/**
 * segment.js — MobileSAM segmentation in the browser, via ONNX Runtime Web.
 *
 * Why segmentation and not detection: a detector gives bounding boxes, and a 2D
 * box cannot yield a physical dimension. What the measurement needs is the
 * object's outline as a polygon, which is then projected onto the QR sticker's
 * known plane to become millimetres.
 *
 * SAM is class-agnostic — it does not know what a "parcel" is. The unlock is
 * that the QR sticker sits ON the parcel face, so a point prompt at the
 * sticker's centre lands on the right object without any training. Negative
 * points around the sticker's surroundings push it away from segmenting the
 * sticker itself, which is a real failure mode: a printed sticker is a distinct
 * visual object and SAM will happily return just that.
 *
 * Architecture matters for the interaction: the encoder is the expensive half
 * (~1024x1024 through a 28 MB network), and the decoder is small. So the encoder
 * runs ONCE per captured frame and the resulting embedding is reused for every
 * prompt — which is what makes "tap to refine the outline" feel responsive.
 *
 * Model provenance (verified by inspecting the graphs):
 *   encoder  input  input_image       [H, W, 3]        NHWC float32
 *            output image_embeddings  [1, 256, 64, 64]
 *   decoder  inputs image_embeddings, point_coords [1,N,2], point_labels [1,N],
 *            mask_input [1,1,256,256], has_mask_input [1], orig_im_size [2]
 *            outputs masks [1,k,H,W] at orig_im_size, iou_predictions [1,k]
 */

const MODEL_URLS = {
  encoder: 'models/mobilesam.encoder.onnx',
  decoder: 'models/mobilesam.decoder.quant.onnx',
};

import { findContours, contourArea, contourPerimeter, approxPolyDP } from './contour.js';

/** SAM always encodes at 1024x1024 regardless of the input's own size. */
const SAM_INPUT = 1024;

export class Segmenter {
  constructor() {
    this.ready = false;
    this.loading = null;
    this.encoder = null;
    this.decoder = null;
    this.embedding = null;
    this.embeddingSize = null;
    this.lastTiming = {};
  }

  /**
   * Load ONNX Runtime and both models. Safe to call repeatedly; the in-flight
   * promise is shared so two callers cannot start two loads.
   *
   * onProgress receives (stage, detail) so the UI can say something more useful
   * than a spinner while 37 MB arrives.
   */
  async load(onProgress = () => {}) {
    if (this.ready) return this;
    if (this.loading) return this.loading;

    this.loading = (async () => {
      onProgress('runtime', 'Loading ONNX Runtime…');
      const ort = await loadOrtRuntime();

      // Single-threaded: the threaded build needs COOP/COEP headers that a
      // plain static host does not send, and silently falling back mid-run
      // would be worse than just not using threads.
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;
      ort.env.logLevel = 'error';

      onProgress('encoder', 'Loading encoder (28 MB)…');
      this.encoder = await ort.InferenceSession.create(MODEL_URLS.encoder, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });

      onProgress('decoder', 'Loading decoder (9 MB)…');
      this.decoder = await ort.InferenceSession.create(MODEL_URLS.decoder, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });

      this.ort = ort;
      this.ready = true;
      onProgress('ready', 'Ready');
      return this;
    })();

    return this.loading;
  }

  /**
   * Encode a frame once. `rgba` is the full-resolution image the masks should
   * come back at, because the decoder resizes to `orig_im_size` for us.
   */
  async setImage(rgba, width, height) {
    if (!this.ready) throw new Error('Segmenter.load() must be awaited first');

    const t0 = performance.now();

    // Resize to SAM's 1024x1024 with a plain box filter. Doing this on the CPU
    // avoids needing a canvas and keeps the module usable outside a browser.
    const resized = new Float32Array(3 * SAM_INPUT * SAM_INPUT);
    const scaleX = width / SAM_INPUT;
    const scaleY = height / SAM_INPUT;

    for (let y = 0; y < SAM_INPUT; y++) {
      const sy = Math.min(height - 1, Math.floor(y * scaleY));
      for (let x = 0; x < SAM_INPUT; x++) {
        const sx = Math.min(width - 1, Math.floor(x * scaleX));
        const p = (sy * width + sx) * 4;
        const i = (y * SAM_INPUT + x);
        // SAM's preprocessing: scale to 0..1 then normalise by the ImageNet
        // mean and standard deviation it was trained with.
        resized[i] = (rgba[p] / 255 - 0.485) / 0.229;
        resized[i + SAM_INPUT * SAM_INPUT] = (rgba[p + 1] / 255 - 0.456) / 0.224;
        resized[i + 2 * SAM_INPUT * SAM_INPUT] = (rgba[p + 2] / 255 - 0.406) / 0.225;
      }
    }

    // The graph declares NHWC, so interleave rather than staying planar.
    const nhwc = new Float32Array(SAM_INPUT * SAM_INPUT * 3);
    for (let i = 0; i < SAM_INPUT * SAM_INPUT; i++) {
      nhwc[i * 3] = resized[i];
      nhwc[i * 3 + 1] = resized[i + SAM_INPUT * SAM_INPUT];
      nhwc[i * 3 + 2] = resized[i + 2 * SAM_INPUT * SAM_INPUT];
    }

    const tensor = new this.ort.Tensor('float32', nhwc, [SAM_INPUT, SAM_INPUT, 3]);
    const out = await this.encoder.run({ input_image: tensor });

    this.embedding = out.image_embeddings;
    this.embeddingSize = [width, height];
    this.lastTiming.encodeMs = performance.now() - t0;
    return { encodeMs: this.lastTiming.encodeMs };
  }

  /**
   * Decode a mask from the cached embedding.
   *
   * points are [x, y] in ORIGINAL image pixels; labels are SAM's convention:
   *   1 = foreground, 0 = background, 2 = box top-left, 3 = box bottom-right.
   *
   * Returns { mask, width, height, score, decodeMs } where mask is a Uint8Array
   * of 0/1 at the original image size.
   */
  async prompt(points, labels) {
    if (!this.embedding) throw new Error('setImage() must run before prompt()');
    const [w, h] = this.embeddingSize;
    const t0 = performance.now();

    // The decoder expects coordinates in SAM's 1024 space, not image pixels.
    const coords = new Float32Array(points.length * 2);
    for (let i = 0; i < points.length; i++) {
      coords[i * 2] = (points[i][0] / w) * SAM_INPUT;
      coords[i * 2 + 1] = (points[i][1] / h) * SAM_INPUT;
    }

    const feeds = {
      image_embeddings: this.embedding,
      point_coords: new this.ort.Tensor('float32', coords, [1, points.length, 2]),
      point_labels: new this.ort.Tensor('float32', Float32Array.from(labels), [1, points.length]),
      // No mask input on the first pass; this is what makes it a fresh prompt
      // rather than a refinement of a previous mask.
      mask_input: new this.ort.Tensor('float32', new Float32Array(1 * 1 * 256 * 256), [1, 1, 256, 256]),
      has_mask_input: new this.ort.Tensor('float32', Float32Array.from([0]), [1]),
      orig_im_size: new this.ort.Tensor('float32', Float32Array.from([h, w]), [2]),
    };

    const out = await this.decoder.run(feeds);
    const masks = out.masks;
    const iou = out.iou_predictions.data;

    // SAM returns several candidate masks with predicted IoU; take the best.
    const numMasks = iou.length;
    let best = 0;
    for (let i = 1; i < numMasks; i++) if (iou[i] > iou[best]) best = i;

    const plane = masks.dims[2] * masks.dims[3];
    const src = masks.data;
    const offset = best * plane;
    const mask = new Uint8Array(plane);
    for (let i = 0; i < plane; i++) mask[i] = src[offset + i] > 0 ? 1 : 0;

    this.lastTiming.decodeMs = performance.now() - t0;
    return {
      mask,
      width: masks.dims[3],
      height: masks.dims[2],
      score: iou[best],
      decodeMs: this.lastTiming.decodeMs,
    };
  }

  /**
   * Prompt from a QR sticker: a foreground point at its centre, and background
   * points ringing it.
   *
   * The ring matters. Without it SAM frequently returns the sticker itself,
   * because a printed label is a coherent visual object — and that mask is
   * exactly the wrong answer, since the sticker is far smaller than the face.
   */
  async promptFromQr(qr, frameWidth, frameHeight) {
    const c = qr.corners;
    const cx = (c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4;
    const cy = (c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4;

    const points = [[cx, cy]];
    const labels = [1];

    // Background samples at a ring well outside the sticker but inside the
    // frame, so they land on whatever is around the parcel rather than on it.
    const size = Math.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1]);
    const r = size * 2.5;
    for (const a of [Math.PI / 4, (3 * Math.PI) / 4, (5 * Math.PI) / 4, (7 * Math.PI) / 4]) {
      const px = cx + Math.cos(a) * r;
      const py = cy + Math.sin(a) * r;
      if (px > 4 && py > 4 && px < frameWidth - 4 && py < frameHeight - 4) {
        points.push([px, py]);
        labels.push(0);
      }
    }

    return this.prompt(points, labels);
  }
}

/**
 * Load ONNX Runtime Web from the vendored UMD build.
 *
 * Vendored rather than CDN so the app keeps working offline, which is the whole
 * reason it is a PWA. The wasm binary is pointed at the same directory.
 */
function loadOrtRuntime() {
  if (globalThis.ort) {
    globalThis.ort.env.wasm.wasmPaths = 'vendor/ort/';
    return Promise.resolve(globalThis.ort);
  }

  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = 'vendor/ort/ort.min.js';
    s.async = true;
    s.onload = () => {
      if (!globalThis.ort) { reject(new Error('ONNX Runtime loaded but exposed no global')); return; }
      // Must be set before any session is created.
      globalThis.ort.env.wasm.wasmPaths = 'vendor/ort/';
      resolve(globalThis.ort);
    };
    s.onerror = () => reject(new Error('Could not load vendor/ort/ort.min.js'));
    document.head.appendChild(s);
  });
}

/** Fetch the current frame as RGBA, for handing to the segmenter. */
export function frameToRGBA(videoEl) {
  const w = videoEl.videoWidth;
  const h = videoEl.videoHeight;
  const c = frameToRGBA._c || (frameToRGBA._c = document.createElement('canvas'));
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(videoEl, 0, 0, w, h);
  return { rgba: ctx.getImageData(0, 0, w, h).data, width: w, height: h };
}

/**
 * Turn a segmentation mask into a simplified outline polygon.
 *
 * Two steps that both matter:
 *
 *   - Keep only the LARGEST connected component. SAM's mask is sometimes
 *     speckled, and a stray blob elsewhere in the frame would otherwise drag
 *     the polygon's bounding box across half the image.
 *   - Simplify with a tolerance proportional to the mask's size. A raw traced
 *     boundary has thousands of points; a parcel face is a handful of corners.
 *     The tolerance has to scale, or a small object gets over-simplified into a
 *     triangle and a large one stays unusably jagged.
 *
 * Returns { polygon, area, bbox, pointCount } or null if the mask is empty.
 */
export function maskOutline(mask, width, height, opts = {}) {
  const { minAreaFrac = 0.002, epsilonFrac = 0.02 } = opts;

  // findContours expects 255 as foreground.
  const binary = new Uint8Array(width * height);
  for (let i = 0; i < binary.length; i++) binary[i] = mask[i] ? 255 : 0;

  const contours = findContours(binary, width, height, 24);
  if (!contours.length) return null;

  let best = null;
  for (const c of contours) {
    const a = contourArea(c);
    if (!best || a > best.area) best = { contour: c, area: a };
  }
  if (!best || best.area < minAreaFrac * width * height) return null;

  const perimeter = contourPerimeter(best.contour);
  const polygon = approxPolyDP(best.contour, perimeter * epsilonFrac, true);
  if (polygon.length < 3) return null;

  let minX = Infinity; let maxX = -Infinity;
  let minY = Infinity; let maxY = -Infinity;
  for (const p of polygon) {
    minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
    minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
  }

  return {
    polygon,
    area: best.area,
    bbox: [minX, minY, maxX, maxY],
    pointCount: polygon.length,
  };
}

export { SAM_INPUT };
