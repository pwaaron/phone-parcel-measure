/**
 * image.js — the image-processing primitives both detectors sit on.
 *
 * Everything works on a plain { data: Uint8ClampedArray, width, height } grey
 * buffer. No canvas, no DOM, so the whole detection stack is testable in Node
 * against synthetic images.
 *
 * The adaptive threshold matters more than it looks: a parcel photographed on a
 * hub floor is lit unevenly (overhead strip lights, deep shadows at the base),
 * and a single global threshold loses the code in the dark half of the frame.
 * A local mean over an integral image keeps it.
 */

/** Rec. 601 luma from RGBA, matching what the eye reads as brightness. */
export function lumaFromRGBA(rgba, width, height) {
  const out = new Uint8ClampedArray(width * height);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    out[i] = (rgba[p] * 299 + rgba[p + 1] * 587 + rgba[p + 2] * 114) / 1000;
  }
  return out;
}

/**
 * Summed-area table, one row/column of padding so any rectangle sums in four
 * lookups. Returned as Float64Array because a 12 MP frame overflows 32-bit
 * integer sums at the far corner.
 */
export function integralImage(gray, width, height) {
  const stride = width + 1;
  const I = new Float64Array(stride * (height + 1));
  for (let y = 0; y < height; y++) {
    let rowSum = 0;
    for (let x = 0; x < width; x++) {
      rowSum += gray[y * width + x];
      I[(y + 1) * stride + (x + 1)] = I[y * stride + (x + 1)] + rowSum;
    }
  }
  return I;
}

/** Sum over the inclusive rectangle [x0..x1] x [y0..y1] using the integral image. */
function rectSum(I, stride, x0, y0, x1, y1) {
  return (
    I[(y1 + 1) * stride + (x1 + 1)] -
    I[y0 * stride + (x1 + 1)] -
    I[(y1 + 1) * stride + x0] +
    I[y0 * stride + x0]
  );
}

/**
 * Adaptive mean threshold. A pixel becomes 255 when it is brighter than the mean
 * of its blockSize x blockSize neighbourhood minus C.
 *
 * blockSize must be odd and is clamped to at least 3. C is in grey levels; the
 * usual 5-10 range works well for printed black-on-white codes.
 */
export function adaptiveThreshold(gray, width, height, blockSize = 31, C = 7) {
  let bs = Math.max(3, blockSize | 0);
  if (bs % 2 === 0) bs++;
  const r = (bs - 1) / 2;

  const I = integralImage(gray, width, height);
  const stride = width + 1;
  const out = new Uint8Array(width * height);

  for (let y = 0; y < height; y++) {
    const y0 = Math.max(0, y - r);
    const y1 = Math.min(height - 1, y + r);
    for (let x = 0; x < width; x++) {
      const x0 = Math.max(0, x - r);
      const x1 = Math.min(width - 1, x + r);
      const count = (x1 - x0 + 1) * (y1 - y0 + 1);
      const mean = rectSum(I, stride, x0, y0, x1, y1) / count;
      out[y * width + x] = gray[y * width + x] > mean - C ? 255 : 0;
    }
  }
  return out;
}

/** Otsu's method: the global threshold that maximises between-class variance. */
export function otsuThreshold(gray) {
  const hist = new Float64Array(256);
  for (let i = 0; i < gray.length; i++) hist[gray[i]]++;

  const total = gray.length;
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * hist[i];

  let sumB = 0;
  let wB = 0;
  let best = 0;
  let bestVar = -1;

  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;

    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > bestVar) { bestVar = between; best = t; }
  }
  return best;
}

/** Global threshold to a binary mask (255 = foreground/bright). */
export function thresholdBinary(gray, t) {
  const out = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i++) out[i] = gray[i] > t ? 255 : 0;
  return out;
}

/**
 * Swap foreground and background in a binary mask.
 *
 * Fiducials are printed dark-on-light, so the features we want to trace are the
 * DARK regions. Both thresholding helpers above mark bright pixels as
 * foreground, which is the natural convention but the wrong polarity for marker
 * detection — OpenCV reaches for THRESH_BINARY_INV for the same reason.
 *
 * Getting this wrong is subtle rather than loud: the detector still finds
 * quadrilaterals, they are just the wrong ones, and the failure shows up as
 * slightly-off corners or an unexplained refusal to decode.
 */
export function invertMask(mask) {
  const out = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i++) out[i] = mask[i] ? 0 : 255;
  return out;
}

/** Separable box blur, used to suppress print grain and sensor noise. */
export function boxBlur(gray, width, height, radius = 1) {
  const r = Math.max(1, radius | 0);
  const tmp = new Uint8ClampedArray(width * height);
  const out = new Uint8ClampedArray(width * height);
  const win = 2 * r + 1;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let s = 0;
      for (let k = -r; k <= r; k++) {
        const xx = Math.min(width - 1, Math.max(0, x + k));
        s += gray[y * width + xx];
      }
      tmp[y * width + x] = s / win;
    }
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let s = 0;
      for (let k = -r; k <= r; k++) {
        const yy = Math.min(height - 1, Math.max(0, y + k));
        s += tmp[yy * width + x];
      }
      out[y * width + x] = s / win;
    }
  }
  return out;
}

/** Bilinear downsample by an integer factor — detection runs faster on a smaller frame. */
export function downsample(gray, width, height, factor) {
  const f = Math.max(1, factor | 0);
  if (f === 1) return { data: gray, width, height };

  const nw = Math.floor(width / f);
  const nh = Math.floor(height / f);
  const out = new Uint8ClampedArray(nw * nh);

  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      let s = 0;
      for (let dy = 0; dy < f; dy++) {
        const sy = y * f + dy;
        for (let dx = 0; dx < f; dx++) {
          s += gray[sy * width + x * f + dx];
        }
      }
      out[y * nw + x] = s / (f * f);
    }
  }
  return { data: out, width: nw, height: nh };
}

/**
 * Bilinear sample at a fractional coordinate, clamped at the borders.
 * Subpixel corner refinement reads between pixels, so this is on the accuracy path.
 */
export function sampleBilinear(gray, width, height, x, y) {
  if (x < 0 || y < 0 || x > width - 1 || y > height - 1) return 0;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(width - 1, x0 + 1);
  const y1 = Math.min(height - 1, y0 + 1);
  const fx = x - x0;
  const fy = y - y0;

  const a = gray[y0 * width + x0];
  const b = gray[y0 * width + x1];
  const c = gray[y1 * width + x0];
  const d = gray[y1 * width + x1];

  return (
    a * (1 - fx) * (1 - fy) +
    b * fx * (1 - fy) +
    c * (1 - fx) * fy +
    d * fx * fy
  );
}

/** Sobel gradient magnitude and orientation, for edge-aware refinement. */
export function sobelGradient(gray, width, height) {
  const mag = new Float32Array(width * height);
  const dirX = new Float32Array(width * height);
  const dirY = new Float32Array(width * height);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const gx =
        -gray[i - width - 1] + gray[i - width + 1] +
        -2 * gray[i - 1] + 2 * gray[i + 1] +
        -gray[i + width - 1] + gray[i + width + 1];
      const gy =
        -gray[i - width - 1] - 2 * gray[i - width] - gray[i - width + 1] +
        gray[i + width - 1] + 2 * gray[i + width] + gray[i + width + 1];
      dirX[i] = gx;
      dirY[i] = gy;
      mag[i] = Math.hypot(gx, gy);
    }
  }
  return { magnitude: mag, dx: dirX, dy: dirY };
}

/**
 * Extract a rectangular patch of a grey image as RGBA, for handing to a QR
 * decoder that expects ImageData-like input.
 */
export function greyToRGBA(gray, width, height) {
  const out = new Uint8ClampedArray(width * height * 4);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    const v = gray[i];
    out[p] = v;
    out[p + 1] = v;
    out[p + 2] = v;
    out[p + 3] = 255;
  }
  return out;
}

/** Extract an axis-aligned sub-rectangle as a new grey buffer. */
export function cropGrey(gray, width, height, x0, y0, w, h) {
  const out = new Uint8ClampedArray(w * h);
  for (let y = 0; y < h; y++) {
    const sy = y0 + y;
    for (let x = 0; x < w; x++) {
      const sx = x0 + x;
      out[y * w + x] = sx >= 0 && sx < width && sy >= 0 && sy < height ? gray[sy * width + sx] : 255;
    }
  }
  return out;
}
