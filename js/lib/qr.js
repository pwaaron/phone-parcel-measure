/**
 * qr.js — QR code detection and payload parsing.
 *
 * The QR sticker is what makes a measurement metric: it is the one object in
 * frame whose real size is known, and its four corners give the surface plane
 * and orientation at the same time.
 *
 * Two detection paths, best first:
 *   1. The native BarcodeDetector API. Fast, hardware-backed, and it reports
 *      every code in frame with corner points.
 *   2. A pure-JS fallback: locate finder patterns by their 1:1:3:1:1 module
 *      signature, group them into candidate symbols, and hand each warped patch
 *      to jsQR to decode.
 *
 * The fallback exists because BarcodeDetector is not universal. It also gives
 * multi-code support without depending on the browser.
 *
 * Payload format (fixed, and emitted by tools/make_qr_sheet.py):
 *   SPXM|1|<id>|<sizeMm>
 * where <sizeMm> is the printed symbol side length in millimetres, excluding the
 * quiet zone. Both detectors report the symbol's own corners, which is the same
 * extent, so the size maps straight onto them with no offset correction.
 */

import {
  adaptiveThreshold, thresholdBinary, otsuThreshold, boxBlur, invertMask, sampleBilinear,
} from './image.js';
import { findContours, approxPolyDP, contourArea, contourPerimeter, isConvex } from './contour.js';
import { homographyFromPoints, applyH } from './camera.js';

const PAYLOAD_RE = /^SPXM\|(\d+)\|([A-Za-z0-9_-]{1,32})\|([0-9]+(?:\.[0-9]+)?)$/;

/**
 * Parse a SPXM payload.
 * Returns { schema, id, sizeMm } or null. A null result is not an error: the
 * sticker may be an ordinary QR, in which case the app asks for its size once
 * and remembers it by id.
 */
export function parseQrPayload(text) {
  if (typeof text !== 'string') return null;
  const m = PAYLOAD_RE.exec(text.trim());
  if (!m) return null;
  const sizeMm = Number(m[3]);
  if (!Number.isFinite(sizeMm) || sizeMm <= 0) return null;
  return { schema: Number(m[1]), id: m[2], sizeMm };
}

/** Build a payload string. Mirrors the Python generator exactly. */
export function buildQrPayload(id, sizeMm) {
  return `SPXM|1|${id}|${Number(sizeMm)}`;
}

/* --------------------------------------------------------- native detector */

/**
 * Detect using the browser's BarcodeDetector.
 *
 * The spec fixes cornerPoints as clockwise from the top-left, which is the same
 * order the rest of the pipeline expects, so no re-ordering is needed.
 */
async function detectNative(rgba, width, height, DetectorCtor) {
  const detector = new DetectorCtor({ formats: ['qr_code'] });
  const imageData = new ImageData(rgba, width, height);
  const results = await detector.detect(imageData);

  return results.map((r) => ({
    text: r.rawValue || '',
    corners: (r.cornerPoints || []).map((p) => [p.x, p.y]),
    source: 'native',
  })).filter((r) => r.corners.length === 4);
}

/* ------------------------------------------- finder-pattern locator (fallback) */

/**
 * Check the 1:1:3:1:1 dark/light run-length signature of a QR finder pattern
 * along a horizontal line through (cx, cy).
 *
 * This ratio is the defining feature of a finder pattern and is what makes it
 * distinguishable from arbitrary square blobs in the scene.
 *
 * The scan starts at the pattern's centre, which lies INSIDE the central 3-wide
 * dark run. Measuring runs from there yields [3,1,1,...] — not the signature.
 * So we first walk out to the bounds of the central run, then measure the
 * light/dark pair on each side, giving the true five runs.
 */
function hasFinderRatio(binary, width, height, cx, cy, expectedSize) {
  const y = Math.round(cy);
  const x0 = Math.round(cx);
  if (y < 0 || y >= height || x0 < 0 || x0 >= width) return false;

  const isDark = (x) => x >= 0 && x < width && binary[y * width + x] === 255;
  if (!isDark(x0)) return false;

  let midL = x0;
  while (midL > 0 && isDark(midL - 1)) midL--;
  let midR = x0;
  while (midR < width - 1 && isDark(midR + 1)) midR++;
  const mid = midR - midL + 1;
  if (mid < 3) return false;

  let p = midL - 1;
  let lightL = 0;
  while (p >= 0 && !isDark(p)) { lightL++; p--; }
  let darkL = 0;
  while (p >= 0 && isDark(p)) { darkL++; p--; }

  p = midR + 1;
  let lightR = 0;
  while (p < width && !isDark(p)) { lightR++; p++; }
  let darkR = 0;
  while (p < width && isDark(p)) { darkR++; p++; }

  const runs = [darkL, lightL, mid, lightR, darkR];
  const total = runs.reduce((a, b) => a + b, 0);
  if (total < 7) return false;

  const unit = total / 7;
  if (unit < 1) return false;

  // Each run must be within tolerance of its expected multiple of the unit.
  const expected = [1, 1, 3, 1, 1];
  const tol = Math.max(0.6, unit * 0.75);
  for (let i = 0; i < 5; i++) {
    if (Math.abs(runs[i] - expected[i] * unit) > tol * expected[i]) return false;
  }

  if (expectedSize && Math.abs(total - expectedSize) > expectedSize * 0.6) return false;
  return true;
}

/**
 * Find candidate finder-pattern centres.
 *
 * Finder patterns appear as nested square contours, so a contour whose bounding
 * box is roughly square and whose centre passes the ratio test is a candidate.
 */
function locateFinderPatterns(binary, gray, width, height, opts = {}) {
  const { minSizePx = 7, maxSizePx = null } = opts;
  const maxSize = maxSizePx || Math.min(width, height) * 0.5;
  const contours = findContours(binary, width, height, 8);
  const found = [];

  for (const contour of contours) {
    const per = contourPerimeter(contour);
    const approx = approxPolyDP(contour, 0.06 * per, true);
    if (approx.length !== 4 || !isConvex(approx)) continue;

    let minX = Infinity; let maxX = -Infinity; let minY = Infinity; let maxY = -Infinity;
    for (const p of approx) {
      minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
      minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
    }
    const w = maxX - minX;
    const h = maxY - minY;
    if (w < minSizePx || h < minSizePx) continue;
    if (w > maxSize || h > maxSize) continue;

    // A finder pattern is square; the outer black ring is its full extent.
    if (Math.max(w / h, h / w) > 1.4) continue;

    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    if (!hasFinderRatio(binary, width, height, cx, cy, (w + h) / 2)) continue;

    found.push({ center: [cx, cy], size: (w + h) / 2, area: contourArea(approx) });
  }

  // Collapse duplicates: nested contours of the same pattern land on nearly the
  // same centre.
  const merged = [];
  for (const f of found.sort((a, b) => b.size - a.size)) {
    const near = merged.find(
      (m) => Math.hypot(m.center[0] - f.center[0], m.center[1] - f.center[1]) < f.size * 0.5
    );
    if (!near) merged.push(f);
  }
  return merged;
}

/**
 * Group three finder patterns into a QR symbol and estimate the fourth corner.
 *
 * The three patterns sit at the top-left, top-right and bottom-left of the
 * symbol and form a right isosceles triangle. The right-angle vertex is the
 * top-left; the hypotenuse joins the other two. The fourth corner follows from
 * the parallelogram relation.
 */
function symbolFromFinders(a, b, c) {
  const d = (p, q) => Math.hypot(p.center[0] - q.center[0], p.center[1] - q.center[1]);
  const dab = d(a, b);
  const dbc = d(b, c);
  const dca = d(c, a);

  // The two closest points are the non-right-angle vertices; the third is TL.
  let tl;
  let p1;
  let p2;
  if (dab >= dbc && dab >= dca) { tl = c; p1 = a; p2 = b; }
  else if (dbc >= dab && dbc >= dca) { tl = a; p1 = b; p2 = c; }
  else { tl = b; p1 = a; p2 = c; }

  const s1 = Math.hypot(p1.center[0] - tl.center[0], p1.center[1] - tl.center[1]);
  const s2 = Math.hypot(p2.center[0] - tl.center[0], p2.center[1] - tl.center[1]);
  // The two arms should be comparable in length.
  if (Math.max(s1 / s2, s2 / s1) > 1.6) return null;

  // Orient: cross product decides which arm is the top edge.
  const cross =
    (p1.center[0] - tl.center[0]) * (p2.center[1] - tl.center[1]) -
    (p1.center[1] - tl.center[1]) * (p2.center[0] - tl.center[0]);

  const tr = cross > 0 ? p2 : p1;
  const bl = cross > 0 ? p1 : p2;

  // The finder centres sit half a module-set inside the symbol, so expand each
  // arm outward by the pattern's own half-width to reach the symbol corners.
  const halfTl = tl.size / 2;
  const halfTr = tr.size / 2;
  const halfBl = bl.size / 2;

  const dir = (from, to, len) => [(to[0] - from[0]) / len, (to[1] - from[1]) / len];
  const armX = [tr.center[0] - tl.center[0], tr.center[1] - tl.center[1]];
  const armY = [bl.center[0] - tl.center[0], bl.center[1] - tl.center[1]];
  const ux = dir(tl.center, tr.center, s1);
  const uy = dir(tl.center, bl.center, s2);

  const cornerTl = [tl.center[0] - ux[0] * halfTl - uy[0] * halfTl,
                    tl.center[1] - ux[1] * halfTl - uy[1] * halfTl];
  const cornerTr = [tr.center[0] + ux[0] * halfTr - uy[0] * halfTr,
                    tr.center[1] + ux[1] * halfTr - uy[1] * halfTr];
  const cornerBl = [bl.center[0] - ux[0] * halfBl + uy[0] * halfBl,
                    bl.center[1] - ux[1] * halfBl + uy[1] * halfBl];
  const cornerBr = [cornerTr[0] + cornerBl[0] - cornerTl[0],
                    cornerTr[1] + cornerBl[1] - cornerTl[1]];

  void armX; void armY;

  return {
    corners: [cornerTl, cornerTr, cornerBr, cornerBl],
    finders: [tl, tr, bl],
    estimated: true,
  };
}

/**
 * Warp a quadrilateral out of a grey image into a square RGBA patch, for a QR
 * decoder that expects ImageData.
 *
 * Sampling at a few points per output pixel and averaging suppresses print grain
 * and sensor noise, which is what lets jsQR succeed on a phone photo where it
 * would fail on a single-pixel sample.
 */
function warpToRGBA(gray, width, height, quad, outSize) {
  const canonical = [[0, 0], [outSize, 0], [outSize, outSize], [0, outSize]];
  let H;
  try {
    H = homographyFromPoints(canonical, quad);
  } catch {
    return null;
  }

  const out = new Uint8ClampedArray(outSize * outSize * 4);
  const sub = [-0.25, 0, 0.25];

  for (let y = 0; y < outSize; y++) {
    for (let x = 0; x < outSize; x++) {
      let sum = 0;
      let n = 0;
      for (const dy of sub) {
        for (const dx of sub) {
          const [sx, sy] = applyH(H, [x + 0.5 + dx, y + 0.5 + dy]);
          if (sx < 0 || sy < 0 || sx > width - 1 || sy > height - 1) continue;
          sum += sampleBilinear(gray, width, height, sx, sy);
          n++;
        }
      }
      const v = n ? sum / n : 255;
      const p = (y * outSize + x) * 4;
      out[p] = v;
      out[p + 1] = v;
      out[p + 2] = v;
      out[p + 3] = 255;
    }
  }
  return out;
}

/**
 * Locate and decode QR symbols without the native API.
 *
 * jsQrFn is injected so this module stays testable outside a browser; it has the
 * signature (rgba, width, height) -> { data, location } | null, matching jsQR.
 */
function detectFallback(gray, width, height, jsQrFn, opts = {}) {
  const { blockSize = 41, adaptiveC = 7, patchSize = 240, useOtsu = true } = opts;

  const smoothed = boxBlur(gray, width, height, 1);
  const masks = [];
  if (useOtsu) masks.push(invertMask(thresholdBinary(smoothed, otsuThreshold(smoothed))));
  masks.push(invertMask(adaptiveThreshold(smoothed, width, height, blockSize, adaptiveC)));

  const results = [];
  const seenCentres = [];

  for (const binary of masks) {
    const finders = locateFinderPatterns(binary, smoothed, width, height);
    if (finders.length < 3) continue;

    // Try every triple. Real scenes rarely contain more than a couple of codes,
    // so the cubic cost is negligible and it avoids needing a clustering step.
    for (let i = 0; i < finders.length; i++) {
      for (let j = i + 1; j < finders.length; j++) {
        for (let k = j + 1; k < finders.length; k++) {
          const symbol = symbolFromFinders(finders[i], finders[j], finders[k]);
          if (!symbol) continue;

          const centre = symbol.corners.reduce(
            (acc, p) => [acc[0] + p[0] / 4, acc[1] + p[1] / 4], [0, 0]
          );
          if (seenCentres.some((c) => Math.hypot(c[0] - centre[0], c[1] - centre[1]) < 12)) continue;

          const rgba = warpToRGBA(smoothed, width, height, symbol.corners, patchSize);
          if (!rgba) continue;

          const decoded = jsQrFn(rgba, patchSize, patchSize);
          if (!decoded || !decoded.data) continue;

          // Prefer the decoder's own corner estimate when it gives one: it is
          // derived from the actual module grid rather than from three blob
          // centres, so it is materially more accurate.
          let corners = symbol.corners;
          if (decoded.location) {
            const l = decoded.location;
            corners = [
              [l.topLeftCorner.x, l.topLeftCorner.y],
              [l.topRightCorner.x, l.topRightCorner.y],
              [l.bottomRightCorner.x, l.bottomRightCorner.y],
              [l.bottomLeftCorner.x, l.bottomLeftCorner.y],
            ].map((p) => applyH(
              homographyFromPoints(
                [[0, 0], [patchSize, 0], [patchSize, patchSize], [0, patchSize]],
                symbol.corners
              ),
              p
            ));
          }

          seenCentres.push(centre);
          results.push({ text: decoded.data, corners, source: 'fallback' });
        }
      }
    }
    if (results.length) break;
  }

  return results;
}

/* ------------------------------------------------------------------ public */

/**
 * Detect every QR code in a grey frame.
 *
 * opts:
 *   barcodeDetector — constructor to use; defaults to the global if present
 *   jsQrFn          — fallback decoder; defaults to the global if present
 *   preferNative    (default true)
 *   defaultSizeMm   — size applied to any code that does not carry one
 *   sizeRegistry    — Map of key -> sizeMm, for codes whose payload omits a size
 *
 * Size resolution, most specific first:
 *   1. the SPXM payload's own sizeMm
 *   2. a size remembered against this exact payload
 *   3. the caller's default (the user's "all my labels are 40 mm" setting)
 *
 * Tier 3 is what makes ARBITRARY QR codes usable. Any code works as a ruler so
 * long as its printed size is known, and for parcels the payload is a different
 * tracking number every time — so remembering sizes per payload would mean
 * re-typing the size for every parcel. A single default covers the whole run.
 *
 * Returns [{ text, corners, source, id, sizeMm, payloadOk, sizeSource }] with
 * corners ordered top-left, top-right, bottom-right, bottom-left — the order the
 * measurement code assumes when building the object-frame correspondences.
 */
export async function detectQrCodes(gray, width, height, opts = {}) {
  const {
    barcodeDetector = (typeof globalThis !== 'undefined' ? globalThis.BarcodeDetector : null),
    jsQrFn = (typeof globalThis !== 'undefined' ? globalThis.jsQR : null),
    preferNative = true,
    defaultSizeMm = null,
    sizeRegistry = null,
  } = opts;

  let raw = [];

  if (preferNative && barcodeDetector) {
    try {
      const rgba = new Uint8ClampedArray(width * height * 4);
      for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
        const v = gray[i];
        rgba[p] = v; rgba[p + 1] = v; rgba[p + 2] = v; rgba[p + 3] = 255;
      }
      raw = await detectNative(rgba, width, height, barcodeDetector);
    } catch {
      raw = [];
    }
  }

  if (raw.length === 0 && jsQrFn) {
    raw = detectFallback(gray, width, height, jsQrFn, opts);
  }

  return raw.map((r) => {
    const parsed = parseQrPayload(r.text);
    // Any code is identified by its own payload, so a remembered size can be
    // attached to an ordinary QR just as well as to one of ours.
    const id = parsed ? parsed.id : (r.text || '').slice(0, 64);

    let sizeMm = parsed ? parsed.sizeMm : null;
    let sizeSource = parsed ? 'payload' : null;

    if (sizeMm === null && id && sizeRegistry && sizeRegistry.has(id)) {
      sizeMm = sizeRegistry.get(id);
      sizeSource = 'remembered';
    }
    if (sizeMm === null && Number.isFinite(defaultSizeMm) && defaultSizeMm > 0) {
      sizeMm = defaultSizeMm;
      sizeSource = 'default';
    }

    return {
      text: r.text,
      corners: r.corners,
      source: r.source,
      id,
      sizeMm,
      sizeSource,
      payloadOk: !!parsed,
    };
  });
}

/**
 * Mean side length of a detected QR quad, in pixels — the quantity that sets
 * measurement precision, and worth surfacing in the UI.
 */
export function qrApparentSizePx(qr) {
  const c = qr.corners;
  const w = Math.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1]);
  const h = Math.hypot(c[3][0] - c[0][0], c[3][1] - c[0][1]);
  return (w + h) / 2;
}

/**
 * A sticker is only useful as a ruler if it covers enough pixels for its corners
 * to be located precisely. Below roughly 80 px the module grid is too coarse to
 * decode reliably at all.
 */
export function qrIsUsable(qr, minPx = 60) {
  return !!qr.sizeMm && qrApparentSizePx(qr) >= minPx;
}

export { isConvex, contourArea };
