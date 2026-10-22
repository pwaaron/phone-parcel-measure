/**
 * aruco.js — ArUco marker detection, written from scratch.
 *
 * The browser exposes no ArUco API, so this implements the standard pipeline:
 *
 *   threshold -> trace borders -> simplify to quads -> perspective-normalise ->
 *   sample the bit grid -> match against the dictionary under 4 rotations
 *
 * Verified conventions (checked against OpenCV's own renderer):
 *   - Bits are row-major from the top-left, with 1 = WHITE.
 *   - A rendered marker with a 1-module border spans (markerSize + 2) modules,
 *     and that black border is what the contour detector actually finds. So a
 *     detected quad is mapped onto a (markerSize + 2) square, and the data cells
 *     live at offset 1 from its edge.
 *
 * The dictionary is passed in rather than imported, so this module stays pure
 * and testable; see aruco-dict.js for the generated tables.
 */

import { adaptiveThreshold, otsuThreshold, thresholdBinary, boxBlur, sampleBilinear, invertMask } from './image.js';
import {
  findContours, approxPolyDP, contourArea, contourPerimeter,
  isConvex, orderQuadCorners,
} from './contour.js';
import { homographyFromPoints, applyH } from './camera.js';
import { refineQuadSubPix } from './subpix.js';

/**
 * Rotate an n x n bit grid, given as a '0'/'1' string, by 90 degrees clockwise.
 *
 * Strings rather than arrays because the dictionary stores strings, and mixing
 * the two types is a silent disaster: `0 !== '0'` is true, so a numeric grid
 * compared against a string reference reports every bit as a mismatch and no
 * marker ever decodes.
 */
function rotateBits(bitsStr, n, times) {
  let cur = bitsStr;
  for (let t = 0; t < times; t++) {
    const next = new Array(n * n);
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        // (r, c) -> (c, n - 1 - r)
        next[c * n + (n - 1 - r)] = cur[r * n + c];
      }
    }
    cur = next.join('');
  }
  return cur;
}

/**
 * Normalise a dictionary's bit patterns to '0'/'1' strings, cached per object.
 *
 * Both directions of the numeric/string mismatch are possible: the generated
 * dictionary stores strings while the sampler yields numbers, but hand-built
 * dictionaries in tests do the opposite. Comparing across the two types makes
 * `0 !== '0'` true for every bit, so nothing decodes and nothing errors. Doing
 * the conversion once per dictionary keeps the inner loop a plain string
 * comparison.
 */
const normalizedDicts = new WeakMap();

function normalizeDict(dict) {
  const cached = normalizedDicts.get(dict);
  if (cached) return cached;

  const normalized = {
    markerSize: dict.markerSize,
    maxCorrectionBits: dict.maxCorrectionBits,
    markers: dict.markers.map((m) => (typeof m === 'string' ? m : m.join(''))),
  };
  normalizedDicts.set(dict, normalized);
  return normalized;
}

/**
 * Match a sampled bit grid against the dictionary under all four rotations,
 * returning the best id within the dictionary's correction budget.
 *
 * `bits` may be an array of 0/1 numbers (as produced by the sampler) or a
 * string; both it and the dictionary are normalised before comparison.
 *
 * The reported `rotation` is the number of 90-degree CLOCKWISE turns that take
 * the canonical marker to the observed one. Internally we rotate the observed
 * grid until it matches canonical, which is the opposite direction, so the
 * result is converted. Callers rely on this sign to order corners correctly; get
 * it backwards and the marker's reference frame is mirrored.
 */
export function matchMarker(bits, dict) {
  const { markers, markerSize, maxCorrectionBits } = normalizeDict(dict);
  const bitsStr = typeof bits === 'string' ? bits : bits.join('');

  let bestId = -1;
  let bestDist = Infinity;
  let bestRot = 0;

  for (let rot = 0; rot < 4; rot++) {
    const rotated = rot === 0 ? bitsStr : rotateBits(bitsStr, markerSize, rot);
    for (let id = 0; id < markers.length; id++) {
      const ref = markers[id];
      let dist = 0;
      for (let i = 0; i < rotated.length; i++) {
        if (rotated[i] !== ref[i]) {
          dist++;
          if (dist > maxCorrectionBits) break;
        }
      }
      if (dist < bestDist) {
        bestDist = dist;
        bestId = id;
        bestRot = rot;
        if (dist === 0) return { id: bestId, distance: 0, rotation: (4 - bestRot) % 4 };
      }
    }
  }

  if (bestId < 0 || bestDist > maxCorrectionBits) return null;
  return { id: bestId, distance: bestDist, rotation: (4 - bestRot) % 4 };
}

/**
 * Sample the markerSize x markerSize data cells from a detected quad.
 *
 * Each cell is sampled at several points around its centre and averaged, rather
 * than read at a single pixel: print speckle, JPEG artefacts and sensor noise
 * all corrupt individual pixels, and a marker that decodes 90% of the time is
 * far less useful than one that decodes 99.9% of the time.
 */
function extractBits(gray, width, height, quad, markerSize) {
  const n = markerSize + 2; // include the 1-module border the quad encloses
  const canonical = [[0, 0], [n, 0], [n, n], [0, n]];

  let H;
  try {
    H = homographyFromPoints(canonical, quad);
  } catch {
    return null;
  }

  const bits = new Array(markerSize * markerSize);
  const offsets = [-0.22, 0, 0.22];

  for (let r = 0; r < markerSize; r++) {
    for (let c = 0; c < markerSize; c++) {
      const mx = 1 + c + 0.5;
      const my = 1 + r + 0.5;

      let sum = 0;
      let count = 0;
      for (const oy of offsets) {
        for (const ox of offsets) {
          const [px, py] = applyH(H, [mx + ox, my + oy]);
          if (px < 0 || py < 0 || px > width - 1 || py > height - 1) continue;
          sum += sampleBilinear(gray, width, height, px, py);
          count++;
        }
      }
      if (count === 0) return null;

      // Otsu-derived binarisation threshold: a cell is white if it is clearly
      // brighter than the midpoint between the local black and white levels.
      bits[r * markerSize + c] = sum / count > 127 ? 1 : 0;
    }
  }

  return bits;
}

/**
 * Detect every ArUco marker in a grey image.
 *
 * opts:
 *   blurRadius   (default 1)  — suppress print grain before thresholding
 *   blockSize    (default 31) — adaptive threshold window
 *   adaptiveC    (default 7)
 *   minAreaFrac  (default 1e-5) — ignore quads smaller than this share of the frame
 *   maxAreaFrac  (default 0.95)
 *   refine       (default true) — sub-pixel corner refinement
 *
 * Returns [{ id, corners, distance, rotation }] where corners are the marker's
 * own top-left, top-right, bottom-right, bottom-left in that order, in image
 * pixels. That ordering is what lets a caller treat the marker as a metric
 * reference frame.
 */
/**
 * Candidate window sizes for the adaptive threshold, scaled to the frame.
 *
 * A single window cannot work across distances. Adaptive thresholding compares
 * each pixel to the mean of its neighbourhood, so any uniform region LARGER
 * than the window has mean == pixel value and reads as "bright" — the black
 * border of a close-up marker disappears entirely, while a distant marker's
 * modules are smaller than the window and come through fine.
 *
 * Sweeping several scales and merging is how OpenCV's detector handles this
 * (adaptiveThreshWinSizeMin/Max/Step). Same idea here.
 */
function defaultBlockSizes(width, height) {
  const m = Math.min(width, height);
  const sizes = new Set();
  for (const frac of [1 / 64, 1 / 32, 1 / 16, 1 / 8]) {
    let s = Math.round(m * frac);
    s = Math.max(9, Math.min(s, 151));
    if (s % 2 === 0) s++;
    if (s <= m / 3) sizes.add(s);
  }
  return [...sizes].sort((a, b) => a - b);
}

/** One detection pass over a single binary mask. */
function detectInBinary(binary, gray, width, height, dict, opts = {}) {
  const { minAreaFrac = 1e-5, maxAreaFrac = 0.95, refine = true } = opts;

  const frameArea = width * height;
  const contours = findContours(binary, width, height, 12);
  const results = [];

  for (const contour of contours) {
    const perimeter = contourPerimeter(contour);
    if (perimeter < 20) continue;

    // A marker's border is four straight lines; anything needing a large
    // tolerance to flatten is not a marker.
    const approx = approxPolyDP(contour, 0.05 * perimeter, true);
    if (approx.length !== 4) continue;
    if (!isConvex(approx)) continue;

    const area = contourArea(approx);
    if (area < minAreaFrac * frameArea || area > maxAreaFrac * frameArea) continue;

    // Reject slivers: a genuine marker quad has no side wildly longer than another.
    const sides = approx.map((p, i) => {
      const q = approx[(i + 1) % 4];
      return Math.hypot(q[0] - p[0], q[1] - p[1]);
    });
    const minSide = Math.min(...sides);
    const maxSide = Math.max(...sides);
    if (minSide < 4 || maxSide / minSide > 6) continue;

    let quad = orderQuadCorners(approx);
    if (refine) quad = refineQuadSubPix(gray, width, height, quad, 5, 3);

    const bits = extractBits(gray, width, height, quad, dict.markerSize);
    if (!bits) continue;

    const match = matchMarker(bits, dict);
    if (!match) continue;

    // Re-order so corners are the marker's own TL, TR, BR, BL. A rotation of
    // `r` steps means the marker's canonical top-left sits at index r of the
    // image-ordered quad.
    const ordered = [];
    for (let i = 0; i < 4; i++) ordered.push(quad[(match.rotation + i) % 4]);

    results.push({
      id: match.id,
      corners: ordered,
      distance: match.distance,
      rotation: match.rotation,
    });
  }

  return dedupeMarkers(results);
}

/**
 * The same physical marker can be found twice when its border produces more
 * than one qualifying contour. Keep the best-scoring detection per id.
 */
function dedupeMarkers(markers) {
  const byId = new Map();
  for (const m of markers) {
    const prev = byId.get(m.id);
    if (!prev) { byId.set(m.id, m); continue; }
    const areaOf = (q) => contourArea(q);
    if (m.distance < prev.distance ||
        (m.distance === prev.distance && areaOf(m.corners) > areaOf(prev.corners))) {
      byId.set(m.id, m);
    }
  }
  return Array.from(byId.values());
}

/**
 * Detect every ArUco marker in a grey image.
 *
 * opts:
 *   blurRadius   (default 1)    — suppress print grain before thresholding
 *   blockSizes   (default null) — adaptive threshold windows; null = auto-scaled
 *   adaptiveC    (default 7)
 *   minAreaFrac  (default 1e-5)
 *   maxAreaFrac  (default 0.95)
 *   refine       (default true) — sub-pixel corner refinement
 *
 * Returns [{ id, corners, distance, rotation }] where corners are the marker's
 * own top-left, top-right, bottom-right, bottom-left in that order, in image
 * pixels — a metric reference frame the caller can measure against.
 */
export function detectMarkers(gray, width, height, dict, opts = {}) {
  const {
    blurRadius = 1, blockSizes = null, adaptiveC = 7, useOtsu = true, invert = true, ...rest
  } = opts;

  const smoothed = blurRadius > 0 ? boxBlur(gray, width, height, blurRadius) : gray;
  const all = [];
  // Markers are dark-on-light, so the mask is flipped: the features we trace are
  // the dark regions.
  const fix = (mask) => (invert ? invertMask(mask) : mask);

  // A global Otsu pass first. It is the right tool for evenly lit scenes and for
  // markers so close that their modules exceed any sensible local window, which
  // is exactly where the adaptive sweep below gives up.
  if (useOtsu) {
    const binary = fix(thresholdBinary(smoothed, otsuThreshold(smoothed)));
    for (const m of detectInBinary(binary, smoothed, width, height, dict, rest)) all.push(m);
  }

  const windows = blockSizes && blockSizes.length ? blockSizes : defaultBlockSizes(width, height);
  for (const bs of windows) {
    const binary = fix(adaptiveThreshold(smoothed, width, height, bs, adaptiveC));
    for (const m of detectInBinary(binary, smoothed, width, height, dict, rest)) all.push(m);
  }

  return dedupeMarkers(all);
}

/** Detect using a global Otsu threshold. Useful for flat, evenly lit scenes. */
export function detectMarkersOtsu(gray, width, height, dict, opts = {}) {
  const { invert = true, ...rest } = opts;
  const raw = thresholdBinary(gray, otsuThreshold(gray));
  const binary = invert ? invertMask(raw) : raw;
  return dedupeMarkers(detectInBinary(binary, gray, width, height, dict, rest));
}
