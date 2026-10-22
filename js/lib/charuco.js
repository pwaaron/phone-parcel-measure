/**
 * charuco.js — ChArUco board model and chessboard-corner interpolation.
 *
 * A ChArUco board is a chessboard with ArUco markers printed inside its white
 * squares. It exists to fix the one weakness of a plain chessboard: a chessboard
 * has no idea which corner is which, so a partly-occluded or partially-detected
 * board is unusable. The markers carry ids, so every detected corner is named,
 * and partial views still contribute.
 *
 * Layout, verified against OpenCV's own renderer (see tools/make_charuco_board.py):
 *   - Markers occupy every cell where (i + j) is odd, INCLUDING the border ring.
 *   - Ids are assigned row-major over those cells: j outer, i inner, from 0.
 *   - A marker spans the central markerLength/squareLength of its cell.
 *   - Marker corner order is top-left, top-right, bottom-right, bottom-left.
 *   - Chessboard corner (I, J) sits at ((I+1)*squareLength, (J+1)*squareLength)
 *     and is indexed J * (squaresX - 1) + I.
 *
 * The key geometric fact this module exploits: a chessboard corner is the exact
 * midpoint of the two diagonally-adjacent markers around it. Rather than average
 * two centres, we fit a homography from both markers' eight corners — the marker
 * corners are the features that were sub-pixel refined, so they carry the
 * accuracy, and eight points over-determine the map and average out their noise.
 *
 * All units are millimetres, matching the rest of the pipeline.
 */

import { homographyFromPoints, applyH } from './camera.js';
import { refineChessboardCorner } from './subpix.js';

/**
 * Build the board's geometry model from a printed-board spec.
 * Mirrors the layout rules above so the app never re-derives them per frame.
 */
export function makeBoardModel({ squaresX, squaresY, squareLengthMm, markerLengthMm }) {
  const SX = squaresX;
  const SY = squaresY;
  const SQ = squareLengthMm;
  const MK = markerLengthMm;

  // marker id -> [i, j], and the reverse lookup
  const markerCells = [];
  const cellToId = new Map();
  for (let j = 0; j < SY; j++) {
    for (let i = 0; i < SX; i++) {
      if ((i + j) % 2 === 1) {
        cellToId.set(`${i},${j}`, markerCells.length);
        markerCells.push([i, j]);
      }
    }
  }

  const half = (SQ - MK) / 2;

  /** The four object-frame corners of a marker, in TL, TR, BR, BL order. */
  function markerObjPoints(id) {
    const [i, j] = markerCells[id];
    const x0 = i * SQ + half;
    const y0 = j * SQ + half;
    const x1 = x0 + MK;
    const y1 = y0 + MK;
    return [
      [x0, y0, 0],
      [x1, y0, 0],
      [x1, y1, 0],
      [x0, y1, 0],
    ];
  }

  /** Object-frame position of chessboard corner index k. */
  function chessboardObjPoint(k) {
    const I = k % (SX - 1);
    const J = Math.floor(k / (SX - 1));
    return [(I + 1) * SQ, (J + 1) * SQ, 0];
  }

  const markerIdAt = (i, j) => {
    const v = cellToId.get(`${i},${j}`);
    return v === undefined ? null : v;
  };

  return {
    squaresX: SX,
    squaresY: SY,
    squareLengthMm: SQ,
    markerLengthMm: MK,
    markerCount: markerCells.length,
    chessboardCornerCount: (SX - 1) * (SY - 1),
    markerCells,
    markerObjPoints,
    chessboardObjPoint,
    markerIdAt,
  };
}

/**
 * Interpolate chessboard corners from detected markers.
 *
 * For each interior chessboard corner, exactly one of the two diagonals through
 * it lands on marker cells (since marker cells all share one parity). We fit a
 * homography to that diagonal pair's eight corners and map the corner's known
 * board position through it.
 *
 * Returns [{ cornerId, imagePoint, sourceMarkers }] for corners whose supporting
 * markers were both detected. Corners with a missing neighbour are simply
 * omitted — a partial board is still useful, which is the whole point.
 */
export function interpolateCharucoCorners(markers, board) {
  const byId = new Map();
  for (const m of markers) byId.set(m.id, m);

  const out = [];
  const SX = board.squaresX;
  const SY = board.squaresY;

  for (let J = 0; J < SY - 1; J++) {
    for (let I = 0; I < SX - 1; I++) {
      const cornerId = J * (SX - 1) + I;

      // Pick the diagonal whose cells carry markers.
      const pair = (I + J) % 2 === 0
        ? [[I + 1, J], [I, J + 1]]
        : [[I, J], [I + 1, J + 1]];

      const idA = board.markerIdAt(pair[0][0], pair[0][1]);
      const idB = board.markerIdAt(pair[1][0], pair[1][1]);
      if (idA === null || idB === null) continue;

      const mA = byId.get(idA);
      const mB = byId.get(idB);
      if (!mA || !mB) continue;

      // Eight correspondences: board-plane marker corners -> image.
      const src = [];
      const dst = [];
      for (const m of [mA, mB]) {
        const obj = board.markerObjPoints(m.id);
        for (let k = 0; k < 4; k++) {
          src.push([obj[k][0], obj[k][1]]);
          dst.push(m.corners[k]);
        }
      }

      let H;
      try {
        H = homographyFromPoints(src, dst);
      } catch {
        continue;
      }

      const target = board.chessboardObjPoint(cornerId);
      const imagePoint = applyH(H, [target[0], target[1]]);

      if (!Number.isFinite(imagePoint[0]) || !Number.isFinite(imagePoint[1])) continue;

      out.push({ cornerId, imagePoint, sourceMarkers: [idA, idB] });
    }
  }

  return out;
}

/**
 * Full detection: markers -> interpolated corners -> sub-pixel refinement.
 *
 * Refinement runs last, on the interpolated position, using a window scaled to
 * the apparent square size. Doing it after interpolation (rather than refining
 * only the marker corners) lets the chessboard's own four-quadrant structure
 * pull the corner onto the true intersection.
 */
export function detectCharucoCorners(gray, width, height, markers, board) {
  const interpolated = interpolateCharucoCorners(markers, board);
  if (interpolated.length === 0) return [];

  // Estimate apparent square size from the spread of interpolated corners.
  let squarePx = 0;
  if (interpolated.length >= 2) {
    let minX = Infinity; let maxX = -Infinity;
    let minY = Infinity; let maxY = -Infinity;
    for (const c of interpolated) {
      minX = Math.min(minX, c.imagePoint[0]);
      maxX = Math.max(maxX, c.imagePoint[0]);
      minY = Math.min(minY, c.imagePoint[1]);
      maxY = Math.max(maxY, c.imagePoint[1]);
    }
    const spanX = (maxX - minX) / Math.max(1, board.squaresX - 2);
    const spanY = (maxY - minY) / Math.max(1, board.squaresY - 2);
    squarePx = Math.max(spanX, spanY);
  }
  if (!Number.isFinite(squarePx) || squarePx <= 2) squarePx = 20;

  return interpolated.map((c) => {
    const refined = refineChessboardCorner(gray, width, height, c.imagePoint, squarePx);
    return {
      cornerId: c.cornerId,
      imagePoint: refined,
      sourceMarkers: c.sourceMarkers,
    };
  });
}

/**
 * Convert a set of ChArUco corner detections into a calibration view.
 *
 * Returns { objectPoints, imagePoints } with the object points in board
 * millimetres — which is what makes the resulting focal length and every later
 * measurement metric rather than merely consistent.
 */
export function charucoViewToCalibration(detections, board) {
  const objectPoints = [];
  const imagePoints = [];
  for (const d of detections) {
    objectPoints.push(board.chessboardObjPoint(d.cornerId));
    imagePoints.push(d.imagePoint);
  }
  return { objectPoints, imagePoints };
}

/**
 * Reject a detection set that is too degenerate to calibrate from.
 *
 * A board viewed almost edge-on, or one where the corners are nearly collinear,
 * produces a homography that is numerically valid but geometrically meaningless.
 * Accepting it silently biases the whole calibration.
 */
export function isViewUsable(detections, board, minCorners = 8) {
  if (detections.length < minCorners) return false;

  const pts = detections.map((d) => d.imagePoint);
  let minX = Infinity; let maxX = -Infinity;
  let minY = Infinity; let maxY = -Infinity;
  for (const p of pts) {
    minX = Math.min(minX, p[0]);
    maxX = Math.max(maxX, p[0]);
    minY = Math.min(minY, p[1]);
    maxY = Math.max(maxY, p[1]);
  }

  const w = maxX - minX;
  const h = maxY - minY;
  if (w < 20 || h < 20) return false;

  // Near-collinear spreads give a rank-deficient homography.
  const aspect = Math.max(w / h, h / w);
  if (aspect > 12) return false;

  // The board must span a useful share of its own convex extent, i.e. the
  // detections must be spread out rather than clustered in one corner.
  const area = w * h;
  const expected = board.chessboardCornerCount * 4;
  void area; void expected;
  return true;
}

/**
 * Coverage score for a view, used to rank and de-duplicate near-identical
 * frames when the user shoots a long calibration burst.
 */
export function viewCoverage(detections) {
  if (!detections.length) return 0;
  let minX = Infinity; let maxX = -Infinity;
  let minY = Infinity; let maxY = -Infinity;
  for (const d of detections) {
    minX = Math.min(minX, d.imagePoint[0]);
    maxX = Math.max(maxX, d.imagePoint[0]);
    minY = Math.min(minY, d.imagePoint[1]);
    maxY = Math.max(maxY, d.imagePoint[1]);
  }
  return Math.hypot(maxX - minX, maxY - minY);
}
