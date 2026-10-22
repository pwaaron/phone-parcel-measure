/**
 * subpix.js — sub-pixel corner refinement.
 *
 * A traced contour corner lands on a whole pixel, so it carries up to half a
 * pixel of error. At 0.5 m that half-pixel is already a millimetre of scale
 * error, and it feeds straight into the pose. Refining corners to ~0.05 px is
 * one of the cheapest accuracy wins in the pipeline.
 *
 * Method is the classic gradient-weighted centroid (OpenCV's cornerSubPix): the
 * true corner is where the intensity gradient is maximally inconsistent with a
 * flat edge, so we solve for the point that best explains the local gradient
 * field, and iterate to convergence.
 */

import { sampleBilinear } from './image.js';

/** Central-difference gradient at a fractional position. */
function gradientAt(gray, width, height, x, y) {
  const gx = (sampleBilinear(gray, width, height, x + 1, y) -
              sampleBilinear(gray, width, height, x - 1, y)) / 2;
  const gy = (sampleBilinear(gray, width, height, x, y + 1) -
              sampleBilinear(gray, width, height, x, y - 1)) / 2;
  return [gx, gy];
}

/**
 * Refine one corner. Returns [x, y] moved to sub-pixel accuracy.
 *
 * winSize is the half-width of the sampling window in pixels; 5 is the usual
 * choice — big enough to see real structure, small enough not to be dragged by
 * a neighbouring corner.
 */
export function cornerSubPix(gray, width, height, corner, winSize = 5, maxIter = 20, eps = 0.01) {
  let cx = corner[0];
  let cy = corner[1];

  for (let iter = 0; iter < maxIter; iter++) {
    let a11 = 0;
    let a12 = 0;
    let a22 = 0;
    let b1 = 0;
    let b2 = 0;

    for (let dy = -winSize; dy <= winSize; dy++) {
      for (let dx = -winSize; dx <= winSize; dx++) {
        const x = cx + dx;
        const y = cy + dy;
        if (x < 1 || y < 1 || x > width - 2 || y > height - 2) continue;

        const [gx, gy] = gradientAt(gray, width, height, x, y);
        const gxx = gx * gx;
        const gxy = gx * gy;
        const gyy = gy * gy;

        a11 += gxx;
        a12 += gxy;
        a22 += gyy;
        b1 += gxx * x + gxy * y;
        b2 += gxy * x + gyy * y;
      }
    }

    const det = a11 * a22 - a12 * a12;
    if (Math.abs(det) < 1e-12) break;

    const nx = (a22 * b1 - a12 * b2) / det;
    const ny = (a11 * b2 - a12 * b1) / det;

    if (!Number.isFinite(nx) || !Number.isFinite(ny)) break;
    if (Math.abs(nx - cx) + Math.abs(ny - cy) < eps) { cx = nx; cy = ny; break; }

    cx = nx;
    cy = ny;
  }

  return [cx, cy];
}

/**
 * Refine a quadrilateral's four corners, with a guard.
 *
 * A corner can be dragged far off by a neighbouring strong edge; if the refined
 * point moves more than maxShift from where the contour put it, the contour was
 * probably right and the refinement is trusting the wrong structure. Keeping the
 * original in that case is strictly safer than accepting a wild jump.
 */
export function refineQuadSubPix(gray, width, height, quad, winSize = 5, maxShift = 3) {
  return quad.map((corner) => {
    const [nx, ny] = cornerSubPix(gray, width, height, corner, winSize);
    if (Math.hypot(nx - corner[0], ny - corner[1]) > maxShift) return [corner[0], corner[1]];
    return [nx, ny];
  });
}

/**
 * Sub-pixel refinement of a ChArUco chessboard corner.
 *
 * Chessboard corners sit where four alternating squares meet. The gradient
 * centroid converges well here because the structure is symmetric, but the
 * window must be small enough to stay inside one square.
 */
export function refineChessboardCorner(gray, width, height, corner, squarePx) {
  const win = Math.max(2, Math.min(7, Math.round(squarePx * 0.35)));
  return cornerSubPix(gray, width, height, corner, win, 25, 0.005);
}
