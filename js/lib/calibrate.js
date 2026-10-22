/**
 * calibrate.js — recover a phone camera's intrinsics and lens distortion from
 * views of a printed ChArUco board.
 *
 * Method
 *   1. Initialise: for each view, fit the board->image homography, then read the
 *      focal length off the two Zhang constraints that hold when the principal
 *      point is at the image centre and pixels are square (both true to within a
 *      few percent on phone cameras). Take the median across views.
 *   2. Refine: joint Levenberg-Marquardt over intrinsics, distortion and every
 *      view's pose simultaneously, minimising reprojection error.
 *
 * Why this shape: the closed-form Zhang extraction is elegant but fragile on
 * real data — it can return a negative f^2 from one unlucky view and there is no
 * obvious symptom. A robust initialisation plus a joint nonlinear refine gets
 * the same answer without that failure mode, and the LM step is what actually
 * buys the accuracy.
 *
 * Units: object points are millimetres, so every recovered translation is in mm.
 *
 * A view is { objectPoints: [[x,y,0],...], imagePoints: [[u,v],...] } where the
 * object points are the board's chessboard corners in its own plane.
 */

import { levenbergMarquardt, transpose, matMul } from './linalg.js';
import {
  homographyFromPoints, projectPoint, rodrigues, rodriguesInverse, makeK,
} from './camera.js';

/**
 * Robust initial guess for the intrinsic matrix.
 *
 * With the principal point at the image centre and fx = fy = f, the image of the
 * absolute conic is diag(1/f^2, 1/f^2, 1), and the two Zhang orthogonality
 * constraints become linear in f^2. Each view yields two estimates; we take the
 * median so a single bad view cannot drag the whole calibration.
 */
export function initialIntrinsics(views, imageSize) {
  const [w, h] = imageSize;
  const cx = w / 2;
  const cy = h / 2;
  const fSquares = [];

  for (const view of views) {
    if (!view || view.objectPoints.length < 4) continue;

    const src = view.objectPoints.map((p) => [p[0], p[1]]);
    const dst = view.imagePoints.map((p) => [p[0] - cx, p[1] - cy]);

    let H;
    try {
      H = homographyFromPoints(src, dst);
    } catch {
      continue;
    }

    // The Zhang constraints are written in terms of the COLUMNS of H, since
    // H = K [r1 r2 t] stacks the rotation columns side by side. Reading rows
    // here instead silently returns a focal length off by orders of magnitude.
    const h1 = [H[0][0], H[1][0], H[2][0]];
    const h2 = [H[0][1], H[1][1], H[2][1]];

    // Constraint A:  h1^T w h2 = 0        (with w = diag(1/f^2, 1/f^2, 1))
    if (Math.abs(h1[2] * h2[2]) > 1e-12) {
      const f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / (h1[2] * h2[2]);
      if (f2 > 0 && Number.isFinite(f2)) fSquares.push(f2);
    }
    // Constraint B:  h1^T w h1 = h2^T w h2
    if (Math.abs(h2[2] * h2[2] - h1[2] * h1[2]) > 1e-12) {
      const f2 = (h1[0] * h1[0] + h1[1] * h1[1] - h2[0] * h2[0] - h2[1] * h2[1]) /
                 (h2[2] * h2[2] - h1[2] * h1[2]);
      if (f2 > 0 && Number.isFinite(f2)) fSquares.push(f2);
    }
  }

  let f;
  if (fSquares.length) {
    fSquares.sort((a, b) => a - b);
    const mid = Math.floor(fSquares.length / 2);
    const med = fSquares.length % 2
      ? fSquares[mid]
      : 0.5 * (fSquares[mid - 1] + fSquares[mid]);
    f = Math.sqrt(med);
  } else {
    // Nothing usable — fall back to a typical phone's ~68 degree horizontal FOV.
    f = w * 0.75;
  }

  // Guard against a wildly implausible focal length from a degenerate set of views.
  f = Math.min(Math.max(f, 0.15 * Math.max(w, h)), 8 * Math.max(w, h));

  return makeK(f, f, cx, cy);
}

/** Per-view pose initialisation from the board homography. */
function initialPoses(views, K, imageSize) {
  const cx = imageSize[0] / 2;
  const cy = imageSize[1] / 2;
  const poses = [];

  for (const view of views) {
    const src = view.objectPoints.map((p) => [p[0], p[1]]);
    const dst = view.imagePoints.map((p) => [p[0], p[1]]);

    let pose = null;
    try {
      const H = homographyFromPoints(src, dst);
      const Kinv = [
        [1 / K[0][0], 0, -K[0][2] / K[0][0]],
        [0, 1 / K[1][1], -K[1][2] / K[1][1]],
        [0, 0, 1],
      ];
      const A = matMul(Kinv, H);
      const a1 = [A[0][0], A[1][0], A[2][0]];
      const a2 = [A[0][1], A[1][1], A[2][1]];
      const a3 = [A[0][2], A[1][2], A[2][2]];
      const n1 = Math.hypot(...a1);
      const n2 = Math.hypot(...a2);
      const lambda = 2 / (n1 + n2);
      const sign = a3[2] * lambda < 0 ? -1 : 1;
      const s = lambda * sign;

      let r1 = a1.map((v) => v * s);
      let r2 = a2.map((v) => v * s);
      const t = a3.map((v) => v * s);

      // Orthonormalise by Gram-Schmidt, then complete the frame.
      const n1l = Math.hypot(...r1) || 1;
      r1 = r1.map((v) => v / n1l);
      const proj = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2];
      r2 = r2.map((v, i) => v - proj * r1[i]);
      const n2l = Math.hypot(...r2) || 1;
      r2 = r2.map((v) => v / n2l);
      const r3 = [
        r1[1] * r2[2] - r1[2] * r2[1],
        r1[2] * r2[0] - r1[0] * r2[2],
        r1[0] * r2[1] - r1[1] * r2[0],
      ];

      const R = [
        [r1[0], r2[0], r3[0]],
        [r1[1], r2[1], r3[1]],
        [r1[2], r2[2], r3[2]],
      ];
      pose = { R, t, rvec: rodriguesInverse(R) };
    } catch {
      pose = null;
    }

    if (!pose || !Number.isFinite(pose.t[0]) || pose.t[2] <= 0) {
      // Unusable view: park the board one metre in front, facing the camera.
      pose = { R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], t: [0, 0, 1000], rvec: [0, 0, 0] };
    }
    poses.push(pose);
  }
  void cx; void cy;
  return poses;
}

/**
 * Calibrate from a set of board views.
 *
 * opts:
 *   refineIntrinsics  (default true)
 *   refineDistortion  (default true)
 *   refinePoses       (default true)
 *   useK3             (default false — the third radial term needs many views
 *                      and destabilises small sets)
 *   maxIter           (default 60)
 *
 * Returns { K, dist, rms, perView, imageSize, iterations, warnings, viewCount }.
 */
export function calibrateCamera(views, imageSize, opts = {}) {
  const {
    refineIntrinsics = true,
    refineDistortion = true,
    refinePoses = true,
    useK3 = false,
    maxIter = 60,
  } = opts;

  const usable = views.filter((v) => v && v.objectPoints.length >= 4);
  if (usable.length < 3) {
    throw new Error(`calibration needs at least 3 usable views, got ${usable.length}`);
  }

  const K0 = initialIntrinsics(usable, imageSize);
  const poses0 = initialPoses(usable, K0, imageSize);

  const nDist = useK3 ? 5 : 4;
  const full = [
    K0[0][0], K0[1][1], K0[0][2], K0[1][2],
    ...new Array(nDist).fill(0),
    ...poses0.flatMap((p) => [...p.rvec, ...p.t]),
  ];

  const free = [
    refineIntrinsics, refineIntrinsics, refineIntrinsics, refineIntrinsics,
    ...new Array(nDist).fill(refineDistortion),
    ...usable.flatMap(() => new Array(6).fill(refinePoses)),
  ];

  const freeIdx = [];
  for (let i = 0; i < free.length; i++) if (free[i]) freeIdx.push(i);

  const unpack = (reduced) => {
    const f = full.slice();
    for (let k = 0; k < freeIdx.length; k++) f[freeIdx[k]] = reduced[k];
    return f;
  };
  const pack = (f) => freeIdx.map((i) => f[i]);

  const residualFn = (reduced) => {
    const p = unpack(reduced);
    const K = makeK(p[0], p[1], p[2], p[3]);
    const dist = p.slice(4, 4 + nDist);
    const res = [];
    let base = 4 + nDist;

    for (let vi = 0; vi < usable.length; vi++) {
      const rvec = [p[base], p[base + 1], p[base + 2]];
      const t = [p[base + 3], p[base + 4], p[base + 5]];
      base += 6;

      const R = rodrigues(rvec);
      const view = usable[vi];
      for (let i = 0; i < view.objectPoints.length; i++) {
        const proj = projectPoint(K, dist, R, t, view.objectPoints[i]);
        if (!proj) {
          // Point fell behind the camera: a huge but finite penalty steers the
          // optimiser back rather than producing NaN and killing the whole run.
          res.push(50, 50);
          continue;
        }
        res.push(proj[0] - view.imagePoints[i][0], proj[1] - view.imagePoints[i][1]);
      }
    }
    return res;
  };

  let x = pack(full);
  let lm = { x, cost: NaN, iterations: 0 };

  if (freeIdx.length > 0) {
    lm = levenbergMarquardt(residualFn, x, { maxIter, fdStep: 1e-6 });
    x = lm.x;
  }

  const p = unpack(x);
  const K = makeK(p[0], p[1], p[2], p[3]);
  const dist = p.slice(4, 4 + nDist);
  const distFull = [...dist, ...new Array(5 - nDist).fill(0)];

  const perView = [];
  let base = 4 + nDist;
  for (let vi = 0; vi < usable.length; vi++) {
    const rvec = [p[base], p[base + 1], p[base + 2]];
    const t = [p[base + 3], p[base + 4], p[base + 5]];
    base += 6;

    const R = rodrigues(rvec);
    let sum = 0;
    let n = 0;
    let maxErr = 0;
    for (let i = 0; i < usable[vi].objectPoints.length; i++) {
      const proj = projectPoint(K, distFull, R, t, usable[vi].objectPoints[i]);
      if (!proj) { n++; sum += 1e6; continue; }
      const dx = proj[0] - usable[vi].imagePoints[i][0];
      const dy = proj[1] - usable[vi].imagePoints[i][1];
      const e = dx * dx + dy * dy;
      sum += e;
      maxErr = Math.max(maxErr, Math.sqrt(e));
      n++;
    }
    perView.push({
      index: vi,
      rms: n ? Math.sqrt(sum / n) : Infinity,
      maxError: maxErr,
      pointCount: n,
      R,
      t,
      distanceMm: Math.hypot(t[0], t[1], t[2]),
    });
  }

  const totalPts = perView.reduce((a, v) => a + v.pointCount, 0);
  const rms = Math.sqrt(
    perView.reduce((a, v) => a + (v.rms ** 2) * v.pointCount, 0) / Math.max(1, totalPts)
  );

  const result = {
    K,
    dist: distFull,
    rms,
    perView,
    imageSize,
    iterations: lm.iterations,
    viewCount: usable.length,
    totalPoints: totalPts,
    refine: { refineIntrinsics, refineDistortion, refinePoses, useK3 },
  };
  result.warnings = calibrationWarnings(result);
  return result;
}

/**
 * Flag calibrations that are technically converged but physically implausible.
 *
 * A silently bad calibration is the worst failure this app can have: every
 * measurement stays self-consistent and wrong, with no error surfaced anywhere.
 * These checks are the guard against that.
 */
export function calibrationWarnings(result) {
  const w = [];
  const [iw, ih] = result.imageSize;
  const maxDim = Math.max(iw, ih);
  const fx = result.K[0][0];
  const fy = result.K[1][1];
  const cx = result.K[0][2];
  const cy = result.K[1][2];

  if (result.viewCount < 6) {
    w.push(`Only ${result.viewCount} views — aim for 10-20 covering the corners of the frame.`);
  }
  if (result.rms > 1.0) {
    w.push(`Reprojection RMS is ${result.rms.toFixed(2)} px (want < 0.5). Re-shoot sharper, flatter views.`);
  }
  if (Math.abs(fx - fy) / Math.max(fx, fy) > 0.05) {
    w.push('fx and fy differ by more than 5% — unusual for a phone; check the board printed at true scale.');
  }
  if (fx < 0.2 * maxDim || fx > 5 * maxDim) {
    w.push(`Focal length ${fx.toFixed(0)} px is implausible for a ${iw}x${ih} image.`);
  }
  if (Math.abs(cx - iw / 2) > 0.2 * iw || Math.abs(cy - ih / 2) > 0.2 * ih) {
    w.push('Principal point is far from the image centre — calibration may be under-constrained.');
  }
  const [k1, k2] = result.dist;
  if (Math.abs(k1) > 0.5 || Math.abs(k2) > 1.0) {
    w.push(`Distortion k1=${k1.toFixed(3)} k2=${k2.toFixed(3)} is extreme — likely a bad board scale or a bad view.`);
  }

  // A view that is far worse than its peers usually means motion blur or a
  // mis-detected corner; dropping it is better than letting it skew the fit.
  const goodRms = result.perView.filter((v) => Number.isFinite(v.rms)).map((v) => v.rms);
  if (goodRms.length > 3) {
    const mean = goodRms.reduce((a, b) => a + b, 0) / goodRms.length;
    const bad = result.perView.filter((v) => v.rms > Math.max(2.0, 3 * mean));
    if (bad.length) {
      w.push(`View(s) ${bad.map((v) => v.index + 1).join(', ')} fit poorly — consider re-shooting them.`);
    }
  }
  return w;
}

/** Horizontal field of view in degrees, for a sanity readout in the UI. */
export function horizontalFovDeg(result) {
  const fx = result.K[0][0];
  const w = result.imageSize[0];
  return (2 * Math.atan(w / (2 * fx)) * 180) / Math.PI;
}

export { transpose };
