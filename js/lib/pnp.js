/**
 * pnp.js — perspective-n-point for planar targets.
 *
 * Every fiducial in this app is flat: a QR sticker or an ArUco marker lies on
 * one plane of the parcel. That is a lucky special case, because for coplanar
 * points the pose has a closed-form solution — the homography factorises as
 * H = K [r1 r2 t], so the rotation and translation fall out algebraically with
 * no iteration and no initial guess.
 *
 * The closed form is then refined by Levenberg-Marquardt against the true
 * projection model (including lens distortion), which is what turns a
 * geometrically valid pose into a metrologically usable one.
 *
 * A note on ambiguity: a planar target has a genuine two-fold pose ambiguity
 * that worsens as the target shrinks or tilts in frame. It is resolved here by
 * requiring positive depth. When a device orientation is available it should be
 * used to confirm that choice — that is what the optional `gravityPrior` is for.
 */

import { levenbergMarquardt } from './linalg.js';
import {
  homographyFromPoints, poseFromHomography, projectPoint,
  rodrigues, rodriguesInverse, undistortPixel, makeK,
} from './camera.js';

/**
 * Solve the pose of a planar target.
 *
 * objectPoints: [[x, y, 0], ...] in millimetres, in the target's own frame
 * imagePoints:  [[u, v], ...] in pixels, same order
 *
 * opts:
 *   refine     (default true)  — LM refinement against the distorted model
 *   maxIter    (default 50)
 *   gravityPrior — optional unit vector giving the expected target normal in
 *                  camera coordinates (from the device IMU). Used only to
 *                  choose between the two ambiguous branches.
 *
 * Returns { R, t, rvec, tvec, rms, maxError, pointCount } or null.
 */
export function solvePnPPlanar(objectPoints, imagePoints, K, dist, opts = {}) {
  const { refine = true, maxIter = 50, gravityPrior = null } = opts;
  if (objectPoints.length < 4) return null;

  // The linear stage assumes a pinhole, so undistort first.
  const undistorted = imagePoints.map((p) => undistortPixel(K, dist, p));

  let H;
  try {
    H = homographyFromPoints(
      objectPoints.map((p) => [p[0], p[1]]),
      undistorted
    );
  } catch {
    return null;
  }

  const pose = poseFromHomography(H, K, objectPoints, undistorted);
  if (!pose) return null;

  let rvec = rodriguesInverse(pose.R);
  let tvec = pose.t.slice();

  // Resolve the sign branch: the target must be in front of the camera.
  if (tvec[2] < 0) {
    rvec = rvec.map((v) => -v);
    tvec = tvec.map((v) => -v);
  }

  // If the IMU says which way the target faces, prefer the branch that agrees.
  // This is the cheapest fix for the planar flip ambiguity.
  if (gravityPrior) {
    const R = rodrigues(rvec);
    const normal = [R[0][2], R[1][2], R[2][2]];
    const dot = normal[0] * gravityPrior[0] + normal[1] * gravityPrior[1] + normal[2] * gravityPrior[2];
    if (dot < 0) {
      rvec = rvec.map((v) => -v);
      tvec = tvec.map((v) => -v);
    }
  }

  const Kc = makeK(K[0][0], K[1][1], K[0][2], K[1][2]);

  if (refine) {
    const residual = (params) => {
      const R = rodrigues([params[0], params[1], params[2]]);
      const t = [params[3], params[4], params[5]];
      const out = [];
      for (let i = 0; i < objectPoints.length; i++) {
        const p = projectPoint(Kc, dist, R, t, objectPoints[i]);
        if (!p) { out.push(50, 50); continue; }
        out.push(p[0] - imagePoints[i][0], p[1] - imagePoints[i][1]);
      }
      return out;
    };

    const lm = levenbergMarquardt(residual, [...rvec, ...tvec], { maxIter, fdStep: 1e-6 });
    rvec = lm.x.slice(0, 3);
    tvec = lm.x.slice(3, 6);
  }

  const R = rodrigues(rvec);

  // Final error report, measured against the real observations.
  let sum = 0;
  let maxError = 0;
  for (let i = 0; i < objectPoints.length; i++) {
    const p = projectPoint(Kc, dist, R, tvec, objectPoints[i]);
    if (!p) { sum += 1e6; continue; }
    const e = Math.hypot(p[0] - imagePoints[i][0], p[1] - imagePoints[i][1]);
    sum += e * e;
    maxError = Math.max(maxError, e);
  }

  return {
    R,
    t: tvec,
    rvec,
    tvec,
    rms: Math.sqrt(sum / objectPoints.length),
    maxError,
    pointCount: objectPoints.length,
  };
}

/**
 * The plane a solved pose describes, in camera coordinates.
 * The target's own z = 0 plane maps to a plane through t with normal R's third
 * column.
 */
export function posePlane(pose) {
  return {
    point: pose.t.slice(),
    normal: [pose.R[0][2], pose.R[1][2], pose.R[2][2]],
  };
}

/**
 * Angular difference between two plane normals, in degrees, ignoring sign —
 * two faces of a box viewed from either side are still the same orientation.
 */
export function normalAngleDeg(a, b) {
  const d = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
  return (Math.acos(Math.min(1, Math.max(-1, d))) * 180) / Math.PI;
}
