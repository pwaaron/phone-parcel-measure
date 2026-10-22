/**
 * camera.js — the pinhole camera model, lens distortion, planar homography and
 * pose recovery.
 *
 * This is the mathematical heart of the app. Everything downstream (calibration,
 * PnP, tap-to-measure) is expressed in terms of the primitives here.
 *
 * Camera model (OpenCV convention)
 *   K    = [[fx, 0, cx],
 *           [ 0, fy, cy],
 *           [ 0,  0,  1]]
 *   dist = [k1, k2, p1, p2, k3]      Brown-Conrady radial + tangential
 *
 * A 3D point X (object frame, millimetres) maps to pixels by
 *   x_cam = R X + t,  then normalise,  then distort,  then apply K.
 *
 * Units: the object frame is in MILLIMETRES throughout. The calibration board's
 * square size is given in mm, so every pose translation comes out in mm, and
 * measured distances are mm with no conversion step to get wrong.
 */

import {
  zeros, eye, matMul, matVec, transpose, dot, norm, nullVector,
  jacobiEigenSymmetric, invert,
} from './linalg.js';

/** Build an intrinsic matrix from its four free parameters. */
export function makeK(fx, fy, cx, cy) {
  return [
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1],
  ];
}

/** Convert a rotation vector (axis * angle, radians) to a 3x3 rotation matrix. */
export function rodrigues(r) {
  const theta = norm(r);
  if (theta < 1e-12) {
    // First-order limit: R ~ I + [r]_x
    return [
      [1, -r[2], r[1]],
      [r[2], 1, -r[0]],
      [-r[1], r[0], 1],
    ];
  }
  const k = [r[0] / theta, r[1] / theta, r[2] / theta];
  const K = [
    [0, -k[2], k[1]],
    [k[2], 0, -k[0]],
    [-k[1], k[0], 0],
  ];
  const K2 = matMul(K, K);
  const s = Math.sin(theta);
  const c = Math.cos(theta);
  const R = zeros(3, 3);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) R[i][j] = (i === j ? 1 : 0) + s * K[i][j] + (1 - c) * K2[i][j];
  }
  return R;
}

/** Inverse of rodrigues(): rotation matrix back to a rotation vector. */
export function rodriguesInverse(R) {
  const tr = R[0][0] + R[1][1] + R[2][2];
  const cosTheta = Math.max(-1, Math.min(1, (tr - 1) / 2));
  const theta = Math.acos(cosTheta);
  if (theta < 1e-9) return [0, 0, 0];
  if (Math.abs(Math.PI - theta) < 1e-6) {
    // Near 180 degrees the antisymmetric part vanishes; read the axis off the
    // diagonal instead. R = 2 aa^T - I  =>  a_i = sqrt((R_ii + 1) / 2).
    const a = [
      Math.sqrt(Math.max(0, (R[0][0] + 1) / 2)),
      Math.sqrt(Math.max(0, (R[1][1] + 1) / 2)),
      Math.sqrt(Math.max(0, (R[2][2] + 1) / 2)),
    ];
    // Fix the signs using the off-diagonal terms.
    if (R[0][1] + R[1][0] < 0) a[1] = -a[1];
    if (R[0][2] + R[2][0] < 0) a[2] = -a[2];
    const n = norm(a) || 1;
    return [a[0] / n * theta, a[1] / n * theta, a[2] / n * theta];
  }
  // The antisymmetric part gives the unit axis; the rotation vector is that
  // axis scaled by theta, not the axis itself.
  const f = theta / (2 * Math.sin(theta));
  return [
    f * (R[2][1] - R[1][2]),
    f * (R[0][2] - R[2][0]),
    f * (R[1][0] - R[0][1]),
  ];
}

/**
 * Nearest rotation matrix to A, via polar decomposition R = A (A^T A)^(-1/2).
 *
 * A homography decomposition hands back a matrix whose columns are *nearly*
 * orthonormal; the residual error is pure noise amplification. Snapping to the
 * nearest true rotation removes it and is what makes the recovered pose usable
 * for measurement rather than merely plausible.
 */
export function orthonormalize3(A) {
  const At = transpose(A);
  const AtA = matMul(At, A);
  const { values, vectors } = jacobiEigenSymmetric(AtA);

  // V * diag(1/sqrt(d)) * V^T  ==  (A^T A)^(-1/2)
  const V = zeros(3, 3);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) V[i][j] = vectors[j][i];

  const invSqrt = zeros(3, 3);
  for (let i = 0; i < 3; i++) {
    const d = values[i];
    invSqrt[i][i] = d > 1e-18 ? 1 / Math.sqrt(d) : 0;
  }

  let R = matMul(A, matMul(V, matMul(invSqrt, transpose(V))));

  // Force a right-handed frame. A reflection here would mirror the whole scene
  // and silently invert every measured distance.
  const det =
    R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) -
    R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) +
    R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
  if (det < 0) {
    R = R.map((row) => [row[0], row[1], -row[2]]);
  }
  return R;
}

/** Apply the Brown-Conrady distortion model to a normalised (x, y) point. */
export function distortNormalized(x, y, dist) {
  const [k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0] = dist;
  const r2 = x * x + y * y;
  const radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
  const xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
  const yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
  return [xd, yd];
}

/**
 * Invert the distortion model: given a distorted normalised point, recover the
 * undistorted one. There is no closed form, so iterate a fixed point — it
 * converges in a handful of steps for realistic phone lenses.
 */
export function undistortNormalized(xd, yd, dist, iterations = 30) {
  let x = xd;
  let y = yd;
  for (let i = 0; i < iterations; i++) {
    const [px, py] = distortNormalized(x, y, dist);
    const ex = px - xd;
    const ey = py - yd;
    x -= ex;
    y -= ey;
    if (Math.abs(ex) + Math.abs(ey) < 1e-14) break;
  }
  return [x, y];
}

/** Project a single object-frame point (mm) into pixels. */
export function projectPoint(K, dist, R, t, X) {
  const xc = R[0][0] * X[0] + R[0][1] * X[1] + R[0][2] * X[2] + t[0];
  const yc = R[1][0] * X[0] + R[1][1] * X[1] + R[1][2] * X[2] + t[1];
  const zc = R[2][0] * X[0] + R[2][1] * X[1] + R[2][2] * X[2] + t[2];
  if (zc <= 1e-9) return null; // behind the camera
  const x = xc / zc;
  const y = yc / zc;
  const [xd, yd] = distortNormalized(x, y, dist);
  return [K[0][0] * xd + K[0][2], K[1][1] * yd + K[1][2]];
}

/** Project many object-frame points at once; null for any behind the camera. */
export function projectPoints(K, dist, R, t, Xs) {
  return Xs.map((X) => projectPoint(K, dist, R, t, X));
}

/**
 * Pixel + intrinsics + distortion -> the undistorted ray direction in camera
 * coordinates, normalised so z = 1.
 *
 * This is the bridge from a screen tap to a 3D ray, and therefore the first half
 * of every measurement.
 */
export function pixelToNormalizedRay(K, dist, u, v) {
  const xd = (u - K[0][2]) / K[0][0];
  const yd = (v - K[1][2]) / K[1][1];
  const [x, y] = undistortNormalized(xd, yd, dist);
  return [x, y, 1];
}

/**
 * Pixel -> undistorted pixel.
 *
 * The linear solvers (homography DLT, planar pose) assume a pinhole with no
 * lens distortion. Feeding them raw pixels from a wide phone lens biases the
 * result, so every observation is undistorted on the way in.
 */
export function undistortPixel(K, dist, p) {
  const [x, y] = pixelToNormalizedRay(K, dist, p[0], p[1]);
  return [K[0][0] * x + K[0][2], K[1][1] * y + K[1][2]];
}

/**
 * Hartley normalisation: shift the centroid to the origin and scale so the mean
 * distance from it is sqrt(2).
 *
 * Skipping this is the classic DLT failure — raw pixel coordinates span
 * hundreds, the design matrix becomes badly conditioned, and the homography
 * degrades without any obvious symptom.
 */
function normalizePoints2D(pts) {
  const n = pts.length;
  let mx = 0;
  let my = 0;
  for (const p of pts) { mx += p[0]; my += p[1]; }
  mx /= n;
  my /= n;

  let meanDist = 0;
  for (const p of pts) meanDist += Math.hypot(p[0] - mx, p[1] - my);
  meanDist /= n;
  const s = meanDist > 1e-12 ? Math.SQRT2 / meanDist : 1;

  const T = [
    [s, 0, -s * mx],
    [0, s, -s * my],
    [0, 0, 1],
  ];
  const out = pts.map((p) => [s * (p[0] - mx), s * (p[1] - my)]);
  return { T, points: out };
}

/** Rescale a homography so that H[2][2] = 1 (when it is not near zero). */
export function normalizeH(H) {
  const s = H[2][2];
  if (Math.abs(s) < 1e-12) return H;
  return H.map((row) => row.map((v) => v / s));
}

/** Apply a homography to a 2D point. */
export function applyH(H, p) {
  const x = H[0][0] * p[0] + H[0][1] * p[1] + H[0][2];
  const y = H[1][0] * p[0] + H[1][1] * p[1] + H[1][2];
  const w = H[2][0] * p[0] + H[2][1] * p[1] + H[2][2];
  return [x / w, y / w];
}

/**
 * Planar homography from >= 4 point correspondences, by normalised DLT.
 * Returns H mapping src -> dst.
 */
export function homographyFromPoints(src, dst) {
  if (src.length !== dst.length) throw new Error('homographyFromPoints: length mismatch');
  if (src.length < 4) throw new Error('homographyFromPoints: need at least 4 points');

  const ns = normalizePoints2D(src);
  const nd = normalizePoints2D(dst);

  const A = [];
  for (let i = 0; i < ns.points.length; i++) {
    const [x, y] = ns.points[i];
    const [u, v] = nd.points[i];
    A.push([-x, -y, -1, 0, 0, 0, u * x, u * y, u]);
    A.push([0, 0, 0, -x, -y, -1, v * x, v * y, v]);
  }

  const h = nullVector(A);
  const Hn = [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], h[8]],
  ];

  // Undo the normalisation: H = T_dst^-1 * Hn * T_src
  const H = matMul(invert(nd.T), matMul(Hn, ns.T));
  return normalizeH(H);
}

/**
 * Recover the pose of a plane from its homography.
 *
 * For a target lying on Z = 0 the homography factorises as H = K [r1 r2 t], so
 * the first two columns of K^-1 H are the first two rotation columns up to a
 * common scale, and the third column is the translation. Orthonormalising the
 * pair and taking their cross product completes the rotation.
 *
 * The global sign is resolved by requiring the plane to be in FRONT of the
 * camera (positive depth), which is the physically meaningful branch.
 *
 * Returns { R, t, reprojectionRmse } with t in the same units as the object
 * points (millimetres, in this app).
 */
export function poseFromHomography(H, K, objectPoints, imagePoints) {
  const Kinv = invert(K);
  const A = matMul(Kinv, H); // columns: a1, a2, a3

  const a1 = [A[0][0], A[1][0], A[2][0]];
  const a2 = [A[0][1], A[1][1], A[2][1]];
  const a3 = [A[0][2], A[1][2], A[2][2]];

  const n1 = norm(a1);
  const n2 = norm(a2);
  const lambda = 2 / (n1 + n2);
  if (!Number.isFinite(lambda) || lambda === 0) return null;

  // Sign: the plane must sit at positive depth along the camera's z axis.
  const sign = a3[2] * lambda < 0 ? -1 : 1;
  const scale = lambda * sign;

  let r1 = a1.map((v) => v * scale);
  let r2 = a2.map((v) => v * scale);
  const t = a3.map((v) => v * scale);

  const Rraw = [
    [r1[0], r2[0], 0],
    [r1[1], r2[1], 0],
    [r1[2], r2[2], 0],
  ];
  const Rorth = orthonormalize3(Rraw);
  r1 = [Rorth[0][0], Rorth[1][0], Rorth[2][0]];
  r2 = [Rorth[0][1], Rorth[1][1], Rorth[2][1]];

  // Third column = r1 x r2, which keeps the frame right-handed by construction.
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

  let rmse = 0;
  if (objectPoints && imagePoints) {
    let sum = 0;
    let n = 0;
    for (let i = 0; i < objectPoints.length; i++) {
      const p = projectPoint(K, [0, 0, 0, 0, 0], R, t, objectPoints[i]);
      if (!p) { sum += 1e6; n++; continue; }
      sum += (p[0] - imagePoints[i][0]) ** 2 + (p[1] - imagePoints[i][1]) ** 2;
      n++;
    }
    rmse = n ? Math.sqrt(sum / n) : 0;
  }

  return { R, t, reprojectionRmse: rmse };
}

/** Distance between two 3D points. */
export function distance3D(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

/**
 * Intersect a ray (origin + s * direction, s > 0) with a plane given by a point
 * and unit normal. Returns null when the ray is parallel to the plane or the hit
 * is behind the camera.
 *
 * This is what turns a screen tap into a 3D point on the parcel's surface.
 */
export function rayPlaneIntersect(origin, direction, planePoint, planeNormal) {
  const denom = dot(planeNormal, direction);
  if (Math.abs(denom) < 1e-12) return null;
  const diff = [
    planePoint[0] - origin[0],
    planePoint[1] - origin[1],
    planePoint[2] - origin[2],
  ];
  const s = dot(planeNormal, diff) / denom;
  if (s <= 0) return null;
  return [
    origin[0] + s * direction[0],
    origin[1] + s * direction[1],
    origin[2] + s * direction[2],
  ];
}

/** Unit normal of a plane given three points, oriented to face the camera. */
export function planeNormalFrom3Points(a, b, c, towardPoint) {
  const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  let n = [
    u[1] * v[2] - u[2] * v[1],
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0],
  ];
  const len = norm(n);
  if (len < 1e-12) return null;
  n = n.map((x) => x / len);
  if (towardPoint) {
    const d = [
      towardPoint[0] - a[0],
      towardPoint[1] - a[1],
      towardPoint[2] - a[2],
    ];
    if (dot(n, d) < 0) n = n.map((x) => -x);
  }
  return n;
}

/** Column-major 3x4 [R|t], as used by OpenCV's projectPoints. */
export function RtTo34(R, t) {
  return [
    [R[0][0], R[0][1], R[0][2], t[0]],
    [R[1][0], R[1][1], R[1][2], t[1]],
    [R[2][0], R[2][1], R[2][2], t[2]],
  ];
}

export { eye };
