/**
 * measure.js — turning detected fiducials into measurements.
 *
 * The chain, end to end:
 *
 *   QR sticker (known printed size) -> plane pose in camera coordinates
 *   -> screen tap -> undistorted ray -> ray/plane intersection -> 3D point
 *   -> distance between 3D points
 *
 * Because every point is recovered in one shared camera frame, distances work
 * ACROSS planes as well as within one. That is what makes it possible to
 * measure a box edge that spans two faces, which a flat homography cannot do.
 *
 * All distances are millimetres.
 */

import { solvePnPPlanar, posePlane, normalAngleDeg } from './pnp.js';
import {
  pixelToNormalizedRay, rayPlaneIntersect, distance3D, undistortPixel,
} from './camera.js';

/**
 * The four object-frame corners of a QR sticker of side `sizeMm`.
 * Order matches the detected corner order: top-left, top-right, bottom-right,
 * bottom-left, as reported by the QR detector.
 */
export function qrObjectCorners(sizeMm) {
  return [
    [0, 0, 0],
    [sizeMm, 0, 0],
    [sizeMm, sizeMm, 0],
    [0, sizeMm, 0],
  ];
}

/**
 * Solve one plane per detected QR code.
 *
 * Returns [{ qr, pose, plane, corners3D, rms }] with the ones that failed to
 * solve dropped.
 */
export function solveQrPlanes(qrDetections, K, dist, opts = {}) {
  const out = [];

  for (const qr of qrDetections) {
    if (!qr.sizeMm || qr.sizeMm <= 0) continue;
    if (!qr.corners || qr.corners.length !== 4) continue;

    const obj = qrObjectCorners(qr.sizeMm);
    const pose = solvePnPPlanar(obj, qr.corners, K, dist, opts);
    if (!pose) continue;

    // A pose that reprojects badly means the corners were mis-detected; using
    // it would silently corrupt every measurement taken on this plane.
    if (!Number.isFinite(pose.rms) || pose.rms > (opts.maxRmsPx ?? 3.0)) continue;

    out.push({
      qr,
      pose,
      plane: posePlane(pose),
      corners3D: obj.map((o) => applyPose(pose.R, pose.t, o)),
      rms: pose.rms,
    });
  }

  return out;
}

/** Apply a pose to a point in the target's frame, giving camera coordinates. */
export function applyPose(R, t, X) {
  return [
    R[0][0] * X[0] + R[0][1] * X[1] + R[0][2] * X[2] + t[0],
    R[1][0] * X[0] + R[1][1] * X[1] + R[1][2] * X[2] + t[1],
    R[2][0] * X[0] + R[2][1] * X[1] + R[2][2] * X[2] + t[2],
  ];
}

/**
 * Group QR planes that describe the same physical surface.
 *
 * Two stickers on one box face are coplanar: their normals agree and each one's
 * origin lies in the other's plane. Merging them is not cosmetic — a joint solve
 * over 8+ corners instead of 4 materially reduces the scale error, which is the
 * dominant error term in every measurement downstream.
 */
export function clusterPlanes(planes, opts = {}) {
  const { maxNormalAngleDeg = 6, maxOffsetMm = 4 } = opts;
  const clusters = [];
  const used = new Array(planes.length).fill(false);

  for (let i = 0; i < planes.length; i++) {
    if (used[i]) continue;
    const group = [planes[i]];
    used[i] = true;

    for (let j = i + 1; j < planes.length; j++) {
      if (used[j]) continue;
      if (sameSurface(planes[i], planes[j], maxNormalAngleDeg, maxOffsetMm)) {
        group.push(planes[j]);
        used[j] = true;
      }
    }
    clusters.push(group);
  }

  return clusters;
}

/** Whether two solved planes are close enough to be the same physical surface. */
function sameSurface(a, b, maxNormalAngleDeg, maxOffsetMm) {
  if (normalAngleDeg(a.plane.normal, b.plane.normal) > maxNormalAngleDeg) return false;

  // Distance from b's origin to a's plane.
  const n = a.plane.normal;
  const d = [
    b.plane.point[0] - a.plane.point[0],
    b.plane.point[1] - a.plane.point[1],
    b.plane.point[2] - a.plane.point[2],
  ];
  const offset = Math.abs(n[0] * d[0] + n[1] * d[1] + n[2] * d[2]);

  // Allow the offset to scale with distance, since a 1-degree pose error is a
  // larger positional error further from the camera.
  const dist = Math.hypot(a.plane.point[0], a.plane.point[1], a.plane.point[2]);
  return offset < Math.max(maxOffsetMm, dist * 0.01);
}

/**
 * Re-solve a cluster as a single plane using every member's corners jointly.
 *
 * The members sit at different places on the surface, so their combined corners
 * span a much wider baseline than any one sticker — and scale error falls as
 * the baseline grows.
 */
export function mergePlaneGroup(group, K, dist, opts = {}) {
  if (group.length === 0) return null;
  if (group.length === 1) return group[0];

  const anchor = group[0];
  const objectPoints = [];
  const imagePoints = [];

  for (const member of group) {
    const obj = qrObjectCorners(member.qr.sizeMm);
    // Express each member's corners in the anchor's frame using its own pose,
    // then re-project them back into the image. The joint solve then works in
    // one consistent frame.
    for (let i = 0; i < 4; i++) {
      const camPt = applyPose(member.pose.R, member.pose.t, obj[i]);
      const inAnchor = applyPoseInverse(anchor.pose.R, anchor.pose.t, camPt);
      objectPoints.push(inAnchor);
      imagePoints.push(member.qr.corners[i]);
    }
  }

  const pose = solvePnPPlanar(objectPoints, imagePoints, K, dist, opts);
  if (!pose) return anchor;

  return {
    qr: anchor.qr,
    qrGroup: group.map((g) => g.qr),
    pose,
    plane: posePlane(pose),
    corners3D: objectPoints.map((o) => applyPose(pose.R, pose.t, o)),
    rms: pose.rms,
    mergedCount: group.length,
  };
}

/** Inverse of applyPose: camera coordinates back into a target's frame. */
export function applyPoseInverse(R, t, X) {
  const d = [X[0] - t[0], X[1] - t[1], X[2] - t[2]];
  return [
    R[0][0] * d[0] + R[1][0] * d[1] + R[2][0] * d[2],
    R[0][1] * d[0] + R[1][1] * d[1] + R[2][1] * d[2],
    R[0][2] * d[0] + R[1][2] * d[1] + R[2][2] * d[2],
  ];
}

/**
 * Build the full set of measurement surfaces from QR detections.
 * Convenience wrapper: solve, cluster, merge.
 */
export function buildSurfaces(qrDetections, K, dist, opts = {}) {
  const solved = solveQrPlanes(qrDetections, K, dist, opts);
  const clusters = clusterPlanes(solved, opts);
  return clusters.map((g) => mergePlaneGroup(g, K, dist, opts)).filter(Boolean);
}

/**
 * Cast a screen tap onto a surface.
 *
 * With several surfaces in view, the tap could belong to any of them — every
 * plane meets the same ray at a different depth. The visible one is the nearest,
 * so that is the default. `bias` lets the caller prefer a surface the user has
 * explicitly selected, and the extent guard rejects hits that land absurdly far
 * from the fiducial that defines the plane.
 *
 * Returns { point, surface, distanceMm, onPlane } or null.
 */
export function tapToPoint(pixel, surfaces, K, dist, opts = {}) {
  const { preferSurface = null, maxExtentFactor = 20 } = opts;
  const ray = pixelToNormalizedRay(K, dist, pixel[0], pixel[1]);
  const origin = [0, 0, 0];

  let best = null;

  for (const surface of surfaces) {
    const hit = rayPlaneIntersect(origin, ray, surface.plane.point, surface.plane.normal);
    if (!hit) continue;

    const depth = Math.hypot(hit[0], hit[1], hit[2]);

    // How far the hit is from the fiducial that anchors the plane. A tap
    // hundreds of times the sticker's size away is almost certainly landing on
    // an extension of the plane that does not exist in reality.
    const anchor = surface.qr.sizeMm || 50;
    const anchorPt = surface.plane.point;
    const offset = Math.hypot(hit[0] - anchorPt[0], hit[1] - anchorPt[1], hit[2] - anchorPt[2]);
    const onPlane = offset < anchor * maxExtentFactor;

    const preferred = preferSurface && surface === preferSurface;
    const score = (preferred ? -1e9 : 0) + depth;

    if (!best || score < best.score) {
      best = { point: hit, surface, distanceMm: depth, onPlane, score };
    }
  }

  if (!best) return null;
  delete best.score;
  return best;
}

/** Total length of a polyline through 3D points. */
export function polylineLength(points3D) {
  let total = 0;
  for (let i = 1; i < points3D.length; i++) total += distance3D(points3D[i - 1], points3D[i]);
  return total;
}

/** Per-segment lengths of a polyline. */
export function segmentLengths(points3D) {
  const out = [];
  for (let i = 1; i < points3D.length; i++) out.push(distance3D(points3D[i - 1], points3D[i]));
  return out;
}

/**
 * Area of a closed 3D polygon, by Newell's method.
 *
 * Newell handles a polygon that is not exactly planar and gives the area of its
 * best-fit plane, which is the right answer for a hand-tapped loop on a real
 * surface.
 */
export function polygonArea3D(points) {
  if (points.length < 3) return 0;
  const n = [0, 0, 0];
  for (let i = 0; i < points.length; i++) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    n[0] += (a[1] - b[1]) * (a[2] + b[2]);
    n[1] += (a[2] - b[2]) * (a[0] + b[0]);
    n[2] += (a[0] - b[0]) * (a[1] + b[1]);
  }
  return Math.hypot(n[0], n[1], n[2]) / 2;
}

/** Best-fit plane through 3D points, for reporting orientation. */
export function fitPlane(points) {
  if (points.length < 3) return null;
  const c = [0, 0, 0];
  for (const p of points) { c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; }
  c[0] /= points.length; c[1] /= points.length; c[2] /= points.length;

  // Normal from the two dominant directions via Newell on the centroid-centred
  // points; adequate for the orientation readout this feeds.
  const n = [0, 0, 0];
  for (let i = 0; i < points.length; i++) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    const u = [a[0] - c[0], a[1] - c[1], a[2] - c[2]];
    const v = [b[0] - c[0], b[1] - c[1], b[2] - c[2]];
    n[0] += u[1] * v[2] - u[2] * v[1];
    n[1] += u[2] * v[0] - u[0] * v[2];
    n[2] += u[0] * v[1] - u[1] * v[0];
  }
  const len = Math.hypot(n[0], n[1], n[2]);
  if (len < 1e-12) return null;
  return { point: c, normal: [n[0] / len, n[1] / len, n[2] / len] };
}

/**
 * Express a 3D point in a surface's own frame. This is what lets the app report
 * a measurement as "along the face" rather than in arbitrary camera axes.
 */
export function toSurfaceFrame(surface, point) {
  const local = applyPoseInverse(surface.pose.R, surface.pose.t, point);
  return [local[0], local[1], local[2]];
}

/**
 * Distance from the camera to a surface, in mm — the working distance, which
 * the UI shows because it predicts the achievable accuracy.
 */
export function surfaceDistanceMm(surface) {
  const t = surface.pose.t;
  return Math.hypot(t[0], t[1], t[2]);
}

/**
 * Predicted measurement uncertainty at a given working distance.
 *
 * Error in a measured length is dominated by angular error in the two tapped
 * rays, which grows with distance and shrinks with the fiducial's pixel size:
 *
 *   sigma_mm ~= distance * tapErrorPx / fiducialPx
 *
 * Reporting this honestly is more useful than a bare number, because it tells
 * the user whether to move closer.
 */
export function predictedUncertaintyMm(surface, imageSize, tapErrorPx = 3) {
  const sizeMm = surface.qr ? surface.qr.sizeMm : 50;
  const corners = surface.qr ? surface.qr.corners : null;
  if (!corners) return null;

  // Apparent pixel width of the fiducial, averaged over opposite edges.
  const w = Math.hypot(corners[1][0] - corners[0][0], corners[1][1] - corners[0][1]);
  const h = Math.hypot(corners[3][0] - corners[0][0], corners[3][1] - corners[0][1]);
  const apparentPx = (w + h) / 2;
  if (apparentPx < 1) return null;

  const mmPerPx = sizeMm / apparentPx;
  const dist = surfaceDistanceMm(surface);
  const distFactor = Math.sqrt(1 + (dist / Math.max(1, dist)) ** 0);

  return {
    mmPerPx,
    apparentPx,
    sigmaMm: tapErrorPx * mmPerPx * distFactor,
    workingDistanceMm: dist,
    imageDiagonal: Math.hypot(imageSize[0], imageSize[1]),
  };
}

export { undistortPixel, distance3D };
