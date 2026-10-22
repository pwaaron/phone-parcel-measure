/**
 * contour.js — border following, polygon simplification and polygon predicates.
 *
 * These are the shape primitives the ArUco and QR detectors share. A marker is
 * found by thresholding, tracing every connected border, keeping the ones that
 * simplify to a quadrilateral of plausible area, and reading the bits inside.
 */

/**
 * Trace the outer border of the connected component containing (sx, sy) using
 * Moore-neighbour following.
 *
 * The walk always hugs the boundary clockwise, tracking the background pixel it
 * arrived from so it knows where to resume scanning. Jacob's criterion stops it
 * when it re-enters the start pixel from the same direction, which is what makes
 * it terminate correctly on shapes with thin necks.
 */
function traceBorder(binary, visited, width, height, sx, sy) {
  // 8-neighbourhood in clockwise order, starting due west.
  const N8 = [
    [-1, 0], [-1, -1], [0, -1], [1, -1],
    [1, 0], [1, 1], [0, 1], [-1, 1],
  ];

  const contour = [];
  let cx = sx;
  let cy = sy;
  let bx = sx - 1; // arrived from the left; that pixel is background by construction
  let by = sy;
  let secondPoint = null;
  const maxSteps = width * height * 4;
  let steps = 0;

  while (steps++ < maxSteps) {
    contour.push([cx, cy]);
    visited[cy * width + cx] = 1;

    let backDir = 0;
    for (let k = 0; k < 8; k++) {
      if (cx + N8[k][0] === bx && cy + N8[k][1] === by) { backDir = k; break; }
    }

    let found = -1;
    for (let k = 1; k <= 8; k++) {
      const d = (backDir + k) % 8;
      const nx = cx + N8[d][0];
      const ny = cy + N8[d][1];
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      if (binary[ny * width + nx]) { found = d; break; }
    }
    if (found < 0) break; // isolated pixel, no border to follow

    const nx = cx + N8[found][0];
    const ny = cy + N8[found][1];

    // Jacob's stopping criterion: the loop is closed when we are back on the
    // start pixel AND about to step to the same second point we set off toward.
    // Merely counting returns to the start traces the boundary twice, which
    // doubles the reported area and corrupts every polygon that follows.
    if (secondPoint === null) {
      secondPoint = [nx, ny];
    } else if (cx === sx && cy === sy && nx === secondPoint[0] && ny === secondPoint[1]) {
      break;
    }

    bx = cx;
    by = cy;
    cx = nx;
    cy = ny;
  }

  return contour;
}

/**
 * All outer contours in a binary mask (255 = foreground), as arrays of [x, y].
 * Contours shorter than minPoints are discarded.
 */
export function findContours(binary, width, height, minPoints = 8) {
  const visited = new Uint8Array(width * height);
  const contours = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (!binary[idx] || visited[idx]) continue;
      // Only begin a contour at the LEFT edge of a foreground run. An interior
      // pixel has a foreground neighbour to its left and would otherwise spawn
      // its own degenerate one-pixel contour — a filled 30x30 square produced
      // 785 contours instead of 1 before this guard.
      if (x > 0 && binary[idx - 1]) continue;
      const c = traceBorder(binary, visited, width, height, x, y);
      if (c.length >= minPoints) contours.push(c);
    }
  }
  return contours;
}

/** Signed polygon area by the shoelace formula; sign encodes winding. */
export function signedArea(poly) {
  let a = 0;
  for (let i = 0, n = poly.length; i < n; i++) {
    const p = poly[i];
    const q = poly[(i + 1) % n];
    a += p[0] * q[1] - q[0] * p[1];
  }
  return a / 2;
}

/** Absolute polygon area. */
export function contourArea(poly) {
  return Math.abs(signedArea(poly));
}

/** Perimeter length. */
export function contourPerimeter(poly) {
  let p = 0;
  for (let i = 0, n = poly.length; i < n; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % n];
    p += Math.hypot(b[0] - a[0], b[1] - a[1]);
  }
  return p;
}

/**
 * Ramer-Douglas-Peucker for an OPEN polyline.
 *
 * Anchors on the first and last points and recursively keeps whichever interior
 * point lies farthest from the current chord.
 */
function rdpOpen(points, epsilon) {
  if (points.length < 3) return points.slice();

  const keep = new Uint8Array(points.length);
  keep[0] = 1;
  keep[points.length - 1] = 1;

  const stack = [[0, points.length - 1]];
  while (stack.length) {
    const [first, last] = stack.pop();
    if (last <= first + 1) continue;

    const a = points[first];
    const b = points[last];
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const len = Math.hypot(dx, dy);

    let maxDist = -1;
    let maxIdx = -1;
    for (let i = first + 1; i < last; i++) {
      const p = points[i];
      const dist = len < 1e-12
        ? Math.hypot(p[0] - a[0], p[1] - a[1])
        : Math.abs(dy * p[0] - dx * p[1] + b[0] * a[1] - b[1] * a[0]) / len;
      if (dist > maxDist) { maxDist = dist; maxIdx = i; }
    }

    if (maxDist > epsilon && maxIdx > 0) {
      keep[maxIdx] = 1;
      stack.push([first, maxIdx]);
      stack.push([maxIdx, last]);
    }
  }

  const out = [];
  for (let i = 0; i < points.length; i++) if (keep[i]) out.push(points[i]);
  return out;
}

/**
 * Ramer-Douglas-Peucker simplification.
 *
 * A traced border is a dense pixel chain; a marker's edge is four straight
 * lines. Reducing the chain to its corners is what lets us recognise a
 * quadrilateral at all.
 *
 * For a CLOSED contour the plain open-chain algorithm is wrong: it anchors on
 * the first and last points, which on a loop are adjacent pixels rather than
 * corners, so the segment spanning the wrap-around gets flattened away and a
 * real corner is lost. Splitting the loop at the point farthest from the start
 * puts both anchors on genuine extremes and fixes it.
 */
export function approxPolyDP(points, epsilon, closed = true) {
  if (points.length < 3) return points.slice();

  if (!closed) return rdpOpen(points, epsilon);

  let far = 0;
  let best = -1;
  const p0 = points[0];
  for (let i = 1; i < points.length; i++) {
    const d = Math.hypot(points[i][0] - p0[0], points[i][1] - p0[1]);
    if (d > best) { best = d; far = i; }
  }

  // Degenerate loop where every point coincides with the start.
  if (far === 0 || best < 1e-12) return rdpOpen(points, epsilon);

  const chainA = points.slice(0, far + 1);
  const chainB = points.slice(far).concat([points[0]]);

  const a = rdpOpen(chainA, epsilon);
  const b = rdpOpen(chainB, epsilon);

  // b starts where a ends and finishes back at the start, so both duplicated
  // endpoints are dropped when merging.
  return a.concat(b.slice(1, b.length - 1));
}

/** Whether a polygon is convex, using consistent cross-product signs. */
export function isConvex(poly) {
  const n = poly.length;
  if (n < 4) return false;
  let sign = 0;
  for (let i = 0; i < n; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % n];
    const c = poly[(i + 2) % n];
    const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    if (Math.abs(cross) < 1e-9) continue;
    const s = cross > 0 ? 1 : -1;
    if (sign === 0) sign = s;
    else if (s !== sign) return false;
  }
  return true;
}

/** Order four points as top-left, top-right, bottom-right, bottom-left. */
export function orderQuadCorners(pts) {
  const n = pts.length;
  let cx = 0;
  let cy = 0;
  for (const p of pts) { cx += p[0]; cy += p[1]; }
  cx /= n;
  cy /= n;

  // Sort by angle around the centroid to get a consistent winding, then rotate
  // so the corner closest to the top-left of the bounding box comes first.
  const withAngle = pts.map((p) => ({ p, a: Math.atan2(p[1] - cy, p[0] - cx) }));
  withAngle.sort((u, v) => u.a - v.a);
  const ordered = withAngle.map((o) => o.p);

  let best = 0;
  let bestScore = Infinity;
  for (let i = 0; i < ordered.length; i++) {
    const score = ordered[i][0] + ordered[i][1];
    if (score < bestScore) { bestScore = score; best = i; }
  }
  return ordered.slice(best).concat(ordered.slice(0, best));
}

/** Ray-casting point-in-polygon test. */
export function pointInPolygon(poly, x, y) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0];
    const yi = poly[i][1];
    const xj = poly[j][0];
    const yj = poly[j][1];
    const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

/** Centroid of a polygon's vertices. */
export function polygonCentroid(poly) {
  let x = 0;
  let y = 0;
  for (const p of poly) { x += p[0]; y += p[1]; }
  return [x / poly.length, y / poly.length];
}

/**
 * Side lengths of a quadrilateral, in order. For a square marker seen at an
 * angle these differ; their ratio is a cheap foreshortening signal.
 */
export function quadSideLengths(q) {
  const out = [];
  for (let i = 0; i < q.length; i++) {
    const a = q[i];
    const b = q[(i + 1) % q.length];
    out.push(Math.hypot(b[0] - a[0], b[1] - a[1]));
  }
  return out;
}
