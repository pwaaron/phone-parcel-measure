/**
 * capture.js — guidance logic for live calibration capture.
 *
 * The problem this solves: a calibration can converge, report a low RMS, and
 * still be wrong — because the intrinsics it is least able to pin down are the
 * principal point and the distortion terms, and the only thing that constrains
 * them is WHERE IN THE FRAME the board appeared.
 *
 * Fifteen centred views leave those parameters effectively free, so the
 * optimiser drifts them to absorb error elsewhere. The result looks fine and
 * measures badly. Warning about it afterwards is the wrong fix; the right fix is
 * to refuse to accept views that do not add information.
 *
 * So this module answers three questions per frame:
 *   - is the board sharp enough to be worth trusting?
 *   - does this pose teach us anything we do not already know?
 *   - which part of the frame should the user fill next?
 */

/**
 * Sharpness by variance of the Laplacian.
 *
 * Blur is the most common way a calibration view is quietly useless: corners
 * still detect, but half a pixel of smear biases every one of them. The
 * Laplacian is a high-pass filter, so a sharp image has a wide spread of
 * responses and a blurred one has almost none.
 */
export function laplacianVariance(gray, width, height) {
  let sum = 0;
  let sumSq = 0;
  let n = 0;

  for (let y = 1; y < height - 1; y++) {
    const row = y * width;
    for (let x = 1; x < width - 1; x++) {
      const i = row + x;
      const v = gray[i - width] + gray[i + width] + gray[i - 1] + gray[i + 1] - 4 * gray[i];
      sum += v;
      sumSq += v * v;
      n++;
    }
  }
  if (n === 0) return 0;
  const mean = sum / n;
  return sumSq / n - mean * mean;
}

/**
 * A rolling sharpness reference.
 *
 * An absolute threshold does not travel: the same board under different lighting
 * and different sensor gain produces very different Laplacian variances. Tracking
 * the best recent value and accepting frames near it adapts to the scene instead
 * of to a number someone guessed.
 */
export function makeSharpnessGate({ fraction = 0.55, decay = 0.985 } = {}) {
  let reference = 0;
  return {
    /** Feed a measurement; returns whether the frame is sharp enough. */
    test(value) {
      reference = Math.max(value, reference * decay);
      if (reference <= 0) return { ok: true, value, reference };
      return { ok: value >= reference * fraction, value, reference };
    },
    get reference() { return reference; },
  };
}

/**
 * Track which parts of the frame the board has visited.
 *
 * A coarse grid is the right granularity: the point is not to measure coverage
 * precisely but to notice "every view was in the middle", which a 4x4 grid
 * makes obvious at a glance.
 */
export function makeCoverageTracker(imageSize, gridN = 4) {
  const cells = new Array(gridN * gridN).fill(0);
  return {
    imageSize,
    gridN,
    cells,
    views: [],
  };
}

/**
 * Summarise where a view sits and how it is shaped, in normalised coordinates.
 *
 * `tilt` is the useful one: a board photographed head-on fills its own bounding
 * box almost completely, while a tilted one traces a thin quadrilateral inside a
 * much larger box. That difference is what lets the novelty check tell "same
 * place, same angle" apart from "same place, new angle" — and a new angle is
 * exactly what constrains the distortion terms.
 */
function viewSignature(corners, w, h) {
  let minX = Infinity; let maxX = -Infinity;
  let minY = Infinity; let maxY = -Infinity;
  for (const c of corners) {
    minX = Math.min(minX, c[0]); maxX = Math.max(maxX, c[0]);
    minY = Math.min(minY, c[1]); maxY = Math.max(maxY, c[1]);
  }

  const centroid = [(minX + maxX) / 2 / w, (minY + maxY) / 2 / h];
  const span = Math.hypot((maxX - minX) / w, (maxY - minY) / h);

  const boxArea = Math.max(1e-9, ((maxX - minX) / w) * ((maxY - minY) / h));
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const spread = corners.reduce((a, c) => {
    const dx = (c[0] - cx) / w;
    const dy = (c[1] - cy) / h;
    return a + Math.hypot(dx, dy);
  }, 0) / corners.length;
  const tilt = spread / Math.sqrt(boxArea);

  return { centroid, span, tilt };
}

/** Record where a view's corners landed, and how much of the frame it spanned. */
export function recordView(tracker, corners) {
  const [w, h] = tracker.imageSize;
  const n = tracker.gridN;

  let minX = Infinity; let maxX = -Infinity;
  let minY = Infinity; let maxY = -Infinity;
  for (const c of corners) {
    minX = Math.min(minX, c[0]); maxX = Math.max(maxX, c[0]);
    minY = Math.min(minY, c[1]); maxY = Math.max(maxY, c[1]);
  }

  // Mark every cell the board actually covers, not just its centre.
  for (let gy = 0; gy < n; gy++) {
    for (let gx = 0; gx < n; gx++) {
      const cx0 = (gx / n) * w;
      const cx1 = ((gx + 1) / n) * w;
      const cy0 = (gy / n) * h;
      const cy1 = ((gy + 1) / n) * h;
      // Cell is covered when the board's bounding box overlaps it.
      if (maxX >= cx0 && minX <= cx1 && maxY >= cy0 && minY <= cy1) {
        tracker.cells[gy * n + gx]++;
      }
    }
  }

  const sig = viewSignature(corners, w, h);
  tracker.views.push({
    ...sig,
    corners: corners.map((c) => [c[0] / w, c[1] / h]),
  });
  return tracker.views[tracker.views.length - 1];
}

/**
 * How much a new view would add, in 0..1.
 *
 * Zero means "we already have a view almost exactly like this"; one means
 * "nothing like it yet". Compared on normalised position, apparent size, and
 * tilt — the last of these is what distinguishes a head-on view from an angled
 * one taken from the same spot.
 */
export function noveltyScore(tracker, corners) {
  const [w, h] = tracker.imageSize;
  if (tracker.views.length === 0) return 1;

  const { centroid, span, tilt } = viewSignature(corners, w, h);

  let best = 0;
  for (const v of tracker.views) {
    const dPos = Math.hypot(v.centroid[0] - centroid[0], v.centroid[1] - centroid[1]);
    const dSpan = Math.abs(v.span - span);
    const dTilt = Math.abs(v.tilt - tilt);
    // Normalised distances: a view is "the same" when all three are small.
    const sim = Math.max(0, 1 - (dPos / 0.22) - (dSpan / 0.35) - (dTilt / 0.25));
    best = Math.max(best, sim);
  }
  return 1 - best;
}

/** Fraction of grid cells that have been covered at least once. */
export function coverageFraction(tracker) {
  const covered = tracker.cells.filter((c) => c > 0).length;
  return covered / tracker.cells.length;
}

/**
 * Which region to send the user to next, as a grid coordinate plus a plain
 * instruction. Prefers empty cells, then the least-visited ones.
 */
export function suggestNextRegion(tracker) {
  const n = tracker.gridN;
  const names = [
    ['top-left', 'top-centre', 'top-right'],
    ['mid-left', 'centre', 'mid-right'],
    ['bottom-left', 'bottom-centre', 'bottom-right'],
  ];
  const nameFor = (gx, gy) => {
    const cx = Math.min(2, Math.floor((gx / n) * 3));
    const cy = Math.min(2, Math.floor((gy / n) * 3));
    return names[cy][cx];
  };

  let best = null;
  for (let gy = 0; gy < n; gy++) {
    for (let gx = 0; gx < n; gx++) {
      const count = tracker.cells[gy * n + gx];
      // Slight bias toward the outer ring: corner coverage is what actually
      // constrains the principal point, so fill the edges before the middle.
      const edge = (gx === 0 || gy === 0 || gx === n - 1 || gy === n - 1) ? 0 : 0.5;
      const score = count + edge;
      if (!best || score < best.score) best = { gx, gy, score, count };
    }
  }
  if (!best || best.count > 0) return null; // nothing empty left worth naming

  return {
    gx: best.gx,
    gy: best.gy,
    label: nameFor(best.gx, best.gy),
    instruction: `Move the board to the ${nameFor(best.gx, best.gy)} of the frame`,
  };
}

/**
 * Overall capture verdict for one frame.
 *
 * Returns { accept, reason, guidance } where guidance is the single sentence to
 * show the user. Deliberately one sentence: a live capture screen is read at a
 * glance while holding a board in the other hand.
 */
export function assessFrame({ cornerCount, sharpness, novelty, minCorners = 10, minNovelty = 0.18 }) {
  if (cornerCount < minCorners) {
    return {
      accept: false,
      reason: 'few-corners',
      guidance: `Only ${cornerCount} corners — get the whole board in frame`,
    };
  }
  if (!sharpness.ok) {
    return { accept: false, reason: 'blurry', guidance: 'Hold still — the frame is soft' };
  }
  if (novelty < minNovelty) {
    return { accept: false, reason: 'duplicate', guidance: 'Too similar to a view you already have' };
  }
  return { accept: true, reason: 'ok', guidance: 'Captured' };
}
