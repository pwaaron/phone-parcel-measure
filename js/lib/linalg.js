/**
 * linalg.js — minimal dense linear algebra + Levenberg-Marquardt.
 *
 * Pure JavaScript, zero dependencies. Everything here is deliberately small and
 * synchronous; the largest systems this app ever solves are ~9x9 (homography DLT)
 * and the LM normal equations during camera calibration (~10 params).
 *
 * Conventions
 *   vector  = number[]
 *   matrix  = number[][]   (row-major: A[i][j] is row i, column j)
 *
 * No third-party code. Every routine is exercised by test/run_tests.mjs against
 * problems with a known closed-form answer.
 */

/** Zero matrix of shape n x m. */
export function zeros(n, m) {
  const A = new Array(n);
  for (let i = 0; i < n; i++) A[i] = new Float64Array(m);
  return A.map((r) => Array.from(r));
}

/** Identity matrix of order n. */
export function eye(n) {
  const A = zeros(n, n);
  for (let i = 0; i < n; i++) A[i][i] = 1;
  return A;
}

/** Matrix product A (n x k) times B (k x m). */
export function matMul(A, B) {
  const n = A.length;
  const k = B.length;
  const m = B[0].length;
  const C = zeros(n, m);
  for (let i = 0; i < n; i++) {
    const Ai = A[i];
    const Ci = C[i];
    for (let p = 0; p < k; p++) {
      const a = Ai[p];
      if (a === 0) continue;
      const Bp = B[p];
      for (let j = 0; j < m; j++) Ci[j] += a * Bp[j];
    }
  }
  return C;
}

/** Matrix-vector product A (n x m) times x (m). */
export function matVec(A, x) {
  const n = A.length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const Ai = A[i];
    let s = 0;
    for (let j = 0; j < Ai.length; j++) s += Ai[j] * x[j];
    out[i] = s;
  }
  return out;
}

/** Transpose. */
export function transpose(A) {
  const n = A.length;
  const m = A[0].length;
  const T = zeros(m, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) T[j][i] = A[i][j];
  return T;
}

/** Scalar multiple of a matrix. */
export function scaleMat(A, s) {
  return A.map((row) => row.map((v) => v * s));
}

/** Elementwise A + B. */
export function addMat(A, B) {
  return A.map((row, i) => row.map((v, j) => v + B[i][j]));
}

/** Dot product. */
export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Euclidean norm. */
export function norm(a) {
  return Math.sqrt(dot(a, a));
}

/** Sum of squares — the cost function LM minimises. */
export function sumSq(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return s;
}

/**
 * Solve A x = b for square A via LU decomposition with partial pivoting.
 * Throws if A is numerically singular, so callers can treat that as "no solution"
 * rather than silently returning NaNs that poison everything downstream.
 */
export function solve(A, b) {
  const n = A.length;
  const M = new Array(n);
  for (let i = 0; i < n; i++) M[i] = A[i].slice().concat([b[i]]);

  for (let c = 0; c < n; c++) {
    let pivot = c;
    let best = Math.abs(M[c][c]);
    for (let r = c + 1; r < n; r++) {
      const v = Math.abs(M[r][c]);
      if (v > best) { best = v; pivot = r; }
    }
    if (best < 1e-300) throw new Error('solve: singular matrix');
    if (pivot !== c) { const t = M[pivot]; M[pivot] = M[c]; M[c] = t; }

    const piv = M[c][c];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / piv;
      if (f === 0) continue;
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }

  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

/** Matrix inverse via repeated solve against the identity columns. */
export function invert(A) {
  const n = A.length;
  const inv = zeros(n, n);
  const e = new Array(n).fill(0);
  for (let j = 0; j < n; j++) {
    e[j] = 1;
    const col = solve(A, e);
    for (let i = 0; i < n; i++) inv[i][j] = col[i];
    e[j] = 0;
  }
  return inv;
}

/**
 * Eigen-decomposition of a real symmetric matrix by the cyclic Jacobi method.
 *
 * This single routine carries the whole calibration pipeline: homography DLT
 * (null space of a 9x9 normal matrix), Zhang's intrinsics (eigen of the 4x4
 * absolute-conic matrix), and rank checks.
 *
 * Returns eigenvalues ascending, with the matching eigenvector for each
 * (vectors[k] is the unit eigenvector for values[k]).
 */
export function jacobiEigenSymmetric(Ain, { maxSweeps = 100, tol = 1e-18 } = {}) {
  const n = Ain.length;
  const A = Ain.map((r) => r.slice());

  // Jacobi requires symmetry; average away any asymmetry from round-off.
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const v = 0.5 * (A[i][j] + A[j][i]);
      A[i][j] = v;
      A[j][i] = v;
    }
  }

  let V = eye(n);

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
    if (off < tol) break;

    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p][q];
        if (Math.abs(apq) < 1e-300) continue;

        const theta = (A[q][q] - A[p][p]) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;

        // A <- J^T A J, applied as two one-sided rotations.
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }

  const pairs = [];
  for (let i = 0; i < n; i++) {
    const vec = new Array(n);
    for (let k = 0; k < n; k++) vec[k] = V[k][i];
    pairs.push({ value: A[i][i], vector: vec });
  }
  pairs.sort((a, b) => a.value - b.value);
  return { values: pairs.map((p) => p.value), vectors: pairs.map((p) => p.vector) };
}

/**
 * Unit vector spanning the null space of M — i.e. the minimiser of |M x| subject
 * to |x| = 1. Computed as the eigenvector of M^T M for its smallest eigenvalue.
 *
 * This is the workhorse behind every DLT in the app (homography, PnP, calibration).
 */
export function nullVector(M) {
  const Mt = transpose(M);
  const MtM = matMul(Mt, M);
  const { vectors } = jacobiEigenSymmetric(MtM);
  return vectors[0];
}

/** Numerical Jacobian of a residual function, by forward differences. */
function numericJacobian(fn, x, r0, fdStep) {
  const m = r0.length;
  const n = x.length;
  const J = new Array(m);
  for (let k = 0; k < m; k++) J[k] = new Array(n).fill(0);

  for (let j = 0; j < n; j++) {
    const step = fdStep * Math.max(1, Math.abs(x[j]));
    const xs = x.slice();
    xs[j] += step;
    const rs = fn(xs);
    const inv = 1 / step;
    for (let k = 0; k < m; k++) J[k][j] = (rs[k] - r0[k]) * inv;
  }
  return J;
}

/**
 * Levenberg-Marquardt least squares.
 *
 *   fn(x) -> residuals (number[]). Minimises sum of squares.
 *
 * Damped Gauss-Newton with an adaptive trust parameter: accept a step and relax
 * the damping when the cost drops, reject it and stiffen the damping when it does
 * not. This is what turns Zhang's closed-form calibration into an accurate one,
 * and what refines a homography-decomposed pose into a metrologically usable one.
 */
export function levenbergMarquardt(fn, x0, opts = {}) {
  const {
    maxIter = 200,
    costTol = 1e-15,
    stepTol = 1e-14,
    lambda0 = 1e-3,
    fdStep = 1e-6,
    minLambda = 1e-12,
    maxLambda = 1e12,
    maxDampingTries = 12,
  } = opts;

  let x = x0.slice();
  let r = fn(x);
  let cost = sumSq(r);
  let lambda = lambda0;
  const n = x.length;
  let iter = 0;

  for (; iter < maxIter; iter++) {
    const J = numericJacobian(fn, x, r, fdStep);
    const m = r.length;

    // Normal equations: JtJ dx = -g
    const JtJ = zeros(n, n);
    const g = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        let s = 0;
        for (let k = 0; k < m; k++) s += J[k][i] * J[k][j];
        JtJ[i][j] = s;
        JtJ[j][i] = s;
      }
      let s = 0;
      for (let k = 0; k < m; k++) s += J[k][i] * r[k];
      g[i] = s;
    }

    let accepted = false;
    let stepNorm = 0;

    for (let attempt = 0; attempt < maxDampingTries; attempt++) {
      // Marquardt scaling: damp in proportion to each parameter's own curvature,
      // so parameters on wildly different scales (focal length vs distortion)
      // still take sensible steps.
      const H = JtJ.map((row, i) =>
        row.map((v, j) => (i === j ? v * (1 + lambda) + (v === 0 ? lambda : 0) : v))
      );

      let delta;
      try {
        delta = solve(H, g.map((v) => -v));
      } catch {
        lambda = Math.min(lambda * 10, maxLambda);
        continue;
      }

      const xn = x.map((v, i) => v + delta[i]);
      const rn = fn(xn);
      const costn = sumSq(rn);

      if (Number.isFinite(costn) && costn < cost) {
        stepNorm = norm(delta);
        x = xn;
        r = rn;
        cost = costn;
        lambda = Math.max(lambda / 10, minLambda);
        accepted = true;
        break;
      }
      lambda = Math.min(lambda * 10, maxLambda);
    }

    if (!accepted) break;
    if (cost < costTol) break;
    if (stepNorm < stepTol) break;
  }

  return { x, cost, iterations: iter, residuals: r };
}
