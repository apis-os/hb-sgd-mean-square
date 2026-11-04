/**
 * @file labs/mean-square-boundary/build-eigenbasis-sample-hessian-deviations.ts
 * @description Dense per-sample curvature deviations in the eigenbasis of the mean Hessian (`buildEigenbasisSampleHessianDeviations`),
 * for small `P`. The mean is built as in Phase 4 (`evaluatePhase4NetworkRow`): rows `H e_j` from `applyMeanHessian`, symmetrised
 * as `(H + H^T) / 2`, and eigendecomposed by `computeSymmetricEigendecompositionHouseholderQl`. Each per-sample Hessian is read
 * column by column from `applySampleHessian` on unit vectors, and `D_i = Q^T (H_i - H) Q` (rows of `qt` are the eigenvectors) is
 * symmetrised and stored as its upper triangle, packed pair by pair: pair `p(a, b)`, `a <= b`, at `a n - a (a - 1) / 2 + b - a`,
 * sample-major (`packed[i * nPair + p]`). `pairIndex[a * n + b]` gives `p` for either order.
 * @cold
 */
import { computeSymmetricEigendecompositionHouseholderQl } from './compute-symmetric-eigendecomposition-householder-ql.ts';
import type { Phase4SampleHessianProvider } from './phase4-types.ts';

export function buildEigenbasisSampleHessianDeviations(provider: Phase4SampleHessianProvider): {
  readonly n: number;
  readonly numSamples: number;
  readonly nPair: number;
  readonly hSym: Float64Array;
  readonly values: Float64Array;
  readonly qt: Float64Array;
  readonly packed: Float64Array;
  readonly pairIndex: Int32Array;
  readonly pairA: Int32Array;
  readonly pairB: Int32Array;
} {
  const n = provider.numParams | 0;
  const nS = provider.numSamples | 0;
  const nn = Math.imul(n, n);
  const nPair = ((n * (n + 1)) / 2) | 0;
  const e = new Float64Array(n);
  const col = new Float64Array(n);
  const hm = new Float64Array(nn);
  for (let j = 0; j < n; (j = (j + 1) | 0)) {
    e.fill(0.0);
    e[j] = 1.0;
    provider.applyMeanHessian(e, col);
    for (let k = 0; k < n; (k = (k + 1) | 0)) hm[(Math.imul(j, n) + k) | 0] = col[k]!;
  }
  const hSym = new Float64Array(nn);
  for (let j = 0; j < n; (j = (j + 1) | 0)) {
    for (let k = 0; k < n; (k = (k + 1) | 0)) hSym[(Math.imul(j, n) + k) | 0] = 0.5 * (hm[(Math.imul(j, n) + k) | 0]! + hm[(Math.imul(k, n) + j) | 0]!);
  }
  const eig = computeSymmetricEigendecompositionHouseholderQl(hSym, n);
  const qt = eig.qt;
  const pairIndex = new Int32Array(nn);
  const pairA = new Int32Array(nPair);
  const pairB = new Int32Array(nPair);
  for (let a = 0; a < n; (a = (a + 1) | 0)) {
    for (let b = a; b < n; (b = (b + 1) | 0)) {
      const p = (Math.imul(a, n) - ((a * (a - 1)) / 2) + b - a) | 0;
      pairIndex[(Math.imul(a, n) + b) | 0] = p;
      pairIndex[(Math.imul(b, n) + a) | 0] = p;
      pairA[p] = a;
      pairB[p] = b;
    }
  }
  const packed = new Float64Array(Math.imul(nS, nPair));
  const dev = new Float64Array(nn);
  const t1 = new Float64Array(nn);
  const rot = new Float64Array(nn);
  for (let i = 0; i < nS; (i = (i + 1) | 0)) {
    // Column j of H_i is H_i e_j; dev = H_i - H.
    for (let j = 0; j < n; (j = (j + 1) | 0)) {
      e.fill(0.0);
      e[j] = 1.0;
      provider.applySampleHessian(i, e, col);
      for (let k = 0; k < n; (k = (k + 1) | 0)) dev[(Math.imul(k, n) + j) | 0] = col[k]! - hSym[(Math.imul(k, n) + j) | 0]!;
    }
    // t1 = dev Q (Q = qt^T, so t1[r][c] = sum_m dev[r][m] qt[c][m]); rot = Q^T t1.
    for (let r = 0; r < n; (r = (r + 1) | 0)) {
      for (let c = 0; c < n; (c = (c + 1) | 0)) {
        let s = 0.0;
        for (let m = 0; m < n; (m = (m + 1) | 0)) s += dev[(Math.imul(r, n) + m) | 0]! * qt[(Math.imul(c, n) + m) | 0]!;
        t1[(Math.imul(r, n) + c) | 0] = s;
      }
    }
    for (let r = 0; r < n; (r = (r + 1) | 0)) {
      for (let c = 0; c < n; (c = (c + 1) | 0)) {
        let s = 0.0;
        for (let m = 0; m < n; (m = (m + 1) | 0)) s += qt[(Math.imul(r, n) + m) | 0]! * t1[(Math.imul(m, n) + c) | 0]!;
        rot[(Math.imul(r, n) + c) | 0] = s;
      }
    }
    const base = Math.imul(i, nPair);
    for (let p = 0; p < nPair; (p = (p + 1) | 0)) {
      const a = pairA[p]!;
      const b = pairB[p]!;
      packed[(base + p) | 0] = 0.5 * (rot[(Math.imul(a, n) + b) | 0]! + rot[(Math.imul(b, n) + a) | 0]!);
    }
  }
  return { n, numSamples: nS, nPair, hSym, values: eig.values, qt, packed, pairIndex, pairA, pairB };
}
