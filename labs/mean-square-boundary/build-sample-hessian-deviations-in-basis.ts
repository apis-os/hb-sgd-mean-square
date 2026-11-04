/**
 * @file labs/mean-square-boundary/build-sample-hessian-deviations-in-basis.ts
 * @description Per-sample curvature deviations `D_i = Q^T (H_i - H) Q` in a given orthonormal basis (`buildSampleHessianDeviationsInBasis`;
 * amendment 3, for the hybrid criterion: the full Hessians of `provider` in the eigenbasis of another operator, e.g. the kept
 * Gauss–Newton modes). `basis` holds `m` rows of length `n` (`basis[r * n + a]`); the mean `H`, the per-sample columns and the
 * packing are those of `buildEigenbasisSampleHessianDeviations` with the same loop order, so with the provider's own
 * eigenvectors as `basis` (all `n` rows) the packed deviations are bit-identical to it. Returns the layout of that function on
 * `m` modes (`values` are the Rayleigh quotients `q_r^T H q_r`, for reference only).
 * @cold
 */
import type { buildEigenbasisSampleHessianDeviations } from './build-eigenbasis-sample-hessian-deviations.ts';
import type { Phase4SampleHessianProvider } from './phase4-types.ts';

type Deviations = ReturnType<typeof buildEigenbasisSampleHessianDeviations>;

function meanHessianSymmetrised(provider: Phase4SampleHessianProvider, n: number): Float64Array {
  const e = new Float64Array(n);
  const col = new Float64Array(n);
  const hm = new Float64Array(Math.imul(n, n));
  for (let j = 0; j < n; (j = (j + 1) | 0)) {
    e.fill(0.0);
    e[j] = 1.0;
    provider.applyMeanHessian(e, col);
    for (let k = 0; k < n; (k = (k + 1) | 0)) hm[(Math.imul(j, n) + k) | 0] = col[k]!;
  }
  const hSym = new Float64Array(Math.imul(n, n));
  for (let j = 0; j < n; (j = (j + 1) | 0)) {
    for (let k = 0; k < n; (k = (k + 1) | 0)) hSym[(Math.imul(j, n) + k) | 0] = 0.5 * (hm[(Math.imul(j, n) + k) | 0]! + hm[(Math.imul(k, n) + j) | 0]!);
  }
  return hSym;
}

// rot = B (dev) B^T for the m x n basis B, in the loop order of buildEigenbasisSampleHessianDeviations.
function rotate(dev: Float64Array, basis: Float64Array, n: number, m: number, t1: Float64Array, rot: Float64Array): void {
  for (let r = 0; r < n; (r = (r + 1) | 0)) {
    for (let c = 0; c < m; (c = (c + 1) | 0)) {
      let s = 0.0;
      for (let k = 0; k < n; (k = (k + 1) | 0)) s += dev[(Math.imul(r, n) + k) | 0]! * basis[(Math.imul(c, n) + k) | 0]!;
      t1[(Math.imul(r, m) + c) | 0] = s;
    }
  }
  for (let r = 0; r < m; (r = (r + 1) | 0)) {
    for (let c = 0; c < m; (c = (c + 1) | 0)) {
      let s = 0.0;
      for (let k = 0; k < n; (k = (k + 1) | 0)) s += basis[(Math.imul(r, n) + k) | 0]! * t1[(Math.imul(k, m) + c) | 0]!;
      rot[(Math.imul(r, m) + c) | 0] = s;
    }
  }
}

export function buildSampleHessianDeviationsInBasis(provider: Phase4SampleHessianProvider, basis: Float64Array, m: number): Deviations {
  const n = provider.numParams | 0;
  const nS = provider.numSamples | 0;
  const nPair = ((m * (m + 1)) / 2) | 0;
  const hSym = meanHessianSymmetrised(provider, n);
  const pairIndex = new Int32Array(Math.imul(m, m));
  const pairA = new Int32Array(nPair);
  const pairB = new Int32Array(nPair);
  for (let a = 0; a < m; (a = (a + 1) | 0)) {
    for (let b = a; b < m; (b = (b + 1) | 0)) {
      const p = (Math.imul(a, m) - ((a * (a - 1)) / 2) + b - a) | 0;
      pairIndex[(Math.imul(a, m) + b) | 0] = p;
      pairIndex[(Math.imul(b, m) + a) | 0] = p;
      pairA[p] = a;
      pairB[p] = b;
    }
  }
  const e = new Float64Array(n);
  const col = new Float64Array(n);
  const dev = new Float64Array(Math.imul(n, n));
  const t1 = new Float64Array(Math.imul(n, m));
  const rot = new Float64Array(Math.imul(m, m));
  const packed = new Float64Array(Math.imul(nS, nPair));
  for (let i = 0; i < nS; (i = (i + 1) | 0)) {
    for (let j = 0; j < n; (j = (j + 1) | 0)) {
      e.fill(0.0);
      e[j] = 1.0;
      provider.applySampleHessian(i, e, col);
      for (let k = 0; k < n; (k = (k + 1) | 0)) dev[(Math.imul(k, n) + j) | 0] = col[k]! - hSym[(Math.imul(k, n) + j) | 0]!;
    }
    rotate(dev, basis, n, m, t1, rot);
    const base = Math.imul(i, nPair);
    for (let p = 0; p < nPair; (p = (p + 1) | 0)) {
      const a = pairA[p]!;
      const b = pairB[p]!;
      packed[(base + p) | 0] = 0.5 * (rot[(Math.imul(a, m) + b) | 0]! + rot[(Math.imul(b, m) + a) | 0]!);
    }
  }
  const values = new Float64Array(m);
  for (let r = 0; r < m; (r = (r + 1) | 0)) {
    let s = 0.0;
    for (let a = 0; a < n; (a = (a + 1) | 0)) {
      let hq = 0.0;
      for (let b = 0; b < n; (b = (b + 1) | 0)) hq += hSym[(Math.imul(a, n) + b) | 0]! * basis[(Math.imul(r, n) + b) | 0]!;
      s += basis[(Math.imul(r, n) + a) | 0]! * hq;
    }
    values[r] = s;
  }
  return { n: m, numSamples: nS, nPair, hSym, values, qt: basis.slice(0, Math.imul(m, n)), packed, pairIndex, pairA, pairB };
}
