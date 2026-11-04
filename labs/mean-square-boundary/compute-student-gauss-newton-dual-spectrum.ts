/**
 * @file labs/mean-square-boundary/compute-student-gauss-newton-dual-spectrum.ts
 * @description Factored dual Gauss–Newton spectrum of the Phase 6 two-layer student network
 * (`computeStudentGaussNewtonDualSpectrum`). For `f(x_i) = W_2 phi(W_1 x_i + b_1) + b_2`, row `(i, c)` of the stacked output
 * Jacobian `J` (`m = N dOut` rows, `P` columns) has first-layer block `(W_2[c, :] .* d1_i) kron (x_i, 1)` and output-layer block
 * `e_c kron (act_i, 1)`. The dual Gram matrix `G = (1/N) J J^T` therefore factors through the two `N x N` sample kernels
 * `K_x[i, i'] = 1 + <x_i, x_{i'}>` and `K_a[i, i'] = 1 + <act_i, act_{i'}>` and the `m x dHidden` backpropagated slopes
 * `dz[(i, c), h] = W_2[c, h] d1_i[h]`:
 * `G[(i, c), (i', c')] = (K_x[i, i'] <dz_{(i, c)}, dz_{(i', c')}> + delta_{c, c'} K_a[i, i']) / N`.
 * That avoids allocating the `m x P` Jacobian (`65.7 MB` at `P = 12 826`) and replaces `O(m^2 P)` inner products with
 * `O(N^2 dIn + m^2 dHidden)`, followed by the same `computeSymmetricEigendecompositionHouseholderQl` solve and `relCut` mode cut as
 * `computeGaussNewtonDualSpectrum`. Never reads `yData`.
 * Exactness contract: `bound` (eigenvalues in `values` relative error `<= 1e-10` of the top eigenvalue, and `traceGaussNewton`,
 * `traceGaussNewtonSq`, `effectiveRank` relative error `<= 1e-12` against `computeGaussNewtonDualSpectrum` on
 * `computeSampleOutputJacobian`).
 * With `withDualResidual` (Phase 6 stage 2, after the second and fourth audit rounds), `dualResidualMax` is
 * `max_k ||G v_k - lambda_k v_k||_2` over the kept modes (`computeDualEigenResidualMax`), computed in floating point (a diagnostic, not certified; the Phase 6 factors
 * are built from these `v_k`, and the factors' own consistency `meanResidual` cannot see an eigensolver error); otherwise `null`, and
 * nothing else changes.
 * @cold
 */
import { computeDualEigenResidualMax } from './compute-dual-eigen-residual-max.ts';
import { computeSymmetricEigendecompositionHouseholderQl } from './compute-symmetric-eigendecomposition-householder-ql.ts';
import type { createPhase6StudentNetwork } from './create-phase6-student-network.ts';

function storeSymmetric(g: Float64Array, m: number, r: number, c: number, v: number): void {
  g[(Math.imul(r, m) + c) | 0] = v;
  g[(Math.imul(c, m) + r) | 0] = v;
}

function buildSampleKernels(
  xData: Float64Array,
  actAll: Float64Array,
  numSamples: number,
  dIn: number,
  dHidden: number,
  kx: Float64Array,
  ka: Float64Array,
): void {
  for (let i = 0; i < numSamples; (i = (i + 1) | 0)) {
    const xiOff = Math.imul(i, dIn);
    const aiOff = Math.imul(i, dHidden);
    for (let ip = i; ip < numSamples; (ip = (ip + 1) | 0)) {
      const xipOff = Math.imul(ip, dIn);
      const aipOff = Math.imul(ip, dHidden);
      let sx = 1.0;
      for (let j = 0; j < dIn; (j = (j + 1) | 0)) sx += xData[(xiOff + j) | 0]! * xData[(xipOff + j) | 0]!;
      let sa = 1.0;
      for (let h = 0; h < dHidden; (h = (h + 1) | 0)) sa += actAll[(aiOff + h) | 0]! * actAll[(aipOff + h) | 0]!;
      storeSymmetric(kx, numSamples, i, ip, sx);
      storeSymmetric(ka, numSamples, i, ip, sa);
    }
  }
}

function assembleFactoredDualGram(
  kx: Float64Array,
  ka: Float64Array,
  dz: Float64Array,
  numSamples: number,
  dOut: number,
  dHidden: number,
  g: Float64Array,
): void {
  const m = Math.imul(numSamples, dOut) | 0;
  const invN = 1.0 / numSamples;
  for (let r = 0; r < m; (r = (r + 1) | 0)) {
    const ir = (r / dOut) | 0;
    const cr = (r - Math.imul(ir, dOut)) | 0;
    const kRow = Math.imul(ir, numSamples);
    const rOff = Math.imul(r, dHidden);
    let c = r;
    for (; ((c + 3) | 0) < m; c = (c + 4) | 0) {
      const c0 = Math.imul(c, dHidden);
      const c1 = (c0 + dHidden) | 0;
      const c2 = (c1 + dHidden) | 0;
      const c3 = (c2 + dHidden) | 0;
      let s0 = 0.0;
      let s1 = 0.0;
      let s2 = 0.0;
      let s3 = 0.0;
      for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
        const x = dz[(rOff + h) | 0]!;
        s0 += x * dz[(c0 + h) | 0]!;
        s1 += x * dz[(c1 + h) | 0]!;
        s2 += x * dz[(c2 + h) | 0]!;
        s3 += x * dz[(c3 + h) | 0]!;
      }
      for (let lane = 0; lane < 4; (lane = (lane + 1) | 0)) {
        const col = (c + lane) | 0;
        const ic = (col / dOut) | 0;
        const cc = (col - Math.imul(ic, dOut)) | 0;
        const dot = lane === 0 ? s0 : lane === 1 ? s1 : lane === 2 ? s2 : s3;
        const val = (kx[(kRow + ic) | 0]! * dot + (cr === cc ? ka[(kRow + ic) | 0]! : 0.0)) * invN;
        storeSymmetric(g, m, r, col, val);
      }
    }
    for (; c < m; (c = (c + 1) | 0)) {
      const ic = (c / dOut) | 0;
      const cc = (c - Math.imul(ic, dOut)) | 0;
      const cOff = Math.imul(c, dHidden);
      let s = 0.0;
      for (let h = 0; h < dHidden; (h = (h + 1) | 0)) s += dz[(rOff + h) | 0]! * dz[(cOff + h) | 0]!;
      const val = (kx[(kRow + ic) | 0]! * s + (cr === cc ? ka[(kRow + ic) | 0]! : 0.0)) * invN;
      storeSymmetric(g, m, r, c, val);
    }
  }
}

export function computeStudentGaussNewtonDualSpectrum(
  net: ReturnType<typeof createPhase6StudentNetwork>,
  theta: Float64Array,
  relCut = 1e-10,
  precomputedActAndD1?: { readonly act: Float64Array; readonly d1: Float64Array },
  withDualResidual = false,
): {
  readonly m: number;
  readonly rank: number;
  readonly values: Float64Array;
  readonly vt: Float64Array;
  readonly finite: boolean;
  readonly traceGaussNewton: number | null;
  readonly traceGaussNewtonSq: number | null;
  readonly effectiveRank: number | null;
  readonly dualResidualMax: number | null;
} {
  const { numSamples, dIn, dHidden, dOut, w2Base, xData } = net;
  const m = Math.imul(numSamples, dOut) | 0;
  let actAll: Float64Array;
  let d1All: Float64Array;
  if (precomputedActAndD1 !== undefined) {
    actAll = precomputedActAndD1.act;
    d1All = precomputedActAndD1.d1;
  } else {
    actAll = new Float64Array(Math.imul(numSamples, dHidden));
    d1All = new Float64Array(Math.imul(numSamples, dHidden));
    const actBuf = new Float64Array(dHidden);
    const d1Buf = new Float64Array(dHidden);
    for (let i = 0; i < numSamples; (i = (i + 1) | 0)) {
      net.computeHiddenActivations(theta, i, actBuf, d1Buf);
      const off = Math.imul(i, dHidden);
      actAll.set(actBuf, off);
      d1All.set(d1Buf, off);
    }
  }
  const kx = new Float64Array(Math.imul(numSamples, numSamples));
  const ka = new Float64Array(Math.imul(numSamples, numSamples));
  buildSampleKernels(xData, actAll, numSamples, dIn, dHidden, kx, ka);
  const dz = new Float64Array(Math.imul(m, dHidden));
  for (let i = 0; i < numSamples; (i = (i + 1) | 0)) {
    const hOff = Math.imul(i, dHidden);
    for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
      const rOff = Math.imul((Math.imul(i, dOut) + c) | 0, dHidden);
      const wOff = (w2Base + Math.imul(c, dHidden)) | 0;
      for (let h = 0; h < dHidden; (h = (h + 1) | 0)) dz[(rOff + h) | 0] = theta[(wOff + h) | 0]! * d1All[(hOff + h) | 0]!;
    }
  }
  const g = new Float64Array(Math.imul(m, m));
  assembleFactoredDualGram(kx, ka, dz, numSamples, dOut, dHidden, g);
  let trace = 0.0;
  let traceSq = 0.0;
  for (let r = 0; r < m; (r = (r + 1) | 0)) {
    trace += g[(Math.imul(r, m) + r) | 0]!;
    for (let c = 0; c < m; (c = (c + 1) | 0)) {
      const x = g[(Math.imul(r, m) + c) | 0]!;
      traceSq += x * x;
    }
  }
  if (!Number.isFinite(trace) || !Number.isFinite(traceSq)) {
    return { m, rank: 0, values: new Float64Array(0), vt: new Float64Array(0), finite: false, traceGaussNewton: null, traceGaussNewtonSq: null, effectiveRank: null, dualResidualMax: null };
  }
  const eig = computeSymmetricEigendecompositionHouseholderQl(g, m);
  const top = eig.values[(m - 1) | 0]!;
  let rank = 0;
  while (rank < m && eig.values[(m - 1 - rank) | 0]! > relCut * top) rank = (rank + 1) | 0;
  const values = new Float64Array(rank);
  const vt = new Float64Array(Math.imul(rank, m));
  for (let k = 0; k < rank; (k = (k + 1) | 0)) {
    const src = (m - 1 - k) | 0;
    values[k] = eig.values[src]!;
    for (let e = 0; e < m; (e = (e + 1) | 0)) vt[(Math.imul(k, m) + e) | 0] = eig.qt[(Math.imul(src, m) + e) | 0]!;
  }
  const dualResidualMax = withDualResidual ? computeDualEigenResidualMax(g, m, values, vt, rank) : null;
  return { m, rank, values, vt, finite: true, traceGaussNewton: trace, traceGaussNewtonSq: traceSq, effectiveRank: traceSq > 0.0 ? (trace * trace) / traceSq : 0.0, dualResidualMax };
}
