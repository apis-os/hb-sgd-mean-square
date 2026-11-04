/**
 * @file labs/mean-square-boundary/compute-trace-weighted-lower-bound.ts
 * @description A lower bound on the threshold of Theorem G.6 from a trace identity (`computeTraceWeightedLowerBound`; derived in
 * this lab, 2026-10-28, not a statement of the manuscript). For any `M = sum_k m_k u_k u_k^T`, `m_k > 0`, commuting with the mean
 * curvature (`l_k` its eigenvalues as seen by the iteration), the double commutator has zero trace against `M`, and the critical
 * equation `L_0 X* = eta* B_beta X*` with `X* >= 0` gives `2 (1 + beta) tr Z = eta* tr(Z N(M))`, `Z = (M L)^{1/2} X* (M L)^{1/2}`,
 * `N(M) = L + (kappa_beta / B) (M L)^{-1/2} E[D_i M D_i] (M L)^{-1/2}`; since `tr(Z N) <= lambda_max(N) tr Z`,
 * `eta* >= eta_tr(M) = 2 (1 + beta) / lambda_max(N(M))`. Here `E[D_i M D_i]` is formed in the eigenbasis from the packed
 * deviations of `buildEigenbasisSampleHessianDeviations` (`D_i` symmetric), and `lambda_max` by Householder QL (small `n`).
 * Bit-identical to `computeTraceWeightedLowerBoundReference` (the test oracle): the same per-sample sums in the same order, formed
 * in `3 x 3` blocks (`accumulateWeightedDeviationProducts`), and the top eigenvalue without the eigenvectors
 * (`computeSymmetricTopEigenvalueHouseholderQl`).
 * @cold
 */
import { accumulateWeightedDeviationProducts } from './accumulate-weighted-deviation-products.ts';
import { computeSymmetricTopEigenvalueHouseholderQl } from './compute-symmetric-top-eigenvalue-householder-ql.ts';
import { isRowMajorPairIndex } from './is-row-major-pair-index.ts';
import { unpackWeightedSampleDeviation } from './unpack-weighted-sample-deviation.ts';

export function computeTraceWeightedLowerBound(
  packed: Float64Array,
  numSamples: number,
  n: number,
  nPair: number,
  pairIndex: Int32Array,
  curvature: Float64Array,
  weights: Float64Array,
  beta: number,
  batchB: number,
): number {
  const nn = Math.imul(n, n);
  const s = new Float64Array(nn);
  const d = new Float64Array(nn);
  const dm = new Float64Array(nn);
  const rowMajor = isRowMajorPairIndex(pairIndex, n, nPair);
  for (let i = 0; i < numSamples; (i = (i + 1) | 0)) {
    unpackWeightedSampleDeviation(packed, Math.imul(i, nPair), n, pairIndex, weights, rowMajor, d, dm);
    accumulateWeightedDeviationProducts(dm, d, s, n);
  }
  const kappa = (1.0 + beta) / (1.0 - beta);
  const scale = kappa / (batchB * numSamples);
  const nm = new Float64Array(nn);
  for (let k = 0; k < n; (k = (k + 1) | 0)) {
    const rk = 1.0 / Math.sqrt(weights[k]! * curvature[k]!);
    for (let l = 0; l < n; (l = (l + 1) | 0)) {
      const rl = 1.0 / Math.sqrt(weights[l]! * curvature[l]!);
      const sym = 0.5 * (s[(Math.imul(k, n) + l) | 0]! + s[(Math.imul(l, n) + k) | 0]!);
      nm[(Math.imul(k, n) + l) | 0] = scale * rk * sym * rl + (k === l ? curvature[k]! : 0.0);
    }
  }
  return (2.0 * (1.0 + beta)) / computeSymmetricTopEigenvalueHouseholderQl(nm, n);
}
