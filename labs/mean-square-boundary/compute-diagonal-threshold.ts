/**
 * @file labs/mean-square-boundary/compute-diagonal-threshold.ts
 * @description The diagonal threshold of Theorem diag-threshold (`computeDiagonalThreshold`): `eta_D = 2 (1 + beta) /
 * lambda_max(Lambda + kappa_beta Gamma~)`, `Gamma~_kl = Gamma_kl / sqrt(l_k l_l)`, `kappa_beta = (1 + beta) / (1 - beta)`, with
 * `Gamma_kl` the diagonal of the Gram matrix of `assembleCurvatureNoiseGram` at the pair `(k, l)` and `l` the curvature of the
 * iteration. Also returns the Perron vector `v*` of `Lambda + kappa_beta Gamma~` (entrywise positive for an irreducible
 * `Gamma`), the top eigenvalue and unit eigenvector `X_inf` of `Gamma~` (the operator `V~` of Proposition diag-expansion on the
 * diagonal block) and the second eigenvalue of the diagonal block `G_DD = (Lambda + kappa_beta Gamma~) / (2 (1 + beta))`. By
 * full Householder-QL eigendecomposition of the `n x n` matrices (small `n`).
 * @cold
 */
import { computeSymmetricEigendecompositionHouseholderQl } from './compute-symmetric-eigendecomposition-householder-ql.ts';

export function computeDiagonalThreshold(
  g: Float64Array,
  n: number,
  nPair: number,
  pairIndex: Int32Array,
  curvature: Float64Array,
  beta: number,
): {
  readonly etaD: number;
  readonly perron: Float64Array;
  readonly gD: number;
  readonly lambda2GDD: number;
  readonly vTildeMax: number;
  readonly xInf: Float64Array;
} {
  const kappa = (1.0 + beta) / (1.0 - beta);
  const nn = Math.imul(n, n);
  const m = new Float64Array(nn);
  const vt = new Float64Array(nn);
  for (let k = 0; k < n; (k = (k + 1) | 0)) {
    for (let l = 0; l < n; (l = (l + 1) | 0)) {
      const p = pairIndex[(Math.imul(k, n) + l) | 0]!;
      const gt = g[(Math.imul(p, nPair) + p) | 0]! / Math.sqrt(curvature[k]! * curvature[l]!);
      vt[(Math.imul(k, n) + l) | 0] = gt;
      m[(Math.imul(k, n) + l) | 0] = kappa * gt + (k === l ? curvature[k]! : 0.0);
    }
  }
  const em = computeSymmetricEigendecompositionHouseholderQl(m, n);
  const top = em.values[(n - 1) | 0]!;
  const perron = Float64Array.from(em.qt.subarray(Math.imul((n - 1) | 0, n), nn));
  let sum = 0.0;
  for (let k = 0; k < n; (k = (k + 1) | 0)) sum += perron[k]!;
  if (sum < 0.0) for (let k = 0; k < n; (k = (k + 1) | 0)) perron[k] = -perron[k]!;
  const ev = computeSymmetricEigendecompositionHouseholderQl(vt, n);
  const xInf = Float64Array.from(ev.qt.subarray(Math.imul((n - 1) | 0, n), nn));
  return {
    etaD: (2.0 * (1.0 + beta)) / top,
    perron,
    gD: top / (2.0 * (1.0 + beta)),
    lambda2GDD: n > 1 ? em.values[(n - 2) | 0]! / (2.0 * (1.0 + beta)) : Number.NEGATIVE_INFINITY,
    vTildeMax: ev.values[(n - 1) | 0]!,
    xInf,
  };
}
