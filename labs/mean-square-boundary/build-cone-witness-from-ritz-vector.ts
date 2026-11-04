/**
 * @file labs/mean-square-boundary/build-cone-witness-from-ritz-vector.ts
 * @description The test matrix of the Phase 6 Collatz–Wielandt certificate (`buildConeWitnessFromRitzVector`): from a Ritz vector
 * `y` of `G = L0^{-1/2} B_beta L0^{-1/2}` (row-major `n x n`, eigenbasis of the mean curvature), `z_ab = y_ab / sqrt(lt_a + lt_b)` is
 * an approximation of the critical eigenmatrix `X*` (`L0 X* = eta* B_beta X*`, positive semidefinite by Theorem G.6(1), when `eta* < eta_det`); its sign
 * is chosen so that `tr z >= 0`, its positive part `z_+` is taken from `computeSymmetricEigendecompositionHouseholderQl`, and the
 * witness is `X = z_+ / lambda_max(z_+) + eps I`, exactly symmetric as stored. If `z` has no positive eigenvalue the witness is
 * `I`. Positive definiteness of the stored `X` is not assumed here; the certificate verifies it.
 * @cold
 */
import { computeSymmetricEigendecompositionHouseholderQl } from './compute-symmetric-eigendecomposition-householder-ql.ts';

export function buildConeWitnessFromRitzVector(lt: Float64Array, n: number, y: Float64Array, eps: number): Float64Array {
  const nn = n * n;
  const z = new Float64Array(nn);
  let tr = 0.0;
  for (let a = 0; a < n; a++) {
    for (let b = a; b < n; b++) {
      const v = (0.5 * (y[a * n + b]! + y[b * n + a]!)) / Math.sqrt(lt[a]! + lt[b]!);
      z[a * n + b] = v;
      z[b * n + a] = v;
    }
    tr += z[a * n + a]!;
  }
  if (tr < 0.0) for (let e = 0; e < nn; e++) z[e] = -z[e]!;
  const eig = computeSymmetricEigendecompositionHouseholderQl(z, n);
  const top = eig.values[n - 1]!;
  const x = new Float64Array(nn);
  if (top > 0.0) {
    for (let k = 0; k < n; k++) {
      const lam = eig.values[k]!;
      if (!(lam > 0.0)) continue;
      const s = lam / top;
      const off = k * n;
      for (let a = 0; a < n; a++) {
        const ca = s * eig.qt[off + a]!;
        for (let b = a; b < n; b++) x[a * n + b] = x[a * n + b]! + ca * eig.qt[off + b]!;
      }
    }
  }
  for (let a = 0; a < n; a++) {
    x[a * n + a] = top > 0.0 ? x[a * n + a]! + eps : 1.0;
    for (let b = a + 1; b < n; b++) x[b * n + a] = x[a * n + b]!;
  }
  return x;
}
