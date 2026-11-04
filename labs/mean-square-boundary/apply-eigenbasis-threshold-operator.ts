/**
 * @file labs/mean-square-boundary/apply-eigenbasis-threshold-operator.ts
 * @description One application of `G = L0^{-1/2} B_beta L0^{-1/2}` of Theorem G.6(3) on symmetric `n x n` matrices in the
 * eigenbasis of the mean curvature of the iteration, eigenvalues `lt` (`applyEigenbasisThresholdOperator`): with
 * `z_ab = x_ab / sqrt(lt_a + lt_b)`, `out_ab = (d_ab z_ab + V(z)_ab / (1 - beta)) / sqrt(lt_a + lt_b)`,
 * `d_ab = lt_a lt_b / (1 + beta) - c_beta (lt_a - lt_b)^2`, `c_beta = beta / ((1 - beta)(1 - beta^2))`, `V` from `noise.apply`
 * without its error bound. `lambda_max(G) = 1 / eta*`. Matrices are full row-major `n x n` arrays, so the Frobenius inner product
 * of two of them is the plain dot product of the arrays; `zBuf` and `wBuf` are `n^2` scratch.
 * @cold
 */
import { computeMomentumCommutatorCoefficient } from './compute-momentum-commutator-coefficient.ts';

export function applyEigenbasisThresholdOperator(
  lt: Float64Array,
  n: number,
  beta: number,
  noise: { apply(x: Float64Array, out: Float64Array, errOut: Float64Array | null): void },
  x: Float64Array,
  out: Float64Array,
  zBuf: Float64Array,
  wBuf: Float64Array,
): void {
  const inv1pb = 1.0 / (1.0 + beta);
  const inv1mb = 1.0 / (1.0 - beta);
  const cb = computeMomentumCommutatorCoefficient(beta);
  for (let a = 0; a < n; a++) {
    for (let b = 0; b < n; b++) zBuf[a * n + b] = x[a * n + b]! / Math.sqrt(lt[a]! + lt[b]!);
  }
  noise.apply(zBuf, wBuf, null);
  for (let a = 0; a < n; a++) {
    for (let b = 0; b < n; b++) {
      const diff = lt[a]! - lt[b]!;
      const d = lt[a]! * lt[b]! * inv1pb - cb * diff * diff;
      out[a * n + b] = (d * zBuf[a * n + b]! + wBuf[a * n + b]! * inv1mb) / Math.sqrt(lt[a]! + lt[b]!);
    }
  }
}
