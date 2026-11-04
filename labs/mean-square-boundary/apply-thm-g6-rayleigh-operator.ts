/**
 * @file labs/mean-square-boundary/apply-thm-g6-rayleigh-operator.ts
 * @description One application of the symmetric operator `A = L_0^{-1/2} B_beta L_0^{-1/2}` of Theorem G.6(3) on `S^P`, whose
 * largest eigenvalue is `1 / eta*` (`applyThmG6RayleighOperator`). `x` and `out` are in the eigenbasis of the tethered Hessian
 * `H + mu I` (eigenvalues `lambdaT`, eigenvectors the columns of `q`, `qt = q^T`), where `L_0` multiplies entry `(j, k)` by
 * `lambdaT_j + lambdaT_k`, `H X H / (1 + beta)` by `lambdaT_j lambdaT_k / (1 + beta)` and the double commutator
 * `[H, [H, X]]` by `(lambda_j - lambda_k)^2` (the tether cancels). The curvature-noise term is applied in the original basis:
 * `Y = q (D^{-1/2} . x) q^T`, `Z = V(Y)` by `applySampleHessianVarianceOperator`, back by `qt Z q`. The second product of each
 * rotation is formed as an exactly symmetric matrix (`matmulSymmetric`), so `A` is symmetric to rounding and the Lanczos iteration sees a symmetric operator. Scratch `s1 .. s6` are
 * `n x n`.
 * @cold
 */
import { applySampleHessianVarianceOperator } from './apply-sample-hessian-variance-operator.ts';
import type { createPhase4RowExecutor } from './create-phase4-row-executor.ts';

export async function applyThmG6RayleighOperator(
  exec: Awaited<ReturnType<typeof createPhase4RowExecutor>>,
  lambdaT: Float64Array,
  q: Float64Array,
  qt: Float64Array,
  beta: number,
  batchB: number,
  x: Float64Array,
  out: Float64Array,
  s: readonly [Float64Array, Float64Array, Float64Array, Float64Array, Float64Array, Float64Array],
): Promise<void> {
  const n = exec.n;
  const [s1, s2, s3, s4, s5, s6] = s;
  const n4 = n & ~3;
  for (let j = 0; j < n; (j = (j + 1) | 0)) {
    const row = Math.imul(j, n);
    const lj = lambdaT[j]!;
    let k = 0;
    for (; k < n4; (k = (k + 4) | 0)) {
      const e = (row + k) | 0;
      s1[e] = x[e]! / Math.sqrt(lj + lambdaT[k]!);
      s1[(e + 1) | 0] = x[(e + 1) | 0]! / Math.sqrt(lj + lambdaT[(k + 1) | 0]!);
      s1[(e + 2) | 0] = x[(e + 2) | 0]! / Math.sqrt(lj + lambdaT[(k + 2) | 0]!);
      s1[(e + 3) | 0] = x[(e + 3) | 0]! / Math.sqrt(lj + lambdaT[(k + 3) | 0]!);
    }
    for (; k < n; (k = (k + 1) | 0)) s1[(row + k) | 0] = x[(row + k) | 0]! / Math.sqrt(lj + lambdaT[k]!);
  }
  await exec.matmul(q, s1, s2);
  await exec.matmulSymmetric(s2, qt, s3);
  await applySampleHessianVarianceOperator(exec, s3, s4, batchB, s2, s5, s6);
  await exec.matmul(qt, s4, s2);
  await exec.matmulSymmetric(s2, q, s5);
  const invOnePlusBeta = 1.0 / (1.0 + beta);
  const commutator = beta / ((1.0 - beta) * (1.0 - beta * beta));
  const invOneMinusBeta = 1.0 / (1.0 - beta);
  for (let j = 0; j < n; (j = (j + 1) | 0)) {
    const row = Math.imul(j, n);
    const lj = lambdaT[j]!;
    const ljScaled = lj * invOnePlusBeta;
    let k = 0;
    for (; k < n4; (k = (k + 4) | 0)) {
      const lk0 = lambdaT[k]!;
      const lk1 = lambdaT[(k + 1) | 0]!;
      const lk2 = lambdaT[(k + 2) | 0]!;
      const lk3 = lambdaT[(k + 3) | 0]!;
      const e0 = (row + k) | 0;
      const e1 = (e0 + 1) | 0;
      const e2 = (e0 + 2) | 0;
      const e3 = (e0 + 3) | 0;
      const d0 = lj - lk0;
      const d1 = lj - lk1;
      const d2 = lj - lk2;
      const d3 = lj - lk3;
      out[e0] = ((lj * lk0 * invOnePlusBeta - commutator * d0 * d0) * s1[e0]! + s5[e0]! * invOneMinusBeta) / Math.sqrt(lj + lk0);
      out[e1] = ((lj * lk1 * invOnePlusBeta - commutator * d1 * d1) * s1[e1]! + s5[e1]! * invOneMinusBeta) / Math.sqrt(lj + lk1);
      out[e2] = ((lj * lk2 * invOnePlusBeta - commutator * d2 * d2) * s1[e2]! + s5[e2]! * invOneMinusBeta) / Math.sqrt(lj + lk2);
      out[e3] = ((lj * lk3 * invOnePlusBeta - commutator * d3 * d3) * s1[e3]! + s5[e3]! * invOneMinusBeta) / Math.sqrt(lj + lk3);
    }
    for (; k < n; (k = (k + 1) | 0)) {
      const lk = lambdaT[k]!;
      const e = (row + k) | 0;
      const diff = lj - lk;
      const multiplier = lj * lk * invOnePlusBeta - commutator * diff * diff;
      const noise = s5[e]! * invOneMinusBeta;
      out[e] = (multiplier * s1[e]! + noise) / Math.sqrt(lj + lk);
    }
    void ljScaled;
  }
}
