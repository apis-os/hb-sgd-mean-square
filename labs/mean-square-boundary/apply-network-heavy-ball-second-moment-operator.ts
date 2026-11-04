/**
 * @file labs/mean-square-boundary/apply-network-heavy-ball-second-moment-operator.ts
 * @description One application of the exact second-moment operator `T(Q) = E[F_t Q F_t^T]` of tethered heavy-ball SGD on
 * per-sample Hessians (`applyNetworkHeavyBallSecondMomentOperator`), on symmetric `2P x 2P` matrices stored as blocks `q11`,
 * `q12`, `q22` (`q21 = q12^T`). With `A_t = (1 + beta - eta mu) I - eta H_{B_t}`, `A = E[A_t]`, `c = 1 + beta - eta mu`:
 * `T_11 = A Q11 A + eta^2 V(Q11) - beta (A Q12 + (A Q12)^T) + beta^2 Q22`, `T_12 = A Q11 - beta Q21`, `T_22 = Q11`, where
 * `A Q11 A = c^2 Q11 - c eta (H Q11 + Q11 H) + eta^2 H Q11 H` and `V` is `applySampleHessianVarianceOperator`, which also
 * returns `Q11 H`, `H Q11` and `H Q11 H`. `H Q12` comes from the rows of `Q21`. This is the operator of Theorem F in the
 * original basis, with no use of the eigenbasis or of Theorem G.6, so its spectral radius is an independent check of the
 * threshold `eta*` of Theorem G.6. Scratch `s1 .. s6` are `P x P`.
 * @cold
 */
import { applySampleHessianVarianceOperator } from './apply-sample-hessian-variance-operator.ts';
import type { createPhase4RowExecutor } from './create-phase4-row-executor.ts';

export async function applyNetworkHeavyBallSecondMomentOperator(
  exec: Awaited<ReturnType<typeof createPhase4RowExecutor>>,
  beta: number,
  eta: number,
  mu: number,
  batchB: number,
  q11: Float64Array,
  q12: Float64Array,
  q22: Float64Array,
  o11: Float64Array,
  o12: Float64Array,
  o22: Float64Array,
  s: readonly [Float64Array, Float64Array, Float64Array, Float64Array, Float64Array, Float64Array],
): Promise<void> {
  const [v11, yh, hy, hyh, q21, g] = s;
  await applySampleHessianVarianceOperator(exec, q11, v11, batchB, yh, hy, hyh);
  await exec.transpose(q12, q21);
  // Rows of H applied to the rows of Q21 (the columns of Q12): g = (H Q12)^T.
  await exec.hessianRows(-1, q21, g, 1.0, false);
  const c = 1.0 + beta - eta * mu;
  const c2 = c * c;
  const cEta = c * eta;
  const eta2 = eta * eta;
  const beta2 = beta * beta;
  const n = exec.n;
  const tile = 32;
  for (let jb = 0; jb < n; (jb = (jb + tile) | 0)) {
    const jEnd = jb + tile < n ? (jb + tile) | 0 : n;
    for (let kb = 0; kb < n; (kb = (kb + tile) | 0)) {
      const kEnd = kb + tile < n ? (kb + tile) | 0 : n;
      for (let j = jb; j < jEnd; (j = (j + 1) | 0)) {
        const row = Math.imul(j, n);
        let kj = (Math.imul(kb, n) + j) | 0;
        for (let k = kb; k < kEnd; (k = (k + 1) | 0)) {
          const jk = (row + k) | 0;
          const q11jk = q11[jk]!;
          const q21jk = q21[jk]!;
          const hyjk = hy[jk]!;
          // (A Q12)_jk = c Q12_jk - eta (H Q12)_jk = c q12[jk] - eta g[kj]; its transpose at (j, k) is c q21[jk] - eta g[jk].
          const aq12 = c * q12[jk]! - eta * g[kj]!;
          const aq12t = c * q21jk - eta * g[jk]!;
          const aqa = c2 * q11jk - cEta * (hyjk + yh[jk]!) + eta2 * hyh[jk]!;
          o11[jk] = aqa + eta2 * v11[jk]! - beta * (aq12 + aq12t) + beta2 * q22[jk]!;
          o12[jk] = c * q11jk - eta * hyjk - beta * q21jk;
          o22[jk] = q11jk;
          kj = (kj + n) | 0;
        }
      }
    }
  }
}
