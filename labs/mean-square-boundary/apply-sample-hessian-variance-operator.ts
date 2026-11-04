/**
 * @file labs/mean-square-boundary/apply-sample-hessian-variance-operator.ts
 * @description The curvature-noise operator of Theorem G.6 for a minibatch of `B` samples drawn with replacement
 * (`applySampleHessianVarianceOperator`): `V(Y) = (1/B) E[(H_i - H) Y (H_i - H)] = (1/B) [ (1/N) sum_i H_i Y H_i - H Y H ]` for
 * symmetric `Y`, from Hessian-vector products only. For each sample, `t1 = rows of H_i Y` (`= Y H_i`), `t2 = t1^T = H_i Y`, and
 * the rows of `H_i t2` add column `j` of `H_i Y H_i` to row `j` of `out`; samples are added in increasing order. The mean term
 * leaves `t1 = Y H`, `t2 = H Y` and `t3 = H Y H` for the caller. A tether `mu I` added to every `H_i` does not change `V`.
 * @cold
 */
import type { createPhase4RowExecutor } from './create-phase4-row-executor.ts';

export async function applySampleHessianVarianceOperator(
  exec: Awaited<ReturnType<typeof createPhase4RowExecutor>>,
  y: Float64Array,
  out: Float64Array,
  batchB: number,
  t1: Float64Array,
  t2: Float64Array,
  t3: Float64Array,
): Promise<void> {
  const nS = exec.provider.numSamples;
  const invN = 1.0 / nS;
  out.fill(0.0);
  for (let i = 0; i < nS; (i = (i + 1) | 0)) {
    await exec.hessianRows(i, y, t1, 1.0, false);
    await exec.transpose(t1, t2);
    await exec.hessianRows(i, t2, out, invN, true);
  }
  await exec.hessianRows(-1, y, t1, 1.0, false);
  await exec.transpose(t1, t2);
  await exec.hessianRows(-1, t2, t3, 1.0, false);
  const invB = 1.0 / batchB;
  const total = out.length;
  for (let e = 0; e < total; (e = (e + 1) | 0)) out[e] = (out[e]! - t3[e]!) * invB;
}
