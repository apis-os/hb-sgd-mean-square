/**
 * @file labs/mean-square-boundary/compute-exact-hessian-trace.ts
 * @description Exact trace of a symmetric operator from `numParams` matrix-vector products with the canonical basis (`computeExactHessianTrace`): `tr H = sum_i e_i^T H e_i`. No probing estimate, no variance; the cost is one product per parameter.
 * @cold
 */
export function computeExactHessianTrace(
  applyHvp: (vec: Float64Array, outHv: Float64Array) => void,
  numParams: number,
): number {
  const p = numParams | 0;
  const e = new Float64Array(p);
  const hv = new Float64Array(p);
  let trace = 0.0;
  let compensation = 0.0;
  for (let i = 0; i < p; (i = (i + 1) | 0)) {
    e[i] = 1.0;
    applyHvp(e, hv);
    e[i] = 0.0;
    const y = hv[i]! - compensation;
    const t = trace + y;
    compensation = t - trace - y;
    trace = t;
  }
  return trace;
}
