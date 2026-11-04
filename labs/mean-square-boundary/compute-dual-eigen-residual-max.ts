/**
 * @file labs/mean-square-boundary/compute-dual-eigen-residual-max.ts
 * @description The largest eigen-residual `max_k ||G v_k - lambda_k v_k||_2` of `rank` approximate eigenpairs of a symmetric
 * row-major `m x m` matrix `g` (`computeDualEigenResidualMax`; `values[k]`, `vt[k m + e]`), computed in floating point (a diagnostic,
 * not certified). Used by `computeStudentGaussNewtonDualSpectrum` for Phase 6 stage 2; split out after the sixth audit round, whose
 * mutation `dualResidualMax = 0` passed the only test.
 * @cold
 */
export function computeDualEigenResidualMax(g: Float64Array, m: number, values: Float64Array, vt: Float64Array, rank: number): number {
  let worst = 0.0;
  for (let k = 0; k < rank; (k = (k + 1) | 0)) {
    const vOff = Math.imul(k, m);
    let r2 = 0.0;
    for (let row = 0; row < m; (row = (row + 1) | 0)) {
      const gOff = Math.imul(row, m);
      let sum = -values[k]! * vt[(vOff + row) | 0]!;
      for (let c = 0; c < m; (c = (c + 1) | 0)) sum += g[(gOff + c) | 0]! * vt[(vOff + c) | 0]!;
      r2 += sum * sum;
    }
    worst = Math.max(worst, Math.sqrt(r2));
  }
  return worst;
}
