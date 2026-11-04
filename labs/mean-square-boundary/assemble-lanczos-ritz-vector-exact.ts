/**
 * @file labs/mean-square-boundary/assemble-lanczos-ritz-vector-exact.ts
 * @description The top Ritz vector of a dense Lanczos run stopped at step `m` (`assembleLanczosRitzVectorExact`), as
 * `computeLanczosTopEigenvalue` writes it: the coefficients are the top eigenvector of the `m x m` tridiagonal (one full-vector
 * QL, `solveTridiagonalQlExact` from `initTred2EquivalentTridiagonal` and `vt = diag(-1, ..., -1, +1)`, or the Householder
 * solver itself when `householder` is set), the vector is `sum_i c_i basis_i` accumulated in `i` order into `ritz` (zero on
 * entry) and scaled by `1 / |ritz|`.
 * @cold
 */
import { buildDenseTridiagonal } from './build-dense-tridiagonal.ts';
import { computeSymmetricEigendecompositionHouseholderQl } from './compute-symmetric-eigendecomposition-householder-ql.ts';
import { initTred2EquivalentTridiagonal } from './init-tred2-equivalent-tridiagonal.ts';
import { solveTridiagonalQlExact } from './solve-tridiagonal-ql-exact.ts';

export function assembleLanczosRitzVectorExact(
  basis: Float32Array,
  alphas: Float64Array,
  betas: Float64Array,
  m: number,
  dim: number,
  householder: boolean,
  ritz: Float64Array,
): void {
  let vt: Float64Array;
  let top: number;
  if (householder) {
    vt = computeSymmetricEigendecompositionHouseholderQl(buildDenseTridiagonal(alphas, betas, m), m).qt;
    top = Math.imul((m - 1) | 0, m);
  } else {
    vt = new Float64Array(Math.imul(m, m));
    for (let i = 0; i < m; (i = (i + 1) | 0)) vt[(Math.imul(i, m) + i) | 0] = i < ((m - 1) | 0) ? -1.0 : 1.0;
    const d = new Float64Array(m);
    const e = new Float64Array(m);
    const perm = new Int32Array(m);
    initTred2EquivalentTridiagonal(alphas, betas, m, d, e);
    solveTridiagonalQlExact(d, e, m, vt, null, perm, false);
    top = Math.imul(perm[(m - 1) | 0]!, m);
  }
  for (let i = 0; i < m; (i = (i + 1) | 0)) {
    const c = vt[(top + i) | 0]!;
    const off = Math.imul(i, dim);
    for (let e = 0; e < dim; (e = (e + 1) | 0)) ritz[e] = ritz[e]! + c * basis[(off + e) | 0]!;
  }
  let nr = 0.0;
  for (let e = 0; e < dim; (e = (e + 1) | 0)) nr += ritz[e]! * ritz[e]!;
  const inv = 1.0 / Math.sqrt(nr);
  for (let e = 0; e < dim; (e = (e + 1) | 0)) ritz[e] = ritz[e]! * inv;
}
