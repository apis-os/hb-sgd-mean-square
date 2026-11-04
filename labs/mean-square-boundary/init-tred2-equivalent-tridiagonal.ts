/**
 * @file labs/mean-square-boundary/init-tred2-equivalent-tridiagonal.ts
 * @description The diagonal `d` and post-shift off-diagonal `e` that `computeSymmetricEigendecompositionHouseholderQl` hands to its
 * implicit QL when its input is the Lanczos tridiagonal (`initTred2EquivalentTridiagonal`; diagonal `alphas[0 .. m-1]`,
 * off-diagonal `betas[0 .. m-2] > 0`). The Householder reflections of `tred2` on a tridiagonal are `1 x 1` sign flips with exact
 * scaling (`beta / beta = 1`), so they leave `d = alphas` (with `-0` turned into `+0` for `i < m - 1`, as `alpha - 0` does),
 * set `e[m-2] = -beta[m-2]` and `e[i] = +beta[i]` below, `e[m-1] = 0`, and accumulate `vt = diag(-1, ..., -1, +1)` (2026-10-28
 * audits: 3 000 random and 1 219 hard tridiagonals, every value and vector entry `Object.is`). Valid only when every entry passes
 * `isTred2IdentityEntrySafe`.
 * @cold
 */
export function initTred2EquivalentTridiagonal(alphas: Float64Array, betas: Float64Array, m: number, d: Float64Array, e: Float64Array): void {
  const m1 = (m - 1) | 0;
  const m2 = (m - 2) | 0;
  for (let i = 0; i < m; (i = (i + 1) | 0)) {
    d[i] = i < m1 ? alphas[i]! + 0.0 : alphas[i]!;
    e[i] = i < m1 ? (i === m2 ? -betas[i]! : betas[i]!) : 0.0;
  }
}
