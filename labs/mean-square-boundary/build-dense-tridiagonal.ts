/**
 * @file labs/mean-square-boundary/build-dense-tridiagonal.ts
 * @description The dense row-major `m x m` symmetric tridiagonal with diagonal `alphas[0 .. m-1]` and off-diagonal
 * `betas[0 .. m-2]` (`buildDenseTridiagonal`), the matrix `computeLanczosTopEigenvalue` hands to
 * `computeSymmetricEigendecompositionHouseholderQl` at step `m`.
 * @cold
 */
export function buildDenseTridiagonal(alphas: Float64Array, betas: Float64Array, m: number): Float64Array {
  const tri = new Float64Array(Math.imul(m, m));
  for (let i = 0; i < m; (i = (i + 1) | 0)) {
    tri[(Math.imul(i, m) + i) | 0] = alphas[i]!;
    if (((i + 1) | 0) < m) {
      tri[(Math.imul(i, m) + i + 1) | 0] = betas[i]!;
      tri[(Math.imul((i + 1) | 0, m) + i) | 0] = betas[i]!;
    }
  }
  return tri;
}
