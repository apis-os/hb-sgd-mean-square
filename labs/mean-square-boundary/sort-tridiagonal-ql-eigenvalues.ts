/**
 * @file labs/mean-square-boundary/sort-tridiagonal-ql-eigenvalues.ts
 * @description The ascending selection sort that ends `computeSymmetricEigendecompositionHouseholderQl` (strict `<`), replayed on
 * an index permutation instead of moving eigenvector rows (`sortTridiagonalQlEigenvalues`): after the call `d` is sorted as there
 * and `perm[j]` is the unsorted row that the reference moves to position `j`. With `topOnly` and a unique, non-NaN maximum the
 * sort is skipped: the maximum is moved to position `m - 1` by every swap sequence of that sort, so only `perm[m - 1]` and
 * `d[m - 1]` are set; ties (including `+0` against `-0`) and NaN fall back to the full replay.
 * @cold
 */
export function sortTridiagonalQlEigenvalues(d: Float64Array, perm: Int32Array, m: number, topOnly: boolean): void {
  if (topOnly) {
    let arg = 0;
    let mx = d[0]!;
    let ties = 0;
    let nan = mx !== mx;
    for (let i = 1; i < m; (i = (i + 1) | 0)) {
      const v = d[i]!;
      if (v !== v) nan = true;
      if (v > mx) {
        mx = v;
        arg = i;
        ties = 0;
      } else if (v === mx) ties = (ties + 1) | 0;
    }
    if (ties === 0 && !nan) {
      perm[(m - 1) | 0] = arg;
      d[(m - 1) | 0] = mx;
      return;
    }
  }
  for (let i = 0; i < m; (i = (i + 1) | 0)) perm[i] = i;
  for (let i = 0; i < ((m - 1) | 0); (i = (i + 1) | 0)) {
    let k = i;
    let p = d[i]!;
    for (let j = (i + 1) | 0; j < m; (j = (j + 1) | 0)) {
      if (d[j]! < p) {
        k = j;
        p = d[j]!;
      }
    }
    if (k !== i) {
      d[k] = d[i]!;
      d[i] = p;
      const t = perm[i]!;
      perm[i] = perm[k]!;
      perm[k] = t;
    }
  }
}
