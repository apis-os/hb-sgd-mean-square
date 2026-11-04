/**
 * @file labs/mean-square-boundary/reduce-symmetric-to-tridiagonal-householder.ts
 * @description The Householder reduction (EISPACK `tred2`, JAMA form) of `computeSymmetricEigendecompositionHouseholderQl`, copied
 * operation for operation, followed by what that solver hands to its QL when the eigenvectors are not needed
 * (`reduceSymmetricToTridiagonalHouseholder`): `d[j] = vt[j][j]` (the vector accumulation saves exactly these entries before it
 * modifies them, and its own use of `d` is scratch) and the off-diagonal shifted down by one with `e[n-1] = 0`. `vt` holds the
 * matrix on entry and is overwritten; `d` and `e` have length `n`.
 * @cold
 */
export function reduceSymmetricToTridiagonalHouseholder(vt: Float64Array, d: Float64Array, e: Float64Array, n: number): void {
  for (let j = 0; j < n; (j = (j + 1) | 0)) d[j] = vt[(Math.imul(j, n) + n - 1) | 0]!;
  for (let i = (n - 1) | 0; i > 0; (i = (i - 1) | 0)) {
    let scale = 0.0;
    let h = 0.0;
    for (let k = 0; k < i; (k = (k + 1) | 0)) scale += Math.abs(d[k]!);
    if (scale === 0.0) {
      e[i] = d[(i - 1) | 0]!;
      for (let j = 0; j < i; (j = (j + 1) | 0)) {
        d[j] = vt[(Math.imul(j, n) + i - 1) | 0]!;
        vt[(Math.imul(j, n) + i) | 0] = 0.0;
        vt[(Math.imul(i, n) + j) | 0] = 0.0;
      }
    } else {
      for (let k = 0; k < i; (k = (k + 1) | 0)) {
        d[k] = d[k]! / scale;
        h += d[k]! * d[k]!;
      }
      let f = d[(i - 1) | 0]!;
      let g = Math.sqrt(h);
      if (f > 0) g = -g;
      e[i] = scale * g;
      h = h - f * g;
      d[(i - 1) | 0] = f - g;
      for (let j = 0; j < i; (j = (j + 1) | 0)) e[j] = 0.0;
      for (let j = 0; j < i; (j = (j + 1) | 0)) {
        const rowJ = Math.imul(j, n);
        f = d[j]!;
        vt[(Math.imul(i, n) + j) | 0] = f;
        g = e[j]! + vt[(rowJ + j) | 0]! * f;
        for (let k = (j + 1) | 0; k <= ((i - 1) | 0); (k = (k + 1) | 0)) {
          const vkj = vt[(rowJ + k) | 0]!;
          g += vkj * d[k]!;
          e[k] = e[k]! + vkj * f;
        }
        e[j] = g;
      }
      f = 0.0;
      for (let j = 0; j < i; (j = (j + 1) | 0)) {
        e[j] = e[j]! / h;
        f += e[j]! * d[j]!;
      }
      const hh = f / (h + h);
      for (let j = 0; j < i; (j = (j + 1) | 0)) e[j] = e[j]! - hh * d[j]!;
      for (let j = 0; j < i; (j = (j + 1) | 0)) {
        const rowJ = Math.imul(j, n);
        f = d[j]!;
        g = e[j]!;
        for (let k = j; k <= ((i - 1) | 0); (k = (k + 1) | 0)) vt[(rowJ + k) | 0] = vt[(rowJ + k) | 0]! - (f * e[k]! + g * d[k]!);
        d[j] = vt[(rowJ + i - 1) | 0]!;
        vt[(rowJ + i) | 0] = 0.0;
      }
    }
    d[i] = h;
  }
  for (let j = 0; j < n; (j = (j + 1) | 0)) d[j] = vt[(Math.imul(j, n) + j) | 0]!;
  for (let i = 1; i < n; (i = (i + 1) | 0)) e[(i - 1) | 0] = e[i]!;
  if (n > 0) e[(n - 1) | 0] = 0.0;
}
