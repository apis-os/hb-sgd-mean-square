/**
 * @file labs/mean-square-boundary/transpose-dense-rows.ts
 * @description Rows `r0 .. r1` of the transpose of a row-major `n x n` matrix (`transposeDenseRows`): `dst[j][k] = src[k][j]`.
 * Tiled in 32x32 cache blocks with 4x unrolled inner copy; pure copies, so the result is bit-identical regardless of tiling or worker split.
 * @cold
 */

export function transposeDenseRows(src: Float64Array, dst: Float64Array, n: number, r0: number, r1: number): void {
  const tile = 32;
  const n2 = (n << 1) | 0;
  const n3 = (n2 + n) | 0;
  const n4 = (n << 2) | 0;
  for (let jb = r0; jb < r1; jb = (jb + tile) | 0) {
    const jEnd = jb + tile < r1 ? (jb + tile) | 0 : r1;
    for (let kb = 0; kb < n; kb = (kb + tile) | 0) {
      const kEnd = kb + tile < n ? (kb + tile) | 0 : n;
      const kEnd4 = (kEnd - 3) | 0;
      for (let j = jb; j < jEnd; j = (j + 1) | 0) {
        const row = Math.imul(j, n);
        let k = kb;
        let srcIdx = (Math.imul(k, n) + j) | 0;
        for (; k < kEnd4; k = (k + 4) | 0) {
          dst[(row + k) | 0] = src[srcIdx]!;
          dst[(row + k + 1) | 0] = src[(srcIdx + n) | 0]!;
          dst[(row + k + 2) | 0] = src[(srcIdx + n2) | 0]!;
          dst[(row + k + 3) | 0] = src[(srcIdx + n3) | 0]!;
          srcIdx = (srcIdx + n4) | 0;
        }
        for (; k < kEnd; k = (k + 1) | 0) {
          dst[(row + k) | 0] = src[srcIdx]!;
          srcIdx = (srcIdx + n) | 0;
        }
      }
    }
  }
}
