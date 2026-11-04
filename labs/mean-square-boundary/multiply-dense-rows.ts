/**
 * @file labs/mean-square-boundary/multiply-dense-rows.ts
 * @description Rows `r0 .. r1` of the dense product `C = A B` of row-major `n x n` matrices (`multiplyDenseRows`), tiled
 * over 32 rows of `C`, 128 rows of `B` and 512 columns so that a tile of `B` stays in the L2 cache. Every entry of `C` is
 * accumulated in increasing `k`, in groups of four, whatever the tiling and whatever the split of rows between workers,
 * so a product computed by one thread and by eight is the same to the bit.
 * @cold
 */

export function multiplyDenseRows(a: Float64Array, b: Float64Array, c: Float64Array, n: number, r0: number, r1: number): void {
  const rowTile = 32;
  const kTile = 128;
  const colTile = 512;
  for (let i = r0; i < r1; (i = (i + 1) | 0)) c.fill(0.0, Math.imul(i, n), (Math.imul(i, n) + n) | 0);
  for (let ib = r0; ib < r1; ib = (ib + rowTile) | 0) {
    const iEnd = ib + rowTile < r1 ? (ib + rowTile) | 0 : r1;
    const iEnd2 = (iEnd - 1) | 0;
    for (let jb = 0; jb < n; jb = (jb + colTile) | 0) {
      const jEnd = jb + colTile < n ? (jb + colTile) | 0 : n;
      for (let kb = 0; kb < n; kb = (kb + kTile) | 0) {
        const kEnd = kb + kTile < n ? (kb + kTile) | 0 : n;
        let i = ib;
        for (; i < iEnd2; i = (i + 2) | 0) {
          const ci0 = Math.imul(i, n);
          const ci1 = (ci0 + n) | 0;
          let k = kb;
          for (; ((k + 3) | 0) < kEnd; k = (k + 4) | 0) {
            const a00 = a[(ci0 + k) | 0]!, a01 = a[(ci0 + k + 1) | 0]!, a02 = a[(ci0 + k + 2) | 0]!, a03 = a[(ci0 + k + 3) | 0]!;
            const a10 = a[(ci1 + k) | 0]!, a11 = a[(ci1 + k + 1) | 0]!, a12 = a[(ci1 + k + 2) | 0]!, a13 = a[(ci1 + k + 3) | 0]!;
            const b0 = Math.imul(k, n), b1 = (b0 + n) | 0, b2 = (b1 + n) | 0, b3 = (b2 + n) | 0;
            for (let j = jb; j < jEnd; (j = (j + 1) | 0)) {
              const b0j = b[(b0 + j) | 0]!, b1j = b[(b1 + j) | 0]!, b2j = b[(b2 + j) | 0]!, b3j = b[(b3 + j) | 0]!;
              c[(ci0 + j) | 0] = c[(ci0 + j) | 0]! + (a00 * b0j + a01 * b1j + a02 * b2j + a03 * b3j);
              c[(ci1 + j) | 0] = c[(ci1 + j) | 0]! + (a10 * b0j + a11 * b1j + a12 * b2j + a13 * b3j);
            }
          }
          for (; k < kEnd; (k = (k + 1) | 0)) {
            const ak0 = a[(ci0 + k) | 0]!, ak1 = a[(ci1 + k) | 0]!;
            const bk = Math.imul(k, n);
            for (let j = jb; j < jEnd; (j = (j + 1) | 0)) {
              const bkj = b[(bk + j) | 0]!;
              c[(ci0 + j) | 0] = c[(ci0 + j) | 0]! + ak0 * bkj;
              c[(ci1 + j) | 0] = c[(ci1 + j) | 0]! + ak1 * bkj;
            }
          }
        }
        for (; i < iEnd; (i = (i + 1) | 0)) {
          const ci = Math.imul(i, n);
          let k = kb;
          for (; ((k + 3) | 0) < kEnd; k = (k + 4) | 0) {
            const a0 = a[(ci + k) | 0]!, a1 = a[(ci + k + 1) | 0]!, a2 = a[(ci + k + 2) | 0]!, a3 = a[(ci + k + 3) | 0]!;
            const b0 = Math.imul(k, n), b1 = (b0 + n) | 0, b2 = (b1 + n) | 0, b3 = (b2 + n) | 0;
            for (let j = jb; j < jEnd; (j = (j + 1) | 0)) {
              c[(ci + j) | 0] = c[(ci + j) | 0]! + (a0 * b[(b0 + j) | 0]! + a1 * b[(b1 + j) | 0]! + a2 * b[(b2 + j) | 0]! + a3 * b[(b3 + j) | 0]!);
            }
          }
          for (; k < kEnd; (k = (k + 1) | 0)) {
            const ak = a[(ci + k) | 0]!;
            const bk = Math.imul(k, n);
            for (let j = jb; j < jEnd; (j = (j + 1) | 0)) c[(ci + j) | 0] = c[(ci + j) | 0]! + ak * b[(bk + j) | 0]!;
          }
        }
      }
    }
  }
}
