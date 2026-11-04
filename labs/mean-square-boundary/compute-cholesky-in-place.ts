/**
 * @file labs/mean-square-boundary/compute-cholesky-in-place.ts
 * @description Cholesky factorisation `A = L L^T` of a dense symmetric row-major `n x n` matrix in place
 * (`computeCholeskyInPlace`): the lower triangle is overwritten with `L` (4-way row-blocked contiguous dot products in
 * exact sequential `k = 0 .. j - 1` order), the strict upper triangle is left as it was. Returns `false` at the first
 * pivot that is not positive, which is how the Phase 4 certificates test `Q - T(Q) > 0` and `T(Q) - Q > 0`.
 * @cold
 */

export function computeCholeskyInPlace(a: Float64Array, n: number): boolean {
  const n3 = (n - 3) | 0;
  for (let j = 0; j < n; (j = (j + 1) | 0)) {
    const rowJ = Math.imul(j, n);
    let d = a[(rowJ + j) | 0]!;
    for (let k = 0; k < j; (k = (k + 1) | 0)) {
      const ljk = a[(rowJ + k) | 0]!;
      d -= ljk * ljk;
    }
    if (!(d > 0.0)) return false;
    const ljj = Math.sqrt(d);
    a[(rowJ + j) | 0] = ljj;
    const inv = 1.0 / ljj;
    let i = (j + 1) | 0;
    for (; i < n3; i = (i + 4) | 0) {
      const r0 = Math.imul(i, n);
      const r1 = (r0 + n) | 0;
      const r2 = (r1 + n) | 0;
      const r3 = (r2 + n) | 0;
      let s0 = a[(r0 + j) | 0]!;
      let s1 = a[(r1 + j) | 0]!;
      let s2 = a[(r2 + j) | 0]!;
      let s3 = a[(r3 + j) | 0]!;
      for (let k = 0; k < j; (k = (k + 1) | 0)) {
        const ajk = a[(rowJ + k) | 0]!;
        s0 -= a[(r0 + k) | 0]! * ajk;
        s1 -= a[(r1 + k) | 0]! * ajk;
        s2 -= a[(r2 + k) | 0]! * ajk;
        s3 -= a[(r3 + k) | 0]! * ajk;
      }
      a[(r0 + j) | 0] = s0 * inv;
      a[(r1 + j) | 0] = s1 * inv;
      a[(r2 + j) | 0] = s2 * inv;
      a[(r3 + j) | 0] = s3 * inv;
    }
    for (; i < n; (i = (i + 1) | 0)) {
      const rowI = Math.imul(i, n);
      let s = a[(rowI + j) | 0]!;
      for (let k = 0; k < j; (k = (k + 1) | 0)) s -= a[(rowI + k) | 0]! * a[(rowJ + k) | 0]!;
      a[(rowI + j) | 0] = s * inv;
    }
  }
  return true;
}
