/**
 * @file labs/mean-square-boundary/solve-tridiagonal-ql-exact.ts
 * @description The implicit QL iteration of `computeSymmetricEigendecompositionHouseholderQl` (EISPACK `tql2`), on a diagonal `d`
 * and a post-shift off-diagonal `e` of length `m`, followed by `sortTridiagonalQlEigenvalues` (`solveTridiagonalQlExact`). The
 * updates of `d` and `e` never read the vectors, and each column of the vector array is rotated independently, so the vectors are
 * kept either in full (`vt`, `m x m`, rows rotated as there) or as the last column only (`z[i] = vt[i][m-1]`, `O(m)` per sweep
 * instead of `O(m^2)`); `null` skips them. Every operation on `d`, `e` and the kept vector entries is the same and in the same
 * order as in the reference, so the results are bit-identical given the same `d`, `e` and initial vectors.
 * @cold
 */
import { computeScaledHypot } from './compute-scaled-hypot.ts';
import { sortTridiagonalQlEigenvalues } from './sort-tridiagonal-ql-eigenvalues.ts';

export function solveTridiagonalQlExact(
  d: Float64Array,
  e: Float64Array,
  m: number,
  vt: Float64Array | null,
  z: Float64Array | null,
  perm: Int32Array,
  topOnly: boolean,
): void {
  let f = 0.0;
  let tst1 = 0.0;
  const eps = 2.220446049250313e-16;
  for (let l = 0; l < m; (l = (l + 1) | 0)) {
    const t = Math.abs(d[l]!) + Math.abs(e[l]!);
    if (t > tst1) tst1 = t;
    let mm = l;
    while (mm < m) {
      if (Math.abs(e[mm]!) <= eps * tst1) break;
      mm = (mm + 1) | 0;
    }
    if (mm > l) {
      do {
        let g = d[l]!;
        let p = (d[(l + 1) | 0]! - g) / (2.0 * e[l]!);
        let r = computeScaledHypot(p, 1.0);
        if (p < 0) r = -r;
        d[l] = e[l]! / (p + r);
        d[(l + 1) | 0] = e[l]! * (p + r);
        const dl1 = d[(l + 1) | 0]!;
        let h = g - d[l]!;
        for (let i = (l + 2) | 0; i < m; (i = (i + 1) | 0)) d[i] = d[i]! - h;
        f = f + h;
        p = d[mm]!;
        let c = 1.0;
        let c2 = c;
        let c3 = c;
        const el1 = e[(l + 1) | 0]!;
        let s = 0.0;
        let s2 = 0.0;
        for (let i = (mm - 1) | 0; i >= l; (i = (i - 1) | 0)) {
          c3 = c2;
          c2 = c;
          s2 = s;
          g = c * e[i]!;
          h = c * p;
          r = computeScaledHypot(p, e[i]!);
          e[(i + 1) | 0] = s * r;
          s = e[i]! / r;
          c = p / r;
          p = c * d[i]! - s * g;
          d[(i + 1) | 0] = h + s * (c * g + s * d[i]!);
          if (vt !== null) {
            const rowI = Math.imul(i, m);
            const rowI1 = (rowI + m) | 0;
            for (let k = 0; k < m; (k = (k + 1) | 0)) {
              const hk = vt[(rowI1 + k) | 0]!;
              const vk = vt[(rowI + k) | 0]!;
              vt[(rowI1 + k) | 0] = s * vk + c * hk;
              vt[(rowI + k) | 0] = c * vk - s * hk;
            }
          } else if (z !== null) {
            const hk = z[(i + 1) | 0]!;
            const vk = z[i]!;
            z[(i + 1) | 0] = s * vk + c * hk;
            z[i] = c * vk - s * hk;
          }
        }
        p = (-s * s2 * c3 * el1 * e[l]!) / dl1;
        e[l] = s * p;
        d[l] = c * p;
      } while (Math.abs(e[l]!) > eps * tst1);
    }
    d[l] = d[l]! + f;
    e[l] = 0.0;
  }
  sortTridiagonalQlEigenvalues(d, perm, m, topOnly);
}
