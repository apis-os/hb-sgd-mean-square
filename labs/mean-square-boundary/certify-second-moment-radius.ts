/**
 * @file labs/mean-square-boundary/certify-second-moment-radius.ts
 * @description Collatz–Wielandt certificates for the spectral radius of the second-moment operator at an iterate
 * (`certifySecondMomentRadius`). `T(Q) = E[F Q F^T]` maps the positive semidefinite cone of `S^{2P}` into itself, so for
 * `Q > 0`: `T(Q) <= r Q` gives `rho(T) <= r` and `T(Q) >= s Q` gives `rho(T) >= s`; the tightest `s`, `r` are the extreme
 * eigenvalues of `Q^{-1/2} T(Q) Q^{-1/2}`. With `bracket = true` (small `P`) they are computed: Cholesky `Q = L L^T`,
 * `M = L^{-1} T(Q) L^{-T}`, full eigendecomposition of `M`. With `bracket = false` (large `P`) only the two decisions are
 * made, by Cholesky of `Q - T(Q)` (success: `rho < 1`) and of `T(Q) - Q` (success: `rho > 1`), which costs `O(P^3 / 3)` each.
 * `q*` hold `Q` and `o*` hold `T(Q)` as blocks (`*21 = *12^T`). A failed factorisation decides nothing; it does not mean the
 * opposite inequality.
 * @cold
 */
import { computeCholeskyInPlace } from './compute-cholesky-in-place.ts';
import { computeSymmetricEigendecompositionHouseholderQl } from './compute-symmetric-eigendecomposition-householder-ql.ts';

function assembleFull(b11: Float64Array, b12: Float64Array, b22: Float64Array, n: number, sign: number, other: readonly [Float64Array, Float64Array, Float64Array] | null): Float64Array {
  const m = (2 * n) | 0;
  const out = new Float64Array(Math.imul(m, m));
  for (let j = 0; j < n; (j = (j + 1) | 0)) {
    for (let k = 0; k < n; (k = (k + 1) | 0)) {
      const jk = (Math.imul(j, n) + k) | 0;
      const kj = (Math.imul(k, n) + j) | 0;
      const v11 = sign * b11[jk]! + (other === null ? 0.0 : other[0][jk]!);
      const v12 = sign * b12[jk]! + (other === null ? 0.0 : other[1][jk]!);
      const v21 = sign * b12[kj]! + (other === null ? 0.0 : other[1][kj]!);
      const v22 = sign * b22[jk]! + (other === null ? 0.0 : other[2][jk]!);
      out[(Math.imul(j, m) + k) | 0] = v11;
      out[(Math.imul(j, m) + n + k) | 0] = v12;
      out[(Math.imul((n + j) | 0, m) + k) | 0] = v21;
      out[(Math.imul((n + j) | 0, m) + n + k) | 0] = v22;
    }
  }
  for (let r = 0; r < m; (r = (r + 1) | 0)) {
    for (let c = (r + 1) | 0; c < m; (c = (c + 1) | 0)) {
      const v = 0.5 * (out[(Math.imul(r, m) + c) | 0]! + out[(Math.imul(c, m) + r) | 0]!);
      out[(Math.imul(r, m) + c) | 0] = v;
      out[(Math.imul(c, m) + r) | 0] = v;
    }
  }
  return out;
}

export function certifySecondMomentRadius(
  q: readonly [Float64Array, Float64Array, Float64Array],
  o: readonly [Float64Array, Float64Array, Float64Array],
  n: number,
  bracket: boolean,
): {
  readonly iteratePositiveDefinite: boolean;
  readonly certifiedBelowOne: boolean;
  readonly certifiedAboveOne: boolean;
  readonly lowerBound: number | null;
  readonly upperBound: number | null;
} {
  const m = (2 * n) | 0;
  const qFull = assembleFull(q[0], q[1], q[2], n, 1.0, null);
  const pd = computeCholeskyInPlace(qFull, m);
  if (!pd) return { iteratePositiveDefinite: false, certifiedBelowOne: false, certifiedAboveOne: false, lowerBound: null, upperBound: null };
  if (bracket) {
    const t = assembleFull(o[0], o[1], o[2], n, 1.0, null);
    // Y = L^{-1} T (forward substitution on each column), then M = L^{-1} Y^T (Y^T = T L^{-T}).
    const m4 = m & ~3;
    for (let pass = 0; pass < 2; (pass = (pass + 1) | 0)) {
      for (let r = 0; r < m; (r = (r + 1) | 0)) {
        const rowR = Math.imul(r, m);
        const diagR = qFull[(rowR + r) | 0]!;
        let c = 0;
        for (; c < m4; (c = (c + 4) | 0)) {
          let s0 = t[(rowR + c) | 0]!;
          let s1 = t[(rowR + c + 1) | 0]!;
          let s2 = t[(rowR + c + 2) | 0]!;
          let s3 = t[(rowR + c + 3) | 0]!;
          for (let k = 0; k < r; (k = (k + 1) | 0)) {
            const lRk = qFull[(rowR + k) | 0]!;
            const kOff = (Math.imul(k, m) + c) | 0;
            s0 -= lRk * t[kOff]!;
            s1 -= lRk * t[(kOff + 1) | 0]!;
            s2 -= lRk * t[(kOff + 2) | 0]!;
            s3 -= lRk * t[(kOff + 3) | 0]!;
          }
          t[(rowR + c) | 0] = s0 / diagR;
          t[(rowR + c + 1) | 0] = s1 / diagR;
          t[(rowR + c + 2) | 0] = s2 / diagR;
          t[(rowR + c + 3) | 0] = s3 / diagR;
        }
        for (; c < m; (c = (c + 1) | 0)) {
          let s = t[(rowR + c) | 0]!;
          for (let k = 0; k < r; (k = (k + 1) | 0)) s -= qFull[(rowR + k) | 0]! * t[(Math.imul(k, m) + c) | 0]!;
          t[(rowR + c) | 0] = s / diagR;
        }
      }
      for (let r = 0; r < m; (r = (r + 1) | 0)) {
        for (let c = (r + 1) | 0; c < m; (c = (c + 1) | 0)) {
          const tmp = t[(Math.imul(r, m) + c) | 0]!;
          t[(Math.imul(r, m) + c) | 0] = t[(Math.imul(c, m) + r) | 0]!;
          t[(Math.imul(c, m) + r) | 0] = tmp;
        }
      }
    }
    for (let r = 0; r < m; (r = (r + 1) | 0)) {
      for (let c = (r + 1) | 0; c < m; (c = (c + 1) | 0)) {
        const v = 0.5 * (t[(Math.imul(r, m) + c) | 0]! + t[(Math.imul(c, m) + r) | 0]!);
        t[(Math.imul(r, m) + c) | 0] = v;
        t[(Math.imul(c, m) + r) | 0] = v;
      }
    }
    const ev = computeSymmetricEigendecompositionHouseholderQl(t, m).values;
    const lowerBound = ev[0]!;
    const upperBound = ev[(m - 1) | 0]!;
    return { iteratePositiveDefinite: true, certifiedBelowOne: upperBound < 1.0, certifiedAboveOne: lowerBound > 1.0, lowerBound, upperBound };
  }
  const below = assembleFull(o[0], o[1], o[2], n, -1.0, q);
  const certifiedBelowOne = computeCholeskyInPlace(below, m);
  const above = assembleFull(q[0], q[1], q[2], n, -1.0, o);
  const certifiedAboveOne = computeCholeskyInPlace(above, m);
  return { iteratePositiveDefinite: true, certifiedBelowOne, certifiedAboveOne, lowerBound: null, upperBound: null };
}
