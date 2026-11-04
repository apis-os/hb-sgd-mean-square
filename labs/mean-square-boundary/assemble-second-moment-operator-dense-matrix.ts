/**
 * @file labs/mean-square-boundary/assemble-second-moment-operator-dense-matrix.ts
 * @description The second-moment operator `T` of `applyEigenbasisSecondMomentOperator` as a dense matrix
 * (`assembleSecondMomentOperatorDenseMatrix`; amendment 4, for the Perron certificate at small `P`). Coordinates of a symmetric
 * `2n x 2n` matrix `Q = [[Q11, Q12], [Q12^T, Q22]]`: the upper triangle of `Q11` in pair order (`nPair` entries, pair `p` is
 * `(pairA[p], pairB[p])`), then `Q12` row-major (`n^2`), then the upper triangle of `Q22` (`nPair`); `dim = 2 nPair + n^2`.
 * Column `c` is `T` applied to the coordinate matrix of basis vector `c` (a pair `(a, b)` sets both `(a, b)` and `(b, a)`), read
 * back in the same coordinates (`m[r * dim + c]`). The coordinates are not orthonormal; the eigenvalues of `T` are those of the
 * matrix, and an eigenvector maps back to the eigenmatrix by `coordinatesToBlocks`.
 * @cold
 */
import { applyEigenbasisSecondMomentOperator } from './apply-eigenbasis-second-moment-operator.ts';

export function assembleSecondMomentOperatorDenseMatrix(
  vmat: Float64Array,
  n: number,
  nPair: number,
  pairA: Int32Array,
  pairB: Int32Array,
  curvature: Float64Array,
  beta: number,
  eta: number,
): { readonly m: Float64Array; readonly dim: number } {
  const nn = Math.imul(n, n);
  const dim = (2 * nPair + nn) | 0;
  const m = new Float64Array(Math.imul(dim, dim));
  const q11 = new Float64Array(nn);
  const q12 = new Float64Array(nn);
  const q22 = new Float64Array(nn);
  const o11 = new Float64Array(nn);
  const o12 = new Float64Array(nn);
  const o22 = new Float64Array(nn);
  const x = new Float64Array(nPair);
  const y = new Float64Array(nPair);
  const setPair = (blk: Float64Array, p: number, v: number): void => {
    const a = pairA[p]!;
    const b = pairB[p]!;
    blk[(Math.imul(a, n) + b) | 0] = v;
    blk[(Math.imul(b, n) + a) | 0] = v;
  };
  for (let c = 0; c < dim; (c = (c + 1) | 0)) {
    if (c < nPair) setPair(q11, c, 1.0);
    else if (c < nPair + nn) q12[(c - nPair) | 0] = 1.0;
    else setPair(q22, (c - nPair - nn) | 0, 1.0);
    applyEigenbasisSecondMomentOperator(vmat, n, nPair, pairA, pairB, curvature, beta, eta, q11, q12, q22, o11, o12, o22, x, y);
    for (let p = 0; p < nPair; (p = (p + 1) | 0)) {
      const ab = (Math.imul(pairA[p]!, n) + pairB[p]!) | 0;
      m[(Math.imul(p, dim) + c) | 0] = o11[ab]!;
      m[(Math.imul((nPair + nn + p) | 0, dim) + c) | 0] = o22[ab]!;
    }
    for (let e = 0; e < nn; (e = (e + 1) | 0)) m[(Math.imul((nPair + e) | 0, dim) + c) | 0] = o12[e]!;
    if (c < nPair) setPair(q11, c, 0.0);
    else if (c < nPair + nn) q12[(c - nPair) | 0] = 0.0;
    else setPair(q22, (c - nPair - nn) | 0, 0.0);
  }
  return { m, dim };
}
