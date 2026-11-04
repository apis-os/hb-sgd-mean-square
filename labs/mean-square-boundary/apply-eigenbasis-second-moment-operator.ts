/**
 * @file labs/mean-square-boundary/apply-eigenbasis-second-moment-operator.ts
 * @description One application of the exact second-moment operator `T(Q) = E[F_t Q F_t^T]` of heavy-ball SGD in the eigenbasis of
 * the mean curvature (`applyEigenbasisSecondMomentOperator`), for small `P`. With `a_k = 1 + beta - eta l_k` (`l` the curvature of
 * the iteration) the mean part acts entry by entry, as in `applyNetworkHeavyBallSecondMomentOperator` with `A` diagonal:
 * `T_11 = a_j a_k Q11 - beta (a_j Q12_jk + a_k Q12_kj) + beta^2 Q22 + eta^2 V(Q11)`, `T_12 = a_j Q11 - beta Q12_kj`, `T_22 = Q11`;
 * `V(Q11)` is the dense matrix of `assembleCurvatureNoiseOperatorMatrix` applied to the orthonormal coordinates of `Q11`. Blocks
 * are full `n x n` arrays (`q21 = q12^T`); `x`, `y` are scratch of length `nPair`. The dense product runs six rows per pass
 * (`multiplyDenseMatrixVectorSixRows`, each row summed in the same order), so the result is bit-identical to
 * `applyEigenbasisSecondMomentOperatorReference` (the test oracle).
 * @cold
 */
import { multiplyDenseMatrixVectorSixRows } from './multiply-dense-matrix-vector-six-rows.ts';

export function applyEigenbasisSecondMomentOperator(
  vmat: Float64Array,
  n: number,
  nPair: number,
  pairA: Int32Array,
  pairB: Int32Array,
  curvature: Float64Array,
  beta: number,
  eta: number,
  q11: Float64Array,
  q12: Float64Array,
  q22: Float64Array,
  o11: Float64Array,
  o12: Float64Array,
  o22: Float64Array,
  x: Float64Array,
  y: Float64Array,
): void {
  for (let p = 0; p < nPair; (p = (p + 1) | 0)) {
    const a = pairA[p]!;
    const b = pairB[p]!;
    const e = q11[(Math.imul(a, n) + b) | 0]!;
    x[p] = a === b ? e : Math.SQRT2 * e;
  }
  multiplyDenseMatrixVectorSixRows(vmat, x, y, nPair);
  const eta2 = eta * eta;
  const beta2 = beta * beta;
  for (let j = 0; j < n; (j = (j + 1) | 0)) {
    const aj = 1.0 + beta - eta * curvature[j]!;
    for (let k = 0; k < n; (k = (k + 1) | 0)) {
      const ak = 1.0 + beta - eta * curvature[k]!;
      const jk = (Math.imul(j, n) + k) | 0;
      const kj = (Math.imul(k, n) + j) | 0;
      const q11jk = q11[jk]!;
      o11[jk] = aj * ak * q11jk - beta * (aj * q12[jk]! + ak * q12[kj]!) + beta2 * q22[jk]!;
      o12[jk] = aj * q11jk - beta * q12[kj]!;
      o22[jk] = q11jk;
    }
  }
  for (let p = 0; p < nPair; (p = (p + 1) | 0)) {
    const a = pairA[p]!;
    const b = pairB[p]!;
    const v = a === b ? y[p]! : y[p]! * Math.SQRT1_2;
    const ab = (Math.imul(a, n) + b) | 0;
    o11[ab] = o11[ab]! + eta2 * v;
    if (a !== b) {
      const ba = (Math.imul(b, n) + a) | 0;
      o11[ba] = o11[ba]! + eta2 * v;
    }
  }
}
