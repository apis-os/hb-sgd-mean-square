/**
 * @file labs/mean-square-boundary/apply-heavy-ball-second-moment-operator.ts
 * @description One application of the second-moment operator `T(Q) = E[F_t Q F_t^T]` of heavy-ball SGD (`applyHeavyBallSecondMomentOperator`) on a quadratic with random-feature minibatches `H_B = (1/B) sum_b z_b z_b^T`, `z = H^{1/2} omega`, i.i.d. draws, coordinates of excess kurtosis `kappaFeat`, in the eigenbasis of `H = diag(lambda)`. With `F_t = [[A_t, -beta I], [I, 0]]`, `A_t = (1 + beta) I - eta H_B` and `A = E[A_t]`: `T(Q)_11 = A Q11 A + eta^2 V(Q11) - beta A Q12 - beta Q21 A + beta^2 Q22`, `T(Q)_12 = A Q11 - beta Q21`, `T(Q)_21 = Q11 A - beta Q12`, `T(Q)_22 = Q11`, where `V(X)_kk = (lambda_k / B) sum_j lambda_j X_jj + ((1 + kappaFeat) / B) lambda_k^2 X_kk` and `V(X)_jk = (lambda_j lambda_k / B) X_kj` for `j != k` (the transpose matters off the symmetric matrices). `q` and `out` are row-major `2P x 2P` matrices; the cost is `O(P^2)` per application, which is what makes a matrix-free power iteration at `P` in the hundreds affordable. The loops run over 64 x 64 tiles so that the transposed read `X_kj` stays in cache; every entry is computed by the same expression whatever the tiling.
 * @cold
 */

export function applyHeavyBallSecondMomentOperator(
  lambdas: Float64Array,
  beta: number,
  eta: number,
  batchB: number,
  kappaFeat: number,
  q: Float64Array,
  out: Float64Array,
): void {
  const p = lambdas.length | 0;
  const n = (2 * p) | 0;
  const tile = 64;
  let traceHX = 0.0;
  for (let j = 0; j < p; (j = (j + 1) | 0)) traceHX += lambdas[j]! * q[(Math.imul(j, n) + j) | 0]!;
  const invB = 1.0 / batchB;
  const eta2 = eta * eta;
  const beta2 = beta * beta;
  const kurtFactor = (1.0 + kappaFeat) * invB;
  const onePlusBeta = 1.0 + beta;
  for (let jb = 0; jb < p; jb = (jb + tile) | 0) {
    const jEnd = jb + tile < p ? (jb + tile) | 0 : p;
    for (let kb = 0; kb < p; kb = (kb + tile) | 0) {
      const kEnd = kb + tile < p ? (kb + tile) | 0 : p;
      for (let j = jb; j < jEnd; (j = (j + 1) | 0)) {
        const lj = lambdas[j]!;
        const aj = onePlusBeta - eta * lj;
        const rowJ = Math.imul(j, n);
        const rowJp = Math.imul((j + p) | 0, n);
        for (let k = kb; k < kEnd; (k = (k + 1) | 0)) {
          const lk = lambdas[k]!;
          const ak = onePlusBeta - eta * lk;
          const q11 = q[(rowJ + k) | 0]!;
          const q12 = q[(rowJ + p + k) | 0]!;
          const q21 = q[(rowJp + k) | 0]!;
          const q22 = q[(rowJp + p + k) | 0]!;
          const v =
            j === k
              ? lj * traceHX * invB + kurtFactor * lj * lj * q11
              : lj * lk * invB * q[(Math.imul(k, n) + j) | 0]!;
          out[(rowJ + k) | 0] = aj * ak * q11 + eta2 * v - beta * aj * q12 - beta * q21 * ak + beta2 * q22;
          out[(rowJ + p + k) | 0] = aj * q11 - beta * q21;
          out[(rowJp + k) | 0] = q11 * ak - beta * q12;
          out[(rowJp + p + k) | 0] = q11;
        }
      }
    }
  }
}
