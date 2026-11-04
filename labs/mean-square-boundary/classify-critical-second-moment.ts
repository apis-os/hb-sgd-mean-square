/**
 * @file labs/mean-square-boundary/classify-critical-second-moment.ts
 * @description Where the critical second moment of a threshold lives (`classifyCriticalSecondMoment`), from the top eigenvector
 * `x` of the threshold operator in the coordinates of `assembleThmG6OperatorMatrix` (kept pairs `pairs`). As in Phase 4,
 * `Sigma_0*_kk = x_kk / sqrt(2 l_k)` (`l` the curvature of the iteration), the sign is fixed by a positive trace, and the lag-1
 * ratio is `sum_k (1 - eta l_k / (1 + beta)) Sigma_kk / tr Sigma`. Modes are split by `|lambda_k|` (raw eigenvalues of the mean
 * Hessian) against `cut` into rigid (`>= cut`) and flat, each by the sign of `lambda_k`; mode `topIndex` is reported on its own.
 * With `cut = 2 |lambda_min|` and `topIndex = n - 1` this is the classification of Phase 4 (rigid = rigid positive, positive
 * flat = flat positive, negative = both negative shares). Modes outside the kept pairs carry no weight.
 * @cold
 */
export function classifyCriticalSecondMoment(
  x: Float64Array,
  pairs: Int32Array,
  pairA: Int32Array,
  pairB: Int32Array,
  values: Float64Array,
  curvature: Float64Array,
  eta: number,
  beta: number,
  cut: number,
  topIndex: number,
): {
  readonly shareTop: number;
  readonly shareRigidPositive: number;
  readonly shareRigidNegative: number;
  readonly shareFlatPositive: number;
  readonly shareFlatNegative: number;
  readonly lagOneRatio: number;
} {
  let tot = 0.0;
  let top = 0.0;
  let rigidPos = 0.0;
  let rigidNeg = 0.0;
  let flatPos = 0.0;
  let flatNeg = 0.0;
  let lag = 0.0;
  const dim = pairs.length | 0;
  for (let r = 0; r < dim; (r = (r + 1) | 0)) {
    const p = pairs[r]!;
    const k = pairA[p]!;
    if (pairB[p] !== k) continue;
    const lk = curvature[k]!;
    const d = x[r]! / Math.sqrt(2.0 * lk);
    tot += d;
    lag += (1.0 - (eta * lk) / (1.0 + beta)) * d;
    const v = values[k]!;
    if (k === topIndex) top += d;
    else if (Math.abs(v) >= cut) {
      if (v >= 0.0) rigidPos += d;
      else rigidNeg += d;
    } else if (v >= 0.0) flatPos += d;
    else flatNeg += d;
  }
  return {
    shareTop: top / tot,
    shareRigidPositive: rigidPos / tot,
    shareRigidNegative: rigidNeg / tot,
    shareFlatPositive: flatPos / tot,
    shareFlatNegative: flatNeg / tot,
    lagOneRatio: lag / tot,
  };
}
