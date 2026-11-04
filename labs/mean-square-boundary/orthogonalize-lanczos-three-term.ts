/**
 * @file labs/mean-square-boundary/orthogonalize-lanczos-three-term.ts
 * @description One three-term Lanczos orthogonalisation (`orthogonalizeLanczosThreeTerm`): `alpha = w . vCur`, then
 * `w <- w - alpha vCur - betaPrev vPrev` and `beta = |w|`, with the four-way left-associative sums of the private step of
 * `computeLanczosTopEigenvalue`, so `alpha`, `beta` and `w` are bit-identical to it.
 * @cold
 */
export function orthogonalizeLanczosThreeTerm(
  w: Float64Array,
  vCur: Float64Array,
  vPrev: Float64Array,
  betaPrev: number,
  total: number,
): { readonly alpha: number; readonly beta: number } {
  const total4 = total & ~3;
  let alpha = 0.0;
  let e = 0;
  for (; e < total4; (e = (e + 4) | 0)) {
    alpha = alpha + w[e]! * vCur[e]! + w[(e + 1) | 0]! * vCur[(e + 1) | 0]! + w[(e + 2) | 0]! * vCur[(e + 2) | 0]! + w[(e + 3) | 0]! * vCur[(e + 3) | 0]!;
  }
  for (; e < total; (e = (e + 1) | 0)) alpha += w[e]! * vCur[e]!;
  let betaSq = 0.0;
  e = 0;
  for (; e < total4; (e = (e + 4) | 0)) {
    const r0 = w[e]! - alpha * vCur[e]! - betaPrev * vPrev[e]!;
    const r1 = w[(e + 1) | 0]! - alpha * vCur[(e + 1) | 0]! - betaPrev * vPrev[(e + 1) | 0]!;
    const r2 = w[(e + 2) | 0]! - alpha * vCur[(e + 2) | 0]! - betaPrev * vPrev[(e + 2) | 0]!;
    const r3 = w[(e + 3) | 0]! - alpha * vCur[(e + 3) | 0]! - betaPrev * vPrev[(e + 3) | 0]!;
    w[e] = r0;
    w[(e + 1) | 0] = r1;
    w[(e + 2) | 0] = r2;
    w[(e + 3) | 0] = r3;
    betaSq = betaSq + r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3;
  }
  for (; e < total; (e = (e + 1) | 0)) {
    const r = w[e]! - alpha * vCur[e]! - betaPrev * vPrev[e]!;
    w[e] = r;
    betaSq += r * r;
  }
  return { alpha, beta: Math.sqrt(betaSq) };
}
