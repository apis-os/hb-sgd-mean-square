/**
 * @file labs/mean-square-boundary/compute-roundoff-gamma.ts
 * @description The rounding constant `gamma_k = k u / (1 - k u)` of Higham (*Accuracy and Stability of Numerical Algorithms*,
 * 2nd ed., §3.1) for IEEE-754 binary64, `u = 2^-53` (`computeRoundoffGamma`): a sum of `k` floating-point terms in any order
 * satisfies `|fl(s) - s| <= gamma_{k - 1} sum |terms|`, an inner product of length `k` errs by at most `gamma_k sum |x_i y_i|`, and `k` rounded multiplications or divisions in a row give a relative error
 * at most `gamma_k`. The value is enlarged by the factor `1 + 1e-7`, so that its own rounding never makes it smaller than the
 * exact constant; `Infinity` when `k u >= 1`. Used by the certificates of Phase 6. These relative bounds assume no underflow and
 * no overflow (eighth audit round: `fl(2^-1074 / 2) = 0` has relative error 1); an underflowing product or quotient adds an absolute
 * error of at most `2^-1075`, which each Phase 6 routine budgets separately.
 * @cold
 */
export function computeRoundoffGamma(k: number): number {
  const ku = k * 1.1102230246251565e-16;
  if (!(ku < 1.0)) return Infinity;
  return (ku / (1.0 - ku)) * 1.0000001;
}
