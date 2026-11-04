/**
 * @file labs/mean-square-boundary/compute-momentum-commutator-coefficient.ts
 * @description The coefficient `c_beta = beta / ((1 - beta)(1 - beta^2))` of the double commutator in Theorem G.6
 * (`computeMomentumCommutatorCoefficient`), evaluated as `beta / ((1 - beta)(1 - beta)(1 + beta))`. The textbook form rounds
 * `beta^2` and then subtracts it from 1, which multiplies that rounding by `beta^2 / (1 - beta^2)` (relative error 129 u at
 * `beta = 0.999`, 2 264 u at `0.9999`; found by an external audit of Phase 6, 2026-10-29). Here `1 - beta` is exact for
 * `beta` in `[0.5, 1]` (Sterbenz) and has one rounding otherwise, `1 + beta` has one, and two products and a division follow, so the
 * relative error is at most `gamma_6` for every `beta` in `[0, 1)` (at most `gamma_4` on `[0.5, 1)`; the certificates budget more); `tests/phase6-certified-bracket.test.ts` checks this
 * against exact rational arithmetic.
 * @cold
 */
export function computeMomentumCommutatorCoefficient(beta: number): number {
  const omb = 1.0 - beta;
  return beta / (omb * omb * (1.0 + beta));
}
