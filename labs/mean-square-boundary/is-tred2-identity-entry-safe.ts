/**
 * @file labs/mean-square-boundary/is-tred2-identity-entry-safe.ts
 * @description Range test for an entry of a Lanczos tridiagonal (`isTred2IdentityEntrySafe`). The Householder reduction
 * (`tred2` of `computeSymmetricEigendecompositionHouseholderQl`) applied to a tridiagonal with positive off-diagonal reproduces
 * `initTred2EquivalentTridiagonal` exactly only while no intermediate (`2 alpha`, `alpha / 2`, `beta / beta`) overflows or
 * becomes subnormal: the 2026-10-28 audits found `alpha >= 2^1023 / 2` (overflow to NaN) and subnormal `alpha` (a lost last bit)
 * as counterexamples. Zero and magnitudes in `[2^-960, 2^960)` are accepted with a wide margin; NaN, infinities and the rest send
 * the solve to the Householder path.
 * @cold
 */
export function isTred2IdentityEntrySafe(x: number): boolean {
  const ax = Math.abs(x);
  return x === 0.0 || (ax >= 1.0261342003245941e-289 && ax < 9.7453140114e288);
}
