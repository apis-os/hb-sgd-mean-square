/**
 * @file labs/mean-square-boundary/is-phase5-c6-seed.ts
 * @description Whether a run carries the `c6` check (`isPhase5C6Seed`): the registration computes it once per phase, on the
 * control run `eta = 0.01`, `mu_train = 0`, of the first seed of the phase, which is 1 for Phase 5 and 1401 for Phase 5b.
 * @cold
 */
export function isPhase5C6Seed(seed: number): boolean {
  return seed === 1 || seed === 1401;
}
