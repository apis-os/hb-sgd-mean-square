/**
 * @file labs/mean-square-boundary/measure-empirical-eoss-ceiling.ts
 * @description Bisection on the learning rate for the empirical mean-square ceiling of heavy-ball SGD with random-feature minibatches (`measureEmpiricalEossCeiling`): a learning rate is unstable when the log energy growth of `simulateHeavyBallFeatureNoiseEnergy` exceeds the threshold; returns the midpoint and half-width of the final bracket.
 * @cold
 */
import { simulateHeavyBallFeatureNoiseEnergy } from './simulate-heavy-ball-feature-noise-energy.ts';

export function measureEmpiricalEossCeiling(
  sqrtLambdas: Float64Array,
  beta: number,
  etaUpper: number,
  batchB: number,
  featureKind: 0 | 1,
  pool: Float64Array | null,
  poolSize: number,
  sigmaXi: number,
  streams: number,
  steps: number,
  iterations: number,
  logGrowthThreshold: number,
  seedLo: number,
  seedHi: number,
): { readonly etaCeiling: number; readonly bracketHalfWidth: number; readonly upperWasStable: boolean } {
  let lo = 0.0;
  let hi = etaUpper;
  const topGrowth = simulateHeavyBallFeatureNoiseEnergy(
    sqrtLambdas, beta, etaUpper, batchB, featureKind, pool, poolSize, sigmaXi, streams, steps, seedLo, seedHi,
  );
  const upperWasStable = topGrowth <= logGrowthThreshold;
  if (upperWasStable) {
    return { etaCeiling: etaUpper, bracketHalfWidth: 0.0, upperWasStable };
  }
  for (let it = 0; it < iterations; (it = (it + 1) | 0)) {
    const mid = 0.5 * (lo + hi);
    const growth = simulateHeavyBallFeatureNoiseEnergy(
      sqrtLambdas, beta, mid, batchB, featureKind, pool, poolSize, sigmaXi, streams, steps,
      (seedLo + Math.imul(it + 1, 0x9e3779b9)) | 0, (seedHi ^ Math.imul(it + 7, 0x85ebca6b)) | 0,
    );
    if (growth > logGrowthThreshold) hi = mid;
    else lo = mid;
  }
  return { etaCeiling: 0.5 * (lo + hi), bracketHalfWidth: 0.5 * (hi - lo), upperWasStable };
}
