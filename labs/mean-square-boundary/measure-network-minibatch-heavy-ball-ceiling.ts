/**
 * @file labs/mean-square-boundary/measure-network-minibatch-heavy-ball-ceiling.ts
 * @description Bisection on the learning rate for the empirical ceiling of heavy-ball SGD with real minibatches on a network at its checkpoint (`measureNetworkMinibatchHeavyBallCeiling`): a learning rate is unstable when the log energy growth of `simulateNetworkMinibatchHeavyBallEnergy` exceeds the threshold. Same protocol as `measureEmpiricalEossCeiling` of Phase 1; returns the midpoint and half-width of the final bracket and whether the upper end was already stable.
 * @cold
 */
import { simulateNetworkMinibatchHeavyBallEnergy } from './simulate-network-minibatch-heavy-ball-energy.ts';

export function measureNetworkMinibatchHeavyBallCeiling(
  mlp: Parameters<typeof simulateNetworkMinibatchHeavyBallEnergy>[0],
  sampleGrads0: Float64Array,
  fullGrad0: Float64Array,
  startDirection: Float64Array,
  startNorm: number,
  beta: number,
  etaUpper: number,
  batchB: number,
  noiseScaleTau: number,
  streams: number,
  steps: number,
  divergenceEnergy: number,
  iterations: number,
  logGrowthThreshold: number,
  seedLo: number,
  seedHi: number,
): { readonly etaCeiling: number; readonly bracketHalfWidth: number; readonly upperWasStable: boolean } {
  let lo = 0.0;
  let hi = etaUpper;
  const topGrowth = simulateNetworkMinibatchHeavyBallEnergy(
    mlp, sampleGrads0, fullGrad0, startDirection, startNorm, beta, etaUpper, batchB, noiseScaleTau, streams, steps, divergenceEnergy, seedLo, seedHi,
  );
  const upperWasStable = topGrowth <= logGrowthThreshold;
  if (upperWasStable) {
    return { etaCeiling: etaUpper, bracketHalfWidth: 0.0, upperWasStable };
  }
  for (let it = 0; it < iterations; (it = (it + 1) | 0)) {
    const mid = 0.5 * (lo + hi);
    const growth = simulateNetworkMinibatchHeavyBallEnergy(
      mlp, sampleGrads0, fullGrad0, startDirection, startNorm, beta, mid, batchB, noiseScaleTau, streams, steps, divergenceEnergy,
      (seedLo + Math.imul(it + 1, 0x9e3779b9)) | 0, (seedHi ^ Math.imul(it + 7, 0x85ebca6b)) | 0,
    );
    if (growth > logGrowthThreshold) hi = mid;
    else lo = mid;
  }
  return { etaCeiling: 0.5 * (lo + hi), bracketHalfWidth: 0.5 * (hi - lo), upperWasStable };
}
