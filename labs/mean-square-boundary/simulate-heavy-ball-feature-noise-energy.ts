/**
 * @file labs/mean-square-boundary/simulate-heavy-ball-feature-noise-energy.ts
 * @description Heavy-ball SGD on a quadratic in the eigenbasis of `H` with minibatch Hessians `H_B = (1/B) sum z z^T`, `z = H^{1/2} omega` (`simulateHeavyBallFeatureNoiseEnergy`): Gaussian or Rademacher i.i.d. features, or a fixed pool sampled without replacement; returns the log ratio of the mean-square energy of the last quarter to that of the second quarter, the mean-square divergence statistic used by the Phase 1 bisection.
 * @cold
 */
import { fdlibmLog } from '../../src/math/index.ts';
import { createXorshift128Plus } from './prng-xorshift128p.ts';

export function simulateHeavyBallFeatureNoiseEnergy(
  sqrtLambdas: Float64Array,
  beta: number,
  eta: number,
  batchB: number,
  featureKind: 0 | 1,
  pool: Float64Array | null,
  poolSize: number,
  sigmaXi: number,
  streams: number,
  steps: number,
  seedLo: number,
  seedHi: number,
): number {
  const p = sqrtLambdas.length | 0;
  const prng = createXorshift128Plus(seedLo | 0, seedHi | 0);
  const q = new Float64Array(p);
  const qPrev = new Float64Array(p);
  const hq = new Float64Array(p);
  const z = new Float64Array(p);
  const idx = new Int32Array(Math.max(1, poolSize | 0));
  for (let i = 0; i < poolSize; (i = (i + 1) | 0)) idx[i] = i;
  const w1Start = (steps >> 2) | 0;
  const w1End = (steps >> 1) | 0;
  const w2Start = ((3 * steps) >> 2) | 0;
  const invB = 1.0 / batchB;
  const onePlusBeta = 1.0 + beta;
  let energyEarly = 0.0;
  let energyLate = 0.0;
  let signBits = 0;
  let signLeft = 0;

  for (let s = 0; s < streams; (s = (s + 1) | 0)) {
    q.fill(0.0);
    qPrev.fill(0.0);
    for (let t = 0; t < steps; (t = (t + 1) | 0)) {
      hq.fill(0.0);
      for (let b = 0; b < batchB; (b = (b + 1) | 0)) {
        if (pool !== null) {
          const j = (b + Math.floor(prng.nextFloat01() * (poolSize - b))) | 0;
          const tmp = idx[b]!;
          idx[b] = idx[j]!;
          idx[j] = tmp;
          const off = Math.imul(idx[b]!, p);
          for (let k = 0; k < p; (k = (k + 1) | 0)) z[k] = pool[(off + k) | 0]!;
        } else if (featureKind === 0) {
          for (let k = 0; k < p; (k = (k + 1) | 0)) z[k] = sqrtLambdas[k]! * prng.nextGaussian();
        } else {
          for (let k = 0; k < p; (k = (k + 1) | 0)) {
            if (signLeft === 0) {
              signBits = prng.nextU32();
              signLeft = 32;
            }
            z[k] = (signBits & 1) === 0 ? sqrtLambdas[k]! : -sqrtLambdas[k]!;
            signBits = signBits >>> 1;
            signLeft = (signLeft - 1) | 0;
          }
        }
        let dot = 0.0;
        for (let k = 0; k < p; (k = (k + 1) | 0)) dot += z[k]! * q[k]!;
        for (let k = 0; k < p; (k = (k + 1) | 0)) hq[k] = hq[k]! + dot * z[k]!;
      }
      let energy = 0.0;
      for (let k = 0; k < p; (k = (k + 1) | 0)) {
        const cur = q[k]!;
        const next = onePlusBeta * cur - beta * qPrev[k]! - eta * (hq[k]! * invB + sigmaXi * prng.nextGaussian());
        qPrev[k] = cur;
        q[k] = next;
        energy += next * next;
      }
      if (!Number.isFinite(energy) || energy > 1e200) return 1e3;
      if (t >= w1Start && t < w1End) energyEarly += energy;
      else if (t >= w2Start) energyLate += energy;
    }
  }
  const ratio = energyLate / Math.max(1e-300, energyEarly);
  return fdlibmLog(ratio);
}
