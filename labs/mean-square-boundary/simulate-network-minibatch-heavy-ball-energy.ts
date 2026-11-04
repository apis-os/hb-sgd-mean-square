/**
 * @file labs/mean-square-boundary/simulate-network-minibatch-heavy-ball-energy.ts
 * @description Log energy growth of heavy-ball SGD on a network at its checkpoint with the mean gradient subtracted (`simulateNetworkMinibatchHeavyBallEnergy`): `w_{t+1} = w_t + beta (w_t - w_{t-1}) - eta [ (g_B(w_t) - g_B(Theta_0)) + tau (g_B(Theta_0) - grad L(Theta_0)) ]`, real minibatches of `B` samples drawn with replacement, so the multiplicative curvature noise is the one of the network. Every stream starts at `Theta_0 + startNorm * startDirection` with zero velocity; the registered run uses `tau = 0` and a start in the range of `H`, because the network Hessian has rank at most `N * dOut << P` and an additive noise diffuses freely in its null space, where the energy grows linearly in `t` at every learning rate (seen in the smoke run with `tau = 1e-3`). Returns `log(energyLate / energyEarly)` over the second and fourth quarters of the run, summed over streams (the statistic of Phase 1), or `1e3` once the squared excursion exceeds `divergenceEnergy`.
 * @cold
 */
import { fdlibmLog } from '../../src/math/index.ts';
import { createXorshift128Plus } from './prng-xorshift128p.ts';
import { sumPrecomputedSampleGrads } from './sum-precomputed-sample-grads.ts';

export function simulateNetworkMinibatchHeavyBallEnergy(
  mlp: {
    readonly numParams: number;
    readonly numSamples: number;
    readonly checkpointTheta0: Float64Array;
    computeMinibatchGradient(theta: Float64Array, batchIndices: Int32Array, batchSize: number, outGrad: Float64Array): number;
  },
  sampleGrads0: Float64Array,
  fullGrad0: Float64Array,
  startDirection: Float64Array,
  startNorm: number,
  beta: number,
  eta: number,
  batchB: number,
  noiseScaleTau: number,
  streams: number,
  steps: number,
  divergenceEnergy: number,
  seedLo: number,
  seedHi: number,
): number {
  const p = mlp.numParams | 0;
  const theta0 = mlp.checkpointTheta0;
  const prng = createXorshift128Plus(seedLo | 0, seedHi | 0);
  const w = new Float64Array(p);
  const wPrev = new Float64Array(p);
  const gB = new Float64Array(p);
  const g0B = new Float64Array(p);
  const batchIdx = new Int32Array(batchB);
  const w1Start = (steps >> 2) | 0;
  const w1End = (steps >> 1) | 0;
  const w2Start = ((3 * steps) >> 2) | 0;
  let energyEarly = 0.0;
  let energyLate = 0.0;

  for (let s = 0; s < streams; (s = (s + 1) | 0)) {
    for (let i = 0; i < p; (i = (i + 1) | 0)) w[i] = theta0[i]! + startNorm * startDirection[i]!;
    wPrev.set(w);
    for (let t = 0; t < steps; (t = (t + 1) | 0)) {
      for (let b = 0; b < batchB; (b = (b + 1) | 0)) {
        batchIdx[b] = Math.floor(prng.nextFloat01() * mlp.numSamples);
      }
      mlp.computeMinibatchGradient(w, batchIdx, batchB, gB);
      sumPrecomputedSampleGrads(sampleGrads0, batchIdx, batchB, p, g0B);
      let energy = 0.0;
      for (let i = 0; i < p; (i = (i + 1) | 0)) {
        const cur = w[i]!;
        const restoring = gB[i]! - g0B[i]!;
        const noise = (g0B[i]! - fullGrad0[i]!) * noiseScaleTau;
        const next = cur + beta * (cur - wPrev[i]!) - eta * (restoring + noise);
        wPrev[i] = cur;
        w[i] = next;
        const d = next - theta0[i]!;
        energy += d * d;
      }
      if (!Number.isFinite(energy) || energy > divergenceEnergy) return 1e3;
      if (t >= w1Start && t < w1End) energyEarly += energy;
      else if (t >= w2Start) energyLate += energy;
    }
  }
  return fdlibmLog(energyLate / Math.max(1e-300, energyEarly));
}
