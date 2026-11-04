/**
 * @file labs/mean-square-boundary/simulate-tethered-heavy-ball-energy.ts
 * @description Log energy growth of tethered heavy-ball SGD with real minibatches at a network checkpoint
 * (`simulateTetheredHeavyBallEnergy`), the statistic of Phases 1 and 2: `log(energyLate / energyEarly)` over the second and
 * fourth quarters of the run, summed over streams, or `1e3` once the squared excursion exceeds `divergenceEnergy`. The
 * deviation `x_t = w_t - Theta_0` follows `x_{t+1} = x_t + beta (x_t - x_{t-1}) - eta (r_B(x_t) + mu x_t)` with `B` samples drawn
 * with replacement; `mode = 0` uses the linearised restoring force `r_B(x) = (1/B) sum_b H_{i_b} x` (the linear system of
 * Theorem G.6), `mode = 1` the network's `r_B(x) = g_B(Theta_0 + x) - g_B(Theta_0)` (mean gradient removed, as in Phase 2).
 * Each stream starts at `startNorm` times a seeded Gaussian unit vector with zero velocity; the tether gives every direction
 * a restoring force, so no start in the range of `H` is needed. Both modes draw the same minibatches and starts from the
 * same seeds.
 * @cold
 */
import { fdlibmLog } from '../../src/math/index.ts';
import type { Phase4SampleHessianProvider } from './phase4-types.ts';
import { createXorshift128Plus } from './prng-xorshift128p.ts';

export function simulateTetheredHeavyBallEnergy(
  provider: Phase4SampleHessianProvider,
  mode: 0 | 1,
  mu: number,
  beta: number,
  eta: number,
  batchB: number,
  streams: number,
  steps: number,
  startNorm: number,
  divergenceEnergy: number,
  seedLo: number,
  seedHi: number,
): number {
  const p = provider.numParams | 0;
  const theta0 = provider.checkpointTheta0;
  const prng = createXorshift128Plus(seedLo | 0, seedHi | 0);
  const x = new Float64Array(p);
  const xPrev = new Float64Array(p);
  const w = new Float64Array(p);
  const r = new Float64Array(p);
  const tmp = new Float64Array(p);
  const g0 = new Float64Array(p);
  const batchIdx = new Int32Array(batchB);
  const invB = 1.0 / batchB;
  const w1Start = (steps >> 2) | 0;
  const w1End = (steps >> 1) | 0;
  const w2Start = ((3 * steps) >> 2) | 0;
  let energyEarly = 0.0;
  let energyLate = 0.0;
  const p4 = p & ~3;
  for (let s = 0; s < streams; (s = (s + 1) | 0)) {
    let normSq = 0.0;
    for (let i = 0; i < p; (i = (i + 1) | 0)) {
      x[i] = prng.nextGaussian();
      normSq += x[i]! * x[i]!;
    }
    const scale = startNorm / Math.sqrt(normSq);
    for (let i = 0; i < p; (i = (i + 1) | 0)) {
      x[i] = x[i]! * scale;
      xPrev[i] = x[i]!;
    }
    for (let t = 0; t < steps; (t = (t + 1) | 0)) {
      for (let b = 0; b < batchB; (b = (b + 1) | 0)) batchIdx[b] = Math.floor(prng.nextFloat01() * provider.numSamples);
      if (mode === 0) {
        r.fill(0.0);
        for (let b = 0; b < batchB; (b = (b + 1) | 0)) {
          provider.applySampleHessian(batchIdx[b]!, x, tmp);
          let i = 0;
          for (; i < p4; (i = (i + 4) | 0)) {
            r[i] = r[i]! + invB * tmp[i]!;
            r[(i + 1) | 0] = r[(i + 1) | 0]! + invB * tmp[(i + 1) | 0]!;
            r[(i + 2) | 0] = r[(i + 2) | 0]! + invB * tmp[(i + 2) | 0]!;
            r[(i + 3) | 0] = r[(i + 3) | 0]! + invB * tmp[(i + 3) | 0]!;
          }
          for (; i < p; (i = (i + 1) | 0)) r[i] = r[i]! + invB * tmp[i]!;
        }
      } else {
        for (let i = 0; i < p; (i = (i + 1) | 0)) w[i] = theta0[i]! + x[i]!;
        provider.computeMinibatchGradient(w, batchIdx, batchB, r);
        provider.computeMinibatchGradient(theta0, batchIdx, batchB, g0);
        for (let i = 0; i < p; (i = (i + 1) | 0)) r[i] = r[i]! - g0[i]!;
      }
      let energy = 0.0;
      for (let i = 0; i < p; (i = (i + 1) | 0)) {
        const cur = x[i]!;
        const next = cur + beta * (cur - xPrev[i]!) - eta * (r[i]! + mu * cur);
        xPrev[i] = cur;
        x[i] = next;
        energy += next * next;
      }
      if (!Number.isFinite(energy) || energy > divergenceEnergy) return 1e3;
      if (t >= w1Start && t < w1End) energyEarly += energy;
      else if (t >= w2Start) energyLate += energy;
    }
  }
  return fdlibmLog(energyLate / Math.max(1e-300, energyEarly));
}
