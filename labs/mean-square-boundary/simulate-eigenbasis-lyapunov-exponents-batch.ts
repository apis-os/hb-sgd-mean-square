/**
 * @file labs/mean-square-boundary/simulate-eigenbasis-lyapunov-exponents-batch.ts
 * @description Top Lyapunov exponents of the linearised heavy-ball iteration for several step sizes at once
 * (`simulateEigenbasisLyapunovExponentsBatch`), in lockstep: every stream draws its minibatches once and every step size uses
 * the same draws, as separate calls of `simulateEigenbasisLyapunovExponentReference` with the same seeds do (common random
 * numbers). For each `etas[g]` the per-stream exponents, mean and standard error are bit-identical to that reference called with
 * `etas[g]` alone: the starts and draws are the same PRNG calls in the same order, the scaled minibatch deviation is the same
 * value (`sumEigenbasisLyapunovMinibatch`), and each state goes through the same operations (`stepEigenbasisLyapunovG1`,
 * `stepEigenbasisLyapunovG2`, `stepEigenbasisLyapunovG3` for `G <= 3` on the row-major pair layout, `stepEigenbasisLyapunovGeneral`
 * otherwise). The cost of a step is one minibatch sum plus `G` products, instead of `G` sums and `G` products.
 * @cold
 */
import { isRowMajorPairLayout } from './is-row-major-pair-layout.ts';
import { createXorshift128Plus } from './prng-xorshift128p.ts';
import { stepEigenbasisLyapunovG1 } from './step-eigenbasis-lyapunov-g1.gen.ts';
import { stepEigenbasisLyapunovG2 } from './step-eigenbasis-lyapunov-g2.gen.ts';
import { stepEigenbasisLyapunovG3 } from './step-eigenbasis-lyapunov-g3.gen.ts';
import { stepEigenbasisLyapunovGeneral } from './step-eigenbasis-lyapunov-general.ts';
import { sumEigenbasisLyapunovMinibatch } from './sum-eigenbasis-lyapunov-minibatch.ts';

export function simulateEigenbasisLyapunovExponentsBatch(
  packed: Float64Array,
  numSamples: number,
  n: number,
  nPair: number,
  pairA: Int32Array,
  pairB: Int32Array,
  curvature: Float64Array,
  beta: number,
  etas: ArrayLike<number>,
  batchB: number,
  streams: number,
  steps: number,
  burnIn: number,
  seedLo: number,
  seedHi: number,
): Array<{ readonly mean: number; readonly standardError: number; readonly perStream: Float64Array }> {
  const gCount = etas.length | 0;
  const eta = Float64Array.from(etas);
  const rowMajor = isRowMajorPairLayout(pairA, pairB, n, nPair);
  const kernel = rowMajor && gCount >= 1 && gCount <= 3 ? gCount : 0;
  const x = new Float64Array(Math.imul(gCount, n));
  const xPrev = new Float64Array(Math.imul(gCount, n));
  const r = new Float64Array(Math.imul(gCount, n));
  const sum = new Float64Array(nPair);
  const rows = new Int32Array(batchB > 0 ? Math.ceil(batchB) : 0);
  const logs = new Float64Array(gCount);
  const invB = 1.0 / batchB;
  const perStream = Array.from({ length: gCount }, () => new Float64Array(streams));
  for (let s = 0; s < streams; (s = (s + 1) | 0)) {
    const prng = createXorshift128Plus((seedLo + Math.imul(s, 0x9e3779b1)) | 0, seedHi | 0);
    let normSq = 0.0;
    const x0 = new Float64Array(n);
    for (let k = 0; k < n; (k = (k + 1) | 0)) {
      x0[k] = prng.nextGaussian();
      normSq += x0[k]! * x0[k]!;
    }
    const inv0 = 1.0 / Math.sqrt(normSq);
    for (let k = 0; k < n; (k = (k + 1) | 0)) x0[k] = x0[k]! * inv0;
    for (let g = 0; g < gCount; (g = (g + 1) | 0)) {
      x.set(x0, Math.imul(g, n));
      xPrev.set(x0, Math.imul(g, n));
      logs[g] = 0.0;
    }
    for (let t = 0; t < steps; (t = (t + 1) | 0)) {
      sumEigenbasisLyapunovMinibatch(prng, packed, numSamples, nPair, batchB, invB, rows, sum);
      const record = t >= burnIn;
      switch (kernel) {
        case 1: stepEigenbasisLyapunovG1(sum, curvature, x, xPrev, r, n, beta, eta, logs, record); break;
        case 2: stepEigenbasisLyapunovG2(sum, curvature, x, xPrev, r, n, beta, eta, logs, record); break;
        case 3: stepEigenbasisLyapunovG3(sum, curvature, x, xPrev, r, n, beta, eta, logs, record); break;
        default: stepEigenbasisLyapunovGeneral(sum, curvature, pairA, pairB, rowMajor, x, xPrev, r, n, nPair, beta, eta, logs, record);
      }
    }
    for (let g = 0; g < gCount; (g = (g + 1) | 0)) perStream[g]![s] = logs[g]! / (steps - burnIn);
  }
  return perStream.map((ps) => {
    let mean = 0.0;
    for (let s = 0; s < streams; (s = (s + 1) | 0)) mean += ps[s]!;
    mean /= streams;
    let v = 0.0;
    for (let s = 0; s < streams; (s = (s + 1) | 0)) v += (ps[s]! - mean) * (ps[s]! - mean);
    const standardError = streams > 1 ? Math.sqrt(v / (streams - 1) / streams) : Number.NaN;
    return { mean, standardError, perStream: ps };
  });
}
