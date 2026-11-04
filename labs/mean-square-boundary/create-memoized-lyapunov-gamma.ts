/**
 * @file labs/mean-square-boundary/create-memoized-lyapunov-gamma.ts
 * @description The top Lyapunov exponent `gamma(eta)` of one linearised iteration as a memoised, batched function
 * (`createMemoizedLyapunovGamma`), for `analyzePhase5FinalPoint`: `batch(etas)` runs one `simulateEigenbasisLyapunovExponentsBatch`
 * over the step sizes not yet evaluated (first-appearance order) and returns, for every requested `eta`, a fresh
 * `{ mean, standardError, perStream }` with `perStream` as a plain array; `one(eta)` is `batch([eta])[0]`. With common random
 * numbers the exponent at a step size does not depend on the other step sizes of the call, so every value equals a separate
 * `simulateEigenbasisLyapunovExponent` call with the same seeds, and a repeated `eta` is not simulated again.
 * @cold
 */
import type { buildEigenbasisSampleHessianDeviations } from './build-eigenbasis-sample-hessian-deviations.ts';
import { simulateEigenbasisLyapunovExponentsBatch } from './simulate-eigenbasis-lyapunov-exponents-batch.ts';

type Dev = ReturnType<typeof buildEigenbasisSampleHessianDeviations>;
type Gamma = { readonly mean: number; readonly standardError: number; readonly perStream: number[] };

export function createMemoizedLyapunovGamma(
  dev: Dev,
  curvature: Float64Array,
  beta: number,
  batchB: number,
  streams: number,
  steps: number,
  burnIn: number,
  seedLo: number,
  seedHi: number,
): { readonly batch: (etas: readonly number[]) => Gamma[]; readonly one: (eta: number) => Gamma } {
  const memo = new Map<number, { mean: number; standardError: number; perStream: Float64Array }>();
  const batch = (etas: readonly number[]): Gamma[] => {
    const missing: number[] = [];
    for (const e of etas) if (!memo.has(e) && !missing.includes(e)) missing.push(e);
    if (missing.length > 0) {
      const res = simulateEigenbasisLyapunovExponentsBatch(dev.packed, dev.numSamples, dev.n, dev.nPair, dev.pairA, dev.pairB, curvature, beta, missing, batchB, streams, steps, burnIn, seedLo, seedHi);
      for (let i = 0; i < missing.length; (i = (i + 1) | 0)) memo.set(missing[i]!, res[i]!);
    }
    return etas.map((e) => {
      const g = memo.get(e)!;
      return { mean: g.mean, standardError: g.standardError, perStream: Array.from(g.perStream) };
    });
  };
  return { batch, one: (eta: number) => batch([eta])[0]! };
}
