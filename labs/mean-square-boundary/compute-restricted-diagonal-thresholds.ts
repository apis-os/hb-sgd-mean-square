/**
 * @file labs/mean-square-boundary/compute-restricted-diagonal-thresholds.ts
 * @description The diagonal threshold of Theorem diag-threshold restricted to the top `r` modes (`computeRestrictedDiagonalThresholds`;
 * amendment 3, the Gauss–Newton `restrictedEtaD`): for each `r` of `ranks` (at most the number of modes), the deviations restricted
 * to the `r` modes of largest `|values|` (ties by index, the order of `evaluateDenseThresholdCriterion`), their Gram matrix and
 * `computeDiagonalThreshold` with the restricted curvature. Like `eta_D` itself it is an upper bound of the restricted `eta*_r`
 * (a Rayleigh compression), and the restriction to fewer modes moves it up further.
 * @cold
 */
import { assembleCurvatureNoiseGram } from './assemble-curvature-noise-gram.ts';
import type { buildEigenbasisSampleHessianDeviations } from './build-eigenbasis-sample-hessian-deviations.ts';
import { computeDiagonalThreshold } from './compute-diagonal-threshold.ts';
import { restrictEigenbasisDeviations } from './restrict-eigenbasis-deviations.ts';

export function computeRestrictedDiagonalThresholds(
  dev: ReturnType<typeof buildEigenbasisSampleHessianDeviations>,
  curvature: Float64Array,
  values: Float64Array,
  ranks: readonly number[],
  beta: number,
  batchB: number,
): Array<{ readonly r: number; readonly etaD: number }> {
  const n = dev.n;
  const order = Array.from({ length: n }, (_, k) => k).sort((x, y) => Math.abs(values[y]!) - Math.abs(values[x]!) || x - y);
  const out: Array<{ r: number; etaD: number }> = [];
  for (const r of ranks) {
    if (r > n) continue;
    const mask = new Uint8Array(n);
    for (let j = 0; j < r; (j = (j + 1) | 0)) mask[order[j]!] = 1;
    const sub = restrictEigenbasisDeviations(dev, mask);
    const gram = assembleCurvatureNoiseGram(sub.packed, sub.numSamples, sub.nPair, batchB);
    const curv = Float64Array.from(sub.modes, (k) => curvature[k]!);
    out.push({ r, etaD: computeDiagonalThreshold(gram, sub.n, sub.nPair, sub.pairIndex, curv, beta).etaD });
  }
  return out;
}
