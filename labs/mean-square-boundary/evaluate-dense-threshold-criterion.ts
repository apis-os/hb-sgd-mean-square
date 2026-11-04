/**
 * @file labs/mean-square-boundary/evaluate-dense-threshold-criterion.ts
 * @description One threshold criterion of Phase 5 at one checkpoint (`evaluateDenseThresholdCriterion`), for small `P`: given the
 * eigenbasis deviations (possibly restricted to kept modes), their Gram matrix, the curvature `l` of the iteration and the signed
 * eigenvalues `values` of the training objective on the same modes: `eta*` of Theorem G.6 from the dense operator
 * (`assembleThmG6OperatorMatrix`, Lanczos residual `relTol`), the critical-mode shares at each cut of `cuts`
 * (`classifyCriticalSecondMoment`, top mode `topIndex`), the bounds `eta_D`, `eta_tr(I)`, `eta_tr(L^{-1})`, `eta_tr(M*)` (null
 * when the Perron vector has a non-positive entry), and the restrictions to the top `r` modes by `|values|` for each `r` of
 * `ranks` (`r` at most the number of modes).
 * @cold
 */
import { assembleThmG6OperatorMatrix } from './assemble-thm-g6-operator-matrix.ts';
import type { buildEigenbasisSampleHessianDeviations } from './build-eigenbasis-sample-hessian-deviations.ts';
import { classifyCriticalSecondMoment } from './classify-critical-second-moment.ts';
import { computeDenseSymmetricTopEigenpair } from './compute-dense-symmetric-top-eigenpair.ts';
import { computeDiagonalThreshold } from './compute-diagonal-threshold.ts';
import { computeTraceWeightedLowerBound } from './compute-trace-weighted-lower-bound.ts';

export async function evaluateDenseThresholdCriterion(
  dev: ReturnType<typeof buildEigenbasisSampleHessianDeviations>,
  gram: Float64Array,
  curvature: Float64Array,
  values: Float64Array,
  beta: number,
  batchB: number,
  relTol: number,
  cuts: readonly number[],
  topIndex: number,
  ranks: readonly number[],
): Promise<{
  readonly etaStar: number;
  readonly lanczosIterations: number;
  readonly lanczosRelResidual: number;
  readonly criticalModes: ReadonlyArray<ReturnType<typeof classifyCriticalSecondMoment>>;
  readonly etaD: number;
  readonly etaTraceIdentity: number;
  readonly etaTraceInverse: number;
  readonly etaTracePerron: number | null;
  readonly restricted: ReadonlyArray<{ readonly r: number; readonly eta: number }>;
}> {
  const n = dev.n;
  const op = assembleThmG6OperatorMatrix(gram, n, dev.nPair, dev.pairIndex, dev.pairA, dev.pairB, curvature, beta);
  const top = await computeDenseSymmetricTopEigenpair(op.a, op.dim, relTol, 400);
  const etaStar = 1.0 / top.value;
  const criticalModes = cuts.map((cut) => classifyCriticalSecondMoment(top.ritz, op.pairs, dev.pairA, dev.pairB, values, curvature, etaStar, beta, cut, topIndex));
  const diag = computeDiagonalThreshold(gram, n, dev.nPair, dev.pairIndex, curvature, beta);
  const etaTraceIdentity = computeTraceWeightedLowerBound(dev.packed, dev.numSamples, n, dev.nPair, dev.pairIndex, curvature, new Float64Array(n).fill(1.0), beta, batchB);
  const etaTraceInverse = computeTraceWeightedLowerBound(dev.packed, dev.numSamples, n, dev.nPair, dev.pairIndex, curvature, Float64Array.from(curvature, (x) => 1.0 / x), beta, batchB);
  const mStar = Float64Array.from(diag.perron, (v, k) => v / Math.sqrt(curvature[k]!));
  let positive = true;
  for (let k = 0; k < n; (k = (k + 1) | 0)) if (!(mStar[k]! > 0.0)) positive = false;
  const etaTracePerron = positive ? computeTraceWeightedLowerBound(dev.packed, dev.numSamples, n, dev.nPair, dev.pairIndex, curvature, mStar, beta, batchB) : null;
  const order = Array.from({ length: n }, (_, k) => k).sort((x, y) => Math.abs(values[y]!) - Math.abs(values[x]!) || x - y);
  const restricted: Array<{ r: number; eta: number }> = [];
  for (const r of ranks) {
    if (r > n) continue;
    const mask = new Uint8Array(n);
    for (let j = 0; j < r; (j = (j + 1) | 0)) mask[order[j]!] = 1;
    const sub = assembleThmG6OperatorMatrix(gram, n, dev.nPair, dev.pairIndex, dev.pairA, dev.pairB, curvature, beta, mask);
    restricted.push({ r, eta: 1.0 / (await computeDenseSymmetricTopEigenpair(sub.a, sub.dim, relTol, 400)).value });
  }
  return { etaStar, lanczosIterations: top.iterations, lanczosRelResidual: top.relResidual, criticalModes, etaD: diag.etaD, etaTraceIdentity, etaTraceInverse, etaTracePerron, restricted };
}
