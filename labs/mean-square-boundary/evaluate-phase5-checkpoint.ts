/**
 * @file labs/mean-square-boundary/evaluate-phase5-checkpoint.ts
 * @description Everything Phase 5 measures at one training checkpoint `theta_t` of the `P = 33` network
 * (`evaluatePhase5Checkpoint`; registration items (a)-(f) and amendments 1-2). Dense per-sample Hessians of the loss at
 * `theta_t` (central differences, `createPhase4SampleHessianProvider`), the training objective `H + mu_train I`, its eigenbasis
 * deviations and their Gram matrix (`B = 16`). (a) spectrum, negative count, the flag `|lambda_min| >= 0.5 lambda_1`, and the
 * decoupling ratio of every eigenvector with `|lambda| < 1e-8 lambda_1` (removed from the `|H|` criterion below `1e-6`).
 * (b) the `|H|` criterion (curvature `|lambda|` on the kept modes), (c) the uniform tether `mu_e = g + max(0, -lambda_min)` for
 * `g` in `{1/32, 1/128}`, each with critical modes at the cuts `1e-2 max|lambda|` and `0.05 lambda_1` (top mode = largest
 * `|lambda|`), `eta_D` and the trace-weighted bounds; restrictions to the top `r` in `{1, 4, 8, 16}` and the Gauss–Newton
 * threshold (per-sample `grad f_i grad f_i^T` plus `mu_train I`, same null-mode rule) for `|H|`. (d) the one-dimensional
 * ceiling of `|H|` (the `r = 1` restriction). (e) batch sharpness over 256 minibatches drawn from a probe stream re-seeded at
 * every checkpoint (the same minibatches at every checkpoint of a run), as the expectation of the ratio and the ratio of
 * expectations, times `eta / (2 (1 + beta))`, and `(tr H / B) eta / (2 (1 - beta))`, from per-sample gradients computed once;
 * the `p6` formula `(tr H_GN^2 + tr V_GN(I)) / tr H_GN`. (f) `lambda_1 eta / (2 (1 + beta))`. The increment-stream block `e13`
 * of the companion lab (built from the last 200 steps, `_deltas`) is not computed here and is `null`. Amendment 3 (reported, no prediction): the full Gauss–Newton criterion
 * (bounds, critical modes, restrictions to `r` in `{1, 4, 8, 16}` and `computeRestrictedDiagonalThresholds`), the criteria of
 * `evaluatePhase5ReportedCriteria`, the data loss without the regulariser, `computeHiddenUnitSaturation`, and the `p6` quantities of
 * `computeBatchSharpnessDecomposition`. Returns
 * the record and the deviations for the final-point analysis.
 * @cold
 */
import { assembleCurvatureNoiseGram } from './assemble-curvature-noise-gram.ts';
import { buildEigenbasisSampleHessianDeviations } from './build-eigenbasis-sample-hessian-deviations.ts';
import { computeBatchSharpnessDecomposition } from './compute-batch-sharpness-decomposition.ts';
import { computeHiddenUnitSaturation } from './compute-hidden-unit-saturation.ts';
import { computeRestrictedDiagonalThresholds } from './compute-restricted-diagonal-thresholds.ts';
import { createPhase4SampleHessianProvider } from './create-phase4-sample-hessian-provider.ts';
import { evaluateDenseThresholdCriterion } from './evaluate-dense-threshold-criterion.ts';
import { evaluatePhase5ReportedCriteria } from './evaluate-phase5-reported-criteria.ts';
import type { createTwoLayerMlpDatasetAndOps } from './mlp-network.ts';
import { createXorshift128Plus } from './prng-xorshift128p.ts';
import { restrictEigenbasisDeviations } from './restrict-eigenbasis-deviations.ts';

type Mlp = ReturnType<typeof createTwoLayerMlpDatasetAndOps>;
type Dev = ReturnType<typeof buildEigenbasisSampleHessianDeviations>;

export async function evaluatePhase5Checkpoint(
  mlp: Mlp,
  theta: Float64Array,
  step: number,
  muTrain: number,
  eta: number,
  beta: number,
  seed: number,
  _deltas: Float64Array | null,
): Promise<{
  readonly record: Record<string, unknown>;
  readonly full: { readonly dev: Dev; readonly gram: Float64Array; readonly values: Float64Array };
  readonly abs: { readonly dev: Dev; readonly gram: Float64Array; readonly curvature: Float64Array; readonly values: Float64Array };
}> {
  const batchB = 16;
  const relTol = 1e-10;
  const p = mlp.numParams;
  const nS = mlp.numSamples;
  const th = Float64Array.from(theta);
  const provider = createPhase4SampleHessianProvider({ kind: 'mlp33', theta: th });
  const dev = buildEigenbasisSampleHessianDeviations(provider);
  const n = dev.n;
  const values = Float64Array.from(dev.values, (v) => v + muTrain);
  const lambda1 = values[(n - 1) | 0]!;
  const lambdaMin = values[0]!;
  let negative = 0;
  let maxAbs = 0.0;
  let topIndex = 0;
  for (let k = 0; k < n; (k = (k + 1) | 0)) {
    if (values[k]! < 0.0) negative = (negative + 1) | 0;
    if (Math.abs(values[k]!) >= maxAbs) { maxAbs = Math.abs(values[k]!); topIndex = k; }
  }
  // (a) Null modes: decoupling ratio max_i |(H_i + mu I) v| / max_i |(H_i + mu I) u_top|.
  const vec = new Float64Array(p);
  const out = new Float64Array(p);
  const sampleNorm = (k: number): number => {
    for (let a = 0; a < p; (a = (a + 1) | 0)) vec[a] = dev.qt[(Math.imul(k, n) + a) | 0]!;
    let best = 0.0;
    for (let i = 0; i < nS; (i = (i + 1) | 0)) {
      provider.applySampleHessian(i, vec, out);
      let s = 0.0;
      for (let a = 0; a < p; (a = (a + 1) | 0)) { const x = out[a]! + muTrain * vec[a]!; s += x * x; }
      if (s > best) best = s;
    }
    return Math.sqrt(best);
  };
  const keep = new Uint8Array(n).fill(1);
  const nullModes: Array<{ k: number; eigenvalue: number; ratio: number; removed: boolean }> = [];
  const topNorm = sampleNorm(topIndex);
  for (let k = 0; k < n; (k = (k + 1) | 0)) {
    if (!(Math.abs(values[k]!) < 1e-8 * lambda1)) continue;
    const ratio = sampleNorm(k) / topNorm;
    const removed = ratio < 1e-6;
    if (removed) keep[k] = 0;
    nullModes.push({ k, eigenvalue: values[k]!, ratio, removed });
  }
  const gram = assembleCurvatureNoiseGram(dev.packed, nS, dev.nPair, batchB);
  const cuts = [1e-2 * maxAbs, 0.05 * lambda1];
  const anyRemoved = nullModes.some((m) => m.removed);
  const devAbs = anyRemoved ? restrictEigenbasisDeviations(dev, keep) : dev;
  const gramAbs = anyRemoved ? assembleCurvatureNoiseGram(devAbs.packed, nS, devAbs.nPair, batchB) : gram;
  const valuesAbs = anyRemoved ? Float64Array.from((devAbs as Dev & { modes: Int32Array }).modes, (k) => values[k]!) : values;
  const curvAbs = Float64Array.from(valuesAbs, (v) => Math.abs(v));
  let topAbs = 0;
  for (let k = 0; k < devAbs.n; (k = (k + 1) | 0)) if (curvAbs[k]! >= curvAbs[topAbs]!) topAbs = k;
  const absH = await evaluateDenseThresholdCriterion(devAbs, gramAbs, curvAbs, valuesAbs, beta, batchB, relTol, cuts, topAbs, [1, 4, 8, 16]);
  const tethered: Record<string, unknown> = {};
  for (const g of [1 / 32, 1 / 128]) {
    const muE = g + (lambdaMin < 0.0 ? -lambdaMin : 0.0);
    const curv = Float64Array.from(values, (v) => v + muE);
    const res = await evaluateDenseThresholdCriterion(dev, gram, curv, values, beta, batchB, relTol, cuts, topIndex, []);
    tethered[String(g)] = { muE, muOverLambda1: muE / lambda1, ...res };
  }
  // Gauss-Newton threshold for |H| conventions (PSD; the null-mode rule on its own spectrum).
  const providerGn = createPhase4SampleHessianProvider({ kind: 'mlp33', theta: th, gaussNewton: true });
  const devGn = buildEigenbasisSampleHessianDeviations(providerGn);
  const valuesGn = Float64Array.from(devGn.values, (v) => v + muTrain);
  const keepGn = new Uint8Array(n).fill(1);
  const lGn1 = valuesGn[(n - 1) | 0]!;
  let removedGn = 0;
  for (let k = 0; k < n; (k = (k + 1) | 0)) if (Math.abs(valuesGn[k]!) < 1e-8 * lGn1) { keepGn[k] = 0; removedGn = (removedGn + 1) | 0; }
  const devGnK = removedGn > 0 ? restrictEigenbasisDeviations(devGn, keepGn) : devGn;
  const valuesGnK = removedGn > 0 ? Float64Array.from((devGnK as Dev & { modes: Int32Array }).modes, (k) => valuesGn[k]!) : valuesGn;
  const gramGn = assembleCurvatureNoiseGram(devGnK.packed, nS, devGnK.nPair, batchB);
  const curvGnK = Float64Array.from(valuesGnK, (v) => Math.abs(v));
  const gn = await evaluateDenseThresholdCriterion(devGnK, gramGn, curvGnK, valuesGnK, beta, batchB, relTol, cuts, devGnK.n - 1, [1, 4, 8, 16]);
  // Amendment 3: the Gauss-Newton diagonal bound on the top r modes, and the added criteria (reported, no prediction).
  const gnRestrictedEtaD = computeRestrictedDiagonalThresholds(devGnK, curvGnK, valuesGnK, [1, 4, 8, 16], beta, batchB);
  const reported = await evaluatePhase5ReportedCriteria(provider, dev, values, devGnK, valuesGnK, eta, beta, batchB, relTol, cuts);
  // (e) Batch sharpness from per-sample gradients computed once.
  const grads = new Float64Array(Math.imul(nS, p));
  const outputGrads = new Float64Array(Math.imul(nS, p));
  const one = new Int32Array(1);
  const gi = new Float64Array(p);
  for (let i = 0; i < nS; (i = (i + 1) | 0)) {
    one[0] = i;
    mlp.computeMinibatchGradient(th, one, 1, gi);
    grads.set(gi, Math.imul(i, p));
    mlp.computeSampleOutputGradient(th, i, gi);
    outputGrads.set(gi, Math.imul(i, p));
  }
  const probe = createXorshift128Plus((0x14b00001 + Math.imul(seed, 104729)) | 0, 0x85ebca6b);
  const idx = new Int32Array(batchB);
  const gb = new Float64Array(p);
  const hg = new Float64Array(p);
  let sumRatio = 0.0;
  let sumNum = 0.0;
  let sumDen = 0.0;
  const draws = 256;
  for (let rIdx = 0; rIdx < draws; (rIdx = (rIdx + 1) | 0)) {
    for (let b = 0; b < batchB; (b = (b + 1) | 0)) idx[b] = Math.floor(probe.nextFloat01() * nS);
    gb.fill(0.0);
    for (let b = 0; b < batchB; (b = (b + 1) | 0)) {
      const base = Math.imul(idx[b]!, p);
      for (let a = 0; a < p; (a = (a + 1) | 0)) gb[a] = gb[a]! + grads[(base + a) | 0]! / batchB;
    }
    for (let a = 0; a < p; (a = (a + 1) | 0)) gb[a] = gb[a]! + muTrain * th[a]!;
    hg.fill(0.0);
    for (let b = 0; b < batchB; (b = (b + 1) | 0)) {
      provider.applySampleHessian(idx[b]!, gb, out);
      for (let a = 0; a < p; (a = (a + 1) | 0)) hg[a] = hg[a]! + out[a]! / batchB;
    }
    let num = 0.0;
    let den = 0.0;
    for (let a = 0; a < p; (a = (a + 1) | 0)) {
      num += gb[a]! * (hg[a]! + muTrain * gb[a]!);
      den += gb[a]! * gb[a]!;
    }
    sumRatio += num / den;
    sumNum += num;
    sumDen += den;
  }
  let traceH = 0.0;
  for (let k = 0; k < n; (k = (k + 1) | 0)) traceH += values[k]!;
  let trGn = 0.0;
  let trGn2 = 0.0;
  for (let k = 0; k < n; (k = (k + 1) | 0)) { trGn += devGn.values[k]!; trGn2 += devGn.values[k]! * devGn.values[k]!; }
  let trVGn = 0.0;
  for (let i = 0; i < nS; (i = (i + 1) | 0)) {
    const base = Math.imul(i, devGn.nPair);
    for (let q = 0; q < devGn.nPair; (q = (q + 1) | 0)) {
      const x = devGn.packed[(base + q) | 0]!;
      trVGn += (devGn.pairA[q] === devGn.pairB[q] ? 1.0 : 2.0) * x * x;
    }
  }
  trVGn /= batchB * nS;
  const p6Formula = (trGn2 + trVGn) / trGn;
  const p6 = computeBatchSharpnessDecomposition(grads, outputGrads, th, nS, p, muTrain, batchB, (i, v, o) => provider.applySampleHessian(i, v, o), (i, v, o) => providerGn.applySampleHessian(i, v, o), p6Formula, sumNum / sumDen);
  // Proposition E13(1) along u_1 (largest lambda) from the last 200 steps, and the increment-stream direction of amendment 3.
  // The increment-stream block (e13) belongs to the estimator of the companion manuscript and is not computed here.
  const e13 = null;
  let thetaSq = 0.0;
  for (let a = 0; a < p; (a = (a + 1) | 0)) thetaSq += th[a]! * th[a]!;
  const record: Record<string, unknown> = {
    step,
    loss: mlp.evalFullLoss(th) + 0.5 * muTrain * thetaSq,
    dataLoss: mlp.evalFullLoss(th),
    hiddenUnits: computeHiddenUnitSaturation(th, mlp.xData, nS, mlp.dIn, mlp.dHidden),
    lambda1,
    lambdaMin,
    negativeEigenvalues: negative,
    flaggedNegativeVsTop: Math.abs(lambdaMin) >= 0.5 * lambda1,
    nullModes,
    absH: { ...absH, etaOneDim: absH.restricted.find((r) => r.r === 1)?.eta ?? null, ratioToEta: absH.etaStar / eta, keptModes: devAbs.n },
    tethered,
    gaussNewton: { ...gn, removedModes: removedGn, ratioToEta: gn.etaStar / eta, etaDRatioToEta: gn.etaD / eta, restrictedEtaD: gnRestrictedEtaD },
    ...reported,
    batchSharpness: {
      expectationOfRatio: sumRatio / draws,
      ratioOfExpectations: sumNum / sumDen,
      expectationOfRatioScaled: (sumRatio / draws) * eta / (2.0 * (1.0 + beta)),
      ratioOfExpectationsScaled: (sumNum / sumDen) * eta / (2.0 * (1.0 + beta)),
      traceOverBScaled: (traceH / batchB) * eta / (2.0 * (1.0 - beta)),
    },
    lambda1Scaled: (lambda1 * eta) / (2.0 * (1.0 + beta)),
    p6Formula,
    ...p6,
    e13,
  };
  return { record, full: { dev, gram, values }, abs: { dev: devAbs, gram: gramAbs, curvature: curvAbs, values: valuesAbs } };
}
