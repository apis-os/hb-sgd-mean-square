/**
 * @file labs/mean-square-boundary/evaluate-phase2-trace-ceiling-cells.ts
 * @description Phase 2 (`evaluatePhase2TraceCeilingCells`): on the `P = 3,610` and `P = 12,826` network checkpoints of `networks.json`, the exact Hessian trace and noise-rank ratio `kappa_beta tr H / (B lambda_1)` at `beta = 0.9`, `B = 16`; the random-feature secular bracket of Theorem G calibrated on `(lambda_1, lambda_2, tr H, nu_11)`; the destructive minibatch heavy-ball ceiling in `R^P` (bisection on sample paths); a finite-difference HVP power-iteration baseline that needs no second-order code; and errors against both ground-truth thresholds. Predictions `q1`–`q4` are evaluated from the rows; their texts are written before the run.
 * @cold
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { computeExactHessianTrace } from './compute-exact-hessian-trace.ts';
import { computeFloat64Quantile } from './compute-float64-quantile.ts';
import { computeSecularCeilingBracketFromTrace } from './compute-secular-ceiling-bracket-from-trace.ts';
import { computeTopEigenpairPowerIteration } from './compute-top-eigenpair-power-iteration.ts';
import { measureNetworkMinibatchHeavyBallCeiling } from './measure-network-minibatch-heavy-ball-ceiling.ts';
import { simulateNetworkMinibatchHeavyBallEnergy } from './simulate-network-minibatch-heavy-ball-energy.ts';
import type { Phase2GrowthGridRow, Phase2NetworkCell, Phase2Report } from './phase2-types.ts';
import { precomputeStaticSampleGrads } from './precompute-static-sample-grads.ts';
import { createXorshift128Plus } from './prng-xorshift128p.ts';
import { createScaleMlpDatasetAndOps } from './scale-mlp-network.ts';

const PREDICTION_TEXTS: Readonly<Record<string, string>> = {
  q1: 'Exact trace: tr(H) / lambda_1 >= 4 on both networks, so the noise-rank ratio kappa_beta tr(H) / (B lambda_1) at beta = 0.9, B = 16 is >= 4.75 and the bulk carries most of the curvature noise that Theorem G feeds into u_1.',
  q2: 'The destructive ceiling of heavy-ball SGD with real minibatches in R^P (beta = 0.9, B = 16, homogeneous run from a start in the range of H) lies below the one-dimensional EoSS ceiling of Theorem E(3) along u_1 by a factor >= 1.5 on both networks.',
  q3: 'The median destructive ceiling lies inside the trace bracket of the random-feature model (Theorem G with kappa_feat calibrated from nu_11, lambda_2 and tr H), widened by 10 % on each side plus the bisection half-width, on both networks.',
  q4: 'A power iteration with central-difference Hessian-vector products from two full gradients each (15 products, 1,920 sample-gradient equivalents, no second-order code) estimates lambda_1 to a relative error below 1e-6 on both networks, so its error against the quadratic threshold is below 0.01 % and against the destructive threshold equals the quadratic-destructive gap within 0.01 points.',
};

function buildNetworkCell(
  cellId: string,
  dIn: number,
  seedLo: number,
  seedHi: number,
  raw: {
    readonly lambda1AnalyticalHvp: number;
    readonly lambda2AnalyticalHvp: number;
    readonly etaCritQuadraticHb09: number;
    readonly etaCritDestructiveHb09: number;
    readonly minibatchHessianStdB16: number;
  },
  numSeeds: number,
  streams: number,
  steps: number,
  bisectionIterations: number,
  logGrowthThreshold: number,
  noiseScaleTau: number,
): Phase2NetworkCell {
  const beta = 0.9;
  const batchB = 16;
  const mlp = createScaleMlpDatasetAndOps(dIn, 48, 10, 64, seedLo, seedHi);
  const p = mlp.numParams;
  const theta0 = mlp.checkpointTheta0;
  const lambda1 = raw.lambda1AnalyticalHvp;
  const lambda2 = raw.lambda2AnalyticalHvp;
  const nuB = raw.minibatchHessianStdB16;
  const applyAnalytic = (v: Float64Array, out: Float64Array): void => mlp.computeAnalyticalHvp(theta0, v, out);

  const tTrace = Date.now();
  const traceH = computeExactHessianTrace(applyAnalytic, p);
  const traceSeconds = (Date.now() - tTrace) / 1000;

  const analytic40 = computeTopEigenpairPowerIteration(applyAnalytic, p, 40, 0x10a00001 ^ dIn, 0x9e3779b9);
  const applyShifted = (v: Float64Array, out: Float64Array): void => {
    mlp.computeAnalyticalHvp(theta0, v, out);
    for (let i = 0; i < p; (i = (i + 1) | 0)) out[i] = lambda1 * v[i]! - out[i]!;
  };
  const shifted100 = computeTopEigenpairPowerIteration(applyShifted, p, 100, 0x10a00002 ^ dIn, 0x85ebca6b);
  const lambdaMin = lambda1 - shifted100.lambda;

  // Finite-difference HVP from two full gradients per product (the eps of hessian-ground-truth.ts).
  const eps = 1e-5;
  const thetaPlus = new Float64Array(p);
  const thetaMinus = new Float64Array(p);
  const gradPlus = new Float64Array(p);
  const gradMinus = new Float64Array(p);
  const applyFd = (v: Float64Array, out: Float64Array): void => {
    for (let i = 0; i < p; (i = (i + 1) | 0)) {
      thetaPlus[i] = theta0[i]! + eps * v[i]!;
      thetaMinus[i] = theta0[i]! - eps * v[i]!;
    }
    mlp.computeFullGradient(thetaPlus, gradPlus);
    mlp.computeFullGradient(thetaMinus, gradMinus);
    const inv2Eps = 0.5 / eps;
    for (let i = 0; i < p; (i = (i + 1) | 0)) out[i] = (gradPlus[i]! - gradMinus[i]!) * inv2Eps;
  };
  const fdIterations = 15;
  const fd15 = computeTopEigenpairPowerIteration(applyFd, p, fdIterations, 0x10a00003 ^ dIn, 0x27182818);
  let cosFd = 0.0;
  for (let i = 0; i < p; (i = (i + 1) | 0)) cosFd += fd15.vec[i]! * analytic40.vec[i]!;
  const fdEtaHat = (2.0 * (1.0 + beta)) / fd15.lambda;
  const fdErrQuad = (Math.abs(fdEtaHat - raw.etaCritQuadraticHb09) / raw.etaCritQuadraticHb09) * 100.0;
  const fdErrDestr = (Math.abs(fdEtaHat - raw.etaCritDestructiveHb09) / raw.etaCritDestructiveHb09) * 100.0;

  const kappaFeat = (batchB * nuB * nuB) / (lambda1 * lambda1) - 2.0;
  const bracket = computeSecularCeilingBracketFromTrace(lambda1, lambda2, traceH, beta, batchB, kappaFeat);

  const fullGrad0 = new Float64Array(p);
  mlp.computeFullGradient(theta0, fullGrad0);
  const sampleGrads0 = precomputeStaticSampleGrads(mlp);
  // Start in the range of H: H applied to a seeded Gaussian vector, normalised. A Gaussian start itself puts
  // (P - rank H) / P of its energy in the null space of H, where heavy-ball keeps it constant.
  const startPrng = createXorshift128Plus(0x10a00004 ^ dIn, 0x7f4a7c15);
  const gaussStart = new Float64Array(p);
  for (let i = 0; i < p; (i = (i + 1) | 0)) gaussStart[i] = startPrng.nextGaussian();
  const startDirection = new Float64Array(p);
  applyAnalytic(gaussStart, startDirection);
  let startNormSq = 0.0;
  for (let i = 0; i < p; (i = (i + 1) | 0)) startNormSq += startDirection[i]! * startDirection[i]!;
  const invStart = 1.0 / Math.sqrt(startNormSq);
  for (let i = 0; i < p; (i = (i + 1) | 0)) startDirection[i] = startDirection[i]! * invStart;
  const startNorm = 1e-6;
  const tCeil = Date.now();
  const ceilings = new Float64Array(numSeeds);
  const halfWidths = new Float64Array(numSeeds);
  let upperStable = 0;
  const divergenceEnergy = 2.5e-5;
  for (let s = 0; s < numSeeds; (s = (s + 1) | 0)) {
    const r = measureNetworkMinibatchHeavyBallCeiling(
      mlp, sampleGrads0, fullGrad0, startDirection, startNorm, beta, 1.1 * raw.etaCritQuadraticHb09, batchB, noiseScaleTau, streams, steps,
      divergenceEnergy, bisectionIterations, logGrowthThreshold, (0x10a10000 + s * 7919) ^ dIn, (0x3c6ef372 ^ Math.imul(s + 1, 0x9e3779b9)) | 0,
    );
    ceilings[s] = r.etaCeiling;
    halfWidths[s] = r.bracketHalfWidth;
    if (r.upperWasStable) upperStable = (upperStable + 1) | 0;
  }
  // Post-hoc diagnostic (no prediction registered): log energy growth on a fixed grid of learning rates at two run
  // lengths. A diffusive process gives the same window ratio at any length; exponential growth does not.
  const gridFractions = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5];
  const growthGrid: Phase2GrowthGridRow[] = [];
  for (let gi = 0; gi < gridFractions.length; (gi = (gi + 1) | 0)) {
    const eta = gridFractions[gi]! * raw.etaCritQuadraticHb09;
    for (let li = 0; li < 2; (li = (li + 1) | 0)) {
      const gridSteps = li === 0 ? steps : 4 * steps;
      const logGrowth = simulateNetworkMinibatchHeavyBallEnergy(
        mlp, sampleGrads0, fullGrad0, startDirection, startNorm, beta, eta, batchB, noiseScaleTau, 4, gridSteps,
        divergenceEnergy, 0x10a20000 ^ dIn, 0x2545f491,
      );
      const guardTriggered = logGrowth >= 1e3;
      growthGrid.push({
        etaOverQuadratic: gridFractions[gi]!,
        eta,
        steps: gridSteps,
        logGrowth,
        guardTriggered,
        ratePerStep: guardTriggered ? Number.POSITIVE_INFINITY : logGrowth / (0.5 * gridSteps),
      });
    }
  }
  const ceilingSeconds = (Date.now() - tCeil) / 1000;
  const median = computeFloat64Quantile(ceilings, 0.5);
  const halfWidthPct = (computeFloat64Quantile(halfWidths, 0.5) / median) * 100.0;
  const tol = 0.10 + halfWidthPct / 100.0;
  const inside = median >= bracket.etaTraceBracketLower * (1.0 - tol) && median <= bracket.etaTraceBracketUpper * (1.0 + tol);

  return {
    cellId,
    numParams: p,
    beta,
    batchB,
    lambda1,
    lambda2,
    lambda1AnalyticPowerIter40: analytic40.lambda,
    lambdaMinShiftedPowerIter100: lambdaMin,
    traceH,
    traceOverLambda1: traceH / lambda1,
    noiseRankRatio: bracket.noiseRankRatio,
    nuB,
    kappaFeatEffective: kappaFeat,
    alphaDiag: bracket.alphaDiag,
    etaCritQuadratic: raw.etaCritQuadraticHb09,
    etaCritDestructiveRay: raw.etaCritDestructiveHb09,
    quadraticVsDestructiveGapPct:
      (Math.abs(raw.etaCritQuadraticHb09 - raw.etaCritDestructiveHb09) / raw.etaCritDestructiveHb09) * 100.0,
    etaCritOneDimEoss: bracket.etaCritOneDim,
    etaSandwichLower: bracket.etaSandwichLower,
    etaSandwichUpper: bracket.etaSandwichUpper,
    etaTraceBracketLower: bracket.etaTraceBracketLower,
    etaTraceBracketUpper: bracket.etaTraceBracketUpper,
    empiricalCeilingMedian: median,
    empiricalCeilingP05: computeFloat64Quantile(ceilings, 0.05),
    empiricalCeilingP95: computeFloat64Quantile(ceilings, 0.95),
    bisectionHalfWidthPct: halfWidthPct,
    upperWasStableCount: upperStable,
    oneDimOverEmpiricalRatio: bracket.etaCritOneDim / median,
    quadraticOverEmpiricalRatio: raw.etaCritQuadraticHb09 / median,
    empiricalInsideTraceBracket: inside,
    growthGrid,
    fdHvpIterations: fdIterations,
    fdHvpSampleGradEquivalents: fdIterations * 2 * mlp.numSamples,
    fdHvpLambda1Hat: fd15.lambda,
    fdHvpLambda1RelErr: Math.abs(fd15.lambda - lambda1) / lambda1,
    fdHvpCosWithAnalyticU1: Math.abs(cosFd),
    fdHvpEtaCritHat: fdEtaHat,
    fdHvpErrVsQuadraticPct: fdErrQuad,
    fdHvpErrVsDestructivePct: fdErrDestr,
    traceSeconds,
    ceilingSeconds,
  };
}

export function evaluatePhase2TraceCeilingCells(
  numSeeds: number,
  streams: number,
  steps: number,
): Phase2Report {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const evidenceDir = path.resolve(here, '../../evidence/mean-square-boundary');
  const raw4 = JSON.parse(fs.readFileSync(path.join(evidenceDir, 'networks.json'), 'utf8')) as {
    readonly networks: ReadonlyArray<Parameters<typeof buildNetworkCell>[4] & { readonly dIn: number; readonly numParams: number }>;
  };
  const bisectionIterations = 14;
  const growthRatioThreshold = 2.5;
  const logGrowthThreshold = 0.9162907318741551; // log(2.5), the Phase 1 threshold
  const noiseScaleTau = 0.0;

  const net3k = raw4.networks.find((n) => n.numParams === 3610);
  const net12k = raw4.networks.find((n) => n.numParams === 12826);
  if (net3k === undefined || net12k === undefined) throw new Error('networks.json: scale networks missing');
  const cells: Phase2NetworkCell[] = [
    buildNetworkCell('X1_P3610_hb09_B16', 64, 0x67400001, 0x9e3779b9, net3k, numSeeds, streams, steps, bisectionIterations, logGrowthThreshold, noiseScaleTau),
    buildNetworkCell('X2_P12826_hb09_B16', 256, 0x67400002, 0x85ebca6b, net12k, numSeeds, streams, steps, bisectionIterations, logGrowthThreshold, noiseScaleTau),
  ];

  const every = (f: (c: Phase2NetworkCell) => boolean): boolean => cells.every(f);
  const predictions: Record<string, boolean> = {
    q1: every((c) => c.traceOverLambda1 >= 4.0),
    q2: every((c) => c.oneDimOverEmpiricalRatio >= 1.5),
    q3: every((c) => c.empiricalInsideTraceBracket),
    q4: every(
      (c) =>
        c.fdHvpLambda1RelErr < 1e-6 &&
        c.fdHvpErrVsQuadraticPct < 0.01 &&
        Math.abs(c.fdHvpErrVsDestructivePct - c.quadraticVsDestructiveGapPct) < 0.01,
    ),
  };

  return {
    labId: 'mean-square-boundary-phase2',
    date: new Date().toISOString(),
    nodeVersion: process.version,
    numSeeds,
    streams,
    steps,
    bisectionIterations,
    growthRatioThreshold,
    noiseScaleTau,
    predictionTexts: PREDICTION_TEXTS,
    predictions,
    networks: cells,
  };
}
