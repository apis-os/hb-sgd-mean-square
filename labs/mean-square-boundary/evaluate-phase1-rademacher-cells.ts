/**
 * @file labs/mean-square-boundary/evaluate-phase1-rademacher-cells.ts
 * @description Phase 1 evaluator (`evaluatePhase1RademacherCells`): for each cell (`P`, `beta`, `B`, feature family) and seed, the empirical mean-square ceiling of heavy-ball SGD by bisection is compared with the secular root of Theorem G, its Lee–Cheng–Paquette–Paquette limit, the sandwich bounds and the one-dimensional ceiling of Theorem E(3); predictions `p1`–`p4` are evaluated from the per-cell medians.
 * @cold
 */
import { computeFloat64Quantile } from './compute-float64-quantile.ts';
import { computeSymmetricEigenvaluesJacobi } from './compute-symmetric-eigenvalues-jacobi.ts';
import { measureEmpiricalEossCeiling } from './measure-empirical-eoss-ceiling.ts';
import type { Phase1CellRow, Phase1Report } from './phase1-types.ts';
import { createXorshift128Plus } from './prng-xorshift128p.ts';
import { solveSecularEossThreshold } from './solve-secular-eoss-threshold.ts';

interface CellSpec {
  readonly numParams: number;
  readonly beta: number;
  readonly batchB: number;
  readonly family: 'gaussian-iid' | 'rademacher-iid' | 'gaussian-pool';
  readonly poolSize: number;
}

function buildCellSpecs(): CellSpec[] {
  const specs: CellSpec[] = [];
  const betas = [0.0, 0.9];
  for (let pi = 0; pi < 2; (pi = (pi + 1) | 0)) {
    const numParams = pi === 0 ? 64 : 256;
    const batches = pi === 0 ? [4, 16, 64] : [4, 16];
    const poolSize = pi === 0 ? 512 : 1024;
    for (let bi = 0; bi < betas.length; (bi = (bi + 1) | 0)) {
      for (let ki = 0; ki < batches.length; (ki = (ki + 1) | 0)) {
        specs.push({ numParams, beta: betas[bi]!, batchB: batches[ki]!, family: 'gaussian-iid', poolSize: 0 });
        specs.push({ numParams, beta: betas[bi]!, batchB: batches[ki]!, family: 'rademacher-iid', poolSize: 0 });
        specs.push({ numParams, beta: betas[bi]!, batchB: batches[ki]!, family: 'gaussian-pool', poolSize });
      }
    }
  }
  return specs;
}

function buildSpectrum(numParams: number): Float64Array {
  const lambdas = new Float64Array(numParams);
  for (let k = 0; k < numParams; (k = (k + 1) | 0)) lambdas[k] = 1.0 / Math.sqrt(k + 1);
  return lambdas;
}

function buildPool(sqrtLambdas: Float64Array, poolSize: number, seed: number): { readonly pool: Float64Array; readonly eigen: Float64Array } {
  const p = sqrtLambdas.length | 0;
  const prng = createXorshift128Plus((0x91000000 + seed * 7919) | 0, (0x3c6ef372 ^ Math.imul(seed + 1, 0x9e3779b9)) | 0);
  const pool = new Float64Array(Math.imul(poolSize, p));
  for (let i = 0; i < poolSize; (i = (i + 1) | 0)) {
    const off = Math.imul(i, p);
    for (let k = 0; k < p; (k = (k + 1) | 0)) pool[(off + k) | 0] = sqrtLambdas[k]! * prng.nextGaussian();
  }
  const cov = new Float64Array(Math.imul(p, p));
  const invN = 1.0 / poolSize;
  for (let i = 0; i < poolSize; (i = (i + 1) | 0)) {
    const off = Math.imul(i, p);
    for (let a = 0; a < p; (a = (a + 1) | 0)) {
      const za = pool[(off + a) | 0]! * invN;
      const rowA = Math.imul(a, p);
      for (let b = a; b < p; (b = (b + 1) | 0)) cov[(rowA + b) | 0] = cov[(rowA + b) | 0]! + za * pool[(off + b) | 0]!;
    }
  }
  for (let a = 0; a < p; (a = (a + 1) | 0)) {
    for (let b = (a + 1) | 0; b < p; (b = (b + 1) | 0)) cov[(b * p + a) | 0] = cov[(a * p + b) | 0]!;
  }
  return { pool, eigen: computeSymmetricEigenvaluesJacobi(cov, p) };
}

export function evaluatePhase1RademacherCells(
  numSeeds: number,
  streams = 8,
  steps = 1000,
  bisectionIterations = 12,
  growthRatioThreshold = 2.5,
): Phase1Report {
  const specs = buildCellSpecs();
  const logThreshold = growthRatioThreshold > 1 ? 0.9162907318741551 * (growthRatioThreshold / 2.5) : 0.0;
  const cells: Phase1CellRow[] = [];
  const sigmaXi = 0.01;

  for (let ci = 0; ci < specs.length; (ci = (ci + 1) | 0)) {
    const spec = specs[ci]!;
    const lambdasModel = buildSpectrum(spec.numParams);
    const sqrtLambdas = new Float64Array(spec.numParams);
    for (let k = 0; k < spec.numParams; (k = (k + 1) | 0)) sqrtLambdas[k] = Math.sqrt(lambdasModel[k]!);
    const kappaFeat = spec.family === 'rademacher-iid' ? -2.0 : 0.0;
    const zeta = spec.family === 'gaussian-pool' ? spec.batchB / spec.poolSize : 0.0;
    const empirical = new Float64Array(numSeeds);
    const secErr = new Float64Array(numSeeds);
    const leeErr = new Float64Array(numSeeds);
    const oneErr = new Float64Array(numSeeds);
    const halfWidths = new Float64Array(numSeeds);
    let sumSecular = 0.0;
    let sumLee = 0.0;
    let sumLower = 0.0;
    let sumUpper = 0.0;
    let sumOneDim = 0.0;
    let sumDet = 0.0;
    let sumLambda1 = 0.0;
    let sumTrace = 0.0;
    let sumOneOverEmp = 0.0;

    for (let seed = 0; seed < numSeeds; (seed = (seed + 1) | 0)) {
      let lambdas = lambdasModel;
      let pool: Float64Array | null = null;
      if (spec.family === 'gaussian-pool') {
        const built = buildPool(sqrtLambdas, spec.poolSize, (seed + Math.imul(ci, 101)) | 0);
        pool = built.pool;
        lambdas = built.eigen;
      }
      const pred = solveSecularEossThreshold(lambdas, spec.beta, spec.batchB, kappaFeat, zeta);
      // Bracket: expand upward from four times the predicted root until unstable, capped at 1.02 x the deterministic ceiling.
      let etaUpper = Math.min(4.0 * pred.etaCritSecular, 1.02 * pred.etaCritDeterministic);
      let result = measureEmpiricalEossCeiling(
        sqrtLambdas, spec.beta, etaUpper, spec.batchB, spec.family === 'rademacher-iid' ? 1 : 0, pool, spec.poolSize,
        sigmaXi, streams, steps, bisectionIterations, logThreshold,
        (0x95000000 + Math.imul(ci + 1, 0x1000) + seed * 613) | 0, (0x2545f491 ^ Math.imul(seed + 3, 0x9e3779b9) ^ Math.imul(ci + 1, 0x85ebca6b)) | 0,
      );
      let expansions = 0;
      while (result.upperWasStable && etaUpper < 1.02 * pred.etaCritDeterministic && expansions < 6) {
        etaUpper = Math.min(2.0 * etaUpper, 1.02 * pred.etaCritDeterministic);
        expansions = (expansions + 1) | 0;
        result = measureEmpiricalEossCeiling(
          sqrtLambdas, spec.beta, etaUpper, spec.batchB, spec.family === 'rademacher-iid' ? 1 : 0, pool, spec.poolSize,
          sigmaXi, streams, steps, bisectionIterations, logThreshold,
          (0x95000000 + Math.imul(ci + 1, 0x1000) + seed * 613 + expansions) | 0, (0x2545f491 ^ Math.imul(seed + 3, 0x9e3779b9) ^ Math.imul(ci + 1, 0x85ebca6b)) | 0,
        );
      }
      const emp = result.etaCeiling;
      empirical[seed] = emp;
      halfWidths[seed] = (result.bracketHalfWidth / Math.max(1e-300, emp)) * 100.0;
      secErr[seed] = (Math.abs(pred.etaCritSecular - emp) / emp) * 100.0;
      leeErr[seed] = (Math.abs(pred.etaCritSecularLeeLimit - emp) / emp) * 100.0;
      oneErr[seed] = (Math.abs(pred.etaCritOneDim - emp) / emp) * 100.0;
      sumSecular += pred.etaCritSecular;
      sumLee += pred.etaCritSecularLeeLimit;
      sumLower += pred.etaCritSandwichLower;
      sumUpper += pred.etaCritSandwichUpper;
      sumOneDim += pred.etaCritOneDim;
      sumDet += pred.etaCritDeterministic;
      sumLambda1 += pred.lambda1;
      sumTrace += pred.traceH;
      sumOneOverEmp += pred.etaCritOneDim / emp;
    }
    const invS = 1.0 / numSeeds;
    const empMedian = computeFloat64Quantile(empirical, 0.5);
    const lower = sumLower * invS;
    const upper = sumUpper * invS;
    const tolerance = 0.03 + computeFloat64Quantile(halfWidths, 0.5) / 100.0;
    const kappaBeta = (1.0 + spec.beta) / (1.0 - spec.beta);
    const row: Phase1CellRow = {
      cellId: `P${spec.numParams}_beta${spec.beta}_B${spec.batchB}_${spec.family}`,
      numParams: spec.numParams,
      beta: spec.beta,
      batchB: spec.batchB,
      featureFamily: spec.family,
      kappaFeat,
      poolSize: spec.poolSize,
      zeta,
      lambda1: sumLambda1 * invS,
      traceH: sumTrace * invS,
      noiseRankRatio: (kappaBeta * (sumTrace * invS)) / (spec.batchB * (sumLambda1 * invS)),
      etaCritDeterministic: sumDet * invS,
      etaCritOneDim: sumOneDim * invS,
      etaCritSecular: sumSecular * invS,
      etaCritSecularLeeLimit: sumLee * invS,
      etaCritSandwichLower: lower,
      etaCritSandwichUpper: upper,
      empiricalCeilingMedian: empMedian,
      empiricalCeilingP05: computeFloat64Quantile(empirical, 0.05),
      empiricalCeilingP95: computeFloat64Quantile(empirical, 0.95),
      bisectionHalfWidthPct: computeFloat64Quantile(halfWidths, 0.5),
      secularMedianErrPct: computeFloat64Quantile(secErr, 0.5),
      secularLeeLimitMedianErrPct: computeFloat64Quantile(leeErr, 0.5),
      oneDimMedianErrPct: computeFloat64Quantile(oneErr, 0.5),
      oneDimOverEmpiricalRatio: sumOneOverEmp * invS,
      insideSandwich: empMedian >= lower * (1.0 - tolerance) && empMedian <= upper * (1.0 + tolerance),
    };
    cells.push(row);
  }

  let p1 = true;
  let p2 = true;
  let p3 = true;
  let p3Count = 0;
  let p4 = true;
  let p4Count = 0;
  for (let i = 0; i < cells.length; (i = (i + 1) | 0)) {
    const c = cells[i]!;
    const iid = c.featureFamily !== 'gaussian-pool';
    if (iid && !c.insideSandwich) p1 = false;
    if (iid && c.secularMedianErrPct >= 5.0) p2 = false;
    if (c.featureFamily === 'rademacher-iid' && c.noiseRankRatio >= 1.0) {
      p3Count = (p3Count + 1) | 0;
      if (c.oneDimOverEmpiricalRatio <= 2.0 || c.secularMedianErrPct >= 5.0) p3 = false;
    }
    if (c.featureFamily === 'gaussian-pool' && c.zeta <= 0.125) {
      p4Count = (p4Count + 1) | 0;
      if (c.secularLeeLimitMedianErrPct >= 8.0) p4 = false;
    }
  }
  if (p3Count === 0) p3 = false;
  if (p4Count === 0) p4 = false;

  return {
    labId: 'mean-square-boundary-phase1',
    date: new Date().toISOString(),
    nodeVersion: process.version,
    numSeeds,
    streams,
    steps,
    bisectionIterations,
    growthRatioThreshold,
    cells,
    predictionTexts: {
      p1: 'p1 (Theorem G(3)): in every i.i.d. cell (Gaussian and Rademacher features), the median empirical mean-square ceiling lies inside the sandwich [2(1+beta)/(alpha lambda_1 + kappa_beta tr(H)/B), 2(1+beta)/max(alpha lambda_1 + kappa_beta lambda_1/B, kappa_beta tr(H)/B)] within 3 % plus the bisection half-width.',
      p2: 'p2 (Theorem G(2)): in every i.i.d. cell the secular root predicts the empirical ceiling with median relative error below 5 %.',
      p3: 'p3 (Theorem G(4), Rademacher separation): in every Rademacher cell with kappa_beta tr(H)/(B lambda_1) >= 1, the one-dimensional ceiling 2(1+beta)/lambda_1 of Theorem E(3) (nu_11 = 0) exceeds the empirical ceiling by more than a factor 2 while the secular root stays within 5 %.',
      p4: 'p4 (Remark G.1, Lee–Cheng–Paquette–Paquette 2022): in the finite-pool arm (Gaussian features sampled without replacement, zeta = B/n <= 1/8), the root of the kernel-norm equation ||K|| = 1 with the factor (1 - zeta) and no kurtosis term predicts the empirical ceiling with median relative error below 8 %.',
    },
    predictions: {
      p1IidCeilingsInsideSandwich: p1,
      p2SecularRootWithin5PctMedianIid: p2,
      p3RademacherOneDimOverestimatesBy2xWhileSecularWithin5Pct: p3,
      p4PoolArmLeeRootWithin8PctAtSmallZeta: p4,
    },
  };
}
