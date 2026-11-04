/**
 * @file labs/mean-square-boundary/evaluate-phase3-operator-radius-cell.ts
 * @description One Phase 3 cell (`evaluatePhase3OperatorRadiusCell`): on the spectrum `lambda_k = k^{-1/2}`, the spectral radius of the exact second-moment operator of heavy-ball SGD (`measureSecondMomentSpectralRadius`, power iteration from the identity on the symmetric part, warm-started across the three probes) at `(1 - d) eta*`, `eta*` and `(1 + d) eta*`, with `eta*` the secular root of Theorem G (`solveSecularEossThreshold`), the relative distance from `eta*` to the crossing `rho = 1` by the local slope, and the radius of the antisymmetric part at `eta*` (Gelfand estimate over a fixed horizon, from a seeded antisymmetric start).
 * @cold
 */
import { measureSecondMomentSpectralRadius } from './measure-second-moment-spectral-radius.ts';
import type { Phase3CellRow, Phase3CellSpec } from './phase3-types.ts';
import { createXorshift128Plus } from './prng-xorshift128p.ts';
import { solveSecularEossThreshold } from './solve-secular-eoss-threshold.ts';

function setIdentity(q: Float64Array, n: number): void {
  q.fill(0.0);
  const v = 1.0 / Math.sqrt(n);
  for (let i = 0; i < n; (i = (i + 1) | 0)) q[(Math.imul(i, n) + i) | 0] = v;
}

function setRandomAntisymmetric(q: Float64Array, n: number, seed: number): void {
  const prng = createXorshift128Plus((0x11a00000 + seed) | 0, 0x2545f491);
  q.fill(0.0);
  let normSq = 0.0;
  for (let i = 0; i < n; (i = (i + 1) | 0)) {
    for (let j = (i + 1) | 0; j < n; (j = (j + 1) | 0)) {
      const v = prng.nextGaussian();
      q[(Math.imul(i, n) + j) | 0] = v;
      q[(Math.imul(j, n) + i) | 0] = -v;
      normSq += 2.0 * v * v;
    }
  }
  const inv = 1.0 / Math.sqrt(normSq);
  const total = Math.imul(n, n);
  for (let i = 0; i < total; (i = (i + 1) | 0)) q[i] = q[i]! * inv;
}

export function evaluatePhase3OperatorRadiusCell(spec: Phase3CellSpec): Phase3CellRow {
  const p = spec.numParams | 0;
  const n = (2 * p) | 0;
  const lambdas = new Float64Array(p);
  for (let k = 0; k < p; (k = (k + 1) | 0)) lambdas[k] = 1.0 / Math.sqrt(k + 1);
  const q = new Float64Array(Math.imul(n, n));
  const scratch = new Float64Array(Math.imul(n, n));
  const beta = spec.beta;
  const batchB = spec.batchB;
  const kappaFeat = spec.featureFamily === 'gaussian-iid' ? 0.0 : -2.0;
  const d = spec.relativeOffset;
  const etaStar = solveSecularEossThreshold(lambdas, beta, batchB, kappaFeat, 0.0).etaCritSecular;
  setIdentity(q, n);
  const below = measureSecondMomentSpectralRadius(lambdas, beta, etaStar * (1.0 - d), batchB, kappaFeat, q, scratch, 0, spec.maxIterations, spec.relTol);
  const atRoot = measureSecondMomentSpectralRadius(lambdas, beta, etaStar, batchB, kappaFeat, q, scratch, 0, spec.maxIterations, spec.relTol);
  const above = measureSecondMomentSpectralRadius(lambdas, beta, etaStar * (1.0 + d), batchB, kappaFeat, q, scratch, 0, spec.maxIterations, spec.relTol);
  setRandomAntisymmetric(q, n, Math.imul(p, 31) + Math.imul(batchB, 7) + spec.betaIndex * 3 + spec.familyIndex);
  const anti = measureSecondMomentSpectralRadius(lambdas, beta, etaStar, batchB, kappaFeat, q, scratch, 1, spec.antisymmetricHorizon, spec.relTol);
  const slope = (above.radius - below.radius) / (2.0 * d);
  return {
    cellId: `P${p}_b${beta}_B${batchB}_${spec.featureFamily}`,
    numParams: p,
    beta,
    batchB,
    featureFamily: spec.featureFamily,
    kappaFeat,
    etaCritSecular: etaStar,
    radiusBelow: below.radius,
    radiusAtRoot: atRoot.radius,
    radiusAbove: above.radius,
    radiusAntisymmetricAtRoot: anti.radius,
    crossingRelErr: Math.abs(atRoot.radius - 1.0) / Math.abs(slope),
    iterationsAtRoot: atRoot.iterations,
    allConverged: below.converged && atRoot.converged && above.converged && anti.converged,
  };
}
