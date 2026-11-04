/**
 * @file tests/phase1-secular-ceiling.test.ts
 * @description Phase 1: the secular equation of Theorem G (`solveSecularEossThreshold`), the Jacobi eigenvalue
 * solver used for the finite-pool arm, the mean-square divergence statistic of the heavy-ball simulator on both sides of
 * the predicted ceiling, a reduced real execution of `evaluatePhase1RademacherCells` (in a worker thread), and the committed `raw-phase1.json`
 * with its pre-registered predictions, whatever their outcome.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Worker } from 'node:worker_threads';
import { describe, it } from 'vitest';
import { computeSymmetricEigenvaluesJacobi } from '../labs/mean-square-boundary/compute-symmetric-eigenvalues-jacobi.ts';
import { measureEmpiricalEossCeiling } from '../labs/mean-square-boundary/measure-empirical-eoss-ceiling.ts';
import type { Phase1Report } from '../labs/mean-square-boundary/phase1-types.ts';
import { simulateHeavyBallFeatureNoiseEnergy } from '../labs/mean-square-boundary/simulate-heavy-ball-feature-noise-energy.ts';
import { solveSecularEossThreshold } from '../labs/mean-square-boundary/solve-secular-eoss-threshold.ts';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');

describe('Phase 1 — the stochastic ceiling in R^P (Theorem G)', () => {
  it('reduces to the one-dimensional ceiling of Theorem E(3) at P = 1 and orders the sandwich', () => {
    const lambdas = new Float64Array([3.0]);
    for (const beta of [0.0, 0.5, 0.9]) {
      for (const kappa of [-2.0, 0.0, 2.0]) {
        const r = solveSecularEossThreshold(lambdas, beta, 16, kappa, 0.0);
        const kappaBeta = (1.0 + beta) / (1.0 - beta);
        const nu11Sq = (9.0 * (2.0 + kappa)) / 16.0;
        const oneDim = (2.0 * (1.0 + beta) * 3.0) / (9.0 + kappaBeta * nu11Sq);
        assert.ok(Math.abs(r.etaCritSecular - oneDim) < 1e-10 * oneDim, `P=1 secular = 1D at beta=${beta} kappa=${kappa}`);
        assert.ok(Math.abs(r.etaCritOneDim - oneDim) < 1e-12 * oneDim);
        assert.ok(r.etaCritSandwichLower <= r.etaCritSecular + 1e-12 && r.etaCritSecular <= r.etaCritSandwichUpper + 1e-12);
      }
    }
  });

  it('satisfies the secular equation at the root, lies inside the sandwich, and the Rademacher separation holds', () => {
    const p = 64;
    const lambdas = new Float64Array(p);
    for (let k = 0; k < p; k++) lambdas[k] = 1.0 / Math.sqrt(k + 1);
    for (const beta of [0.0, 0.9]) {
      for (const batchB of [4, 16]) {
        for (const kappa of [0.0, -2.0]) {
          const r = solveSecularEossThreshold(lambdas, beta, batchB, kappa, 0.0);
          const kappaBeta = (1.0 + beta) / (1.0 - beta);
          const alpha = 1.0 + (kappaBeta * (1.0 + kappa)) / batchB;
          let sum = 0.0;
          for (let k = 0; k < p; k++) sum += lambdas[k]! / (1.0 - (r.etaCritSecular * lambdas[k]! * alpha) / (2.0 * (1.0 + beta)));
          const lhs = (r.etaCritSecular * sum) / (2.0 * (1.0 - beta) * batchB);
          assert.ok(Math.abs(lhs - 1.0) < 1e-9, `secular residual at beta=${beta} B=${batchB} kappa=${kappa}: ${lhs}`);
          assert.ok(r.etaCritSandwichLower <= r.etaCritSecular && r.etaCritSecular <= r.etaCritSandwichUpper);
          if (kappa === -2.0) {
            assert.ok(Math.abs(r.etaCritOneDim - r.etaCritDeterministic) < 1e-12, 'Rademacher: 1D ceiling equals 2(1+beta)/lambda_1');
            const tracePerp = r.traceH - r.lambda1;
            const bound = batchB <= kappaBeta
              ? (2.0 * (1.0 + beta)) / (r.lambda1 + (kappaBeta * tracePerp) / batchB)
              : (2.0 * (1.0 + beta)) / Math.max(r.lambda1, (kappaBeta * r.traceH) / batchB);
            assert.ok(r.etaCritSecular <= bound + 1e-12, 'Theorem G(4) upper bound');
            if (batchB <= kappaBeta) {
              assert.ok(r.etaCritOneDim / r.etaCritSecular > 1.0 + (kappaBeta * tracePerp) / (batchB * r.lambda1) - 1e-9, 'separation factor');
            }
          }
        }
      }
    }
    // Mutation: scaling the bulk changes the root (the ceiling depends on tr(H), not only on lambda_1)
    const bulkScaled = new Float64Array(lambdas);
    for (let k = 1; k < p; k++) bulkScaled[k] = 0.5 * lambdas[k]!;
    const base = solveSecularEossThreshold(lambdas, 0.9, 4, 0.0, 0.0);
    const mutated = solveSecularEossThreshold(bulkScaled, 0.9, 4, 0.0, 0.0);
    assert.ok(mutated.etaCritSecular > 1.3 * base.etaCritSecular);
    assert.ok(Math.abs(mutated.etaCritDeterministic - base.etaCritDeterministic) < 1e-12);
  });

  it('computes symmetric eigenvalues by cyclic Jacobi (diagonal-plus-rank-one check against the secular equation)', () => {
    const n = 5;
    const d = [5.0, 4.0, 3.0, 2.0, 1.0];
    const v = [1.0, 0.5, 0.25, 0.125, 0.0625];
    const m = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) m[i * n + j] = (i === j ? d[i]! : 0.0) + v[i]! * v[j]!;
    const eig = computeSymmetricEigenvaluesJacobi(m, n);
    let trace = 0.0;
    for (let i = 0; i < n; i++) trace += eig[i]!;
    assert.ok(Math.abs(trace - (15.0 + 1.0 + 0.25 + 0.0625 + 0.015625 + 0.00390625)) < 1e-10);
    for (let i = 0; i < n; i++) {
      let s = 0.0;
      for (let j = 0; j < n; j++) s += (v[j]! * v[j]!) / (eig[i]! - d[j]!);
      assert.ok(Math.abs(1.0 - s) < 1e-7, `secular residual of eigenvalue ${i}`);
    }
    assert.ok(eig[0]! >= eig[1]! && eig[1]! >= eig[4]!);
  });

  it('declares the heavy-ball process stable below and unstable above the secular root, and the bisection lands near it', () => {
    const p = 32;
    const lambdas = new Float64Array(p);
    const sqrtLambdas = new Float64Array(p);
    for (let k = 0; k < p; k++) {
      lambdas[k] = 1.0 / Math.sqrt(k + 1);
      sqrtLambdas[k] = Math.sqrt(lambdas[k]!);
    }
    const beta = 0.9;
    const batchB = 4;
    const r = solveSecularEossThreshold(lambdas, beta, batchB, -2.0, 0.0);
    const below = simulateHeavyBallFeatureNoiseEnergy(sqrtLambdas, beta, 0.8 * r.etaCritSecular, batchB, 1, null, 0, 0.01, 8, 1200, 11, 22);
    const above = simulateHeavyBallFeatureNoiseEnergy(sqrtLambdas, beta, 1.3 * r.etaCritSecular, batchB, 1, null, 0, 0.01, 8, 1200, 11, 22);
    assert.ok(below < 0.9162907318741551, `stable side grows by ${below}`);
    assert.ok(above > 0.9162907318741551, `unstable side grows by ${above}`);
    const ceiling = measureEmpiricalEossCeiling(
      sqrtLambdas, beta, 4.0 * r.etaCritSecular, batchB, 1, null, 0, 0.01, 8, 1200, 10, 0.9162907318741551, 11, 22,
    );
    assert.ok(!ceiling.upperWasStable);
    assert.ok(Math.abs(ceiling.etaCeiling - r.etaCritSecular) / r.etaCritSecular < 0.12, `bisection ${ceiling.etaCeiling} vs root ${r.etaCritSecular}`);
    // Mutation: a wrong momentum in the prediction misplaces the root by more than the measured tolerance
    const wrongBeta = solveSecularEossThreshold(lambdas, 0.5, batchB, -2.0, 0.0);
    assert.ok(Math.abs(ceiling.etaCeiling - wrongBeta.etaCritSecular) / ceiling.etaCeiling > 0.3);
  });

  it('executes a reduced evaluatePhase1RademacherCells and validates the committed raw-phase1.json', async () => {
    // The reduced run takes about 16 s on an idle machine; it runs in a worker thread so that the Vitest worker can answer
    // its own RPC calls meanwhile (their timeout is 60 s, which a synchronous run can exceed on a loaded machine).
    const fast = await new Promise<Phase1Report>((resolve, reject) => {
      const worker = new Worker(new URL('./workers/evaluate-phase1-reduced.ts', import.meta.url));
      worker.once('message', (report: Phase1Report) => resolve(report));
      worker.once('error', reject);
      worker.once('exit', (code) => {
        if (code !== 0) reject(new Error(`evaluate-phase1-reduced worker exited with code ${code}`));
      });
    });
    assert.equal(fast.labId, 'mean-square-boundary-phase1');
    assert.equal(fast.cells.length, 30);
    for (const c of fast.cells) {
      assert.ok(c.etaCritSandwichLower <= c.etaCritSecular && c.etaCritSecular <= c.etaCritSandwichUpper);
      assert.ok(Number.isFinite(c.empiricalCeilingMedian) && c.empiricalCeilingMedian > 0);
    }
    const rawPath = path.join(repoRoot, 'evidence', 'mean-square-boundary', 'raw-phase1.json');
    assert.ok(fs.existsSync(rawPath), 'evidence/mean-square-boundary/raw-phase1.json must be committed');
    const report = JSON.parse(fs.readFileSync(rawPath, 'utf8')) as Phase1Report;
    assert.equal(report.labId, 'mean-square-boundary-phase1');
    assert.equal(report.numSeeds, 8);
    assert.equal(report.cells.length, 30);
    for (const key of Object.keys(report.predictions)) {
      assert.equal(typeof report.predictions[key as keyof Phase1Report['predictions']], 'boolean');
    }
    // The committed prediction outcomes are pinned as literals so that a silent regeneration is detected.
    assert.deepEqual(report.predictions, EXPECTED_PREDICTIONS);
  }, 600_000);
});

// Filled from raw-phase1.json (field date 2026-10-27). p1 and p2 failed on one cell (P = 64, beta = 0, B = 64, Gaussian
// features), where the sample-path ceiling lies above the mean-square one (Proposition G.3); they stay false.
const EXPECTED_PREDICTIONS: Phase1Report['predictions'] = {
  p1IidCeilingsInsideSandwich: false,
  p2SecularRootWithin5PctMedianIid: false,
  p3RademacherOneDimOverestimatesBy2xWhileSecularWithin5Pct: true,
  p4PoolArmLeeRootWithin8PctAtSmallZeta: true,
};
