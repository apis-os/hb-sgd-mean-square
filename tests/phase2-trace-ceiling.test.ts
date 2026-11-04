/**
 * @file tests/phase2-trace-ceiling.test.ts
 * @description Phase 2: the exact Hessian trace against a finite-difference diagonal oracle, the seeded power
 * iteration on an explicit matrix, the trace bracket of the secular root against the full root of Theorem G, the
 * minibatch heavy-ball energy statistic on the `P = 33` network on both sides of the deterministic threshold, the
 * bisection harness, and the committed `raw-phase2.json` with its
 * pre-registered predictions, whatever their outcome.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it } from 'vitest';
import { computeExactHessianTrace } from '../labs/mean-square-boundary/compute-exact-hessian-trace.ts';
import { computeSecularCeilingBracketFromTrace } from '../labs/mean-square-boundary/compute-secular-ceiling-bracket-from-trace.ts';
import { computeTopEigenpairPowerIteration } from '../labs/mean-square-boundary/compute-top-eigenpair-power-iteration.ts';
import { computeMlpGroundTruthStability } from '../labs/mean-square-boundary/hessian-ground-truth.ts';
import { measureNetworkMinibatchHeavyBallCeiling } from '../labs/mean-square-boundary/measure-network-minibatch-heavy-ball-ceiling.ts';
import { createTwoLayerMlpDatasetAndOps } from '../labs/mean-square-boundary/mlp-network.ts';
import type { Phase2Report } from '../labs/mean-square-boundary/phase2-types.ts';
import { precomputeStaticSampleGrads } from '../labs/mean-square-boundary/precompute-static-sample-grads.ts';
import { simulateNetworkMinibatchHeavyBallEnergy } from '../labs/mean-square-boundary/simulate-network-minibatch-heavy-ball-energy.ts';
import { solveSecularEossThreshold } from '../labs/mean-square-boundary/solve-secular-eoss-threshold.ts';
import { sumPrecomputedSampleGrads } from '../labs/mean-square-boundary/sum-precomputed-sample-grads.ts';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');
const LOG_2_5 = 0.9162907318741551;

function applyExplicit(m: Float64Array, n: number): (v: Float64Array, out: Float64Array) => void {
  return (v, out) => {
    for (let i = 0; i < n; i++) {
      let acc = 0.0;
      for (let j = 0; j < n; j++) acc += m[i * n + j]! * v[j]!;
      out[i] = acc;
    }
  };
}

describe('Phase 2 — trace, the ceiling in R^P on the networks, the FD-HVP baseline', () => {
  it('computes the exact trace of an explicit symmetric matrix and of the P = 33 network Hessian (FD diagonal oracle)', () => {
    const m = new Float64Array([4, 1, 0.5, 1, 3, -0.25, 0.5, -0.25, 2]);
    assert.ok(Math.abs(computeExactHessianTrace(applyExplicit(m, 3), 3) - 9.0) < 1e-15);
    const mlp = createTwoLayerMlpDatasetAndOps();
    const p = mlp.numParams;
    const theta0 = mlp.checkpointTheta0;
    const eps = 1e-5;
    const tp = new Float64Array(p);
    const tm = new Float64Array(p);
    const gp = new Float64Array(p);
    const gm = new Float64Array(p);
    const applyFd = (v: Float64Array, out: Float64Array): void => {
      for (let i = 0; i < p; i++) {
        tp[i] = theta0[i]! + eps * v[i]!;
        tm[i] = theta0[i]! - eps * v[i]!;
      }
      mlp.computeFullGradient(tp, gp);
      mlp.computeFullGradient(tm, gm);
      for (let i = 0; i < p; i++) out[i] = (gp[i]! - gm[i]!) * (0.5 / eps);
    };
    // independent oracle: the diagonal from second differences of the loss itself
    let diagOracle = 0.0;
    const t = new Float64Array(p);
    const h = 1e-3;
    const l0 = mlp.evalFullLoss(theta0);
    for (let i = 0; i < p; i++) {
      t.set(theta0);
      t[i] = theta0[i]! + h;
      const lp = mlp.evalFullLoss(t);
      t[i] = theta0[i]! - h;
      const lm = mlp.evalFullLoss(t);
      diagOracle += (lp - 2.0 * l0 + lm) / (h * h);
    }
    const trace = computeExactHessianTrace(applyFd, p);
    assert.ok(trace > 0.0);
    assert.ok(Math.abs(trace - diagOracle) < 1e-3 * Math.abs(trace), `trace ${trace} vs second-difference oracle ${diagOracle}`);
    // mutation: dropping one diagonal entry changes the trace by that entry
    const e = new Float64Array(p);
    const hv = new Float64Array(p);
    e[0] = 1.0;
    applyFd(e, hv);
    const skipFirst = (v: Float64Array, out: Float64Array): void => {
      applyFd(v, out);
      if (v[0] === 1.0) out[0] = 0.0;
    };
    assert.ok(Math.abs(computeExactHessianTrace(skipFirst, p) - (trace - hv[0]!)) < 1e-12 * Math.abs(trace));
  });

  it('power iteration returns the top eigenpair of an explicit matrix and the Rayleigh quotient of the last product', () => {
    const m = new Float64Array([2, 1, 0, 1, 2, 0, 0, 0, 1]); // eigenvalues 3, 1, 1
    const r = computeTopEigenpairPowerIteration(applyExplicit(m, 3), 3, 60, 0x1234, 0x5678);
    assert.ok(Math.abs(r.lambda - 3.0) < 1e-12, `lambda ${r.lambda}`);
    assert.ok(Math.abs(Math.abs(r.vec[0]!) - Math.SQRT1_2) < 1e-9 && Math.abs(r.vec[2]!) < 1e-9);
    assert.equal(r.products, 60);
    const r1 = computeTopEigenpairPowerIteration(applyExplicit(m, 3), 3, 1, 0x1234, 0x5678);
    assert.ok(r1.lambda < 3.0 && r1.lambda >= 1.0, 'one product gives a Rayleigh quotient between the extreme eigenvalues');
  });

  it('trace bracket contains the full secular root of Theorem G and sits inside the sandwich; it widens with the trace', () => {
    const p = 64;
    const lambdas = new Float64Array(p);
    let trace = 0.0;
    for (let k = 0; k < p; k++) {
      lambdas[k] = 1.0 / Math.sqrt(k + 1);
      trace += lambdas[k]!;
    }
    for (const beta of [0.0, 0.9]) {
      for (const batchB of [4, 16]) {
        for (const kappa of [0.0, -1.0, -1.5]) {
          const full = solveSecularEossThreshold(lambdas, beta, batchB, kappa, 0.0);
          const br = computeSecularCeilingBracketFromTrace(lambdas[0]!, lambdas[1]!, trace, beta, batchB, kappa);
          if (br.alphaDiag <= 0.0) {
            assert.ok(Number.isNaN(br.etaTraceBracketLower) && Number.isNaN(br.etaTraceBracketUpper), 'no bracket when alpha_diag <= 0');
            assert.ok(Math.abs(br.etaSandwichLower - full.etaCritSandwichLower) < 1e-12 && Math.abs(br.etaSandwichUpper - full.etaCritSandwichUpper) < 1e-12);
            continue;
          }
          assert.ok(br.etaTraceBracketLower <= full.etaCritSecular * (1.0 + 1e-9), `lower ${br.etaTraceBracketLower} root ${full.etaCritSecular} (beta ${beta} B ${batchB} kappa ${kappa})`);
          assert.ok(br.etaTraceBracketUpper >= full.etaCritSecular * (1.0 - 1e-9), `upper ${br.etaTraceBracketUpper} root ${full.etaCritSecular}`);
          assert.ok(br.etaTraceBracketLower >= full.etaCritSandwichLower * (1.0 - 1e-9) && br.etaTraceBracketUpper <= full.etaCritSandwichUpper * (1.0 + 1e-9), 'inside the sandwich');
          assert.ok(Math.abs(br.etaSandwichLower - full.etaCritSandwichLower) < 1e-12 && Math.abs(br.etaSandwichUpper - full.etaCritSandwichUpper) < 1e-12);
          assert.ok(Math.abs(br.etaCritOneDim - full.etaCritOneDim) < 1e-12);
          const wider = computeSecularCeilingBracketFromTrace(lambdas[0]!, lambdas[1]!, 2.0 * trace, beta, batchB, kappa);
          assert.ok(wider.etaTraceBracketLower < br.etaTraceBracketLower && wider.etaTraceBracketUpper < br.etaTraceBracketUpper, 'more trace lowers both ends');
        }
      }
    }
    const degenerate = computeSecularCeilingBracketFromTrace(1.0, 0.5, 3.0, 0.9, 1, -2.0);
    assert.ok(Number.isNaN(degenerate.etaTraceBracketLower) && degenerate.alphaDiag <= 0.0, 'alpha_diag <= 0 yields no bracket');
  });

  it('minibatch heavy-ball energy on the P = 33 network decays far below and grows above the deterministic threshold; the bisection lands between', () => {
    const mlp = createTwoLayerMlpDatasetAndOps();
    const gt = computeMlpGroundTruthStability(mlp, mlp.checkpointTheta0);
    const p = mlp.numParams;
    const fullGrad0 = new Float64Array(p);
    mlp.computeFullGradient(mlp.checkpointTheta0, fullGrad0);
    const sampleGrads0 = precomputeStaticSampleGrads(mlp);
    // sum of rows at batch size 16 is bit-identical to a 1/16 mean
    const idx = new Int32Array(16);
    for (let b = 0; b < 16; b++) idx[b] = (b * 5) % mlp.numSamples;
    const g0 = new Float64Array(p);
    sumPrecomputedSampleGrads(sampleGrads0, idx, 16, p, g0);
    let acc = 0.0;
    for (let b = 0; b < 16; b++) acc += sampleGrads0[idx[b]! * p + 3]!;
    assert.equal(g0[3], acc * 0.0625);
    const start = new Float64Array(p);
    start.set(gt.topEigenvectorU1);
    const args = [mlp, sampleGrads0, fullGrad0, start, 1e-6, 0.9] as const;
    // At P = 33, B = 16 the minibatch curvature noise is large (nu_B ~ lambda_1): the R^P ceiling of heavy-ball SGD sits
    // near 0.03 x the deterministic threshold (growth -0.49 at 0.02 x, +3.6 at 0.2 x, explosion from 0.3 x; probe of 2026-10-24).
    const low = simulateNetworkMinibatchHeavyBallEnergy(...args, 0.005 * gt.etaCritQuadraticHb09, 16, 0.0, 4, 400, 2.5e-5, 0x11, 0x22);
    const mid = simulateNetworkMinibatchHeavyBallEnergy(...args, 0.2 * gt.etaCritQuadraticHb09, 16, 0.0, 4, 400, 2.5e-5, 0x11, 0x22);
    const high = simulateNetworkMinibatchHeavyBallEnergy(...args, 1.2 * gt.etaCritQuadraticHb09, 16, 0.0, 4, 400, 2.5e-5, 0x11, 0x22);
    assert.ok(low < 0.0, `energy decays at 0.005 x the quadratic threshold: ${low}`);
    assert.ok(mid > LOG_2_5, `growth at 0.2 x quadratic threshold ${mid}`);
    assert.equal(high, 1e3, 'excursion guard at 1.2 x the quadratic threshold');
    const r = measureNetworkMinibatchHeavyBallCeiling(...args, 1.2 * gt.etaCritQuadraticHb09, 16, 0.0, 4, 400, 2.5e-5, 5, LOG_2_5, 0x11, 0x22);
    assert.equal(r.upperWasStable, false);
    assert.ok(r.etaCeiling > 0.005 * gt.etaCritQuadraticHb09 && r.etaCeiling < 0.2 * gt.etaCritQuadraticHb09, `ceiling ${r.etaCeiling / gt.etaCritQuadraticHb09} x quadratic`);
    assert.ok(Math.abs(r.bracketHalfWidth - (1.2 * gt.etaCritQuadraticHb09) / 64) < 1e-15);
  });

  it('committed raw-phase2.json carries the pre-registered predictions with their recorded outcomes', () => {
    const raw = JSON.parse(fs.readFileSync(path.join(repoRoot, 'evidence/mean-square-boundary/raw-phase2.json'), 'utf8')) as Phase2Report;
    assert.equal(raw.networks.length, 2);
    assert.deepEqual(raw.networks.map((c) => c.numParams), [3610, 12826]);
    for (const c of raw.networks) {
      assert.ok(c.traceH > c.lambda1 && c.traceOverLambda1 === c.traceH / c.lambda1);
      assert.ok(Math.abs(c.noiseRankRatio - (19.0 * c.traceH) / (16.0 * c.lambda1)) < 1e-9 * c.noiseRankRatio);
      assert.ok(c.etaTraceBracketLower >= c.etaSandwichLower * (1.0 - 1e-9) && c.etaTraceBracketUpper <= c.etaSandwichUpper * (1.0 + 1e-9));
      assert.ok(c.empiricalCeilingMedian > 0.0 && c.empiricalCeilingMedian < 1.1 * c.etaCritQuadratic);
      assert.equal(c.fdHvpSampleGradEquivalents, 1920);
      assert.ok(c.fdHvpLambda1RelErr < 1e-5);
    }
    const EXPECTED_PREDICTIONS: Readonly<Record<string, boolean>> = { q1: true, q2: true, q3: false, q4: true };
    assert.deepEqual(Object.keys(raw.predictions), Object.keys(EXPECTED_PREDICTIONS));
    for (const [k, v] of Object.entries(EXPECTED_PREDICTIONS)) {
      assert.equal(raw.predictions[k], v, `prediction ${k} outcome recorded in raw-phase2.json`);
      assert.ok(typeof raw.predictionTexts[k] === 'string' && raw.predictionTexts[k]!.length > 40);
    }
  });
});
