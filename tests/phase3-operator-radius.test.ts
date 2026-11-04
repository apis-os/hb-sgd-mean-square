/**
 * @file tests/phase3-operator-radius.test.ts
 * @description Phase 3: the closed-form second-moment operator against its exact expectation by enumeration (non-symmetric input, so the transpose in `V(X)_jk = lambda_j lambda_k X_kj / B` is tested), a reduced run of the operator-radius check of Theorem G, and the committed `raw-phase3.json`.
 */
import { strict as assert } from 'node:assert';
import fs from 'node:fs';
import path from 'node:path';
import { describe, it } from 'vitest';
import { applyHeavyBallSecondMomentOperator } from '../labs/mean-square-boundary/apply-heavy-ball-second-moment-operator.ts';
import { evaluatePhase3OperatorRadiusCells } from '../labs/mean-square-boundary/evaluate-phase3-operator-radius-cells.ts';
import type { Phase3Report } from '../labs/mean-square-boundary/phase3-types.ts';

function exactOperator(
  lambdas: number[],
  beta: number,
  eta: number,
  batchB: number,
  values: number[],
  probs: number[],
  q: Float64Array,
): Float64Array {
  const p = lambdas.length;
  const n = 2 * p;
  const out = new Float64Array(n * n);
  // enumerate omega for the B draws: (values^p)^B outcomes
  const perDraw: { w: number[]; pr: number }[] = [];
  const total = values.length ** p;
  for (let code = 0; code < total; code++) {
    let c = code;
    const w: number[] = [];
    let pr = 1;
    for (let k = 0; k < p; k++) {
      const idx = c % values.length;
      c = Math.floor(c / values.length);
      w.push(values[idx]!);
      pr *= probs[idx]!;
    }
    perDraw.push({ w, pr });
  }
  const recurse = (b: number, hb: number[][], pr: number): void => {
    if (b === batchB) {
      const f: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
      for (let i = 0; i < p; i++) {
        for (let j = 0; j < p; j++) f[i]![j] = (i === j ? 1 + beta : 0) - eta * hb[i]![j]!;
        f[i]![p + i] = -beta;
        f[p + i]![i] = 1;
      }
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          let s = 0;
          for (let a = 0; a < n; a++) {
            if (f[i]![a] === 0) continue;
            for (let c2 = 0; c2 < n; c2++) s += f[i]![a]! * q[a * n + c2]! * f[j]![c2]!;
          }
          out[i * n + j] = out[i * n + j]! + pr * s;
        }
      }
      return;
    }
    for (const d of perDraw) {
      const next = hb.map((row) => row.slice());
      for (let i = 0; i < p; i++) {
        for (let j = 0; j < p; j++) next[i]![j] = next[i]![j]! + (Math.sqrt(lambdas[i]! * lambdas[j]!) * d.w[i]! * d.w[j]!) / batchB;
      }
      recurse(b + 1, next, pr * d.pr);
    }
  };
  recurse(0, Array.from({ length: p }, () => new Array<number>(p).fill(0)), 1);
  return out;
}

describe('Phase 3 — exact second-moment operator against the secular root of Theorem G', () => {
  it('matches the exact expectation on a non-symmetric input (three-point and Rademacher features)', () => {
    const lambdas = [1.0, 0.45, 0.2];
    const p = lambdas.length;
    const n = 2 * p;
    const q = new Float64Array(n * n);
    let s = 0.37;
    for (let i = 0; i < n * n; i++) {
      s = (s * 9301 + 0.49297) % 1;
      q[i] = s - 0.5;
    }
    const cases: { values: number[]; probs: number[]; kappa: number; batchB: number }[] = [
      { values: [-2, 0, 2], probs: [0.125, 0.75, 0.125], kappa: 1, batchB: 1 },
      { values: [-1, 1], probs: [0.5, 0.5], kappa: -2, batchB: 2 },
    ];
    for (const c of cases) {
      const exact = exactOperator(lambdas, 0.7, 0.3, c.batchB, c.values, c.probs, q);
      const got = new Float64Array(n * n);
      applyHeavyBallSecondMomentOperator(new Float64Array(lambdas), 0.7, 0.3, c.batchB, c.kappa, q, got);
      let maxErr = 0;
      for (let i = 0; i < n * n; i++) maxErr = Math.max(maxErr, Math.abs(got[i]! - exact[i]!));
      assert.ok(maxErr < 1e-12, `max error ${maxErr} for kappa ${c.kappa}`);
    }
  });

  it('reduced run at P = 16: the radius crosses 1 at the secular root and the antisymmetric part stays below', () => {
    const r = evaluatePhase3OperatorRadiusCells([16]);
    assert.equal(r.cells.length, 12);
    for (const c of r.cells) {
      assert.ok(c.allConverged, c.cellId);
      assert.ok(Math.abs(c.radiusAtRoot - 1.0) < 1e-10, `${c.cellId} rho(eta*) = ${c.radiusAtRoot}`);
      assert.ok(c.radiusBelow < 1.0 && c.radiusAbove > 1.0, c.cellId);
      assert.ok(c.radiusAntisymmetricAtRoot < 1.0, c.cellId);
    }
    assert.equal(r.predictions.q1RadiusCrossesOneAtSecularRootEveryCell, true);
    assert.equal(r.predictions.q2CrossingWithin1e8RelativeEveryCell, true);
    assert.equal(r.predictions.q3AntisymmetricRadiusBelowOneAtRootEveryCell, true);
  });

  it('validates the committed raw-phase3.json', () => {
    const raw = JSON.parse(
      fs.readFileSync(path.resolve(__dirname, '../evidence/mean-square-boundary/raw-phase3.json'), 'utf8'),
    ) as Phase3Report;
    assert.equal(raw.labId, 'mean-square-boundary-phase3');
    assert.equal(raw.cells.length, 36);
    let q1 = true;
    let q2 = true;
    let q3 = true;
    for (const c of raw.cells) {
      if (!(c.radiusBelow < 1.0 && c.radiusAbove > 1.0)) q1 = false;
      if (!(c.crossingRelErr < 1e-8)) q2 = false;
      if (!(c.radiusAntisymmetricAtRoot < 1.0)) q3 = false;
    }
    assert.equal(raw.predictions.q1RadiusCrossesOneAtSecularRootEveryCell, q1);
    assert.equal(raw.predictions.q2CrossingWithin1e8RelativeEveryCell, q2);
    assert.equal(raw.predictions.q3AntisymmetricRadiusBelowOneAtRootEveryCell, q3);
    assert.ok(raw.cells.some((c) => c.numParams === 1024));
  });
});
