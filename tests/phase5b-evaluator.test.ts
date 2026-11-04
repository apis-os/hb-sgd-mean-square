/**
 * @file tests/phase5b-evaluator.test.ts
 * @description The Phase 5b evaluator on the Phase 5 records relabelled to seeds 1401-1403 (the predictions were written from
 * these data, so all six hold and `c5b` lists the 7 sides of amendment 4), and one mutation per rule: a missing run, a
 * saturated checkpoint, a solver residual, a certificate contradiction, a failed `c6`, a divergent run; and the `c6` seed rule.
 * @cold
 */
import fs from 'node:fs';
import { describe, expect, it } from 'vitest';
import { evaluatePhase5Predictions } from '../labs/mean-square-boundary/evaluate-phase5-predictions.ts';
import { evaluatePhase5bPredictions } from '../labs/mean-square-boundary/evaluate-phase5b-predictions.ts';
import { isPhase5C6Seed } from '../labs/mean-square-boundary/is-phase5-c6-seed.ts';

type Rec = Record<string, unknown> & { eta: number; muTrain: number; seed: number; divergedAt: number | null; checkpoints: Array<Record<string, unknown>>; finalPoint: Record<string, unknown> | null };
const dir = new URL('../evidence/mean-square-boundary/phase5-runs/', import.meta.url);
const runs: Rec[] = [];
for (const mu of [0, 1e-3]) for (const eta of [0.01, 0.03, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]) for (const seed of [1, 2, 3]) {
  const r = JSON.parse(fs.readFileSync(new URL(`eta${eta}-mu${mu}-seed${seed}.json`, dir), 'utf8')) as Rec;
  runs.push({ ...r, seed: seed + 1400 });
}
const checks = evaluatePhase5Predictions(runs as never).checks as { c4: boolean; c6: boolean };
type Out = Record<string, { holds: boolean; suspended: boolean; reasons: string[] }> & { replication: boolean; gridComplete: boolean; c5b: boolean; c5bUndecidedSides: unknown[]; certificateContradictions: unknown[] };
const evalOn = (rs: Rec[], ch = checks): Out => evaluatePhase5bPredictions(rs as never, ch) as unknown as Out;
const replaceRun = (eta: number, mu: number, seed: number, edit: (r: Rec) => void): Rec[] => runs.map((r) => {
  if (r.eta !== eta || r.muTrain !== mu || r.seed !== seed) return r;
  const c = structuredClone(r);
  edit(c);
  return c;
});

describe('Phase 5b evaluator', () => {
  it('holds on the data the predictions were written from, and c5b lists the 7 undecided sides', () => {
    expect(checks.c4).toBe(true);
    expect(checks.c6).toBe(true);
    const out = evalOn(runs);
    expect(out.gridComplete).toBe(true);
    for (const q of ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']) {
      expect(out[q]!.holds, `${q}: ${out[q]!.reasons.slice(0, 3).join('; ')}`).toBe(true);
      expect(out[q]!.suspended).toBe(false);
    }
    expect(out.replication).toBe(true);
    expect(out.c5b).toBe(false);
    expect(out.c5bUndecidedSides.length).toBe(7);
    expect(out.certificateContradictions.length).toBe(0);
  });

  it('each rule moves its prediction', () => {
    expect(evalOn(runs.slice(1)).gridComplete).toBe(false);
    const sat = evalOn(replaceRun(0.2, 0, 1401, (r) => { (r.checkpoints[60]!.hiddenUnits as Record<string, number>).saturatedHiddenUnits1e4 = 1; }));
    expect(sat.q3!.holds).toBe(false);
    expect(sat.q1!.holds).toBe(true);
    const res = evalOn(replaceRun(0.3, 1e-3, 1402, (r) => { (r.checkpoints[70]!.absH as Record<string, number>).lanczosRelResidual = 1e-9; }));
    expect(res.q1!.suspended && res.q2!.suspended).toBe(true);
    expect(res.replication).toBe(false);
    const contra = evalOn(replaceRun(0.1, 0, 1403, (r) => { ((r.finalPoint!.operatorBelow as Record<string, Record<string, boolean>>).iterateCertificate)!.certifiedAboveOne = true; }));
    expect(contra.certificateContradictions.length).toBe(1);
    expect(contra.q4!.suspended && !contra.q5!.suspended).toBe(true);
    const noC6 = evalOn(runs, { c4: true, c6: false });
    expect(noC6.q5!.suspended && !noC6.q6!.suspended).toBe(true);
    const div = evalOn(replaceRun(0.4, 0, 1401, (r) => { r.divergedAt = 15000; }));
    expect(div.q5!.holds || div.q6!.holds).toBe(false);
  });

  it('carries c6 on the first seed of each phase only', () => {
    expect([1, 1401].map(isPhase5C6Seed)).toEqual([true, true]);
    expect([0, 2, 3, 97, 1402, 1403].map(isPhase5C6Seed).some(Boolean)).toBe(false);
  });
});
