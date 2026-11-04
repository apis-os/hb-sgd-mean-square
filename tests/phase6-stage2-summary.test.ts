import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { summarizePhase6Stage2 } from '../labs/mean-square-boundary/summarize-phase6-stage2.ts';

type Rec = Parameters<typeof summarizePhase6Stage2>[0][number];
type Sum = {
  checkpoints: number;
  matchesList: boolean;
  invalidAgainstReference: number;
  readingAll: Record<string, number>;
  runsAtSurrogateEdge: number;
  decision: { validity: string; adoption: string };
  expectations: { e16_2: { holds: boolean }; e16_3: { holds: boolean }; e16_4: { holds: boolean } };
};

function cp(step: number, eta: number, low: number, up: number, o: { valid?: boolean | null; top64?: number } = {}): Rec['checkpoints'][number] {
  const reading = 0.8 * eta <= low && up <= 1.25 * eta ? 'inside' : up < 0.8 * eta ? 'below' : low > 1.25 * eta ? 'above' : 'undecided';
  return {
    step,
    rank: 640,
    rankCompressed: false,
    meanResidualFrobeniusOverMinCurvature: 1e-10,
    dualResidualOverTop: 1e-15,
    bracket: { etaLow: low, etaUp: up, lowOverUp: low / up, seconds: 1 },
    reference: { etaLow: low, etaUp: up, lowOverUp: low / up, seconds: 1 },
    upOverTop64: up / (o.top64 ?? up / 0.97),
    lowOverEta: low / eta,
    upOverEta: up / eta,
    reading,
    validAgainstReference: o.valid === undefined ? true : o.valid,
  };
}

function rec(run: string, eta: number, step: number, low: number, up: number, o: { valid?: boolean | null; top64?: number; mismatchAt?: number } = {}): Rec {
  return { run, network: 3610, activation: 'tanh', eta, divergedAt: null, mismatchAt: o.mismatchAt ?? null, checkpoints: [cp(step, eta, low, up, o)] };
}

const steps = [10000, 11000];
const expected = [{ run: 'a', steps }, { run: 'b', steps }];

describe('summarizePhase6Stage2', () => {
  it('groups per-checkpoint files by run and applies the plan rules', () => {
    const recs = [rec('a', 0.05, 11000, 0.049, 0.0495), rec('a', 0.05, 10000, 0.049, 0.0495), rec('b', 0.001, 10000, 0.01, 0.0101), rec('b', 0.001, 11000, 0.01, 0.0101)];
    const s = summarizePhase6Stage2(recs, expected) as Sum;
    expect(s.matchesList).toBe(true);
    expect(s.checkpoints).toBe(4);
    expect(s.readingAll).toEqual({ inside: 2, below: 0, above: 2, undecided: 0 });
    expect(s.runsAtSurrogateEdge).toBe(1);
    expect(s.decision).toEqual({ validity: 'no violation', adoption: 'adopted' });
    expect(s.expectations.e16_3.holds).toBe(true);
    expect(s.expectations.e16_4.holds).toBe(true);
  });

  it('refuses a decision on an incomplete or mismatched set, and rejects a wrong-side bracket', () => {
    const ok = (r: string, e: number, st: number) => rec(r, e, st, 0.049, 0.0495);
    expect((summarizePhase6Stage2([ok('a', 0.05, 10000), ok('a', 0.05, 11000), ok('b', 0.05, 10000)], expected) as Sum).decision.validity).toBe('incomplete: no decision');
    expect((summarizePhase6Stage2([ok('a', 0.05, 10000), ok('a', 0.05, 11000), ok('b', 0.05, 10000), rec('b', 0.05, 11000, 0.049, 0.0495, { mismatchAt: 11000 })], expected) as Sum).decision.validity).toBe('incomplete: no decision');
    const dup = summarizePhase6Stage2([ok('a', 0.05, 10000), ok('a', 0.05, 10000), ok('b', 0.05, 10000), ok('b', 0.05, 11000)], expected) as Sum;
    expect(dup.matchesList).toBe(false);
    const bad = summarizePhase6Stage2([ok('a', 0.05, 10000), ok('a', 0.05, 11000), ok('b', 0.05, 10000), rec('b', 0.05, 11000, 0.049, 0.0495, { valid: false })], expected) as Sum;
    expect(bad.invalidAgainstReference).toBe(1);
    expect(bad.decision).toEqual({ validity: 'violation: method rejected', adoption: 'not adopted' });
    const noRef = summarizePhase6Stage2([ok('a', 0.05, 10000), ok('a', 0.05, 11000), ok('b', 0.05, 10000), rec('b', 0.05, 11000, 0.049, 0.0495, { valid: null })], expected) as Sum;
    expect(noRef.decision.validity).toBe('violation: method rejected');
  });

  it('adoption needs eta_low / eta_up >= 0.98 at 90 % of the checkpoints', () => {
    const wide = (r: string, st: number) => rec(r, 0.05, st, 0.045, 0.0495);
    const s = summarizePhase6Stage2([wide('a', 10000), rec('a', 0.05, 11000, 0.049, 0.0495), rec('b', 0.05, 10000, 0.049, 0.0495), rec('b', 0.05, 11000, 0.049, 0.0495)], expected) as Sum;
    expect(s.decision).toEqual({ validity: 'no violation', adoption: 'not adopted' });
  });

  it('reads the expectations e16-2, e16-3 and e16-4', () => {
    // e16-3 fails when a run with eta >= 0.01 is not inside at most checkpoints; e16-4 fails when a run is below at most.
    const above = summarizePhase6Stage2([rec('a', 0.01, 10000, 0.02, 0.0201), rec('a', 0.01, 11000, 0.02, 0.0201), rec('b', 0.05, 10000, 0.049, 0.0495), rec('b', 0.05, 11000, 0.049, 0.0495)], expected) as Sum;
    expect(above.expectations.e16_3.holds).toBe(false);
    expect(above.expectations.e16_4.holds).toBe(true);
    const below = summarizePhase6Stage2([rec('a', 0.05, 10000, 0.02, 0.0201), rec('a', 0.05, 11000, 0.02, 0.0201), rec('b', 0.05, 10000, 0.049, 0.0495), rec('b', 0.05, 11000, 0.049, 0.0495)], expected) as Sum;
    expect(below.expectations.e16_4.holds).toBe(false);
    const top64 = summarizePhase6Stage2([rec('a', 0.05, 10000, 0.049, 0.0495, { top64: 0.06 }), rec('a', 0.05, 11000, 0.049, 0.0495), rec('b', 0.05, 10000, 0.049, 0.0495), rec('b', 0.05, 11000, 0.049, 0.0495)], expected) as Sum;
    expect(top64.expectations.e16_2.holds).toBe(false);
  });

  it('regenerates the committed stage 2 summary from the committed records', () => {
    const evid = 'evidence/mean-square-boundary';
    const dir = path.join(evid, 'phase6-stage2');
    const files = fs.readdirSync(dir).filter((f) => f.endsWith('.json') && f !== 'PROVENANCE.json').sort();
    const records = files.map((f) => JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8')) as Rec);
    const list = JSON.parse(fs.readFileSync(path.join(evid, 'phase6-checkpoints.json'), 'utf8')) as { stage2: { runs: Array<{ run: string; steps: number[] }> } };
    const committed = JSON.parse(fs.readFileSync(path.join(evid, 'phase6-stage2-summary.json'), 'utf8')) as Record<string, unknown>;
    delete committed.labId;
    delete committed.date;
    expect(JSON.parse(JSON.stringify(summarizePhase6Stage2(records, list.stage2.runs)))).toEqual(committed);
  });
});
