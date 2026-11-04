/**
 * @file tests/phase4-certified-bracket-p33.test.ts
 * @description Checks the committed `phase4-certified-bracket-p33.json` written by `runPhase4CertifiedBracketP33`: 16 cells,
 * each bracket contains the stored and the dense threshold, `eta_low / eta_up >= 0.9993`, and both offsets `(1 -+ 1e-3) eta*`
 * of check `t1` lie outside the bracket; the stored thresholds equal those of `raw-phase4.json`.
 */
import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

type Row = { beta: number; tetherGrid: number; etaLow: number; etaUp: number; etaThmG6Stored: number; etaDense: number; witnessPositiveDefinite: boolean };
const evid = path.resolve('evidence', 'mean-square-boundary');

describe('Phase 4 P = 33 certified brackets', () => {
  const r = JSON.parse(fs.readFileSync(path.join(evid, 'phase4-certified-bracket-p33.json'), 'utf8')) as { allInside: boolean; rows: Row[] };
  const raw = JSON.parse(fs.readFileSync(path.join(evid, 'raw-phase4.json'), 'utf8')) as { networks: Array<{ networkId: string; thresholds: Array<{ beta: number; tetherGrid: number; etaThmG6: number }> }> };
  const cells = raw.networks.find((n) => n.networkId === 'N33_mlp')!.thresholds;

  it('every bracket contains the stored and dense thresholds and excludes both t1 offsets', () => {
    expect(r.rows.length).toBe(16);
    for (const x of r.rows) {
      expect(x.witnessPositiveDefinite).toBe(true);
      expect(x.etaLow <= x.etaThmG6Stored && x.etaThmG6Stored <= x.etaUp).toBe(true);
      expect(x.etaLow <= x.etaDense && x.etaDense <= x.etaUp).toBe(true);
      expect(x.etaLow / x.etaUp).toBeGreaterThanOrEqual(0.9993);
      expect((1 - 1e-3) * x.etaThmG6Stored < x.etaLow && (1 + 1e-3) * x.etaThmG6Stored > x.etaUp).toBe(true);
    }
    expect(r.allInside).toBe(true);
  });

  it('the stored thresholds are those of raw-phase4.json', () => {
    for (const x of r.rows) expect(x.etaThmG6Stored).toBe(cells.find((c) => c.beta === x.beta && c.tetherGrid === x.tetherGrid)!.etaThmG6);
  });
});
