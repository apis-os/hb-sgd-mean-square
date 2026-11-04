/**
 * @file tests/phase4-5-raw-files.test.ts
 * @description Checks the committed raw files of Phases 4, 5 and 5b. Phase 4: the registered predictions t1, t2, t5, t6
 * and the implementation checks c2, c3 recomputed from the cell records of `raw-phase4.json`, and the values the draft
 * quotes. Phases 5 and 5b: the decisions of `evaluatePhase5Predictions` and `evaluatePhase5bPredictions` recomputed from the
 * 48 committed run records equal the decisions stored in `raw-phase5.json` and `raw-phase5b.json`, and the run-file hashes
 * stored there are those of the committed records.
 */
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { evaluatePhase5Predictions } from '../labs/mean-square-boundary/evaluate-phase5-predictions.ts';
import { evaluatePhase5bPredictions } from '../labs/mean-square-boundary/evaluate-phase5b-predictions.ts';

const evid = path.resolve('evidence', 'mean-square-boundary');
const readJson = (f: string): Record<string, unknown> => JSON.parse(fs.readFileSync(path.join(evid, f), 'utf8')) as Record<string, unknown>;

type Cert = { iteratePositiveDefinite: boolean; certifiedBelowOne: boolean; certifiedAboveOne: boolean };
type OpCheck = { beta: number; tetherGrid: number; radiusBelow: number; radiusAbove: number; certificateBelow?: Cert; certificateAbove?: Cert; fallback?: { certificateBelow: Cert; certificateAbove: Cert } };
type Threshold = { beta: number; tetherGrid: number; mu: number; etaThmG6: number; fractionOfDeterministic: number };
type Network = { networkId: string; lambdaMin: number; negativeEigenvalues: number; collapseSlopes: Array<{ beta: number; slope: number }>; thresholds: Threshold[]; operatorChecks: OpCheck[] };

describe('Phase 4 raw file', () => {
  const raw = readJson('raw-phase4.json') as { predictions: Record<string, boolean>; implementationChecks: Record<string, boolean>; networks: Network[] };
  const n33 = raw.networks.find((n) => n.networkId === 'N33_mlp')!;
  const n3610 = raw.networks.find((n) => n.networkId === 'X1_P3610')!;

  it('t1: every operator cell has radius below 1 at (1 - 1e-3) eta* and above 1 at (1 + 1e-3) eta*', () => {
    const all = raw.networks.flatMap((n) => n.operatorChecks);
    expect(all.length).toBe(17);
    expect(all.every((c) => c.radiusBelow < 1 && c.radiusAbove > 1)).toBe(raw.predictions.t1);
    expect(raw.predictions.t1).toBe(true);
  });

  it('t2: the mean Hessian at P = 3 610 is indefinite, with 1 528 negative eigenvalues', () => {
    expect(n3610.lambdaMin < 0).toBe(raw.predictions.t2);
    expect(n3610.negativeEigenvalues).toBe(1528);
    expect(n3610.lambdaMin).toBeCloseTo(-0.0833, 4);
    expect(n33.negativeEigenvalues).toBe(2);
  });

  it('t5: the flat-mode slope at P = 3 610 is outside [0.4, 1.1], so the registered prediction fails', () => {
    const slope = n3610.collapseSlopes.find((c) => c.beta === 0.9)!.slope;
    const eta = (g: number): number => n3610.thresholds.find((t) => t.beta === 0.9 && t.tetherGrid === g)!.etaThmG6;
    const holds = eta(1 / 128) < eta(1 / 32) && slope >= 0.4 && slope <= 1.1;
    expect(holds).toBe(raw.predictions.t5);
    expect(raw.predictions.t5).toBe(false);
    expect(slope).toBeCloseTo(0.0022, 4);
  });

  it('t6: the fraction of the deterministic threshold is non-decreasing in mu on both networks and for every beta', () => {
    let ok = true;
    for (const n of raw.networks) {
      for (const beta of [...new Set(n.thresholds.map((t) => t.beta))]) {
        const rows = n.thresholds.filter((t) => t.beta === beta).sort((a, b) => a.mu - b.mu);
        for (let i = 1; i < rows.length; i++) if (rows[i]!.fractionOfDeterministic < rows[i - 1]!.fractionOfDeterministic) ok = false;
      }
    }
    expect(ok).toBe(raw.predictions.t6);
  });

  it('c2, c3: the P = 33 cells are decided by their certificates, the P = 3 610 cell at neither offset', () => {
    expect(n33.operatorChecks.every((c) => c.certificateBelow!.certifiedBelowOne && c.certificateAbove!.certifiedAboveOne)).toBe(true);
    const c = n3610.operatorChecks[0]!;
    expect(c.fallback!.certificateBelow.iteratePositiveDefinite || c.fallback!.certificateAbove.iteratePositiveDefinite).toBe(false);
    expect(raw.implementationChecks.c2).toBe(false);
    expect(raw.implementationChecks.c3).toBe(false);
  });

  it('values quoted in the draft (Table of Phase 4)', () => {
    const eta = (n: Network, beta: number, g: number): number => n.thresholds.find((t) => t.beta === beta && t.tetherGrid === g)!.etaThmG6;
    expect(eta(n33, 0.9, 8)).toBeCloseTo(0.19406782, 8);
    expect(eta(n3610, 0.9, 32)).toBeCloseTo(0.017880637, 9);
    expect(eta(n3610, 0.9, 1 / 128)).toBeCloseTo(0.015640812, 9);
  });
});

function loadRuns(dir: string, seeds: readonly number[]): { runs: unknown[]; files: Array<{ file: string; sha256: string }> } {
  const runs: unknown[] = [];
  const files: Array<{ file: string; sha256: string }> = [];
  for (const mu of [0, 1e-3]) {
    for (const eta of [0.01, 0.03, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]) {
      for (const seed of seeds) {
        const file = `eta${eta}-mu${mu}-seed${seed}.json`;
        const text = fs.readFileSync(path.join(evid, dir, file), 'utf8');
        files.push({ file, sha256: crypto.createHash('sha256').update(text).digest('hex') });
        runs.push(JSON.parse(text));
      }
    }
  }
  return { runs, files };
}

const decisionKeys = ['predictions', 'checks', 'perMuTrain', 'pairedVerdict', 'summaries'] as const;

describe('Phase 5 and 5b raw files', () => {
  it('raw-phase5.json holds the decisions recomputed from the 48 committed run records', () => {
    const { runs, files } = loadRuns('phase5-runs', [1, 2, 3]);
    const stored = readJson('raw-phase5.json');
    const decisions = evaluatePhase5Predictions(runs as never) as Record<string, unknown>;
    for (const k of decisionKeys) expect(JSON.parse(JSON.stringify(decisions[k] ?? null))).toEqual(stored[k] ?? null);
    expect(stored.files).toEqual(files);
  });

  it('raw-phase5b.json holds the Phase 5b decisions recomputed from the 48 committed run records', () => {
    const { runs, files } = loadRuns('phase5b-runs', [1401, 1402, 1403]);
    const stored = readJson('raw-phase5b.json');
    const decisions = evaluatePhase5Predictions(runs as never) as Record<string, unknown>;
    for (const k of decisionKeys) expect(JSON.parse(JSON.stringify(decisions[k] ?? null))).toEqual(stored[k] ?? null);
    const p5b = evaluatePhase5bPredictions(runs as never, decisions.checks as { c4: boolean; c6: boolean });
    expect(JSON.parse(JSON.stringify(p5b))).toEqual(stored.phase5b);
    expect(stored.files).toEqual(files);
    const q = stored.phase5b as Record<string, { holds: boolean }>;
    expect(['q1', 'q2', 'q3', 'q4', 'q5', 'q6'].every((k) => q[k]!.holds)).toBe(true);
  });
});
