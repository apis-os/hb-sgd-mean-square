/**
 * @file labs/mean-square-boundary/run-phase5-aggregate.ts
 * @description Collects the 48 run records of Phase 5 (`runPhase5Aggregate`; argument: the directory of
 * `eta<eta>-mu<mu>-seed<seed>.json` files, default `evidence/mean-square-boundary/phase5-runs`) and writes
 * `evidence/mean-square-boundary/raw-phase5.json` with the decisions of `evaluatePhase5Predictions`, the list of run files with
 * their sha256, and the provenance passed in `provenance.json` of that directory when present (commit, package manifest).
 * Refuses to write unless all 48 registered `(eta, mu_train, seed)` are present. Phase 5b (README, "Phase 5b"): `seeds`
 * 1401-1403 and the output `raw-phase5b.json`, which adds `phase5b`, the decisions of `evaluatePhase5bPredictions` (the
 * Phase 5 decisions stay in the file for comparison). CLI: `[dir] [seeds, comma-separated] [output file name]`.
 * @cold
 */
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { evaluatePhase5Predictions } from './evaluate-phase5-predictions.ts';
import { evaluatePhase5bPredictions } from './evaluate-phase5b-predictions.ts';

export function runPhase5Aggregate(dir: string, seeds: readonly number[] = [1, 2, 3], outName = 'raw-phase5.json'): Record<string, unknown> {
  const etas = [0.01, 0.03, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5];
  const runs: Array<Parameters<typeof evaluatePhase5Predictions>[0][number]> = [];
  const files: Array<{ file: string; sha256: string }> = [];
  for (const mu of [0, 1e-3]) {
    for (const eta of etas) {
      for (const seed of seeds) {
        const f = path.join(dir, `eta${eta}-mu${mu}-seed${seed}.json`);
        if (!fs.existsSync(f)) throw new Error(`missing run ${f}`);
        const text = fs.readFileSync(f, 'utf8');
        files.push({ file: path.basename(f), sha256: crypto.createHash('sha256').update(text).digest('hex') });
        runs.push(JSON.parse(text));
      }
    }
  }
  const provPath = path.join(dir, 'provenance.json');
  const provenance = fs.existsSync(provPath) ? JSON.parse(fs.readFileSync(provPath, 'utf8')) : null;
  const decisions = evaluatePhase5Predictions(runs);
  const phase5b = outName === 'raw-phase5b.json' ? { phase5b: evaluatePhase5bPredictions(runs as never, decisions.checks as { c4: boolean; c6: boolean }) } : {};
  const labId = outName === 'raw-phase5b.json' ? 'mean-square-boundary-phase5b' : 'mean-square-boundary-phase5';
  const report = { labId, date: new Date().toISOString(), nodeVersion: process.version, provenance, files, ...decisions, ...phase5b };
  const here = path.dirname(fileURLToPath(import.meta.url));
  fs.writeFileSync(path.resolve(here, '..', '..', 'evidence', 'mean-square-boundary', outName), `${JSON.stringify(report, null, 2)}\n`);
  return report;
}

if (process.argv[1] !== undefined && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const seeds = process.argv[3] === undefined ? [1, 2, 3] : process.argv[3].split(',').map((x) => Number.parseInt(x, 10));
  const r = runPhase5Aggregate(process.argv[2] ?? path.resolve(here, '..', '..', 'evidence', 'mean-square-boundary', 'phase5-runs'), seeds, process.argv[4] ?? 'raw-phase5.json');
  process.stdout.write(`${JSON.stringify({ predictions: r.predictions, checks: r.checks, perMuTrain: r.perMuTrain, pairedVerdict: r.pairedVerdict })}\n`);
}
