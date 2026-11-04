/**
 * @file labs/mean-square-boundary/summarize-phase6-stage2.ts
 * @description Summary of stage 2 of Phase 6 (`summarizePhase6Stage2`) from the per-checkpoint files of `runPhase6Stage2Run`
 * (one `<run>-step<k>.json` per checkpoint, each holding one record of `buildPhase6Stage2CheckpointRecord`), with the rules of the
 * plan (section 5) as written before the stage 2 batch: (1) validity, every bracket must overlap the reference bracket with no
 * tolerance (`validAgainstReference`; a `false` or a missing reference rejects the method); (2) adoption, `eta_low / eta_up >= 0.98`
 * at 90 % of the listed checkpoints (an undefined `eta_low` counts as a miss); (3) the reading per checkpoint (`inside`, `below`,
 * `above`, `undecided` against `[0.8 eta, 1.25 eta]`, as stored by the record builder); (4) per run, the share of each reading, a run
 * being "at the surrogate's edge" when at least 80 % of its checkpoints read `inside`. Also the expectations `e16-2` (`eta_up /
 * eta*_GN,64 >= 0.95` at 90 % of the checkpoints), `e16-3` (runs with `eta >= 0.01` read `inside` at more than half of their
 * checkpoints, runs with `eta <= 0.003` read `above` at every checkpoint) and `e16-4` (no run reads `below` at more than half of its
 * checkpoints), the agreement of the two upper bounds, the relative width `(eta_up - eta_low) / eta_up`, the factor residuals, the
 * row-space ranks and the per-run window medians of `eta_low / eta` and `eta_up / eta`. A decision needs the records to match the
 * committed list `phase6-checkpoints.json` (field `stage2`) exactly, every run regenerated without mismatch and no divergence.
 * Pure over its arguments; the CLI reads the directory and the list: `node summarize-phase6-stage2.ts [dir] [out]`.
 * @cold
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

type Side = { etaLow: number | null; etaUp: number; lowOverUp: number | null; seconds: number } | null;

type Stage2Checkpoint = {
  step: number;
  rank: number;
  rankCompressed: boolean;
  meanResidualFrobeniusOverMinCurvature: number;
  dualResidualOverTop: number;
  bracket: Side;
  reference: Side;
  upOverTop64: number | null;
  lowOverEta: number | null;
  upOverEta: number | null;
  reading: string;
  validAgainstReference: boolean | null;
};

type Stage2Record = { run: string; network: number; activation: string; eta: number; divergedAt: number | null; mismatchAt: number | null; checkpoints: Stage2Checkpoint[] };

function quantiles(xs: number[]): Record<string, number> | null {
  if (xs.length === 0) return null;
  const s = xs.slice().sort((a, b) => a - b);
  const q = (p: number): number => s[Math.min(s.length - 1, Math.floor(p * (s.length - 1) + 0.5))]!;
  return { n: s.length, min: s[0]!, p10: q(0.1), median: q(0.5), p90: q(0.9), max: s[s.length - 1]! };
}

function median(xs: number[]): number | null {
  const m = quantiles(xs);
  return m === null ? null : m.median!;
}

export function summarizePhase6Stage2(records: readonly Stage2Record[], expected: ReadonlyArray<{ readonly run: string; readonly steps: readonly number[] }>): Record<string, unknown> {
  // Group the per-checkpoint files by run, in step order.
  const byRun = new Map<string, { rec: Stage2Record; cps: Stage2Checkpoint[]; mismatch: boolean; diverged: boolean }>();
  let duplicate = false;
  for (const r of records) {
    const g = byRun.get(r.run) ?? { rec: r, cps: [], mismatch: false, diverged: false };
    for (const c of r.checkpoints) {
      if (g.cps.some((x) => x.step === c.step)) duplicate = true;
      g.cps.push(c);
    }
    if (r.mismatchAt !== null) g.mismatch = true;
    if (r.divergedAt !== null) g.diverged = true;
    byRun.set(r.run, g);
  }
  for (const g of byRun.values()) g.cps.sort((a, b) => a.step - b.step);
  let matchesList = !duplicate && byRun.size === expected.length && expected.length > 0;
  for (const e of expected) {
    const g = byRun.get(e.run);
    if (g === undefined || g.cps.length !== e.steps.length) { matchesList = false; break; }
    for (let i = 0; i < e.steps.length; i++) if (g.cps[i]!.step !== e.steps[i]) { matchesList = false; break; }
  }
  const lowOverUp: number[] = [];
  const width: number[] = [];
  const upAgreement: number[] = [];
  const upOverTop64: number[] = [];
  const bracketSeconds: number[] = [];
  let total = 0, narrow = 0, undefinedLow = 0, invalid = 0, missingReference = 0, compressed = 0, top64AtLeast095 = 0, top64Total = 0;
  let maxMeanResidual = 0, maxDualResidual = 0, minRank = Infinity, maxRank = 0;
  const readingAll: Record<string, number> = { inside: 0, below: 0, above: 0, undecided: 0 };
  const readingEta001: Record<string, number> = { inside: 0, below: 0, above: 0, undecided: 0 };
  const runs: Array<Record<string, unknown>> = [];
  let mismatches = 0, divergences = 0, edgeRuns = 0;
  let e163 = true, e164 = true;
  for (const name of [...byRun.keys()].sort()) {
    const g = byRun.get(name)!;
    if (g.mismatch) mismatches++;
    if (g.diverged) divergences++;
    const eta = g.rec.eta;
    const reading: Record<string, number> = { inside: 0, below: 0, above: 0, undecided: 0 };
    const lowEta: number[] = [];
    const upEta: number[] = [];
    for (const c of g.cps) {
      total++;
      minRank = Math.min(minRank, c.rank);
      maxRank = Math.max(maxRank, c.rank);
      if (c.rankCompressed) compressed++;
      maxMeanResidual = Math.max(maxMeanResidual, c.meanResidualFrobeniusOverMinCurvature);
      maxDualResidual = Math.max(maxDualResidual, c.dualResidualOverTop);
      if (c.validAgainstReference === false) invalid++;
      if (c.validAgainstReference === null || c.reference === null) missingReference++;
      reading[c.reading] = (reading[c.reading] ?? 0) + 1;
      readingAll[c.reading] = (readingAll[c.reading] ?? 0) + 1;
      if (eta >= 0.01) readingEta001[c.reading] = (readingEta001[c.reading] ?? 0) + 1;
      if (c.upOverTop64 !== null) { top64Total++; upOverTop64.push(c.upOverTop64); if (c.upOverTop64 >= 0.95) top64AtLeast095++; }
      if (c.lowOverEta !== null) lowEta.push(c.lowOverEta);
      if (c.upOverEta !== null) upEta.push(c.upOverEta);
      const b = c.bracket;
      if (b === null) { undefinedLow++; continue; }
      bracketSeconds.push(b.seconds);
      if (c.reference !== null) upAgreement.push(Math.abs(b.etaUp / c.reference.etaUp - 1));
      if (b.etaLow === null || b.lowOverUp === null) { undefinedLow++; continue; }
      lowOverUp.push(b.lowOverUp);
      width.push((b.etaUp - b.etaLow) / b.etaUp);
      if (b.lowOverUp >= 0.98) narrow++;
    }
    const n = g.cps.length;
    const insideShare = n === 0 ? 0 : reading.inside! / n;
    if (insideShare >= 0.8) edgeRuns++;
    if (eta >= 0.01 && !(reading.inside! > n / 2)) e163 = false;
    if (eta <= 0.003 && reading.above !== n) e163 = false;
    if (reading.below! > n / 2) e164 = false;
    runs.push({ run: name, network: g.rec.network, activation: g.rec.activation, eta, checkpoints: n, reading, insideShare, atSurrogateEdge: insideShare >= 0.8, medianLowOverEta: median(lowEta), medianUpOverEta: median(upEta), minLowOverEta: lowEta.length === 0 ? null : Math.min(...lowEta), maxUpOverEta: upEta.length === 0 ? null : Math.max(...upEta) });
  }
  const complete = matchesList && mismatches === 0 && divergences === 0 && total > 0;
  const validity = !complete ? 'incomplete: no decision' : invalid === 0 && missingReference === 0 ? 'no violation' : 'violation: method rejected';
  return {
    runs: byRun.size,
    expectedRuns: expected.length,
    checkpoints: total,
    matchesList,
    regenerationMismatches: mismatches,
    divergences,
    rank: { min: minRank, max: maxRank, compressed },
    invalidAgainstReference: invalid,
    missingReference,
    undefinedLow,
    fractionLowOverUpAtLeast098: total === 0 ? 0 : narrow / total,
    lowOverUp: quantiles(lowOverUp),
    relativeWidth: quantiles(width),
    upperBoundsRelativeDisagreement: quantiles(upAgreement),
    upOverTop64: quantiles(upOverTop64),
    maxMeanResidualFrobeniusOverMinCurvature: maxMeanResidual,
    maxDualResidualOverTop: maxDualResidual,
    bracketSeconds: quantiles(bracketSeconds),
    readingAll,
    readingEtaAtLeast001: readingEta001,
    runsAtSurrogateEdge: edgeRuns,
    decision: {
      validity,
      adoption: validity === 'no violation' && narrow >= 0.9 * total ? 'adopted' : 'not adopted',
    },
    expectations: {
      e16_2: { fractionUpOverTop64AtLeast095: top64Total === 0 ? null : top64AtLeast095 / top64Total, holds: top64Total > 0 && top64AtLeast095 >= 0.9 * top64Total },
      e16_3: { holds: complete && e163 },
      e16_4: { holds: complete && e164 },
    },
    perRun: runs,
  };
}

if (process.argv[1] !== undefined && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const evid = path.resolve(here, '..', '..', 'evidence', 'mean-square-boundary');
  const dir = process.argv[2] ?? path.join(evid, 'phase6-stage2');
  const out = process.argv[3] ?? path.join(evid, 'phase6-stage2-summary.json');
  const files = fs.readdirSync(dir).filter((f) => f.endsWith('.json') && f !== 'PROVENANCE.json').sort();
  const records = files.map((f) => JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8')) as Stage2Record);
  const list = JSON.parse(fs.readFileSync(path.join(evid, 'phase6-checkpoints.json'), 'utf8')) as { stage2: { runs: Array<{ run: string; steps: number[] }> } };
  const s = summarizePhase6Stage2(records, list.stage2.runs);
  fs.writeFileSync(out, `${JSON.stringify({ labId: 'mean-square-boundary-phase6-stage2-summary', date: new Date().toISOString(), ...s }, null, 1)}\n`);
  process.stdout.write(`written ${out}\n`);
}
