/**
 * @file labs/mean-square-boundary/summarize-phase6-stage1.ts
 * @description Summary of stage 1 of Phase 6 (`summarizePhase6Stage1`) from the per-run files of `runPhase6Stage1Run`, with the
 * rules of the plan (section 5) as written before the run: (1) validity, any checkpoint with `eta_low > eta*` or `eta* > eta_up`
 * beyond `1e-8` relative rejects the method; (2) adoption, for the Gauss–Newton object at `mu_train > 0`, `eta_low / eta_up >= 0.98` at 90 % of
 * the listed checkpoints (an undefined `eta_low` counts as a miss; `mu_train = 0` is the quotient object and enters no adoption
 * decision, plan section 5, reported as `gaussNewtonMu0`); the `|H|` object is a method check, reported with the same statistics and
 * no adoption decision. An invalid bracket of either still rejects the method under rule (1) (twelfth audit round: the text said they
 * entered no decision). When a checkpoint carries `validDense` (the dense reference of `runPhase6Stage1Run`), validity is
 * decided against it, and the count against the stored threshold is reported beside it (`invalidVsStored`): the first stage 1 run
 * found 94 Gauss–Newton checkpoints outside the bracket of the stored value, all where the stored dense Lanczos had not converged,
 * and none outside against the dense reference. Also: regeneration mismatches, disagreements of the removed-mode sets with the stored records,
 * Lanczos reruns, the quantiles of `eta_low / eta_up` and `eta_up / eta*` per object and per `(mu_train, eta)`, and, for the
 * Gauss–Newton object, the reading of rule 3 per checkpoint (`inside`, `below`, `above`, `undecided` against `[0.8 eta, 1.25 eta]`)
 * over the window (steps 10 000–20 000). A decision needs the records to match the committed list `phase6-checkpoints.json`
 * exactly (run names and checkpoint steps). Pure over its arguments; the CLI reads the directory and the list:
 * `node summarize-phase6-stage1.ts [dir] [out]`.
 * @cold
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

type Obj = { etaLow: number | null; etaUp: number | null; etaStarStored: number; lowOverUp: number | null; upOverStored: number | null; valid: boolean; validDense?: boolean | null; etaStarDense?: number | null; lanczosRerun: boolean; witnessPositiveDefinite: boolean };

function quantiles(xs: number[]): Record<string, number> | null {
  if (xs.length === 0) return null;
  const s = xs.slice().sort((a, b) => a - b);
  const q = (p: number): number => s[Math.min(s.length - 1, Math.floor(p * (s.length - 1) + 0.5))]!;
  return { n: s.length, min: s[0]!, p10: q(0.1), median: q(0.5), p90: q(0.9), max: s[s.length - 1]! };
}

type RunRecord = { run: string; eta: number; muTrain: number; mismatchAt: number | null; storedCheckpoints: number; checkpoints: Array<{ step: number; gaussNewton: Obj & { removedModesAgree: boolean }; absH: Obj & { nullModesAgree: boolean } }> };

export function summarizePhase6Stage1(records: readonly RunRecord[], expected: ReadonlyArray<{ readonly run: string; readonly steps: readonly number[] }>): Record<string, unknown> {
  const objects = ['gaussNewton', 'gaussNewtonMu0', 'absH'] as const;
  const acc: Record<string, { total: number; invalid: Array<Record<string, unknown>>; invalidVsStored: number; undefinedLow: number; narrow: number; reruns: number; witnessFail: number; ratios: number[]; upRatios: number[]; agreeFail: number }> = {};
  for (const o of objects) acc[o] = { total: 0, invalid: [], invalidVsStored: 0, undefinedLow: 0, narrow: 0, reruns: 0, witnessFail: 0, ratios: [], upRatios: [], agreeFail: 0 };
  const perCell: Record<string, { low: number[]; reading: Record<string, number> }> = {};
  const mismatches: Array<{ run: string; step: number }> = [];
  let runs = 0;
  for (const r of records) {
    runs++;
    if (r.mismatchAt !== null) mismatches.push({ run: r.run, step: r.mismatchAt });
    const cell = `mu${r.muTrain}-eta${r.eta}`;
    perCell[cell] ??= { low: [], reading: { inside: 0, below: 0, above: 0, undecided: 0 } };
    for (const c of r.checkpoints) {
      for (const o of objects) {
        if ((o === 'gaussNewton' && r.muTrain === 0) || (o === 'gaussNewtonMu0' && r.muTrain !== 0)) continue;
        const x = o === 'gaussNewtonMu0' ? c.gaussNewton : c[o];
        const a = acc[o]!;
        a.total++;
        if (!x.valid) a.invalidVsStored++;
        const ok = x.validDense === undefined || x.validDense === null ? x.valid : x.validDense;
        if (!ok) a.invalid.push({ run: r.run, step: c.step, etaLow: x.etaLow, etaUp: x.etaUp, etaStar: x.etaStarStored, etaStarDense: x.etaStarDense ?? null });
        if (x.lanczosRerun) a.reruns++;
        if (!x.witnessPositiveDefinite) a.witnessFail++;
        if (x.upOverStored !== null) a.upRatios.push(x.upOverStored);
        if (o === 'absH' ? !(c.absH.nullModesAgree) : !(c.gaussNewton.removedModesAgree)) a.agreeFail++;
        if (x.lowOverUp === null) { a.undefinedLow++; continue; }
        a.ratios.push(x.lowOverUp);
        if (x.lowOverUp >= 0.98) a.narrow++;
      }
      const g = c.gaussNewton;
      if (g.lowOverUp !== null) perCell[cell]!.low.push(g.lowOverUp);
      if (c.step < 10000 || c.step > 20000) continue;
      const lo = g.etaLow;
      const up = g.etaUp ?? Infinity;
      const reading = lo !== null && 0.8 * r.eta <= lo && up <= 1.25 * r.eta ? 'inside' : up < 0.8 * r.eta ? 'below' : lo !== null && lo > 1.25 * r.eta ? 'above' : 'undecided';
      perCell[cell]!.reading[reading] = perCell[cell]!.reading[reading]! + 1;
    }
  }
  const result: Record<string, unknown> = { runs, mismatches, expectedRuns: expected.length };
  for (const o of objects) {
    const a = acc[o]!;
    result[o] = {
      checkpoints: a.total,
      invalid: a.invalid.length,
      invalidVsStored: a.invalidVsStored,
      invalidCases: a.invalid.slice(0, 20),
      undefinedLow: a.undefinedLow,
      witnessNotPositiveDefinite: a.witnessFail,
      lanczosReruns: a.reruns,
      removedModeDisagreements: a.agreeFail,
      fractionLowOverUpAtLeast098: a.narrow / a.total,
      lowOverUp: quantiles(a.ratios),
      upOverStored: quantiles(a.upRatios),
    };
  }
  const gn = acc.gaussNewton!;
  const listed = acc.gaussNewton!.total + acc.gaussNewtonMu0!.total;
  // A decision needs every run regenerated without mismatch, the removed-mode sets agreeing with the records, and checkpoints to
  // decide on (second audit round: an empty or mismatched directory used to read 'no violation' and 'adopted').
  // The records must be exactly the committed list: every listed run once, with exactly its listed checkpoint steps (third audit
  // round: a directory holding one run of 48 used to read 'no violation' and 'adopted').
  const byRun = new Map<string, RunRecord>();
  let duplicate = false;
  for (const r of records) { if (byRun.has(r.run)) duplicate = true; byRun.set(r.run, r); }
  let matchesList = !duplicate && records.length === expected.length && expected.length > 0;
  for (const e of expected) {
    const r = byRun.get(e.run);
    if (r === undefined || r.checkpoints.length !== e.steps.length) { matchesList = false; break; }
    for (let i = 0; i < e.steps.length; i++) if (r.checkpoints[i]!.step !== e.steps[i]) { matchesList = false; break; }
  }
  const complete = matchesList && mismatches.length === 0 && listed > 0 && acc.gaussNewton!.agreeFail === 0 && acc.gaussNewtonMu0!.agreeFail === 0 && acc.absH!.agreeFail === 0;
  // Adoption needs the validity rule to pass on every object (sixth audit round: an invalid |H| bracket rejected the method while the
  // Gauss–Newton object still read 'adopted').
  const validity = !complete ? 'incomplete: no decision' : acc.gaussNewton!.invalid.length === 0 && acc.gaussNewtonMu0!.invalid.length === 0 && acc.absH!.invalid.length === 0 ? 'no violation' : 'violation: method rejected';
  result.decision = {
    validity,
    adoptionGaussNewton: validity === 'no violation' && gn.total > 0 && gn.narrow >= 0.9 * gn.total ? 'adopted' : 'not adopted',
  };
  const cells: Record<string, unknown> = {};
  for (const k of Object.keys(perCell).sort()) cells[k] = { lowOverUp: quantiles(perCell[k]!.low), windowReading: perCell[k]!.reading };
  result.gaussNewtonPerCell = cells;
  return result;
}

if (process.argv[1] !== undefined && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const evid = path.resolve(here, '..', '..', 'evidence', 'mean-square-boundary');
  const dir = process.argv[2] ?? path.join(evid, 'phase6-stage1');
  const out = process.argv[3] ?? path.join(evid, 'phase6-stage1-summary.json');
  const files = fs.readdirSync(dir).filter((f) => f.endsWith('.json')).sort();
  const records = files.map((f) => JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8')) as RunRecord);
  const list = JSON.parse(fs.readFileSync(path.join(evid, 'phase6-checkpoints.json'), 'utf8')) as { stage1: { runs: Array<{ run: string; steps: number[] }> } };
  const s = summarizePhase6Stage1(records, list.stage1.runs);
  fs.writeFileSync(out, `${JSON.stringify({ labId: 'mean-square-boundary-phase6-stage1-summary', date: new Date().toISOString(), ...s }, null, 1)}\n`);
  process.stdout.write(`written ${out}\n`);
}
