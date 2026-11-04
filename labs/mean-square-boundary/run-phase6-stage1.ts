/**
 * @file labs/mean-square-boundary/run-phase6-stage1.ts
 * @description Stage 1 of the Phase 6 plan for one Phase 5b run (`runPhase6Stage1Run`): the certified bracket
 * `computeCertifiedThresholdBracket` at every stored checkpoint of `evidence/mean-square-boundary/phase5b-runs/<run>.json`, against
 * the dense threshold stored there, for two objects built exactly as `evaluatePhase5Checkpoint` builds them:
 * - `gaussNewton`: per-sample Gauss–Newton Hessians (`createPhase4SampleHessianProvider` with `gaussNewton: true`), eigenbasis
 *   deviations, curvature `|lambda_GN + mu_train|`, modes with `|lambda_GN + mu_train| < 1e-8` times the top one removed; the stored
 *   `removedModes` must agree.
 * - `absH`: the loss Hessians, curvature `|lambda + mu_train|`, the modes listed as removed in the stored `nullModes` taken out; the
 *   candidate set `|lambda + mu_train| < 1e-8 lambda_1` must equal the stored list.
 * Iterates come from `regeneratePhase5CheckpointIterates`; the loss at each checkpoint must equal the stored `loss` bit for bit,
 * otherwise the run stops with `mismatchAt`. Bracket by `computeAdaptiveCertifiedThresholdBracket` (Lanczos 30, 60, 120 steps; the
 * plan's rule was 30, then 60 on the residual, changed after the first stage 1 run, see that file); `B = 16`, `beta = 0.9`. Per checkpoint: `etaLow`, `etaUp`, the stored `etaStar`, the
 * validity flags at relative tolerance `1e-8` and the Lanczos diagnostics. With `denseReference`, also the threshold from
 * `computeEigenbasisThresholdDenseReference` (full eigendecomposition of the assembled operator, no Lanczos) and the validity
 * against it: added after stage 1 found 94 Gauss–Newton checkpoints, all at `mu_train = 0`, `eta = 0.5`, whose stored threshold
 * came from a dense Lanczos that had not reached its residual (`lanczosRelResidual` up to `9e-3`). CLI:
 * `node run-phase6-stage1.ts <run-name> [out] [dense]`; output `evidence/mean-square-boundary/phase6-stage1/<run-name>.json`.
 * @cold
 */
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildEigenbasisSampleHessianDeviations } from './build-eigenbasis-sample-hessian-deviations.ts';
import { computeAdaptiveCertifiedThresholdBracket } from './compute-adaptive-certified-threshold-bracket.ts';
import { createDenseDeviationNoiseOperator } from './create-dense-deviation-noise-operator.ts';
import { computeEigenbasisThresholdDenseReference } from './compute-eigenbasis-threshold-dense.reference.ts';
import { createPhase4SampleHessianProvider } from './create-phase4-sample-hessian-provider.ts';
import { decidePhase6RegenerationMismatch } from './decide-phase6-regeneration-mismatch.ts';
import { regeneratePhase5CheckpointIterates } from './regenerate-phase5-checkpoint-iterates.ts';
import { restrictEigenbasisDeviations } from './restrict-eigenbasis-deviations.ts';
import { unpackEigenbasisDeviationsDense } from './unpack-eigenbasis-deviations-dense.ts';

type Dev = ReturnType<typeof buildEigenbasisSampleHessianDeviations>;
type Stored = { step: number; loss: number; nullModes: Array<{ k: number; removed: boolean }>; absH: { etaStar: number }; gaussNewton: { etaStar: number; removedModes: number } };

function bracketObject(dev: Dev, values: Float64Array, keep: Uint8Array, beta: number, batchB: number, ref: number, dense: boolean): Record<string, unknown> {
  let removed = 0;
  for (let k = 0; k < dev.n; k++) if (keep[k] === 0) removed++;
  const d = removed > 0 ? restrictEigenbasisDeviations(dev, keep) : dev;
  const vals = removed > 0 ? Float64Array.from((d as Dev & { modes: Int32Array }).modes, (k) => values[k]!) : values;
  const lt = Float64Array.from(vals, (v) => Math.abs(v));
  const devDense = unpackEigenbasisDeviationsDense(d);
  const noise = createDenseDeviationNoiseOperator(devDense, d.n, d.numSamples, batchB);
  const etaStarDense = dense ? computeEigenbasisThresholdDenseReference(lt, d.n, beta, devDense, d.numSamples, batchB) : null;
  const adaptive = computeAdaptiveCertifiedThresholdBracket(lt, d.n, beta, noise);
  const br = adaptive.bracket;
  const lanczosRerun = adaptive.lengthsTried.length > 1;
  if (br === null) return { removedModes: removed, keptModes: d.n, etaStarStored: ref, etaLow: null, etaUp: null, lowOverUp: null, upOverStored: null, valid: false, validDense: false, witnessPositiveDefinite: false, lanczosRerun: false, bracketNull: true };
  // A null lower bound is never a pass (plan section 5, rule 1; second audit round).
  const lowOk = br.etaLow !== null && br.etaLow <= ref * (1 + 1e-8);
  const upOk = ref <= br.etaUp * (1 + 1e-8);
  return {
    removedModes: removed,
    keptModes: d.n,
    etaStarStored: ref,
    etaLow: br.etaLow,
    etaUp: br.etaUp,
    etaDet: br.etaDet,
    ritzEta: br.ritzEta,
    ritzResidual: br.ritzResidual,
    lanczosSteps: br.lanczosSteps,
    lanczosRerun,
    lengthsTried: adaptive.lengthsTried,
    bestEps: br.bestEps,
    witnessPositiveDefinite: br.witnessPositiveDefinite,
    lowOverStored: br.etaLow === null ? null : br.etaLow / ref,
    upOverStored: br.etaUp / ref,
    lowOverUp: br.etaLow === null ? null : br.etaLow / br.etaUp,
    valid: lowOk && upOk,
    etaStarDense,
    validDense: etaStarDense === null ? null : br.etaLow !== null && br.etaLow <= etaStarDense * (1 + 1e-8) && etaStarDense <= br.etaUp * (1 + 1e-8),
    storedOverDense: etaStarDense === null ? null : ref / etaStarDense,
  };
}

export function runPhase6Stage1Run(runFile: string, denseReference = false): Record<string, unknown> {
  const run = JSON.parse(fs.readFileSync(runFile, 'utf8')) as { eta: number; muTrain: number; seed: number; beta: number; batchB: number; steps: number; divergedAt: number | null; checkpoints: Stored[] };
  const byStep = new Map<number, Stored>();
  for (let c = 0; c < run.checkpoints.length; c++) byStep.set(run.checkpoints[c]!.step, run.checkpoints[c]!);
  // Regenerate to the end of the stored run (its divergence step or its length), so that a divergence after the last checkpoint is
  // compared too (second audit round).
  const lastStep = run.divergedAt ?? run.steps;
  const out: Array<Record<string, unknown>> = [];
  let mismatchAt: number | null = null;
  const mu = run.muTrain;
  const regenDivergedAt = regeneratePhase5CheckpointIterates(run.eta, mu, run.seed, lastStep, (t, theta, loss) => {
    const stored = byStep.get(t);
    if (stored === undefined) return true;
    if (loss !== stored.loss) { mismatchAt = t; return false; }
    const th = Float64Array.from(theta);
    const devGn = buildEigenbasisSampleHessianDeviations(createPhase4SampleHessianProvider({ kind: 'mlp33', theta: th, gaussNewton: true }));
    const n = devGn.n;
    const valuesGn = Float64Array.from(devGn.values, (v) => v + mu);
    const keepGn = new Uint8Array(n).fill(1);
    const top = valuesGn[n - 1]!;
    for (let k = 0; k < n; k++) if (Math.abs(valuesGn[k]!) < 1e-8 * top) keepGn[k] = 0;
    const gn = bracketObject(devGn, valuesGn, keepGn, run.beta, run.batchB, stored.gaussNewton.etaStar, denseReference);
    const gnRemovedAgrees = gn.removedModes === stored.gaussNewton.removedModes;
    const dev = buildEigenbasisSampleHessianDeviations(createPhase4SampleHessianProvider({ kind: 'mlp33', theta: th }));
    const values = Float64Array.from(dev.values, (v) => v + mu);
    const lambda1 = values[n - 1]!;
    const keep = new Uint8Array(n).fill(1);
    const candidates: number[] = [];
    for (let k = 0; k < n; k++) if (Math.abs(values[k]!) < 1e-8 * lambda1) candidates.push(k);
    for (let m = 0; m < stored.nullModes.length; m++) if (stored.nullModes[m]!.removed) keep[stored.nullModes[m]!.k] = 0;
    const storedCandidates = stored.nullModes.map((m) => m.k);
    const nullModesAgree = candidates.length === storedCandidates.length && candidates.every((k, i) => k === storedCandidates[i]);
    const absH = bracketObject(dev, values, keep, run.beta, run.batchB, stored.absH.etaStar, denseReference);
    out.push({ step: t, gaussNewton: { ...gn, removedModesAgree: gnRemovedAgrees }, absH: { ...absH, nullModesAgree } });
    return true;
  });
  // A regeneration that diverges where the stored run did not, or that yields fewer checkpoints than stored, is a mismatch.
  mismatchAt = decidePhase6RegenerationMismatch(mismatchAt, regenDivergedAt, run.divergedAt, [[out.length, run.checkpoints.length]]);
  return { eta: run.eta, muTrain: mu, seed: run.seed, beta: run.beta, batchB: run.batchB, divergedAt: run.divergedAt, storedCheckpoints: run.checkpoints.length, mismatchAt, checkpoints: out };
}

if (process.argv[1] !== undefined && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const name = process.argv[2] ?? '';
  const here = path.dirname(fileURLToPath(import.meta.url));
  const evid = path.resolve(here, '..', '..', 'evidence', 'mean-square-boundary');
  const out = process.argv[3] ?? path.join(evid, 'phase6-stage1', `${name}.json`);
  const t0 = Date.now();
  const r = runPhase6Stage1Run(path.join(evid, 'phase5b-runs', `${name}.json`), process.argv[4] === 'dense');
  fs.mkdirSync(path.dirname(out), { recursive: true });
  // Provenance (ninth audit round): the commit of this code and the provenance file of the original runs.
  // In a package without git (Modal), the commit comes from L67_BRACKET_COMMIT, written by the package builder.
  let bracketCommit: string | null = process.env.L67_BRACKET_COMMIT ?? null;
  try { bracketCommit = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: here, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim(); } catch { /* no git */ }
  const provFile = path.join(evid, 'phase5b-runs', 'provenance.json');
  const originalProvenance = fs.existsSync(provFile) ? (JSON.parse(fs.readFileSync(provFile, 'utf8')) as unknown) : null;
  fs.writeFileSync(out, `${JSON.stringify({ labId: 'mean-square-boundary-phase6-stage1-run', date: new Date().toISOString(), nodeVersion: process.version, elapsedSeconds: (Date.now() - t0) / 1000, bracketCommit, originalProvenance, run: name, ...r })}\n`);
  process.stdout.write(`written ${out}\n`);
}
