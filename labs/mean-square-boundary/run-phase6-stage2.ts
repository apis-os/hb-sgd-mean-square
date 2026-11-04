/**
 * @file labs/mean-square-boundary/run-phase6-stage2.ts
 * @description Stage 2 of the Phase 6 plan for one Phase 6 pilot run (`runPhase6Stage2Run`): at the requested checkpoint steps,
 * the certified bracket of the Gauss–Newton mean-square threshold on the row space of the stacked Jacobian. Iterates come from
 * `regeneratePhase6PilotCheckpointIterates`; at each requested step the first-layer and output-layer squared norms and the data
 * loss (`net.evalLoss`) must equal the stored `thetaSqFirstLayer`, `thetaSqOutputLayer`, `dataLoss` of
 * `evidence/mean-square-boundary/phase6-pilot/<run>.json` bit for bit, otherwise the run stops with `mismatchAt`. Row space:
 * `computeStudentGaussNewtonDualSpectrum` (cut `1e-10`, as the pilot), factors `M_i[c, k] = sqrt(N lambda_k) vt[k, (i, c)]`, curvature
 * `lambda_k + mu_train`, noise `createRowSpaceFactorNoiseOperator` (`B = 16`, `beta = 0.9`); `rankCompressed` marks a checkpoint whose
 * kept rank is below `N dOut` (then the bracket is of a Rayleigh compression and every lower bound is dropped; the rank cut does not involve the tether, so this
 * can happen at any `mu_train`, though the pilots showed it only at `mu_train = 0`; eleventh audit round). Pilot B runs (`34186`) are read from `phase6-pilot-b`. Bracket by
 * `computeAdaptiveCertifiedThresholdBracket` (Lanczos 30, 60, 120 steps, witness `eps` ladder). With `referenceSteps > 0`, a second bracket from a
 * Lanczos of at most `referenceSteps` steps to relative residual `1e-10` (the stage 2 reference). Per checkpoint, the record of
 * `buildPhase6Stage2CheckpointRecord` (both brackets, with every lower bound `null` on a compressed row space; the stored top-64
 * restricted threshold; the mean residual `||Hbar - diag(lambda)||_F` over the smallest curvature; the dual residual
 * `max_k ||G v_k - lambda_k v_k||_2 / lambda_0`; the reading of rule 3 against `[0.8 eta, 1.25 eta]`; the overlap checks) and wall
 * seconds (local timing of a cold script). CLI:
 * `node run-phase6-stage2.ts <run> <steps comma-separated> <referenceSteps> <out>`.
 * @cold
 */
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildPhase6Stage2CheckpointRecord } from './build-phase6-stage2-checkpoint-record.ts';
import { computeAdaptiveCertifiedThresholdBracket } from './compute-adaptive-certified-threshold-bracket.ts';
import { computeCertifiedThresholdBracket } from './compute-certified-threshold-bracket.ts';
import { computeStudentGaussNewtonDualSpectrum } from './compute-student-gauss-newton-dual-spectrum.ts';
import { createRowSpaceFactorNoiseOperator } from './create-row-space-factor-noise-operator.ts';
import { decidePhase6RegenerationMismatch } from './decide-phase6-regeneration-mismatch.ts';
import { regeneratePhase6PilotCheckpointIterates } from './regenerate-phase6-pilot-checkpoint-iterates.ts';

type Stored = { step: number; dataLoss: number; thetaSqFirstLayer: number; thetaSqOutputLayer: number; gaussNewton: { restrictedThreshold: { etaStar: number } | null } | null };
type Bracket = NonNullable<ReturnType<typeof computeCertifiedThresholdBracket>>;

function parseRun(name: string): { network: 3610 | 12826 | 34186; activation: 'tanh' | 'leakyRelu01' | 'softsign'; eta: number; mu: number; seed: number } {
  const m = /^p(\d+)-(tanh|leakyRelu01|softsign)-eta([0-9.e-]+)-mu([0-9.e-]+)-seed(\d+)$/.exec(name);
  if (m === null) throw new Error(`bad run name ${name}`);
  return { network: Number(m[1]) as 3610 | 12826 | 34186, activation: m[2] as 'tanh' | 'leakyRelu01' | 'softsign', eta: Number(m[3]), mu: Number(m[4]), seed: Number(m[5]) };
}

export function runPhase6Stage2Run(name: string, steps: number[], referenceSteps: number, evid: string): Record<string, unknown> {
  const spec = parseRun(name);
  const stored = JSON.parse(fs.readFileSync(path.join(evid, spec.network === 34186 ? 'phase6-pilot-b' : 'phase6-pilot', `${name}.json`), 'utf8')) as { steps: number; divergedAt: number | null; checkpoints: Stored[] };
  const byStep = new Map<number, Stored>();
  for (let c = 0; c < stored.checkpoints.length; c++) byStep.set(stored.checkpoints[c]!.step, stored.checkpoints[c]!);
  const want = new Set(steps);
  // Regenerate to the end of the stored run and compare every stored checkpoint, not only the requested ones (third audit round).
  const last = stored.divergedAt ?? stored.steps;
  let compared = 0;
  const out: Array<Record<string, unknown>> = [];
  let mismatchAt: number | null = null;
  const divergedAt = regeneratePhase6PilotCheckpointIterates(spec.network, spec.activation, spec.eta, spec.mu, spec.seed, last, (t, theta, net, normFirst, normOut) => {
    const st = byStep.get(t);
    if (st === undefined) { mismatchAt = t; return false; }
    const dataLoss = net.evalLoss(theta);
    if (normFirst !== st.thetaSqFirstLayer || normOut !== st.thetaSqOutputLayer || dataLoss !== st.dataLoss) { mismatchAt = t; return false; }
    compared++;
    if (!want.has(t)) return true;
    const t0 = Date.now();
    const spectrum = computeStudentGaussNewtonDualSpectrum(net, Float64Array.from(theta), 1e-10, undefined, true);
    const r = spectrum.rank;
    const nS = net.numSamples;
    const dOut = net.dOut;
    const factors = new Float64Array(nS * dOut * r);
    for (let k = 0; k < r; k++) {
      const sc = Math.sqrt(nS * spectrum.values[k]!);
      for (let i = 0; i < nS; i++) for (let c = 0; c < dOut; c++) factors[i * dOut * r + c * r + k] = sc * spectrum.vt[k * spectrum.m + i * dOut + c]!;
    }
    const lambda = Float64Array.from(spectrum.values.subarray(0, r));
    const lt = Float64Array.from(lambda, (v) => v + spec.mu);
    let ltMin = Infinity;
    for (let k = 0; k < r; k++) ltMin = Math.min(ltMin, lt[k]!);
    const noise = createRowSpaceFactorNoiseOperator(factors, lambda, r, nS, dOut, 16);
    const tSpec = (Date.now() - t0) / 1000;
    const t1 = Date.now();
    const adaptive = computeAdaptiveCertifiedThresholdBracket(lt, r, 0.9, noise);
    const br = adaptive.bracket;
    const rerun = adaptive.lengthsTried;
    const tBr = (Date.now() - t1) / 1000;
    let refBr: Bracket | null = null;
    let tRef = 0;
    if (referenceSteps > 0) {
      const t2 = Date.now();
      refBr = computeCertifiedThresholdBracket(lt, r, 0.9, noise, referenceSteps, 1e-10);
      tRef = (Date.now() - t2) / 1000;
    }
    out.push(buildPhase6Stage2CheckpointRecord({
      step: t,
      eta: spec.eta,
      rank: r,
      m: spectrum.m,
      bracket: br,
      bracketSeconds: tBr,
      lengthsTried: rerun,
      reference: refBr,
      referenceSeconds: tRef,
      referenceRequested: referenceSteps > 0,
      top64: st.gaussNewton?.restrictedThreshold?.etaStar ?? null,
      meanResidualOverMinCurvature: noise.meanResidual / ltMin,
      dualResidualOverTop: spectrum.dualResidualMax === null || r === 0 ? NaN : spectrum.dualResidualMax / spectrum.values[0]!,
      spectrumSeconds: tSpec,
    }));
    return true;
  });
  // A regeneration that diverges where the pilot did not (or the reverse), or that stops before the requested steps, is a mismatch.
  mismatchAt = decidePhase6RegenerationMismatch(mismatchAt, divergedAt, stored.divergedAt, [[compared, stored.checkpoints.length], [out.length, steps.length]]);
  return { run: name, ...spec, steps, referenceSteps, divergedAt, storedDivergedAt: stored.divergedAt, mismatchAt, checkpoints: out };
}

if (process.argv[1] !== undefined && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const evid = path.resolve(here, '..', '..', 'evidence', 'mean-square-boundary');
  const name = process.argv[2] ?? '';
  const steps = (process.argv[3] ?? '10000').split(',').map((s) => Number(s));
  const referenceSteps = Number(process.argv[4] ?? '0');
  const out = process.argv[5] ?? path.join(evid, 'phase6-stage2', `${name}.json`);
  const t0 = Date.now();
  const r = runPhase6Stage2Run(name, steps, referenceSteps, evid);
  fs.mkdirSync(path.dirname(out), { recursive: true });
  // Provenance (ninth audit round): the commit of this code and the provenance file of the original pilot runs.
  // In a package without git (Modal), the commit comes from L67_BRACKET_COMMIT, written by the package builder.
  let bracketCommit: string | null = process.env.L67_BRACKET_COMMIT ?? null;
  try { bracketCommit = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: here, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim(); } catch { /* no git */ }
  const provFile = path.join(evid, name.startsWith('p34186') ? 'phase6-pilot-b' : 'phase6-pilot', 'provenance.json');
  const originalProvenance = fs.existsSync(provFile) ? (JSON.parse(fs.readFileSync(provFile, 'utf8')) as unknown) : null;
  fs.writeFileSync(out, `${JSON.stringify({ labId: 'mean-square-boundary-phase6-stage2-run', date: new Date().toISOString(), nodeVersion: process.version, elapsedSeconds: (Date.now() - t0) / 1000, bracketCommit, originalProvenance, ...r })}\n`);
  process.stdout.write(`written ${out}\n`);
}
