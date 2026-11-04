/**
 * @file labs/mean-square-boundary/run-phase4-certified-bracket-p33.ts
 * @description Post hoc (2026-11-01, after the audits of the second-moment draft): the certified bracket of Phase 6
 * (`computeAdaptiveCertifiedThresholdBracket`: Rayleigh upper bound with forward error, cone Collatz–Wielandt lower bound with a
 * shifted Cholesky and Rump's bound) on the 16 tethered `P = 33` cells of Phase 4 (`runPhase4CertifiedBracketP33`), in place of
 * the floating-point Collatz–Wielandt bracket of the last power iterate that Phase 4 used (check `t1`). Per cell
 * (`beta` in {0, 0.9}, the eight tethers of `raw-phase4.json`, `B = 16`): per-sample Hessians from
 * `createPhase4SampleHessianProvider({ kind: 'mlp33' })` at the checkpoint, eigenbasis deviations (unchanged by the tether), curvature
 * `lambda_k + mu`, the dense noise operator `createDenseDeviationNoiseOperator`, the bracket, the stored Lanczos threshold `etaThmG6`
 * and the dense reference `computeEigenbasisThresholdDenseReference`.
 * Expectation written before the run: in every cell both the stored threshold and the dense reference lie inside the certified
 * bracket with no tolerance, and `eta_low / eta_up >= 0.98` (stage 1 of Phase 6 gave at least 0.9934 on the Gauss–Newton
 * object at `P = 33`); a cell where either fails is reported as it is. The bracket is certified for the computed per-sample
 * Hessians (central differences, step `1e-5`) and `fl(lambda_k + mu)`, not for the exact network Hessians. Writes
 * `evidence/mean-square-boundary/phase4-certified-bracket-p33.json`.
 * @cold
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildEigenbasisSampleHessianDeviations } from './build-eigenbasis-sample-hessian-deviations.ts';
import { computeAdaptiveCertifiedThresholdBracket } from './compute-adaptive-certified-threshold-bracket.ts';
import { computeEigenbasisThresholdDenseReference } from './compute-eigenbasis-threshold-dense.reference.ts';
import { createDenseDeviationNoiseOperator } from './create-dense-deviation-noise-operator.ts';
import { createPhase4SampleHessianProvider } from './create-phase4-sample-hessian-provider.ts';
import { unpackEigenbasisDeviationsDense } from './unpack-eigenbasis-deviations-dense.ts';

export function runPhase4CertifiedBracketP33(): Record<string, unknown> {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const evid = path.resolve(here, '..', '..', 'evidence', 'mean-square-boundary');
  const raw = JSON.parse(fs.readFileSync(path.join(evid, 'raw-phase4.json'), 'utf8')) as {
    networks: Array<{ networkId: string; thresholds: Array<{ beta: number; batchB: number; tetherGrid: number; mu: number; etaThmG6: number }> }>;
  };
  const cells = raw.networks.find((r) => r.networkId === 'N33_mlp')!.thresholds;
  const dev = buildEigenbasisSampleHessianDeviations(createPhase4SampleHessianProvider({ kind: 'mlp33' }));
  const devDense = unpackEigenbasisDeviationsDense(dev);
  const rows: Array<Record<string, unknown>> = [];
  let allInside = true;
  let minLowOverUp = Infinity;
  for (const cell of cells) {
    const lt = Float64Array.from(dev.values, (v) => v + cell.mu);
    const noise = createDenseDeviationNoiseOperator(devDense, dev.n, dev.numSamples, cell.batchB);
    const adaptive = computeAdaptiveCertifiedThresholdBracket(lt, dev.n, cell.beta, noise);
    const br = adaptive.bracket;
    const dense = computeEigenbasisThresholdDenseReference(lt, dev.n, cell.beta, devDense, dev.numSamples, cell.batchB);
    const low = br === null ? null : br.etaLow;
    const up = br === null ? null : br.etaUp;
    const inside = (x: number): boolean => low !== null && up !== null && low <= x && x <= up;
    const storedInside = inside(cell.etaThmG6);
    const denseInside = inside(dense);
    if (!storedInside || !denseInside) allInside = false;
    const lowOverUp = low !== null && up !== null ? low / up : null;
    if (lowOverUp !== null && lowOverUp < minLowOverUp) minLowOverUp = lowOverUp;
    rows.push({ beta: cell.beta, tetherGrid: cell.tetherGrid, mu: cell.mu, batchB: cell.batchB, etaLow: low, etaUp: up, lowOverUp, etaThmG6Stored: cell.etaThmG6, etaDense: dense, storedInside, denseInside, lanczosSteps: br === null ? null : br.lanczosSteps, lengthsTried: adaptive.lengthsTried, witnessPositiveDefinite: br === null ? false : br.witnessPositiveDefinite, bestEps: br === null ? null : br.bestEps });
  }
  return { labId: 'mean-square-boundary-phase4-certified-bracket-p33', cells: rows.length, allInside, minLowOverUp, rows };
}

if (process.argv[1] !== undefined && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const r = runPhase4CertifiedBracketP33();
  const out = path.resolve(here, '..', '..', 'evidence', 'mean-square-boundary', 'phase4-certified-bracket-p33.json');
  fs.writeFileSync(out, `${JSON.stringify({ ...r, date: new Date().toISOString(), nodeVersion: process.version }, null, 1)}\n`);
  process.stdout.write(`${JSON.stringify({ cells: r.cells, allInside: r.allInside, minLowOverUp: r.minLowOverUp })}\n`);
}
