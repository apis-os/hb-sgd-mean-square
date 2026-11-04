/**
 * @file labs/mean-square-boundary/run-phase1-cap-diagnostic.ts
 * @description Post-hoc diagnostic of the failed Phase 1 cell (`runPhase1CapDiagnostic`): the cell `P = 64`, `beta = 0`,
 * `B = 64` rerun for Gaussian and Rademacher features with the seeds, streams, steps, bisection and growth threshold of
 * Phase 1 and a single bisection on `[0, 1.15]` times the deterministic ceiling instead of the expanding bracket capped at `1.02`; writes
 * `evidence/mean-square-boundary/phase1-diag-as-vs-ms.json`. Not a pre-registered prediction. Usage:
 * `node --experimental-strip-types labs/mean-square-boundary/run-phase1-cap-diagnostic.ts`.
 * @cold
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { computeFloat64Quantile } from './compute-float64-quantile.ts';
import { measureEmpiricalEossCeiling } from './measure-empirical-eoss-ceiling.ts';
import { solveSecularEossThreshold } from './solve-secular-eoss-threshold.ts';

interface DiagnosticFamilyRow {
  readonly family: 'gaussian-iid' | 'rademacher-iid';
  readonly ceilings: number[];
  readonly median: number;
}

export function runPhase1CapDiagnostic(): { readonly cap: number; readonly results: DiagnosticFamilyRow[] } {
  const numParams = 64;
  const beta = 0.0;
  const batchB = 64;
  const numSeeds = 8;
  const streams = 8;
  const steps = 1000;
  const bisectionIterations = 12;
  const logThreshold = 0.9162907318741551;
  const sigmaXi = 0.01;
  const capFactor = 1.15;
  const sqrtLambdas = new Float64Array(numParams);
  const lambdas = new Float64Array(numParams);
  for (let k = 0; k < numParams; (k = (k + 1) | 0)) {
    lambdas[k] = 1.0 / Math.sqrt(k + 1);
    sqrtLambdas[k] = Math.sqrt(lambdas[k]!);
  }
  // Cell indices of P = 64, beta = 0, B = 64 in the Phase 1 cell order (Gaussian 6, Rademacher 7), so the seeds are Phase 1's.
  const families = [
    { family: 'gaussian-iid' as const, ci: 6, kappaFeat: 0.0, rademacher: 0 as const },
    { family: 'rademacher-iid' as const, ci: 7, kappaFeat: -2.0, rademacher: 1 as const },
  ];
  const results: DiagnosticFamilyRow[] = [];
  let cap = 0.0;
  for (const f of families) {
    const pred = solveSecularEossThreshold(lambdas, beta, batchB, f.kappaFeat, 0.0);
    cap = capFactor * pred.etaCritDeterministic;
    const ceilings: number[] = [];
    for (let seed = 0; seed < numSeeds; (seed = (seed + 1) | 0)) {
      // One bisection on [0, cap], with the Phase 1 seeds of the first bracket of this cell.
      const result = measureEmpiricalEossCeiling(
        sqrtLambdas, beta, cap, batchB, f.rademacher, null, 0, sigmaXi, streams, steps, bisectionIterations, logThreshold,
        (0x95000000 + Math.imul(f.ci + 1, 0x1000) + seed * 613) | 0,
        (0x2545f491 ^ Math.imul(seed + 3, 0x9e3779b9) ^ Math.imul(f.ci + 1, 0x85ebca6b)) | 0,
      );
      ceilings.push(result.etaCeiling);
    }
    ceilings.sort((a, b) => a - b);
    results.push({ family: f.family, ceilings, median: computeFloat64Quantile(Float64Array.from(ceilings), 0.5) });
  }
  return { cap, results };
}

if (process.argv[1] !== undefined && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { cap, results } = runPhase1CapDiagnostic();
  const report = {
    labId: 'mean-square-boundary-phase1-diag-as-vs-ms',
    date: new Date().toISOString(),
    cell: 'P64_beta0_B64',
    note: 'post-hoc diagnostic, not a pre-registered prediction: bisection cap raised to 1.15 x deterministic ceiling; same streams/steps/threshold as Phase 1',
    streams: 8,
    steps: 1000,
    iterations: 12,
    cap,
    results,
  };
  const here = path.dirname(fileURLToPath(import.meta.url));
  const out = path.resolve(here, '..', '..', 'evidence', 'mean-square-boundary', 'phase1-diag-as-vs-ms.json');
  fs.writeFileSync(out, `${JSON.stringify(report, null, 2)}\n`);
  for (const r of results) process.stdout.write(`${r.family}: median ${r.median}, range ${r.ceilings[0]}-${r.ceilings[r.ceilings.length - 1]}\n`);
}
