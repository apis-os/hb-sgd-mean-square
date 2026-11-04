/**
 * @file labs/mean-square-boundary/run-phase2.ts
 * @description Executes Phase 2 (`runPhase2`): trace, noise-rank ratio, secular bracket, destructive minibatch ceiling in `R^P` and finite-difference HVP baseline on the scale networks; writes `evidence/mean-square-boundary/raw-phase2.json` and `phase2-summary.md`. Usage: `node --experimental-strip-types labs/mean-square-boundary/run-phase2.ts [seeds] [streams] [steps]`.
 * @cold
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { evaluatePhase2TraceCeilingCells } from './evaluate-phase2-trace-ceiling-cells.ts';
import type { Phase2Report } from './phase2-types.ts';

function formatPhase2Markdown(r: Phase2Report): string {
  const lines: string[] = [];
  lines.push('# Phase 2: Hessian trace, the ceiling in R^P on the networks, and the finite-difference HVP baseline');
  lines.push('');
  lines.push(`- **Date**: \`${r.date}\``);
  lines.push(`- **Node**: \`${r.nodeVersion}\``);
  lines.push(`- **Seeds per network cell**: \`${r.numSeeds}\`; streams \`${r.streams}\`, steps \`${r.steps}\`, bisection iterations \`${r.bisectionIterations}\`, growth-ratio threshold \`${r.growthRatioThreshold}\`, additive-noise scale \`tau = ${r.noiseScaleTau}\``);
  lines.push('');
  lines.push('## 1. Pre-registered predictions (`q1`–`q4`)');
  lines.push('');
  for (const [k, text] of Object.entries(r.predictionTexts)) lines.push(`- **\`${k}\`**: ${text}`);
  lines.push('');
  lines.push('### Prediction outcomes');
  lines.push('');
  for (const [k, v] of Object.entries(r.predictions)) lines.push(`- \`${k}\`: \`${String(v)}\``);
  lines.push('');
  lines.push('## 2. Networks (beta = 0.9, B = 16)');
  lines.push('');
  lines.push(
    '| Cell | `P` | `lambda_1` | `lambda_2` | `lambda_min` (shifted PI) | `tr H` | `tr H / lambda_1` | `rho_B` | `nu_B` | `kappa_feat,eff` | `alpha_diag` | `eta_quad` | `eta_destr` (ray) | gap % | 1D EoSS | sandwich lower | trace lower | trace upper | sandwich upper | empirical median | p05 | p95 | bisection ±% | upper stable | 1D / emp | quad / emp | inside |',
  );
  lines.push('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|');
  for (const c of r.networks) {
    lines.push(
      `| \`${c.cellId}\` | ${c.numParams} | ${c.lambda1.toFixed(3)} | ${c.lambda2.toFixed(3)} | ${c.lambdaMinShiftedPowerIter100.toFixed(4)} | ${c.traceH.toFixed(2)} | ${c.traceOverLambda1.toFixed(2)} | ${c.noiseRankRatio.toFixed(2)} | ${c.nuB.toFixed(2)} | ${c.kappaFeatEffective.toFixed(3)} | ${c.alphaDiag.toFixed(3)} | ${c.etaCritQuadratic.toFixed(6)} | ${c.etaCritDestructiveRay.toFixed(6)} | ${c.quadraticVsDestructiveGapPct.toFixed(3)} | ${c.etaCritOneDimEoss.toFixed(6)} | ${c.etaSandwichLower.toFixed(6)} | ${c.etaTraceBracketLower.toFixed(6)} | ${c.etaTraceBracketUpper.toFixed(6)} | ${c.etaSandwichUpper.toFixed(6)} | ${c.empiricalCeilingMedian.toFixed(6)} | ${c.empiricalCeilingP05.toFixed(6)} | ${c.empiricalCeilingP95.toFixed(6)} | ${c.bisectionHalfWidthPct.toFixed(3)} | ${c.upperWasStableCount} | ${c.oneDimOverEmpiricalRatio.toFixed(2)} | ${c.quadraticOverEmpiricalRatio.toFixed(2)} | ${c.empiricalInsideTraceBracket ? 'yes' : 'no'} |`,
    );
  }
  lines.push('');
  lines.push('### Post-hoc growth grid (log energy growth over the window ratio; rate per step = growth / (steps / 2); no prediction registered)');
  lines.push('');
  lines.push('| Cell | `eta / eta_quad` | `eta` | steps | log growth | rate per step | guard |');
  lines.push('|---|---:|---:|---:|---:|---:|---|');
  for (const c of r.networks) {
    for (const g of c.growthGrid) {
      lines.push(`| \`${c.cellId}\` | ${g.etaOverQuadratic} | ${g.eta.toExponential(3)} | ${g.steps} | ${g.guardTriggered ? 'guard' : g.logGrowth.toFixed(3)} | ${g.guardTriggered ? 'guard' : g.ratePerStep.toExponential(3)} | ${g.guardTriggered ? 'yes' : 'no'} |`);
    }
  }
  lines.push('');
  lines.push('## 3. Finite-difference HVP power iteration (two full gradients per product, no second-order code)');
  lines.push('');
  lines.push('| Cell | `P` | products | sample-grad equiv. | `lambda_1` hat | rel. err | cos with analytic `u_1` | `eta_crit` hat (0.9) | err vs quadratic % | err vs destructive % | trace s | ceiling s |');
  lines.push('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|');
  for (const c of r.networks) {
    lines.push(
      `| \`${c.cellId}\` | ${c.numParams} | ${c.fdHvpIterations} | ${c.fdHvpSampleGradEquivalents} | ${c.fdHvpLambda1Hat.toFixed(6)} | ${c.fdHvpLambda1RelErr.toExponential(2)} | ${c.fdHvpCosWithAnalyticU1.toFixed(8)} | ${c.fdHvpEtaCritHat.toFixed(6)} | ${c.fdHvpErrVsQuadraticPct.toFixed(4)} | ${c.fdHvpErrVsDestructivePct.toFixed(4)} | ${c.traceSeconds.toFixed(1)} | ${c.ceilingSeconds.toFixed(1)} |`,
    );
  }
  lines.push('');
  lines.push('');
  return lines.join('\n');
}

export function runPhase2(numSeeds: number, streams: number, steps: number): Phase2Report {
  const report = evaluatePhase2TraceCeilingCells(numSeeds, streams, steps);
  const here = path.dirname(fileURLToPath(import.meta.url));
  const evidenceDir = path.resolve(here, '..', '..', 'evidence', 'mean-square-boundary');
  fs.mkdirSync(evidenceDir, { recursive: true });
  fs.writeFileSync(path.join(evidenceDir, 'raw-phase2.json'), `${JSON.stringify(report, null, 2)}\n`);
  fs.writeFileSync(path.join(evidenceDir, 'phase2-summary.md'), `${formatPhase2Markdown(report)}\n`);
  return report;
}

if (process.argv[1] !== undefined && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const seeds = Number.parseInt(process.argv[2] ?? '4', 10);
  const streams = Number.parseInt(process.argv[3] ?? '8', 10);
  const steps = Number.parseInt(process.argv[4] ?? '600', 10);
  const t0 = Date.now();
  const report = runPhase2(seeds, streams, steps);
  for (const [k, v] of Object.entries(report.predictions)) process.stdout.write(`${k}: ${String(v)}\n`);
  process.stdout.write(`networks ${report.networks.length}, elapsed ${((Date.now() - t0) / 1000).toFixed(1)} s\n`);
}
