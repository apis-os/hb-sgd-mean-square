/**
 * @file labs/mean-square-boundary/run-phase1.ts
 * @description Executes the Phase 1 study of the stochastic ceiling in `R^P` (`runPhase1`), writing `evidence/mean-square-boundary/raw-phase1.json` and `evidence/mean-square-boundary/phase1-summary.md`. Usage: `node --experimental-strip-types labs/mean-square-boundary/run-phase1.ts [seeds] [streams] [steps]`.
 * @cold
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { evaluatePhase1RademacherCells } from './evaluate-phase1-rademacher-cells.ts';
import type { Phase1Report } from './phase1-types.ts';

function formatPhase1Markdown(r: Phase1Report): string {
  const lines: string[] = [];
  lines.push('# Phase 1: the stochastic ceiling in R^P, Gaussian against Rademacher features (Theorem G)');
  lines.push('');
  lines.push(`- **Date**: \`${r.date}\``);
  lines.push(`- **Node**: \`${r.nodeVersion}\``);
  lines.push(`- **Seeds per cell**: \`${r.numSeeds}\`; streams \`${r.streams}\`, steps \`${r.steps}\`, bisection iterations \`${r.bisectionIterations}\`, growth-ratio threshold \`${r.growthRatioThreshold}\``);
  lines.push('');
  lines.push('## 1. Pre-registered predictions (`p1`–`p4`)');
  lines.push('');
  for (const [k, text] of Object.entries(r.predictionTexts)) lines.push(`- **\`${k}\`**: ${text}`);
  lines.push('');
  lines.push('### Prediction outcomes');
  lines.push('');
  for (const [k, v] of Object.entries(r.predictions)) lines.push(`- \`${k}\`: \`${String(v)}\``);
  lines.push('');
  lines.push('## 2. Cells');
  lines.push('');
  lines.push(
    '| Cell | `P` | `beta` | `B` | Family | `zeta` | `kappa_beta tr(H)/(B lambda_1)` | `eta_det` | 1D (Thm E(3)) | Secular (Thm G) | Lee limit | Sandwich lower | Sandwich upper | Empirical median | Emp. p05 | Emp. p95 | Bisection ±% | Secular err % | Lee err % | 1D err % | 1D / emp | Inside sandwich |',
  );
  lines.push('|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|');
  for (let i = 0; i < r.cells.length; (i = (i + 1) | 0)) {
    const c = r.cells[i]!;
    lines.push(
      `| \`${c.cellId}\` | ${c.numParams} | ${c.beta} | ${c.batchB} | ${c.featureFamily} | ${c.zeta.toFixed(4)} | ${c.noiseRankRatio.toFixed(2)} | ${c.etaCritDeterministic.toFixed(4)} | ${c.etaCritOneDim.toFixed(4)} | ${c.etaCritSecular.toFixed(4)} | ${c.etaCritSecularLeeLimit.toFixed(4)} | ${c.etaCritSandwichLower.toFixed(4)} | ${c.etaCritSandwichUpper.toFixed(4)} | ${c.empiricalCeilingMedian.toFixed(4)} | ${c.empiricalCeilingP05.toFixed(4)} | ${c.empiricalCeilingP95.toFixed(4)} | ${c.bisectionHalfWidthPct.toFixed(2)} | ${c.secularMedianErrPct.toFixed(2)} | ${c.secularLeeLimitMedianErrPct.toFixed(2)} | ${c.oneDimMedianErrPct.toFixed(2)} | ${c.oneDimOverEmpiricalRatio.toFixed(2)} | ${c.insideSandwich ? 'yes' : 'no'} |`,
    );
  }
  lines.push('');
  return lines.join('\n');
}

export function runPhase1(numSeeds: number, streams: number, steps: number): Phase1Report {
  const report = evaluatePhase1RademacherCells(numSeeds, streams, steps);
  const here = path.dirname(fileURLToPath(import.meta.url));
  const evidenceDir = path.resolve(here, '..', '..', 'evidence', 'mean-square-boundary');
  fs.mkdirSync(evidenceDir, { recursive: true });
  fs.writeFileSync(path.join(evidenceDir, 'raw-phase1.json'), `${JSON.stringify(report, null, 2)}\n`);
  fs.writeFileSync(path.join(evidenceDir, 'phase1-summary.md'), `${formatPhase1Markdown(report)}\n`);
  return report;
}

if (process.argv[1] !== undefined && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const seeds = Number.parseInt(process.argv[2] ?? '8', 10);
  const streams = Number.parseInt(process.argv[3] ?? '8', 10);
  const steps = Number.parseInt(process.argv[4] ?? '1000', 10);
  const t0 = Date.now();
  const report = runPhase1(seeds, streams, steps);
  for (const [k, v] of Object.entries(report.predictions)) process.stdout.write(`${k}: ${String(v)}\n`);
  process.stdout.write(`cells ${report.cells.length}, elapsed ${((Date.now() - t0) / 1000).toFixed(1)} s\n`);
}
