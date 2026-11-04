/**
 * @file labs/mean-square-boundary/run-phase3.ts
 * @description Executes the Phase 3 check of Theorem G against the exact second-moment operator (`runPhase3`), writing `evidence/mean-square-boundary/raw-phase3.json` and `evidence/mean-square-boundary/phase3-summary.md`. Cells run on a pool of worker threads (largest first) and the rows are put back in spec order, so the raw file does not depend on the worker count. Usage: `node --experimental-strip-types labs/mean-square-boundary/run-phase3.ts [P1,P2,...] [workers]` (default `64,256,1024`, available cores minus 2).
 * @cold
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import os from 'node:os';
import { assemblePhase3Report } from './assemble-phase3-report.ts';
import { buildPhase3CellSpecs } from './build-phase3-cell-specs.ts';
import type { Phase3CellRow, Phase3CellSpec, Phase3Report } from './phase3-types.ts';
import { runSeedBlocksParallel } from './run-seed-blocks-parallel.ts';

function formatPhase3Markdown(r: Phase3Report): string {
  const lines: string[] = [];
  lines.push('# Phase 3: the secular root of Theorem G against the exact second-moment operator');
  lines.push('');
  lines.push(`- **Date**: \`${r.date}\``);
  lines.push(`- **Node**: \`${r.nodeVersion}\``);
  lines.push(`- **Power iteration**: from the identity on the symmetric part, relative tolerance \`${r.relTol}\`, at most \`${r.maxIterations}\` steps; probes at \`(1 ± ${r.relativeOffset}) eta*\`.`);
  lines.push(`- **Provenance**: ${r.provenanceNote}`);
  lines.push('');
  lines.push('## 1. Predictions (`q1`–`q3`)');
  lines.push('');
  for (const [k, text] of Object.entries(r.predictionTexts)) lines.push(`- **\`${k}\`**: ${text}`);
  lines.push('');
  lines.push('### Outcomes');
  lines.push('');
  for (const [k, v] of Object.entries(r.predictions)) lines.push(`- \`${k}\`: \`${String(v)}\``);
  lines.push('');
  lines.push('## 2. Cells');
  lines.push('');
  lines.push('| Cell | `P` | `beta` | `B` | Family | `eta*` (secular) | `rho` at `(1-d) eta*` | `rho` at `eta*` | `rho` at `(1+d) eta*` | antisymmetric `rho` at `eta*` | crossing rel. err | iterations | converged |');
  lines.push('|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|');
  for (let i = 0; i < r.cells.length; (i = (i + 1) | 0)) {
    const c = r.cells[i]!;
    lines.push(
      `| \`${c.cellId}\` | ${c.numParams} | ${c.beta} | ${c.batchB} | ${c.featureFamily} | ${c.etaCritSecular.toPrecision(10)} | ${c.radiusBelow.toFixed(9)} | ${c.radiusAtRoot.toFixed(12)} | ${c.radiusAbove.toFixed(9)} | ${c.radiusAntisymmetricAtRoot.toFixed(6)} | ${c.crossingRelErr.toExponential(2)} | ${c.iterationsAtRoot} | ${c.allConverged ? 'yes' : 'no'} |`,
    );
  }
  lines.push('');
  return lines.join('\n');
}

export async function runPhase3(sizes: readonly number[], maxWorkers: number): Promise<Phase3Report> {
  const specs = buildPhase3CellSpecs(sizes);
  // Largest cells first so the pool stays busy; rows are put back in spec order.
  const order = specs.map((_, i) => i).sort((a, b) => specs[b]!.numParams - specs[a]!.numParams || a - b);
  const jobs: Phase3CellSpec[] = order.map((i) => specs[i]!);
  const done = await runSeedBlocksParallel<Phase3CellSpec, Phase3CellRow>('./phase3-cell-worker.ts', jobs, maxWorkers);
  const rows: Phase3CellRow[] = new Array(specs.length);
  for (let j = 0; j < order.length; (j = (j + 1) | 0)) rows[order[j]!] = done[j]!;
  const report = assemblePhase3Report(rows, specs[0]!);
  const here = path.dirname(fileURLToPath(import.meta.url));
  const evidenceDir = path.resolve(here, '..', '..', 'evidence', 'mean-square-boundary');
  fs.mkdirSync(evidenceDir, { recursive: true });
  fs.writeFileSync(path.join(evidenceDir, 'raw-phase3.json'), `${JSON.stringify(report, null, 2)}\n`);
  fs.writeFileSync(path.join(evidenceDir, 'phase3-summary.md'), `${formatPhase3Markdown(report)}\n`);
  return report;
}

if (process.argv[1] !== undefined && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const sizes = (process.argv[2] ?? '64,256,1024').split(',').map((s) => Number.parseInt(s, 10));
  const workers = Number.parseInt(process.argv[3] ?? String(Math.max(1, os.availableParallelism() - 2)), 10);
  const t0 = Date.now();
  void runPhase3(sizes, workers).then((report) => {
    for (const [k, v] of Object.entries(report.predictions)) process.stdout.write(`${k}: ${String(v)}\n`);
    process.stdout.write(`cells ${report.cells.length}, workers ${workers}, elapsed ${((Date.now() - t0) / 1000).toFixed(1)} s\n`);
  });
}
