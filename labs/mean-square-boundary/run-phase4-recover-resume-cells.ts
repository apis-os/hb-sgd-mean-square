/**
 * @file labs/mean-square-boundary/run-phase4-recover-resume-cells.ts
 * @description Collects the Lanczos outcomes of the `X1_P3610` cells from the progress log of an interrupted Phase 4 launch
 * (`runPhase4RecoverResumeCells`, by `parsePhase4CellLogLine`) into `evidence/mean-square-boundary/phase4-resume-x1-p3610.json`,
 * the file `run-phase4.ts` takes as its resume argument. Arguments: the log path (relative to the repository root) and the
 * commit of the launch that wrote it.
 * @cold
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parsePhase4CellLogLine } from './parse-phase4-cell-log-line.ts';

export function runPhase4RecoverResumeCells(logPath: string, launchCommit: string): object {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const root = path.resolve(here, '..', '..');
  const lines = fs.readFileSync(path.resolve(root, logPath), 'utf8').split('\n');
  const cells = [];
  for (let i = 0; i < lines.length; (i = (i + 1) | 0)) {
    const c = parsePhase4CellLogLine(lines[i]!);
    if (c === null || c.networkId !== 'X1_P3610') continue;
    const { networkId: _id, ...cell } = c;
    cells.push(cell);
  }
  const report = { labId: 'mean-square-boundary-phase4-resume-x1-p3610', date: new Date().toISOString(), sourceLog: logPath, launchCommit, cells };
  fs.writeFileSync(path.resolve(root, 'evidence', 'mean-square-boundary', 'phase4-resume-x1-p3610.json'), `${JSON.stringify(report, null, 2)}\n`);
  process.stdout.write(`${cells.length} cells recovered\n`);
  return report;
}

if (process.argv[1] !== undefined && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  runPhase4RecoverResumeCells(process.argv[2] ?? 'evidence/mean-square-boundary/phase4-progress-launch6-crashed.log', process.argv[3] ?? 'unknown');
}
