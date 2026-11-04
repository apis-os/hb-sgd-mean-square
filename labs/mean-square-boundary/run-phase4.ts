/**
 * @file labs/mean-square-boundary/run-phase4.ts
 * @description Executes Phase 4 (`runPhase4`), the exact mean-square threshold of Theorem G.6 on tethered network
 * Hessians, writing `evidence/mean-square-boundary/raw-phase4.json` and `phase4-summary.md`; progress lines are appended to
 * `phase4-progress.log` as each cell finishes, so a run stopped midway keeps what it measured. The matrix kernels split rows
 * over `matrixWorkers` threads and the sample-path cells run on `trajectoryWorkers` threads; neither count changes a number.
 * Usage: `node --no-liftoff --experimental-strip-types labs/mean-square-boundary/run-phase4.ts [matrixWorkers]
 * [trajectoryWorkers] [seeds] [operatorMaxIterations3610] [resumeFile]` (defaults 4, 4, 4, 1500, none). `--no-liftoff` makes
 * the Wasm product run in TurboFan (speed only, no number changes). `resumeFile` is the output of
 * `run-phase4-recover-resume-cells.ts`; its cells skip their Lanczos run and the file is named in the provenance.
 * @cold
 */
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { evaluatePhase4TetheredMeanSquareCells } from './evaluate-phase4-tethered-mean-square-cells.ts';
import type { Phase4Report, Phase4ResumedLanczosCell } from './phase4-types.ts';

function fmtCert(c: Phase4Report['networks'][number]['operatorChecks'][number]['certificateBelow'], side: 'below' | 'above'): string {
  const decided = side === 'below' ? c.certifiedBelowOne : c.certifiedAboveOne;
  const range = c.lowerBound === null || c.upperBound === null ? '' : ` [${c.lowerBound.toPrecision(10)}, ${c.upperBound.toPrecision(10)}]`;
  return `${c.iteratePositiveDefinite ? '' : 'Q not PD; '}${decided ? 'decided' : 'not decided'}${range}`;
}

function formatPhase4Markdown(r: Phase4Report): string {
  const lines: string[] = [];
  lines.push('# Phase 4: the exact mean-square threshold of Theorem G.6 on tethered network Hessians');
  lines.push('');
  lines.push(`- **Date**: \`${r.date}\``);
  lines.push(`- **Node**: \`${r.nodeVersion}\``);
  lines.push(`- **Batch**: \`B = ${r.batchB}\`, tether grid \`g\` = ${r.tetherGrid.join(', ')}, \`mu = g + max(0, -lambda_min)\``);
  lines.push('');
  lines.push('## 1. Predictions (`t1`–`t6`, written before the run)');
  lines.push('');
  for (const [k, text] of Object.entries(r.predictionTexts)) lines.push(`- **\`${k}\`**: ${text}`);
  lines.push('');
  lines.push('### Outcomes');
  lines.push('');
  for (const [k, v] of Object.entries(r.predictions)) lines.push(`- \`${k}\`: \`${String(v)}\``);
  lines.push('');
  lines.push('### Implementation checks (added after the registration; consequences of the theorem, not predictions)');
  lines.push('');
  for (const [k, text] of Object.entries(r.implementationCheckTexts)) lines.push(`- **\`${k}\`** (\`${String(r.implementationChecks[k])}\`): ${text}`);
  for (const net of r.networks) {
    lines.push('');
    lines.push(`## ${net.networkId} (P = ${net.numParams}, N = ${net.numSamples})`);
    lines.push('');
    lines.push(`- Hessians: ${net.hessianSource}`);
    lines.push(`- \`lambda_1 = ${net.lambda1}\`, \`lambda_2 = ${net.lambda2}\`, \`lambda_min = ${net.lambdaMin}\`, ${net.negativeEigenvalues} negative eigenvalues, second smallest \`${net.lambdaSecond}\`, ${net.eigenvaluesBelowTwiceAbsMin} eigenvalues below \`2 |lambda_min|\`, \`tr H = ${net.traceH}\`, \`nu_11,B^2 = ${net.nuB11Sq}\``);
    lines.push('');
    lines.push('| beta | g | mu | lambda_min(H + mu) | eta_det | 1D ceiling | eta* (Thm G.6) | f* = eta* / eta_det | 1D / eta* | Lanczos steps | linearised / eta* | network / eta* | horizon informative |');
    lines.push('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|');
    for (const th of net.thresholds) {
      const lin = net.trajectories.find((t) => t.beta === th.beta && t.tetherGrid === th.tetherGrid && t.mode === 'linearised');
      const nw = net.trajectories.find((t) => t.beta === th.beta && t.tetherGrid === th.tetherGrid && t.mode === 'network');
      lines.push(
        `| ${th.beta} | ${th.tetherGrid} | ${th.mu.toPrecision(6)} | ${th.lambdaMinTethered.toPrecision(6)} | ${th.etaDeterministic.toPrecision(6)} | ${th.etaOneDim.toPrecision(6)} | ${th.etaThmG6.toPrecision(8)} | ${th.fractionOfDeterministic.toPrecision(6)} | ${th.oneDimOverThmG6.toFixed(3)} | ${th.lanczosIterations}${th.lanczosConverged ? '' : ' (not converged)'} | ${lin?.medianOverThmG6.toFixed(3) ?? ''} | ${nw?.medianOverThmG6.toFixed(3) ?? ''} | ${lin?.horizonInformative ? 'yes' : 'no'} |`,
      );
    }
    lines.push('');
    lines.push('| beta | g | eta* | rho(T) at (1 - d) eta* | rho(T) at (1 + d) eta* | 1 - rho below | rho above - 1 | steps | converged | crosses | certificate below | certificate above | horizon (multiplier: rho, steps to 2.5x) |');
    lines.push('|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|');
    for (const oc of net.operatorChecks) {
      lines.push(
        `| ${oc.beta} | ${oc.tetherGrid} | ${oc.etaThmG6.toPrecision(8)} | ${oc.radiusBelow.toFixed(10)} | ${oc.radiusAbove.toFixed(10)} | ${(1 - oc.radiusBelow).toExponential(2)} | ${(oc.radiusAbove - 1).toExponential(2)} | ${oc.iterationsBelow}, ${oc.iterationsAbove} | ${oc.convergedBelow && oc.convergedAbove ? 'yes' : 'no'} | ${oc.crossesAtThmG6 ? 'yes' : 'no'} | ${fmtCert(oc.certificateBelow, 'below')} | ${fmtCert(oc.certificateAbove, 'above')} | ${oc.horizon.map((h) => `${h.multiplier}: ${h.radius.toFixed(10)}, ${Number.isFinite(h.stepsToGrow2p5) ? h.stepsToGrow2p5.toFixed(0) : 'inf'}`).join('; ')}${oc.fallback === null ? '' : `; fallback ±${oc.fallback.relOffset}: ${oc.fallback.radiusBelow.toFixed(8)} / ${oc.fallback.radiusAbove.toFixed(8)}, ${fmtCert(oc.fallback.certificateBelow, 'below')} / ${fmtCert(oc.fallback.certificateAbove, 'above')}`} |`,
      );
    }
    lines.push('');
    for (const m of net.interiorMaxima) lines.push(`- Interior maxima of eta* along the grid, beta ${m.beta}: ${m.tetherGrid.length === 0 ? 'none' : m.tetherGrid.map((g) => `g = ${g}`).join(', ')}`);
    for (const s of net.collapseSlopes) lines.push(`- Collapse slope (log eta* against log lambda_min(H + mu), two smallest g), beta ${s.beta}: \`${s.slope}\``);
  }
  lines.push('');
  return lines.join('\n');
}

export async function runPhase4(
  matrixWorkers: number,
  trajectoryWorkers: number,
  seeds: number,
  operatorMaxIterations3610: number,
  resumeFile?: string,
): Promise<Phase4Report> {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const evidenceDir = path.resolve(here, '..', '..', 'evidence', 'mean-square-boundary');
  fs.mkdirSync(evidenceDir, { recursive: true });
  const startCommit = (() => {
    try {
      return execFileSync('git', ['rev-parse', 'HEAD'], { cwd: here, encoding: 'utf8' }).trim();
    } catch {
      return 'unknown';
    }
  })();
  const startDirty = (() => {
    try {
      return execFileSync('git', ['status', '--porcelain', '--', 'labs/mean-square-boundary', 'src'], { cwd: path.resolve(here, '..', '..'), encoding: 'utf8' }).trim().length > 0;
    } catch {
      return true;
    }
  })();
  const progress = path.join(evidenceDir, 'phase4-progress.log');
  fs.writeFileSync(progress, '');
  const log = (line: string): void => {
    const stamped = `${new Date().toISOString()} ${line}\n`;
    fs.appendFileSync(progress, stamped);
    process.stdout.write(stamped);
  };
  const resume = resumeFile === undefined ? undefined : (JSON.parse(fs.readFileSync(resumeFile, 'utf8')) as { readonly cells: readonly Phase4ResumedLanczosCell[] });
  const report = await evaluatePhase4TetheredMeanSquareCells(
    { matrixWorkers, trajectoryWorkers, trajectorySeeds: seeds, operatorMaxIterations3610, ...(resume === undefined ? {} : { resumeCells3610: resume.cells }) },
    log,
  );
  // The commit that produced the file, and whether the tree had uncommitted changes when the run started.
  const git = (args: string[]): string => {
    try {
      return execFileSync('git', args, { cwd: here, encoding: 'utf8' }).trim();
    } catch {
      return 'unknown';
    }
  };
  const provenance = { gitCommit: startCommit, gitDirtyAtStart: startDirty, gitCommitAtEnd: git(['rev-parse', 'HEAD']), resumeFile: resumeFile ?? null };
  fs.writeFileSync(path.join(evidenceDir, 'raw-phase4.json'), `${JSON.stringify({ ...report, provenance }, null, 2)}\n`);
  fs.writeFileSync(path.join(evidenceDir, 'phase4-summary.md'), formatPhase4Markdown(report));
  return report;
}

if (process.argv[1] !== undefined && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const matrixWorkers = Number.parseInt(process.argv[2] ?? '4', 10);
  const trajectoryWorkers = Number.parseInt(process.argv[3] ?? '4', 10);
  const seeds = Number.parseInt(process.argv[4] ?? '4', 10);
  const opMax = Number.parseInt(process.argv[5] ?? '1500', 10);
  const t0 = Date.now();
  void runPhase4(matrixWorkers, trajectoryWorkers, seeds, opMax, process.argv[6]).then((report) => {
    for (const [k, v] of Object.entries(report.predictions)) process.stdout.write(`${k}: ${String(v)}\n`);
    process.stdout.write(`elapsed ${((Date.now() - t0) / 1000).toFixed(1)} s\n`);
  });
}
