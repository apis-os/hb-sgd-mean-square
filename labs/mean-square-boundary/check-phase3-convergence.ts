/**
 * @file labs/mean-square-boundary/check-phase3-convergence.ts
 * @description Post hoc (2026-11-01, after a reviewer asked which estimate failed): reruns, for the six Phase 3 cells whose
 * row has `allConverged` false (`P = 64`, `beta = 0.9`, `B` in {4, 16, 64}, Gaussian and Rademacher features), the four
 * power iterations that `evaluatePhase3OperatorRadiusCell` combines into that flag (`checkPhase3Convergence`): the symmetric
 * radius at `(1 - 1e-3) eta*`, `eta*` and `(1 + 1e-3) eta*` from the identity (tolerance `1e-13`, at most `200 000` steps), and
 * the Gelfand estimate on the antisymmetric matrices over the fixed horizon `4 096` from a random antisymmetric start. Prints one
 * line per cell with each flag and step count. Run: `node labs/mean-square-boundary/check-phase3-convergence.ts`.
 * @cold
 */
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { measureSecondMomentSpectralRadius } from './measure-second-moment-spectral-radius.ts';
import { solveSecularEossThreshold } from './solve-secular-eoss-threshold.ts';

export function checkPhase3Convergence(): string[] {
  const p = 64;
  const n = 128;
  const beta = 0.9;
  const lam = new Float64Array(p);
  for (let k = 0; k < p; k++) lam[k] = 1 / Math.sqrt(k + 1);
  const lines: string[] = [];
  for (const batchB of [4, 16, 64]) {
    for (const kappaFeat of [0, -2]) {
      const eta = solveSecularEossThreshold(lam, beta, batchB, kappaFeat, 0).etaCritSecular;
      const q = new Float64Array(n * n);
      const s = new Float64Array(n * n);
      const sym: string[] = [];
      for (const e of [eta * (1 - 1e-3), eta, eta * (1 + 1e-3)]) {
        q.fill(0);
        for (let i = 0; i < n; i++) q[i * n + i] = 1 / Math.sqrt(n);
        const r = measureSecondMomentSpectralRadius(lam, beta, e, batchB, kappaFeat, q, s, 0, 200000, 1e-13);
        sym.push(`${r.converged}/${r.iterations}`);
      }
      q.fill(0);
      let normSq = 0;
      let x = 12345;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          x = (x * 1103515245 + 12345) % 2147483648;
          const v = x / 2147483648 - 0.5;
          q[i * n + j] = v;
          q[j * n + i] = -v;
          normSq += 2 * v * v;
        }
      }
      for (let i = 0; i < n * n; i++) q[i] = q[i]! / Math.sqrt(normSq);
      const anti = measureSecondMomentSpectralRadius(lam, beta, eta, batchB, kappaFeat, q, s, 1, 4096, 1e-13);
      lines.push(`B=${batchB} kappaFeat=${kappaFeat} symmetric below/root/above converged/steps ${sym.join(' ')} antisymmetric converged ${anti.converged} radius ${anti.radius.toFixed(6)}`);
    }
  }
  return lines;
}

if (process.argv[1] !== undefined && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  for (const line of checkPhase3Convergence()) process.stdout.write(`${line}\n`);
}
