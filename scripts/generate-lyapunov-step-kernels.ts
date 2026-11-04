/**
 * @file scripts/generate-lyapunov-step-kernels.ts
 * @description Writes (or with `--check` verifies) `labs/mean-square-boundary/step-eigenbasis-lyapunov-g<G>.gen.ts` for `G = 1, 2, 3`
 * from `buildLyapunovStepKernelSource`. Exit 1 on drift.
 * @cold
 */
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildLyapunovStepKernelSource } from './build-lyapunov-step-kernel-source.ts';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const check = process.argv.includes('--check');
let drift = false;
for (const g of [1, 2, 3]) {
  const rel = `labs/mean-square-boundary/step-eigenbasis-lyapunov-g${g}.gen.ts`;
  const target = path.join(rootDir, rel);
  const source = buildLyapunovStepKernelSource(g);
  if (check) {
    const existing = fs.existsSync(target) ? fs.readFileSync(target, 'utf8') : '';
    if (existing !== source) {
      console.error(`${rel} differs from regeneration (drift)`);
      drift = true;
    }
  } else {
    fs.writeFileSync(target, source, 'utf8');
    console.log(`wrote ${rel}`);
  }
}
if (drift) process.exit(1);
if (check) console.log('Lyapunov step kernels are up to date');
