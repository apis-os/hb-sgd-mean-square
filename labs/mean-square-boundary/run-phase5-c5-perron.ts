/**
 * @file labs/mean-square-boundary/run-phase5-c5-perron.ts
 * @description Amendment 4 (post hoc): for each named Phase 5 run, rebuilds `theta_T` from the training loop of `run-phase5.ts`
 * (checked bit for bit against the recorded final checkpoint), lists the smallest `|lambda_k|` of the kept `|H|` modes with the
 * second-moment eigenvalue `1 - 2 eta |lambda| / (1 - beta)` they carry, and applies `certifySecondMomentRadiusPerron` at
 * `(1 -+ 1e-3) eta*_|H|` with shift `max(rho_power, 1) + 1e-5`, beside the recorded bracket and witness decisions. Writes
 * `evidence/mean-square-boundary/phase5-c5-perron.json`. CLI: run names (`eta<eta>-mu<mu>-seed<seed>`).
 * @cold
 */
import fs from 'node:fs';
import { performance } from 'node:perf_hooks';
const { createTwoLayerMlpDatasetAndOps } = await import('./mlp-network.ts');
const { createXorshift128Plus } = await import('./prng-xorshift128p.ts');
const { evaluatePhase5Checkpoint } = await import('./evaluate-phase5-checkpoint.ts');
const { assembleCurvatureNoiseOperatorMatrix } = await import('./assemble-curvature-noise-operator-matrix.ts');
const { certifySecondMomentRadiusPerron } = await import('./certify-second-moment-radius-perron.ts');
const mlp = createTwoLayerMlpDatasetAndOps();
const out: unknown[] = [];
for (const name of process.argv.slice(2)) {
  const rec = JSON.parse(fs.readFileSync(new URL(`../../evidence/mean-square-boundary/phase5-runs/${name}.json`, import.meta.url), 'utf8'));
  const { eta, muTrain, seed } = rec; const beta = 0.9; const p = mlp.numParams;
  const prng = createXorshift128Plus((0x14a00001 + Math.imul(seed, 7919)) | 0, 0x2545f491);
  const theta = new Float64Array(p);
  for (let i = 0; i < mlp.dIn * mlp.dHidden; i++) theta[i] = prng.nextGaussian() / Math.sqrt(mlp.dIn);
  for (let h = 0; h < mlp.dHidden; h++) theta[mlp.dIn * mlp.dHidden + mlp.dHidden + h] = prng.nextGaussian() / Math.sqrt(mlp.dHidden);
  const prev = new Float64Array(theta); const g = new Float64Array(p); const idx = new Int32Array(16);
  for (let t = 1; t <= 20000; t++) {
    for (let b = 0; b < 16; b++) idx[b] = Math.floor(prng.nextFloat01() * mlp.numSamples);
    mlp.computeMinibatchGradient(theta, idx, 16, g);
    for (let a = 0; a < p; a++) { const cur = theta[a]!; const next = cur + beta * (cur - prev[a]!) - eta * (g[a]! + muTrain * cur); prev[a] = cur; theta[a] = next; }
  }
  const ev = await evaluatePhase5Checkpoint(mlp, theta, 20000, muTrain, eta, beta, seed, null);
  const last = rec.checkpoints[rec.checkpoints.length - 1];
  const same = Object.is((ev.record.absH as { etaStar: number }).etaStar, last.absH.etaStar);
  const abs = ev.abs; const n = abs.dev.n;
  const l1 = Math.max(...Array.from(abs.values, (v: number) => Math.abs(v)));
  const flat = Array.from(abs.values, (v: number) => ({ rel: Math.abs(v) / l1, oneMinusEig: (2 * (1 - 1e-3) * last.absH.etaStar * Math.abs(v)) / (1 - beta) })).sort((a, b) => a.rel - b.rel).slice(0, 3);
  const vmat = assembleCurvatureNoiseOperatorMatrix(abs.gram, n, abs.dev.nPair, abs.dev.pairIndex, abs.dev.pairA, abs.dev.pairB);
  for (const [side, key, s] of [['below', 'operatorBelow', -1], ['above', 'operatorAbove', 1]] as const) {
    const o = rec.finalPoint[key];
    const sigma = Math.max(o.radius, 1.0) + 1e-5;
    const t0 = performance.now();
    const r = certifySecondMomentRadiusPerron(vmat, n, abs.dev.nPair, abs.dev.pairA, abs.dev.pairB, abs.curvature, beta, (1 + s * 1e-3) * last.absH.etaStar, sigma, 4000) as Record<string, unknown>;
    const c = r.certificate as { iteratePositiveDefinite: boolean; certifiedBelowOne: boolean; certifiedAboveOne: boolean; lowerBound: number | null; upperBound: number | null } | null;
    const registered = side === 'below' ? o.iterateCertificate.certifiedBelowOne : (o.iterateCertificate.certifiedAboveOne || o.aboveDeterministicThreshold);
    const witness = o.witness === null ? null : (side === 'below' ? o.witness.certificate.certifiedBelowOne : o.witness.certificate.certifiedAboveOne);
    out.push({ name, side, thetaReproduced: same, smallestKeptModes: side === 'below' ? flat : undefined, n, dim: r.dim, it: r.iterations, est1m: 1 - (r.eigenvalueEstimate as number), pd: c?.iteratePositiveDefinite, below: c?.certifiedBelowOne, above: c?.certifiedAboveOne, lb1m: c?.lowerBound == null ? null : 1 - c.lowerBound, ub1m: c?.upperBound == null ? null : 1 - c.upperBound, registered, witness, powerRadius1m: 1 - o.radius, sec: +((performance.now() - t0) / 1000).toFixed(1) });
    console.log(JSON.stringify(out[out.length - 1]));
  }
}
fs.writeFileSync(new URL('../../evidence/mean-square-boundary/phase5-c5-perron.json', import.meta.url), `${JSON.stringify({ labId: 'mean-square-boundary-phase5-c5-perron', date: new Date().toISOString(), nodeVersion: process.version, sides: out }, null, 1)}\n`);
