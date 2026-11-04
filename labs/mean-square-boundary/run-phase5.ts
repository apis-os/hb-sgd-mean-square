/**
 * @file labs/mean-square-boundary/run-phase5.ts
 * @description One registered Phase 5 run (`runPhase5Run`): the `P = 33` network and data of Phase 4, the initialisation of the
 * pilot (`W_1 ~ N(0, 1/d_in)`, `W_2 ~ N(0, 1/d_hidden)`, biases zero, seed-dependent stream), heavy-ball SGD
 * `theta_{t+1} = theta_t + beta (theta_t - theta_{t-1}) - eta (g_{B_t}(theta_t) + mu_train theta_t)`, `beta = 0.9`, `B = 16` with
 * replacement. The full loss is checked at every step (divergent: non-finite or above `1e3`, recorded with its step). At steps
 * `200, 400, ..., steps` the checkpoint record of `evaluatePhase5Checkpoint` (with the last 200 steps for Proposition E13); for
 * `eta >= 0.1` the `|H|` almost-sure threshold at steps `11 000, 12 000, ..., 20 000` (amendment 1, item 2); at the final point
 * the analysis of `analyzePhase5FinalPoint`. CLI: `node --no-liftoff --experimental-strip-types run-phase5.ts <eta> <mu_train>
 * <seed> [steps] [out.json] [lyapunovSteps]`; defaults 20 000 steps, `2e5` Lyapunov steps, output
 * `evidence/mean-square-boundary/phase5-runs/eta<eta>-mu<mu>-seed<seed>.json`. Smoke tests use other seeds and fewer steps
 * (amendment 2, item 6).
 * @cold
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { analyzePhase5FinalPoint } from './analyze-phase5-final-point.ts';
import { evaluatePhase5Checkpoint } from './evaluate-phase5-checkpoint.ts';
import { isPhase5C6Seed } from './is-phase5-c6-seed.ts';
import { createPhase4SampleHessianProvider } from './create-phase4-sample-hessian-provider.ts';
import { createTwoLayerMlpDatasetAndOps } from './mlp-network.ts';
import { createXorshift128Plus } from './prng-xorshift128p.ts';

// c6 (identity (vii)): at B = 1 the minibatch gradient is g_i = r_i grad f_i and the Gauss-Newton minibatch Hessian is
// grad f_i grad f_i^T, so the per-batch ratio is |grad f_i|^2 and its expectation is tr H_GN exactly; also a 256-draw Monte Carlo.
function checkC6(mlp: ReturnType<typeof createTwoLayerMlpDatasetAndOps>, theta: Float64Array, seed: number): Record<string, number> {
  const p = mlp.numParams;
  const nS = mlp.numSamples;
  const gn = createPhase4SampleHessianProvider({ kind: 'mlp33', theta: Float64Array.from(theta), gaussNewton: true });
  const one = new Int32Array(1);
  const gi = new Float64Array(p);
  const hg = new Float64Array(p);
  const e = new Float64Array(p);
  const ratios = new Float64Array(nS);
  let traceGn = 0.0;
  for (let i = 0; i < nS; (i = (i + 1) | 0)) {
    one[0] = i;
    mlp.computeMinibatchGradient(theta, one, 1, gi);
    gn.applySampleHessian(i, gi, hg);
    let num = 0.0;
    let den = 0.0;
    for (let a = 0; a < p; (a = (a + 1) | 0)) { num += gi[a]! * hg[a]!; den += gi[a]! * gi[a]!; }
    ratios[i] = num / den;
    for (let a = 0; a < p; (a = (a + 1) | 0)) {
      e.fill(0.0);
      e[a] = 1.0;
      gn.applySampleHessian(i, e, hg);
      traceGn += hg[a]! / nS;
    }
  }
  let exact = 0.0;
  for (let i = 0; i < nS; (i = (i + 1) | 0)) exact += ratios[i]! / nS;
  const probe = createXorshift128Plus((0x14e00001 + Math.imul(seed, 104729)) | 0, 0x85ebca6b);
  let mc = 0.0;
  let mc2 = 0.0;
  for (let r = 0; r < 256; (r = (r + 1) | 0)) {
    const x = ratios[Math.floor(probe.nextFloat01() * nS) | 0]!;
    mc += x / 256;
    mc2 += (x * x) / 256;
  }
  return { traceGaussNewton: traceGn, exactExpectationOfRatio: exact, relativeDifference: Math.abs(exact - traceGn) / traceGn, monteCarloMean: mc, monteCarloStandardError: Math.sqrt(Math.max(0.0, mc2 - mc * mc) / 255) };
}

export async function runPhase5Run(eta: number, muTrain: number, seed: number, steps: number, lyapunovSteps: number, log: (s: string) => void): Promise<Record<string, unknown>> {
  const beta = 0.9;
  const batchB = 16;
  const history = 200;
  const mlp = createTwoLayerMlpDatasetAndOps();
  const p = mlp.numParams;
  const dIn = mlp.dIn;
  const dH = mlp.dHidden;
  const prng = createXorshift128Plus((0x14a00001 + Math.imul(seed, 7919)) | 0, 0x2545f491);
  const theta = new Float64Array(p);
  for (let i = 0; i < Math.imul(dIn, dH); (i = (i + 1) | 0)) theta[i] = prng.nextGaussian() / Math.sqrt(dIn);
  for (let h = 0; h < dH; (h = (h + 1) | 0)) theta[(Math.imul(dIn, dH) + dH + h) | 0] = prng.nextGaussian() / Math.sqrt(dH);
  const prev = new Float64Array(theta);
  const g = new Float64Array(p);
  const idx = new Int32Array(batchB);
  const ring = new Float64Array(Math.imul(history, p));
  let ringCount = 0;
  const checkpoints: Array<Record<string, unknown>> = [];
  const window: Array<Record<string, unknown>> = [];
  let divergedAt: number | null = null;
  let last: Awaited<ReturnType<typeof evaluatePhase5Checkpoint>> | null = null;
  const windowSteps = new Set<number>();
  if (eta >= 0.1) for (let t = 11000; t <= 20000; t += 1000) windowSteps.add(t);
  for (let t = 1; t <= steps; (t = (t + 1) | 0)) {
    for (let b = 0; b < batchB; (b = (b + 1) | 0)) idx[b] = Math.floor(prng.nextFloat01() * mlp.numSamples);
    mlp.computeMinibatchGradient(theta, idx, batchB, g);
    const slot = Math.imul(ringCount % history, p);
    for (let a = 0; a < p; (a = (a + 1) | 0)) {
      const cur = theta[a]!;
      const next = cur + beta * (cur - prev[a]!) - eta * (g[a]! + muTrain * cur);
      prev[a] = cur;
      theta[a] = next;
      ring[(slot + a) | 0] = next - cur;
    }
    ringCount = (ringCount + 1) | 0;
    let thetaSq = 0.0;
    for (let a = 0; a < p; (a = (a + 1) | 0)) thetaSq += theta[a]! * theta[a]!;
    const loss = mlp.evalFullLoss(theta) + 0.5 * muTrain * thetaSq;
    if (!Number.isFinite(loss) || loss > 1e3) {
      divergedAt = t;
      log(`diverged at step ${t}`);
      break;
    }
    if (t % 200 !== 0) continue;
    let deltas: Float64Array | null = null;
    if (ringCount >= history) {
      deltas = new Float64Array(Math.imul(history, p));
      for (let s = 0; s < history; (s = (s + 1) | 0)) {
        const src = Math.imul((ringCount - history + s) % history, p);
        deltas.set(ring.subarray(src, src + p), Math.imul(s, p));
      }
    }
    last = await evaluatePhase5Checkpoint(mlp, theta, t, muTrain, eta, beta, seed, deltas);
    checkpoints.push(last.record);
    if (t % 2000 === 0) log(`step ${t}: lambda1 ${String(last.record.lambda1)}, eta*_|H|/eta ${String((last.record.absH as { ratioToEta: number }).ratioToEta)}`);
    if (windowSteps.has(t)) {
      const etaStarAbs = (last.record.absH as { etaStar: number }).etaStar;
      const as = await analyzePhase5FinalPoint(last.full, last.abs, etaStarAbs, eta, beta, seed + t, true, lyapunovSteps);
      window.push({ step: t, etaStarAbs, ...as });
      log(`window step ${t}: eta_a.s.,|H| ${String((as.almostSureAbs as { etaAs: number | null }).etaAs)}`);
    }
  }
  let finalPoint: Record<string, unknown> | null = null;
  if (divergedAt === null && last !== null) {
    finalPoint = await analyzePhase5FinalPoint(last.full, last.abs, (last.record.absH as { etaStar: number }).etaStar, eta, beta, seed, false, lyapunovSteps);
    log('final point done');
  }
  const c6 = divergedAt === null && eta === 0.01 && muTrain === 0 && isPhase5C6Seed(seed) ? checkC6(mlp, theta, seed) : null;
  return { eta, muTrain, seed, beta, batchB, steps, lyapunovSteps, divergedAt, checkpoints, window, finalPoint, c6 };
}

if (process.argv[1] !== undefined && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const eta = Number(process.argv[2]);
  const mu = Number(process.argv[3]);
  const seed = Number.parseInt(process.argv[4] ?? '1', 10);
  const steps = Number.parseInt(process.argv[5] ?? '20000', 10);
  const here = path.dirname(fileURLToPath(import.meta.url));
  const out = process.argv[6] ?? path.resolve(here, '..', '..', 'evidence', 'mean-square-boundary', 'phase5-runs', `eta${eta}-mu${mu}-seed${seed}.json`);
  const lyap = Number.parseInt(process.argv[7] ?? '200000', 10);
  const t0 = Date.now();
  void runPhase5Run(eta, mu, seed, steps, lyap, (s) => process.stdout.write(`${new Date().toISOString()} eta ${eta} mu ${mu} seed ${seed}: ${s}\n`)).then((r) => {
    fs.mkdirSync(path.dirname(out), { recursive: true });
    fs.writeFileSync(out, `${JSON.stringify({ labId: 'mean-square-boundary-phase5-run', date: new Date().toISOString(), nodeVersion: process.version, elapsedSeconds: (Date.now() - t0) / 1000, ...r })}\n`);
    process.stdout.write(`written ${out}\n`);
  });
}
