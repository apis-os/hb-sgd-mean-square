/**
 * @file labs/mean-square-boundary/regenerate-phase6-pilot-checkpoint-iterates.ts
 * @description The training iterates of one Phase 6 pilot run at its checkpoints (`regeneratePhase6PilotCheckpointIterates`), for
 * Phase 6 stage 2: the pilot records do not store `theta`, so the training of `runPhase6Pilot` is repeated without its
 * measurements: the data and student of `PHASE6_NETWORK_CONFIGS` (`createScaleMlpDatasetAndOps`, `createPhase6StudentNetwork`), the
 * initialisation from the training stream `createXorshift128Plus((0x15a00001 + 7919 seed) | 0, 0x2545f491)`, the minibatch draws and
 * the heavy-ball update in the same order of operations, and the divergence rule at the steps where the pilot checks it. At every
 * checkpoint step `onCheckpoint(t, theta, net, normFirst, normOut)` receives the first-layer and output-layer squared norms summed
 * as `evaluatePhase6Checkpoint` sums them; the caller compares them, and the data loss, bit for bit with the stored record (the
 * detector of drift that changes those scalars, not of every drift of the parameters: a sign flip of a `tanh` unit keeps them
 * identical; twelfth audit round). Returns the divergence step or `null`.
 * @cold
 */
import { createPhase6StudentNetwork } from './create-phase6-student-network.ts';
import { PHASE6_NETWORK_CONFIGS } from './phase6-network-configs.ts';
import { createXorshift128Plus } from './prng-xorshift128p.ts';
import { createScaleMlpDatasetAndOps } from './scale-mlp-network.ts';

export function regeneratePhase6PilotCheckpointIterates(
  network: 3610 | 12826 | 34186,
  activation: 'tanh' | 'leakyRelu01' | 'softsign',
  eta: number,
  muTrain: number,
  seed: number,
  steps: number,
  onCheckpoint: (t: number, theta: Float64Array, net: ReturnType<typeof createPhase6StudentNetwork>, normFirst: number, normOut: number) => boolean,
): number | null {
  const cfg = PHASE6_NETWORK_CONFIGS[String(network) as '3610' | '12826' | '34186'];
  const data = createScaleMlpDatasetAndOps(cfg.dIn, cfg.dHidden, cfg.dOut, cfg.numSamples, cfg.seedLo, cfg.seedHi);
  const net = createPhase6StudentNetwork(data.xData, data.yData, cfg.dIn, cfg.dHidden, cfg.dOut, cfg.numSamples, activation);
  const checkEvery = cfg.divergenceCheckEvery | 0;
  const cpEvery = cfg.checkpointEvery | 0;
  const p = net.numParams | 0;
  const beta = 0.9;
  const batchB = 16;
  const prng = createXorshift128Plus((0x15a00001 + Math.imul(seed, 7919)) | 0, 0x2545f491);
  const theta = new Float64Array(p);
  for (let i = 0; i < Math.imul(net.dIn, net.dHidden); (i = (i + 1) | 0)) theta[i] = prng.nextGaussian() / Math.sqrt(net.dIn);
  for (let i = 0; i < Math.imul(net.dOut, net.dHidden); (i = (i + 1) | 0)) theta[(net.w2Base + i) | 0] = prng.nextGaussian() / Math.sqrt(net.dHidden);
  const prev = new Float64Array(theta);
  const g = new Float64Array(p);
  const idx = new Int32Array(batchB);
  for (let t = 1; t <= steps; (t = (t + 1) | 0)) {
    for (let b = 0; b < batchB; (b = (b + 1) | 0)) idx[b] = Math.floor(prng.nextFloat01() * net.numSamples) | 0;
    net.computeMinibatchGradient(theta, idx, batchB, g);
    let thetaSq = 0.0;
    for (let a = 0; a < p; (a = (a + 1) | 0)) {
      const cur = theta[a]!;
      const next = cur + beta * (cur - prev[a]!) - eta * (g[a]! + muTrain * cur);
      prev[a] = cur;
      theta[a] = next;
      thetaSq += next * next;
    }
    if (t % checkEvery !== 0 && t % cpEvery !== 0 && Number.isFinite(thetaSq)) continue;
    const loss = net.evalLoss(theta) + 0.5 * muTrain * thetaSq;
    if (!Number.isFinite(loss) || loss > 1e3) return t;
    if (t % cpEvery !== 0) continue;
    let normFirst = 0.0;
    let normOut = 0.0;
    for (let a = 0; a < p; (a = (a + 1) | 0)) {
      const x = theta[a]! * theta[a]!;
      if (a < net.w2Base) normFirst += x;
      else normOut += x;
    }
    if (!onCheckpoint(t, theta, net, normFirst, normOut)) return null;
  }
  return null;
}
