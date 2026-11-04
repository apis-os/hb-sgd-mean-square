/**
 * @file labs/mean-square-boundary/regenerate-phase5-checkpoint-iterates.ts
 * @description The training iterates of one Phase 5 run at its checkpoints (`regeneratePhase5CheckpointIterates`), for Phase 6:
 * the run records do not store `theta`, so the training of `runPhase5Run` is repeated (same network, initialisation stream
 * `createXorshift128Plus((0x14a00001 + 7919 seed) | 0, 0x2545f491)`, minibatch draws, update, loss and divergence rule, in the same
 * order of operations) without its measurements, and `onCheckpoint(t, theta, loss)` is called at every step `t` that is a multiple
 * of 200, with the loss exactly as the checkpoint record stores it (`evalFullLoss + 0.5 mu_train |theta|^2`). The caller compares
 * that loss bit for bit with the stored one; a mismatch means a different run, and stops the comparison. This repeats the loop of
 * `runPhase5Run` rather than sharing it, so that the registered runner stays as it ran; the bit-for-bit check detects any drift that
 * changes a stored loss, not every drift of the parameters (twelfth audit round: flipping the sign of a `tanh` unit's weights keeps
 * the losses identical; the records store no parameters). Returns the divergence step or `null`.
 * @cold
 */
import { createTwoLayerMlpDatasetAndOps } from './mlp-network.ts';
import { createXorshift128Plus } from './prng-xorshift128p.ts';

export function regeneratePhase5CheckpointIterates(
  eta: number,
  muTrain: number,
  seed: number,
  steps: number,
  onCheckpoint: (t: number, theta: Float64Array, loss: number) => boolean,
): number | null {
  const beta = 0.9;
  const batchB = 16;
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
  for (let t = 1; t <= steps; (t = (t + 1) | 0)) {
    for (let b = 0; b < batchB; (b = (b + 1) | 0)) idx[b] = Math.floor(prng.nextFloat01() * mlp.numSamples);
    mlp.computeMinibatchGradient(theta, idx, batchB, g);
    for (let a = 0; a < p; (a = (a + 1) | 0)) {
      const cur = theta[a]!;
      const next = cur + beta * (cur - prev[a]!) - eta * (g[a]! + muTrain * cur);
      prev[a] = cur;
      theta[a] = next;
    }
    let thetaSq = 0.0;
    for (let a = 0; a < p; (a = (a + 1) | 0)) thetaSq += theta[a]! * theta[a]!;
    const loss = mlp.evalFullLoss(theta) + 0.5 * muTrain * thetaSq;
    if (!Number.isFinite(loss) || loss > 1e3) return t;
    if (t % 200 !== 0) continue;
    if (!onCheckpoint(t, theta, loss)) return null;
  }
  return null;
}
