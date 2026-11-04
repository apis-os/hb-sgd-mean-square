/**
 * @file labs/mean-square-boundary/precompute-static-sample-grads.ts
 * @description Per-sample gradients of a two-layer MLP at its checkpoint, stored row-major as `numSamples x numParams` floats (`precomputeStaticSampleGrads`), so a minibatch gradient at the checkpoint is a sum of rows and not a forward pass.
 * @cold
 */
import type { createTwoLayerMlpDatasetAndOps } from './mlp-network.ts';

export function precomputeStaticSampleGrads(
  mlp: Pick<ReturnType<typeof createTwoLayerMlpDatasetAndOps>, 'numParams' | 'numSamples' | 'checkpointTheta0' | 'computeMinibatchGradient'>,
): Float64Array {
  const p = mlp.numParams;
  const sampleGrads0 = new Float64Array(Math.imul(mlp.numSamples, p));
  const oneIdx = new Int32Array(1);
  const gTmp = new Float64Array(p);
  for (let n = 0; n < mlp.numSamples; (n = (n + 1) | 0)) {
    oneIdx[0] = n;
    mlp.computeMinibatchGradient(mlp.checkpointTheta0, oneIdx, 1, gTmp);
    sampleGrads0.set(gTmp, Math.imul(n, p));
  }
  return sampleGrads0;
}
