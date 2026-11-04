/**
 * @file labs/mean-square-boundary/sum-precomputed-sample-grads.ts
 * @description Minibatch gradient at the checkpoint as the mean of precomputed per-sample gradient rows (`sumPrecomputedSampleGrads`); with `batchSize = 16` the factor `1 / 16` is exact in binary, so the result is bit-identical to the former hard-coded `0.0625` sum.
 * @cold
 */
export function sumPrecomputedSampleGrads(

  sampleGrads0: Float64Array,
  batchIdx: Int32Array,
  batchSize: number,
  p: number,
  outGrad: Float64Array,
): void {
  outGrad.fill(0.0);
  const invB = 1.0 / batchSize;
  for (let b = 0; b < batchSize; (b = (b + 1) | 0)) {
    const off = Math.imul(batchIdx[b]!, p);
    for (let i = 0; i < p; (i = (i + 1) | 0)) {
      outGrad[i] = outGrad[i]! + sampleGrads0[(off + i) | 0]!;
    }
  }
  for (let i = 0; i < p; (i = (i + 1) | 0)) {
    outGrad[i] = outGrad[i]! * invB;
  }
}
