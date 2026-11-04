/**
 * @file labs/mean-square-boundary/unpack-eigenbasis-deviations-dense.ts
 * @description The packed eigenbasis deviations of `buildEigenbasisSampleHessianDeviations` (or of `restrictEigenbasisDeviations`)
 * as full symmetric `n x n` matrices, sample-major (`unpackEigenbasisDeviationsDense`; entry `(a, b)` of sample `i` at
 * `i n^2 + a n + b`, both triangles from the same packed value), the input layout of `createDenseDeviationNoiseOperator`.
 * @cold
 */
export function unpackEigenbasisDeviationsDense(dev: {
  readonly n: number;
  readonly numSamples: number;
  readonly nPair: number;
  readonly packed: Float64Array;
  readonly pairIndex: Int32Array;
}): Float64Array {
  const n = dev.n;
  const nn = n * n;
  const out = new Float64Array(dev.numSamples * nn);
  for (let i = 0; i < dev.numSamples; i++) {
    const src = i * dev.nPair;
    const dst = i * nn;
    for (let a = 0; a < n; a++) {
      for (let b = 0; b < n; b++) out[dst + a * n + b] = dev.packed[src + dev.pairIndex[a * n + b]!]!;
    }
  }
  return out;
}
