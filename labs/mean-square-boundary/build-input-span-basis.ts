/**
 * @file labs/mean-square-boundary/build-input-span-basis.ts
 * @description An orthonormal basis of `S = span{(x_i, 1)}` in `R^{dIn + 1}` (`buildInputSpanBasis`): modified Gram–Schmidt
 * applied twice to each `(x_i, 1)` in sample order, a vector being dropped when its residual norm is below `1e-10` of its own
 * norm. Row `k` of `basis` (length `dIn + 1`) is the `k`-th basis vector, `size` the number kept. Every per-sample gradient and
 * Hessian product of the two-layer network reads and writes a first-layer row `(W1[h, :], b1[h])` only along `S`, so `S` per
 * hidden unit, with the whole output layer, is the data-active subspace (`computeActiveNullWeightSplit`,
 * `createActiveSubspaceProjector`).
 * @cold
 */
export function buildInputSpanBasis(xData: Float64Array, numSamples: number, dIn: number): { readonly basis: Float64Array; readonly size: number; readonly dim: number } {
  const d = (dIn + 1) | 0;
  const basis = new Float64Array(Math.imul(Math.min(numSamples, d), d));
  const w = new Float64Array(d);
  let size = 0;
  for (let i = 0; i < numSamples && size < d; (i = (i + 1) | 0)) {
    const xOff = Math.imul(i, dIn);
    let norm0 = 1.0;
    for (let j = 0; j < dIn; (j = (j + 1) | 0)) { w[j] = xData[(xOff + j) | 0]!; norm0 += w[j]! * w[j]!; }
    w[dIn] = 1.0;
    for (let pass = 0; pass < 2; (pass = (pass + 1) | 0)) {
      for (let k = 0; k < size; (k = (k + 1) | 0)) {
        const off = Math.imul(k, d);
        let c = 0.0;
        for (let j = 0; j < d; (j = (j + 1) | 0)) c += basis[(off + j) | 0]! * w[j]!;
        for (let j = 0; j < d; (j = (j + 1) | 0)) w[j] = w[j]! - c * basis[(off + j) | 0]!;
      }
    }
    let norm = 0.0;
    for (let j = 0; j < d; (j = (j + 1) | 0)) norm += w[j]! * w[j]!;
    if (!(norm > 1e-20 * norm0)) continue;
    const inv = 1.0 / Math.sqrt(norm);
    const off = Math.imul(size, d);
    for (let j = 0; j < d; (j = (j + 1) | 0)) basis[(off + j) | 0] = w[j]! * inv;
    size = (size + 1) | 0;
  }
  return { basis, size, dim: d };
}
