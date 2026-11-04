/**
 * @file labs/mean-square-boundary/accumulate-weighted-deviation-products.ts
 * @description `s += (D M) D` for one sample (`accumulateWeightedDeviationProducts`), with `dm[k n + j] = D_kj m_j` and `d` the
 * symmetric deviation `D` as full `n x n` arrays: entry `(k, l)` is `sum_j dm[k][j] d[j][l]` summed from `0.0` in `j` order and
 * then added to `s[k][l]`, exactly as the per-entry loop of `computeTraceWeightedLowerBoundReference`. Entries are formed in
 * `3 x 3` register blocks (each `dm` and `d` load serves three entries), which changes no operation of any entry.
 * @cold
 */
export function accumulateWeightedDeviationProducts(dm: Float64Array, d: Float64Array, s: Float64Array, n: number): void {
  let k = 0;
  for (; ((k + 2) | 0) < n; k = (k + 3) | 0) {
    const k0 = Math.imul(k, n);
    const k1 = (k0 + n) | 0;
    const k2 = (k1 + n) | 0;
    let l = 0;
    for (; ((l + 2) | 0) < n; l = (l + 3) | 0) {
      let a00 = 0.0, a01 = 0.0, a02 = 0.0, a10 = 0.0, a11 = 0.0, a12 = 0.0, a20 = 0.0, a21 = 0.0, a22 = 0.0;
      for (let j = 0; j < n; (j = (j + 1) | 0)) {
        const jl = (Math.imul(j, n) + l) | 0;
        const m0 = dm[(k0 + j) | 0]!, m1 = dm[(k1 + j) | 0]!, m2 = dm[(k2 + j) | 0]!;
        const d0 = d[jl]!, d1 = d[(jl + 1) | 0]!, d2 = d[(jl + 2) | 0]!;
        a00 += m0 * d0; a01 += m0 * d1; a02 += m0 * d2;
        a10 += m1 * d0; a11 += m1 * d1; a12 += m1 * d2;
        a20 += m2 * d0; a21 += m2 * d1; a22 += m2 * d2;
      }
      s[(k0 + l) | 0] = s[(k0 + l) | 0]! + a00; s[(k0 + l + 1) | 0] = s[(k0 + l + 1) | 0]! + a01; s[(k0 + l + 2) | 0] = s[(k0 + l + 2) | 0]! + a02;
      s[(k1 + l) | 0] = s[(k1 + l) | 0]! + a10; s[(k1 + l + 1) | 0] = s[(k1 + l + 1) | 0]! + a11; s[(k1 + l + 2) | 0] = s[(k1 + l + 2) | 0]! + a12;
      s[(k2 + l) | 0] = s[(k2 + l) | 0]! + a20; s[(k2 + l + 1) | 0] = s[(k2 + l + 1) | 0]! + a21; s[(k2 + l + 2) | 0] = s[(k2 + l + 2) | 0]! + a22;
    }
    for (; l < n; (l = (l + 1) | 0)) {
      let a0 = 0.0, a1 = 0.0, a2 = 0.0;
      for (let j = 0; j < n; (j = (j + 1) | 0)) {
        const dj = d[(Math.imul(j, n) + l) | 0]!;
        a0 += dm[(k0 + j) | 0]! * dj; a1 += dm[(k1 + j) | 0]! * dj; a2 += dm[(k2 + j) | 0]! * dj;
      }
      s[(k0 + l) | 0] = s[(k0 + l) | 0]! + a0; s[(k1 + l) | 0] = s[(k1 + l) | 0]! + a1; s[(k2 + l) | 0] = s[(k2 + l) | 0]! + a2;
    }
  }
  for (; k < n; (k = (k + 1) | 0)) {
    const kRow = Math.imul(k, n);
    for (let l = 0; l < n; (l = (l + 1) | 0)) {
      let acc = 0.0;
      for (let j = 0; j < n; (j = (j + 1) | 0)) acc += dm[(kRow + j) | 0]! * d[(Math.imul(j, n) + l) | 0]!;
      s[(kRow + l) | 0] = s[(kRow + l) | 0]! + acc;
    }
  }
}
