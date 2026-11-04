/**
 * @file labs/mean-square-boundary/assemble-curvature-noise-operator-matrix.ts
 * @description The curvature-noise operator `V(Y) = (1/B) E[D_i Y D_i]` of a minibatch drawn with replacement as a dense symmetric
 * matrix on `S^n` (`assembleCurvatureNoiseOperatorMatrix`), in the orthonormal basis `E_kk`, `(E_ab + E_ba) / sqrt(2)` of the
 * eigenbasis, ordered by the pairs of `buildEigenbasisSampleHessianDeviations`: `V_{(ab),(cd)} = w_ab w_cd (G[ac][bd] + G[ad][bc])`,
 * `w = 1` off the diagonal and `1/sqrt(2)` on it, from the Gram matrix `G` of `assembleCurvatureNoiseGram`. A matrix `Y` has
 * coordinates `Y_kk` and `sqrt(2) Y_ab`.
 * @cold
 */
export function assembleCurvatureNoiseOperatorMatrix(
  g: Float64Array,
  n: number,
  nPair: number,
  pairIndex: Int32Array,
  pairA: Int32Array,
  pairB: Int32Array,
): Float64Array {
  const v = new Float64Array(Math.imul(nPair, nPair));
  for (let p = 0; p < nPair; (p = (p + 1) | 0)) {
    const ia = pairA[p]!;
    const ib = pairB[p]!;
    const wp = ia === ib ? Math.SQRT1_2 : 1.0;
    for (let q = p; q < nPair; (q = (q + 1) | 0)) {
      const ic = pairA[q]!;
      const id = pairB[q]!;
      const wq = ic === id ? Math.SQRT1_2 : 1.0;
      const gac = pairIndex[(Math.imul(ia, n) + ic) | 0]!;
      const gbd = pairIndex[(Math.imul(ib, n) + id) | 0]!;
      const gad = pairIndex[(Math.imul(ia, n) + id) | 0]!;
      const gbc = pairIndex[(Math.imul(ib, n) + ic) | 0]!;
      const e = wp * wq * (g[(Math.imul(gac, nPair) + gbd) | 0]! + g[(Math.imul(gad, nPair) + gbc) | 0]!);
      v[(Math.imul(p, nPair) + q) | 0] = e;
      v[(Math.imul(q, nPair) + p) | 0] = e;
    }
  }
  return v;
}
