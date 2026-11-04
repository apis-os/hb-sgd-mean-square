/**
 * @file labs/mean-square-boundary/assemble-thm-g6-operator-matrix.ts
 * @description The threshold operator `A = L_0^{-1/2} B_beta L_0^{-1/2}` of Theorem G.6(3) as a dense symmetric matrix on `S^n`
 * (`assembleThmG6OperatorMatrix`), in the orthonormal basis `E_kk`, `(E_ab + E_ba) / sqrt(2)` of the eigenbasis of the mean
 * curvature, ordered by the pairs of `buildEigenbasisSampleHessianDeviations`; `lambda_max(A) = 1 / eta*`. With `s_p = l_a + l_b`
 * (`l` the curvature of the iteration: `lambda + mu` for a tether, `|lambda|` for the reflected one; the double commutator
 * uses differences of `l`, in which a tether cancels), the diagonal part is
 * `(l_a l_b / (1 + beta) - beta (l_a - l_b)^2 / ((1 - beta)(1 - beta^2))) / s_p`, the same multiplier as
 * `applyThmG6RayleighOperator`, and the curvature noise is `V_pq / ((1 - beta) sqrt(s_p s_q))` with
 * `V_{(ab),(cd)} = w_ab w_cd (G[ac][bd] + G[ad][bc])`, `w = 1` off the diagonal and `1/sqrt(2)` on it, from the Gram matrix
 * `G` of `assembleCurvatureNoiseGram`. Optional `restrict` (a 0/1 mask over modes) keeps only the pairs inside the masked modes,
 * the Rayleigh restriction of Theorem G.6 to matrices supported on their span; the returned `pairs` lists the kept pairs.
 * @cold
 */
export function assembleThmG6OperatorMatrix(
  g: Float64Array,
  n: number,
  nPair: number,
  pairIndex: Int32Array,
  pairA: Int32Array,
  pairB: Int32Array,
  curvature: Float64Array,
  beta: number,
  restrict?: Uint8Array,
): { readonly a: Float64Array; readonly dim: number; readonly pairs: Int32Array } {
  const keep: number[] = [];
  for (let p = 0; p < nPair; (p = (p + 1) | 0)) {
    if (restrict === undefined || (restrict[pairA[p]!] === 1 && restrict[pairB[p]!] === 1)) keep.push(p);
  }
  const dim = keep.length | 0;
  const pairs = Int32Array.from(keep);
  const a = new Float64Array(Math.imul(dim, dim));
  const invOnePlusBeta = 1.0 / (1.0 + beta);
  const commutator = beta / ((1.0 - beta) * (1.0 - beta * beta));
  const invOneMinusBeta = 1.0 / (1.0 - beta);
  const rootS = new Float64Array(dim);
  const w = new Float64Array(dim);
  for (let r = 0; r < dim; (r = (r + 1) | 0)) {
    const p = pairs[r]!;
    rootS[r] = Math.sqrt(curvature[pairA[p]!]! + curvature[pairB[p]!]!);
    w[r] = pairA[p] === pairB[p] ? Math.SQRT1_2 : 1.0;
  }
  for (let r = 0; r < dim; (r = (r + 1) | 0)) {
    const p = pairs[r]!;
    const ia = pairA[p]!;
    const ib = pairB[p]!;
    for (let c = r; c < dim; (c = (c + 1) | 0)) {
      const q = pairs[c]!;
      const ic = pairA[q]!;
      const id = pairB[q]!;
      const gac = pairIndex[(Math.imul(ia, n) + ic) | 0]!;
      const gbd = pairIndex[(Math.imul(ib, n) + id) | 0]!;
      const gad = pairIndex[(Math.imul(ia, n) + id) | 0]!;
      const gbc = pairIndex[(Math.imul(ib, n) + ic) | 0]!;
      const v = w[r]! * w[c]! * (g[(Math.imul(gac, nPair) + gbd) | 0]! + g[(Math.imul(gad, nPair) + gbc) | 0]!);
      let entry = (v * invOneMinusBeta) / (rootS[r]! * rootS[c]!);
      if (r === c) {
        const la = curvature[ia]!;
        const lb = curvature[ib]!;
        const diff = la - lb;
        entry += (la * lb * invOnePlusBeta - commutator * diff * diff) / (la + lb);
      }
      a[(Math.imul(r, dim) + c) | 0] = entry;
      a[(Math.imul(c, dim) + r) | 0] = entry;
    }
  }
  return { a, dim, pairs };
}
