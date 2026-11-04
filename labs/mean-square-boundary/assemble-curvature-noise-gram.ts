/**
 * @file labs/mean-square-boundary/assemble-curvature-noise-gram.ts
 * @description Gram matrix of the packed eigenbasis deviations (`assembleCurvatureNoiseGram`): with `D_i` the deviations of
 * `buildEigenbasisSampleHessianDeviations` and pairs `p, q` of `S^n`, `G[p][q] = (1 / (B N)) sum_i (D_i)_p (D_i)_q`, the second
 * moment of the curvature noise of a minibatch of `B` samples drawn with replacement. The deviations are first transposed to
 * pair-major rows of `N` contiguous samples; the upper triangle is formed by dot products and mirrored. Its diagonal entry at
 * the pair `(k, l)` is `Gamma_kl = (1/B) E[(D_i)_kl^2]`. The curvature-noise operator `V` and the threshold operator of
 * Theorem G.6 are read off `G` by `assembleThmG6OperatorMatrix`. Bit-identical to `assembleCurvatureNoiseGramReference` (the test
 * oracle): every entry is the same dot product summed from `0.0` over `i` in order and scaled once; the entries are formed two
 * rows by four columns at a time (one load of each sample value serves several entries; the one entry below the diagonal that a
 * block may form is the same product sum as its mirror, by commutativity of each product).
 * @cold
 */
export function assembleCurvatureNoiseGram(packed: Float64Array, numSamples: number, nPair: number, batchB: number): Float64Array {
  const nS = numSamples | 0;
  const np = nPair | 0;
  const rows = new Float64Array(Math.imul(np, nS));
  for (let i = 0; i < nS; (i = (i + 1) | 0)) {
    const base = Math.imul(i, np);
    for (let p = 0; p < np; (p = (p + 1) | 0)) rows[(Math.imul(p, nS) + i) | 0] = packed[(base + p) | 0]!;
  }
  const scale = 1.0 / (batchB * nS);
  const g = new Float64Array(Math.imul(np, np));
  let p = 0;
  for (; ((p + 1) | 0) < np; p = (p + 2) | 0) {
    const ra = Math.imul(p, nS);
    const rb = (ra + nS) | 0;
    let q = p;
    for (; ((q + 3) | 0) < np; q = (q + 4) | 0) {
      const r0 = Math.imul(q, nS);
      const r1 = (r0 + nS) | 0;
      const r2 = (r1 + nS) | 0;
      const r3 = (r2 + nS) | 0;
      let a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0, b0 = 0.0, b1 = 0.0, b2 = 0.0, b3 = 0.0;
      for (let i = 0; i < nS; (i = (i + 1) | 0)) {
        const x = rows[(ra + i) | 0]!;
        const y = rows[(rb + i) | 0]!;
        const z0 = rows[(r0 + i) | 0]!, z1 = rows[(r1 + i) | 0]!, z2 = rows[(r2 + i) | 0]!, z3 = rows[(r3 + i) | 0]!;
        a0 += x * z0; a1 += x * z1; a2 += x * z2; a3 += x * z3;
        b0 += y * z0; b1 += y * z1; b2 += y * z2; b3 += y * z3;
      }
      storeGramPair(g, np, p, q, a0 * scale); storeGramPair(g, np, p, (q + 1) | 0, a1 * scale);
      storeGramPair(g, np, p, (q + 2) | 0, a2 * scale); storeGramPair(g, np, p, (q + 3) | 0, a3 * scale);
      storeGramPair(g, np, (p + 1) | 0, q, b0 * scale); storeGramPair(g, np, (p + 1) | 0, (q + 1) | 0, b1 * scale);
      storeGramPair(g, np, (p + 1) | 0, (q + 2) | 0, b2 * scale); storeGramPair(g, np, (p + 1) | 0, (q + 3) | 0, b3 * scale);
    }
    for (; q < np; (q = (q + 1) | 0)) {
      const rq = Math.imul(q, nS);
      let a = 0.0;
      let b = 0.0;
      for (let i = 0; i < nS; (i = (i + 1) | 0)) {
        const z = rows[(rq + i) | 0]!;
        a += rows[(ra + i) | 0]! * z;
        b += rows[(rb + i) | 0]! * z;
      }
      storeGramPair(g, np, p, q, a * scale);
      storeGramPair(g, np, (p + 1) | 0, q, b * scale);
    }
  }
  for (; p < np; (p = (p + 1) | 0)) {
    const rp = Math.imul(p, nS);
    for (let q = p; q < np; (q = (q + 1) | 0)) {
      const rq = Math.imul(q, nS);
      let s = 0.0;
      for (let i = 0; i < nS; (i = (i + 1) | 0)) s += rows[(rp + i) | 0]! * rows[(rq + i) | 0]!;
      storeGramPair(g, np, p, q, s * scale);
    }
  }
  return g;
}

function storeGramPair(g: Float64Array, np: number, p: number, q: number, v: number): void {
  g[(Math.imul(p, np) + q) | 0] = v;
  g[(Math.imul(q, np) + p) | 0] = v;
}
