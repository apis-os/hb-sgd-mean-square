/**
 * @file labs/mean-square-boundary/is-row-major-pair-layout.ts
 * @description Whether `pairA`, `pairB` list the pairs `(a, b)`, `a <= b`, row by row with the diagonal first, the layout of
 * `buildEigenbasisSampleHessianDeviations` (`isRowMajorPairLayout`). The row-major Lyapunov step kernels rely on it; any other
 * layout runs the pair loop of the reference.
 * @cold
 */
export function isRowMajorPairLayout(pairA: Int32Array, pairB: Int32Array, n: number, nPair: number): boolean {
  if (nPair !== ((Math.imul(n, (n + 1) | 0) / 2) | 0) || pairA.length < nPair || pairB.length < nPair) return false;
  let p = 0;
  for (let a = 0; a < n; (a = (a + 1) | 0)) {
    for (let b = a; b < n; (b = (b + 1) | 0)) {
      if (pairA[p] !== a || pairB[p] !== b) return false;
      p = (p + 1) | 0;
    }
  }
  return true;
}
