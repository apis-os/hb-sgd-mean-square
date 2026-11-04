/**
 * @file labs/mean-square-boundary/is-row-major-pair-index.ts
 * @description Whether `pairIndex` is the layout of `buildEigenbasisSampleHessianDeviations` (`isRowMajorPairIndex`): pairs
 * `(a, b)`, `a <= b`, numbered row by row from `0` with the diagonal first, and `pairIndex[a n + b] = pairIndex[b n + a]`. The
 * fast kernels that read the packed deviations sequentially use it to decide whether they may skip `pairIndex`.
 * @cold
 */
export function isRowMajorPairIndex(pairIndex: Int32Array, n: number, nPair: number): boolean {
  if (nPair !== ((Math.imul(n, (n + 1) | 0) / 2) | 0) || pairIndex.length < Math.imul(n, n)) return false;
  let p = 0;
  for (let a = 0; a < n; (a = (a + 1) | 0)) {
    for (let b = a; b < n; (b = (b + 1) | 0)) {
      if (pairIndex[(Math.imul(a, n) + b) | 0] !== p || pairIndex[(Math.imul(b, n) + a) | 0] !== p) return false;
      p = (p + 1) | 0;
    }
  }
  return true;
}
