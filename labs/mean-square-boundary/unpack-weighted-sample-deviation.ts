/**
 * @file labs/mean-square-boundary/unpack-weighted-sample-deviation.ts
 * @description One packed eigenbasis deviation as full `n x n` arrays (`unpackWeightedSampleDeviation`): `d[a][b] = D_ab` and
 * `dm[a][b] = D_ab m_b`, the arrays of `computeTraceWeightedLowerBoundReference`. With `rowMajor` (the layout of
 * `isRowMajorPairIndex`) the packed row is read once in order and mirrored; otherwise every entry goes through `pairIndex` as in
 * the reference. Both give the same values, since the reference also reads one packed value per unordered pair.
 * @cold
 */
export function unpackWeightedSampleDeviation(
  packed: Float64Array,
  base: number,
  n: number,
  pairIndex: Int32Array,
  weights: Float64Array,
  rowMajor: boolean,
  d: Float64Array,
  dm: Float64Array,
): void {
  if (!rowMajor) {
    for (let a = 0; a < n; (a = (a + 1) | 0)) {
      for (let b = 0; b < n; (b = (b + 1) | 0)) {
        const v = packed[(base + pairIndex[(Math.imul(a, n) + b) | 0]!) | 0]!;
        d[(Math.imul(a, n) + b) | 0] = v;
        dm[(Math.imul(a, n) + b) | 0] = v * weights[b]!;
      }
    }
    return;
  }
  let p = base;
  for (let a = 0; a < n; (a = (a + 1) | 0)) {
    const aRow = Math.imul(a, n);
    const wa = weights[a]!;
    const vd = packed[p]!;
    p = (p + 1) | 0;
    d[(aRow + a) | 0] = vd;
    dm[(aRow + a) | 0] = vd * wa;
    for (let b = (a + 1) | 0; b < n; (b = (b + 1) | 0)) {
      const v = packed[p]!;
      p = (p + 1) | 0;
      const bRow = Math.imul(b, n);
      d[(aRow + b) | 0] = v;
      d[(bRow + a) | 0] = v;
      dm[(aRow + b) | 0] = v * weights[b]!;
      dm[(bRow + a) | 0] = v * wa;
    }
  }
}
