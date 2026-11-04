/**
 * @file labs/mean-square-boundary/apply-hessian-rows.ts
 * @description Rows `r0 .. r1` of `dst[j] = (accumulate ? dst[j] : 0) + scale * H src[j]` for the Hessian of one sample
 * (`sampleIdx >= 0`) or the mean Hessian (`sampleIdx = -1`) of a `Phase4SampleHessianProvider` (`applyHessianRows`), with
 * `src` and `dst` row-major `n x n`. For symmetric `X`, applying it to the rows of `X` gives `(H X)^T = X H`; the operators of
 * Phase 4 are built from this and `transposeDenseRows`. Each row is one Hessian-vector product, so the result does not
 * depend on the split of rows between workers.
 * @cold
 */
import type { Phase4SampleHessianProvider } from './phase4-types.ts';

export function applyHessianRows(
  provider: Phase4SampleHessianProvider,
  sampleIdx: number,
  src: Float64Array,
  dst: Float64Array,
  n: number,
  r0: number,
  r1: number,
  scale: number,
  accumulate: boolean,
  tmpOut: Float64Array,
): void {
  const n4 = n & ~3;
  for (let j = r0; j < r1; (j = (j + 1) | 0)) {
    const row = Math.imul(j, n);
    const vec = src.subarray(row, (row + n) | 0);
    if (sampleIdx < 0) provider.applyMeanHessian(vec, tmpOut);
    else provider.applySampleHessian(sampleIdx, vec, tmpOut);
    if (accumulate) {
      let k = 0;
      for (; k < n4; (k = (k + 4) | 0)) {
        const idx = (row + k) | 0;
        dst[idx] = dst[idx]! + scale * tmpOut[k]!;
        dst[(idx + 1) | 0] = dst[(idx + 1) | 0]! + scale * tmpOut[(k + 1) | 0]!;
        dst[(idx + 2) | 0] = dst[(idx + 2) | 0]! + scale * tmpOut[(k + 2) | 0]!;
        dst[(idx + 3) | 0] = dst[(idx + 3) | 0]! + scale * tmpOut[(k + 3) | 0]!;
      }
      for (; k < n; (k = (k + 1) | 0)) dst[(row + k) | 0] = dst[(row + k) | 0]! + scale * tmpOut[k]!;
    } else {
      let k = 0;
      for (; k < n4; (k = (k + 4) | 0)) {
        const idx = (row + k) | 0;
        dst[idx] = scale * tmpOut[k]!;
        dst[(idx + 1) | 0] = scale * tmpOut[(k + 1) | 0]!;
        dst[(idx + 2) | 0] = scale * tmpOut[(k + 2) | 0]!;
        dst[(idx + 3) | 0] = scale * tmpOut[(k + 3) | 0]!;
      }
      for (; k < n; (k = (k + 1) | 0)) dst[(row + k) | 0] = scale * tmpOut[k]!;
    }
  }
}
