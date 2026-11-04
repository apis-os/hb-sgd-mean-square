/**
 * @file labs/mean-square-boundary/detect-float64-buffer-overlap.ts
 * @description Whether the first `lenA` entries of `a` and the first `lenB` entries of `b` may share memory
 * (`detectFloat64BufferOverlap`): same underlying buffer and intersecting byte ranges, read through `readFloat64View`. The Phase 6
 * certificate, the verified Cholesky and both noise operators refuse overlapping scratch; the fourth audit round passed `work === w` and
 * `work === x` to `certifyConeCollatzWielandtAtRate`, whose Cholesky of `x` then overwrote the noise or the witness and certified a
 * rate above `eta*`. Empty ranges never overlap. Anything `readFloat64View` refuses (shared memory of any realm, a `Proxy`, another
 * typed-array kind, a short or detached view) counts as overlapping, and `buffer` and `byteOffset` are read through the intrinsic
 * getters, so a subclass or an own property cannot misreport them (sixth, eighth and ninth audit rounds); a genuine `Float64Array` of
 * another realm is accepted.
 * @cold
 */
import { readFloat64View } from './read-float64-view.ts';

export function detectFloat64BufferOverlap(a: Float64Array, lenA: number, b: Float64Array, lenB: number): boolean {
  if (lenA <= 0 || lenB <= 0) return false;
  const ra = readFloat64View(a, lenA);
  const rb = readFloat64View(b, lenB);
  if (ra === null || rb === null) return true;
  if (ra.buffer !== rb.buffer) return false;
  return ra.byteOffset < rb.byteOffset + lenB * 8 && rb.byteOffset < ra.byteOffset + lenA * 8;
}
