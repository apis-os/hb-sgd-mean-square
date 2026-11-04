/**
 * @file labs/mean-square-boundary/compute-power-of-two-scale.ts
 * @description The power of two `s = 2^-e` that brings a magnitude `m` into `[0.5, 1)` (`computePowerOfTwoScale`), with its exponent
 * `e`, both from the IEEE-754 bit pattern (no native math). Multiplying a binary64 number by `s` is exact unless the product leaves
 * the normal range. The Phase 6 noise operators scale their inputs by `s` before any chained product, so that every entry is at most
 * 1 in magnitude and an intermediate underflow (error at most `2^-1075`) can only be multiplied afterwards by factors bounded by the
 * matrix sizes; they undo the scaling with `multiplyByPowerOfTwo` on the net exponent (second and third external audit rounds,
 * 2026-10-29). Returns `null` unless `m` is finite and in `[2^-1000, 2^1000]`, so that `s` is a normal number and `e` lies in
 * `[-999, 1001]`.
 * @cold
 */
export function computePowerOfTwoScale(m: number): { readonly scale: number; readonly exponent: number } | null {
  if (!(m >= 9.332636185032189e-302 && m <= 1.0715086071862673e301)) return null;
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, m);
  // m is normal: m = f 2^(field - 1023), f in [1, 2); m 2^-e in [0.5, 1) for e = field - 1022
  const e = ((view.getUint32(0) >>> 20) & 0x7ff) - 1022;
  view.setUint32(0, ((1023 - e) << 20) >>> 0);
  view.setUint32(4, 0);
  return { scale: view.getFloat64(0), exponent: e };
}
