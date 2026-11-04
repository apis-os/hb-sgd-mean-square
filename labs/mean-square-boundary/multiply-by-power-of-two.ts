/**
 * @file labs/mean-square-boundary/multiply-by-power-of-two.ts
 * @description `v 2^k` for an integer `k` (`multiplyByPowerOfTwo`), in steps `2^j` with `|j| <= 1000` that all have the sign of
 * `k` (each step a normal power of two built from the exponent field). With `k >= 0` every partial product is at most the final one
 * in magnitude, so nothing overflows unless the result does; with `k < 0` a step can round only once the value has entered the
 * subnormal range, after which every further step is at most 1, so the total error is at most `steps 2^-1075`, with
 * `steps = ceil(|k| / 1000)`. The factors are never reordered: an earlier version sorted them by size and overflowed on products whose
 * exact value was finite (third external audit round, 2026-10-29).
 * @cold
 */
export function multiplyByPowerOfTwo(v: number, k: number): number {
  const view = new DataView(new ArrayBuffer(8));
  let r = v;
  let left = k;
  while (left !== 0) {
    const j = left > 1000 ? 1000 : left < -1000 ? -1000 : left;
    view.setUint32(0, ((j + 1023) << 20) >>> 0);
    view.setUint32(4, 0);
    r = r * view.getFloat64(0);
    left -= j;
  }
  return r;
}
