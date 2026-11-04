// src/math/log.ts — natural logarithm (FDLIBM e_log.c)
// Ported from an earlier TypeScript port of FDLIBM 5.3 (fdlibm-exp-log.ts).
import { getLowWord, getSignedHighWord, setHighWord } from './float-words.ts';
import { Lg1, Lg2, Lg3, Lg4, Lg5, Lg6, Lg7, ln2_hi, ln2_lo, two54 } from './log-coefficients.ts';

export function fdlibmLog(xIn: number): number {
  let x = xIn;
  let hx = getSignedHighWord(x);
  const lx = getLowWord(x);
  let k = 0;
  if (hx < 0x00100000 && ((hx & 0x7fffffff) | lx) === 0) return -Infinity;
  if (hx < 0x00100000 && hx < 0) return Number.NaN;
  if (hx < 0x00100000) {
    k -= 54;
    x *= two54;
    hx = getSignedHighWord(x);
  }
  if (hx >= 0x7ff00000) return x + x;
  k += (hx >> 20) - 1023;
  hx &= 0x000fffff;
  const i = (hx + 0x95f64) & 0x100000;
  x = setHighWord(x, hx | (i ^ 0x3ff00000));
  k += i >> 20;
  const f = x - 1.0;
  if ((0x000fffff & (2 + hx)) < 3 && f === 0) {
    const dk = k;
    return k === 0 ? 0 : dk * ln2_hi + dk * ln2_lo;
  }
  if ((0x000fffff & (2 + hx)) < 3) {
    const R = f * f * (0.5 - 0.33333333333333333 * f);
    const dk = k;
    return k === 0 ? f - R : dk * ln2_hi - ((R - dk * ln2_lo) - f);
  }
  const s = f / (2.0 + f);
  const dk = k;
  const z = s * s;
  const i2 = hx - 0x6147a;
  const w = z * z;
  const j = 0x6b851 - hx;
  const t1 = w * (Lg2 + w * (Lg4 + w * Lg6));
  const t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)));
  const R = t2 + t1;
  if ((i2 | j) > 0) {
    const hfsq = 0.5 * f * f;
    return k === 0 ? f - (hfsq - s * (hfsq + R)) : dk * ln2_hi - ((hfsq - (s * (hfsq + R) + dk * ln2_lo)) - f);
  }
  return k === 0 ? f - s * (f - R) : dk * ln2_hi - ((s * (f - R) - dk * ln2_lo) - f);
}
