// src/math/scalbn.ts — x * 2^n by exponent arithmetic (FDLIBM s_scalbn.c), two variants kept bit-identical to the earlier port
// Ported from an earlier TypeScript port of FDLIBM 5.3: fdlibmScalbn from fdlibm-core.ts; scalbn (with copysign overflow) from the files of pow
// and of the Payne-Hanek reduction in that port. Here rem-pio2-kernel.ts uses it.
import { getLowWord, getSignedHighWord, setHighWord, setWords } from './float-words.ts';

const TWO54 = 1.8014398509481984e16;
const TWOM54 = 5.5511151231257827e-17;
const HUGE = 1.0e+300;
const TINY = 1.0e-300;

export function fdlibmScalbn(xIn: number, n: number): number {
  let x = xIn;
  let hx = getSignedHighWord(x);
  const lx = getLowWord(x);
  let k = (hx & 0x7ff00000) >>> 20;

  if (k === 0 && (lx | (hx & 0x7fffffff)) === 0) return x;
  if (k === 0 && n < -50000) return 1e-300 * (x * TWO54);
  if (k === 0) {
    x *= TWO54;
    hx = getSignedHighWord(x);
    k = ((hx & 0x7ff00000) >>> 20) - 54;
  }
  if (k === 0x7ff) return x + x;
  k = k + (n | 0);
  if (k > 0x7fe) return x > 0 ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY;
  if (k > 0) return setWords((hx & 0x800fffff) | (k << 20), lx);
  if (k <= -54) return (n > 50000) ? (x > 0 ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY) : (x > 0 ? 0.0 : -0.0);
  k += 54;
  const scaled = setWords((hx & 0x800fffff) | (k << 20), lx);
  return scaled * TWOM54;
}

/** Magnitude of x with the sign of y. */
export function copysign(x: number, y: number): number {
  return setHighWord(x, (getSignedHighWord(x) & 0x7fffffff) | (getSignedHighWord(y) & 0x80000000));
}

export function scalbn(xIn: number, n: number): number {
  let x = xIn;
  let hx = getSignedHighWord(x);
  const lx = getLowWord(x);
  let k = (hx & 0x7ff00000) >> 20;
  if (k === 0 && (lx | (hx & 0x7fffffff)) === 0) return x;
  if (k === 0) {
    x *= TWO54;
    hx = getSignedHighWord(x);
    k = ((hx & 0x7ff00000) >> 20) - 54;
  }
  if (k === 0x7ff) return x + x;
  if (n < -50000) return TINY * x;
  if (n > 50000 || k + n > 0x7fe) return HUGE * copysign(HUGE, x);
  k = k + n;
  if (k > 0) return setHighWord(x, (hx & 0x800fffff) | (k << 20));
  if (k <= -54) return TINY * copysign(TINY, x);
  k += 54;
  return setHighWord(x, (hx & 0x800fffff) | (k << 20)) * TWOM54;
}
