// src/math/expm1.ts — e^x - 1 accurate near zero (FDLIBM s_expm1.c)
// Ported from an earlier TypeScript port of FDLIBM 5.3 (fdlibm-hyperbolic.ts).
import { getLowWord, getSignedHighWord, setWords } from './float-words.ts';

const one = 1.0;
const em_huge = 1.0e+300;
const em_tiny = 1.0e-300;
const o_threshold = 7.09782712893383973096e+02;
const ln2_hi = 6.93147180369123816490e-01;
const ln2_lo = 1.90821492927058770002e-10;
const invln2 = 1.44269504088896338700e+00;
const two1023 = 8.988465674311579539e307;
const Q1 = -3.33333333333331316428e-02;
const Q2 = 1.58730158725481460165e-03;
const Q3 = -7.93650757867487942473e-05;
const Q4 = 4.00821782732936239552e-06;
const Q5 = -2.01099218183624371326e-07;

function expm1ScaleBack(r: number, e: number, k: number): number {
  const twopk = setWords((0x3ff00000 + (k << 20)) >>> 0, 0);
  if (k === -1) return 0.5 * (r - e) - 0.5;
  if (k === 1 && r < -0.25) return -2.0 * (e - (r + 0.5));
  if (k === 1) return one + 2.0 * (r - e);
  if ((k <= -2 || k > 56) && k === 1024) return (one - (e - r)) * 2.0 * two1023 - one;
  if (k <= -2 || k > 56) return (one - (e - r)) * twopk - one;
  if (k < 20) {
    const t = setWords((0x3ff00000 - (0x200000 >> k)) >>> 0, 0);
    return (t - (e - r)) * twopk;
  }
  const tHi = setWords(((0x3ff - k) << 20) >>> 0, 0);
  return ((r - (e + tHi)) + one) * twopk;
}

export function fdlibmExpm1(xIn: number): number {
  let x = xIn;
  let c = 0.0;
  let hi = 0.0;
  let lo = 0.0;
  let k = 0;

  let hx = getSignedHighWord(x);
  const xsb = hx & 0x80000000;
  hx &= 0x7fffffff;

  if (hx >= 0x7ff00000) {
    const low = getLowWord(x);
    const isNan = ((hx & 0xfffff) | low) !== 0;
    return isNan ? x + x : (xsb === 0 ? x : -1.0);
  }
  if (hx >= 0x40862E42 && x > o_threshold) return em_huge * em_huge;
  if (hx >= 0x4043687A && xsb !== 0 && x + em_tiny < 0.0) return em_tiny - one;

  if (hx > 0x3fd62e42 && hx < 0x3FF0A2B2) {
    hi = xsb === 0 ? x - ln2_hi : x + ln2_hi;
    lo = xsb === 0 ? ln2_lo : -ln2_lo;
    k = xsb === 0 ? 1 : -1;
    x = hi - lo;
    c = (hi - x) - lo;
  } else if (hx > 0x3fd62e42) {
    const t = xsb === 0 ? invln2 * x + 0.5 : invln2 * x - 0.5;
    k = Math.trunc(t);
    const tk = k;
    hi = x - tk * ln2_hi;
    lo = tk * ln2_lo;
    x = hi - lo;
    c = (hi - x) - lo;
  } else if (hx < 0x3c900000) {
    return x;
  } else {
    k = 0;
  }

  const hfx = 0.5 * x;
  const hxs = x * hfx;
  const r1 = one + hxs * (Q1 + hxs * (Q2 + hxs * (Q3 + hxs * (Q4 + hxs * Q5))));
  const t3 = 3.0 - r1 * hfx;
  let e = hxs * ((r1 - t3) / (6.0 - x * t3));
  if (k === 0) return x - (x * e - hxs);

  e = x * (e - c) - c;
  e -= hxs;
  return expm1ScaleBack(x, e, k);
}
