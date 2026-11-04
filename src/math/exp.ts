// src/math/exp.ts — e^x with argument reduction by ln2 and a Remez polynomial (FDLIBM e_exp.c)
// Ported from an earlier TypeScript port of FDLIBM 5.3 (fdlibm-exp-log.ts).
import { getLowWord, getSignedHighWord, setWords } from './float-words.ts';

const halF = [0.5, -0.5];
const o_threshold = 7.09782712893383973096e+02;
const u_threshold = -7.45133219101941108420e+02;
const ln2HI = [6.93147180369123816490e-01, -6.93147180369123816490e-01];
const ln2LO = [1.90821492927058770002e-10, -1.90821492927058770002e-10];
const invln2 = 1.44269504088896338700e+00;
const P1 = 1.66666666666666019037e-01;
const P2 = -2.77777777770155933842e-03;
const P3 = 6.61375632143793436117e-05;
const P4 = -1.65339022054652515390e-06;
const P5 = 4.13813679705723846039e-08;
const E = 2.718281828459045;
const huge = 1.0e+300;
const twom1000 = 9.33263618503218878990e-302;
const two1023 = 8.988465674311579539e307;

export function fdlibmExp(xIn: number): number {
  let x = xIn;
  let hi = 0.0;
  let lo = 0.0;
  let k = 0;
  let hx = getSignedHighWord(x);
  const xsb = (hx >>> 31) & 1;
  hx &= 0x7fffffff;

  if (hx >= 0x7ff00000) {
    const isNan = ((hx & 0xfffff) | getLowWord(x)) !== 0;
    return isNan ? x + x : (xsb === 0 ? x : 0.0);
  }
  if (hx >= 0x40862E42 && x > o_threshold) return huge * huge;
  if (hx >= 0x40862E42 && x < u_threshold) return twom1000 * twom1000;

  if (hx > 0x3fd62e42 && hx < 0x3FF0A2B2 && x === 1.0) return E;
  if (hx > 0x3fd62e42 && hx < 0x3FF0A2B2) {
    hi = x - ln2HI[xsb]!;
    lo = ln2LO[xsb]!;
    k = 1 - xsb - xsb;
    x = hi - lo;
  } else if (hx > 0x3fd62e42) {
    k = Math.trunc(invln2 * x + halF[xsb]!);
    const t = k;
    hi = x - t * ln2HI[0]!;
    lo = t * ln2LO[0]!;
    x = hi - lo;
  } else if (hx < 0x3e300000 && huge + x > 1.0) {
    return 1.0 + x;
  } else {
    k = 0;
  }

  const t = x * x;
  const twopk = k >= -1021 ? setWords(0x3ff00000 + (k << 20), 0) : setWords(0x3ff00000 + ((k + 1000) << 20), 0);
  const c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
  if (k === 0) return 1.0 - ((x * c) / (c - 2.0) - x);
  const y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi);
  if (k >= -1021 && k === 1024) return y * 2.0 * two1023;
  if (k >= -1021) return y * twopk;
  return y * twopk * twom1000;
}
