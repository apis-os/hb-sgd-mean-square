// src/math/sinh-cosh-tanh.ts — hyperbolic sine, cosine and tangent (FDLIBM e_sinh.c, e_cosh.c, s_tanh.c)
// Ported from an earlier TypeScript port of FDLIBM 5.3 (fdlibm-hyperbolic.ts), including the FreeBSD msun tanh revision.
import { fdlibmAbs } from './abs-sign-min-max.ts';
import { fdlibmExp } from './exp.ts';
import { fdlibmExpm1 } from './expm1.ts';
import { getLowWord, getSignedHighWord } from './float-words.ts';

const one_sh = 1.0;
const shuge = 1.0e+307;

export function fdlibmSinh(xIn: number): number {
  const x = xIn;
  const jx = getSignedHighWord(x);
  const ix = jx & 0x7fffffff;

  if (ix >= 0x7ff00000) return x + x;
  if (ix < 0x3e300000 && shuge + x > one_sh) return x;

  const h = jx < 0 ? -0.5 : 0.5;
  if (ix < 0x40360000) {
    const t = fdlibmExpm1(fdlibmAbs(x));
    return ix < 0x3ff00000 ? h * (2.0 * t - (t * t) / (t + one_sh)) : h * (t + t / (t + one_sh));
  }

  if (ix < 0x40862E42) return h * fdlibmExp(fdlibmAbs(x));

  const lx = getLowWord(x);
  if (ix < 0x408633CE || (ix === 0x408633ce && lx <= 0x8fb9f87d)) {
    const w = fdlibmExp(0.5 * fdlibmAbs(x));
    return (h * w) * w;
  }

  return x * shuge;
}

const one_ch = 1.0;
const half_ch = 0.5;
const huge_ch = 1.0e+300;

export function fdlibmCosh(xIn: number): number {
  const x = xIn;
  const ix = getSignedHighWord(x) & 0x7fffffff;

  if (ix >= 0x7ff00000) return x * x;

  if (ix < 0x3fd62e43) {
    const t = fdlibmExpm1(fdlibmAbs(x));
    const w = one_ch + t;
    return ix < 0x3c800000 ? w : one_ch + (t * t) / (w + w);
  }

  if (ix < 0x40360000) {
    const t = fdlibmExp(fdlibmAbs(x));
    return half_ch * t + half_ch / t;
  }

  if (ix < 0x40862E42) return half_ch * fdlibmExp(fdlibmAbs(x));

  const lx = getLowWord(x);
  if (ix < 0x408633CE || (ix === 0x408633ce && lx <= 0x8fb9f87d)) {
    const w = fdlibmExp(half_ch * fdlibmAbs(x));
    return (half_ch * w) * w;
  }

  return huge_ch * huge_ch;
}

const one_th = 1.0;
const two_th = 2.0;
const tiny_th = 1.0e-300;
const huge_th = 1.0e+300;

export function fdlibmTanh(xIn: number): number {
  const x = xIn;
  const jx = getSignedHighWord(x);
  const ix = jx & 0x7fffffff;

  if (ix >= 0x7ff00000) return jx >= 0 ? one_th / x + one_th : one_th / x - one_th;
  // FreeBSD s_tanh.c: tanh(x) = x below 2^-28 (FDLIBM 5.3 used 2^-55 and went through expm1 in between)
  if (ix < 0x3e300000 && huge_th + x > one_th) return x;

  let z: number = 0;
  if (ix < 0x40360000 && ix >= 0x3ff00000) {
    const t = fdlibmExpm1(two_th * fdlibmAbs(x));
    z = one_th - two_th / (t + two_th);
  } else if (ix < 0x40360000) {
    const t = fdlibmExpm1(-two_th * fdlibmAbs(x));
    z = -t / (t + two_th);
  } else {
    z = one_th - tiny_th;
  }
  return jx >= 0 ? z : -z;
}
