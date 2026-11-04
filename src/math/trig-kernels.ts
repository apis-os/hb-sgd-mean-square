// src/math/trig-kernels.ts — sin, cos and tan kernels on [-pi/4, pi/4] (FDLIBM k_sin.c, k_cos.c, k_tan.c)
// Ported from an earlier TypeScript port of FDLIBM 5.3 (fdlibm-trig.ts).
import { fdlibmAbs } from './abs-sign-min-max.ts';
import { getLowWord, getSignedHighWord, setLowWord, setWords } from './float-words.ts';

export const S1 = -1.66666666666666324348e-01;
export const S2 = 8.33333333332248946124e-03;
export const S3 = -1.98412698298579493134e-04;
export const S4 = 2.75573137070700676789e-06;
export const S5 = -2.50507602534068634195e-08;
export const S6 = 1.58969099521155010221e-10;

export const C1 = 4.16666666666666019037e-02;
export const C2 = -1.38888888888741095749e-03;
export const C3 = 2.48015872894767294178e-05;
export const C4 = -2.75573143513906633035e-07;
export const C5 = 2.08757232129817482790e-09;
export const C6 = -1.13596475577881948265e-11;

const T0 = 3.33333333333334091986e-01;
const T1 = 1.33333333333201242699e-01;
const T2 = 5.39682539762260521377e-02;
const T3 = 2.18694882948595424599e-02;
const T4 = 8.86323982359930005737e-03;
const T5 = 3.59207910759131235356e-03;
const T6 = 1.45620945432529025516e-03;
const T7 = 5.88041240820264096874e-04;
const T8 = 2.46463134818469906812e-04;
const T9 = 7.81794442939557092300e-05;
const T10 = 7.14072491382608190305e-05;
const T11 = -1.85586374855275456654e-05;
const T12 = 2.59073051863633712884e-05;
const pio4 = 7.85398163397448278999e-01;
const pio4lo = 3.06161699786838301793e-17;

export function kernelSin(x: number, yTail: number, iy: number): number {
  const ix = getSignedHighWord(x) & 0x7fffffff;
  if (ix < 0x3e400000 && Math.trunc(x) === 0) return x;
  const z = x * x;
  const v = z * x;
  const r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
  if (iy === 0) return x + v * (S1 + z * r);
  return x - ((z * (0.5 * yTail - v * r) - yTail) - v * S1);
}

export function kernelCos(x: number, yTail: number): number {
  const ix = getSignedHighWord(x) & 0x7fffffff;
  if (ix < 0x3e400000 && Math.trunc(x) === 0) return 1.0;
  const z = x * x;
  const r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
  if (ix < 0x3FD33333) return 1.0 - (0.5 * z - (z * r - x * yTail));
  const qx = ix > 0x3fe90000 ? 0.28125 : setWords(ix - 0x00200000, 0);
  const hz = 0.5 * z - qx;
  const a = 1.0 - qx;
  return a - (hz - (z * r - x * yTail));
}

export function kernelTan(xIn: number, yTailIn: number, iy: number): number {
  let x = xIn;
  let yTail = yTailIn;
  let z = 0.0;
  let w = 0.0;
  let v = 0.0;
  let s = 0.0;
  let r = 0.0;
  const hx = getSignedHighWord(x);
  const ix = hx & 0x7fffffff;

  if (ix < 0x3e300000 && Math.trunc(x) === 0 && ((ix | getLowWord(x)) | (iy + 1)) === 0) {
    return 1.0 / fdlibmAbs(x);
  }
  if (ix < 0x3e300000 && Math.trunc(x) === 0 && iy === 1) {
    return x;
  }
  if (ix < 0x3e300000 && Math.trunc(x) === 0) {
    z = setLowWord(x + yTail, 0);
    w = x + yTail;
    v = yTail - (z - x);
    const a0 = -1.0 / w;
    const t0 = setLowWord(a0, 0);
    s = 1.0 + t0 * z;
    return t0 + a0 * (s + t0 * v);
  }

  if (ix >= 0x3FE59428 && hx < 0) {
    x = -x;
    yTail = -yTail;
    z = pio4 - x;
    w = pio4lo - yTail;
    x = z + w;
    yTail = 0.0;
  } else if (ix >= 0x3FE59428) {
    z = pio4 - x;
    w = pio4lo - yTail;
    x = z + w;
    yTail = 0.0;
  }

  z = x * x;
  w = z * z;
  r = T1 + w * (T3 + w * (T5 + w * (T7 + w * (T9 + w * T11))));
  v = z * (T2 + w * (T4 + w * (T6 + w * (T8 + w * (T10 + w * T12)))));
  s = z * x;
  r = yTail + z * (s * (r + v) + yTail);
  r += T0 * s;
  w = x + r;

  if (ix >= 0x3FE59428) {
    v = iy;
    return (1 - ((hx >> 30) & 2)) * (v - 2.0 * (x - (w * w / (w + v) - r)));
  }
  if (iy === 1) return w;
  z = setLowWord(w, 0);
  v = r - (z - x);
  const a = -1.0 / w;
  const t = setLowWord(a, 0);
  s = 1.0 + t * z;
  return t + a * (s + t * v);
}
