// src/math/sin-cos-tan.ts — sin, cos and tan with Payne-Hanek reduction for large arguments (FDLIBM s_sin.c, s_cos.c, s_tan.c)
// Ported from an earlier TypeScript port of FDLIBM 5.3 (fdlibm-trig.ts).
import { getSignedHighWord, setWords } from './float-words.ts';
import { fdlibmRemPio2 } from './rem-pio2.ts';
import { C1, C2, C3, C4, C5, C6, S1, S2, S3, S4, S5, S6, kernelCos, kernelSin, kernelTan } from './trig-kernels.ts';

const y = new Float64Array(2);

export function fdlibmSin(x: number): number {
  const ix = getSignedHighWord(x) & 0x7fffffff;
  if (ix <= 0x3fe921fb && ix < 0x3e400000 && Math.trunc(x) === 0) return x;
  if (ix <= 0x3fe921fb) {
    const z = x * x;
    const v = z * x;
    const r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    return x + v * (S1 + z * r);
  }
  if (ix >= 0x7ff00000) return x - x;
  const n = fdlibmRemPio2(x, y);
  switch (n & 3) {
    case 0: return kernelSin(y[0]!, y[1]!, 1);
    case 1: return kernelCos(y[0]!, y[1]!);
    case 2: return -kernelSin(y[0]!, y[1]!, 1);
    default: return -kernelCos(y[0]!, y[1]!);
  }
}

export function fdlibmCos(x: number): number {
  const ix = getSignedHighWord(x) & 0x7fffffff;
  if (ix <= 0x3fe921fb && ix < 0x3e400000 && Math.trunc(x) === 0) return 1.0;
  if (ix <= 0x3fe921fb && ix < 0x3FD33333) {
    const z = x * x;
    const r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    return 1.0 - (0.5 * z - (z * r - x * 0.0));
  }
  if (ix <= 0x3fe921fb) {
    const z = x * x;
    const r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    const zr = z * r - x * 0.0;
    const qx = ix > 0x3fe90000 ? 0.28125 : setWords(ix - 0x00200000, 0);
    return (1.0 - qx) - ((0.5 * z - qx) - zr);
  }
  if (ix >= 0x7ff00000) return x - x;
  const n = fdlibmRemPio2(x, y);
  switch (n & 3) {
    case 0: return kernelCos(y[0]!, y[1]!);
    case 1: return -kernelSin(y[0]!, y[1]!, 1);
    case 2: return -kernelCos(y[0]!, y[1]!);
    default: return kernelSin(y[0]!, y[1]!, 1);
  }
}

export function fdlibmTan(x: number): number {
  const ix = getSignedHighWord(x) & 0x7fffffff;
  if (ix <= 0x3fe921fb) return kernelTan(x, 0.0, 1);
  if (ix >= 0x7ff00000) return x - x;
  const n = fdlibmRemPio2(x, y);
  return kernelTan(y[0]!, y[1]!, 1 - ((n & 1) << 1));
}
