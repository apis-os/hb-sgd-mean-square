// src/math/rem-pio2.ts — x mod pi/2 as a double-double y[0] + y[1], returns the quadrant (FDLIBM e_rem_pio2.c)
// Ported from an earlier TypeScript port of FDLIBM 5.3 (fdlibm-trig-large.ts).
import { getLowWord, getSignedHighWord, setHighWord, setLowWord } from './float-words.ts';
import { kernelRemPio2 } from './rem-pio2-kernel.ts';
import { npio2_hw } from './rem-pio2-tables.ts';

const two24 = 1.67772160000000000000e+07;
const invpio2 = 6.36619772367581382433e-01;
const pio2_1 = 1.57079632673412561417e+00;
const pio2_1t = 6.07710050650619224932e-11;
const pio2_2 = 6.07710050630396597660e-11;
const pio2_2t = 2.02226624879595063154e-21;
const pio2_3 = 2.02226624871116645580e-21;
const pio2_3t = 8.47842766036889956997e-32;
const txBuf = new Float64Array(3);

function reduceMediumPio2(ix: number, fn: number, rIn: number, wIn: number, y: Float64Array): void {
  let r = rIn;
  let w = wIn;
  let y0 = r - w;
  const j = ix >> 20;
  let high = getSignedHighWord(y0);
  let i = j - ((high >> 20) & 0x7ff);

  if (i > 16) {
    const t = r;
    w = fn * pio2_2;
    r = t - w;
    w = fn * pio2_2t - ((t - r) - w);
    y0 = r - w;
    high = getSignedHighWord(y0);
    i = j - ((high >> 20) & 0x7ff);
  }
  if (i > 49) {
    const t = r;
    w = fn * pio2_3;
    r = t - w;
    w = fn * pio2_3t - ((t - r) - w);
    y0 = r - w;
  }
  y[0] = y0;
  y[1] = (r - y0) - w;
}

function reduceMediumScale(x: number, hx: number, ix: number, y: Float64Array): number {
  const t = setHighWord(x, ix);
  const n = Math.trunc(t * invpio2 + 0.5);
  const fn = n;
  const r = t - fn * pio2_1;
  const w = fn * pio2_1t;
  if (n < 32 && ix !== npio2_hw[n - 1]) {
    y[0] = r - w;
    y[1] = (r - y[0]) - w;
  } else {
    reduceMediumPio2(ix, fn, r, w, y);
  }
  y[0] = hx < 0 ? -y[0]! : y[0]!;
  y[1] = hx < 0 ? -y[1]! : y[1]!;
  return hx < 0 ? -n : n;
}

export function fdlibmRemPio2(x: number, y: Float64Array): number {
  const hx = getSignedHighWord(x);
  const ix = hx & 0x7fffffff;

  if (ix <= 0x3fe921fb) {
    y[0] = x;
    y[1] = 0;
    return 0;
  }

  if (ix < 0x4002d97c && hx > 0 && ix !== 0x3ff921fb) {
    const z = x - pio2_1;
    y[0] = z - pio2_1t;
    y[1] = (z - y[0]) - pio2_1t;
    return 1;
  }
  if (ix < 0x4002d97c && hx > 0) {
    const z = (x - pio2_1) - pio2_2;
    y[0] = z - pio2_2t;
    y[1] = (z - y[0]) - pio2_2t;
    return 1;
  }
  if (ix < 0x4002d97c && ix !== 0x3ff921fb) {
    const z = x + pio2_1;
    y[0] = z + pio2_1t;
    y[1] = (z - y[0]) + pio2_1t;
    return -1;
  }
  if (ix < 0x4002d97c) {
    const z = (x + pio2_1) + pio2_2;
    y[0] = z + pio2_2t;
    y[1] = (z - y[0]) + pio2_2t;
    return -1;
  }

  if (ix <= 0x413921fb) {
    return reduceMediumScale(x, hx, ix, y);
  }

  if (ix >= 0x7ff00000) {
    y[0] = x - x;
    y[1] = y[0];
    return 0;
  }

  let z = setLowWord(0.0, getLowWord(x));
  const e0 = (ix >> 20) - 1046;
  z = setHighWord(z, ix - (e0 << 20));
  for (let i = 0; i < 2; i++) {
    txBuf[i] = Math.trunc(z);
    z = (z - txBuf[i]!) * two24;
  }
  txBuf[2] = z;
  let nx = 3;
  while (txBuf[nx - 1] === 0.0) nx--;
  const n = kernelRemPio2(txBuf, y, e0, nx);
  if (hx < 0) {
    y[0] = -y[0]!;
    y[1] = -y[1]!;
    return -n;
  }
  return n;
}
