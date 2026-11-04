// src/math/rem-pio2-kernel.ts — Payne-Hanek reduction of huge arguments modulo pi/2 (FDLIBM k_rem_pio2.c)
// Ported from an earlier TypeScript port of FDLIBM 5.3 (fdlibm-trig-large.ts).
import { PIo2, two_over_pi } from './rem-pio2-tables.ts';
import { scalbn } from './scalbn.ts';

const two24 = 1.67772160000000000000e+07;
const twon24 = 5.96046447753906250000e-08;
const JK = 4;
const iq = new Int32Array(20);
const fBuf = new Float64Array(20);
const qBuf = new Float64Array(20);
const fqBuf = new Float64Array(20);

export function kernelRemPio2(x: Float64Array, y: Float64Array, e0: number, nx: number): number {
  const jp = JK;
  let jz = JK;
  let i = 0;
  let j = 0;
  let k = 0;
  let n = 0;
  let ih = 0;
  let carry = 0;
  let z = 0.0;
  let fw = 0.0;

  const jx = nx - 1;
  let jv = Math.trunc((e0 - 3) / 24);
  if (jv < 0) jv = 0;
  let q0 = e0 - 24 * (jv + 1);

  j = jv - jx;
  const m = jx + JK;
  for (i = 0; i <= m; i++, j++) fBuf[i] = j < 0 ? 0.0 : two_over_pi[j]!;

  for (i = 0; i <= JK; i++) {
    fw = 0.0;
    for (j = 0; j <= jx; j++) fw += x[j]! * fBuf[jx + i - j]!;
    qBuf[i] = fw;
  }

  for (;;) {
    for (i = 0, j = jz, z = qBuf[jz]!; j > 0; i++, j--) {
      fw = Math.trunc(twon24 * z);
      iq[i] = Math.trunc(z - two24 * fw);
      z = qBuf[j - 1]! + fw;
    }

    z = scalbn(z, q0);
    z -= 8.0 * Math.floor(z * 0.125);
    n = Math.trunc(z);
    z -= n;
    ih = 0;
    if (q0 > 0) {
      i = iq[jz - 1]! >> (24 - q0);
      n += i;
      iq[jz - 1] = iq[jz - 1]! - (i << (24 - q0));
      ih = iq[jz - 1]! >> (23 - q0);
    } else if (q0 === 0) {
      ih = iq[jz - 1]! >> 23;
    } else if (z >= 0.5) {
      ih = 2;
    }

    if (ih > 0) {
      n += 1;
      carry = 0;
      for (i = 0; i < jz; i++) {
        j = iq[i]!;
        const newCarry = (carry === 0 && j !== 0) ? 1 : carry;
        iq[i] = (carry === 0 && j !== 0) ? 0x1000000 - j : (carry !== 0 ? 0xffffff - j : j);
        carry = newCarry;
      }
      iq[jz - 1] = iq[jz - 1]! & (q0 === 1 ? 0x7fffff : (q0 === 2 ? 0x3fffff : 0xffffff));
      z = ih === 2 ? (1.0 - z) - (carry !== 0 ? scalbn(1.0, q0) : 0.0) : z;
    }

    if (z !== 0.0) break;
    j = 0;
    for (i = jz - 1; i >= JK; i--) j |= iq[i]!;
    if (j !== 0) break;
    for (k = 1; iq[JK - k] === 0; k++) { /* k terms needed */ }
    for (i = jz + 1; i <= jz + k; i++) {
      fBuf[jx + i] = two_over_pi[jv + i]!;
      fw = 0.0;
      for (j = 0; j <= jx; j++) fw += x[j]! * fBuf[jx + i - j]!;
      qBuf[i] = fw;
    }
    jz += k;
  }

  if (z === 0.0) {
    jz -= 1;
    q0 -= 24;
    while (iq[jz] === 0) {
      jz--;
      q0 -= 24;
    }
  } else {
    z = scalbn(z, -q0);
    fw = Math.trunc(twon24 * z);
    iq[jz] = z >= two24 ? Math.trunc(z - two24 * fw) : Math.trunc(z);
    jz += z >= two24 ? 1 : 0;
    q0 += z >= two24 ? 24 : 0;
    iq[jz] = z >= two24 ? Math.trunc(fw) : iq[jz]!;
  }

  fw = scalbn(1.0, q0);
  for (i = jz; i >= 0; i--) {
    qBuf[i] = fw * iq[i]!;
    fw *= twon24;
  }

  for (i = jz; i >= 0; i--) {
    fw = 0.0;
    for (k = 0; k <= jp && k <= jz - i; k++) fw += PIo2[k]! * qBuf[i + k]!;
    fqBuf[jz - i] = fw;
  }

  fw = 0.0;
  for (i = jz; i >= 0; i--) fw += fqBuf[i]!;
  y[0] = ih === 0 ? fw : -fw;
  fw = fqBuf[0]! - fw;
  for (i = 1; i <= jz; i++) fw += fqBuf[i]!;
  y[1] = ih === 0 ? fw : -fw;

  return n & 7;
}
