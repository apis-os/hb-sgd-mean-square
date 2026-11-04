/**
 * @file labs/mean-square-boundary/prng-xorshift128p.ts
 * @description Deterministic 64-bit integer PRNG (`createXorshift128Plus`) with exact Box-Muller Gaussian sampling via FDLIBM (`fdlibmLog`, `fdlibmCos`, `fdlibmSin`).
 */
import { fdlibmCos, fdlibmLog, fdlibmSin } from '../../src/math/index.ts';

interface Xorshift128Plus {
  nextU32(): number;
  nextFloat01(): number;
  nextGaussian(): number;
}

export function createXorshift128Plus(seedLo = 0x6a09e667, seedHi = 0xbb67ae85): Xorshift128Plus {
  let s0 = (seedLo | 0) !== 0 ? seedLo | 0 : 0x12345678;
  let s1 = (seedHi | 0) !== 0 ? seedHi | 0 : 0x9abcdef0;
  let s2 = (s0 ^ 0x3c6ef372) | 0;
  let s3 = (s1 ^ 0xa54ff53a) | 0;
  let hasSpare = false;
  let spare = 0.0;

  const nextU32 = (): number => {
    let t = s3;
    const s = s0;
    s3 = s2;
    s2 = s1;
    s1 = s;
    t = (t ^ (t << 11)) | 0;
    t = (t ^ (t >>> 8)) | 0;
    s0 = (t ^ s ^ (s >>> 19)) | 0;
    return (s0 + s1) >>> 0;
  };

  const nextFloat01 = (): number => {
    return ((nextU32() >>> 8) + 0.5) * 5.960464477539063e-8;
  };

  const nextGaussian = (): number => {
    if (hasSpare) {
      hasSpare = false;
      return spare;
    }
    const u1 = Math.max(1e-15, nextFloat01());
    const u2 = nextFloat01();
    const radius = Math.sqrt(-2.0 * fdlibmLog(u1));
    const theta = 2.0 * Math.PI * u2;
    spare = radius * fdlibmSin(theta);
    hasSpare = true;
    return radius * fdlibmCos(theta);
  };

  return {
    nextU32,
    nextFloat01,
    nextGaussian,
  };
}
