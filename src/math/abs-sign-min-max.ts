// src/math/abs-sign-min-max.ts — bit-level abs and sign, IEEE-754 min/max with signed-zero ordering
// Ported from an earlier TypeScript port of FDLIBM 5.3 (fdlibm-core.ts).
import { getHighWord, setHighWord } from './float-words.ts';

export function fdlibmAbs(x: number): number {
  const hi = getHighWord(x);
  return setHighWord(x, hi & 0x7fffffff);
}

export function fdlibmSign(x: number): number {
  if (Number.isNaN(x) || x === 0) return x;
  const hi = getHighWord(x);
  return (hi >>> 31) === 1 ? -1 : 1;
}

export function fdlibmMin(a: number, b: number): number {
  if (Number.isNaN(a) || Number.isNaN(b)) return Number.NaN;
  if (a === 0 && b === 0) {
    const isANegZero = (1 / a === Number.NEGATIVE_INFINITY);
    const isBNegZero = (1 / b === Number.NEGATIVE_INFINITY);
    return (isANegZero || isBNegZero) ? -0 : 0;
  }
  return a < b ? a : b;
}

export function fdlibmMax(a: number, b: number): number {
  if (Number.isNaN(a) || Number.isNaN(b)) return Number.NaN;
  if (a === 0 && b === 0) {
    const isAPosZero = (1 / a === Number.POSITIVE_INFINITY);
    const isBPosZero = (1 / b === Number.POSITIVE_INFINITY);
    return (isAPosZero || isBPosZero) ? 0 : -0;
  }
  return a > b ? a : b;
}
