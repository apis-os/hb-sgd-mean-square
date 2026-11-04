// src/math/float-words.ts — read and write the 32-bit halves of an IEEE-754 double (FDLIBM GET_HIGH_WORD & co.)
// Ported from an earlier TypeScript port of FDLIBM 5.3 (fdlibm-core.ts). FDLIBM notice: NOTICE.

const buf = new ArrayBuffer(8);
const f64 = new Float64Array(buf);
const u32 = new Uint32Array(buf);
const i32 = new Int32Array(buf);
const isLittleEndian = new Uint8Array(new Uint32Array([0x12345678]).buffer)[0] === 0x78;
export const HI = isLittleEndian ? 1 : 0;
export const LO = isLittleEndian ? 0 : 1;

/** High word as unsigned 32-bit. */
export function getHighWord(x: number): number {
  f64[0] = x;
  return u32[HI]!;
}

/** High word as signed 32-bit (FDLIBM `int hx`). */
export function getSignedHighWord(x: number): number {
  f64[0] = x;
  return i32[HI]!;
}

/** Low word as unsigned 32-bit. */
export function getLowWord(x: number): number {
  f64[0] = x;
  return u32[LO]!;
}

/** Low word as signed 32-bit. */
export function getSignedLowWord(x: number): number {
  f64[0] = x;
  return i32[LO]!;
}

/** Double with the given high and low words. */
export function setWords(hi: number, lo: number): number {
  u32[HI] = hi >>> 0;
  u32[LO] = lo >>> 0;
  return f64[0]!;
}

/** x with its high word replaced. */
export function setHighWord(x: number, hi: number): number {
  f64[0] = x;
  u32[HI] = hi >>> 0;
  return f64[0]!;
}

/** x with its low word replaced. */
export function setLowWord(x: number, lo: number): number {
  f64[0] = x;
  u32[LO] = lo >>> 0;
  return f64[0]!;
}
