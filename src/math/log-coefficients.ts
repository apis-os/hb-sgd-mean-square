// src/math/log-coefficients.ts — ln2 split and the Lg1..Lg7 Remez coefficients used by log (FDLIBM e_log.c)
// Ported from an earlier TypeScript port of FDLIBM 5.3 (fdlibm-exp-log.ts).

export const ln2_hi = 6.93147180369123816490e-01;
export const ln2_lo = 1.90821492927058770002e-10;
export const two54 = 1.80143985094819840000e+16;
export const Lg1 = 6.666666666666735130e-01;
export const Lg2 = 3.999999999940941908e-01;
export const Lg3 = 2.857142874366239149e-01;
export const Lg4 = 2.222219843214978396e-01;
export const Lg5 = 1.818357216161805012e-01;
export const Lg6 = 1.531383769920937332e-01;
export const Lg7 = 1.479819860511658591e-01;

/** k_log.h: log(1+f) - f + f*f/2 for 1+f in [sqrt(2)/2, sqrt(2)]. */
export function kLog1p(f: number): number {
  const s = f / (2.0 + f);
  const z = s * s;
  const w = z * z;
  const t1 = w * (Lg2 + w * (Lg4 + w * Lg6));
  const t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)));
  const R = t2 + t1;
  const hfsq = 0.5 * f * f;
  return s * (hfsq + R);
}
