/**
 * @file labs/mean-square-boundary/create-dense-deviation-noise-operator.ts
 * @description The curvature-noise operator of Theorem G.6 on symmetric `n x n` matrices from dense per-sample deviations `D_i`
 * given in the eigenbasis of the mean curvature (`createDenseDeviationNoiseOperator`; `devInput` is sample-major,
 * `devInput[i n^2 + a n + b]`, symmetrised on construction, `(D_i + D_i^T) / 2`, so that `V` preserves the positive semidefinite
 * cone). The deviations are centred exactly: `V(X) = (1 / (B N)) (sum_i D_i X D_i - S X S / N)`, `S = sum_i D_i`, so the iteration
 * certified by Phase 6 has per-sample curvature `diag(lt) + D_i - mean_j D_j` and mean exactly `diag(lt)`.
 *
 * Scaling. The deviations are multiplied once by the power of two `sD` that brings `max |D_i|` into `[0.5, 1)` (before they are
 * symmetrised, so that the symmetrisation cannot overflow or flush a scaled value; third audit round), and every `x` by the power of
 * two `sX` that does the same for `max |x|` (`computePowerOfTwoScale`); the products are formed on the scaled values, whose entries are
 * at most 1, and the result is multiplied back by `2^(2 eD + eX)` with `multiplyByPowerOfTwo` on that net exponent (steps of one
 * sign, no intermediate overflow when the result is finite), then by `1 / (B N)`. This replaces the trailing `1e-290` of an earlier
 * version, which the second audit round showed does not bound a product that underflows and is then multiplied by a large entry.
 *
 * Error bound `errOut` (entrywise, `|fl(V(x)) - V(x)| <= errOut`), in scaled units and then rescaled the same way: the sample term
 * loses at most `gamma_{2n + N + 3}` times `sum_i |D_i| |x| |D_i|`; the `S` term, with `|fl(S) - S| <= gamma_N Sabs`,
 * `Sabs = sum_i |D_i|`, at most `(gamma_{2n + 2}(1 + g)^2 + 2 g (1 + g) + g^2)(1 + g)^2 (1 + gamma_{2n + 4}) (Sabs |x| Sabs) / N`,
 * `g = gamma_N`; the subtraction, the symmetrisation sum, the reciprocal `1 / (B N)` and the product by it add `gamma_4` times
 * `(|out_ab| + |out_ba|) / 2`; gradual underflow, in scaled units where every factor is at most 1 in magnitude, adds at most
 * `U = 8 (N + 1)(n + 1)^2 2^-1074` (each underflowing product or scaled input errs by at most `2^-1075`, a symmetrised scaled entry
 * by at most `2.5 2^-1075`; in the sample term it is afterwards multiplied by unit-bounded factors and summed over at most `N n^2`
 * terms, and in `S x S`, whose entries of `S` reach `N`, by at most `N` and summed over `n^2` terms before the division by `N`; sixth
 * audit round: the earlier text said every factor was unit-bounded, false for `S`); the rescaling and the scale
 * `1 / (B N)` add `16 2^-1074` (at most four power-of-two steps and one product, each rounding at most `2^-1075` on the value and
 * on its bound). Domain: `n` in `[1, 4096]`, `N`, `B` integers in `[1, 2^20]` (so `B N` is exact and `1 / (B N)` is a
 * normal number; third audit round), finite entries, the largest entry of `D` and of `x` in `[2^-1000, 2^1000]` (or zero), `x`
 * exactly symmetric, `out` and `errOut` of length at least `n^2` sharing no memory with each other or with `x`, deviations not on
 * shared memory; otherwise,
 * or when a rescaled value overflows, `out` = NaN and `errOut` = Infinity, which every certificate rejects. Cost `4 N n^3 + 2 n^3` multiply-adds without the bound (the absolute sums of the sample term are formed
 * even then, `2 N n^3` of them discarded) and `4 N n^3 + 4 n^3` with it (eighth audit round: the text said twice that). For small `n`
 * (stages 0 and 1 of Phase 6); the rank-factor form for the Gauss–Newton row space is `createRowSpaceFactorNoiseOperator`.
 * @cold
 */
import { computePowerOfTwoScale } from './compute-power-of-two-scale.ts';
import { computeRoundoffGamma } from './compute-roundoff-gamma.ts';
import { detectFloat64BufferOverlap } from './detect-float64-buffer-overlap.ts';
import { fillFloat64View } from './fill-float64-view.ts';
import { multiplyByPowerOfTwo } from './multiply-by-power-of-two.ts';
import { phase6NoiseOperatorBrand } from './phase6-noise-operator-brand.ts';
import { readFloat64View } from './read-float64-view.ts';

const TINY = 4.9406564584124654e-324;

function maxAbs(a: Float64Array, len: number): number {
  let m = 0.0;
  for (let e = 0; e < len; e++) {
    const v = a[e]!;
    if (!Number.isFinite(v)) return Infinity;
    const av = Math.abs(v);
    if (av > m) m = av;
  }
  return m;
}

export function createDenseDeviationNoiseOperator(
  devInput: Float64Array,
  n: number,
  numSamples: number,
  batchB: number,
): { readonly n: number; apply(x: Float64Array, out: Float64Array, errOut: Float64Array | null): void } {
  const nn = n * n;
  const inRange = (k: number, hi: number): boolean => Number.isInteger(k) && k >= 1 && k <= hi;
  const validShape = inRange(n, 4096) && inRange(numSamples, 1048576) && inRange(batchB, 1048576) && readFloat64View(devInput, numSamples * nn) !== null;
  const dMax = validShape ? maxAbs(devInput, numSamples * nn) : Infinity;
  const zeroNoise = dMax === 0.0;
  const sD = zeroNoise || !Number.isFinite(dMax) ? null : computePowerOfTwoScale(dMax);
  const invalid = !validShape || !Number.isFinite(dMax) || (!zeroNoise && sD === null);
  const dev = new Float64Array(invalid ? 0 : numSamples * nn);
  if (sD !== null) {
    const sc = sD.scale;
    for (let i = 0; i < numSamples; i++) {
      for (let a = 0; a < n; a++) {
        for (let b = 0; b < n; b++) dev[i * nn + a * n + b] = 0.5 * (devInput[i * nn + a * n + b]! * sc + devInput[i * nn + b * n + a]! * sc);
      }
    }
  }
  const g4 = computeRoundoffGamma(4);
  const g = computeRoundoffGamma(2 * n + numSamples + 3);
  const gN = computeRoundoffGamma(numSamples);
  const gS = (computeRoundoffGamma(2 * n + 2) * (1.0 + gN) * (1.0 + gN) + 2.0 * gN * (1.0 + gN) + gN * gN) * (1.0 + gN) * (1.0 + gN) * (1.0 + computeRoundoffGamma(2 * n + 4));
  const underflow = 8.0 * (numSamples + 1) * (n + 1) * (n + 1) * TINY;
  const scale = 1.0 / (numSamples * batchB);
  // Scratch sized 0 for an invalid shape, which every call refuses (eleventh audit round: n = Infinity threw a RangeError here).
  const sSum = new Float64Array(invalid ? 0 : nn);
  const sAbs = new Float64Array(invalid ? 0 : nn);
  for (let i = 0; i < (invalid ? 0 : numSamples); i++) {
    for (let e = 0; e < nn; e++) {
      sSum[e] = sSum[e]! + dev[i * nn + e]!;
      sAbs[e] = sAbs[e]! + Math.abs(dev[i * nn + e]!);
    }
  }
  const t = new Float64Array(invalid ? 0 : nn);
  const ta = new Float64Array(invalid ? 0 : nn);
  const xs = new Float64Array(invalid ? 0 : nn);
  const ax = new Float64Array(invalid ? 0 : nn);
  const t2 = new Float64Array(invalid ? 0 : nn);
  const ss = new Float64Array(invalid ? 0 : nn);
  const sandwich = (left: Float64Array, mid: Float64Array, outM: Float64Array): void => {
    for (let a = 0; a < n; a++) {
      for (let b = 0; b < n; b++) {
        let s = 0.0;
        for (let k = 0; k < n; k++) s += left[a * n + k]! * mid[k * n + b]!;
        t2[a * n + b] = s;
      }
    }
    for (let a = 0; a < n; a++) {
      for (let b = 0; b < n; b++) {
        let s = 0.0;
        for (let k = 0; k < n; k++) s += t2[a * n + k]! * left[k * n + b]!;
        outM[a * n + b] = s;
      }
    }
  };
  const fail = (out: Float64Array, errOut: Float64Array | null): void => {
    fillFloat64View(out, NaN, nn);
    if (errOut !== null) fillFloat64View(errOut, Infinity, nn);
  };
  // Fourth audit round: a non-symmetric `x` was symmetrised silently (the bound was then for `(x + x^T) / 2`), and `out === errOut`
  // overwrote the value with its bound; both, short buffers and any overlap with `x`, are refused.
  const argumentsValid = (x: Float64Array, out: Float64Array, errOut: Float64Array | null): boolean => {
    // Through readFloat64View only: genuine non-shared Float64Arrays of length at least n^2 (ninth audit round).
    if (readFloat64View(x, nn) === null || readFloat64View(out, nn) === null || detectFloat64BufferOverlap(out, nn, x, nn)) return false;
    if (errOut !== null && (readFloat64View(errOut, nn) === null || detectFloat64BufferOverlap(errOut, nn, out, nn) || detectFloat64BufferOverlap(errOut, nn, x, nn))) return false;
    for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) if (x[a * n + b] !== x[b * n + a]) return false;
    return true;
  };
  const apply = (x: Float64Array, out: Float64Array, errOut: Float64Array | null): void => {
    if (invalid || !argumentsValid(x, out, errOut)) { fail(out, errOut); return; }
    const xMax = maxAbs(x, nn);
    if (!Number.isFinite(xMax)) { fail(out, errOut); return; }
    if (xMax === 0.0) {
      fillFloat64View(out, 0.0, nn);
      if (errOut !== null) fillFloat64View(errOut, 0.0, nn);
      return;
    }
    // The range of x is checked before the zero-noise shortcut (sixth audit round: x = 2^1001 returned 0 with zero noise).
    const sX = computePowerOfTwoScale(xMax);
    if (sX === null) { fail(out, errOut); return; }
    if (zeroNoise) {
      fillFloat64View(out, 0.0, nn);
      if (errOut !== null) fillFloat64View(errOut, 0.0, nn);
      return;
    }
    if (sD === null) { fail(out, errOut); return; }
    for (let e = 0; e < nn; e++) { xs[e] = x[e]! * sX.scale; ax[e] = Math.abs(xs[e]!); }
    fillFloat64View(out, 0.0, nn);
    if (errOut !== null) fillFloat64View(errOut, 0.0, nn);
    for (let i = 0; i < numSamples; i++) {
      const base = i * nn;
      for (let a = 0; a < n; a++) {
        for (let b = 0; b < n; b++) {
          let s = 0.0;
          let sa = 0.0;
          for (let k = 0; k < n; k++) {
            const d = dev[base + a * n + k]!;
            s += d * xs[k * n + b]!;
            sa += Math.abs(d) * ax[k * n + b]!;
          }
          t[a * n + b] = s;
          ta[a * n + b] = sa;
        }
      }
      for (let a = 0; a < n; a++) {
        for (let b = 0; b < n; b++) {
          let s = 0.0;
          let sa = 0.0;
          for (let k = 0; k < n; k++) {
            const d = dev[base + k * n + b]!;
            s += t[a * n + k]! * d;
            sa += ta[a * n + k]! * Math.abs(d);
          }
          out[a * n + b] = out[a * n + b]! + s;
          if (errOut !== null) errOut[a * n + b] = errOut[a * n + b]! + sa;
        }
      }
    }
    if (errOut !== null) for (let e = 0; e < nn; e++) errOut[e] = errOut[e]! * g * (1.0 + g);
    sandwich(sSum, xs, ss);
    for (let e = 0; e < nn; e++) out[e] = out[e]! - ss[e]! / numSamples;
    if (errOut !== null) {
      sandwich(sAbs, ax, ss);
      for (let e = 0; e < nn; e++) errOut[e] = errOut[e]! + (gS * ss[e]!) / numSamples;
    }
    // Undo the scaling: 2^(2 eD + eX) on the net exponent first, then the normal factor 1 / (B N).
    const netExp = 2 * sD.exponent + sX.exponent;
    for (let a = 0; a < n; a++) {
      for (let b = a; b < n; b++) {
        const oab = out[a * n + b]!;
        const oba = out[b * n + a]!;
        const v = multiplyByPowerOfTwo(0.5 * (oab + oba), netExp) * scale;
        let e = 0.0;
        if (errOut !== null) e = multiplyByPowerOfTwo(((0.5 * (errOut[a * n + b]! + errOut[b * n + a]!) + g4 * 0.5 * (Math.abs(oab) + Math.abs(oba))) * (1.0 + g4) + underflow) * (1.0 + g4), netExp) * scale * (1.0 + g4);
        if (!Number.isFinite(v) || !Number.isFinite(e)) { fail(out, errOut); return; }
        out[a * n + b] = v;
        out[b * n + a] = v;
        if (errOut !== null) {
          const eUp = e + 16.0 * TINY;
          errOut[a * n + b] = eUp;
          errOut[b * n + a] = eUp;
        }
      }
    }
  };
  const operator = Object.freeze({ n, apply });
  phase6NoiseOperatorBrand.add(operator);
  return operator;
}
