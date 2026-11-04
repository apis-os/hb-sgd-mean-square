/**
 * @file labs/mean-square-boundary/create-row-space-factor-noise-operator.ts
 * @description The curvature-noise operator of Theorem G.6 for the Gauss–Newton iteration on the row space of the stacked Jacobian
 * (`createRowSpaceFactorNoiseOperator`), from the per-sample factors `M_i` (`dOut x R`, sample-major `factors[i dOut R + c R + k]`,
 * `M_i[c, k] = (J_i u_k)_c`): per-sample curvature `A_i = M_i^T M_i`, exact mean `Hbar = (1/N) sum_i A_i`, and
 * `V(X) = (1 / (B N)) (sum_i A_i X A_i - N Hbar X Hbar)`, i.e. the deviations `A_i - Hbar`. With the mean curvature taken as
 * `diag(lt)` passed to the certificate (`lt = fl(lambda + mu)` in stage 2; plan section 2), the certified iteration has per-sample
 * matrices `A_i - Hbar + diag(lt)`: the deviations of the stored factors and the mean of the certificate exactly; `meanResidual` is `||fl(Hbar) - diag(lambda)||_F`,
 * reported beside the bracket. `S = sum_i A_i` is formed once.
 * `apply(x, out, errOut)`: `Y_i = M_i X` (`dOut x R`), `P_i = Y_i M_i^T` (`dOut x dOut`), `sum_i M_i^T (P_i M_i)`, minus
 * `S X S / N`, scaled by `1 / (B N)`, symmetrised. Error bound (entrywise): every summand of the sample term passes through `R`
 * roundings in `Y`, `R` in `P`, `dOut` in `P M`, one product and `N dOut - 1` additions into the same entry of `out` (all `(i, c)`
 * pairs accumulate in one slot), so it loses at most `gamma_{2R + dOut + N dOut + 4}` times `sum_i |M_i|^T |M_i| |X| |M_i|^T |M_i|`
 * (formed in floating point the same way, then enlarged by the same factor); the `S` term, with `|fl(S) - S| <= gamma_{N dOut} Sabs`,
 * `Sabs = sum_i |M_i|^T |M_i|` (whose own rounding, a factor `1 + gamma_{N dOut}` in each of the two copies in `Sabs |X| Sabs`, is
 * carried as `(1 + g)^2` inside the scalar `gS`; twelfth audit round), loses at most
 * `(gamma_{2R + 2}(1 + g)^2 + 2 g (1 + g) + g^2) (Sabs |X| Sabs) / N`, `g = gamma_{N dOut}`; the subtraction, the symmetrisation sum,
 * the reciprocal `1 / (B N)` and the product by it add `gamma_4` times `(|out_ab| + |out_ba|) / 2`. The counting of the first version
 * (`gamma_{2R + 2 dOut + N + 4}`) missed the `N dOut` accumulations; an external audit of 2026-10-29 built inputs on which it put
 * `eta_up` under `eta*` and `eta_low` above it.
 * Scaling (after the second audit round): the factors are multiplied once by the power of two `sM` that brings `max |M_i|` into
 * `[0.5, 1)`, and every `x` by the power of two `sX` that does the same for `max |x|` (`computePowerOfTwoScale`); all products are
 * formed on scaled values and the result is multiplied back by `2^(4 eM + eX)` with `multiplyByPowerOfTwo` on that net exponent
 * (steps of one sign; third audit round), then by `1 / (B N)`. In scaled units every intermediate entry of the sample term is at most `N dOut^2 R^2` in magnitude, and `S X S`, whose entries
 * of `S` reach `N dOut`, at most `N^2 dOut^2 R^2` before the division by `N` (sixth audit round: the earlier text gave the first bound
 * for both), so a product that underflows (or a scaled input that becomes subnormal), each in error by at most `2^-1075`, contributes at most `U = 16 (N + 1)(dOut + 1)^2 (R + 1)^2 2^-1074`
 * to an entry; the rescaling and the scale add `24 2^-1074`. Domain: `R` in `[1, 4096]`, `dOut` in `[1, 1024]`, `N`, `B` in
 * `[1, 2^20]`, finite entries with the largest factor and `x` entry in `[2^-1000, 2^1000]` (or zero), `x` exactly symmetric, `out`
 * and `errOut` of length at least `R^2` sharing no memory with each other or with `x`; otherwise, or when a rescaled
 * value overflows, `out` = NaN and `errOut` = Infinity, which every certificate rejects.
 * `meanResidual` is computed from the unscaled factors (a diagnostic, not certified). Cost per application
 * `2 N R^2 dOut + 2 N R dOut^2 + 2 R^3` multiply-adds, twice that with the bound (seventh audit round: the `N R dOut^2` term was
 * missing). Factors or `lambda` on shared memory make the operator refuse every call, as an invalid shape does.
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

function multiplySquare(a: Float64Array, b: Float64Array, out: Float64Array, n: number): void {
  out.fill(0.0);
  for (let i = 0; i < n; i++) {
    const ri = i * n;
    for (let k = 0; k < n; k++) {
      const aik = a[ri + k]!;
      if (aik === 0.0) continue;
      const rk = k * n;
      for (let j = 0; j < n; j++) out[ri + j] = out[ri + j]! + aik * b[rk + j]!;
    }
  }
}

export function createRowSpaceFactorNoiseOperator(
  factors: Float64Array,
  lambda: Float64Array,
  rank: number,
  numSamples: number,
  dOut: number,
  batchB: number,
): { readonly n: number; readonly meanResidual: number; apply(x: Float64Array, out: Float64Array, errOut: Float64Array | null): void } {
  const r = rank;
  const rr = r * r;
  const blk = dOut * r;
  const inRange = (k: number, hi: number): boolean => Number.isInteger(k) && k >= 1 && k <= hi;
  const validShape = inRange(r, 4096) && inRange(numSamples, 1048576) && inRange(dOut, 1024) && inRange(batchB, 1048576) && readFloat64View(factors, numSamples * blk) !== null && readFloat64View(lambda, r) !== null;
  let mMax = validShape ? 0.0 : Infinity;
  for (let e = 0; e < (validShape ? numSamples * blk : 0); e++) {
    const v = factors[e]!;
    if (!Number.isFinite(v)) { mMax = Infinity; break; }
    mMax = Math.max(mMax, Math.abs(v));
  }
  const zeroNoise = mMax === 0.0;
  const sM = zeroNoise || !Number.isFinite(mMax) ? null : computePowerOfTwoScale(mMax);
  const invalid = !Number.isFinite(mMax) || (!zeroNoise && sM === null);
  const fs = new Float64Array(invalid ? 0 : numSamples * blk);
  if (sM !== null) for (let e = 0; e < fs.length; e++) fs[e] = factors[e]! * sM.scale;
  const accumulateGram = (src: Float64Array, sum: Float64Array, sumAbs: Float64Array | null): void => {
    for (let i = 0; i < numSamples; i++) {
      const base = i * blk;
      for (let c = 0; c < dOut; c++) {
        const row = base + c * r;
        for (let a = 0; a < r; a++) {
          const ma = src[row + a]!;
          if (ma === 0.0) continue;
          for (let b = 0; b < r; b++) {
            const mb = src[row + b]!;
            sum[a * r + b] = sum[a * r + b]! + ma * mb;
            if (sumAbs !== null) sumAbs[a * r + b] = sumAbs[a * r + b]! + Math.abs(ma) * Math.abs(mb);
          }
        }
      }
    }
  };
  const sRaw = new Float64Array(invalid ? 0 : rr);
  if (!invalid) accumulateGram(factors, sRaw, null);
  // Scratch sized 0 for an invalid shape, which every call refuses (eleventh audit round).
  const s = new Float64Array(invalid ? 0 : rr);
  const sAbs = new Float64Array(invalid ? 0 : rr);
  if (sM !== null) accumulateGram(fs, s, sAbs);
  const s0 = sRaw;
  let res2 = invalid ? NaN : 0.0;
  for (let a = 0; a < (invalid ? 0 : r); a++) {
    for (let b = 0; b < r; b++) {
      const d = s0[a * r + b]! / numSamples - (a === b ? lambda[a]! : 0.0);
      res2 += d * d;
    }
  }
  const y = new Float64Array(invalid ? 0 : blk);
  const ya = new Float64Array(invalid ? 0 : blk);
  const pm = new Float64Array(invalid ? 0 : blk);
  const pma = new Float64Array(invalid ? 0 : blk);
  const p = new Float64Array(invalid ? 0 : dOut * dOut);
  const pa = new Float64Array(invalid ? 0 : dOut * dOut);
  const ax = new Float64Array(invalid ? 0 : rr);
  const xs = new Float64Array(invalid ? 0 : rr);
  const underflow = 16.0 * (numSamples + 1) * (dOut + 1) * (dOut + 1) * (r + 1) * (r + 1) * TINY;
  const t1 = new Float64Array(invalid ? 0 : rr);
  const t2 = new Float64Array(invalid ? 0 : rr);
  const g = computeRoundoffGamma(numSamples * dOut);
  const gSample = computeRoundoffGamma(2 * r + dOut + numSamples * dOut + 4);
  const gS = (computeRoundoffGamma(2 * r + 2) * (1.0 + g) * (1.0 + g) + 2.0 * g * (1.0 + g) + g * g) * (1.0 + g) * (1.0 + g) * (1.0 + computeRoundoffGamma(2 * r + 2));
  const g4 = computeRoundoffGamma(4);
  const scale = 1.0 / (batchB * numSamples);
  // Fourth audit round: a non-symmetric `x` was symmetrised silently (the bound was then for `(x + x^T) / 2`), and `out === errOut`
  // overwrote the value with its bound; both, short buffers and any overlap with `x`, are refused.
  const argumentsValid = (x: Float64Array, out: Float64Array, errOut: Float64Array | null): boolean => {
    // Through readFloat64View only: genuine non-shared Float64Arrays of length at least R^2 (ninth audit round).
    if (readFloat64View(x, rr) === null || readFloat64View(out, rr) === null || detectFloat64BufferOverlap(out, rr, x, rr)) return false;
    if (errOut !== null && (readFloat64View(errOut, rr) === null || detectFloat64BufferOverlap(errOut, rr, out, rr) || detectFloat64BufferOverlap(errOut, rr, x, rr))) return false;
    for (let a = 0; a < r; a++) for (let b = a + 1; b < r; b++) if (x[a * r + b] !== x[b * r + a]) return false;
    return true;
  };
  const apply = (x: Float64Array, out: Float64Array, errOut: Float64Array | null): void => {
    const withErr = errOut !== null;
    const fail = (): void => { fillFloat64View(out, NaN, rr); if (errOut !== null) fillFloat64View(errOut, Infinity, rr); };
    if (invalid || !argumentsValid(x, out, errOut)) { fail(); return; }
    let xMax = 0.0;
    for (let e = 0; e < rr; e++) {
      const v = x[e]!;
      if (!Number.isFinite(v)) { fail(); return; }
      xMax = Math.max(xMax, Math.abs(v));
    }
    if (xMax === 0.0) { fillFloat64View(out, 0.0, rr); if (errOut !== null) fillFloat64View(errOut, 0.0, rr); return; }
    // The range of x is checked before the zero-noise shortcut (sixth audit round).
    const sX = computePowerOfTwoScale(xMax);
    if (sX === null) { fail(); return; }
    if (zeroNoise) { fillFloat64View(out, 0.0, rr); if (errOut !== null) fillFloat64View(errOut, 0.0, rr); return; }
    if (sM === null) { fail(); return; }
    for (let e = 0; e < rr; e++) { xs[e] = x[e]! * sX.scale; ax[e] = Math.abs(xs[e]!); }
    fillFloat64View(out, 0.0, rr);
    if (withErr) fillFloat64View(errOut, 0.0, rr);
    for (let i = 0; i < numSamples; i++) {
      const base = i * blk;
      for (let c = 0; c < dOut; c++) {
        const row = base + c * r;
        for (let b = 0; b < r; b++) { y[c * r + b] = 0.0; ya[c * r + b] = 0.0; }
        for (let k = 0; k < r; k++) {
          const m = fs[row + k]!;
          if (m === 0.0) continue;
          const am = Math.abs(m);
          const xk = k * r;
          for (let b = 0; b < r; b++) {
            y[c * r + b] = y[c * r + b]! + m * xs[xk + b]!;
            if (withErr) ya[c * r + b] = ya[c * r + b]! + am * ax[xk + b]!;
          }
        }
      }
      for (let c = 0; c < dOut; c++) {
        for (let d = 0; d < dOut; d++) {
          let sp = 0.0;
          let spa = 0.0;
          const rowD = base + d * r;
          for (let b = 0; b < r; b++) {
            const m = fs[rowD + b]!;
            sp += y[c * r + b]! * m;
            if (withErr) spa += ya[c * r + b]! * Math.abs(m);
          }
          p[c * dOut + d] = sp;
          pa[c * dOut + d] = spa;
        }
      }
      for (let c = 0; c < dOut; c++) {
        for (let b = 0; b < r; b++) {
          let sp = 0.0;
          let spa = 0.0;
          for (let d = 0; d < dOut; d++) {
            const m = fs[base + d * r + b]!;
            sp += p[c * dOut + d]! * m;
            if (withErr) spa += pa[c * dOut + d]! * Math.abs(m);
          }
          pm[c * r + b] = sp;
          pma[c * r + b] = spa;
        }
      }
      for (let c = 0; c < dOut; c++) {
        const row = base + c * r;
        for (let a = 0; a < r; a++) {
          const m = fs[row + a]!;
          if (m === 0.0) continue;
          const am = Math.abs(m);
          const ra = a * r;
          for (let b = 0; b < r; b++) {
            out[ra + b] = out[ra + b]! + m * pm[c * r + b]!;
            if (withErr) errOut[ra + b] = errOut[ra + b]! + am * pma[c * r + b]!;
          }
        }
      }
    }
    multiplySquare(s, xs, t1, r);
    multiplySquare(t1, s, t2, r);
    for (let e = 0; e < rr; e++) out[e] = out[e]! - t2[e]! / numSamples;
    if (withErr) {
      for (let e = 0; e < rr; e++) errOut[e] = errOut[e]! * gSample * (1.0 + gSample);
      multiplySquare(sAbs, ax, t1, r);
      multiplySquare(t1, sAbs, t2, r);
      for (let e = 0; e < rr; e++) errOut[e] = errOut[e]! + (gS * t2[e]! * (1.0 + gSample)) / numSamples;
    }
    // Undo the scaling: 2^(4 eM + eX) on the net exponent first, then the normal factor 1 / (B N).
    const netExp = 4 * sM.exponent + sX.exponent;
    for (let a = 0; a < r; a++) {
      for (let b = a; b < r; b++) {
        const oab = out[a * r + b]!;
        const oba = out[b * r + a]!;
        const v = multiplyByPowerOfTwo(0.5 * (oab + oba), netExp) * scale;
        let e = 0.0;
        if (withErr) e = multiplyByPowerOfTwo(((0.5 * (errOut[a * r + b]! + errOut[b * r + a]!) + g4 * 0.5 * (Math.abs(oab) + Math.abs(oba))) * (1.0 + g4) + underflow) * (1.0 + g4), netExp) * scale * (1.0 + g4);
        if (!Number.isFinite(v) || !Number.isFinite(e)) { fail(); return; }
        out[a * r + b] = v;
        out[b * r + a] = v;
        if (withErr) {
          const eUp = e + 24.0 * TINY;
          errOut[a * r + b] = eUp;
          errOut[b * r + a] = eUp;
        }
      }
    }
  };
  const operator = Object.freeze({ n: r, meanResidual: Math.sqrt(res2), apply });
  phase6NoiseOperatorBrand.add(operator);
  return operator;
}
