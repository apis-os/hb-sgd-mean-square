/**
 * @file labs/mean-square-boundary/compute-certified-rayleigh-upper-threshold.ts
 * @description A certified upper bound of the threshold of Theorem G.6 from one symmetric test matrix `z` in the eigenbasis of the
 * mean curvature (`computeCertifiedRayleighUpperThreshold`). Since `eta* = 1 / lambda_max(L0^{-1/2} B_beta L0^{-1/2})`, every
 * `z != 0` gives `eta* <= <z, L0 z> / <z, B_beta z>` when the denominator is positive, with
 * `<z, L0 z> = sum_ab (lt_a + lt_b) z_ab^2` and `<z, B_beta z> = sum_ab z_ab (d_ab z_ab + V(z)_ab / (1 - beta))`,
 * `d_ab = lt_a lt_b / (1 + beta) - c_beta (lt_a - lt_b)^2`. Both sums are formed in floating point with error bounds: the quadratic
 * form of `B_beta` loses at most `gamma_{n^2 + 16}` times the sum of the absolute values of its terms (each term at most sixteen
 * rounded operations deep, constants included; `c_beta` from `computeMomentumCommutatorCoefficient`) plus `sum |z_ab| errV_ab / (1 - beta)` from the bound of `noise.apply`; the
 * `L0` form has nonnegative terms and relative error at most `gamma_{n^2 + 4}`. `etaUp` is `null` when the lower end of the `B_beta`
 * form is not positive (the test matrix says nothing); the whole result is `null` when an input is outside the domain below, so a
 * caller can tell a refusal from an uninformative `z` (fourth audit round: the bracket fell back to `eta_det` on a NaN noise operator). `z` is first scaled to unit largest entry (the quotient is invariant), and both forms carry an additive
 * `n^2 1e-290` for gradual underflow (an external audit of 2026-10-29 found that a tiny `z` made the `L0` form underflow to zero and
 * returned `etaUp = 0`). Domain, checked here (the same as `certifyConeCollatzWielandtAtRate`): `n` an integer in `[1, 4096]`,
 * `beta = 0` or in `[2^-20, 1 - 2^-20]`, every `lt` in `[2^-60, 2^60]`, `z` finite and exactly symmetric, the noise output and its
 * bound finite and at most `2^300`, `lt` and `z` not on shared memory (seventh audit round); otherwise `null`. Inside it the `L0` form is at least `2^-60` and the quotient cannot underflow;
 * `2^-1074` is still added after the final division (third audit round). `etaUp` is rounded upwards by a factor `1 + 4u`; `quotient` is the uncertified `<z, B z> / <z, L0 z>`.
 * Dropping the `1 + 4u` and `2^-1074` after the division, or the `n^2 1e-290` terms, is not caught by any test (fourth audit round,
 * mutations `M1`, `M2`): the deepest term of the `B_beta` form, `z ((p1 - (c_beta diff) diff) z + w / (1 - beta))` with `c_beta`
 * counted as six, has at most fourteen roundings before the `n^2 - 1` additions (sixth audit round: the earlier text said twelve),
 * against the sixteen of `gamma_{n^2 + 16}`; the spare of about `3u numAbs >= 3u numLow` covers the two roundings that form `numLow`
 * and the division; `denHigh` has no spare (the `L0` form takes `n^2 + 2` roundings and forming `denHigh` two more, which is
 * `gamma_{n^2 + 4}` exactly; twelfth audit round, the text claimed two spare); inside the domain
 * `numAbs >= 2^-121`, so `gamma numAbs` dwarfs `n^2 1e-290`. The terms stay because the proof uses them, not that slack.
 * @cold
 */
import { computeMomentumCommutatorCoefficient } from './compute-momentum-commutator-coefficient.ts';
import { computeRoundoffGamma } from './compute-roundoff-gamma.ts';
import { copyFloat64View } from './copy-float64-view.ts';
import { phase6NoiseOperatorBrand } from './phase6-noise-operator-brand.ts';

export function computeCertifiedRayleighUpperThreshold(
  ltIn: Float64Array,
  n: number,
  beta: number,
  noise: { readonly n?: number; apply(x: Float64Array, out: Float64Array, errOut: Float64Array | null): void },
  zArg: Float64Array,
): { readonly etaUp: number | null; readonly quotient: number } | null {
  const nn = n * n;
  if (!Number.isInteger(n) || n < 1 || n > 4096) return null;
  // Private copies; anything but a genuine non-shared Float64Array is refused (seventh to ninth audit rounds).
  const lt = copyFloat64View(ltIn, n);
  const zIn = copyFloat64View(zArg, nn);
  if (lt === null || zIn === null) return null;
  // Only operators built by the two Phase 6 factories, which are frozen (thirteenth audit round; phase6NoiseOperatorBrand).
  if (!phase6NoiseOperatorBrand.has(noise)) return null;
  if (noise.n !== n) return null;
  // A primitive number only: a string or a coercible object passed the range test and then entered the arithmetic as another value
  // (tenth audit round: beta = '0.9' made 1 + beta the string '10.9').
  if (typeof beta !== 'number') return null;
  if (!(beta === 0.0 || (beta >= 9.5367431640625e-7 && beta <= 1.0 - 9.5367431640625e-7))) return null;
  for (let a = 0; a < n; a++) if (!(lt[a]! >= 8.673617379884035e-19 && lt[a]! <= 1.152921504606847e18)) return null;
  for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) if (!Number.isFinite(zIn[a * n + b]!) || zIn[a * n + b] !== zIn[b * n + a]) return null;
  let zMax = 0.0;
  for (let e = 0; e < nn; e++) zMax = Math.max(zMax, Math.abs(zIn[e]!));
  if (!(zMax > 0.0) || !Number.isFinite(zMax)) return null;
  const z = new Float64Array(nn);
  for (let e = 0; e < nn; e++) z[e] = zIn[e]! / zMax;
  const w = new Float64Array(nn);
  const werr = new Float64Array(nn);
  // apply receives its own copy; the forms read z. A copy changed by apply means the noise is not a function of the matrix it was
  // given, and nothing is returned (eleventh audit round: an apply that wrote its input made the forms use another z than V).
  const zApply = new Float64Array(nn);
  for (let e = 0; e < nn; e++) zApply[e] = z[e]!;
  w.fill(NaN);
  werr.fill(NaN);
  noise.apply(zApply, w, werr);
  for (let e = 0; e < nn; e++) if (!Object.is(zApply[e], z[e])) return null;
  for (let e = 0; e < nn; e++) if (!(Math.abs(w[e]!) <= 2.037035976334486e90) || !(werr[e]! >= 0.0 && werr[e]! <= 2.037035976334486e90)) return null;
  const inv1pb = 1.0 / (1.0 + beta);
  const inv1mb = 1.0 / (1.0 - beta);
  const cb = computeMomentumCommutatorCoefficient(beta);
  let num = 0.0;
  let numAbs = 0.0;
  let numErrV = 0.0;
  let den = 0.0;
  for (let a = 0; a < n; a++) {
    for (let b = 0; b < n; b++) {
      const zab = z[a * n + b]!;
      const diff = lt[a]! - lt[b]!;
      const p1 = lt[a]! * lt[b]! * inv1pb;
      const p2 = cb * diff * diff;
      const wab = w[a * n + b]!;
      num += zab * ((p1 - p2) * zab + wab * inv1mb);
      numAbs += Math.abs(zab) * ((p1 + p2) * Math.abs(zab) + Math.abs(wab) * inv1mb);
      numErrV += Math.abs(zab) * werr[a * n + b]! * inv1mb;
      den += (lt[a]! + lt[b]!) * zab * zab;
    }
  }
  const gNum = computeRoundoffGamma(nn + 16);
  const numLow = num - (gNum * numAbs + numErrV * (1.0 + gNum)) * (1.0 + gNum) - nn * 1e-290;
  if (!(numLow > 0.0)) return { etaUp: null, quotient: num / den };
  // The factor on den is load-bearing (seventh audit round: without it, z with tiny entries that round away in the L0 form and up in
  // the B form gave etaUp < eta_det = eta* at n >= 12; tested).
  const denHigh = den * (1.0 + computeRoundoffGamma(nn + 4)) + nn * 1e-290;
  const etaUp = (denHigh / numLow) * (1.0 + 4.440892098500626e-16) + 4.9406564584124654e-324;
  if (!Number.isFinite(etaUp)) return null;
  return { etaUp, quotient: num / den };
}
