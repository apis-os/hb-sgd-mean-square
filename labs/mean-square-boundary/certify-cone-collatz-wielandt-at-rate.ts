/**
 * @file labs/mean-square-boundary/certify-cone-collatz-wielandt-at-rate.ts
 * @description The Phase 6 lower-bound test at one rate (`certifyConeCollatzWielandtAtRate`): `true` proves `eta* > eta` for the
 * iteration with mean curvature `diag(lt)` and curvature noise `V` (Theorem G.6), given a symmetric test matrix `x`, `w = fl(V(x))`
 * and an entrywise bound `wErr` of `|w - V(x)|` (`noise.apply` with its bound, computed once and reused for every rate).
 *
 * Argument. For `eta < eta_det = 2(1 + beta) / max lt`, `K(X)_ab = (eta / (1 - beta)) V(X)_ab / l_ab`,
 * `l_ab = (lt_a + lt_b) + eta (-lt_a lt_b / (1 + beta) + c_beta (lt_a - lt_b)^2)`, preserves the cone of positive semidefinite
 * matrices, and so does its adjoint (the cone is self-dual); by Krein–Rutman the adjoint has an eigenmatrix `Y >= 0`, `Y != 0`, with
 * eigenvalue `rho(K)`. If `x > 0` and `x - K(x) > 0`, then `0 < <Y, x - K(x)> = (1 - rho(K)) <Y, x>` with `<Y, x> > 0`, so
 * `rho(K) < 1`, which is `eta < eta*` by Theorem G.6(1) and the final equivalence of its proof (`rho(K) < 1` iff
 * `L0 - eta B_beta > 0`, pointwise in `eta`; with `V = 0` the same chain gives `rho(K) = 0` and `eta* = eta_det`, outside the theorem's
 * hypothesis `V != 0` but proved the same way; eighth audit round). This function verifies both: `x > 0` with
 * `verifyPositiveDefiniteShiftedCholesky` (`rho = 0`; the last verified `x` is remembered by contents, so a bisection over one witness
 * factorises it once), and `x - K(x) > 0` as below. Before the third audit round the `x > 0` check was left to the caller, and a
 * direct call with an indefinite `x` certified rates above `eta*`.
 *
 * Domain (third audit round; inside it every quantity named below is a normal number or exactly zero, which the rounding argument
 * uses): `n` an integer in `[1, 4096]`, `beta = 0` or `beta` in `[2^-20, 1 - 2^-20]`, every `lt` in `[2^-60, 2^60]`, `eta` in
 * `[2^-200, eta_det)` with the margin `16u`, `x`, `w`, `wErr` finite with `|x|, |w|, wErr <= 2^300` and `wErr >= 0`, buffers of length
 * at least `n^2`, and the scratch `m`, `work` sharing no memory with each other or with `lt`, `x`, `w`, `wErr`
 * (`detectFloat64BufferOverlap`; fourth audit round: with `work === w` the Cholesky of `x` overwrote `w` and a rate above `eta*`
 * was certified); otherwise `false`. Then `coef = eta / (1 - beta)` lies in `[2^-200, 2^82]`,
 * `p1 = lt_a lt_b / (1 + beta)` in `[2^-121, 2^120]`, `p2 = c_beta (lt_a - lt_b)^2` is 0 or at least `2^-245` (`c_beta` at least
 * `2^-21`, `|lt_a - lt_b|` 0 or at least `2^-112`), `l_ab` is 0 or at least `2^-500` in magnitude (a multiple of the granularity of its
 * terms); `l_ab - dl` can be negative within about `32u` of `eta_det` (measured 30.5u to 32.5u; twelfth audit
 * round, the text said `64u`; the explicit `1 - 16u` guard is therefore implied by this test), which the test `l_ab - dl > 0` refuses; when positive it can
 * be far below `dl` by cancellation (seventh audit round: `6e-38` with `dl = 1.2e-32`; the sixth round's floor `2^-108` was false), but
 * it is a difference of two normal numbers of magnitude at least `2^-108`, hence a nonzero multiple of their spacing, at least
 * `2^-160`, and normal; the code divides by the actual value; and `dl / l_ab >= gamma_16 > 2^-50`. These are the coefficients and denominators of the data path; the inputs `x`, `w`, `wErr`
 * and their products (for instance `k = coef w / l`) may be subnormal, which the underflow budget `T` covers (eleventh audit round: the
 * earlier text read as if every data-path quantity were normal); `T`
 * below and the trailing `2^-1074` are subnormal on purpose and are terms of the proof, not slack (fourth audit round: an earlier
 * sentence of the plan said that every intermediate is normal or zero, which these terms contradict).
 *
 * Rounding. `l_ab` is formed with an error at most `dl = gamma_16 (lt_a + lt_b + eta (p1 + p2))(1 + gamma_16)` (`c_beta` from
 * `computeMomentumCommutatorCoefficient`, relative error at most `gamma_6`; the deepest term carries at most 13 roundings) and must
 * satisfy `l_ab - dl > 0`. With `q = dl / l_ab`, the entry `k = coef w_ab / l_ab` of `K` has error at most
 * `kErr = ((coef wErr_ab) / (l - dl) + ((coef |w_ab|) / (l - dl)) q)(1 + gamma_12)^2 + gamma_6 |k| (1 + gamma_12)` plus the underflow
 * term `T = ((2 coef + 4) / min(l - dl, 1))(1 + 4u) 2^-1074 + 4 2^-1074`: the `2 coef` part has no derivation in the order of operations used (no underflowed value is multiplied by
 * `coef`) and is extra margin, pinned only by the round-8 conservative test (twelfth audit round); every product or quotient above that underflows errs by at
 * most `2^-1075` and is afterwards divided at most by `l - dl` or multiplied by `q < 1` (the order that the third audit round asked
 * for: forming `|w| dl` first could underflow before a division by `l (l - dl)`). The difference `m = x - k` adds `u |m|`, the entry
 * bound is enlarged by `1 + 4u` plus `2^-1074` (which can fall short of four roundings, `(1 + u)^4`, by `O(u^2)`; the shift of the verified
 * Cholesky absorbs that, tenth audit round), and the Frobenius norm of the bounds (accumulated as `eMax^2 sum (e / eMax)^2`) bounds
 * the spectral norm of the error of the stored `m`; it is passed as `rho` to `verifyPositiveDefiniteShiftedCholesky`. `m` and `work` are
 * `n^2` scratch. Dropping `u |m|`, `1 + 4u` or the trailing `2^-1074` from `e` is not caught by any test (fourth audit round,
 * mutation `M12`). Their Frobenius sum is at most about `u ||m||_F` plus `4u` of the noise bound, which the shift of the verified
 * Cholesky always absorbs, since `rho < lambda_min(m)` whenever the certificate passes (twelfth audit round); the seventh audit round built loose noise bounds (`wErr` near `1/n`, `n = 256`) on which a mutant returns
 * `true` where the code returns `false`, at a rate below `eta*`, so the mutants change the boolean without a wrong-side certificate
 * having been found. The terms stay because the proof uses them.
 * Inputs that separate the `gamma_6 |k|` term of `kErr`, the `eta (p1 + p2)` part of `dl` and the underflow budget `T` (removing any
 * of them turns a refusal into `true` at a rate below `eta*`) were built in the eighth audit round and are pinned by tests; the
 * seventh round had listed the first two as having no separating input.
 * @cold
 */
import { computeMomentumCommutatorCoefficient } from './compute-momentum-commutator-coefficient.ts';
import { computeRoundoffGamma } from './compute-roundoff-gamma.ts';
import { copyFloat64View } from './copy-float64-view.ts';
import { detectFloat64BufferOverlap } from './detect-float64-buffer-overlap.ts';
import { readFloat64View } from './read-float64-view.ts';
import { verifyPositiveDefiniteShiftedCholesky } from './verify-positive-definite-shifted-cholesky.ts';

let lastVerified: Float64Array | null = null;

function witnessVerified(x: Float64Array, n: number, work: Float64Array): boolean {
  const nn = n * n;
  if (lastVerified !== null && lastVerified.length === nn) {
    let same = true;
    for (let q = 0; q < nn; q++) if (lastVerified[q] !== x[q]) { same = false; break; }
    if (same) return true;
  }
  if (!verifyPositiveDefiniteShiftedCholesky(x, n, 0.0, work)) return false;
  // x is the private copy made on entry; copied by index, no method of any argument is called (ninth audit round: a subclass whose
  // subarray returned other contents poisoned this cache).
  const copy = new Float64Array(nn);
  for (let q = 0; q < nn; q++) copy[q] = x[q]!;
  lastVerified = copy;
  return true;
}

export function certifyConeCollatzWielandtAtRate(
  ltIn: Float64Array,
  n: number,
  beta: number,
  xIn: Float64Array,
  wIn: Float64Array,
  wErrIn: Float64Array,
  eta: number,
  m: Float64Array,
  work: Float64Array,
): boolean {
  const u = 1.1102230246251565e-16;
  const tiny = 4.9406564584124654e-324;
  const big = 2.037035976334486e90; // 2^300
  if (!Number.isInteger(n) || n < 1 || n > 4096) return false;
  const nn = n * n;
  if (readFloat64View(m, nn) === null || readFloat64View(work, nn) === null) return false;
  if (detectFloat64BufferOverlap(m, nn, work, nn)) return false;
  if (detectFloat64BufferOverlap(m, nn, ltIn, n) || detectFloat64BufferOverlap(work, nn, ltIn, n)) return false;
  if (detectFloat64BufferOverlap(m, nn, xIn, nn) || detectFloat64BufferOverlap(work, nn, xIn, nn)) return false;
  if (detectFloat64BufferOverlap(m, nn, wIn, nn) || detectFloat64BufferOverlap(work, nn, wIn, nn)) return false;
  if (detectFloat64BufferOverlap(m, nn, wErrIn, nn) || detectFloat64BufferOverlap(work, nn, wErrIn, nn)) return false;
  // Private copies of the inputs (copyFloat64View refuses anything but a genuine, non-shared Float64Array of sufficient length).
  const lt = copyFloat64View(ltIn, n);
  const x = copyFloat64View(xIn, nn);
  const w = copyFloat64View(wIn, nn);
  const wErr = copyFloat64View(wErrIn, nn);
  if (lt === null || x === null || w === null || wErr === null) return false;
  // A primitive number only: a string or a coercible object passed the range test and then entered the arithmetic as another value
  // (tenth audit round: beta = '0.9' made 1 + beta the string '10.9').
  if (typeof beta !== 'number') return false;
  if (!(beta === 0.0 || (beta >= 9.5367431640625e-7 && beta <= 1.0 - 9.5367431640625e-7))) return false;
  if (typeof eta !== 'number' || !(eta >= 6.223015277861142e-61) || !Number.isFinite(eta)) return false;
  let ltMax = 0.0;
  for (let a = 0; a < n; a++) {
    if (!(lt[a]! >= 8.673617379884035e-19 && lt[a]! <= 1.152921504606847e18)) return false;
    ltMax = Math.max(ltMax, lt[a]!);
  }
  for (let q = 0; q < nn; q++) {
    if (!(Math.abs(x[q]!) <= big) || !(Math.abs(w[q]!) <= big) || !(wErr[q]! >= 0.0 && wErr[q]! <= big)) return false;
  }
  if (!(eta * ltMax < 2.0 * (1.0 + beta) * (1.0 - 16.0 * u))) return false;
  if (!witnessVerified(x, n, work)) return false;
  const inv1pb = 1.0 / (1.0 + beta);
  const cb = computeMomentumCommutatorCoefficient(beta);
  const coef = eta / (1.0 - beta);
  const g16 = computeRoundoffGamma(16);
  const g12 = computeRoundoffGamma(12);
  const g6 = computeRoundoffGamma(6);
  let eMax = 0.0;
  for (let a = 0; a < n; a++) {
    for (let b = 0; b < n; b++) {
      const sum = lt[a]! + lt[b]!;
      const diff = lt[a]! - lt[b]!;
      const p1 = lt[a]! * lt[b]! * inv1pb;
      const p2 = cb * diff * diff;
      const l = sum + eta * (p2 - p1);
      const dl = g16 * (sum + eta * (p1 + p2)) * (1.0 + g16);
      const lmd = l - dl;
      if (!(lmd > 0.0)) return false;
      const q = dl / l;
      const wab = w[a * n + b]!;
      const k = (coef * wab) / l;
      const main = ((coef * wErr[a * n + b]!) / lmd + ((coef * Math.abs(wab)) / lmd) * q) * (1.0 + g12) * (1.0 + g12);
      const under = ((2.0 * coef + 4.0) / Math.min(lmd, 1.0)) * (1.0 + 4.0 * u) * tiny + 4.0 * tiny;
      const kErr = main + g6 * Math.abs(k) * (1.0 + g12) + under;
      const mab = x[a * n + b]! - k;
      m[a * n + b] = mab;
      const e = (kErr + u * Math.abs(mab)) * (1.0 + 4.0 * u) + tiny;
      work[a * n + b] = e;
      if (e > eMax) eMax = e;
    }
  }
  let sumE2 = 0.0;
  for (let q = 0; q < nn; q++) { const r = work[q]! / eMax; sumE2 += r * r; }
  const rho = eMax * Math.sqrt(sumE2 * (1.0 + computeRoundoffGamma(nn + 2))) * (1.0 + 8.0 * u);
  return verifyPositiveDefiniteShiftedCholesky(m, n, rho, work);
}
