/**
 * @file labs/mean-square-boundary/verify-positive-definite-shifted-cholesky.ts
 * @description A floating-point proof that a symmetric matrix is positive definite (`verifyPositiveDefiniteShiftedCholesky`), in the
 * manner of Rump (2006, *BIT* 46:433–452, Corollary 2.4): `a` is the stored row-major `n x n` matrix, and the matrix to certify is any
 * `A` with `||A - a||_2 <= rho`. The diagonal is shifted by `c = (2 gamma_{n+2} tr+(a)(1 + gamma_n) + rho)(1 + 8u)` (`tr+` the sum of
 * the positive diagonal entries), giving `at = fl(a - c I)`, and `at` is factorised by `computeCholeskyInPlace` in `work`
 * (left-looking, `l_ij = s fl(1 / l_jj)`, inner products in increasing `k`; the 4-row blocking changes no arithmetic).
 *
 * Without underflow every computed entry of `L L^T` differs from `at` by at most `gamma_{n+1} (|L| |L^T|)_ij` (Higham, 2nd ed.,
 * Theorem 10.3, with one more rounding for the reciprocal pivot; an external audit counted at most `j + 2 <= n + 1` roundings per
 * entry), so `lambda_min(at) >= -gamma_{n+1} ||L||_F^2`. With gradual underflow, each of the at most `j` inner products
 * `l_ik l_jk` that underflows errs by at most `2^-1075` in `s`, and an underflowing `l_ij = fl(s fl(1 / l_jj))` errs by at most
 * `2^-1075`, which is about `l_jj 2^-1075` in the entry of `L L^T - at` (fourth audit round: the earlier text bounded this
 * by `2^-1075`, false for `l_jj > 1`; `fl(sqrt)` and `fl(1 / l_jj)` cannot underflow above the pivot floor). Solving the recurrence
 * `s_{k+1} = fl(s_k - fl(l_ik l_jk))` backwards multiplies an absolute error by the inverse rounding factors of the later
 * subtractions, at most `(1 - u)^-(n + 2) < 2` for `n <= 4096` (sixth audit round: the earlier text carried only `(1 - u)^-2`). So
 * the underflow part `D` has `|D_ij| <= 2 (n + max(1, l_jj)) 2^-1075` and, summing in quadrature with
 * `(n + l)^2 <= 2 n^2 + 2 l^2`, `||D||_2 <= ||D||_F <= 2 sqrt(2) (n (n + 1) + sqrt(n) ||L||_F) 2^-1075` (tenth audit round: the earlier
 * constant was short of the entrywise majorant by a small factor).
 * The test uses `gamma_{n+2}`, one `u` more than needed, and requires every pivot above `1e-140`: then `||L||_F > 1e-140` and, for
 * `n <= 4096`, `||D||_2 < 2^-66 u ||L||_F^2` (the bound evaluates to `2^-66.36` at `n = 4096`, `||L||_F = 1e-140` with the tenth-round
 * constant; the seventh round corrected `2^-70`, the eleventh the value `2^-66.86` of the earlier constant), so that spare `u` absorbs the underflow (an earlier docblock attributed this to an
 * additive `n 1e-290` in the shift, which rounds away beside a normal shift; third audit round). The subtraction on the diagonal moves each entry by at most
 * `(u / (1 - u)) |at_ii|`, and in fact by `u |at_ii|` for round to nearest (`|fl(x) - x| <= u |fl(x)|`; twelfth audit round: the
 * tenth round's change to `u / (1 - u)` was not needed, and either is covered by the factor `1 + 8u` on `need`). Hence
 * `lambda_min(A) >= c - gamma_{n+2} ||L||_F^2 - (u / (1 - u)) max|at_ii| - rho`, and the function returns `true` only when this, with
 * every term enlarged for its own rounding, is positive. It returns `false` unless `n` is an integer in `[1, 4096]`, `work` holds
 * `n^2` entries sharing no memory with `a`, and `a` is finite and exactly symmetric as stored (the factorisation reads only the lower
 * triangle; an external audit built a non-symmetric `a` within `rho` of an indefinite matrix that the first version certified).
 * `false` is not a proof of anything.
 *
 * With this `c` the comparison `c > need` cannot fail once the factorisation succeeds: `||L||_F^2` is `tr(at)` up to rounding,
 * `tr(at) <= tr+(a) - n c` and `max|at_ii| <= tr+(a)`, so `c - need >= (gamma_{n+2} - u) tr+(a) > 0` up to second-order terms (sixth
 * audit round: `return true` in place of the comparison survives every test). The comparison is the step of the proof that ties
 * `c` to the backward error; it rejects a wrong shift (`c = rho` is caught through it), and no input can separate the mutations that
 * drop its terms (fourth audit round, mutation `M10`). The terms stay because `lambda_min(A) > 0` is proved from them. Replacing
 * `gamma_{n+2}` by `gamma_1` (in the shift and in `need`) accepted an indefinite arrow matrix of order 128 built in the eighth audit
 * round (126 small diagonal entries coupled to the last coordinate, whose updates round away before the final one); it is a test,
 * and so is `a = 2^-1000`, which only the pivot floor refuses.
 * @cold
 */
import { computeCholeskyInPlace } from './compute-cholesky-in-place.ts';
import { computeRoundoffGamma } from './compute-roundoff-gamma.ts';
import { copyFloat64View } from './copy-float64-view.ts';
import { detectFloat64BufferOverlap } from './detect-float64-buffer-overlap.ts';
import { readFloat64View } from './read-float64-view.ts';

export function verifyPositiveDefiniteShiftedCholesky(aIn: Float64Array, n: number, rho: number, work: Float64Array): boolean {
  const u = 1.1102230246251565e-16;
  if (!(rho >= 0.0) || !Number.isFinite(rho) || !Number.isInteger(n) || n < 1 || n > 4096) return false;
  if (readFloat64View(work, n * n) === null || detectFloat64BufferOverlap(aIn, n * n, work, n * n)) return false;
  // A private copy of the matrix; a genuine non-shared Float64Array scratch (ninth audit round: an Int32Array scratch quantised the
  // factor and certified an indefinite matrix).
  const a = copyFloat64View(aIn, n * n);
  if (a === null) return false;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = a[i * n + j]!;
      if (!Number.isFinite(v) || v !== a[j * n + i]!) return false;
    }
  }
  let trPos = 0.0;
  for (let i = 0; i < n; i++) trPos += Math.max(a[i * n + i]!, 0.0);
  const g = computeRoundoffGamma(n + 2);
  const c = (2.0 * g * trPos * (1.0 + computeRoundoffGamma(n)) + rho) * (1.0 + 8.0 * u);
  let dMax = 0.0;
  for (let e = 0; e < n * n; e++) work[e] = a[e]!;
  for (let i = 0; i < n; i++) {
    const v = a[i * n + i]! - c;
    work[i * n + i] = v;
    dMax = Math.max(dMax, Math.abs(v));
  }
  if (!computeCholeskyInPlace(work, n)) return false;
  let fro = 0.0;
  let minPivot = Infinity;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) fro += work[i * n + j]! * work[i * n + j]!;
    minPivot = Math.min(minPivot, work[i * n + i]!);
  }
  if (!(minPivot > 1e-140) || !Number.isFinite(fro)) return false;
  const need = (g * fro * (1.0 + computeRoundoffGamma(n * n)) + u * dMax + rho) * (1.0 + 8.0 * u);
  return c > need;
}
