/**
 * @file labs/mean-square-boundary/compute-certified-threshold-bracket.ts
 * @description The Phase 6 bracket `eta_low < eta* <= eta_up` of the mean-square threshold of Theorem G.6
 * (`computeCertifiedThresholdBracket`; plan `phase6-plan-certified-mean-square-bracket.md`, section 4), for the iteration with
 * mean curvature `diag(lt)` (`lt > 0`, the eigenbasis of the mean curvature plus tether) and curvature noise `noise.apply`. The proof
 * needs a noise map that is linear, self-adjoint and cone-preserving; only operators built by `createDenseDeviationNoiseOperator` and
 * `createRowSpaceFactorNoiseOperator` (frozen, members of `phase6NoiseOperatorBrand`) are accepted, anything else returns `null`
 * (thirteenth audit round: arbitrary callbacks, including any function through `Function.prototype.apply` and maps that are fixed but
 * not cone-preserving, gave wrong brackets, and rounds 11-12 had found callbacks that wrote their input or were read repeatedly). The
 * outputs are filled with NaN before each certified call, so an output left unwritten is refused; `apply` receives copies, compared
 * with `Object.is` on return (defence in depth: the library operators do not write their input). The Lanczos products are not
 * checked; they only choose the test matrices. Built-ins replaced after the modules load are outside the contract.
 * 1. `computeReorthogonalisedLanczosTopPair` on `applyEigenbasisThresholdOperator` from `L0^{1/2}(I)` (diagonal `sqrt(2 lt_a)`),
 *    at most `lanczosSteps` steps, relative residual `relTol`.
 * 2. Upper bound: `computeCertifiedRayleighUpperThreshold` at `z = L0^{-1/2}(y)`, `y` the Ritz vector; `etaUp` is the smaller of
 *    that bound and `eta_det = 2(1 + beta) / max lt` rounded upwards (`fl(2(1 + beta) / max lt)(1 + 8u)`: the addition, the division and the final product can each round, the
 *    product by 2 is exact, and `(1 - u)^3 (1 + 8u) > 1`; eleventh audit round, the text said two roundings; the
 *    deterministic part alone is unstable beyond `eta_det`). Before an external audit of 2026-10-29 the unrounded quotient was
 *    used, and on a noise-free input it rounded below the true ceiling and pulled `etaUp` under `eta*`.
 * 3. Lower bound: for each `eps` of `epsilons`, the witness `buildConeWitnessFromRitzVector(lt, n, y, eps)`, verified positive
 *    definite (`verifyPositiveDefiniteShiftedCholesky`, `rho = 0`), `V` of it once with its bound, then `bisectConeLowerThreshold`
 *    on `(best so far, etaUp)`; `etaLow` is the largest rate certified by any of them (each is a proof on its own), `bestEps` the
 *    `eps` that gave it. The ladder (default `1e-1, 1e-2, ..., 1e-9`) was added after stage 1: the certificate loses about `eps`
 *    when the critical eigenmatrix has small directions, and fails for `eps` below the Ritz residual, so no single `eps` fits all
 *    checkpoints (at one `|H|` checkpoint `eta_low / eta_up` went from 0.39 at `1e-3` to 0.999998 at `1e-9`; at one unconverged
 *    Gauss–Newton checkpoint from 0.70 at `1e-3` to 0.994 at `1e-2`). `etaLow` is `null` when no witness is certified positive
 *    definite or no rate passes.
 * `etaDet` is the rounded-up ceiling `fl(2(1 + beta) / max lt)(1 + 8u)`, an upper bound of `eta*` (fifth audit round: the field held
 * the raw quotient, below the exact ceiling at, for instance, `beta = 0.9`, `lt = 1.7`). `ritzEta = 1 / value` is the uncertified estimate. Returns `null` outside the certified domain of `certifyConeCollatzWielandtAtRate`
 * (`n` an integer in `[1, 4096]`, `beta = 0` or in `[2^-20, 1 - 2^-20]`, every `lt` in `[2^-60, 2^60]`), narrowed after the third
 * external audit round, which found underflow paths at the ends of the wider domain `[1e-100, 1e100]`. Also `null` when the noise
 * operator refuses or leaves the domain on the Rayleigh test matrix or a witness (the Lanczos products only generate candidates and are
 * not checked; tenth audit round) (an output or bound not finite or above `2^300`, a
 * negative bound, a non-finite Ritz value, or a refusal by `computeCertifiedRayleighUpperThreshold`): the fourth audit round found
 * that a NaN deviation, `B = 2^25` or `|V(z)| = 2^400` returned `etaUp = eta_det` and `witnessPositiveDefinite = true`, a bracket
 * for an operator outside the domain. `lt` on a `SharedArrayBuffer` is refused (seventh audit round), and so is a noise operator whose `n`
 * differs from `n` (eighth audit round).
 * @cold
 */
import { applyEigenbasisThresholdOperator } from './apply-eigenbasis-threshold-operator.ts';
import { bisectConeLowerThreshold } from './bisect-cone-lower-threshold.ts';
import { buildConeWitnessFromRitzVector } from './build-cone-witness-from-ritz-vector.ts';
import { computeCertifiedRayleighUpperThreshold } from './compute-certified-rayleigh-upper-threshold.ts';
import { computeReorthogonalisedLanczosTopPair } from './compute-reorthogonalised-lanczos-top-pair.ts';
import { copyFloat64View } from './copy-float64-view.ts';
import { phase6NoiseOperatorBrand } from './phase6-noise-operator-brand.ts';
import { verifyPositiveDefiniteShiftedCholesky } from './verify-positive-definite-shifted-cholesky.ts';

export function computeCertifiedThresholdBracket(
  ltIn: Float64Array,
  n: number,
  beta: number,
  noise: { readonly n?: number; apply(x: Float64Array, out: Float64Array, errOut: Float64Array | null): void },
  lanczosSteps = 60,
  relTol = 1e-10,
  epsilons: readonly number[] = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
  bisectionSteps = 50,
): {
  readonly etaLow: number | null;
  readonly etaUp: number;
  readonly etaDet: number;
  readonly ritzEta: number;
  readonly ritzResidual: number;
  readonly lanczosSteps: number;
  readonly witnessPositiveDefinite: boolean;
  readonly bisectionPasses: number;
  readonly bestEps: number | null;
} | null {
  const nn = n * n;
  if (!Number.isInteger(n) || n < 1 || n > 4096) return null;
  // A private copy of lt: shared memory can change during the call (seventh and eighth audit rounds: a worker rewrote a shared lt after
  // eta_det was formed), and a Proxy can change its values between reads (ninth audit round); copyFloat64View refuses both.
  const lt = copyFloat64View(ltIn, n);
  if (lt === null) return null;
  // A noise operator built for another dimension is refused (eighth audit round: a rank-1 operator passed with n = 2 gave a bracket
  // for a different iteration).
  // Only operators built by the two Phase 6 factories, which are frozen (thirteenth audit round; phase6NoiseOperatorBrand). Checked
  // before any property of noise is read; null, primitives and every other object are refused.
  if (!phase6NoiseOperatorBrand.has(noise)) return null;
  if (noise.n !== n) return null;
  const op = noise;
  // A primitive number only: a string or a coercible object passed the range test and then entered the arithmetic as another value
  // (tenth audit round: beta = '0.9' made 1 + beta the string '10.9').
  if (typeof beta !== 'number') return null;
  if (!Number.isInteger(lanczosSteps) || lanczosSteps < 1 || !Number.isInteger(bisectionSteps) || bisectionSteps < 0) return null;
  if (typeof relTol !== 'number' || !(relTol >= 0.0)) return null;
  const eps: number[] = [];
  for (let e = 0; e < epsilons.length; e++) {
    const v = epsilons[e];
    if (typeof v !== 'number' || !(v > 0.0) || !Number.isFinite(v)) return null;
    eps.push(v);
  }
  if (!(beta === 0.0 || (beta >= 9.5367431640625e-7 && beta <= 1.0 - 9.5367431640625e-7))) return null;
  let ltMax = 0.0;
  for (let a = 0; a < n; a++) {
    if (!(lt[a]! >= 8.673617379884035e-19 && lt[a]! <= 1.152921504606847e18)) return null;
    ltMax = Math.max(ltMax, lt[a]!);
  }
  const etaDet = (2.0 * (1.0 + beta)) / ltMax;
  const zBuf = new Float64Array(nn);
  const wBuf = new Float64Array(nn);
  const start = new Float64Array(nn);
  for (let a = 0; a < n; a++) start[a * n + a] = Math.sqrt(2.0 * lt[a]!);
  const ritz = computeReorthogonalisedLanczosTopPair(
    (x, out) => applyEigenbasisThresholdOperator(lt, n, beta, op, x, out, zBuf, wBuf),
    nn,
    start,
    lanczosSteps,
    relTol,
  );
  const z = new Float64Array(nn);
  for (let a = 0; a < n; a++) {
    for (let b = a; b < n; b++) {
      const v = (0.5 * (ritz.vector[a * n + b]! + ritz.vector[b * n + a]!)) / Math.sqrt(lt[a]! + lt[b]!);
      z[a * n + b] = v;
      z[b * n + a] = v;
    }
  }
  // A finite Ritz value that is not positive (fl V dominated by its own rounding, e.g. centring cancellation) is not a refusal: its
  // vector is only a poor test matrix, the Rayleigh bound then says nothing and eta_det rounded up is the upper end (seventh audit
  // round: this case returned null).
  if (!Number.isFinite(ritz.value)) return null;
  const upper = computeCertifiedRayleighUpperThreshold(lt, n, beta, op, z);
  if (upper === null) return null;
  const etaDetUp = etaDet * (1.0 + 8.0 * 1.1102230246251565e-16);
  const etaUp = upper.etaUp === null ? etaDetUp : Math.min(upper.etaUp, etaDetUp);
  const work = new Float64Array(nn);
  const w = new Float64Array(nn);
  const wErr = new Float64Array(nn);
  let witnessPositiveDefinite = false;
  let etaLow: number | null = null;
  let bestEps: number | null = null;
  let passes = 0;
  for (let e = 0; e < eps.length; e++) {
    const x = buildConeWitnessFromRitzVector(lt, n, ritz.vector, eps[e]!);
    if (!verifyPositiveDefiniteShiftedCholesky(x, n, 0.0, work)) continue;
    witnessPositiveDefinite = true;
    // apply receives a copy of the witness, and the certificate uses the witness itself; a changed copy is a refusal (eleventh audit
    // round: an apply that multiplied its input by 1000 after reading it made the bisection certify a rate above eta*).
    const xApply = new Float64Array(nn);
    for (let q = 0; q < nn; q++) xApply[q] = x[q]!;
    w.fill(NaN);
    wErr.fill(NaN);
    op.apply(xApply, w, wErr);
    for (let q = 0; q < nn; q++) if (!Object.is(xApply[q], x[q])) return null;
    for (let q = 0; q < nn; q++) if (!(Math.abs(w[q]!) <= 2.037035976334486e90) || !(wErr[q]! >= 0.0 && wErr[q]! <= 2.037035976334486e90)) return null;
    const res = bisectConeLowerThreshold(lt, n, beta, x, w, wErr, etaUp, bisectionSteps, etaLow ?? 0.0);
    passes += res.passes;
    if (res.etaLow !== null && (etaLow === null || res.etaLow > etaLow)) {
      etaLow = res.etaLow;
      bestEps = eps[e]!;
    }
  }
  // Two certified ends cannot cross; if they do, an input changed under the computation or a bound is wrong, and nothing is returned
  // (ninth audit round).
  if (etaLow !== null && !(etaLow < etaUp)) return null;
  return {
    etaLow,
    etaUp,
    etaDet: etaDetUp,
    ritzEta: 1.0 / ritz.value,
    ritzResidual: ritz.residual,
    lanczosSteps: ritz.steps,
    witnessPositiveDefinite,
    bisectionPasses: passes,
    bestEps,
  };
}
