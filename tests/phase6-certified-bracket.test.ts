/**
 * @file tests/phase6-certified-bracket.test.ts
 * @description Stage 0 of the Phase 6 plan (`labs/mean-square-boundary/phase6-plan-certified-mean-square-bracket.md`, section 6):
 * the certified bracket `computeCertifiedThresholdBracket` against the dense oracle `computeEigenbasisThresholdDenseReference` on
 * random non-commuting instances (three ensembles, `beta` in {0, 0.5, 0.9, 0.99}, `B` in {1, 16}); the oracle itself against the
 * spectral radius of the second-moment operator `T` on `S^{2n}` (power iteration, independent of Theorem G.6); the closed form in
 * dimension 1; `verifyPositiveDefiniteShiftedCholesky` on matrices of known spectrum; and input-level mutations (a noise operator
 * built with `B = 1` for a `B = 16` truth, a reference with the commutator sign flipped, a perturbation radius set below the true
 * perturbation) that the comparisons must detect. Stage 1 helpers: `regeneratePhase5CheckpointIterates` reproduces the stored
 * losses of a committed Phase 5b run bit for bit, and `unpackEigenbasisDeviationsDense` matches the packed deviations. Stage 2:
 * `createRowSpaceFactorNoiseOperator` equals the dense operator built from the deviations `M_i^T M_i - Hbar` within its own error
 * bound, and the bracket built on it encloses the dense threshold. After an external audit (2026-10-29): `c_beta` within `gamma_5`
 * of exact rational arithmetic up to `beta = 0.99999`; on a noise-free input whose `eta_det` rounds downwards, `eta_up` is still
 * at least the exact rational ceiling; a common matrix added to every deviation leaves the bracket unchanged (exact centring).
 * Zero-slack enclosure against exact rational arithmetic (after the second audit): in dimension 1 the threshold has the closed
 * form `2 lt / (lt^2 / (1 + beta) + V / (1 - beta))`, computed exactly from the binary64 inputs with `BigInt` fractions, and
 * `eta_low < eta* <= eta_up` is checked with no tolerance, on random inputs, `beta` up to `1 - 2^-25`, `eta*` within an ulp of
 * `eta_det`, and the rank-factor operator with `N dOut >> R` (the audit's counterexample to the first error count). After the
 * second audit round: the error bounds of both noise operators against the exact `V` on adversarial magnitudes (`2^-600` to `1e307`);
 * the enclosure at `n` in {2, 3} with the commutator, decided exactly through Theorem G.6 (`eta < eta*` iff `L0 - eta B_beta` is
 * positive definite, tested on the rational form); near-singular matrices on which a plain Cholesky succeeds while the exact
 * determinant is negative must be refused. After the third round: the stage 1 summary decides only on the complete committed list;
 * the Phase 6 pilot regeneration reproduces a committed record bit for bit; the adaptive Lanczos policy.
 * @cold
 */
import fs from 'node:fs';
import vm from 'node:vm';
import { describe, expect, it } from 'vitest';
import { computeCertifiedThresholdBracket } from '../labs/mean-square-boundary/compute-certified-threshold-bracket.ts';
import { computeMomentumCommutatorCoefficient } from '../labs/mean-square-boundary/compute-momentum-commutator-coefficient.ts';
import { computeEigenbasisThresholdDenseReference } from '../labs/mean-square-boundary/compute-eigenbasis-threshold-dense.reference.ts';
import { computeSymmetricEigendecompositionHouseholderQl } from '../labs/mean-square-boundary/compute-symmetric-eigendecomposition-householder-ql.ts';
import { bisectConeLowerThreshold } from '../labs/mean-square-boundary/bisect-cone-lower-threshold.ts';
import { certifyConeCollatzWielandtAtRate } from '../labs/mean-square-boundary/certify-cone-collatz-wielandt-at-rate.ts';
import { createDenseDeviationNoiseOperator } from '../labs/mean-square-boundary/create-dense-deviation-noise-operator.ts';
import { createRowSpaceFactorNoiseOperator } from '../labs/mean-square-boundary/create-row-space-factor-noise-operator.ts';
import { computeAdaptiveCertifiedThresholdBracket } from '../labs/mean-square-boundary/compute-adaptive-certified-threshold-bracket.ts';
import { regeneratePhase6PilotCheckpointIterates } from '../labs/mean-square-boundary/regenerate-phase6-pilot-checkpoint-iterates.ts';
import { summarizePhase6Stage1 } from '../labs/mean-square-boundary/summarize-phase6-stage1.ts';
import { createXorshift128Plus } from '../labs/mean-square-boundary/prng-xorshift128p.ts';
import { buildEigenbasisSampleHessianDeviations } from '../labs/mean-square-boundary/build-eigenbasis-sample-hessian-deviations.ts';
import { createPhase4SampleHessianProvider } from '../labs/mean-square-boundary/create-phase4-sample-hessian-provider.ts';
import { regeneratePhase5CheckpointIterates } from '../labs/mean-square-boundary/regenerate-phase5-checkpoint-iterates.ts';
import { unpackEigenbasisDeviationsDense } from '../labs/mean-square-boundary/unpack-eigenbasis-deviations-dense.ts';
import { verifyPositiveDefiniteShiftedCholesky } from '../labs/mean-square-boundary/verify-positive-definite-shifted-cholesky.ts';
import { decidePhase6RegenerationMismatch } from '../labs/mean-square-boundary/decide-phase6-regeneration-mismatch.ts';
import { runPhase6Stage1Run } from '../labs/mean-square-boundary/run-phase6-stage1.ts';
import { runPhase6Stage2Run } from '../labs/mean-square-boundary/run-phase6-stage2.ts';
import { computeCertifiedRayleighUpperThreshold } from '../labs/mean-square-boundary/compute-certified-rayleigh-upper-threshold.ts';
import { buildPhase6Stage2CheckpointRecord } from '../labs/mean-square-boundary/build-phase6-stage2-checkpoint-record.ts';
import { computeStudentGaussNewtonDualSpectrum } from '../labs/mean-square-boundary/compute-student-gauss-newton-dual-spectrum.ts';
import { computeDualEigenResidualMax } from '../labs/mean-square-boundary/compute-dual-eigen-residual-max.ts';
import { createPhase6StudentNetwork } from '../labs/mean-square-boundary/create-phase6-student-network.ts';
import { createScaleMlpDatasetAndOps } from '../labs/mean-square-boundary/scale-mlp-network.ts';

type Instance = { n: number; numSamples: number; beta: number; batchB: number; lt: Float64Array; dev: Float64Array; kind: string };
type Prng = ReturnType<typeof createXorshift128Plus>;

function recentre(dev: Float64Array, n: number, numSamples: number): void {
  const nn = n * n;
  for (let e = 0; e < nn; e++) {
    let s = 0;
    for (let i = 0; i < numSamples; i++) s += dev[i * nn + e]!;
    s /= numSamples;
    for (let i = 0; i < numSamples; i++) dev[i * nn + e] = dev[i * nn + e]! - s;
  }
  for (let i = 0; i < numSamples; i++) for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) {
    const v = 0.5 * (dev[i * nn + a * n + b]! + dev[i * nn + b * n + a]!);
    dev[i * nn + a * n + b] = v;
    dev[i * nn + b * n + a] = v;
  }
}

// Scales the deviations so that the threshold sits at a random fraction of eta_det (between about 0.05 and 0.9).
function scaleToFraction(inst: Instance, prng: Prng): void {
  const etaDet0 = computeEigenbasisThresholdDenseReference(inst.lt, inst.n, inst.beta, new Float64Array(inst.dev.length), inst.numSamples, inst.batchB);
  const target = (0.05 + 0.85 * prng.nextFloat01()) * etaDet0;
  let s = 1;
  for (let it = 0; it < 60; it++) {
    const scaled = Float64Array.from(inst.dev, (v) => v * s);
    const eta = computeEigenbasisThresholdDenseReference(inst.lt, inst.n, inst.beta, scaled, inst.numSamples, inst.batchB);
    if (Math.abs(eta / target - 1) < 0.05) break;
    s *= eta > target ? 1.6 : 0.7;
  }
  for (let e = 0; e < inst.dev.length; e++) inst.dev[e] = inst.dev[e]! * s;
}

function makeInstance(kind: string, beta: number, batchB: number, prng: Prng): Instance {
  const n = 2 + (prng.nextU32() % 5);
  const numSamples = 3 + (prng.nextU32() % 8);
  const nn = n * n;
  const lt = new Float64Array(n);
  const dev = new Float64Array(numSamples * nn);
  if (kind === 'general') {
    for (let a = 0; a < n; a++) lt[a] = 0.05 + 2 * prng.nextFloat01();
    for (let i = 0; i < numSamples; i++) for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) dev[i * nn + a * n + b] = prng.nextGaussian();
  } else if (kind === 'near-degenerate') {
    for (let a = 0; a < n; a++) lt[a] = 1 + Math.floor(a / 2) * 0.3 + (a % 2) * 1e-7 * prng.nextFloat01();
    for (let i = 0; i < numSamples; i++) for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) dev[i * nn + a * n + b] = prng.nextGaussian();
  } else {
    // Multiscale Gauss–Newton: rank-1 per-sample curvature g g^T, eigenvalues of the mean spread over about eight decades, tether 1e-3.
    const gs = new Float64Array(numSamples * n);
    const mean = new Float64Array(nn);
    for (let i = 0; i < numSamples; i++) for (let a = 0; a < n; a++) {
      gs[i * n + a] = prng.nextGaussian() * Math.exp((-9.2 * a) / Math.max(1, n - 1));
    }
    for (let i = 0; i < numSamples; i++) for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
      mean[a * n + b] = mean[a * n + b]! + (gs[i * n + a]! * gs[i * n + b]!) / numSamples;
    }
    const eig = computeSymmetricEigendecompositionHouseholderQl(mean, n);
    for (let k = 0; k < n; k++) lt[k] = Math.max(eig.values[k]!, 0) + 1e-3;
    const gr = new Float64Array(n);
    for (let i = 0; i < numSamples; i++) {
      for (let k = 0; k < n; k++) {
        let s = 0;
        for (let a = 0; a < n; a++) s += eig.qt[k * n + a]! * gs[i * n + a]!;
        gr[k] = s;
      }
      for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) dev[i * nn + a * n + b] = gr[a]! * gr[b]! - (a === b ? Math.max(eig.values[a]!, 0) : 0);
    }
  }
  recentre(dev, n, numSamples);
  const inst = { n, numSamples, beta, batchB, lt, dev, kind };
  if (kind !== 'multiscale-gn') scaleToFraction(inst, prng);
  return inst;
}

// Spectral radius of T(Q) = F Q F^T + eta^2 blkdiag(V(Q11), 0) on S^{2n} by power iteration from I (the mean-square recursion of
// the heavy-ball iteration with H_B = diag(lt) + mean of B deviations drawn with replacement).
function secondMomentRadius(inst: Instance, eta: number, iters: number): number {
  const { n, numSamples, beta, batchB, lt, dev } = inst;
  const m = 2 * n;
  const nn = n * n;
  let q = new Float64Array(m * m);
  for (let a = 0; a < m; a++) q[a * m + a] = 1;
  const f = new Float64Array(m * m);
  for (let a = 0; a < n; a++) {
    f[a * m + a] = 1 + beta - eta * lt[a]!;
    f[a * m + n + a] = -beta;
    f[(n + a) * m + a] = 1;
  }
  let logSum = 0;
  let counted = 0;
  const tmp = new Float64Array(m * m);
  for (let it = 0; it < iters; it++) {
    const out = new Float64Array(m * m);
    for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) {
      let s = 0;
      for (let k = 0; k < m; k++) s += f[a * m + k]! * q[k * m + b]!;
      tmp[a * m + b] = s;
    }
    for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) {
      let s = 0;
      for (let k = 0; k < m; k++) s += tmp[a * m + k]! * f[b * m + k]!;
      out[a * m + b] = s;
    }
    for (let i = 0; i < numSamples; i++) for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
      let s = 0;
      for (let k = 0; k < n; k++) for (let l = 0; l < n; l++) s += dev[i * nn + a * n + k]! * q[k * m + l]! * dev[i * nn + l * n + b]!;
      out[a * m + b] = out[a * m + b]! + (eta * eta * s) / (numSamples * batchB);
    }
    let tr = 0;
    for (let a = 0; a < m; a++) tr += out[a * m + a]!;
    for (let e = 0; e < m * m; e++) out[e] = out[e]! / tr;
    if (it >= iters / 2) { logSum += Math.log(tr); counted++; }
    q = out;
  }
  return Math.exp(logSum / counted);
}

// Exact rational arithmetic on binary64 values, used by the enclosure tests (fourth and fifth audit rounds: the random-instance tests
// compared against the dense oracle with 1e-9 slack, which a bound on the wrong side by less than that passed).
type Q = [bigint, bigint];
const gcd = (x: bigint, y: bigint): bigint => { let p = x < 0n ? -x : x; let q = y; while (q !== 0n) { const t = p % q; p = q; q = t; } return p; };
const red = (a: Q): Q => { const g = gcd(a[0], a[1]); return g > 1n ? [a[0] / g, a[1] / g] : a; };
const frac = (x: number): Q => {
  const dv = new DataView(new ArrayBuffer(8));
  dv.setFloat64(0, x);
  const bits = dv.getBigUint64(0);
  const sign = bits >> 63n ? -1n : 1n;
  const expo = Number((bits >> 52n) & 0x7ffn);
  let mant = bits & ((1n << 52n) - 1n);
  if (expo !== 0) mant |= 1n << 52n;
  const e = (expo === 0 ? 1 : expo) - 1075;
  return e >= 0 ? [sign * (mant << BigInt(e)), 1n] : red([sign * mant, 1n << BigInt(-e)]);
};
const add = (a: Q, b: Q): Q => red([a[0] * b[1] + b[0] * a[1], a[1] * b[1]]);
const sub = (a: Q, b: Q): Q => red([a[0] * b[1] - b[0] * a[1], a[1] * b[1]]);
const mul = (a: Q, b: Q): Q => red([a[0] * b[0], a[1] * b[1]]);
const div = (a: Q, b: Q): Q => red(b[0] < 0n ? [-a[0] * b[1], -a[1] * b[0]] : [a[0] * b[1], a[1] * b[0]]);
const absQ = (a: Q): Q => [a[0] < 0n ? -a[0] : a[0], a[1]];
const le = (a: Q, b: Q): boolean => a[0] * b[1] <= b[0] * a[1];
const zero: Q = [0n, 1n];
const matMul = (A: Q[], B: Q[], n: number): Q[] => {
  const C: Q[] = [];
  for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) { let s = zero; for (let k = 0; k < n; k++) s = add(s, mul(A[a * n + k]!, B[k * n + b]!)); C.push(s); }
  return C;
};
// Exact V(X) = (1 / (B N)) (sum_i D_i X D_i - S X S / N), D_i symmetrised.
const exactV = (dev: Float64Array, n: number, N: number, B: number, x: Float64Array): Q[] => {
  const nn = n * n;
  const X = Array.from(x, frac);
  let acc: Q[] = new Array<Q>(nn).fill(zero);
  const S: Q[] = new Array<Q>(nn).fill(zero);
  for (let i = 0; i < N; i++) {
    const D: Q[] = [];
    for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) D.push(mul([1n, 2n], add(frac(dev[i * nn + a * n + b]!), frac(dev[i * nn + b * n + a]!))));
    const DXD = matMul(matMul(D, X, n), D, n);
    acc = acc.map((v, k) => add(v, DXD[k]!));
    for (let k = 0; k < nn; k++) S[k] = add(S[k]!, D[k]!);
  }
  const SXS = matMul(matMul(S, X, n), S, n);
  return acc.map((v, k) => div(sub(v, div(SXS[k]!, [BigInt(N), 1n])), [BigInt(B * N), 1n]));
};
// Dyadic numbers m 2^e (every binary64 value is one, and so is every sum and product of them): exact without gcd reductions.
type Dy = { m: bigint; e: number };
const dy = (x: number): Dy => {
  const [num, den] = frac(x);
  if (num === 0n) return { m: 0n, e: 0 };
  let e = 0;
  let d = den;
  while (d > 1n) { d >>= 1n; e--; }
  return { m: num, e };
};
const dyAdd = (x: Dy, y: Dy): Dy => {
  if (x.m === 0n) return y;
  if (y.m === 0n) return x;
  const e = Math.min(x.e, y.e);
  return { m: (x.m << BigInt(x.e - e)) + (y.m << BigInt(y.e - e)), e };
};
const dyMul = (x: Dy, y: Dy): Dy => ({ m: x.m * y.m, e: x.e + y.e });
const dyNeg = (x: Dy): Dy => ({ m: -x.m, e: x.e });
const dyInt = (k: number): Dy => ({ m: BigInt(k), e: 0 });
// Sylvester's criterion by fraction-free (Bareiss) elimination: after step k the pivot is the (k + 1)-th leading principal minor.
const isPositiveDefinite = (M: bigint[], d: number): boolean => {
  const A = M.slice();
  let prev = 1n;
  for (let k = 0; k < d; k++) {
    const piv = A[k * d + k]!;
    if (piv <= 0n) return false;
    for (let i = k + 1; i < d; i++) for (let j = k + 1; j < d; j++) A[i * d + j] = (A[i * d + j]! * piv - A[i * d + k]! * A[k * d + j]!) / prev;
    prev = piv;
  }
  return true;
};
// The Gram matrix of K (L0 - eta B_beta) on the basis E_aa, E_ab + E_ba of S^n, with the positive factor
// K = (1 - beta)^2 (1 + beta) B N^2 that clears every non-dyadic denominator, as integers (common power of two removed):
// K B_beta(X)_ab = ((1 - beta)^2 B N^2 lt_a lt_b - beta B N^2 (lt_a - lt_b)^2) X_ab + (1 - beta)(1 + beta)(N sum_i D_i X D_i - S X S)_ab,
// D_i the symmetrised deviations and S their sum. Positive definite exactly when eta < eta* (for eta < eta_det).
const formAt = (lt: Float64Array, n: number, beta: number, dev: Float64Array, N: number, B: number, eta: number): bigint[] => {
  const nn = n * n;
  const pairs: Array<[number, number]> = [];
  for (let a = 0; a < n; a++) for (let b = a; b < n; b++) pairs.push([a, b]);
  const d = pairs.length;
  const be = dy(beta);
  const one = dyInt(1);
  const omb = dyAdd(one, dyNeg(be));
  const opb = dyAdd(one, be);
  // Formed in BigInt: B N^2 can exceed 2^53 inside the domain (eighth audit round: the binary64 product rounded it).
  const bn2: Dy = { m: BigInt(B) * BigInt(N) * BigInt(N), e: 0 };
  const kFac = dyMul(dyMul(dyMul(omb, omb), opb), bn2);
  const L = Array.from(lt, dy);
  const E = dy(eta);
  const half: Dy = { m: 1n, e: -1 };
  const D: Dy[] = [];
  const S: Dy[] = new Array<Dy>(nn).fill(dyInt(0));
  for (let i = 0; i < N; i++) for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
    const v = dyMul(half, dyAdd(dy(dev[i * nn + a * n + b]!), dy(dev[i * nn + b * n + a]!)));
    D.push(v);
    S[a * n + b] = dyAdd(S[a * n + b]!, v);
  }
  // (M X_q M)_ab for X_q = E_cd + E_dc (or E_cc).
  const sand = (M: (r: number, c: number) => Dy, c: number, dd: number, a: number, b: number): Dy =>
    c === dd ? dyMul(M(a, c), M(c, b)) : dyAdd(dyMul(M(a, c), M(dd, b)), dyMul(M(a, dd), M(c, b)));
  const vFac = dyMul(omb, opb);
  const out: Dy[] = new Array<Dy>(d * d).fill(dyInt(0));
  for (let q = 0; q < d; q++) {
    const [c, dd] = pairs[q]!;
    for (let p = 0; p < d; p++) {
      const [a, b] = pairs[p]!;
      let sumD = dyInt(0);
      for (let i = 0; i < N; i++) sumD = dyAdd(sumD, sand((r, k) => D[i * nn + r * n + k]!, c, dd, a, b));
      const vPart = dyAdd(dyMul(dyInt(N), sumD), dyNeg(sand((r, k) => S[r * n + k]!, c, dd, a, b)));
      let bTerm = dyMul(vFac, vPart);
      let l0 = dyInt(0);
      if (p === q) {
        const diff = dyAdd(L[a]!, dyNeg(L[b]!));
        const curv = dyAdd(dyMul(dyMul(dyMul(omb, omb), bn2), dyMul(L[a]!, L[b]!)), dyNeg(dyMul(dyMul(be, bn2), dyMul(diff, diff))));
        bTerm = dyAdd(bTerm, curv);
        l0 = dyMul(kFac, dyAdd(L[a]!, L[b]!));
      }
      const v = dyAdd(l0, dyNeg(dyMul(E, bTerm)));
      out[p * d + q] = a === b ? v : dyMul(dyInt(2), v);
    }
  }
  for (let p = 0; p < d; p++) for (let q = p + 1; q < d; q++) out[q * d + p] = out[p * d + q]!;
  let eMin = 0;
  for (const v of out) if (v.m !== 0n) eMin = Math.min(eMin, v.e);
  return out.map((v) => (v.m === 0n ? 0n : v.m << BigInt(v.e - eMin)));
};

// The enclosure judged exactly: eta < eta* iff L0 - eta B_beta > 0 on S^n (Theorem G.6), so the returned eta_low must make the
// rational form positive definite, and eta_up, when below the exact eta_det, must not. Returns whether the upper end was decided.
function expectExactEnclosure(lt: Float64Array, n: number, beta: number, dev: Float64Array, N: number, B: number, etaLow: number | null, etaUp: number, etaDet?: number): boolean {
  const d = (n * (n + 1)) / 2;
  if (etaLow !== null) expect(isPositiveDefinite(formAt(lt, n, beta, dev, N, B, etaLow), d)).toBe(true);
  let ltMax = 0;
  for (let a = 0; a < n; a++) ltMax = Math.max(ltMax, lt[a]!);
  const [en, ed] = frac(etaUp);
  const [bn, bd] = frac(beta);
  const [ln, ld] = frac(ltMax);
  // The returned eta_det field is the rounded-up ceiling, at least 2 (1 + beta) / max lt exactly (fifth audit round).
  if (etaDet !== undefined) { const [dn, dd] = frac(etaDet); expect(dn * bd * ln >= 2n * (bd + bn) * ld * dd).toBe(true); }
  if (en * bd * ln >= 2n * (bd + bn) * ld * ed) return false;
  expect(isPositiveDefinite(formAt(lt, n, beta, dev, N, B, etaUp), d)).toBe(false);
  return true;
}

const BETAS = [0, 0.5, 0.9, 0.99];
const KINDS = ['general', 'near-degenerate', 'multiscale-gn'];

describe('Phase 6 certified bracket (stage 0)', () => {
  it('the dense oracle agrees with the spectral radius of the second-moment operator', () => {
    const prng = createXorshift128Plus(0x16000001, 0x2545f491);
    for (const batchB of [1, 16]) {
      for (const beta of [0, 0.9]) {
        const inst = makeInstance('general', beta, batchB, prng);
        const eta = computeEigenbasisThresholdDenseReference(inst.lt, inst.n, beta, inst.dev, inst.numSamples, batchB);
        expect(secondMomentRadius(inst, 0.95 * eta, 6000)).toBeLessThan(1);
        expect(secondMomentRadius(inst, 1.05 * eta, 6000)).toBeGreaterThan(1);
      }
    }
  });

  it('dimension 1: the bracket encloses the closed form 2 lambda / (lambda^2 / (1 + beta) + nu^2 / (B (1 - beta)))', () => {
    for (const beta of BETAS) {
      const lt = Float64Array.of(1.7);
      const dev = Float64Array.of(0.9, -0.4, -0.5);
      const nu2 = (0.81 + 0.16 + 0.25) / 3;
      const exact = Math.min((2 * 1.7) / ((1.7 * 1.7) / (1 + beta) + nu2 / (16 * (1 - beta))), (2 * (1 + beta)) / 1.7);
      const br = computeCertifiedThresholdBracket(lt, 1, beta, createDenseDeviationNoiseOperator(dev, 1, 3, 16));
      expect(br).not.toBeNull();
      expect(br!.etaLow).not.toBeNull();
      expectExactEnclosure(lt, 1, beta, dev, 3, 16, br!.etaLow, br!.etaUp);
      expect(Math.abs(br!.etaUp / exact - 1)).toBeLessThan(1e-6);
      expect(br!.etaLow! / br!.etaUp).toBeGreaterThan(0.999);
    }
  });

  it('random non-commuting instances: eta_low <= eta* <= eta_up always, and the bracket is narrow', () => {
    const prng = createXorshift128Plus(0x16000002, 0x9e3779b9);
    let total = 0;
    let narrow = 0;
    let undefinedLow = 0;
    let worst = 1;
    for (const kind of KINDS) for (const beta of BETAS) for (const batchB of [1, 16]) for (let r = 0; r < 10; r++) {
      const inst = makeInstance(kind, beta, batchB, prng);
      const ref = computeEigenbasisThresholdDenseReference(inst.lt, inst.n, beta, inst.dev, inst.numSamples, batchB);
      const br = computeCertifiedThresholdBracket(inst.lt, inst.n, beta, createDenseDeviationNoiseOperator(inst.dev, inst.n, inst.numSamples, batchB));
      expect(br).not.toBeNull();
      total++;
      expectExactEnclosure(inst.lt, inst.n, beta, inst.dev, inst.numSamples, batchB, br!.etaLow, br!.etaUp, br!.etaDet);
      // The oracle is a diagnostic beside the exact test (it shares the c_beta formula and the eigensolver).
      expect(br!.etaUp).toBeGreaterThanOrEqual(ref * (1 - 1e-9));
      if (br!.etaLow === null) { undefinedLow++; continue; }
      const ratio = br!.etaLow / br!.etaUp;
      worst = Math.min(worst, ratio);
      if (ratio >= 0.95) narrow++;
    }
    expect(total).toBe(240);
    expect(narrow).toBeGreaterThanOrEqual(Math.ceil(0.95 * total));
    expect(undefinedLow).toBe(0);
    expect(worst).toBeGreaterThan(0.9);
  });

  it('an unconverged Lanczos (3 steps) still gives a valid bracket on both sides', () => {
    const prng = createXorshift128Plus(0x16000005, 0x85ebca6b);
    let unconverged = 0;
    for (const kind of KINDS) for (const beta of BETAS) for (let r = 0; r < 5; r++) {
      let inst = makeInstance(kind, beta, 16, prng);
      while (inst.n < 5) inst = makeInstance(kind, beta, 16, prng);
      const ref = computeEigenbasisThresholdDenseReference(inst.lt, inst.n, beta, inst.dev, inst.numSamples, 16);
      const br = computeCertifiedThresholdBracket(inst.lt, inst.n, beta, createDenseDeviationNoiseOperator(inst.dev, inst.n, inst.numSamples, 16), 3);
      expect(br!.lanczosSteps).toBe(3);
      if (br!.ritzResidual > 1e-6 * (1 / br!.ritzEta)) unconverged++;
      expectExactEnclosure(inst.lt, inst.n, beta, inst.dev, inst.numSamples, 16, br!.etaLow, br!.etaUp);
      expect(ref).toBeGreaterThan(0);
    }
    expect(unconverged).toBeGreaterThan(10);
  });

  it('verifyPositiveDefiniteShiftedCholesky: certifies known spectra and respects the perturbation radius', () => {
    const prng = createXorshift128Plus(0x16000003, 0x85ebca6b);
    const n = 8;
    for (const lamMin of [1e-6, -1e-3]) {
      const g = new Float64Array(n * n);
      for (let e = 0; e < n * n; e++) g[e] = prng.nextGaussian();
      const sym = new Float64Array(n * n);
      for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) sym[a * n + b] = g[a * n + b]! + g[b * n + a]!;
      const eig = computeSymmetricEigendecompositionHouseholderQl(sym, n);
      const a = new Float64Array(n * n);
      for (let k = 0; k < n; k++) {
        const lam = k === 0 ? lamMin : 1 + k;
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) a[i * n + j] = a[i * n + j]! + lam * eig.qt[k * n + i]! * eig.qt[k * n + j]!;
      }
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const v = 0.5 * (a[i * n + j]! + a[j * n + i]!); a[i * n + j] = v; a[j * n + i] = v; }
      const work = new Float64Array(n * n);
      if (lamMin > 0) {
        expect(verifyPositiveDefiniteShiftedCholesky(a, n, 1e-8, work)).toBe(true);
        // Mutation: the true matrix is a - 1e-5 I (negative definite direction), so a radius below 1e-5 must not be trusted.
        expect(verifyPositiveDefiniteShiftedCholesky(a, n, 1e-5, work)).toBe(false);
      } else {
        expect(verifyPositiveDefiniteShiftedCholesky(a, n, 0, work)).toBe(false);
      }
    }
  });

  it('mutations are detected: noise built with B = 1 for a B = 16 truth, and a reference with the commutator sign flipped', () => {
    const prng = createXorshift128Plus(0x16000004, 0x2545f491);
    let wrongB = 0;
    let flipped = 0;
    for (let r = 0; r < 20; r++) {
      const inst = makeInstance('general', 0.9, 16, prng);
      const ref = computeEigenbasisThresholdDenseReference(inst.lt, inst.n, 0.9, inst.dev, inst.numSamples, 16);
      const brWrong = computeCertifiedThresholdBracket(inst.lt, inst.n, 0.9, createDenseDeviationNoiseOperator(inst.dev, inst.n, inst.numSamples, 1));
      if (brWrong!.etaUp < ref * (1 - 1e-9)) wrongB++;
      const br = computeCertifiedThresholdBracket(inst.lt, inst.n, 0.9, createDenseDeviationNoiseOperator(inst.dev, inst.n, inst.numSamples, 16));
      const refFlipped = computeEigenbasisThresholdDenseReference(inst.lt, inst.n, 0.9, inst.dev, inst.numSamples, 16, -1);
      if (refFlipped > br!.etaUp * (1 + 1e-9) || (br!.etaLow !== null && refFlipped < br!.etaLow * (1 - 1e-9))) flipped++;
    }
    expect(wrongB).toBeGreaterThan(0);
    expect(flipped).toBeGreaterThan(0);
  });

  it('stage 1 helpers: regenerated Phase 5b losses are bit-identical, and the dense deviations match the packed ones', () => {
    const run = JSON.parse(fs.readFileSync('evidence/mean-square-boundary/phase5b-runs/eta0.3-mu0.001-seed1402.json', 'utf8')) as { checkpoints: Array<{ step: number; loss: number }> };
    const seen: number[] = [];
    let thetaAt600: Float64Array | null = null;
    regeneratePhase5CheckpointIterates(0.3, 0.001, 1402, 600, (t, theta, loss) => {
      const stored = run.checkpoints.find((c) => c.step === t)!;
      expect(loss).toBe(stored.loss);
      seen.push(t);
      if (t === 600) thetaAt600 = Float64Array.from(theta);
      return true;
    });
    expect(seen).toEqual([200, 400, 600]);
    // Mutation: a different seed is a different run.
    let differs = false;
    regeneratePhase5CheckpointIterates(0.3, 0.001, 1403, 200, (t, _theta, loss) => { differs = loss !== run.checkpoints.find((c) => c.step === t)!.loss; return true; });
    expect(differs).toBe(true);
    const dev = buildEigenbasisSampleHessianDeviations(createPhase4SampleHessianProvider({ kind: 'mlp33', theta: thetaAt600!, gaussNewton: true }));
    const dense = unpackEigenbasisDeviationsDense(dev);
    const n = dev.n;
    for (let i = 0; i < dev.numSamples; i += 7) for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
      expect(dense[i * n * n + a * n + b]).toBe(dev.packed[i * dev.nPair + dev.pairIndex[a * n + b]!]);
      expect(dense[i * n * n + a * n + b]).toBe(dense[i * n * n + b * n + a]);
    }
  });

  it('stage 2: the rank-factor noise operator equals the dense one within its bound, and its bracket encloses the dense threshold', () => {
    const prng = createXorshift128Plus(0x16000006, 0x2545f491);
    for (let rep = 0; rep < 6; rep++) {
      const r = 3 + (prng.nextU32() % 5);
      const numSamples = 4 + (prng.nextU32() % 6);
      const dOut = 1 + (prng.nextU32() % 3);
      const factors = new Float64Array(numSamples * dOut * r);
      // Factors on the grid 2^-4 (integers up to 64 over 16), so that every product and sum forming A_i = M_i^T M_i is exact and the exact
      // enclosure below judges the stored factors themselves (thirteenth audit round: rounded Gram matrices changed the exact object).
      for (let e = 0; e < factors.length; e++) factors[e] = Math.round(prng.nextGaussian() * (1 + 3 * prng.nextFloat01()) * 16) / 16;
      const rr = r * r;
      const a = new Float64Array(numSamples * rr);
      const hbar = new Float64Array(rr);
      for (let i = 0; i < numSamples; i++) for (let c = 0; c < dOut; c++) for (let j = 0; j < r; j++) for (let k = 0; k < r; k++) {
        const v = factors[i * dOut * r + c * r + j]! * factors[i * dOut * r + c * r + k]!;
        a[i * rr + j * r + k] = a[i * rr + j * r + k]! + v;
        hbar[j * r + k] = hbar[j * r + k]! + v / numSamples;
      }
      const dev = new Float64Array(numSamples * rr);
      for (let i = 0; i < numSamples; i++) for (let e = 0; e < rr; e++) dev[i * rr + e] = a[i * rr + e]! - hbar[e]!;
      const lambda = new Float64Array(r);
      for (let k = 0; k < r; k++) lambda[k] = hbar[k * r + k]!;
      const fac = createRowSpaceFactorNoiseOperator(factors, lambda, r, numSamples, dOut, 16);
      const den = createDenseDeviationNoiseOperator(dev, r, numSamples, 16);
      const x = new Float64Array(rr);
      for (let j = 0; j < r; j++) for (let k = j; k < r; k++) { const v = prng.nextGaussian(); x[j * r + k] = v; x[k * r + j] = v; }
      const o1 = new Float64Array(rr); const e1 = new Float64Array(rr); const o2 = new Float64Array(rr); const e2 = new Float64Array(rr);
      fac.apply(x, o1, e1);
      den.apply(x, o2, e2);
      for (let e = 0; e < rr; e++) {
        expect(Math.abs(o1[e]! - o2[e]!)).toBeLessThanOrEqual(e1[e]! + e2[e]! + 1e-300);
        expect(e1[e]!).toBeLessThan(1e-11 * (1 + Math.abs(o1[e]!)) * 1e3);
      }
      // meanResidual is ||fl(Hbar) - diag(lambda)||_F; with lambda the diagonal of Hbar it is the Frobenius norm of the off-diagonal
      // part (sixth audit round: the earlier assertion, > 0, held for any operator).
      let off2 = 0;
      for (let j = 0; j < r; j++) for (let k = 0; k < r; k++) if (j !== k) off2 += hbar[j * r + k]! * hbar[j * r + k]!;
      expect(Math.abs(fac.meanResidual - Math.sqrt(off2))).toBeLessThan(1e-9 * Math.sqrt(off2) + 1e-12);
      const lt = Float64Array.from(lambda, (v) => v + 1e-3);
      const ref = computeEigenbasisThresholdDenseReference(lt, r, 0.9, dev, numSamples, 16);
      const br = computeCertifiedThresholdBracket(lt, r, 0.9, fac, 30, 1e-12);
      expect(br!.etaLow).not.toBeNull();
      // The oracle centres the uncentred, exactly formed A_i itself.
      expectExactEnclosure(lt, r, 0.9, a, numSamples, 16, br!.etaLow, br!.etaUp);
      expect(br!.etaLow! / ref).toBeGreaterThan(0.9);
    }
  });

  it('audit fixes: c_beta against exact rationals, eta_det rounded upwards, and exact centring of the deviations', () => {
    const toFrac = (x: number): [bigint, bigint] => {
      const buf = new DataView(new ArrayBuffer(8));
      buf.setFloat64(0, x);
      const bits = buf.getBigUint64(0);
      const expo = Number((bits >> 52n) & 0x7ffn);
      let mant = bits & ((1n << 52n) - 1n);
      if (expo !== 0) mant |= 1n << 52n;
      const e = (expo === 0 ? 1 : expo) - 1075;
      return e >= 0 ? [mant << BigInt(e), 1n] : [mant, 1n << BigInt(-e)];
    };
    for (const beta of [0, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.3 + 1e-12]) {
      const [bn, bd] = toFrac(beta);
      // exact c = b / ((1 - b)^2 (1 + b)) with b = bn / bd: c = bn bd^2 / ((bd - bn)^2 (bd + bn))
      const exN = bn * bd * bd;
      const exD = (bd - bn) * (bd - bn) * (bd + bn);
      if (exN === 0n) { expect(computeMomentumCommutatorCoefficient(beta)).toBe(0); continue; }
      const [cn, cd] = toFrac(computeMomentumCommutatorCoefficient(beta));
      // |c - ex| / ex <= 5.0001 u  <=>  |cn exD - exN cd| * 2^53 <= 5.0001 exN cd
      const diff = cn * exD - exN * cd;
      const absDiff = diff < 0n ? -diff : diff;
      expect(Number(absDiff * (1n << 53n)) / Number(exN * cd)).toBeLessThanOrEqual(5.0001);
    }
    // The noise-free input of the audit: the plain quotient 2(1 + beta) / lt rounds below the exact ceiling.
    const beta = 0.03877554246009984;
    const lt1 = 2092.169360320777;
    const br = computeCertifiedThresholdBracket(Float64Array.of(lt1), 1, beta, createDenseDeviationNoiseOperator(new Float64Array(1), 1, 1, 1))!;
    const [bn, bd] = toFrac(beta);
    const [ln, ld] = toFrac(lt1);
    const [un, ud] = toFrac(br.etaUp);
    // eta_up >= 2 (1 + b) / lt  <=>  un ud^-1 >= 2 (bd + bn) ld / (bd ln)
    expect(un * bd * ln >= 2n * (bd + bn) * ld * ud).toBe(true);
    // Mutation guard: on this input the unrounded quotient is below the exact ceiling, so the old code would fail the line above.
    const [qn, qd] = toFrac((2 * (1 + beta)) / lt1);
    expect(qn * bd * ln < 2n * (bd + bn) * ld * qd).toBe(true);
    // Centring: adding the same symmetric matrix to every deviation must not change the certified iteration.
    const prng = createXorshift128Plus(0x16000007, 0x9e3779b9);
    for (let rep = 0; rep < 4; rep++) {
      const inst = makeInstance('general', 0.9, 16, prng);
      const n = inst.n;
      const shifted = Float64Array.from(inst.dev);
      for (let a = 0; a < n; a++) for (let b = a; b < n; b++) {
        const c = 0.5 * prng.nextGaussian();
        for (let i = 0; i < inst.numSamples; i++) { shifted[i * n * n + a * n + b] = shifted[i * n * n + a * n + b]! + c; if (a !== b) shifted[i * n * n + b * n + a] = shifted[i * n * n + b * n + a]! + c; }
      }
      const b0 = computeCertifiedThresholdBracket(inst.lt, n, 0.9, createDenseDeviationNoiseOperator(inst.dev, n, inst.numSamples, 16))!;
      const b1 = computeCertifiedThresholdBracket(inst.lt, n, 0.9, createDenseDeviationNoiseOperator(shifted, n, inst.numSamples, 16))!;
      expect(Math.abs(b1.etaUp / b0.etaUp - 1)).toBeLessThan(1e-9);
      expectExactEnclosure(inst.lt, n, 0.9, shifted, inst.numSamples, 16, b1.etaLow, b1.etaUp);
    }
  });

  it('zero-slack enclosure against exact rationals in dimension 1 (dense and rank-factor operators)', () => {
    type Q = [bigint, bigint];
    const frac = (x: number): Q => {
      const dv = new DataView(new ArrayBuffer(8));
      dv.setFloat64(0, x);
      const bits = dv.getBigUint64(0);
      const sign = bits >> 63n ? -1n : 1n;
      const expo = Number((bits >> 52n) & 0x7ffn);
      let mant = bits & ((1n << 52n) - 1n);
      if (expo !== 0) mant |= 1n << 52n;
      const e = (expo === 0 ? 1 : expo) - 1075;
      return e >= 0 ? [sign * (mant << BigInt(e)), 1n] : [sign * mant, 1n << BigInt(-e)];
    };
    const gcd = (x: bigint, y: bigint): bigint => { let p = x < 0n ? -x : x; let q = y; while (q !== 0n) { const t = p % q; p = q; q = t; } return p; };
    const red = (a: Q): Q => { const g = gcd(a[0], a[1]); return g > 1n ? [a[0] / g, a[1] / g] : a; };
    const add = (a: Q, b: Q): Q => red([a[0] * b[1] + b[0] * a[1], a[1] * b[1]]);
    const sub = (a: Q, b: Q): Q => red([a[0] * b[1] - b[0] * a[1], a[1] * b[1]]);
    const mul = (a: Q, b: Q): Q => red([a[0] * b[0], a[1] * b[1]]);
    const div = (a: Q, b: Q): Q => red(b[0] < 0n ? [-a[0] * b[1], -a[1] * b[0]] : [a[0] * b[1], a[1] * b[0]]);
    const lt = (a: Q, b: Q): boolean => a[0] * b[1] < b[0] * a[1];
    const le = (a: Q, b: Q): boolean => a[0] * b[1] <= b[0] * a[1];
    const one: Q = [1n, 1n];
    // eta* = 2 l / (l^2 / (1 + beta) + v / (1 - beta)) with v the exact centred noise (already divided by B).
    const exactEta = (l: number, beta: number, v: Q): Q => {
      const L = frac(l);
      const b = frac(beta);
      return div(mul([2n, 1n], L), add(div(mul(L, L), add(one, b)), div(v, sub(one, b))));
    };
    const check = (l: number, beta: number, v: Q, noise: { apply(x: Float64Array, out: Float64Array, errOut: Float64Array | null): void }): void => {
      const br = computeCertifiedThresholdBracket(Float64Array.of(l), 1, beta, noise)!;
      const ex = exactEta(l, beta, v);
      expect(br.etaLow).not.toBeNull();
      expect(lt(frac(br.etaLow!), ex)).toBe(true);
      expect(le(ex, frac(br.etaUp))).toBe(true);
    };
    const prng = createXorshift128Plus(0x16000008, 0x2545f491);
    const betas = [0, 0.5, 0.9, 0.99, 0.999, 1 - 2 ** -20];
    for (let rep = 0; rep < 60; rep++) {
      const beta = betas[rep % betas.length]!;
      const N = 2 + (prng.nextU32() % 7);
      const B = rep % 2 === 0 ? 1 : 16;
      const l = 0.01 + 5 * prng.nextFloat01();
      const scaleD = rep % 3 === 0 ? 1e-9 * l : l * (0.1 + 3 * prng.nextFloat01());
      const d = new Float64Array(N);
      for (let i = 0; i < N; i++) d[i] = scaleD * prng.nextGaussian();
      // exact centred variance of the symmetrised (1 x 1) deviations, divided by B
      let mean: Q = [0n, 1n];
      for (let i = 0; i < N; i++) mean = add(mean, div(frac(d[i]!), [BigInt(N), 1n]));
      let v: Q = [0n, 1n];
      for (let i = 0; i < N; i++) { const c = sub(frac(d[i]!), mean); v = add(v, mul(c, c)); }
      v = div(v, [BigInt(N * B), 1n]);
      check(l, beta, v, createDenseDeviationNoiseOperator(d, 1, N, B));
    }
    // Strong cancellation: deviations = a large common value plus small noise, so the centring subtraction loses many digits and
    // the error bound of V decides the enclosure (a mutation that zeroes that bound must fail here).
    for (let rep = 0; rep < 30; rep++) {
      const beta = betas[rep % betas.length]!;
      const N = 50 + (prng.nextU32() % 300);
      const B = rep % 2 === 0 ? 1 : 16;
      const big = 1e3 * (1 + prng.nextFloat01());
      const small = 1e-3 * (1 + prng.nextFloat01());
      const d = new Float64Array(N);
      for (let i = 0; i < N; i++) d[i] = big + small * prng.nextGaussian();
      let mean: Q = [0n, 1n];
      for (let i = 0; i < N; i++) mean = add(mean, div(frac(d[i]!), [BigInt(N), 1n]));
      let v: Q = [0n, 1n];
      for (let i = 0; i < N; i++) { const c = sub(frac(d[i]!), mean); v = add(v, mul(c, c)); }
      v = div(v, [BigInt(N * B), 1n]);
      let mf = 0;
      for (let i = 0; i < N; i++) mf += (d[i]! - big) / N;
      let vf = 0;
      for (let i = 0; i < N; i++) vf += (d[i]! - big - mf) * (d[i]! - big - mf) / (N * B);
      const l = Math.sqrt(vf) * (0.5 + prng.nextFloat01());
      check(l, beta, v, createDenseDeviationNoiseOperator(d, 1, N, B));
    }
    // Rank-factor operator, R = 1, N dOut = 800 (the audit's counterexample to the first error count) and random factors.
    const u = 2 ** -53;
    for (const mult of [1.01, 0.98]) {
      const N = 40;
      const dOut = 20;
      const factors = new Float64Array(N * dOut);
      factors[0] = 1;
      const mSmall = Math.pow((mult * u) / dOut, 0.25);
      for (let i = 1; i < N; i++) for (let c = 0; c < dOut; c++) factors[i * dOut + c] = mSmall;
      const aI: Q[] = [];
      for (let i = 0; i < N; i++) { let a: Q = [0n, 1n]; for (let c = 0; c < dOut; c++) { const m = frac(factors[i * dOut + c]!); a = add(a, mul(m, m)); } aI.push(a); }
      let hbar: Q = [0n, 1n];
      for (const a of aI) hbar = add(hbar, div(a, [BigInt(N), 1n]));
      let v: Q = [0n, 1n];
      for (const a of aI) { const c = sub(a, hbar); v = add(v, mul(c, c)); }
      v = div(v, [BigInt(N), 1n]);
      check(0.5, 0, v, createRowSpaceFactorNoiseOperator(factors, Float64Array.of(0.5), 1, N, dOut, 1));
    }
    for (let rep = 0; rep < 20; rep++) {
      const N = 8 + (prng.nextU32() % 40);
      const dOut = 1 + (prng.nextU32() % 20);
      const B = rep % 2 === 0 ? 1 : 16;
      const beta = betas[rep % betas.length]!;
      const factors = new Float64Array(N * dOut);
      for (let e = 0; e < factors.length; e++) factors[e] = prng.nextGaussian() * (rep % 4 === 0 ? 1e-4 : 1);
      const aI: Q[] = [];
      for (let i = 0; i < N; i++) { let a: Q = [0n, 1n]; for (let c = 0; c < dOut; c++) { const m = frac(factors[i * dOut + c]!); a = add(a, mul(m, m)); } aI.push(a); }
      let hbar: Q = [0n, 1n];
      for (const a of aI) hbar = add(hbar, div(a, [BigInt(N), 1n]));
      let v: Q = [0n, 1n];
      for (const a of aI) { const c = sub(a, hbar); v = add(v, mul(c, c)); }
      v = div(v, [BigInt(N * B), 1n]);
      const l = 0.05 + 4 * prng.nextFloat01();
      check(l, beta, v, createRowSpaceFactorNoiseOperator(factors, Float64Array.of(l), 1, N, dOut, B));
    }
  });

  it('round 2: exact error bounds on adversarial magnitudes, exact enclosure at n = 2, 3, and refused near-singular matrices', () => {
    // 'ok' when every entry is finite and within its bound; 'refused' only when every entry is NaN with an infinite bound; otherwise
    // 'violated' (a refusal is never counted as a checked bound; third audit round).
    const checkBound = (op: { apply(x: Float64Array, out: Float64Array, errOut: Float64Array | null): void }, exact: Q[], x: Float64Array): 'ok' | 'refused' | 'violated' => {
      const nn = exact.length;
      const out = new Float64Array(nn);
      const err = new Float64Array(nn);
      op.apply(x, out, err);
      let nonFinite = 0;
      for (let k = 0; k < nn; k++) if (Number.isNaN(out[k]!) && err[k] === Infinity) nonFinite++;
      if (nonFinite === nn) return 'refused';
      for (let k = 0; k < nn; k++) {
        if (!Number.isFinite(out[k]!) || !Number.isFinite(err[k]!)) return 'violated';
        if (!le(absQ(sub(frac(out[k]!), exact[k]!)), frac(err[k]!))) return 'violated';
      }
      return 'ok';
    };
    // The auditors' counterexamples (round 2) and random wide-exponent instances.
    const p2 = (k: number): number => { let v = 1; if (k >= 0) for (let i = 0; i < k; i++) v *= 2; else for (let i = 0; i < -k; i++) v /= 2; return v; };
    const cross = Float64Array.of(p2(-600), 0, 0, 0, 0, p2(400), p2(400), 0);
    expect(checkBound(createDenseDeviationNoiseOperator(cross, 2, 2, 1), exactV(cross, 2, 2, 1, Float64Array.of(p2(-600), 0, 0, 0)), Float64Array.of(p2(-600), 0, 0, 0))).toBe('ok');
    // round 2: a product flushed in one triangle and not in the other (magnitudes moved into the domain [2^-1000, 2^1000])
    const d4 = new Float64Array(2 * 16);
    d4[1] = p2(-537); d4[4] = p2(-537); d4[11] = p2(990); d4[14] = p2(990);
    for (let k = 0; k < 16; k++) d4[16 + k] = -d4[k]!;
    const z4 = new Float64Array(16);
    z4[0] = 1; z4[3] = -1; z4[12] = -1; z4[6] = p2(-537); z4[9] = p2(-537);
    expect(checkBound(createDenseDeviationNoiseOperator(d4, 4, 2, 1), exactV(d4, 4, 2, 1, z4), z4)).toBe('ok');
    // round 3: exact V = 2^800 I is finite and must not be refused; round 3: D = 2^-500, x = 2^900
    const big = Float64Array.of(p2(600), 0, 0, -p2(600), -p2(600), 0, 0, p2(600));
    const xb = Float64Array.of(p2(-400), 0, 0, p2(-400));
    expect(checkBound(createDenseDeviationNoiseOperator(big, 2, 2, 1), exactV(big, 2, 2, 1, xb), xb)).toBe('ok');
    const small = Float64Array.of(p2(-500), p2(-501), p2(-501), 0, -p2(-500), -p2(-501), -p2(-501), 0);
    const xs = Float64Array.of(p2(900), p2(899), p2(899), p2(900));
    expect(checkBound(createDenseDeviationNoiseOperator(small, 2, 2, 1), exactV(small, 2, 2, 1, xs), xs)).toBe('ok');
    const prng = createXorshift128Plus(0x16000009, 0x85ebca6b);
    for (let rep = 0; rep < 40; rep++) {
      const n = 2 + (rep % 2);
      const N = 2 + (prng.nextU32() % 3);
      const dev = new Float64Array(N * n * n);
      for (let e = 0; e < dev.length; e++) dev[e] = prng.nextFloat01() < 0.3 ? 0 : prng.nextGaussian() * p2((prng.nextU32() % 600) - 300);
      dev[0] = p2((prng.nextU32() % 600) - 300);
      const x = new Float64Array(n * n);
      for (let a = 0; a < n; a++) for (let b = a; b < n; b++) { const v = prng.nextGaussian() * p2((prng.nextU32() % 600) - 300); x[a * n + b] = v; x[b * n + a] = v; }
      expect(checkBound(createDenseDeviationNoiseOperator(dev, n, N, 1 + (rep % 2) * 15), exactV(dev, n, N, 1 + (rep % 2) * 15, x), x)).toBe('ok');
    }
    // Rank-factor operator: the round-2 counterexample (R = 3) and random wide-exponent factors, against V of A_i = M_i^T M_i.
    const rowExact = (f: Float64Array, R: number, N: number, dOut: number, B: number, x: Float64Array): Q[] => {
      const dev = new Float64Array(N * R * R);
      const devQ: Q[][] = [];
      for (let i = 0; i < N; i++) {
        const A: Q[] = [];
        for (let a = 0; a < R; a++) for (let b = 0; b < R; b++) { let s = zero; for (let c = 0; c < dOut; c++) s = add(s, mul(frac(f[i * dOut * R + c * R + a]!), frac(f[i * dOut * R + c * R + b]!))); A.push(s); }
        devQ.push(A);
      }
      const X = Array.from(x, frac);
      let acc: Q[] = new Array<Q>(R * R).fill(zero);
      const S: Q[] = new Array<Q>(R * R).fill(zero);
      for (const A of devQ) { const AXA = matMul(matMul(A, X, R), A, R); acc = acc.map((v, k) => add(v, AXA[k]!)); for (let k = 0; k < R * R; k++) S[k] = add(S[k]!, A[k]!); }
      const SXS = matMul(matMul(S, X, R), S, R);
      void dev;
      return acc.map((v, k) => div(sub(v, div(SXS[k]!, [BigInt(N), 1n])), [BigInt(B * N), 1n]));
    };
    const m3 = Float64Array.of(1.5e-162, 0, 1e151, 1.5e-162, 1e151, 0, 1.5e-162, 0, -1e151, 1.5e-162, -1e151, 0);
    const x3 = Float64Array.of(1, 0, 0, 0, 1e-300, 0, 0, 0, 1e-300);
    expect(checkBound(createRowSpaceFactorNoiseOperator(m3, Float64Array.of(1, 1, 1), 3, 2, 2, 1), rowExact(m3, 3, 2, 2, 1, x3), x3)).toBe('ok');
    // round 3: rank-1 square, exact V = 0, must not be refused
    const sq = Float64Array.of(p2(400), 0, 0, p2(400));
    const xq = Float64Array.of(p2(-700), 0, 0, p2(-700));
    expect(checkBound(createRowSpaceFactorNoiseOperator(sq, Float64Array.of(1, 1), 2, 1, 2, 1), rowExact(sq, 2, 1, 2, 1, xq), xq)).toBe('ok');
    for (let rep = 0; rep < 30; rep++) {
      const R = 2 + (rep % 2);
      const N = 2 + (prng.nextU32() % 3);
      const dOut = 1 + (prng.nextU32() % 3);
      const f = new Float64Array(N * dOut * R);
      for (let e = 0; e < f.length; e++) f[e] = prng.nextFloat01() < 0.3 ? 0 : prng.nextGaussian() * p2((prng.nextU32() % 300) - 150);
      f[0] = p2((prng.nextU32() % 300) - 150);
      const x = new Float64Array(R * R);
      for (let a = 0; a < R; a++) for (let b = a; b < R; b++) { const v = prng.nextGaussian() * p2((prng.nextU32() % 600) - 300); x[a * R + b] = v; x[b * R + a] = v; }
      const lam = new Float64Array(R).fill(1);
      expect(checkBound(createRowSpaceFactorNoiseOperator(f, lam, R, N, dOut, 1), rowExact(f, R, N, dOut, 1, x), x)).toBe('ok');
    }
    // Exact enclosure at n = 2, 3 with the commutator: eta < eta* iff L0 - eta B_beta > 0 on S^n (Theorem G.6), on the rational form
    // in the basis E_aa, E_ab + E_ba.
    let decided = 0;
    for (let rep = 0; rep < 24; rep++) {
      const n = 2 + (rep % 2);
      const beta = [0, 0.5, 0.9, 0.99, 0.999, 1 - p2(-20)][rep % 6]!;
      const N = 2 + (prng.nextU32() % 3);
      const B = rep % 4 < 2 ? 1 : 16;
      const lt = new Float64Array(n);
      for (let a = 0; a < n; a++) lt[a] = 0.05 + 3 * prng.nextFloat01();
      const dev = new Float64Array(N * n * n);
      for (let e = 0; e < dev.length; e++) dev[e] = prng.nextGaussian() * 0.7;
      const br = computeCertifiedThresholdBracket(lt, n, beta, createDenseDeviationNoiseOperator(dev, n, N, B))!;
      expect(br.etaLow).not.toBeNull();
      const d = (n * (n + 1)) / 2;
      expect(isPositiveDefinite(formAt(lt, n, beta, dev, N, B, br.etaLow!), d)).toBe(true);
      let ltMax = 0;
      for (let a = 0; a < n; a++) ltMax = Math.max(ltMax, lt[a]!);
      const [en, ed] = frac(br.etaUp);
      const [bn, bd] = frac(beta);
      const [ln, ld] = frac(ltMax);
      const atOrBeyondDet = en * bd * ln >= 2n * (bd + bn) * ld * ed;
      if (!atOrBeyondDet) { expect(isPositiveDefinite(formAt(lt, n, beta, dev, N, B, br.etaUp), d)).toBe(false); decided++; }
    }
    expect(decided).toBeGreaterThan(12);
    // The certificate called on its own. Round 2: lt = 1e-100 with a witness 1e-230 I certified a rate 1.6e6 eta*; that lt is now
    // outside the domain and refused. Inside the domain (lt = 2^-60, witness 2^-280 I, rates up to the deterministic ceiling) every
    // rate it certifies must satisfy the exact positive-definiteness test; an indefinite witness must be refused (round 3).
    {
      expect(certifyConeCollatzWielandtAtRate(Float64Array.of(1e-100, 1e-100), 2, 0, Float64Array.of(1, 0, 0, 1), new Float64Array(4), new Float64Array(4), 1, new Float64Array(4), new Float64Array(4))).toBe(false);
      const lt = Float64Array.of(p2(-60), p2(-60) * 1.5);
      const dev = Float64Array.of(p2(-70), p2(-70), p2(-70), 0, -p2(-70), -p2(-70), -p2(-70), 0);
      const op = createDenseDeviationNoiseOperator(dev, 2, 2, 1);
      const x = Float64Array.of(p2(-280), 0, 0, p2(-280));
      const w = new Float64Array(4);
      const wErr = new Float64Array(4);
      op.apply(x, w, wErr);
      const res = bisectConeLowerThreshold(lt, 2, 0, x, w, wErr, (2 / (1.5 * p2(-60))) * 0.999999, 60);
      expect(res.etaLow).not.toBeNull();
      expect(isPositiveDefinite(formAt(lt, 2, 0, dev, 2, 1, res.etaLow!), 3)).toBe(true);
      const xNeg = Float64Array.of(-1, 0, 0, 1);
      const wn = new Float64Array(4);
      const en = new Float64Array(4);
      op.apply(xNeg, wn, en);
      expect(bisectConeLowerThreshold(lt, 2, 0, xNeg, wn, en, 1e18, 40).etaLow).toBeNull();
      // round 3: n = 1, lt = 1, beta = 0, an indefinite witness x = -1 with w = V(x) = -4 (deviations +-2); eta* = 0.4, and
      // without the x > 0 check the test passed at eta = 1.5.
      expect(certifyConeCollatzWielandtAtRate(Float64Array.of(1), 1, 0, Float64Array.of(-1), Float64Array.of(-4), Float64Array.of(0), 1.5, new Float64Array(1), new Float64Array(1))).toBe(false);
    }
    // Near-singular matrices: a plain Cholesky succeeds but the exact determinant is negative; the verifier must refuse them.
    let found = 0;
    for (let rep = 0; rep < 4000 && found < 20; rep++) {
      const x = 0.5 + prng.nextFloat01();
      const y = prng.nextGaussian();
      const z = (y * y) / x;
      const a = Float64Array.of(x, y, y, z);
      const det = sub(mul(frac(x), frac(z)), mul(frac(y), frac(y)));
      const plain = Float64Array.from(a);
      const plainOk = (() => { const l00 = Math.sqrt(plain[0]!); const l10 = plain[2]! * (1 / l00); const dd = plain[3]! - l10 * l10; return dd > 0; })();
      if (det[0] < 0n && plainOk) {
        found++;
        expect(verifyPositiveDefiniteShiftedCholesky(a, 2, 0, new Float64Array(4))).toBe(false);
      }
    }
    expect(found).toBeGreaterThan(5);
    // A non-symmetric stored matrix within rho = 1.74 of the indefinite 2.9 I - 1 1^T (round-1 counterexample): refused.
    expect(verifyPositiveDefiniteShiftedCholesky(Float64Array.of(1.9, -2, -2, 0, 1.9, -2, 0, 0, 1.9), 3, 1.74, new Float64Array(9))).toBe(false);
  });

  it('round 3: summary needs the complete list, pilot regeneration is bit-identical, adaptive policy', () => {
    const obj = (ok: boolean) => ({ etaLow: ok ? 0.99 : null, etaUp: 1.0, etaStarStored: 1.0, lowOverUp: ok ? 0.99 : null, upOverStored: 1.0, valid: ok, validDense: ok, lanczosRerun: false, witnessPositiveDefinite: true });
    const rec = (run: string, steps: number[], ok = true) => ({ run, eta: 0.1, muTrain: 0.001, mismatchAt: null, storedCheckpoints: steps.length, checkpoints: steps.map((step) => ({ step, gaussNewton: { ...obj(ok), removedModesAgree: true }, absH: { ...obj(true), nullModesAgree: true } })) });
    const expected = [{ run: 'a', steps: [200, 400] }, { run: 'b', steps: [200, 400] }];
    const full = summarizePhase6Stage1([rec('a', [200, 400]), rec('b', [200, 400])], expected) as { decision: { validity: string; adoptionGaussNewton: string } };
    expect(full.decision.validity).toBe('no violation');
    expect(full.decision.adoptionGaussNewton).toBe('adopted');
    const partial = summarizePhase6Stage1([rec('a', [200, 400])], expected) as { decision: { validity: string; adoptionGaussNewton: string } };
    expect(partial.decision.validity).toBe('incomplete: no decision');
    expect(partial.decision.adoptionGaussNewton).toBe('not adopted');
    const shortRun = summarizePhase6Stage1([rec('a', [200, 400]), rec('b', [200])], expected) as { decision: { validity: string } };
    expect(shortRun.decision.validity).toBe('incomplete: no decision');
    const nullLow = summarizePhase6Stage1([rec('a', [200, 400]), rec('b', [200, 400], false)], expected) as { decision: { validity: string } };
    expect(nullLow.decision.validity).toBe('violation: method rejected');
    // Pilot regeneration against a committed record.
    const stored = JSON.parse(fs.readFileSync('evidence/mean-square-boundary/phase6-pilot/p3610-tanh-eta0.05-mu0.001-seed0.json', 'utf8')) as { checkpoints: Array<{ step: number; dataLoss: number; thetaSqFirstLayer: number; thetaSqOutputLayer: number }> };
    let checked = 0;
    regeneratePhase6PilotCheckpointIterates(3610, 'tanh', 0.05, 0.001, 0, 400, (t, theta, net, normFirst, normOut) => {
      const st = stored.checkpoints.find((c) => c.step === t)!;
      expect(normFirst).toBe(st.thetaSqFirstLayer);
      expect(normOut).toBe(st.thetaSqOutputLayer);
      expect(net.evalLoss(theta)).toBe(st.dataLoss);
      checked++;
      return true;
    });
    expect(checked).toBe(2);
    // Adaptive policy: the first length is 30, a converged small problem stops there, and the bracket is that of the direct call.
    const prng = createXorshift128Plus(0x1600000a, 0x2545f491);
    const inst = makeInstance('general', 0.9, 16, prng);
    const noise = createDenseDeviationNoiseOperator(inst.dev, inst.n, inst.numSamples, 16);
    const ad = computeAdaptiveCertifiedThresholdBracket(inst.lt, inst.n, 0.9, noise);
    expect(ad.lengthsTried[0]).toBe(30);
    const direct = computeCertifiedThresholdBracket(inst.lt, inst.n, 0.9, noise, 30, 1e-12)!;
    expect(ad.lengthsTried).toEqual([30]);
    expect(ad.bracket!.etaUp).toBe(direct.etaUp);
  });

  it('round 4: refusals at every domain boundary, of invalid noise, and of aliased or short buffers', () => {
    const p2 = (e: number) => 2 ** e;
    const noise1 = createDenseDeviationNoiseOperator(Float64Array.of(0.5, -0.5), 1, 2, 1);
    // The bracket: NaN deviations, B above 2^20, and |V(z)| = 2^400 used to return eta_up = eta_det and a positive definite witness.
    const lt2 = Float64Array.of(1, 2);
    const devNaN = new Float64Array(8).fill(0.1);
    devNaN[3] = NaN;
    expect(computeCertifiedThresholdBracket(lt2, 2, 0.5, createDenseDeviationNoiseOperator(devNaN, 2, 2, 1))).toBeNull();
    const devOk = Float64Array.of(0.1, 0.2, 0.2, -0.1, -0.1, -0.2, -0.2, 0.1);
    expect(computeCertifiedThresholdBracket(lt2, 2, 0.5, createDenseDeviationNoiseOperator(devOk, 2, 2, 1))).not.toBeNull();
    expect(computeCertifiedThresholdBracket(lt2, 2, 0.5, createDenseDeviationNoiseOperator(devOk, 2, 2, p2(25)))).toBeNull();
    expect(computeCertifiedThresholdBracket(lt2, 2, 0.5, createDenseDeviationNoiseOperator(Float64Array.from(devOk, (v) => v * p2(200)), 2, 2, 1))).toBeNull();
    // A caller-made noise object (here one that leaves the domain on the witness) is refused before any call: only library operators
    // are accepted (thirteenth audit round).
    const base = createDenseDeviationNoiseOperator(devOk, 2, 2, 1);
    let boundCalls = 0;
    const leaving = { n: 2, apply(x: Float64Array, out: Float64Array, errOut: Float64Array | null): void {
      base.apply(x, out, errOut);
      if (errOut !== null && ++boundCalls > 1) out[0] = p2(301);
    } };
    expect(computeCertifiedThresholdBracket(lt2, 2, 0.5, leaving)).toBeNull();
    expect(boundCalls).toBe(0);
    // N and B at 2^20 are inside, 2^20 + 1 outside (n = 1 keeps the buffer at 8 MB).
    const big = new Float64Array(p2(20) + 1);
    big[0] = 1;
    big[1] = -1;
    expect(computeCertifiedThresholdBracket(Float64Array.of(1), 1, 0, createDenseDeviationNoiseOperator(big, 1, p2(20), 1))).not.toBeNull();
    expect(computeCertifiedThresholdBracket(Float64Array.of(1), 1, 0, createDenseDeviationNoiseOperator(big, 1, p2(20) + 1, 1))).toBeNull();
    expect(computeCertifiedThresholdBracket(Float64Array.of(1), 1, 0, createDenseDeviationNoiseOperator(Float64Array.of(1, -1), 1, 2, p2(20)))).not.toBeNull();
    expect(computeCertifiedThresholdBracket(Float64Array.of(1), 1, 0, createDenseDeviationNoiseOperator(Float64Array.of(1, -1), 1, 2, p2(20) + 1))).toBeNull();
    const fac = (dOut: number) => { const f = new Float64Array(2 * dOut); f[0] = 1; return f; };
    expect(computeCertifiedThresholdBracket(Float64Array.of(1), 1, 0, createRowSpaceFactorNoiseOperator(fac(1024), Float64Array.of(0.5), 1, 2, 1024, 1))).not.toBeNull();
    expect(computeCertifiedThresholdBracket(Float64Array.of(1), 1, 0, createRowSpaceFactorNoiseOperator(fac(1025), Float64Array.of(0.5), 1, 2, 1025, 1))).toBeNull();
    // beta, lt and n at and beyond the boundaries, in the bracket and in the Rayleigh bound.
    const z1 = Float64Array.of(1);
    for (const beta of [0, p2(-20), 1 - p2(-20), 0.9]) {
      expect(computeCertifiedThresholdBracket(Float64Array.of(1), 1, beta, noise1)).not.toBeNull();
      expect(computeCertifiedRayleighUpperThreshold(Float64Array.of(1), 1, beta, noise1, z1)).not.toBeNull();
    }
    for (const beta of [p2(-21), 1 - p2(-21), -0.1, 1, NaN]) {
      expect(computeCertifiedThresholdBracket(Float64Array.of(1), 1, beta, noise1)).toBeNull();
      expect(computeCertifiedRayleighUpperThreshold(Float64Array.of(1), 1, beta, noise1, z1)).toBeNull();
    }
    for (const l of [p2(-60), p2(60)]) expect(computeCertifiedThresholdBracket(Float64Array.of(l), 1, 0, noise1)).not.toBeNull();
    for (const l of [p2(-61), p2(61), 0, -1, NaN]) {
      expect(computeCertifiedThresholdBracket(Float64Array.of(l), 1, 0, noise1)).toBeNull();
      expect(computeCertifiedRayleighUpperThreshold(Float64Array.of(l), 1, 0, noise1, z1)).toBeNull();
    }
    expect(computeCertifiedThresholdBracket(new Float64Array(4097).fill(1), 4097, 0, noise1)).toBeNull();
    // An uninformative test matrix is not a refusal: at beta = 0.9, lt = (1, 100), V = 0 and z = E_01 + E_10, the B form is
    // 2 (100 / 1.9 - c_beta 99^2) < 0, so etaUp is null while the result is not; at z = I it is informative.
    const zero2 = createDenseDeviationNoiseOperator(new Float64Array(8), 2, 2, 1);
    expect(computeCertifiedRayleighUpperThreshold(Float64Array.of(1, 100), 2, 0.9, zero2, Float64Array.of(0, 1, 1, 0))).toEqual({ etaUp: null, quotient: expect.any(Number) as number });
    expect(computeCertifiedRayleighUpperThreshold(Float64Array.of(1, 100), 2, 0.9, zero2, Float64Array.of(1, 0, 0, 1))!.etaUp).not.toBeNull();
  });

  it('round 4: the certificate refuses each domain boundary on its own, and aliased scratch', () => {
    const p2 = (e: number) => 2 ** e;
    // The certificate at one rate: each guard on its own (K = 0 passes for every eta < eta_det, so only the guard can refuse).
    const cert = (lt: number, beta: number, x: number, w: number, wErr: number, eta: number) =>
      certifyConeCollatzWielandtAtRate(Float64Array.of(lt), 1, beta, Float64Array.of(x), Float64Array.of(w), Float64Array.of(wErr), eta, new Float64Array(1), new Float64Array(1));
    expect(cert(1, 0, 1, 0, 0, 1)).toBe(true);
    expect(cert(1, p2(-20), 1, 0, 0, 1)).toBe(true);
    expect(cert(1, p2(-21), 1, 0, 0, 1)).toBe(false);
    expect(cert(1, 1 - p2(-20), 1, 0, 0, 1)).toBe(true);
    expect(cert(1, 1 - p2(-21), 1, 0, 0, 1)).toBe(false);
    expect(cert(p2(-60), 0, 1, 0, 0, p2(-100))).toBe(true);
    expect(cert(p2(-61), 0, 1, 0, 0, p2(-100))).toBe(false);
    expect(cert(p2(60), 0, 1, 0, 0, p2(-100))).toBe(true);
    expect(cert(p2(61), 0, 1, 0, 0, p2(-100))).toBe(false);
    expect(cert(1, 0, 1, 0, 0, p2(-200))).toBe(true);
    expect(cert(1, 0, 1, 0, 0, p2(-201))).toBe(false);
    expect(cert(1, 0, p2(300), 0, 0, 1)).toBe(true);
    expect(cert(1, 0, p2(301), 0, 0, 1)).toBe(false);
    expect(cert(1, 0, p2(300), p2(300), 0, p2(-200))).toBe(true);
    expect(cert(1, 0, p2(300), p2(301), 0, p2(-200))).toBe(false);
    expect(cert(1, 0, p2(300), 0, p2(300), p2(-200))).toBe(true);
    expect(cert(1, 0, p2(300), 0, p2(301), p2(-200))).toBe(false);
    expect(cert(1, 0, 1, 0, -1e-300, 1)).toBe(false);
    // Aliased scratch (audit, fourth round): n = 2, lt = (1, 1), beta = 0, V(X) = X (D = I, -I, B = 1), so eta* = 1 exactly.
    const eye = (s: number) => Float64Array.of(s, 0, 0, s);
    const lt11 = Float64Array.of(1, 1);
    const sep = (eta: number) => certifyConeCollatzWielandtAtRate(lt11, 2, 0, eye(100), eye(100), new Float64Array(4), eta, new Float64Array(4), new Float64Array(4));
    expect(sep(0.9)).toBe(true);
    expect(sep(1.5)).toBe(false);
    const w = eye(100);
    expect(certifyConeCollatzWielandtAtRate(lt11, 2, 0, eye(100), w, new Float64Array(4), 1.5, new Float64Array(4), w)).toBe(false);
    const xs = eye(1e-4);
    expect(certifyConeCollatzWielandtAtRate(lt11, 2, 0, xs, eye(1e-4), new Float64Array(4), 1.5, new Float64Array(4), xs)).toBe(false);
    for (const which of ['m=x', 'm=w', 'm=wErr', 'work=wErr', 'm=work', 'overlap']) {
      const x = eye(100);
      const ww = eye(100);
      const we = new Float64Array(4);
      const pool = new Float64Array(16);
      let m = new Float64Array(4);
      let work = new Float64Array(4);
      if (which === 'm=x') m = x;
      if (which === 'm=w') m = ww;
      if (which === 'm=wErr') m = we;
      if (which === 'work=wErr') work = we;
      if (which === 'm=work') m = work;
      if (which === 'overlap') { m = pool.subarray(0, 4); work = pool.subarray(3, 7); }
      expect(certifyConeCollatzWielandtAtRate(lt11, 2, 0, x, ww, we, 0.9, m, work)).toBe(false);
    }
    const pool = new Float64Array(8);
    expect(certifyConeCollatzWielandtAtRate(lt11, 2, 0, eye(100), eye(100), new Float64Array(4), 0.9, pool.subarray(0, 4), pool.subarray(4, 8))).toBe(true);
  });

  it('round 4: the verified Cholesky and both noise operators refuse aliased, short or non-symmetric arguments', () => {
    const eye = (v: number) => Float64Array.of(v, 0, 0, v);
    // The verified Cholesky refuses scratch that shares memory with its matrix.
    const a = eye(2);
    expect(verifyPositiveDefiniteShiftedCholesky(a, 2, 0, new Float64Array(4))).toBe(true);
    expect(verifyPositiveDefiniteShiftedCholesky(a, 2, 0, a)).toBe(false);
    // Both noise operators: a non-symmetric x, out === errOut, a short out, and out overlapping x are refused.
    const opDense = createDenseDeviationNoiseOperator(Float64Array.of(1, 2, 2, 3, -1, -2, -2, -3), 2, 2, 1);
    const fRow = Float64Array.of(1, 2, -1, 0.5);
    const opRow = createRowSpaceFactorNoiseOperator(fRow, Float64Array.of(1, 1), 2, 2, 1, 1);
    for (const op of [opDense, opRow]) {
      const refused = (o: Float64Array, e: Float64Array | null) => o.every((v) => Number.isNaN(v) || v === Infinity) && (e === null || e.every((v) => v === Infinity || Number.isNaN(v)));
      const out = new Float64Array(4);
      const err = new Float64Array(4);
      op.apply(Float64Array.of(1, 0.5, 0.5, 2), out, err);
      expect(out.every(Number.isFinite) && err.every(Number.isFinite)).toBe(true);
      op.apply(Float64Array.of(0, 1, 0, 0), out, err);
      expect(refused(out, err)).toBe(true);
      const both = new Float64Array(4);
      op.apply(Float64Array.of(1, 0.5, 0.5, 2), both, both);
      expect(both.every((v) => !Number.isFinite(v))).toBe(true);
      const short = new Float64Array(3);
      op.apply(Float64Array.of(1, 0.5, 0.5, 2), short, new Float64Array(4));
      expect(short.every(Number.isNaN)).toBe(true);
      const xo = Float64Array.of(1, 0.5, 0.5, 2);
      op.apply(xo, xo, err);
      expect(xo.every(Number.isNaN) && err.every((v) => v === Infinity)).toBe(true);
    }
  });

  it('round 4: gradual underflow after rescaling on a net exponent of -1100 is inside the bound', () => {
    // Dense: N = 2, D = (2^-400, 0), x = 2^-300: V(x) = d^2 x / 4 = 2^-1102 > 0, below the smallest subnormal, so out = 0 and the
    // bound must be positive; without the 16 2^-1074 after the rescaling it was 0 (fourth audit round, mutation M3).
    const dense = createDenseDeviationNoiseOperator(Float64Array.of(2 ** -400, 0), 1, 2, 1);
    const out = new Float64Array(1);
    const err = new Float64Array(1);
    dense.apply(Float64Array.of(2 ** -300), out, err);
    expect(out[0]).toBe(0);
    expect(err[0]!).toBeGreaterThan(0);
    expect(err[0]!).toBeLessThan(1e-320);
    // Row space: M_0 = 2^-250, M_1 = 0, x = 2^-100: A = (2^-500, 0), deviations +-2^-501, V(x) = 2^-1102 (mutation M4).
    const row = createRowSpaceFactorNoiseOperator(Float64Array.of(2 ** -250, 0), Float64Array.of(2 ** -501), 1, 2, 1, 1);
    row.apply(Float64Array.of(2 ** -100), out, err);
    expect(out[0]).toBe(0);
    expect(err[0]!).toBeGreaterThan(0);
    expect(err[0]!).toBeLessThan(1e-320);
    // One binade up the value is representable and inside its bound.
    dense.apply(Float64Array.of(2 ** -250), out, err);
    expect(Math.abs(out[0]! - 2 ** -1052)).toBeLessThanOrEqual(err[0]!);
  });

  it('round 4: the adaptive policy reruns 60 steps on a stage 1 checkpoint that 30 steps leave open', () => {
    // Phase 5b run eta 0.4, mu 1e-3, seed 1402, step 200, Gauss–Newton object (P = 33, N = 64): stage 1 v1 marked it as rerun.
    const stored = JSON.parse(fs.readFileSync('evidence/mean-square-boundary/phase5b-runs/eta0.4-mu0.001-seed1402.json', 'utf8')) as { checkpoints: Array<{ step: number; loss: number }> };
    let checked = false;
    regeneratePhase5CheckpointIterates(0.4, 0.001, 1402, 200, (t, theta, loss) => {
      if (t !== 200) return true;
      expect(loss).toBe(stored.checkpoints.find((c) => c.step === 200)!.loss);
      const dev = buildEigenbasisSampleHessianDeviations(createPhase4SampleHessianProvider({ kind: 'mlp33', theta: Float64Array.from(theta), gaussNewton: true }));
      const lt = Float64Array.from(dev.values, (v) => Math.abs(v + 0.001));
      const noise = createDenseDeviationNoiseOperator(unpackEigenbasisDeviationsDense(dev), dev.n, dev.numSamples, 16);
      const ad = computeAdaptiveCertifiedThresholdBracket(lt, dev.n, 0.9, noise);
      expect(ad.lengthsTried).toEqual([30, 60]);
      expect(ad.bracket!.etaLow! / ad.bracket!.etaUp).toBeGreaterThanOrEqual(0.98);
      const at30 = computeCertifiedThresholdBracket(lt, dev.n, 0.9, noise, 30, 1e-12)!;
      const narrow30 = at30.etaLow !== null && at30.etaLow / at30.etaUp >= 0.98;
      expect(narrow30 && at30.ritzResidual <= 1e-3 / at30.ritzEta).toBe(false);
      checked = true;
      return false;
    });
    expect(checked).toBe(true);
  });

  it('round 4: stage 1 summary rules (mismatch, mode sets, duplicates, the 90 % rule, mu = 0, the window)', () => {
    const obj = (narrow: boolean) => ({ etaLow: narrow ? 0.99 : 0.5, etaUp: 1.0, etaStarStored: 1.0, lowOverUp: narrow ? 0.99 : 0.5, upOverStored: 1.0, valid: true, validDense: true, lanczosRerun: false, witnessPositiveDefinite: true });
    type Opt = { mu?: number; mismatchAt?: number | null; gnAgree?: boolean; absAgree?: boolean; narrow?: (i: number) => boolean };
    const rec = (run: string, steps: number[], o: Opt = {}) => ({ run, eta: 0.1, muTrain: o.mu ?? 0.001, mismatchAt: o.mismatchAt ?? null, storedCheckpoints: steps.length, checkpoints: steps.map((step, i) => ({ step, gaussNewton: { ...obj(o.narrow === undefined ? true : o.narrow(i)), removedModesAgree: o.gnAgree ?? true }, absH: { ...obj(true), nullModesAgree: o.absAgree ?? true } })) });
    type Sum = { decision: { validity: string; adoptionGaussNewton: string }; gaussNewtonPerCell: Record<string, { windowReading: Record<string, number> }>; gaussNewtonMu0: { checkpoints: number } };
    const steps = Array.from({ length: 20 }, (_, i) => 200 * (i + 1));
    const ex = [{ run: 'a', steps }, { run: 'z', steps }];
    const sum = (rs: ReturnType<typeof rec>[], e = ex) => summarizePhase6Stage1(rs, e) as Sum;
    expect(sum([rec('a', steps), rec('z', steps)]).decision).toEqual({ validity: 'no violation', adoptionGaussNewton: 'adopted' });
    expect(sum([rec('a', steps, { mismatchAt: 400 }), rec('z', steps)]).decision.validity).toBe('incomplete: no decision');
    expect(sum([rec('a', steps, { gnAgree: false }), rec('z', steps)]).decision.validity).toBe('incomplete: no decision');
    expect(sum([rec('a', steps, { absAgree: false }), rec('z', steps)]).decision.validity).toBe('incomplete: no decision');
    expect(sum([rec('a', steps), rec('a', steps)], ex).decision.validity).toBe('incomplete: no decision');
    expect(sum([rec('a', steps, { mu: 0, gnAgree: false }), rec('z', steps)]).decision.validity).toBe('incomplete: no decision');
    // 40 Gauss–Newton checkpoints at mu > 0: 36 narrow (90 %) adopts, 35 does not.
    const k = (miss: number) => (i: number) => i >= miss;
    expect(sum([rec('a', steps, { narrow: k(4) }), rec('z', steps)]).decision.adoptionGaussNewton).toBe('adopted');
    expect(sum([rec('a', steps, { narrow: k(5) }), rec('z', steps)]).decision.adoptionGaussNewton).toBe('not adopted');
    // A wide mu = 0 run is the quotient object: counted apart, never in the adoption rule.
    const withMu0 = sum([rec('a', steps), rec('z', steps, { mu: 0, narrow: () => false })]);
    expect(withMu0.decision.adoptionGaussNewton).toBe('adopted');
    expect(withMu0.gaussNewtonMu0.checkpoints).toBe(20);
    const allMu0 = sum([rec('a', steps, { mu: 0 }), rec('z', steps, { mu: 0 })]);
    expect(allMu0.decision.adoptionGaussNewton).toBe('not adopted');
    // Rule 3 reads the window only (steps 10 000 to 20 000): 11 of the 20 steps.
    const cells = sum([rec('a', Array.from({ length: 20 }, (_, i) => 1000 * (i + 1))), rec('z', Array.from({ length: 20 }, (_, i) => 1000 * (i + 1)))], [{ run: 'a', steps: Array.from({ length: 20 }, (_, i) => 1000 * (i + 1)) }, { run: 'z', steps: Array.from({ length: 20 }, (_, i) => 1000 * (i + 1)) }]).gaussNewtonPerCell;
    const reading = cells['mu0.001-eta0.1']!.windowReading;
    expect(reading.inside! + reading.below! + reading.above! + reading.undecided!).toBe(22);
  });

  it('round 4: stage 2 record masks every lower bound on a compressed row space and compares brackets without tolerance', () => {
    const br = (etaLow: number | null, etaUp: number) => ({ etaLow, etaUp, etaDet: 4, ritzEta: etaUp, ritzResidual: 0, lanczosSteps: 30, witnessPositiveDefinite: true, bestEps: 1e-3 });
    const base = { step: 10000, eta: 0.1, bracketSeconds: 0, lengthsTried: [30], referenceSeconds: 0, referenceRequested: true, top64: 0.2, meanResidualOverMinCurvature: 0, dualResidualOverTop: 0, spectrumSeconds: 0 };
    type Rec = { bracket: { etaLow: number | null; lowOverUp: number | null; bestEps: number | null }; reference: { etaLow: number | null; lowOverUp: number | null }; lowOverEta: number | null; reading: string; validAgainstReference: boolean | null; validAgainstTop64: boolean | null; rankCompressed: boolean };
    const full = buildPhase6Stage2CheckpointRecord({ ...base, rank: 8, m: 8, bracket: br(0.099, 0.1), reference: br(0.0995, 0.1) }) as Rec;
    expect(full.rankCompressed).toBe(false);
    expect(full.bracket.etaLow).toBe(0.099);
    expect(full.reading).toBe('inside');
    expect(full.validAgainstReference).toBe(true);
    expect(full.validAgainstTop64).toBe(true);
    const comp = buildPhase6Stage2CheckpointRecord({ ...base, rank: 6, m: 8, bracket: br(0.099, 0.1), reference: br(0.0995, 0.1) }) as Rec;
    expect(comp.rankCompressed).toBe(true);
    expect([comp.bracket.etaLow, comp.bracket.lowOverUp, comp.bracket.bestEps, comp.reference.etaLow, comp.reference.lowOverUp, comp.lowOverEta]).toEqual([null, null, null, null, null, null]);
    expect(comp.reading).toBe('undecided');
    expect(comp.validAgainstReference).toBe(false);
    expect(comp.validAgainstTop64).toBeNull();
    // Strict overlap: a lower end equal to the other upper end, or above it by 1e-9 relative, is a violation.
    expect((buildPhase6Stage2CheckpointRecord({ ...base, rank: 8, m: 8, bracket: br(0.1, 0.11), reference: br(0.09, 0.1) }) as Rec).validAgainstReference).toBe(false);
    expect((buildPhase6Stage2CheckpointRecord({ ...base, rank: 8, m: 8, bracket: br(0.09, 0.1), reference: br(0.1 * (1 + 1e-9), 0.11) }) as Rec).validAgainstReference).toBe(false);
    expect((buildPhase6Stage2CheckpointRecord({ ...base, rank: 8, m: 8, bracket: br(0.09, 0.1), reference: br(0.095, 0.11), referenceRequested: false }) as Rec).validAgainstReference).toBeNull();
    expect((buildPhase6Stage2CheckpointRecord({ ...base, rank: 8, m: 8, bracket: br(null, 0.1), reference: br(0.095, 0.11) }) as Rec).validAgainstReference).toBe(false);
    expect((buildPhase6Stage2CheckpointRecord({ ...base, rank: 8, m: 8, bracket: br(0.5, 0.6), reference: br(0.5, 0.6) }) as Rec).reading).toBe('above');
    expect((buildPhase6Stage2CheckpointRecord({ ...base, rank: 8, m: 8, bracket: br(0.01, 0.05), reference: br(0.01, 0.05) }) as Rec).reading).toBe('below');
  });

  it('round 4: the dual eigen-residual of the Phase 6 spectrum is small, and computed only on request', () => {
    const mlp = createScaleMlpDatasetAndOps(6, 5, 3, 16, 0x67500011, 0x9e3779b9);
    const net = createPhase6StudentNetwork(mlp.xData, mlp.yData, 6, 5, 3, 16, 'tanh');
    const theta = Float64Array.from(mlp.checkpointTheta0);
    const plain = computeStudentGaussNewtonDualSpectrum(net, theta, 1e-10);
    const withRes = computeStudentGaussNewtonDualSpectrum(net, theta, 1e-10, undefined, true);
    expect(plain.dualResidualMax).toBeNull();
    expect(Array.from(withRes.values)).toEqual(Array.from(plain.values));
    expect(Array.from(withRes.vt)).toEqual(Array.from(plain.vt));
    expect(withRes.rank).toBeGreaterThan(1);
    expect(withRes.dualResidualMax!).toBeLessThan(1e-12 * withRes.values[0]!);
    expect(withRes.dualResidualMax!).toBeGreaterThan(0);
    // The helper on a known pair: G = diag(3, 1), v_0 rotated by t from e_0 gives G v - 3 v = (0, -2 sin t).
    const t = 0.01;
    const res = computeDualEigenResidualMax(Float64Array.of(3, 0, 0, 1), 2, Float64Array.of(3, 1), Float64Array.of(Math.cos(t), Math.sin(t), 0, 1), 2);
    expect(Math.abs(res - 2 * Math.sin(t))).toBeLessThan(1e-15);
    expect(computeDualEigenResidualMax(Float64Array.of(3, 0, 0, 1), 2, Float64Array.of(3, 1), Float64Array.of(1, 0, 0, 1), 2)).toBe(0);
  });

  it('round 6: shared memory is refused, adoption needs validity, zero noise keeps the range of x', () => {
    // Audit: a structuredClone of a SharedArrayBuffer shares its bytes under a different buffer object; with w on one and work on
    // the other the Cholesky of x overwrote w and eta = 1.5 was certified at eta* = 1 (n = 2, lt = 1, V(X) = X).
    const eye = (v: number, buf?: Float64Array) => { const m = buf ?? new Float64Array(4); m.set([v, 0, 0, v]); return m; };
    const sab = new SharedArrayBuffer(32);
    const wShared = eye(100, new Float64Array(sab));
    const workClone = new Float64Array(structuredClone(sab));
    expect(certifyConeCollatzWielandtAtRate(Float64Array.of(1, 1), 2, 0, eye(100), wShared, new Float64Array(4), 1.5, new Float64Array(4), workClone)).toBe(false);
    expect(certifyConeCollatzWielandtAtRate(Float64Array.of(1, 1), 2, 0, eye(100), eye(100), new Float64Array(4), 0.9, new Float64Array(4), new Float64Array(4))).toBe(true);
    expect(certifyConeCollatzWielandtAtRate(Float64Array.of(1, 1), 2, 0, eye(100), wShared, new Float64Array(4), 0.9, new Float64Array(4), new Float64Array(4))).toBe(false);
    expect(verifyPositiveDefiniteShiftedCholesky(eye(1, new Float64Array(new SharedArrayBuffer(32))), 2, 0, new Float64Array(4))).toBe(false);
    // A shared output cannot carry the refusal signal safely: the operator throws (ninth audit round).
    const sharedOut = new Float64Array(new SharedArrayBuffer(32));
    expect(() => createDenseDeviationNoiseOperator(Float64Array.of(1, 0, 0, 1, -1, 0, 0, -1), 2, 2, 1).apply(eye(1), sharedOut, new Float64Array(4))).toThrow(TypeError);
    // Adoption needs the validity rule on every object: a narrow valid Gauss–Newton object beside an invalid |H| object is rejected.
    const obj = (ok: boolean) => ({ etaLow: 0.99, etaUp: 1.0, etaStarStored: 1.0, lowOverUp: 0.99, upOverStored: 1.0, valid: ok, validDense: ok, lanczosRerun: false, witnessPositiveDefinite: true });
    const rec = (run: string) => ({ run, eta: 0.1, muTrain: 0.001, mismatchAt: null, storedCheckpoints: 2, checkpoints: [200, 400].map((step) => ({ step, gaussNewton: { ...obj(true), removedModesAgree: true }, absH: { ...obj(step !== 400), nullModesAgree: true } })) });
    const sum = summarizePhase6Stage1([rec('a'), rec('b')], [{ run: 'a', steps: [200, 400] }, { run: 'b', steps: [200, 400] }]) as { decision: { validity: string; adoptionGaussNewton: string } };
    expect(sum.decision).toEqual({ validity: 'violation: method rejected', adoptionGaussNewton: 'not adopted' });
    // Zero noise: x outside [2^-1000, 2^1000] is refused as with nonzero noise (it returned 0 and a zero bound).
    for (const xv of [2 ** 1001, 2 ** -1074]) {
      for (const op of [createDenseDeviationNoiseOperator(Float64Array.of(0, 0), 1, 2, 1), createRowSpaceFactorNoiseOperator(Float64Array.of(0, 0), Float64Array.of(1), 1, 2, 1, 1)]) {
        const out = new Float64Array(1);
        const err = new Float64Array(1);
        op.apply(Float64Array.of(xv), out, err);
        expect(Number.isNaN(out[0]!) && err[0] === Infinity).toBe(true);
        op.apply(Float64Array.of(1), out, err);
        expect([out[0], err[0]]).toEqual([0, 0]);
      }
    }
  });

  it('round 6: inputs at which the rounding terms decide (Frobenius rho, dense N count, per-z Rayleigh, witness cache, eta_det)', () => {
    // F1: n = 8, lt = 1, beta = 0, V(X) = P X P with P = 1 1^T / 8 (eta* = 1), x = I, w = (7/64) 1 1^T, wErr = 1/64: the error pattern
    // is rank one, so the spectral norm of the error equals its Frobenius bound; the largest entry alone certified eta = 1.05.
    {
      const n = 8;
      const nn = n * n;
      const x = new Float64Array(nn);
      for (let a = 0; a < n; a++) x[a * n + a] = 1;
      const cert = (eta: number) => certifyConeCollatzWielandtAtRate(new Float64Array(n).fill(1), n, 0, x, new Float64Array(nn).fill(7 / 64), new Float64Array(nn).fill(1 / 64), eta, new Float64Array(nn), new Float64Array(nn));
      expect(cert(0.9)).toBe(true);
      for (const eta of [1.0, 1.01, 1.03, 1.05]) expect(cert(eta)).toBe(false);
    }
    // F2: the dense sample sum with N - 1 same-sign roundings (the round-1 instance of the rank-factor operator): D_0 = 1,
    // D_i = sqrt(c u), judged exactly.
    for (const c of [1.01, 0.98]) for (const N of [200, 1000, 4000]) for (const beta of [0, 0.9]) {
      const d = new Float64Array(N);
      d[0] = 1;
      for (let i = 1; i < N; i++) d[i] = Math.sqrt(c * 1.1102230246251565e-16);
      const noise = createDenseDeviationNoiseOperator(d, 1, N, 1);
      for (const l of [0.25, 0.5, 1, 2]) {
        const br = computeCertifiedThresholdBracket(Float64Array.of(l), 1, beta, noise)!;
        // Both ends decided (seventh audit round: a null lower end and eta_up = eta_det passed this check).
        expect(br.etaLow).not.toBeNull();
        expect(expectExactEnclosure(Float64Array.of(l), 1, beta, d, N, 1, br.etaLow, br.etaUp)).toBe(true);
      }
    }
    // F3: the Rayleigh bound of each test matrix, etaUp >= <z, L0 z> / <z, B z> exactly, with a cancelling B form (z = [[1, t], [t, 0]],
    // 2 |d_12| t^2 just below d_11, zero noise).
    {
      const zeroNoise = createDenseDeviationNoiseOperator(new Float64Array(4), 2, 1, 1);
      let seed = 987654321;
      const rnd = (): number => { seed = (Math.imul(seed, 1103515245) + 12345) >>> 0; return seed / 4294967296; };
      let checked = 0;
      for (let rep = 0; rep < 4000; rep++) {
        const beta = [0.5, 0.9, 0.99][rep % 3]!;
        const ltv = Float64Array.of(1 + rnd(), 0.1 + 0.5 * rnd());
        const B = frac(beta);
        const one: Q = [1n, 1n];
        const cb = div(B, mul(mul(sub(one, B), sub(one, B)), add(one, B)));
        const dQ = (a: number, b: number): Q => { const df = sub(frac(ltv[a]!), frac(ltv[b]!)); return sub(div(mul(frac(ltv[a]!), frac(ltv[b]!)), add(one, B)), mul(cb, mul(df, df))); };
        const d11 = Number(dQ(0, 0)[0]) / Number(dQ(0, 0)[1]);
        const d12 = Number(dQ(0, 1)[0]) / Number(dQ(0, 1)[1]);
        const eps = 10 ** -(4 + 6 * rnd());
        if (!(d12 < 0)) continue;
        const t = Math.sqrt((d11 * (1 - eps)) / (2 * -d12));
        const z = Float64Array.of(1, t, t, 0);
        const r = computeCertifiedRayleighUpperThreshold(ltv, 2, beta, zeroNoise, z);
        if (r === null || r.etaUp === null) continue;
        let num: Q = [0n, 1n];
        let den: Q = [0n, 1n];
        for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) {
          const zz = mul(frac(z[a * 2 + b]!), frac(z[a * 2 + b]!));
          num = add(num, mul(dQ(a, b), zz));
          den = add(den, mul(add(frac(ltv[a]!), frac(ltv[b]!)), zz));
        }
        if (!(num[0] > 0n)) continue;
        expect(le(div(den, num), frac(r.etaUp))).toBe(true);
        checked++;
      }
      expect(checked).toBeGreaterThan(3000);
    }
    // F4: the witness cache compares contents: a verified x = 1 at n = 1, then the indefinite x = -1 of round 3 (eta* = 0.4).
    expect(certifyConeCollatzWielandtAtRate(Float64Array.of(1), 1, 0, Float64Array.of(1), new Float64Array(1), new Float64Array(1), 0.1, new Float64Array(1), new Float64Array(1))).toBe(true);
    expect(certifyConeCollatzWielandtAtRate(Float64Array.of(1), 1, 0, Float64Array.of(-1), Float64Array.of(-4), new Float64Array(1), 1.5, new Float64Array(1), new Float64Array(1))).toBe(false);
    // F5: eta_det rounded up by 1 + 8u (1 + 2u falls below the exact ceiling at this pair), noise-free bracket judged exactly.
    {
      const beta = 0.028279689279990246;
      const l = 214.96453751419017;
      const br = computeCertifiedThresholdBracket(Float64Array.of(l), 1, beta, createDenseDeviationNoiseOperator(new Float64Array(2), 1, 2, 1))!;
      const [en, ed] = frac(br.etaUp);
      const [bn, bd] = frac(beta);
      const [ln, ld] = frac(l);
      expect(en * bd * ln >= 2n * (bd + bn) * ld * ed).toBe(true);
    }
  });

  it('round 7: the L0 rounding factor of the Rayleigh bound, and a non-positive Ritz value inside the domain', () => {
    // Zero noise, so eta* = eta_det; all lt = 2^k; z_00 = 1 and every other entry (1 + r) 2^-27, so that each small L0 term rounds
    // away and each small B term rounds up: without the factor (1 + gamma_{n^2 + 4}) on the L0 form etaUp fell below eta_det at n >= 12.
    let checked = 0;
    for (const n of [12, 16, 24, 32]) for (const beta of [0.05, 0.2, 0.4, 0.6]) for (const k of [0, -30, 17]) for (const rj of [1, 200, 20000, 100000, 150000]) {
      const lt = new Float64Array(n).fill(2 ** k);
      const z = new Float64Array(n * n).fill((1 + rj * 2 ** -20) * 2 ** -27);
      z[0] = 1;
      const r = computeCertifiedRayleighUpperThreshold(lt, n, beta, createDenseDeviationNoiseOperator(new Float64Array(n * n), n, 1, 1), z);
      if (r === null || r.etaUp === null) continue;
      const [en, ed] = frac(r.etaUp);
      const [bn, bd] = frac(beta);
      const [ln, ld] = frac(2 ** k);
      expect(en * bd * ln >= 2n * (bd + bn) * ld * ed).toBe(true);
      checked++;
    }
    expect(checked).toBeGreaterThan(150);
    // Centring cancellation: deviations 2^30 + 0.5 +- 1 (exact centred V(1) = 1/B), fl V(1) = 0 with a large bound and a negative Ritz
    // value; the bracket is eta_det rounded up with a verified lower end, judged exactly (it was null).
    const C = 2 ** 30 + 0.5;
    const dev = Float64Array.of(C + 1, C - 1, C + 1, C - 1);
    const br = computeCertifiedThresholdBracket(Float64Array.of(1), 1, 0.5, createDenseDeviationNoiseOperator(dev, 1, 4, 2));
    expect(br).not.toBeNull();
    expect(br!.ritzEta).toBeLessThan(0);
    expect(br!.etaLow).not.toBeNull();
    expectExactEnclosure(Float64Array.of(1), 1, 0.5, dev, 4, 2, br!.etaLow, br!.etaUp);
    // Shared memory is refused at every entry point, also for lt, z and the noise inputs (seventh audit round: a shared lt rewritten
    // by a worker during the call gave an etaUp below the threshold of the final lt).
    const ltShared = new Float64Array(new SharedArrayBuffer(8));
    ltShared[0] = 2;
    const zero1 = createDenseDeviationNoiseOperator(new Float64Array(2), 1, 2, 1);
    expect(computeCertifiedThresholdBracket(Float64Array.of(2), 1, 0, zero1)).not.toBeNull();
    expect(computeCertifiedThresholdBracket(ltShared, 1, 0, zero1)).toBeNull();
    expect(computeCertifiedRayleighUpperThreshold(ltShared, 1, 0, zero1, Float64Array.of(1))).toBeNull();
    const zShared = new Float64Array(new SharedArrayBuffer(8));
    zShared[0] = 1;
    expect(computeCertifiedRayleighUpperThreshold(Float64Array.of(2), 1, 0, zero1, zShared)).toBeNull();
    const devShared = new Float64Array(new SharedArrayBuffer(16));
    devShared.set([1, -1]);
    expect(computeCertifiedThresholdBracket(Float64Array.of(2), 1, 0, createDenseDeviationNoiseOperator(devShared, 1, 2, 1))).toBeNull();
    const facShared = new Float64Array(new SharedArrayBuffer(16));
    facShared.set([1, 0]);
    expect(computeCertifiedThresholdBracket(Float64Array.of(2), 1, 0, createRowSpaceFactorNoiseOperator(facShared, Float64Array.of(0.5), 1, 2, 1, 1))).toBeNull();
    expect(computeCertifiedThresholdBracket(Float64Array.of(2), 1, 0, createRowSpaceFactorNoiseOperator(Float64Array.of(1, 0), Float64Array.of(0.5), 1, 2, 1, 1))).not.toBeNull();
  });

  it('round 8: inputs where l - dl, the 2R and the 2n terms decide, and a noise operator of another size', () => {
    // Certificate, n = 1, eta 34u-64u below eta_det, where the computed l exceeds the exact one by more than 3 %: with V(x) = W such that
    // coef W / fl(l) = 0.99, dividing by l instead of l - dl certified rates with exact x - K(x) < 0. Every certified rate must have
    // exact x - K(x) > 0.
    {
      const u = 1.1102230246251565e-16;
      let seed = 7;
      const rnd = (): number => { seed = (Math.imul(seed, 1103515245) + 12345) >>> 0; return seed / 4294967296; };
      let reached = 0;
      for (let rep = 0; rep < 20000; rep++) {
        const beta = [0, 0.5, 0.9][rep % 3]!;
        const lt = 1 + rnd();
        const eta = ((2 * (1 + beta)) / lt) * (1 - (34 + 30 * rnd()) * u);
        const lComp = lt + lt + eta * (0 - lt * lt * (1.0 / (1.0 + beta)));
        const one: Q = [1n, 1n];
        const B = frac(beta);
        const lEx = sub(add(frac(lt), frac(lt)), div(mul(frac(eta), mul(frac(lt), frac(lt))), add(one, B)));
        if (!(lEx[0] > 0n)) continue;
        if (!(Number((lEx[0] * frac(lComp)[1] * 10n ** 9n) / (lEx[1] * frac(lComp)[0])) / 1e9 < 0.97)) continue;
        const W = (0.99 * lComp) / (eta / (1 - beta));
        const mEx = sub(one, div(mul(div(frac(eta), sub(one, B)), frac(W)), lEx));
        if (!(mEx[0] > 0n)) reached++;
        const ok = certifyConeCollatzWielandtAtRate(Float64Array.of(lt), 1, beta, Float64Array.of(1), Float64Array.of(0), Float64Array.of(W), eta, new Float64Array(1), new Float64Array(1));
        if (ok) expect(mEx[0] > 0n).toBe(true);
      }
      expect(reached).toBeGreaterThan(0);
    }
    // Noise operators: sums whose additions all round the same way (one entry 1, the rest 1.01 u, x all ones); the exact V is
    // c (sum m)^2 m_a m_b with c = (N - 1) / N^2; without 2R (row space) or 2n (dense) in the counts the bound was exceeded 2.5-19 times.
    {
      const u = 1.1102230246251565e-16;
      const R = 256;
      const N = 32;
      const m = new Float64Array(R);
      m[0] = 1;
      for (let k = 1; k < R; k++) m[k] = 1.01 * u;
      const factors = new Float64Array(N * R);
      factors.set(m, 0);
      const out = new Float64Array(R * R);
      const err = new Float64Array(R * R);
      createRowSpaceFactorNoiseOperator(factors, new Float64Array(R).fill(1), R, N, 1, 1).apply(new Float64Array(R * R).fill(1), out, err);
      let sm: Q = [0n, 1n];
      for (let k = 0; k < R; k++) sm = add(sm, frac(m[k]!));
      const c = mul(mul(sm, sm), [BigInt(N - 1), BigInt(N * N)]);
      for (const [a, b] of [[0, 0], [0, 1], [1, 1], [0, R - 1]] as const) {
        expect(le(absQ(sub(frac(out[a * R + b]!), mul(c, mul(frac(m[a]!), frac(m[b]!))))), frac(err[a * R + b]!))).toBe(true);
      }
      const n = 128;
      const Nd = 4;
      const nn = n * n;
      const dev = new Float64Array(Nd * nn);
      dev[0] = 1;
      for (let k = 1; k < n; k++) { dev[k] = 1.01 * u; dev[k * n] = 1.01 * u; }
      const outD = new Float64Array(nn);
      const errD = new Float64Array(nn);
      createDenseDeviationNoiseOperator(dev, n, Nd, 1).apply(new Float64Array(nn).fill(1), outD, errD);
      const v: Q[] = [];
      for (let a = 0; a < n; a++) { let t: Q = [0n, 1n]; for (let k = 0; k < n; k++) t = add(t, frac(dev[a * n + k]!)); v.push(t); }
      const cd: Q = [BigInt(Nd - 1), BigInt(Nd * Nd)];
      for (const [a, b] of [[0, 0], [0, 1], [1, 1], [0, n - 1]] as const) {
        expect(le(absQ(sub(frac(outD[a * n + b]!), mul(cd, mul(v[a]!, v[b]!)))), frac(errD[a * n + b]!))).toBe(true);
      }
    }
    // A noise operator of another dimension is refused (a rank-1 row-space operator passed with n = 2 gave a full bracket).
    const rank1 = createRowSpaceFactorNoiseOperator(Float64Array.of(1, 0), Float64Array.of(0.5), 1, 2, 1, 1);
    expect(computeCertifiedThresholdBracket(Float64Array.of(1), 1, 0, rank1)).not.toBeNull();
    expect(computeCertifiedThresholdBracket(Float64Array.of(1, 1), 2, 0, rank1)).toBeNull();
    expect(computeCertifiedRayleighUpperThreshold(Float64Array.of(1, 1), 2, 0, rank1, Float64Array.of(1, 0, 0, 1))).toBeNull();
  });

  it('round 8b: shared memory from another realm, the exact oracle at large N, the arrow matrix, and conservative pins', () => {
    // A SharedArrayBuffer created in another realm is not an instance of this realm's SharedArrayBuffer; the brand check refuses it.
    const foreign = vm.runInNewContext('new SharedArrayBuffer(16)') as SharedArrayBuffer;
    const ltForeign = new Float64Array(foreign, 0, 1);
    ltForeign[0] = 10;
    const devPair = createDenseDeviationNoiseOperator(Float64Array.of(1, -1), 1, 2, 1);
    expect(computeCertifiedThresholdBracket(Float64Array.of(10), 1, 0, devPair)).not.toBeNull();
    expect(computeCertifiedThresholdBracket(ltForeign, 1, 0, devPair)).toBeNull();
    expect(computeCertifiedRayleighUpperThreshold(ltForeign, 1, 0, devPair, Float64Array.of(1))).toBeNull();
    expect(verifyPositiveDefiniteShiftedCholesky(new Float64Array(foreign, 8, 1).fill(1), 1, 0, new Float64Array(1))).toBe(false);
    expect(computeCertifiedThresholdBracket(Float64Array.of(10), 1, 0, createDenseDeviationNoiseOperator(new Float64Array(foreign, 0, 2), 1, 2, 1))).toBeNull();
    // The exact oracle at N = 100001, B = 1000001 (B N^2 above 2^53): PD at the float just below the exact eta*, not at the one above.
    {
      const N = 100001;
      const B = 1000001;
      const l = 1.226705859381049;
      const dev = new Float64Array(N);
      dev[0] = 1e6;
      dev[1] = -1e6;
      const L = frac(l);
      const etaStar = div(mul([2n, 1n], L), add(mul(L, L), [2000000000000n, BigInt(B) * BigInt(N)]));
      const step = (x: number, up: boolean): number => { const b = new DataView(new ArrayBuffer(8)); b.setFloat64(0, x); b.setBigUint64(0, b.getBigUint64(0) + (up ? 1n : -1n)); return b.getFloat64(0); };
      let lo = Number(etaStar[0]) / Number(etaStar[1]);
      const lessQ = (x: Q, y: Q): boolean => x[0] * y[1] < y[0] * x[1];
      while (!lessQ(frac(lo), etaStar)) lo = step(lo, false);
      let hi = step(lo, true);
      while (!lessQ(etaStar, frac(hi))) hi = step(hi, true);
      expect(isPositiveDefinite(formAt(Float64Array.of(l), 1, 0, dev, N, B, lo), 1)).toBe(true);
      expect(isPositiveDefinite(formAt(Float64Array.of(l), 1, 0, dev, N, B, hi), 1)).toBe(false);
    }
    // Arrow matrix (audit): 126 diagonal entries 2^-40 coupled by 2^-48 to the last coordinate, a unit 2 x 2 block, last diagonal
    // 1 + 12 u; the exact last Schur complement is -30 / 2^56 < 0 (indefinite). gamma_1 in place of gamma_{n+2} accepted it.
    {
      const n = 128;
      const a = new Float64Array(n * n);
      for (let i = 0; i < 126; i++) { a[i * n + i] = 2 ** -40; a[i * n + 127] = 2 ** -48; a[127 * n + i] = 2 ** -48; }
      a[126 * n + 126] = 1;
      a[126 * n + 127] = 1;
      a[127 * n + 126] = 1;
      a[127 * n + 127] = 1 + 12 * 2 ** -53;
      expect(verifyPositiveDefiniteShiftedCholesky(a, n, 0, new Float64Array(n * n))).toBe(false);
    }
    // Conservative pins (audit): inputs at which removing a proof term flips the boolean to true. Each rate is below the
    // exact threshold, so these pin the terms rather than soundness: gamma_6 |k| and eta (p1 + p2) in the certificate (n = 1,
    // V(Y) = Y, eta* = 1), the kErr underflow budget (lt = 2^-60, x = 2^-920, zero noise), and the pivot floor of the Cholesky.
    {
      const u = 1.1102230246251565e-16;
      const cert = (lt: number, x: number, w: number, eta: number) => certifyConeCollatzWielandtAtRate(Float64Array.of(lt), 1, 0, Float64Array.of(x), Float64Array.of(w), new Float64Array(1), eta, new Float64Array(1), new Float64Array(1));
      expect(cert(1, 1, 1, 1 - 25 * u)).toBe(false);
      expect(cert(1, 1, 1, 1 - 20 * u)).toBe(false);
      expect(cert(1, 1, 1, 0.9)).toBe(true);
      expect(cert(2 ** -60, 2 ** -920, 0, 2 ** 61 * (1 - 64 * u))).toBe(false);
      expect(verifyPositiveDefiniteShiftedCholesky(Float64Array.of(2 ** -1000), 1, 0, new Float64Array(1))).toBe(false);
      expect(verifyPositiveDefiniteShiftedCholesky(Float64Array.of(2 ** -400), 1, 0, new Float64Array(1))).toBe(true);
    }
  });

  it('round 9: arguments that misreport themselves (subclass, own buffer property, Proxy, Int32Array) are refused or copied', () => {
    const one = () => Float64Array.of(1);
    // Audit: a subclass whose subarray lies poisoned the witness cache with [-1]; then x = w = -1 at eta = 3/2 (eta* = 1) passed.
    const liar = new Float64Array(1);
    Object.defineProperty(liar, 'subarray', { value: () => Float64Array.of(-1) });
    liar[0] = 1;
    expect(certifyConeCollatzWielandtAtRate(one(), 1, 0, liar, one(), new Float64Array(1), 0.5, new Float64Array(1), new Float64Array(1))).toBe(true);
    expect(certifyConeCollatzWielandtAtRate(one(), 1, 0, Float64Array.of(-1), Float64Array.of(-1), new Float64Array(1), 1.5, new Float64Array(1), new Float64Array(1))).toBe(false);
    // An own buffer property that hides an alias of w and work (V = 4X, x = 2, w = 8, eta = 1/2 > eta* = 2/5).
    const store = new Float64Array(1);
    const w = new Float64Array(store.buffer);
    w[0] = 8;
    const work = new Float64Array(store.buffer);
    Object.defineProperty(work, 'buffer', { value: new ArrayBuffer(8) });
    expect(certifyConeCollatzWielandtAtRate(one(), 1, 0, Float64Array.of(2), w, new Float64Array(1), 0.5, new Float64Array(1), work)).toBe(false);
    // A Proxy that returns 2 for its first reads of lt[0] and 1 afterwards (exact eta* = 2 for the final lt) is refused.
    let reads = 0;
    const target = Float64Array.of(1);
    const proxy = new Proxy(target, { get(t, k) { if (k === '0') return ++reads <= 3 ? 2 : 1; const v = Reflect.get(t, k, t) as unknown; return typeof v === 'function' ? (v as (...a: unknown[]) => unknown).bind(t) : v; } }) as Float64Array;
    const zero1 = createDenseDeviationNoiseOperator(new Float64Array(2), 1, 2, 1);
    expect(computeCertifiedThresholdBracket(proxy, 1, 0, zero1)).toBeNull();
    expect(computeCertifiedRayleighUpperThreshold(proxy, 1, 0, zero1, one())).toBeNull();
    expect(certifyConeCollatzWielandtAtRate(proxy, 1, 0, one(), new Float64Array(1), new Float64Array(1), 0.5, new Float64Array(1), new Float64Array(1))).toBe(false);
    // Int32Array scratch quantised the factor and certified the indefinite [[2.1, 2.2], [2.2, 2.1]].
    expect(verifyPositiveDefiniteShiftedCholesky(Float64Array.of(2.1, 2.2, 2.2, 2.1), 2, 0, new Int32Array(4) as unknown as Float64Array)).toBe(false);
    expect(verifyPositiveDefiniteShiftedCholesky(Float64Array.of(2.2, 2.1, 2.1, 2.2), 2, 0, new Float64Array(4))).toBe(true);
    // A noise operator refuses a Proxy input and throws on an output that cannot carry the refusal.
    const op = createDenseDeviationNoiseOperator(Float64Array.of(1, -1), 1, 2, 1);
    const out = new Float64Array(1);
    op.apply(new Proxy(Float64Array.of(1), {}) as Float64Array, out, null);
    expect(Number.isNaN(out[0]!)).toBe(true);
    expect(() => op.apply(Float64Array.of(1), new Int32Array(1) as unknown as Float64Array, null)).toThrow(TypeError);
    // A genuine Float64Array of another realm is accepted.
    const foreignF64 = vm.runInNewContext('new Float64Array([10])') as Float64Array;
    expect(computeCertifiedThresholdBracket(foreignF64, 1, 0, op)).not.toBeNull();
    // Stage 1 window: steps 10 000 to 20 000 inclusive only.
    const obj = { etaLow: 0.09, etaUp: 0.1, etaStarStored: 0.095, lowOverUp: 0.9, upOverStored: 1, valid: true, validDense: true, lanczosRerun: false, witnessPositiveDefinite: true };
    const steps = [9800, 10000, 20000, 20200];
    const rec = { run: 'a', eta: 0.1, muTrain: 0.001, mismatchAt: null, storedCheckpoints: 4, checkpoints: steps.map((step) => ({ step, gaussNewton: { ...obj, removedModesAgree: true }, absH: { ...obj, nullModesAgree: true } })) };
    const sum = summarizePhase6Stage1([rec], [{ run: 'a', steps }]) as { gaussNewtonPerCell: Record<string, { windowReading: Record<string, number> }> };
    const reading = sum.gaussNewtonPerCell['mu0.001-eta0.1']!.windowReading;
    expect(reading.inside! + reading.below! + reading.above! + reading.undecided!).toBe(2);
  });

  it('round 9: the size factors of the underflow budgets (entries lost by the scaling, x at the bottom of the domain)', () => {
    // |k out - c V'| <= k err exactly, in dyadic arithmetic; without N dOut^2 (row space) or N (dense, N = 4096) in U the bound fails.
    const dyMat = (A: Dy[], Bm: Dy[], n: number): Dy[] => { const C: Dy[] = []; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { let t: Dy = { m: 0n, e: 0 }; for (let k = 0; k < n; k++) t = dyAdd(t, dyMul(A[i * n + k]!, Bm[k * n + j]!)); C.push(t); } return C; };
    const within = (out: number, err: number, k: bigint, exactScaled: Dy): boolean => {
      const d = dyAdd(dyMul(dy(out), { m: k, e: 0 }), dyNeg(exactScaled));
      const slack = dyAdd(dyMul(dy(err), { m: k, e: 0 }), dyNeg(d.m < 0n ? dyNeg(d) : d));
      return slack.m >= 0n;
    };
    {
      const R = 2;
      const dOut = 1024;
      const N = 2;
      const fac = new Float64Array(N * dOut * R);
      for (let c = 0; c < dOut; c++) { fac[c * R] = 2 ** 259; fac[c * R + 1] = 0.49 * 2 ** -814; }
      const x = new Float64Array(R * R).fill(2 ** -1000);
      const A: Dy[] = [];
      for (let a = 0; a < R; a++) for (let b = 0; b < R; b++) { let t: Dy = { m: 0n, e: 0 }; for (let c = 0; c < dOut; c++) t = dyAdd(t, dyMul(dy(fac[c * R + a]!), dy(fac[c * R + b]!))); A.push(t); }
      const AXA = dyMat(dyMat(A, Array.from(x, dy), R), A, R);
      const out = new Float64Array(R * R);
      const err = new Float64Array(R * R);
      createRowSpaceFactorNoiseOperator(fac, new Float64Array(R), R, N, dOut, 1).apply(x, out, err);
      for (const q of [0, 1]) expect(within(out[q]!, err[q]!, BigInt(N * N), dyMul(dyInt(N - 1), AXA[q]!))).toBe(true);
    }
    {
      const n = 2;
      const N = 4096;
      const sm = 0.49 * 2 ** -474;
      const dev = new Float64Array(N * n * n);
      for (let i = 0; i < N; i++) { const g = i % 2 === 0 ? 1 : -1; dev.set([g * 2 ** 599, g * sm, g * sm, 0], i * n * n); }
      const x = new Float64Array(n * n).fill(2 ** -1000);
      const D = [dy(2 ** 599), dy(sm), dy(sm), { m: 0n, e: 0 }];
      const DXD = dyMat(dyMat(D, Array.from(x, dy), n), D, n);
      const out = new Float64Array(n * n);
      const err = new Float64Array(n * n);
      createDenseDeviationNoiseOperator(dev, n, N, 1).apply(x, out, err);
      for (const q of [0, 1]) expect(within(out[q]!, err[q]!, 1n, DXD[q]!)).toBe(true);
    }
  });

  it('round 10: numbers must be primitive numbers, detached views are refused, the dense reference decides, the runners check the tail', () => {
    // beta = '0.9' passed the range test and made 1 + beta the string '10.9' (eta = 4 certified where eta* = 3.8).
    const zero1 = createDenseDeviationNoiseOperator(Float64Array.of(0), 1, 1, 1);
    const one = () => Float64Array.of(1);
    let calls = 0;
    const shifty = { valueOf: () => (++calls <= 2 ? 0.9 : 0.99) };
    expect(computeCertifiedThresholdBracket(one(), 1, 0.9, zero1)).not.toBeNull();
    for (const bad of ['0.9', shifty, new Number(0.9)] as unknown[]) {
      expect(computeCertifiedThresholdBracket(one(), 1, bad as number, zero1)).toBeNull();
      expect(computeCertifiedRayleighUpperThreshold(one(), 1, bad as number, zero1, one())).toBeNull();
      expect(certifyConeCollatzWielandtAtRate(one(), 1, bad as number, one(), new Float64Array(1), new Float64Array(1), 1, new Float64Array(1), new Float64Array(1))).toBe(false);
    }
    expect(certifyConeCollatzWielandtAtRate(one(), 1, 0.9, one(), new Float64Array(1), new Float64Array(1), '1' as unknown as number, new Float64Array(1), new Float64Array(1))).toBe(false);
    expect(computeCertifiedThresholdBracket(one(), 1, 0.9, zero1, 60, 1e-10, [1e-3, '1e-4' as unknown as number])).toBeNull();
    expect(computeCertifiedThresholdBracket(one(), 1, 0.9, zero1, '60' as unknown as number)).toBeNull();
    // A detached view (its buffer transferred away) is refused even where length 0 would pass.
    const det = new Float64Array(1);
    structuredClone(det.buffer, { transfer: [det.buffer] });
    expect(verifyPositiveDefiniteShiftedCholesky(one(), 1, 0, det)).toBe(false);
    expect(() => zero1.apply(one(), det, null)).toThrow(TypeError);
    // The dense reference decides validity when present, in both directions of disagreement with the stored value.
    const obj = (valid: boolean, validDense: boolean) => ({ etaLow: 0.99, etaUp: 1.0, etaStarStored: 1.0, lowOverUp: 0.99, upOverStored: 1, valid, validDense, lanczosRerun: false, witnessPositiveDefinite: true });
    const rec = (run: string, valid: boolean, validDense: boolean) => ({ run, eta: 0.1, muTrain: 0.001, mismatchAt: null, storedCheckpoints: 1, checkpoints: [{ step: 200, gaussNewton: { ...obj(valid, validDense), removedModesAgree: true }, absH: { ...obj(true, true), nullModesAgree: true } }] });
    const ex = [{ run: 'a', steps: [200] }];
    type Sum = { decision: { validity: string }; gaussNewton: { invalid: number; invalidVsStored: number } };
    const storedOnly = summarizePhase6Stage1([rec('a', false, true)], ex) as Sum;
    expect(storedOnly.decision.validity).toBe('no violation');
    expect([storedOnly.gaussNewton.invalid, storedOnly.gaussNewton.invalidVsStored]).toEqual([0, 1]);
    const denseOnly = summarizePhase6Stage1([rec('a', true, false)], ex) as Sum;
    expect(denseOnly.decision.validity).toBe('violation: method rejected');
    expect([denseOnly.gaussNewton.invalid, denseOnly.gaussNewton.invalidVsStored]).toEqual([1, 0]);
    // The runners' final regeneration check.
    expect(decidePhase6RegenerationMismatch(null, null, null, [[5, 5], [2, 2]])).toBeNull();
    expect(decidePhase6RegenerationMismatch(400, null, null, [[5, 5]])).toBe(400);
    expect(decidePhase6RegenerationMismatch(null, null, 1, [[0, 0]])).toBe(-1);
    expect(decidePhase6RegenerationMismatch(null, 30, null, [[0, 0]])).toBe(30);
    expect(decidePhase6RegenerationMismatch(null, null, null, [[4, 5]])).toBe(-1);
    expect(decidePhase6RegenerationMismatch(null, null, null, [[5, 5], [1, 2]])).toBe(-1);
  });

  it('round 10b: inputs where the B_beta rounding count and the S X S / N bounds decide', () => {
    // Rayleigh: gamma_{n^2} in place of gamma_{n^2 + 16} returned etaUp below the exact quotient of this z (zero noise).
    {
      const lt = Float64Array.of(1.438144148690363, 0.37134585264138964);
      const beta = 0.7795518191158775;
      const z = Float64Array.of(1, 0.24735121030121324, 0.24735121030121324, 0.8526181952117157);
      const one1: Q = [1n, 1n];
      const B = frac(beta);
      const cb = div(B, mul(mul(sub(one1, B), sub(one1, B)), add(one1, B)));
      let num: Q = [0n, 1n];
      let den: Q = [0n, 1n];
      for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) {
        const df = sub(frac(lt[a]!), frac(lt[b]!));
        const d = sub(div(mul(frac(lt[a]!), frac(lt[b]!)), add(one1, B)), mul(cb, mul(df, df)));
        const zz = mul(frac(z[a * 2 + b]!), frac(z[a * 2 + b]!));
        num = add(num, mul(d, zz));
        den = add(den, mul(add(frac(lt[a]!), frac(lt[b]!)), zz));
      }
      const r = computeCertifiedRayleighUpperThreshold(lt, 2, beta, createDenseDeviationNoiseOperator(new Float64Array(4), 2, 1, 1), z)!;
      expect(r.etaUp).not.toBeNull();
      expect(le(div(den, num), frac(r.etaUp!))).toBe(true);
    }
    // Dense: alternating d e0 e0^T and d e1 e1^T, x all ones; the sample term at (0, 1) is exactly 0, so the S X S / N bound alone
    // carries the error of entry (0, 1), exact value -S00 S11 / (B N^2).
    {
      let seed = 4242;
      const rnd = (): number => { seed = (Math.imul(seed, 1103515245) + 12345) >>> 0; return seed / 4294967296; };
      for (let rep = 0; rep < 400; rep++) {
        const N = 2 * (8 + Math.floor(rnd() * 200));
        const d = 0.5 + rnd() * 0.49;
        const dev = new Float64Array(N * 4);
        for (let i = 0; i < N; i++) dev[i * 4 + (i % 2 === 0 ? 0 : 3)] = d;
        const S = mul([BigInt(N / 2), 1n], frac(d));
        const v01 = red([-(S[0] * S[0]), S[1] * S[1] * BigInt(N * N)]);
        const out = new Float64Array(4);
        const err = new Float64Array(4);
        createDenseDeviationNoiseOperator(dev, 2, N, 1).apply(Float64Array.of(1, 1, 1, 1), out, err);
        expect(le(absQ(sub(frac(out[1]!), v01)), frac(err[1]!))).toBe(true);
      }
    }
    // Row space: two samples on disjoint columns (R = 2, dOut up to 1024): the sample term at (0, 1) is 0 and V_01 = -s0 s1 / N^2.
    {
      let seed = 99;
      const rnd = (): number => { seed = (Math.imul(seed, 1103515245) + 12345) >>> 0; return seed / 4294967296; };
      for (let rep = 0; rep < 60; rep++) {
        const dOut = 256 + Math.floor(rnd() * 769);
        const fac = new Float64Array(2 * dOut * 2);
        let s0: Q = [0n, 1n];
        let s1: Q = [0n, 1n];
        for (let c = 0; c < dOut; c++) {
          const a = 0.5 + 0.5 * rnd();
          const b = 0.5 + 0.5 * rnd();
          fac[c * 2] = a;
          fac[dOut * 2 + c * 2 + 1] = b;
          s0 = add(s0, mul(frac(a), frac(a)));
          s1 = add(s1, mul(frac(b), frac(b)));
        }
        const v01 = red([-(s0[0] * s1[0]), s0[1] * s1[1] * 4n]);
        const out = new Float64Array(4);
        const err = new Float64Array(4);
        createRowSpaceFactorNoiseOperator(fac, new Float64Array(2), 2, 2, dOut, 1).apply(Float64Array.of(1, 1, 1, 1), out, err);
        expect(le(absQ(sub(frac(out[1]!), v01)), frac(err[1]!))).toBe(true);
      }
    }
  });

  it('round 11: output tails, the later modes of the dual residual, the runners\' final check, invalid shapes', () => {
    // An output longer than n^2 whose tail overlaps the input: only the validated n^2 entries are written, the input survives.
    for (const make of [() => createDenseDeviationNoiseOperator(Float64Array.of(1, -1), 1, 2, 1), () => createRowSpaceFactorNoiseOperator(Float64Array.of(1, 0), Float64Array.of(0.5), 1, 2, 1, 1)]) {
      const buf = new ArrayBuffer(16);
      const out = new Float64Array(buf);
      const x = new Float64Array(buf, 8, 1);
      x[0] = 1;
      make().apply(x, out, new Float64Array(1));
      expect(x[0]).toBe(1);
      expect(Number.isFinite(out[0]!)).toBe(true);
    }
    // The dual residual is the maximum over every retained mode: the error placed in the second mode only.
    const t = 0.01;
    const res = computeDualEigenResidualMax(Float64Array.of(3, 0, 0, 1), 2, Float64Array.of(3, 1), Float64Array.of(1, 0, Math.sin(t), Math.cos(t)), 2);
    expect(res).toBeGreaterThan(0.019);
    expect(res).toBeLessThan(0.021);
    // The runners' final check is wired: stored records that say 'finite' for a run that diverges at step 1 (committed fixtures).
    expect(runPhase6Stage1Run('tests/fixtures/phase6/stage1-divergent-run.json').mismatchAt).toBe(1);
    expect(runPhase6Stage2Run('p3610-tanh-eta100000000-mu0-seed0', [], 0, 'tests/fixtures/phase6').mismatchAt).toBe(1);
    // Invalid shapes construct without throwing (n = Infinity threw a RangeError during construction); a non-finite dimension (Infinity
    // or NaN; with NaN the outputs were left as they were) throws on every call, a finite invalid one refuses with NaN / Infinity.
    const o = Float64Array.of(123);
    const e = Float64Array.of(456);
    for (const dim of [Infinity, NaN]) {
      expect(() => createDenseDeviationNoiseOperator(new Float64Array(0), dim, 1, 1).apply(Float64Array.of(1), o, e)).toThrow(TypeError);
      expect(() => createRowSpaceFactorNoiseOperator(new Float64Array(0), new Float64Array(0), dim, 1, 1, 1).apply(Float64Array.of(1), o, e)).toThrow(TypeError);
    }
    createRowSpaceFactorNoiseOperator(new Float64Array(0), new Float64Array(1), 1, 1, Infinity, 1).apply(Float64Array.of(1), o, e);
    expect([Number.isNaN(o[0]!), e[0]]).toEqual([true, Infinity]);
    createDenseDeviationNoiseOperator(Float64Array.of(1, -1), 1, 2, 2 ** 25).apply(Float64Array.of(1), o, e);
    expect([Number.isNaN(o[0]!), e[0]]).toEqual([true, Infinity]);
    // The |w| / (1 - beta) part of the absolute B_beta sum (audit, round 11): n = 4, lt = 2^-60, beta = 0, exact noise
    // V(X)_ab = d_a d_b X_ab with d = (1, -1, 1, -1), z with z_01 chosen so that the B form cancels; without that part the bound fell
    // below the exact quotient of its own z on 385 of 812 matrices. Every finite etaUp must be at least the exact quotient.
    {
      const n = 4;
      const d = [1, -1, 1, -1];
      // V(X) = D X D with D = diag(d): library dense operator with deviations D and -D, N = 2, B = 1.
      const devD = new Float64Array(2 * n * n);
      for (let a = 0; a < n; a++) { devD[a * n + a] = d[a]!; devD[n * n + a * n + a] = -d[a]!; }
      const noise = createDenseDeviationNoiseOperator(devD, n, 2, 1);
      let informative = 0;
      const lt = new Float64Array(n).fill(2 ** -60);
      let seed = 31337;
      const rnd = (): number => { seed = (Math.imul(seed, 1103515245) + 12345) >>> 0; return seed / 4294967296; };
      let tried = 0;
      for (let rep = 0; rep < 2000; rep++) {
        const z = new Float64Array(n * n);
        for (let a = 0; a < n; a++) for (let b = a; b < n; b++) { const v = (0.5 + 0.5 * rnd()) * (rnd() < 0.5 ? -1 : 1); z[a * n + b] = v; z[b * n + a] = v; }
        z[0] = 1;
        let same = 0;
        let other = 0;
        for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) { if ((a === 0 && b === 1) || (a === 1 && b === 0)) continue; const q = z[a * n + b]! * z[a * n + b]!; if (d[a]! * d[b]! > 0) same += q; else other += q; }
        const t2 = (same - other) / 2;
        if (!(t2 > 0) || !(t2 < 1)) continue;
        z[1] = Math.sqrt(t2);
        z[n] = z[1]!;
        tried++;
        const r = computeCertifiedRayleighUpperThreshold(lt, n, 0, noise, z);
        if (r === null || r.etaUp === null) continue;
        let num: Q = [0n, 1n];
        let den: Q = [0n, 1n];
        const L = frac(2 ** -60);
        for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
          const zz = mul(frac(z[a * n + b]!), frac(z[a * n + b]!));
          num = add(num, mul(zz, add(mul(L, L), [BigInt(d[a]! * d[b]!), 1n])));
          den = add(den, mul(zz, mul([2n, 1n], L)));
        }
        if (!(num[0] > 0n)) continue;
        expect(le(div(den, num), frac(r.etaUp))).toBe(true);
        informative++;
      }
      expect(tried).toBeGreaterThan(500);
      expect(informative).toBe(0);
    }
  });

  it('round 11b/13: only library noise operators are accepted; callbacks of any kind are refused before they are called', () => {
    type Noise = Parameters<typeof computeCertifiedThresholdBracket>[3];
    let calls = 0;
    const writer = { n: 1, apply(x: Float64Array, out: Float64Array, err: Float64Array | null): void { calls++; out[0] = x[0]!; if (err !== null) err[0] = 0; x[0] = 0.5; } };
    const honestCallback = { n: 2, apply(x: Float64Array, out: Float64Array, err: Float64Array | null): void { calls++; out.fill(0); out[0] = 4 * x[0]!; if (err !== null) err.fill(0); } };
    const lib1 = createDenseDeviationNoiseOperator(Float64Array.of(1, -1), 1, 2, 1);
    // A writer (round 11), an honest-looking callback, any function (it inherits Function.prototype.apply; round 13), the factory itself,
    // a copy or a prototype child of a library operator, null and primitives.
    for (const bad of [writer, honestCallback, () => 0, createDenseDeviationNoiseOperator, { ...lib1 }, Object.create(lib1) as object, null, 3, 'x']) {
      expect(computeCertifiedThresholdBracket(Float64Array.of(1), 1, 0, bad as unknown as Noise)).toBeNull();
      expect(computeCertifiedRayleighUpperThreshold(Float64Array.of(1), 1, 0, bad as unknown as Noise, Float64Array.of(1))).toBeNull();
    }
    expect(calls).toBe(0);
    expect(Object.isFrozen(lib1)).toBe(true);
    // The library form of the honest callback: V(X) = D X D, D = diag(2, 0) (deviations D and -D, N = 2, B = 1), so V(X)_00 = 4 X_00 and
    // eta* = 2/5 exactly (lt = 1, beta = 0).
    const op = createDenseDeviationNoiseOperator(Float64Array.of(2, 0, 0, 0, -2, 0, 0, 0), 2, 2, 1);
    const ok = computeCertifiedThresholdBracket(Float64Array.of(1, 1), 2, 0, op)!;
    expect(ok.etaLow!).toBeLessThan(0.4);
    expect(ok.etaUp).toBeGreaterThanOrEqual(0.4);
    expect(computeCertifiedThresholdBracket(Float64Array.of(1, 1, 1), 3, 0, op)).toBeNull();
  });

  it('round 12: the dense oracle centres a common offset exactly, apply is read once, a zero sign change is a refusal', () => {
    // Six deviations equal to 2^60: the centred noise is exactly zero and eta* = 2 (the oracle returned 2 / 16385).
    expect(computeEigenbasisThresholdDenseReference(Float64Array.of(1), 1, 0, new Float64Array(6).fill(2 ** 60), 6, 1)).toBe(2);
    // Caller-made objects with a getter for apply, or an apply that flips a zero's sign, are refused before any property is read.
    let reads = 0;
    const base = createDenseDeviationNoiseOperator(Float64Array.of(1, -1), 1, 2, 1);
    const getterNoise = { n: 1, get apply() { reads++; return base.apply; } };
    expect(computeCertifiedThresholdBracket(Float64Array.of(1), 1, 0, getterNoise)).toBeNull();
    expect(reads).toBe(0);
  });
});
