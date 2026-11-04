/**
 * @file labs/mean-square-boundary/create-phase6-student-network.ts
 * @description The student network of Phase 6 on given data (`createPhase6StudentNetwork`):
 * `f(x) = W_2 phi(W_1 x + b_1) + b_2` with the parameter layout of `createScaleMlpDatasetAndOps` (W1 row-major
 * `dHidden x dIn` at 0, b1 at `b1Base`, W2 row-major `dOut x dHidden` at `w2Base`, b2 at `b2Base`). `phi` is `fdlibmTanh`
 * (`phi' = 1 - tanh^2`, `phi'' = -2 tanh (1 - tanh^2)`, the formulas of the scale network), the leaky ReLU of slope 0.1
 * (`leakyRelu01`: `phi(z) = z` for `z >= 0`, `0.1 z` below, `phi' = 1` or `0.1`, `phi'' = 0`; the kink at 0 takes the right
 * derivative), or `softsign` (`phi(z) = z / (1 + |z|)`, `phi'(z) = 1 / (1 + |z|)^2`, `phi''(z) = -2 sgn(z) / (1 + |z|)^3` for
 * `z != 0` and `0` at `z = 0`, limits `+-1` and `0` derivatives at `+-Infinity`; a NaN pre-activation gives NaN derivatives in
 * all three arms).
 *
 * Activation audit for separating bounded output from fast-vanishing derivatives (`s5`): no scalar activation on `R` can be
 * bounded while keeping `phi'` bounded away from zero everywhere (`integral phi' dz` would diverge), so separation is partial.
 * Among the four candidates: (1) a feature-norm-normalised leaky ReLU couples all `dHidden` units in the Jacobian and Hessian
 * (breaking per-unit hidden statistics and the additive layerwise Gauss–Newton trace) and introduces positive degree-0 radial
 * scale invariance in `(W_1, b_1)` at large norm, adding a normalisation-layer step-size confound; (2) a bounded smooth activation
 * with a derivative floor `c` on `[-M, M]` and `phi'(0) = 1`, `|phi| <= 1` obeys `integral_0^M phi' <= 1` and adds two arbitrary
 * cutoffs `(M, c)` whose outcome depends on whether pre-activations stay inside `[-M, M]`; (3) the algebraic sigmoid
 * `z / sqrt(1 + z^2)` is `C^infinity` and parameter-free with `phi(0) = 0`, `phi'(0) = 1`, `|phi| < 1`, but `phi'(z) = (1 + z^2)^(-3/2)`
 * decays as `|z|^(-3)` (`phi'^2 ~ |z|^(-6)` in the first-layer Gauss–Newton trace, already below `1e-4` at `|z| = 4.54`, and at
 * equal output amplitude `|a| -> 1` its derivative `(1 - a^2)^(3/2)` is smaller than `tanh`'s `1 - a^2`); (4) `softsign`
 * `z / (1 + |z|)` is odd, monotone, non-homogeneous and parameter-free with `phi(0) = 0`, `phi'(0) = 1`, `sup |phi| = 1` (sharing
 * the exact output-layer Gauss–Newton ceiling `dOut (1 + dHidden)` with `tanh`), and has the slowest integer-power polynomial
 * derivative decay `phi'(z) = (1 + |z|)^(-2) ~ |z|^(-2)` (`phi'^2 ~ |z|^(-4)`: at `|z| = 10`, `phi'^2 = 6.83e-5` vs `9.71e-7` for
 * `z / sqrt(1 + z^2)` and `6.80e-17` for `tanh`; at `|z| = 20`, `tanh` rounds to `+-1` in float64 with `phi' = 0` while `softsign`
 * keeps `phi' = 2.27e-3`). Its tradeoff is `C^1` regularity at `z = 0` (`phi''` jumps from `+2` to `-2` across `0`) and that as a
 * function of `|a| = |phi(z)|`, `phi' = (1 - |a|)^2`, so the separation tests polynomial against exponential decay in weight /
 * pre-activation space `|z|`. Loss
 * `mean_i sum_c (f_c(x_i) - y_ic)^2 / 2`, data only (callers add weight decay). Per sample: the gradient of
 * `sum_c (f_c - y_c)^2 / 2` (not divided by `N`), the exact Hessian-vector product (Pearlmutter, forward over reverse), its
 * Gauss–Newton part `J_i^T J_i v`, and the output Jacobian J_i (`dOut x P` row-major). `computeExactTraces` gives in `O(N P)`
 * `tr H = tr H_GN + tr H_res`: `tr H_GN = mean_i ||J_i||_F^2`, split into the output layer `dOut (1 + sum_h phi(z_ih)^2)` and the
 * first layer `(1 + ||x_i||^2) sum_h phi'(z_ih)^2 ||W_2[:, h]||^2`, and
 * `tr H_res = mean_i (1 + ||x_i||^2) sum_h phi''(z_ih) sum_c W_2[c, h] e_ic` with `e_ic = f_c(x_i) - y_ic` (the diagonal second derivatives of f_c in W2 and b2 are zero), and
 * `traceGaussNewtonFirstLayerWithoutInputNorm = mean_i sum_h phi'(z_ih)^2 ||W_2[:, h]||^2`, the first-layer trace without the factor
 * `1 + ||x_i||^2` (about 104 at `dIn = 64`, 415 at 256), comparable across input widths.
 * Every method recomputes the forward pass of the samples it reads, so no result depends on an earlier call; an output buffer must
 * not alias `theta` or `v`. `evalLossOn` and `computeHiddenActivationsOn` read another dataset (the held-out set). `getInputBasis`
 * caches `buildInputSpanBasis` on `xData`.
 * Exactness contract: `exact` (bit-identical to `oracleCreatePhase6StudentNetwork`; 4-sample blocking in `evalLossOn`,
 * `computeMinibatchGradient`, and `computeFullGradient` preserves every IEEE-754 summation order).
 * @cold
 */
import { fdlibmTanh } from '../../src/math/index.ts';
import { buildInputSpanBasis } from './build-input-span-basis.ts';

function evalActOnly(z: number, leaky: boolean, softsign: boolean): number {
  if (leaky) return z >= 0.0 ? z : 0.1 * z;
  if (softsign) {
    if (!Number.isFinite(z)) return z !== z ? Number.NaN : z > 0.0 ? 1.0 : -1.0;
    return z / (1.0 + Math.abs(z));
  }
  return fdlibmTanh(z);
}

function evalActAndD1(z: number, leaky: boolean, softsign: boolean, outAct: Float64Array, outD1: Float64Array, slot: number): void {
  if (leaky) {
    const pos = z >= 0.0;
    outAct[slot] = pos ? z : 0.1 * z;
    outD1[slot] = z !== z ? Number.NaN : pos ? 1.0 : 0.1;
  } else if (softsign) {
    if (!Number.isFinite(z)) {
      const nan = z !== z;
      outAct[slot] = nan ? Number.NaN : z > 0.0 ? 1.0 : -1.0;
      outD1[slot] = nan ? Number.NaN : 0.0;
    } else {
      const denom = 1.0 + Math.abs(z);
      const inv = 1.0 / denom;
      outAct[slot] = z / denom;
      outD1[slot] = inv * inv;
    }
  } else {
    const a = fdlibmTanh(z);
    outAct[slot] = a;
    outD1[slot] = 1.0 - a * a;
  }
}

export function createPhase6StudentNetwork(
  xData: Float64Array,
  yData: Float64Array,
  dIn: number,
  dHidden: number,
  dOut: number,
  numSamples: number,
  activation: 'tanh' | 'leakyRelu01' | 'softsign',
): {
  readonly dIn: number;
  readonly dHidden: number;
  readonly dOut: number;
  readonly numParams: number;
  readonly numSamples: number;
  readonly activation: 'tanh' | 'leakyRelu01' | 'softsign';
  readonly b1Base: number;
  readonly w2Base: number;
  readonly b2Base: number;
  readonly xData: Float64Array;
  readonly yData: Float64Array;
  getInputBasis(): { readonly basis: Float64Array; readonly size: number; readonly dim: number };
  evalLoss(theta: Float64Array): number;
  evalLossOn(theta: Float64Array, x: Float64Array, y: Float64Array, n: number): number;
  computeFullGradient(theta: Float64Array, out: Float64Array): number;
  computeSampleGradient(theta: Float64Array, i: number, out: Float64Array): number;
  computeMinibatchGradient(theta: Float64Array, idx: Int32Array, batchSize: number, out: Float64Array): number;
  computeSampleHvp(theta: Float64Array, v: Float64Array, i: number, out: Float64Array): void;
  computeSampleGaussNewtonHvp(theta: Float64Array, v: Float64Array, i: number, out: Float64Array): void;
  computeFullHvp(theta: Float64Array, v: Float64Array, out: Float64Array): void;
  computeSampleOutputJacobian(theta: Float64Array, i: number, out: Float64Array): void;
  computeHiddenActivations(theta: Float64Array, i: number, outAct: Float64Array, outDeriv: Float64Array): void;
  computeHiddenActivationsOn(theta: Float64Array, x: Float64Array, i: number, outAct: Float64Array, outDeriv: Float64Array): void;
  computeExactTraces(theta: Float64Array): {
    readonly traceH: number;
    readonly traceGaussNewton: number;
    readonly traceGaussNewtonOutputLayer: number;
    readonly traceGaussNewtonFirstLayer: number;
    readonly traceGaussNewtonFirstLayerWithoutInputNorm: number;
    readonly traceResidual: number;
  };
} {
  const b1Base = Math.imul(dHidden, dIn);
  const w2Base = (b1Base + dHidden) | 0;
  const b2Base = (w2Base + Math.imul(dOut, dHidden)) | 0;
  const numParams = (b2Base + dOut) | 0;
  const leaky = activation === 'leakyRelu01';
  const softsign = activation === 'softsign';
  const act = new Float64Array(dHidden);
  const d1 = new Float64Array(dHidden);
  const d2 = new Float64Array(dHidden);
  const rz = new Float64Array(dHidden);
  const pred = new Float64Array(dOut);
  const err = new Float64Array(dOut);
  const rf = new Float64Array(dOut);
  const act4 = new Float64Array(Math.imul(4, dHidden));
  const d14 = new Float64Array(Math.imul(4, dHidden));
  const pred4 = new Float64Array(Math.imul(4, dOut));
  const err4 = new Float64Array(Math.imul(4, dOut));
  let cachedInputBasis: { readonly basis: Float64Array; readonly size: number; readonly dim: number } | null = null;

  function forward(theta: Float64Array, x: Float64Array, xOff: number): void {
    for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
      const row = Math.imul(h, dIn);
      let z = theta[(b1Base + h) | 0]!;
      for (let j = 0; j < dIn; (j = (j + 1) | 0)) z += theta[(row + j) | 0]! * x[(xOff + j) | 0]!;
      if (leaky) {
        const pos = z >= 0.0;
        const nan = z !== z;
        act[h] = pos ? z : 0.1 * z;
        d1[h] = nan ? Number.NaN : pos ? 1.0 : 0.1;
        d2[h] = nan ? Number.NaN : 0.0;
      } else if (softsign) {
        if (!Number.isFinite(z)) {
          const nan = z !== z;
          act[h] = nan ? Number.NaN : z > 0.0 ? 1.0 : -1.0;
          d1[h] = nan ? Number.NaN : 0.0;
          d2[h] = nan ? Number.NaN : 0.0;
        } else {
          const denom = 1.0 + Math.abs(z);
          const inv = 1.0 / denom;
          const d = inv * inv;
          act[h] = z / denom;
          d1[h] = d;
          d2[h] = z > 0.0 ? -2.0 * d * inv : z < 0.0 ? 2.0 * d * inv : 0.0;
        }
      } else {
        const a = fdlibmTanh(z);
        const d = 1.0 - a * a;
        act[h] = a;
        d1[h] = d;
        d2[h] = -2.0 * a * d;
      }
    }
    for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
      const row = (w2Base + Math.imul(c, dHidden)) | 0;
      let s = theta[(b2Base + c) | 0]!;
      for (let h = 0; h < dHidden; (h = (h + 1) | 0)) s += theta[(row + h) | 0]! * act[h]!;
      pred[c] = s;
    }
  }

  function evalLossOn(theta: Float64Array, x: Float64Array, y: Float64Array, n: number): number {
    let total = 0.0;
    let i = 0;
    const h1 = dHidden;
    const h2 = Math.imul(2, dHidden);
    const h3 = Math.imul(3, dHidden);
    for (; ((i + 3) | 0) < n; (i = (i + 4) | 0)) {
      const x0 = Math.imul(i, dIn);
      const x1 = (x0 + dIn) | 0;
      const x2 = (x1 + dIn) | 0;
      const x3 = (x2 + dIn) | 0;
      for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
        const row = Math.imul(h, dIn);
        const b = theta[(b1Base + h) | 0]!;
        let z0 = b;
        let z1 = b;
        let z2 = b;
        let z3 = b;
        for (let j = 0; j < dIn; (j = (j + 1) | 0)) {
          const w = theta[(row + j) | 0]!;
          z0 += w * x[(x0 + j) | 0]!;
          z1 += w * x[(x1 + j) | 0]!;
          z2 += w * x[(x2 + j) | 0]!;
          z3 += w * x[(x3 + j) | 0]!;
        }
        act4[h] = evalActOnly(z0, leaky, softsign);
        act4[(h1 + h) | 0] = evalActOnly(z1, leaky, softsign);
        act4[(h2 + h) | 0] = evalActOnly(z2, leaky, softsign);
        act4[(h3 + h) | 0] = evalActOnly(z3, leaky, softsign);
      }
      for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
        const row = (w2Base + Math.imul(c, dHidden)) | 0;
        const b2 = theta[(b2Base + c) | 0]!;
        let s0 = b2;
        let s1 = b2;
        let s2 = b2;
        let s3 = b2;
        for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
          const w = theta[(row + h) | 0]!;
          s0 += w * act4[h]!;
          s1 += w * act4[(h1 + h) | 0]!;
          s2 += w * act4[(h2 + h) | 0]!;
          s3 += w * act4[(h3 + h) | 0]!;
        }
        pred4[c] = s0;
        pred4[(dOut + c) | 0] = s1;
        pred4[(Math.imul(2, dOut) + c) | 0] = s2;
        pred4[(Math.imul(3, dOut) + c) | 0] = s3;
      }
      for (let s = 0; s < 4; (s = (s + 1) | 0)) {
        const pOff = Math.imul(s, dOut);
        const yOff = Math.imul((i + s) | 0, dOut);
        for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
          const e = pred4[(pOff + c) | 0]! - y[(yOff + c) | 0]!;
          total += 0.5 * e * e;
        }
      }
    }
    for (; i < n; (i = (i + 1) | 0)) {
      const xOff = Math.imul(i, dIn);
      for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
        const row = Math.imul(h, dIn);
        let z = theta[(b1Base + h) | 0]!;
        for (let j = 0; j < dIn; (j = (j + 1) | 0)) z += theta[(row + j) | 0]! * x[(xOff + j) | 0]!;
        act[h] = evalActOnly(z, leaky, softsign);
      }
      const yOff = Math.imul(i, dOut);
      for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
        const row = (w2Base + Math.imul(c, dHidden)) | 0;
        let s = theta[(b2Base + c) | 0]!;
        for (let h = 0; h < dHidden; (h = (h + 1) | 0)) s += theta[(row + h) | 0]! * act[h]!;
        const e = s - y[(yOff + c) | 0]!;
        total += 0.5 * e * e;
      }
    }
    return total / n;
  }

  function accumulateGradient(theta: Float64Array, i: number, scale: number, out: Float64Array): number {
    const xOff = Math.imul(i, dIn);
    const yOff = Math.imul(i, dOut);
    forward(theta, xData, xOff);
    let loss = 0.0;
    for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
      const e = pred[c]! - yData[(yOff + c) | 0]!;
      loss += 0.5 * e * e;
      err[c] = e;
      const se = scale * e;
      out[(b2Base + c) | 0] = out[(b2Base + c) | 0]! + se;
      const row = (w2Base + Math.imul(c, dHidden)) | 0;
      for (let h = 0; h < dHidden; (h = (h + 1) | 0)) out[(row + h) | 0] = out[(row + h) | 0]! + se * act[h]!;
    }
    for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
      let back = 0.0;
      for (let c = 0; c < dOut; (c = (c + 1) | 0)) back += theta[(w2Base + Math.imul(c, dHidden) + h) | 0]! * err[c]!;
      const delta = scale * back * d1[h]!;
      out[(b1Base + h) | 0] = out[(b1Base + h) | 0]! + delta;
      const row = Math.imul(h, dIn);
      for (let j = 0; j < dIn; (j = (j + 1) | 0)) out[(row + j) | 0] = out[(row + j) | 0]! + delta * xData[(xOff + j) | 0]!;
    }
    return loss;
  }

  // 4-sample blocked gradient accumulation preserving exact IEEE-754 summation order (`0 ULP` vs 4 calls to `accumulateGradient`).
  function accumulateGradient4(theta: Float64Array, i0: number, i1: number, i2: number, i3: number, scale: number, lossSoFar: number, out: Float64Array): number {
    const x0 = Math.imul(i0, dIn);
    const x1 = Math.imul(i1, dIn);
    const x2 = Math.imul(i2, dIn);
    const x3 = Math.imul(i3, dIn);
    const h1 = dHidden;
    const h2 = Math.imul(2, dHidden);
    const h3 = Math.imul(3, dHidden);
    for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
      const row = Math.imul(h, dIn);
      const b = theta[(b1Base + h) | 0]!;
      let z0 = b;
      let z1 = b;
      let z2 = b;
      let z3 = b;
      for (let j = 0; j < dIn; (j = (j + 1) | 0)) {
        const w = theta[(row + j) | 0]!;
        z0 += w * xData[(x0 + j) | 0]!;
        z1 += w * xData[(x1 + j) | 0]!;
        z2 += w * xData[(x2 + j) | 0]!;
        z3 += w * xData[(x3 + j) | 0]!;
      }
      evalActAndD1(z0, leaky, softsign, act4, d14, h);
      evalActAndD1(z1, leaky, softsign, act4, d14, (h1 + h) | 0);
      evalActAndD1(z2, leaky, softsign, act4, d14, (h2 + h) | 0);
      evalActAndD1(z3, leaky, softsign, act4, d14, (h3 + h) | 0);
    }
    let lossAcc = lossSoFar;
    for (let s = 0; s < 4; (s = (s + 1) | 0)) {
      const sampleIdx = s === 0 ? i0 : s === 1 ? i1 : s === 2 ? i2 : i3;
      const yOff = Math.imul(sampleIdx, dOut);
      const aOff = Math.imul(s, dHidden);
      const eOff = Math.imul(s, dOut);
      let sampleLoss = 0.0;
      for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
        const row = (w2Base + Math.imul(c, dHidden)) | 0;
        let predC = theta[(b2Base + c) | 0]!;
        for (let h = 0; h < dHidden; (h = (h + 1) | 0)) predC += theta[(row + h) | 0]! * act4[(aOff + h) | 0]!;
        const e = predC - yData[(yOff + c) | 0]!;
        sampleLoss += 0.5 * e * e;
        err4[(eOff + c) | 0] = e;
        const se = scale * e;
        out[(b2Base + c) | 0] = out[(b2Base + c) | 0]! + se;
        for (let h = 0; h < dHidden; (h = (h + 1) | 0)) out[(row + h) | 0] = out[(row + h) | 0]! + se * act4[(aOff + h) | 0]!;
      }
      lossAcc += sampleLoss;
    }
    const e1 = dOut;
    const e2 = Math.imul(2, dOut);
    const e3 = Math.imul(3, dOut);
    for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
      let bk0 = 0.0;
      let bk1 = 0.0;
      let bk2 = 0.0;
      let bk3 = 0.0;
      for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
        const w = theta[(w2Base + Math.imul(c, dHidden) + h) | 0]!;
        bk0 += w * err4[c]!;
        bk1 += w * err4[(e1 + c) | 0]!;
        bk2 += w * err4[(e2 + c) | 0]!;
        bk3 += w * err4[(e3 + c) | 0]!;
      }
      const del0 = scale * bk0 * d14[h]!;
      const del1 = scale * bk1 * d14[(h1 + h) | 0]!;
      const del2 = scale * bk2 * d14[(h2 + h) | 0]!;
      const del3 = scale * bk3 * d14[(h3 + h) | 0]!;
      const bIdx = (b1Base + h) | 0;
      out[bIdx] = out[bIdx]! + del0 + del1 + del2 + del3;
      const row = Math.imul(h, dIn);
      for (let j = 0; j < dIn; (j = (j + 1) | 0)) {
        out[(row + j) | 0] =
          out[(row + j) | 0]! + del0 * xData[(x0 + j) | 0]! + del1 * xData[(x1 + j) | 0]! + del2 * xData[(x2 + j) | 0]! + del3 * xData[(x3 + j) | 0]!;
      }
    }
    return lossAcc;
  }

  function accumulateHvp(theta: Float64Array, v: Float64Array, i: number, scale: number, gaussNewton: boolean, out: Float64Array): void {
    const xOff = Math.imul(i, dIn);
    const yOff = Math.imul(i, dOut);
    forward(theta, xData, xOff);
    for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
      const row = Math.imul(h, dIn);
      let s = v[(b1Base + h) | 0]!;
      for (let j = 0; j < dIn; (j = (j + 1) | 0)) s += v[(row + j) | 0]! * xData[(xOff + j) | 0]!;
      rz[h] = s;
    }
    // The residual terms (those read from `e`, and `phi''` for tanh) are skipped by branching, never multiplied by 0, so a
    // non-finite residual or an overflowed product cannot turn the Gauss–Newton part into NaN through 0 * Infinity.
    for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
      const row = (w2Base + Math.imul(c, dHidden)) | 0;
      let s = v[(b2Base + c) | 0]!;
      for (let h = 0; h < dHidden; (h = (h + 1) | 0)) s += v[(row + h) | 0]! * act[h]! + theta[(row + h) | 0]! * d1[h]! * rz[h]!;
      rf[c] = s;
      out[(b2Base + c) | 0] = out[(b2Base + c) | 0]! + scale * s;
      if (gaussNewton) {
        for (let h = 0; h < dHidden; (h = (h + 1) | 0)) out[(row + h) | 0] = out[(row + h) | 0]! + scale * (s * act[h]!);
        continue;
      }
      const e = pred[c]! - yData[(yOff + c) | 0]!;
      err[c] = e;
      for (let h = 0; h < dHidden; (h = (h + 1) | 0)) out[(row + h) | 0] = out[(row + h) | 0]! + scale * (s * act[h]! + e * d1[h]! * rz[h]!);
    }
    for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
      let backRf = 0.0;
      for (let c = 0; c < dOut; (c = (c + 1) | 0)) backRf += theta[(w2Base + Math.imul(c, dHidden) + h) | 0]! * rf[c]!;
      let rDelta: number;
      if (gaussNewton) {
        rDelta = scale * (d1[h]! * backRf);
      } else {
        let backE = 0.0;
        let backVe = 0.0;
        for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
          const idx = (w2Base + Math.imul(c, dHidden) + h) | 0;
          backE += theta[idx]! * err[c]!;
          backVe += v[idx]! * err[c]!;
        }
        rDelta = leaky ? scale * (d1[h]! * (backRf + backVe)) : scale * (d1[h]! * (backRf + backVe) + d2[h]! * rz[h]! * backE);
      }
      out[(b1Base + h) | 0] = out[(b1Base + h) | 0]! + rDelta;
      const row = Math.imul(h, dIn);
      for (let j = 0; j < dIn; (j = (j + 1) | 0)) out[(row + j) | 0] = out[(row + j) | 0]! + rDelta * xData[(xOff + j) | 0]!;
    }
  }

  function computeSampleOutputJacobian(theta: Float64Array, i: number, out: Float64Array): void {
    const xOff = Math.imul(i, dIn);
    forward(theta, xData, xOff);
    out.fill(0.0);
    for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
      const base = Math.imul(c, numParams);
      const w2Row = (w2Base + Math.imul(c, dHidden)) | 0;
      out[(base + b2Base + c) | 0] = 1.0;
      for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
        out[(base + w2Row + h) | 0] = act[h]!;
        const dz = theta[(w2Row + h) | 0]! * d1[h]!;
        out[(base + b1Base + h) | 0] = dz;
        const row = Math.imul(h, dIn);
        for (let j = 0; j < dIn; (j = (j + 1) | 0)) out[(base + row + j) | 0] = dz * xData[(xOff + j) | 0]!;
      }
    }
  }

  function computeExactTraces(theta: Float64Array): {
    readonly traceH: number;
    readonly traceGaussNewton: number;
    readonly traceGaussNewtonOutputLayer: number;
    readonly traceGaussNewtonFirstLayer: number;
    readonly traceGaussNewtonFirstLayerWithoutInputNorm: number;
    readonly traceResidual: number;
  } {
    const colSq = new Float64Array(dHidden);
    for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
      for (let h = 0; h < dHidden; (h = (h + 1) | 0)) { const w = theta[(w2Base + Math.imul(c, dHidden) + h) | 0]!; colSq[h] = colSq[h]! + w * w; }
    }
    let outLayer = 0.0;
    let firstLayer = 0.0;
    let firstNoNorm = 0.0;
    let res = 0.0;
    for (let i = 0; i < numSamples; (i = (i + 1) | 0)) {
      const xOff = Math.imul(i, dIn);
      const yOff = Math.imul(i, dOut);
      forward(theta, xData, xOff);
      let xSq = 1.0;
      for (let j = 0; j < dIn; (j = (j + 1) | 0)) { const x = xData[(xOff + j) | 0]!; xSq += x * x; }
      for (let c = 0; c < dOut; (c = (c + 1) | 0)) err[c] = pred[c]! - yData[(yOff + c) | 0]!;
      let actSq = 1.0;
      let first = 0.0;
      let curv = 0.0;
      for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
        actSq += act[h]! * act[h]!;
        first += d1[h]! * d1[h]! * colSq[h]!;
        let back = 0.0;
        for (let c = 0; c < dOut; (c = (c + 1) | 0)) back += theta[(w2Base + Math.imul(c, dHidden) + h) | 0]! * err[c]!;
        curv += d2[h]! * back;
      }
      outLayer += dOut * actSq;
      firstLayer += xSq * first;
      firstNoNorm += first;
      res += xSq * curv;
    }
    const inv = 1.0 / numSamples;
    const gn = (outLayer + firstLayer) * inv;
    return { traceH: gn + res * inv, traceGaussNewton: gn, traceGaussNewtonOutputLayer: outLayer * inv, traceGaussNewtonFirstLayer: firstLayer * inv, traceGaussNewtonFirstLayerWithoutInputNorm: firstNoNorm * inv, traceResidual: res * inv };
  }

  return {
    dIn,
    dHidden,
    dOut,
    numParams,
    numSamples,
    activation,
    b1Base,
    w2Base,
    b2Base,
    xData,
    yData,
    getInputBasis: () => {
      if (cachedInputBasis === null) cachedInputBasis = buildInputSpanBasis(xData, numSamples, dIn);
      return cachedInputBasis;
    },
    evalLoss: (theta) => evalLossOn(theta, xData, yData, numSamples),
    evalLossOn,
    computeFullGradient: (theta, out) => {
      out.fill(0.0);
      const scale = 1.0 / numSamples;
      let loss = 0.0;
      let i = 0;
      for (; ((i + 3) | 0) < numSamples; (i = (i + 4) | 0)) {
        loss = accumulateGradient4(theta, i, (i + 1) | 0, (i + 2) | 0, (i + 3) | 0, scale, loss, out);
      }
      for (; i < numSamples; (i = (i + 1) | 0)) loss += accumulateGradient(theta, i, scale, out);
      return loss / numSamples;
    },
    computeSampleGradient: (theta, i, out) => {
      out.fill(0.0);
      return accumulateGradient(theta, i, 1.0, out);
    },
    computeMinibatchGradient: (theta, idx, batchSize, out) => {
      out.fill(0.0);
      const scale = 1.0 / batchSize;
      let loss = 0.0;
      let b = 0;
      for (; ((b + 3) | 0) < batchSize; (b = (b + 4) | 0)) {
        loss = accumulateGradient4(theta, idx[b]!, idx[(b + 1) | 0]!, idx[(b + 2) | 0]!, idx[(b + 3) | 0]!, scale, loss, out);
      }
      for (; b < batchSize; (b = (b + 1) | 0)) loss += accumulateGradient(theta, idx[b]!, scale, out);
      return loss / batchSize;
    },
    computeSampleHvp: (theta, v, i, out) => {
      out.fill(0.0);
      accumulateHvp(theta, v, i, 1.0, false, out);
    },
    computeSampleGaussNewtonHvp: (theta, v, i, out) => {
      out.fill(0.0);
      accumulateHvp(theta, v, i, 1.0, true, out);
    },
    computeFullHvp: (theta, v, out) => {
      out.fill(0.0);
      for (let i = 0; i < numSamples; (i = (i + 1) | 0)) accumulateHvp(theta, v, i, 1.0 / numSamples, false, out);
    },
    computeSampleOutputJacobian,
    computeHiddenActivations: (theta, i, outAct, outDeriv) => {
      forward(theta, xData, Math.imul(i, dIn));
      for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
        outAct[h] = act[h]!;
        outDeriv[h] = d1[h]!;
      }
    },
    computeHiddenActivationsOn: (theta, x, i, outAct, outDeriv) => {
      forward(theta, x, Math.imul(i, dIn));
      for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
        outAct[h] = act[h]!;
        outDeriv[h] = d1[h]!;
      }
    },
    computeExactTraces,
  };
}
