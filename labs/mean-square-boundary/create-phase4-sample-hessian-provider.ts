/**
 * @file labs/mean-square-boundary/create-phase4-sample-hessian-provider.ts
 * @description Per-sample Hessians of a network checkpoint (`createPhase4SampleHessianProvider`). For the scale networks of
 * lab Phase 4 they are the exact Pearlmutter products `computeSampleHvp` and their mean `computeAnalyticalHvp`. The `P = 33`
 * network has no second-order code, so its per-sample Hessians are built once as dense matrices from central differences of
 * the per-sample gradient (`eps = 1e-5`, the step of `hessian-ground-truth.ts`) and symmetrised; the linear system of Phase
 * 13 is then defined by these matrices, and the mean is their average in sample order. A `quadratic` spec takes the
 * per-sample Hessians as given (tests); its gradient is the linear system itself. Deterministic: a worker that builds the
 * same spec gets the same bits. An optional `theta` evaluates the Hessians there instead of at the checkpoint.
 * @cold
 */
import { createTwoLayerMlpDatasetAndOps } from './mlp-network.ts';
import type { Phase4NetworkSpec, Phase4SampleHessianProvider } from './phase4-types.ts';
import { createScaleMlpDatasetAndOps } from './scale-mlp-network.ts';

export function createPhase4SampleHessianProvider(spec: Phase4NetworkSpec): Phase4SampleHessianProvider {
  if (spec.kind === 'scale') {
    const mlp = createScaleMlpDatasetAndOps(spec.dIn, 48, 10, 64, spec.seedLo, spec.seedHi);
    const theta0 = spec.theta ?? mlp.checkpointTheta0;
    if (spec.gaussNewton === true) {
      const tmpGn = new Float64Array(mlp.numParams);
      const invN = 1.0 / mlp.numSamples;
      return {
        numParams: mlp.numParams,
        numSamples: mlp.numSamples,
        checkpointTheta0: theta0,
        applySampleHessian: (i, v, out) => mlp.computeSampleGaussNewtonHvp(theta0, v, i, out),
        applyMeanHessian: (v, out) => {
          out.fill(0.0);
          for (let i = 0; i < mlp.numSamples; (i = (i + 1) | 0)) {
            mlp.computeSampleGaussNewtonHvp(theta0, v, i, tmpGn);
            for (let a = 0; a < mlp.numParams; (a = (a + 1) | 0)) out[a] = out[a]! + invN * tmpGn[a]!;
          }
        },
        computeMinibatchGradient: (theta, idx, b, out) => mlp.computeMinibatchGradient(theta, idx, b, out),
      };
    }
    return {
      numParams: mlp.numParams,
      numSamples: mlp.numSamples,
      checkpointTheta0: theta0,
      applySampleHessian: (i, v, out) => mlp.computeSampleHvp(theta0, v, i, out),
      applyMeanHessian: (v, out) => mlp.computeAnalyticalHvp(theta0, v, out),
      computeMinibatchGradient: (theta, idx, b, out) => mlp.computeMinibatchGradient(theta, idx, b, out),
    };
  }
  if (spec.kind === 'quadratic') {
    const p = spec.numParams;
    const hsq = spec.hessians;
    const tmp = new Float64Array(p);
    // Gradient of the quadratic sum_i (1/2) w^T H_i w at w (checkpoint 0): the linear system itself.
    return buildDenseProvider(p, spec.numSamples, hsq, new Float64Array(p), (theta, ix, b, out) => {
      out.fill(0.0);
      for (let r = 0; r < b; (r = (r + 1) | 0)) {
        const base = Math.imul(ix[r]!, Math.imul(p, p));
        for (let a = 0; a < p; (a = (a + 1) | 0)) {
          let s = 0.0;
          for (let c = 0; c < p; (c = (c + 1) | 0)) s += hsq[(base + Math.imul(a, p) + c) | 0]! * theta[c]!;
          tmp[a] = s;
        }
        for (let a = 0; a < p; (a = (a + 1) | 0)) out[a] = out[a]! + tmp[a]! / b;
      }
      return 0.0;
    });
  }
  const mlp = createTwoLayerMlpDatasetAndOps();
  const p = mlp.numParams;
  const nS = mlp.numSamples;
  const theta0 = spec.theta ?? mlp.checkpointTheta0;
  const pp = Math.imul(p, p);
  const hs = new Float64Array(Math.imul(nS, pp));
  const eps = 1e-5;
  const idx = new Int32Array(1);
  const tp = new Float64Array(p);
  const tm = new Float64Array(p);
  const gp = new Float64Array(p);
  const gm = new Float64Array(p);
  if (spec.gaussNewton === true) {
    // One output and the loss (1/2) err^2: the Gauss-Newton part is grad f_i grad f_i^T, from backpropagating f_i.
    for (let i = 0; i < nS; (i = (i + 1) | 0)) {
      mlp.computeSampleOutputGradient(theta0, i, gp);
      const base = Math.imul(i, pp);
      for (let a = 0; a < p; (a = (a + 1) | 0)) for (let b = 0; b < p; (b = (b + 1) | 0)) hs[(base + Math.imul(a, p) + b) | 0] = gp[a]! * gp[b]!;
    }
    return buildDenseProvider(p, nS, hs, theta0, (theta, ix, b, out) => mlp.computeMinibatchGradient(theta, ix, b, out));
  }
  for (let i = 0; i < nS; (i = (i + 1) | 0)) {
    idx[0] = i;
    const base = Math.imul(i, pp);
    for (let k = 0; k < p; (k = (k + 1) | 0)) {
      tp.set(theta0);
      tm.set(theta0);
      tp[k] = theta0[k]! + eps;
      tm[k] = theta0[k]! - eps;
      mlp.computeMinibatchGradient(tp, idx, 1, gp);
      mlp.computeMinibatchGradient(tm, idx, 1, gm);
      for (let a = 0; a < p; (a = (a + 1) | 0)) hs[(base + Math.imul(a, p) + k) | 0] = (gp[a]! - gm[a]!) / (2.0 * eps);
    }
    for (let a = 0; a < p; (a = (a + 1) | 0)) {
      for (let b = (a + 1) | 0; b < p; (b = (b + 1) | 0)) {
        const ab = (base + Math.imul(a, p) + b) | 0;
        const ba = (base + Math.imul(b, p) + a) | 0;
        const s = 0.5 * (hs[ab]! + hs[ba]!);
        hs[ab] = s;
        hs[ba] = s;
      }
    }
  }
  return buildDenseProvider(p, nS, hs, theta0, (theta, ix, b, out) => mlp.computeMinibatchGradient(theta, ix, b, out));
}

function buildDenseProvider(
  p: number,
  nS: number,
  hs: Float64Array,
  theta0: Float64Array,
  gradient: Phase4SampleHessianProvider['computeMinibatchGradient'],
): Phase4SampleHessianProvider {
  const pp = Math.imul(p, p);
  const mean = new Float64Array(pp);
  const invN = 1.0 / nS;
  for (let i = 0; i < nS; (i = (i + 1) | 0)) {
    const base = Math.imul(i, pp);
    for (let e = 0; e < pp; (e = (e + 1) | 0)) mean[e] = mean[e]! + invN * hs[(base + e) | 0]!;
  }
  const p4 = p & ~3;
  function denseMatVec(m: Float64Array, base: number, v: Float64Array, out: Float64Array): void {
    let a = 0;
    for (; (a + 3) < p; (a = (a + 4) | 0)) {
      const r0 = (base + Math.imul(a, p)) | 0;
      const r1 = (r0 + p) | 0;
      const r2 = (r1 + p) | 0;
      const r3 = (r2 + p) | 0;
      let s0 = 0.0;
      let s1 = 0.0;
      let s2 = 0.0;
      let s3 = 0.0;
      for (let b = 0; b < p; (b = (b + 1) | 0)) {
        const vb = v[b]!;
        s0 += m[(r0 + b) | 0]! * vb;
        s1 += m[(r1 + b) | 0]! * vb;
        s2 += m[(r2 + b) | 0]! * vb;
        s3 += m[(r3 + b) | 0]! * vb;
      }
      out[a] = s0;
      out[(a + 1) | 0] = s1;
      out[(a + 2) | 0] = s2;
      out[(a + 3) | 0] = s3;
    }
    for (; a < p; (a = (a + 1) | 0)) {
      const row = (base + Math.imul(a, p)) | 0;
      let s = 0.0;
      let b = 0;
      for (; b < p4; (b = (b + 4) | 0)) {
        s =
          s +
          m[(row + b) | 0]! * v[b]! +
          m[(row + b + 1) | 0]! * v[(b + 1) | 0]! +
          m[(row + b + 2) | 0]! * v[(b + 2) | 0]! +
          m[(row + b + 3) | 0]! * v[(b + 3) | 0]!;
      }
      for (; b < p; (b = (b + 1) | 0)) s += m[(row + b) | 0]! * v[b]!;
      out[a] = s;
    }
  }
  return {
    numParams: p,
    numSamples: nS,
    checkpointTheta0: theta0,
    applySampleHessian: (i, v, out) => denseMatVec(hs, Math.imul(i, pp), v, out),
    applyMeanHessian: (v, out) => denseMatVec(mean, 0, v, out),
    computeMinibatchGradient: gradient,
  };
}
