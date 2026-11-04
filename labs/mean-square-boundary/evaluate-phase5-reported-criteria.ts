/**
 * @file labs/mean-square-boundary/evaluate-phase5-reported-criteria.ts
 * @description The added threshold criteria of amendment 3 at one checkpoint (`evaluatePhase5ReportedCriteria`; reported, no
 * prediction, H1-H5 stay decided by `|H|`). All use `evaluateDenseThresholdCriterion` with the cuts of the checkpoint.
 * `absHRestricted1e2`: `|H|` on the modes with `|lambda_k| >= 1e-2 max|lambda|` (both signs; the rigid cut of amendment 1), which
 * removes the modes near zero. `rigidPositive[tau]`, `tau` in `{0.02, 0.05}`: the modes with `lambda_k >= tau lambda_1` of
 * `H + mu_train I`, where `|H|` and the real objective coincide on the kept block. Both are Rayleigh compressions of the `|H|`
 * operator (the kept modes avoid the removed null modes), so each `eta*` is at least `eta*_|H|` by construction; neither is the
 * iteration on all modes. The `|H|` operator differs from the real one on pairs of opposite sign, the double commutator included;
 * in `rigidPositive` the two coincide on the kept block, but the restriction is a Rayleigh compression, not the real iteration.
 * `hybridGnCurvatureFullNoise`: the Gauss–Newton curvature (`lambda_k(H_GN) + mu_train`, kept modes) in the Gauss–Newton
 * eigenbasis with the noise of the full per-sample Hessians projected on it (`buildSampleHessianDeviationsInBasis`); against
 * `|H|` it changes the curvature and the basis of every mode at once, so it does not isolate the near-zero modes.
 * @cold
 */
import { assembleCurvatureNoiseGram } from './assemble-curvature-noise-gram.ts';
import type { buildEigenbasisSampleHessianDeviations } from './build-eigenbasis-sample-hessian-deviations.ts';
import { buildSampleHessianDeviationsInBasis } from './build-sample-hessian-deviations-in-basis.ts';
import { evaluateDenseThresholdCriterion } from './evaluate-dense-threshold-criterion.ts';
import type { Phase4SampleHessianProvider } from './phase4-types.ts';
import { restrictEigenbasisDeviations } from './restrict-eigenbasis-deviations.ts';

type Dev = ReturnType<typeof buildEigenbasisSampleHessianDeviations>;

async function onMask(dev: Dev, values: Float64Array, mask: Uint8Array, eta: number, beta: number, batchB: number, relTol: number, cuts: readonly number[]): Promise<Record<string, unknown> | null> {
  const sub = restrictEigenbasisDeviations(dev, mask);
  if (sub.n === 0) return null;
  const vals = Float64Array.from(sub.modes, (k) => values[k]!);
  const curv = Float64Array.from(vals, (v) => Math.abs(v));
  let top = 0;
  for (let k = 0; k < sub.n; (k = (k + 1) | 0)) if (curv[k]! >= curv[top]!) top = k;
  const gram = assembleCurvatureNoiseGram(sub.packed, sub.numSamples, sub.nPair, batchB);
  const res = await evaluateDenseThresholdCriterion(sub, gram, curv, vals, beta, batchB, relTol, cuts, top, []);
  return { keptModes: sub.n, ratioToEta: res.etaStar / eta, ...res };
}

export async function evaluatePhase5ReportedCriteria(
  provider: Phase4SampleHessianProvider,
  dev: Dev,
  values: Float64Array,
  devGnK: Dev,
  valuesGnK: Float64Array,
  eta: number,
  beta: number,
  batchB: number,
  relTol: number,
  cuts: readonly number[],
): Promise<Record<string, unknown>> {
  const n = dev.n;
  const lambda1 = values[(n - 1) | 0]!;
  let maxAbs = 0.0;
  for (let k = 0; k < n; (k = (k + 1) | 0)) if (Math.abs(values[k]!) > maxAbs) maxAbs = Math.abs(values[k]!);
  const absMask = Uint8Array.from(values, (v) => (Math.abs(v) >= 1e-2 * maxAbs ? 1 : 0));
  const rigidPositive: Record<string, unknown> = {};
  for (const tau of [0.02, 0.05]) {
    const mask = Uint8Array.from(values, (v) => (lambda1 > 0.0 && v >= tau * lambda1 ? 1 : 0));
    rigidPositive[String(tau)] = await onMask(dev, values, mask, eta, beta, batchB, relTol, cuts);
  }
  const hyb = buildSampleHessianDeviationsInBasis(provider, devGnK.qt, devGnK.n);
  const hybCurv = Float64Array.from(valuesGnK, (v) => Math.abs(v));
  let hybTop = 0;
  for (let k = 0; k < hyb.n; (k = (k + 1) | 0)) if (hybCurv[k]! >= hybCurv[hybTop]!) hybTop = k;
  const hybGram = assembleCurvatureNoiseGram(hyb.packed, hyb.numSamples, hyb.nPair, batchB);
  const hybRes = await evaluateDenseThresholdCriterion(hyb, hybGram, hybCurv, valuesGnK, beta, batchB, relTol, cuts, hybTop, []);
  return {
    absHRestricted1e2: await onMask(dev, values, absMask, eta, beta, batchB, relTol, cuts),
    rigidPositive,
    hybridGnCurvatureFullNoise: { keptModes: hyb.n, ratioToEta: hybRes.etaStar / eta, ...hybRes },
  };
}
