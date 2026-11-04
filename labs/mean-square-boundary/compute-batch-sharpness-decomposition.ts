/**
 * @file labs/mean-square-boundary/compute-batch-sharpness-decomposition.ts
 * @description The `p6` quantities of amendment 3 at one checkpoint (`computeBatchSharpnessDecomposition`; reported, the
 * registered `p6` is unchanged). Every entry includes the `+ mu_train` that the measured ratio of expectations carries exactly.
 * `p6FormulaWithMu`: the registered formula plus `mu_train`. `p6FormulaResidualWeighted`: the same zero-mean-gradient expansion
 * with the residuals kept, `(mean_i w_i |grad f_i|^4 + (B - 1) tr(H~ H_GN)) / (B tr H~) + mu_train`, `w_i = r_i^2 / mean_j r_j^2`,
 * `H~ = mean_i w_i grad f_i grad f_i^T` (with `w = 1` it is the registered formula; the residual `r_i` is the `b_2` component of
 * the per-sample loss gradient, whose output gradient is 1). The decomposition, in this fixed order (each step changes one
 * assumption, so the differences attribute the gap between the Monte Carlo and the formula; another order would split it
 * differently): `a0` the 256-draw Monte Carlo of the checkpoint; `a1` the exact ratio of expectations
 * (`computeMinibatchRatioOfExpectations`) with the full per-sample Hessians and gradients `g_i + mu theta`; `a2` the same with the
 * Gauss–Newton parts `grad f_i grad f_i^T`; `a3` Gauss–Newton with the mean-gradient terms dropped and `g_i` without
 * `mu theta` (equal to `p6FormulaResidualWeighted` up to rounding); `a4` the registered formula plus `mu_train`.
 * @cold
 */
import { computeMinibatchRatioOfExpectations } from './compute-minibatch-ratio-of-expectations.ts';

export function computeBatchSharpnessDecomposition(
  grads: Float64Array,
  outputGrads: Float64Array,
  theta: Float64Array,
  numSamples: number,
  p: number,
  muTrain: number,
  batchB: number,
  applyFull: (i: number, v: Float64Array, out: Float64Array) => void,
  applyGaussNewton: (i: number, v: Float64Array, out: Float64Array) => void,
  p6Formula: number,
  monteCarloRatioOfExpectations: number,
): Record<string, unknown> {
  const shift = Float64Array.from(theta, (x) => muTrain * x);
  let meanR2 = 0.0;
  for (let i = 0; i < numSamples; (i = (i + 1) | 0)) {
    const r = grads[(Math.imul(i, p) + p - 1) | 0]!;
    meanR2 += (r * r) / numSamples;
  }
  let ew4 = 0.0;
  let trTilde = 0.0;
  let trTildeH = 0.0;
  for (let i = 0; i < numSamples; (i = (i + 1) | 0)) {
    const r = grads[(Math.imul(i, p) + p - 1) | 0]!;
    const w = (r * r) / meanR2;
    let zi2 = 0.0;
    for (let a = 0; a < p; (a = (a + 1) | 0)) zi2 += outputGrads[(Math.imul(i, p) + a) | 0]! * outputGrads[(Math.imul(i, p) + a) | 0]!;
    ew4 += (w * zi2 * zi2) / numSamples;
    trTilde += (w * zi2) / numSamples;
    for (let j = 0; j < numSamples; (j = (j + 1) | 0)) {
      let zz = 0.0;
      for (let a = 0; a < p; (a = (a + 1) | 0)) zz += outputGrads[(Math.imul(i, p) + a) | 0]! * outputGrads[(Math.imul(j, p) + a) | 0]!;
      trTildeH += (w * zz * zz) / (numSamples * numSamples);
    }
  }
  const weighted = (ew4 + (batchB - 1) * trTildeH) / (batchB * trTilde) + muTrain;
  return {
    p6FormulaWithMu: p6Formula + muTrain,
    p6FormulaResidualWeighted: weighted,
    decomposition: {
      a0MonteCarlo: monteCarloRatioOfExpectations,
      a1ExactFullHessian: computeMinibatchRatioOfExpectations(grads, shift, numSamples, p, applyFull, batchB, false) + muTrain,
      a2ExactGaussNewton: computeMinibatchRatioOfExpectations(grads, shift, numSamples, p, applyGaussNewton, batchB, false) + muTrain,
      a3GaussNewtonMeanDropped: computeMinibatchRatioOfExpectations(grads, null, numSamples, p, applyGaussNewton, batchB, true) + muTrain,
      a4RegisteredFormula: p6Formula + muTrain,
    },
  };
}
