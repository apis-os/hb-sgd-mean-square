/**
 * @file labs/mean-square-boundary/phase6-noise-operator-brand.ts
 * @description The set of noise operators built by the two Phase 6 factories (`phase6NoiseOperatorBrand`):
 * `createDenseDeviationNoiseOperator` and `createRowSpaceFactorNoiseOperator` add each operator they return, frozen, and
 * `computeCertifiedThresholdBracket` and `computeCertifiedRayleighUpperThreshold` refuse any other object. The proof needs a noise
 * map that is linear, self-adjoint and cone-preserving; the factories compute `V(X) = (1 / (B N)) sum_i (D_i - Dbar) X (D_i - Dbar)`
 * with a rounding bound, which is all three, and no check on an arbitrary callback can establish them (thirteenth audit round: any
 * function passed the `typeof apply` test through `Function.prototype.apply` and gave the noise-free bracket, and a fixed, exactly
 * computed but not cone-preserving map was certified at 300 times its threshold). A `WeakSet` membership cannot be forged.
 * @cold
 */
export const phase6NoiseOperatorBrand = new WeakSet<object>();
