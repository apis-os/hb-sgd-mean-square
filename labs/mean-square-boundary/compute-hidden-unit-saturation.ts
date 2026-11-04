/**
 * @file labs/mean-square-boundary/compute-hidden-unit-saturation.ts
 * @description Saturation of the `tanh` hidden units of the two-layer network at `theta` (`computeHiddenUnitSaturation`; amendment 3,
 * a reported quantity without prediction). With `z_{i,h} = b_{1,h} + w_{1,h}^T x_i` (layout of `createTwoLayerMlpDatasetAndOps`:
 * `W_1` row-major `d_hidden x d_in`, then `b_1`, `w_2`, `b_2`), `s_h = (1/N) sum_i (1 - tanh^2 z_{i,h})^2` (`fdlibmTanh`); the counts
 * of units with `s_h < 1e-4` and `s_h < 1e-6`, and `activeTangentDim = P - (d_in + 1) * count(s_h < 1e-4)`. The thresholds are
 * arbitrary and the dimension is a heuristic count: a unit saturated with one sign on every sample also makes its `w_{2,h}`
 * collinear with `b_2`, which the count does not subtract.
 * @cold
 */
import { fdlibmTanh } from '../../src/math/index.ts';

export function computeHiddenUnitSaturation(
  theta: Float64Array,
  xData: Float64Array,
  numSamples: number,
  dIn: number,
  dHidden: number,
): { readonly unitDerivativeSq: number[]; readonly saturatedHiddenUnits1e4: number; readonly saturatedHiddenUnits1e6: number; readonly activeTangentDim: number } {
  const b1Base = Math.imul(dIn, dHidden);
  const unitDerivativeSq: number[] = [];
  let sat4 = 0;
  let sat6 = 0;
  for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
    const w1Row = Math.imul(h, dIn);
    let s = 0.0;
    for (let i = 0; i < numSamples; (i = (i + 1) | 0)) {
      const xOff = Math.imul(i, dIn);
      let z = theta[(b1Base + h) | 0]!;
      for (let j = 0; j < dIn; (j = (j + 1) | 0)) z += theta[(w1Row + j) | 0]! * xData[(xOff + j) | 0]!;
      const a = fdlibmTanh(z);
      const d = 1.0 - a * a;
      s += d * d;
    }
    s /= numSamples;
    unitDerivativeSq.push(s);
    if (s < 1e-4) sat4 = (sat4 + 1) | 0;
    if (s < 1e-6) sat6 = (sat6 + 1) | 0;
  }
  const numParams = b1Base + dHidden + dHidden + 1;
  return { unitDerivativeSq, saturatedHiddenUnits1e4: sat4, saturatedHiddenUnits1e6: sat6, activeTangentDim: numParams - Math.imul(dIn + 1, sat4) };
}
