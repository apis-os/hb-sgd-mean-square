/**
 * @file labs/mean-square-boundary/compute-minibatch-ratio-of-expectations.ts
 * @description Closed form of the batch-sharpness ratio of expectations `E[g_B^T H_B g_B] / E|g_B|^2` for minibatches of `B`
 * indices drawn independently and uniformly with replacement from `N` samples (`computeMinibatchRatioOfExpectations`), with
 * `g_B` the mean of the drawn per-sample gradients `g_i` (rows of `grads`, plus the constant `shift`) and `H_B` the mean of the
 * drawn per-sample Hessians (`applySample(i, v, out)` writes `H_i v`). Expanding the triple sum over the draws: with
 * `gbar = mean_i g_i`, `Hbar = mean_i H_i`, `T = gbar^T Hbar gbar`, `E1 = mean_i g_i^T Hbar g_i`, `E2 = mean_i gbar^T H_i g_i`,
 * `E3 = mean_i g_i^T H_i g_i`, `E|g|^2 = mean_i |g_i|^2`:
 * `E[g_B^T H_B g_B] = ((B-1)(B-2) T + (B-1)(E1 + 2 E2) + E3) / B^2` and `E|g_B|^2 = E|g|^2 / B + (1 - 1/B) |gbar|^2`.
 * `dropMean` keeps only the terms without `gbar` (`((B-1) E1 + E3) / (B E|g|^2)`, the zero-mean-gradient assumption of the
 * registration's item (viii)). The training objective's `+ mu I` adds exactly `mu` to the ratio and is left to the caller.
 * @cold
 */
export function computeMinibatchRatioOfExpectations(
  grads: Float64Array,
  shift: Float64Array | null,
  numSamples: number,
  p: number,
  applySample: (i: number, v: Float64Array, out: Float64Array) => void,
  batchB: number,
  dropMean: boolean,
): number {
  const g = new Float64Array(Math.imul(numSamples, p));
  for (let i = 0; i < numSamples; (i = (i + 1) | 0)) {
    for (let a = 0; a < p; (a = (a + 1) | 0)) g[(Math.imul(i, p) + a) | 0] = grads[(Math.imul(i, p) + a) | 0]! + (shift === null ? 0.0 : shift[a]!);
  }
  const gbar = new Float64Array(p);
  for (let i = 0; i < numSamples; (i = (i + 1) | 0)) for (let a = 0; a < p; (a = (a + 1) | 0)) gbar[a] = gbar[a]! + g[(Math.imul(i, p) + a) | 0]! / numSamples;
  const out = new Float64Array(p);
  const v = new Float64Array(p);
  // Hbar g_i for every i (and Hbar gbar), as means of the per-sample products.
  const hbarG = new Float64Array(Math.imul(numSamples + 1, p));
  for (let i = 0; i <= numSamples; (i = (i + 1) | 0)) {
    for (let a = 0; a < p; (a = (a + 1) | 0)) v[a] = i < numSamples ? g[(Math.imul(i, p) + a) | 0]! : gbar[a]!;
    for (let j = 0; j < numSamples; (j = (j + 1) | 0)) {
      applySample(j, v, out);
      for (let a = 0; a < p; (a = (a + 1) | 0)) hbarG[(Math.imul(i, p) + a) | 0] = hbarG[(Math.imul(i, p) + a) | 0]! + out[a]! / numSamples;
    }
  }
  let e1 = 0.0;
  let e2 = 0.0;
  let e3 = 0.0;
  let eg2 = 0.0;
  for (let i = 0; i < numSamples; (i = (i + 1) | 0)) {
    for (let a = 0; a < p; (a = (a + 1) | 0)) v[a] = g[(Math.imul(i, p) + a) | 0]!;
    applySample(i, v, out);
    let s1 = 0.0;
    let s2 = 0.0;
    let s3 = 0.0;
    let s4 = 0.0;
    for (let a = 0; a < p; (a = (a + 1) | 0)) {
      s1 += v[a]! * hbarG[(Math.imul(i, p) + a) | 0]!;
      s2 += gbar[a]! * out[a]!;
      s3 += v[a]! * out[a]!;
      s4 += v[a]! * v[a]!;
    }
    e1 += s1 / numSamples;
    e2 += s2 / numSamples;
    e3 += s3 / numSamples;
    eg2 += s4 / numSamples;
  }
  if (dropMean) return ((batchB - 1) * e1 + e3) / (batchB * eg2);
  let t = 0.0;
  let gbar2 = 0.0;
  for (let a = 0; a < p; (a = (a + 1) | 0)) {
    t += gbar[a]! * hbarG[(Math.imul(numSamples, p) + a) | 0]!;
    gbar2 += gbar[a]! * gbar[a]!;
  }
  const numer = ((batchB - 1) * (batchB - 2) * t + (batchB - 1) * (e1 + 2.0 * e2) + e3) / (batchB * batchB);
  const denom = eg2 / batchB + (1.0 - 1.0 / batchB) * gbar2;
  return numer / denom;
}
