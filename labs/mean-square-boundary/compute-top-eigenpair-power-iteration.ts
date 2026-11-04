/**
 * @file labs/mean-square-boundary/compute-top-eigenpair-power-iteration.ts
 * @description Power iteration with a seeded Gaussian start (`computeTopEigenpairPowerIteration`): `iterations` matrix-vector products, the Rayleigh quotient of the last product as the eigenvalue estimate and the normalised last product as the eigenvector. The operator is a callback, so the same routine runs on analytical and on finite-difference Hessian-vector products.
 * @cold
 */
import { createXorshift128Plus } from './prng-xorshift128p.ts';

export function computeTopEigenpairPowerIteration(
  applyHvp: (vec: Float64Array, outHv: Float64Array) => void,
  numParams: number,
  iterations: number,
  seedLo: number,
  seedHi: number,
): { readonly lambda: number; readonly vec: Float64Array; readonly products: number } {
  const p = numParams | 0;
  const prng = createXorshift128Plus(seedLo | 0, seedHi | 0);
  const v = new Float64Array(p);
  const hv = new Float64Array(p);
  let norm = 0.0;
  for (let i = 0; i < p; (i = (i + 1) | 0)) {
    const g = prng.nextGaussian();
    v[i] = g;
    norm += g * g;
  }
  let inv = 1.0 / Math.sqrt(norm);
  for (let i = 0; i < p; (i = (i + 1) | 0)) v[i] = v[i]! * inv;
  let rayleigh = 0.0;
  for (let it = 0; it < iterations; (it = (it + 1) | 0)) {
    applyHvp(v, hv);
    let dot = 0.0;
    let hvNorm = 0.0;
    for (let i = 0; i < p; (i = (i + 1) | 0)) {
      dot += v[i]! * hv[i]!;
      hvNorm += hv[i]! * hv[i]!;
    }
    rayleigh = dot;
    if (hvNorm <= 0.0) break;
    inv = 1.0 / Math.sqrt(hvNorm);
    for (let i = 0; i < p; (i = (i + 1) | 0)) v[i] = hv[i]! * inv;
  }
  return { lambda: rayleigh, vec: v, products: iterations | 0 };
}
