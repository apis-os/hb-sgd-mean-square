/**
 * @file labs/mean-square-boundary/compute-eigenbasis-threshold-dense.reference.ts
 * @description Test oracle of Phase 6 (`computeEigenbasisThresholdDenseReference`): the threshold of Theorem G.6 for mean
 * curvature `diag(lt)` and dense deviations `dev` (sample-major `n x n`, as `createDenseDeviationNoiseOperator`), by assembling
 * `G = L0^{-1/2} B_beta L0^{-1/2}` as a dense symmetric matrix in the orthonormal basis `E_aa`, `(E_ab + E_ba) / sqrt(2)` of `S^n`
 * with plain loops (deviations symmetrised and centred, `sym(D_i) - mean_i sym(D_i)`, then `V(E)` summed sample by sample, no shared code with the certificate
 * path; an approximate oracle in floating point, judged in the tests only beside the exact enclosure) and taking its largest eigenvalue with
 * `computeSymmetricEigendecompositionHouseholderQl`; `eta* = min(1 / lambda_max, 2 (1 + beta) / max lt)`. `commutatorScale`
 * multiplies `c_beta` (1 for the theorem; other values only build mutated references in tests).
 * @cold
 */
import { computeSymmetricEigendecompositionHouseholderQl } from './compute-symmetric-eigendecomposition-householder-ql.ts';

export function computeEigenbasisThresholdDenseReference(
  lt: Float64Array,
  n: number,
  beta: number,
  dev: Float64Array,
  numSamples: number,
  batchB: number,
  commutatorScale = 1,
): number {
  const nn = n * n;
  const pa: number[] = [];
  const pb: number[] = [];
  for (let a = 0; a < n; a++) for (let b = a; b < n; b++) { pa.push(a); pb.push(b); }
  const dim = pa.length;
  const cb = (commutatorScale * beta) / ((1 - beta) * (1 - beta) * (1 + beta));
  const sym = Float64Array.from(dev, (v, idx) => { const i = Math.floor(idx / nn); const a = Math.floor((idx % nn) / n); const b = idx % n; return 0.5 * (v + dev[i * nn + b * n + a]!); });
  // Centred about the first sample, then about the mean of the differences (twelfth audit round: deviations sharing an offset 2^60
  // were centred to 128 instead of 0 by summing sym / N directly).
  const shift = new Float64Array(nn);
  for (let i = 1; i < numSamples; i++) for (let k = 0; k < nn; k++) shift[k] = shift[k]! + (sym[i * nn + k]! - sym[k]!) / numSamples;
  const dc = Float64Array.from(sym, (v, idx) => (v - sym[idx % nn]!) - shift[idx % nn]!);
  const g = new Float64Array(dim * dim);
  const e = new Float64Array(nn);
  const t = new Float64Array(nn);
  const v = new Float64Array(nn);
  for (let q = 0; q < dim; q++) {
    e.fill(0);
    const a0 = pa[q]!;
    const b0 = pb[q]!;
    const s0 = a0 === b0 ? 1 : 1 / Math.sqrt(2);
    const c0 = s0 / Math.sqrt(lt[a0]! + lt[b0]!);
    e[a0 * n + b0] = c0;
    e[b0 * n + a0] = c0;
    v.fill(0);
    for (let i = 0; i < numSamples; i++) {
      for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += dc[i * nn + a * n + k]! * e[k * n + b]!;
        t[a * n + b] = s;
      }
      for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += t[a * n + k]! * dc[i * nn + k * n + b]!;
        v[a * n + b] = v[a * n + b]! + s / (numSamples * batchB);
      }
    }
    for (let p = 0; p < dim; p++) {
      const a = pa[p]!;
      const b = pb[p]!;
      const diff = lt[a]! - lt[b]!;
      const d = (lt[a]! * lt[b]!) / (1 + beta) - cb * diff * diff;
      const sym = a === b ? 1 : Math.sqrt(2);
      const bx = d * e[a * n + b]! + v[a * n + b]! / (1 - beta);
      g[p * dim + q] = (sym * bx) / Math.sqrt(lt[a]! + lt[b]!);
    }
  }
  for (let p = 0; p < dim; p++) for (let q = p + 1; q < dim; q++) {
    const s = 0.5 * (g[p * dim + q]! + g[q * dim + p]!);
    g[p * dim + q] = s;
    g[q * dim + p] = s;
  }
  const top = computeSymmetricEigendecompositionHouseholderQl(g, dim).values[dim - 1]!;
  let ltMax = 0;
  for (let a = 0; a < n; a++) ltMax = Math.max(ltMax, lt[a]!);
  return Math.min(1 / top, (2 * (1 + beta)) / ltMax);
}
