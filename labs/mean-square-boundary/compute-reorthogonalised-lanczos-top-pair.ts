/**
 * @file labs/mean-square-boundary/compute-reorthogonalised-lanczos-top-pair.ts
 * @description Largest eigenvalue and Ritz vector of a symmetric operator on `R^dim` (`computeReorthogonalisedLanczosTopPair`), by
 * Lanczos from `start` with full reorthogonalisation (classical Gram–Schmidt against every kept vector, two passes) at each step.
 * After step `j` the tridiagonal matrix of order `j + 1` is solved by `computeSymmetricEigendecompositionHouseholderQl`; its top
 * eigenvalue `theta` and the last component `s` of its eigenvector give the residual `beta_{j+1} |s|` of the Ritz pair. Stops when
 * that residual is at most `relTol |theta|`, when `beta_{j+1}` falls below `1e-14 |theta|` (an invariant subspace), or after
 * `maxSteps`. The Ritz vector is returned normalised. The value is not a certified bound by itself; Phase 6 recomputes the
 * Rayleigh quotient of the vector with an error bound (`computeCertifiedRayleighUpperThreshold`).
 * @cold
 */
import { computeSymmetricEigendecompositionHouseholderQl } from './compute-symmetric-eigendecomposition-householder-ql.ts';

export function computeReorthogonalisedLanczosTopPair(
  apply: (x: Float64Array, out: Float64Array) => void,
  dim: number,
  start: Float64Array,
  maxSteps: number,
  relTol: number,
): { readonly value: number; readonly residual: number; readonly steps: number; readonly vector: Float64Array } {
  const kMax = Math.max(1, Math.min(maxSteps, dim));
  const basis = new Float64Array(kMax * dim);
  const alpha = new Float64Array(kMax);
  const offd = new Float64Array(kMax);
  const w = new Float64Array(dim);
  let norm0 = 0.0;
  for (let e = 0; e < dim; e++) norm0 += start[e]! * start[e]!;
  norm0 = Math.sqrt(norm0);
  for (let e = 0; e < dim; e++) basis[e] = start[e]! / norm0;
  let theta = 0.0;
  let residual = Infinity;
  let steps = 0;
  let svec = new Float64Array(1);
  for (let j = 0; j < kMax; j++) {
    const vj = basis.subarray(j * dim, (j + 1) * dim);
    apply(vj, w);
    let a = 0.0;
    for (let e = 0; e < dim; e++) a += w[e]! * vj[e]!;
    alpha[j] = a;
    for (let pass = 0; pass < 2; pass++) {
      for (let i = 0; i <= j; i++) {
        let c = 0.0;
        const off = i * dim;
        for (let e = 0; e < dim; e++) c += w[e]! * basis[off + e]!;
        for (let e = 0; e < dim; e++) w[e] = w[e]! - c * basis[off + e]!;
      }
    }
    let bn = 0.0;
    for (let e = 0; e < dim; e++) bn += w[e]! * w[e]!;
    bn = Math.sqrt(bn);
    offd[j] = bn;
    const m = j + 1;
    const tri = new Float64Array(m * m);
    for (let i = 0; i < m; i++) {
      tri[i * m + i] = alpha[i]!;
      if (i + 1 < m) {
        tri[i * m + i + 1] = offd[i]!;
        tri[(i + 1) * m + i] = offd[i]!;
      }
    }
    const eig = computeSymmetricEigendecompositionHouseholderQl(tri, m);
    theta = eig.values[m - 1]!;
    svec = eig.qt.slice((m - 1) * m, m * m);
    residual = bn * Math.abs(svec[m - 1]!);
    steps = m;
    if (residual <= relTol * Math.abs(theta) || bn <= 1e-14 * Math.abs(theta) || m === kMax) break;
    const off = m * dim;
    for (let e = 0; e < dim; e++) basis[off + e] = w[e]! / bn;
  }
  const vector = new Float64Array(dim);
  for (let i = 0; i < steps; i++) {
    const c = svec[i]!;
    const off = i * dim;
    for (let e = 0; e < dim; e++) vector[e] = vector[e]! + c * basis[off + e]!;
  }
  let vn = 0.0;
  for (let e = 0; e < dim; e++) vn += vector[e]! * vector[e]!;
  vn = Math.sqrt(vn);
  for (let e = 0; e < dim; e++) vector[e] = vector[e]! / vn;
  return { value: theta, residual, steps, vector };
}
