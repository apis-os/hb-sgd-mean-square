/**
 * @file labs/mean-square-boundary/compute-dense-symmetric-top-eigenpair.ts
 * @description Largest eigenvalue and its Ritz vector of a dense symmetric `dim x dim` matrix (`computeDenseSymmetricTopEigenpair`),
 * by the three-term Lanczos recurrence of `computeLanczosTopEigenvalue` from the all-ones start (or `start`, a warm start),
 * stopping at relative Ritz residual `relTol`; the Ritz vector comes from Lanczos vectors kept in single precision (enough to
 * classify a critical mode, not to certify it). Bit-identical to that recurrence with the dense matrix-vector product
 * (`computeDenseSymmetricTopEigenpairReference`, the test oracle) in every returned field, at lower cost: the product runs six
 * rows per pass (`multiplyDenseMatrixVectorSixRows`); the tridiagonal problem of each step is solved from the diagonal and
 * off-diagonal that the Householder reduction would produce (`initTred2EquivalentTridiagonal`) by the same implicit QL tracking
 * only the last eigenvector column (`solveTridiagonalQlExact`, `O(m^2)` per step instead of `O(m^3)`); the full top eigenvector is
 * formed once at the stop (`assembleLanczosRitzVectorExact`). From the first step at which an entry fails
 * `isTred2IdentityEntrySafe` the solve uses `computeSymmetricEigendecompositionHouseholderQl` on the dense tridiagonal, as the
 * reference does.
 * @cold
 */
import { assembleLanczosRitzVectorExact } from './assemble-lanczos-ritz-vector-exact.ts';
import { buildDenseTridiagonal } from './build-dense-tridiagonal.ts';
import { computeSymmetricEigendecompositionHouseholderQl } from './compute-symmetric-eigendecomposition-householder-ql.ts';
import { initTred2EquivalentTridiagonal } from './init-tred2-equivalent-tridiagonal.ts';
import { isTred2IdentityEntrySafe } from './is-tred2-identity-entry-safe.ts';
import { multiplyDenseMatrixVectorSixRows } from './multiply-dense-matrix-vector-six-rows.ts';
import { orthogonalizeLanczosThreeTerm } from './orthogonalize-lanczos-three-term.ts';
import { solveTridiagonalQlExact } from './solve-tridiagonal-ql-exact.ts';

export async function computeDenseSymmetricTopEigenpair(
  a: Float64Array,
  dim: number,
  relTol: number,
  maxIterations: number,
  start?: Float64Array,
): Promise<{ readonly value: number; readonly iterations: number; readonly converged: boolean; readonly relResidual: number; readonly ritz: Float64Array }> {
  const d = dim | 0;
  const v0 = start === undefined ? new Float64Array(d).fill(1.0) : Float64Array.from(start);
  const ritz = new Float64Array(d);
  const maxIt = Math.min(d, maxIterations | 0);
  if (maxIt <= 0) return Promise.resolve({ value: 0.0, iterations: maxIt, converged: false, relResidual: Number.POSITIVE_INFINITY, ritz });
  const vCur = new Float64Array(d);
  const vPrev = new Float64Array(d);
  const w = new Float64Array(d);
  let norm = 0.0;
  for (let e = 0; e < d; (e = (e + 1) | 0)) norm += v0[e]! * v0[e]!;
  const inv0 = 1.0 / Math.sqrt(norm);
  for (let e = 0; e < d; (e = (e + 1) | 0)) vCur[e] = v0[e]! * inv0;
  const alphas = new Float64Array(maxIt);
  const betas = new Float64Array(maxIt);
  const dd = new Float64Array(maxIt);
  const ee = new Float64Array(maxIt);
  const z = new Float64Array(maxIt);
  const perm = new Int32Array(maxIt);
  const basis = new Float32Array(Math.imul(maxIt, d));
  let betaPrev = 0.0;
  let theta = 0.0;
  let relResidual = Number.POSITIVE_INFINITY;
  let householder = false;
  let iterations = maxIt;
  let converged = false;
  for (let k = 0; k < maxIt; (k = (k + 1) | 0)) {
    basis.set(vCur, Math.imul(k, d));
    multiplyDenseMatrixVectorSixRows(a, vCur, w, d);
    const step = orthogonalizeLanczosThreeTerm(w, vCur, vPrev, betaPrev, d);
    const beta = step.beta;
    alphas[k] = step.alpha;
    const m = (k + 1) | 0;
    if (!isTred2IdentityEntrySafe(step.alpha) || (k > 0 && !isTred2IdentityEntrySafe(betas[(k - 1) | 0]!))) householder = true;
    let lastComponent: number;
    if (householder) {
      const eig = computeSymmetricEigendecompositionHouseholderQl(buildDenseTridiagonal(alphas, betas, m), m);
      theta = eig.values[(m - 1) | 0]!;
      lastComponent = eig.qt[(Math.imul((m - 1) | 0, m) + m - 1) | 0]!;
    } else {
      initTred2EquivalentTridiagonal(alphas, betas, m, dd, ee);
      for (let i = 0; i < m; (i = (i + 1) | 0)) z[i] = 0.0;
      z[(m - 1) | 0] = 1.0;
      solveTridiagonalQlExact(dd, ee, m, null, z, perm, true);
      theta = dd[(m - 1) | 0]!;
      lastComponent = z[perm[(m - 1) | 0]!]!;
    }
    relResidual = (beta * Math.abs(lastComponent)) / Math.abs(theta);
    if (relResidual <= relTol || !(beta > 0.0)) {
      iterations = m;
      converged = true;
      break;
    }
    betas[k] = beta;
    const invBeta = 1.0 / beta;
    for (let e = 0; e < d; (e = (e + 1) | 0)) {
      vPrev[e] = vCur[e]!;
      vCur[e] = w[e]! * invBeta;
    }
    betaPrev = beta;
  }
  assembleLanczosRitzVectorExact(basis, alphas, betas, iterations, d, householder, ritz);
  return Promise.resolve({ value: theta, iterations, converged, relResidual, ritz });
}
