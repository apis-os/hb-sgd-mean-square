/**
 * @file labs/mean-square-boundary/compute-symmetric-top-eigenvalue-householder-ql.ts
 * @description Largest eigenvalue of a dense symmetric `n x n` matrix (`computeSymmetricTopEigenvalueHouseholderQl`), equal bit
 * for bit to `computeSymmetricEigendecompositionHouseholderQl(a, n).values[n - 1]`: the same Householder reduction
 * (`reduceSymmetricToTridiagonalHouseholder`), the same implicit QL and selection sort (`solveTridiagonalQlExact` without
 * vectors), skipping only the eigenvector accumulation and rotations, which the eigenvalues never read. `a` is not modified.
 * @cold
 */
import { reduceSymmetricToTridiagonalHouseholder } from './reduce-symmetric-to-tridiagonal-householder.ts';
import { solveTridiagonalQlExact } from './solve-tridiagonal-ql-exact.ts';

export function computeSymmetricTopEigenvalueHouseholderQl(a: Float64Array, n: number): number {
  const vt = new Float64Array(a);
  const d = new Float64Array(n);
  const e = new Float64Array(n);
  reduceSymmetricToTridiagonalHouseholder(vt, d, e, n);
  solveTridiagonalQlExact(d, e, n, null, null, new Int32Array(n), false);
  return d[(n - 1) | 0]!;
}
