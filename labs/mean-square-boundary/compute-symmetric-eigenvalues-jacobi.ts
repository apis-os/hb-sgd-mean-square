/**
 * @file labs/mean-square-boundary/compute-symmetric-eigenvalues-jacobi.ts
 * @description Eigenvalues of a dense symmetric matrix by cyclic Jacobi rotations (`computeSymmetricEigenvaluesJacobi`), in place on a row-major copy, sorted in decreasing order; used for the empirical covariance of the finite feature pool in Phase 1 (`P <= 256`).
 * @cold
 */

export function computeSymmetricEigenvaluesJacobi(matrixRowMajor: Float64Array, n: number, maxSweeps = 40): Float64Array {
  const a = new Float64Array(matrixRowMajor);
  for (let sweep = 0; sweep < maxSweeps; (sweep = (sweep + 1) | 0)) {
    let off = 0.0;
    for (let i = 0; i < n; (i = (i + 1) | 0)) {
      for (let j = (i + 1) | 0; j < n; (j = (j + 1) | 0)) {
        const v = a[(i * n + j) | 0]!;
        off += v * v;
      }
    }
    if (off < 1e-26) break;
    for (let pIdx = 0; pIdx < n; (pIdx = (pIdx + 1) | 0)) {
      for (let q = (pIdx + 1) | 0; q < n; (q = (q + 1) | 0)) {
        const apq = a[(pIdx * n + q) | 0]!;
        if (Math.abs(apq) < 1e-300) continue;
        const app = a[(pIdx * n + pIdx) | 0]!;
        const aqq = a[(q * n + q) | 0]!;
        const theta = (aqq - app) / (2.0 * apq);
        const t = (theta >= 0 ? 1.0 : -1.0) / (Math.abs(theta) + Math.sqrt(theta * theta + 1.0));
        const c = 1.0 / Math.sqrt(t * t + 1.0);
        const s = t * c;
        for (let k = 0; k < n; (k = (k + 1) | 0)) {
          const akp = a[(k * n + pIdx) | 0]!;
          const akq = a[(k * n + q) | 0]!;
          a[(k * n + pIdx) | 0] = c * akp - s * akq;
          a[(k * n + q) | 0] = s * akp + c * akq;
        }
        for (let k = 0; k < n; (k = (k + 1) | 0)) {
          const apk = a[(pIdx * n + k) | 0]!;
          const aqk = a[(q * n + k) | 0]!;
          a[(pIdx * n + k) | 0] = c * apk - s * aqk;
          a[(q * n + k) | 0] = s * apk + c * aqk;
        }
      }
    }
  }
  const eig = new Float64Array(n);
  for (let i = 0; i < n; (i = (i + 1) | 0)) eig[i] = a[(i * n + i) | 0]!;
  eig.sort();
  eig.reverse();
  return eig;
}
