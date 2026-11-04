/**
 * @file labs/mean-square-boundary/measure-second-moment-spectral-radius.ts
 * @description Matrix-free power iteration for the spectral radius of the heavy-ball second-moment operator `T` of `applyHeavyBallSecondMomentOperator` (`measureSecondMomentSpectralRadius`). `q` holds the start matrix on entry (unit Frobenius norm; the identity for the symmetric part, an antisymmetric matrix for the other) and the last normalised iterate on exit, so consecutive calls warm-start. Each step applies `T`, projects (in 64 x 64 tiles) on the symmetric (`part = 0`) or antisymmetric (`part = 1`) matrices to remove rounding drift (`T` commutes with transposition, so both subspaces are invariant), and normalises. On the symmetric part the radius is the ratio of Frobenius norms, which converges to the Perron root: the iteration runs on the positive semidefinite cone and the radius is `rho(T)` by the Russo–Dye lemma; it stops when the ratio changes by less than `relTol` for 20 consecutive steps or after `maxIterations`. On the antisymmetric part the dominant eigenvalues form a cluster of equal modulus (on the deterministic part every product `r_a r_b` of complex roots has modulus `beta`), the ratio of norms oscillates, and the radius is the Gelfand estimate `exp(mean log |T x_k|)` over steps `maxIterations / 4 .. maxIterations` of a fixed horizon; `converged` then means that the means over the second quarter and over the second half agree to `1e-3`.
 * @cold
 */
import { fdlibmExp, fdlibmLog } from '../../src/math/index.ts';
import { applyHeavyBallSecondMomentOperator } from './apply-heavy-ball-second-moment-operator.ts';

export function measureSecondMomentSpectralRadius(
  lambdas: Float64Array,
  beta: number,
  eta: number,
  batchB: number,
  kappaFeat: number,
  q: Float64Array,
  scratch: Float64Array,
  part: 0 | 1,
  maxIterations: number,
  relTol: number,
): { readonly radius: number; readonly iterations: number; readonly converged: boolean } {
  const n = (2 * lambdas.length) | 0;
  const total = Math.imul(n, n);
  const sign = part === 0 ? 1.0 : -1.0;
  const quarter = (maxIterations >> 2) | 0;
  const half = (maxIterations >> 1) | 0;
  let radius = 0.0;
  let stable = 0;
  let sumLogQ2 = 0.0;
  let sumLogH2 = 0.0;
  let it = 0;
  for (; it < maxIterations; (it = (it + 1) | 0)) {
    applyHeavyBallSecondMomentOperator(lambdas, beta, eta, batchB, kappaFeat, q, scratch);
    let normSq = 0.0;
    for (let ib = 0; ib < n; ib = (ib + 64) | 0) {
      const iEnd = ib + 64 < n ? (ib + 64) | 0 : n;
      for (let jb = ib; jb < n; jb = (jb + 64) | 0) {
        const jEnd = jb + 64 < n ? (jb + 64) | 0 : n;
        for (let i = ib; i < iEnd; (i = (i + 1) | 0)) {
          const rowI = Math.imul(i, n);
          const jStart = jb > i ? jb : i;
          for (let j = jStart; j < jEnd; (j = (j + 1) | 0)) {
            const ij = (rowI + j) | 0;
            const ji = (Math.imul(j, n) + i) | 0;
            const v = 0.5 * (scratch[ij]! + sign * scratch[ji]!);
            scratch[ij] = v;
            scratch[ji] = sign * v;
            normSq += i === j ? v * v : 2.0 * v * v;
          }
        }
      }
    }
    const scale = Math.sqrt(normSq);
    if (!(scale > 0.0)) return { radius: 0.0, iterations: (it + 1) | 0, converged: true };
    const inv = 1.0 / scale;
    for (let i = 0; i < total; (i = (i + 1) | 0)) q[i] = scratch[i]! * inv;
    if (part === 0) {
      if (Math.abs(scale - radius) <= relTol * scale) stable = (stable + 1) | 0;
      else stable = 0;
      radius = scale;
      if (stable >= 20) return { radius, iterations: (it + 1) | 0, converged: true };
    } else if (it >= quarter) {
      const lg = fdlibmLog(scale);
      if (it < half) sumLogQ2 += lg;
      else sumLogH2 += lg;
    }
  }
  if (part === 0) return { radius, iterations: it, converged: false };
  const meanQ2 = sumLogQ2 / ((half - quarter) | 0);
  const meanH2 = sumLogH2 / ((maxIterations - half) | 0);
  const estimate = fdlibmExp((sumLogQ2 + sumLogH2) / ((maxIterations - quarter) | 0));
  const agree = Math.abs(fdlibmExp(meanH2 - meanQ2) - 1.0) <= 1e-3;
  return { radius: estimate, iterations: it, converged: agree };
}
