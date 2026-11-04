/**
 * @file labs/mean-square-boundary/measure-network-second-moment-spectral-radius.ts
 * @description Power iteration for the spectral radius of `applyNetworkHeavyBallSecondMomentOperator` on the symmetric
 * `2P x 2P` matrices (`measureNetworkSecondMomentSpectralRadius`), the protocol of Phase 3 (`part = 0`): each step applies
 * `T`, symmetrises the diagonal blocks, normalises in the Frobenius norm of the full matrix (`|Q11|^2 + |Q22|^2 + 2 |Q12|^2`)
 * and takes the ratio of norms as the radius; the iteration runs on the positive semidefinite cone, so the ratio converges to
 * the Perron root `rho(T)`. It stops when the ratio changes by less than `relTol` for 20 consecutive steps or after
 * `maxIterations`. `q11`, `q12`, `q22` hold the start on entry (the caller passes the identity, or the last iterate of a
 * neighbouring learning rate) and the last normalised iterate on exit.
 * @cold
 */
import { applyNetworkHeavyBallSecondMomentOperator } from './apply-network-heavy-ball-second-moment-operator.ts';
import type { createPhase4RowExecutor } from './create-phase4-row-executor.ts';

export async function measureNetworkSecondMomentSpectralRadius(
  exec: Awaited<ReturnType<typeof createPhase4RowExecutor>>,
  beta: number,
  eta: number,
  mu: number,
  batchB: number,
  q11: Float64Array,
  q12: Float64Array,
  q22: Float64Array,
  o: readonly [Float64Array, Float64Array, Float64Array],
  s: Parameters<typeof applyNetworkHeavyBallSecondMomentOperator>[11],
  maxIterations: number,
  relTol: number,
): Promise<{ readonly radius: number; readonly iterations: number; readonly converged: boolean }> {
  const n = exec.n;
  const total = Math.imul(n, n);
  const [o11, o12, o22] = o;
  let radius = 0.0;
  let stable = 0;
  for (let it = 0; it < maxIterations; (it = (it + 1) | 0)) {
    await applyNetworkHeavyBallSecondMomentOperator(exec, beta, eta, mu, batchB, q11, q12, q22, o11, o12, o22, s);
    let normSq = 0.0;
    for (let j = 0; j < n; (j = (j + 1) | 0)) {
      for (let k = j; k < n; (k = (k + 1) | 0)) {
        const jk = (Math.imul(j, n) + k) | 0;
        const kj = (Math.imul(k, n) + j) | 0;
        const a = 0.5 * (o11[jk]! + o11[kj]!);
        const d = 0.5 * (o22[jk]! + o22[kj]!);
        o11[jk] = a;
        o11[kj] = a;
        o22[jk] = d;
        o22[kj] = d;
        const w = j === k ? 1.0 : 2.0;
        normSq += w * (a * a + d * d);
      }
    }
    for (let e = 0; e < total; (e = (e + 1) | 0)) normSq += 2.0 * o12[e]! * o12[e]!;
    const scale = Math.sqrt(normSq);
    if (!(scale > 0.0)) return { radius: 0.0, iterations: (it + 1) | 0, converged: true };
    const inv = 1.0 / scale;
    for (let e = 0; e < total; (e = (e + 1) | 0)) {
      q11[e] = o11[e]! * inv;
      q12[e] = o12[e]! * inv;
      q22[e] = o22[e]! * inv;
    }
    if (Math.abs(scale - radius) <= relTol * scale) stable = (stable + 1) | 0;
    else stable = 0;
    radius = scale;
    if (stable >= 20) return { radius, iterations: (it + 1) | 0, converged: true };
  }
  return { radius, iterations: maxIterations, converged: false };
}
