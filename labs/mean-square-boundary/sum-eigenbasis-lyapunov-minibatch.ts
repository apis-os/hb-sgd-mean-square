/**
 * @file labs/mean-square-boundary/sum-eigenbasis-lyapunov-minibatch.ts
 * @description The scaled minibatch deviation of one linearised step (`sumEigenbasisLyapunovMinibatch`): draws `batchB` sample
 * indices from `prng` exactly as `simulateEigenbasisLyapunovExponentReference` does (`floor(u N)` with `u = nextFloat01()`, in
 * order) and writes `sum[p] = (((0 + D_{i_0,p}) + D_{i_1,p}) + ...) * invB`, the value the reference forms as `sum[p] * invB` at
 * every use. The rows are added four at a time per pass over `p` (left-associative, so each partial sum is the sequential one)
 * and the scaling is folded into the last pass (`x * 1.0 = x` exactly for the other passes); `batchB <= 0` gives `0 * invB`, as
 * the reference does. `rows` is scratch of length at least `ceil(batchB)`.
 * @cold
 */
import type { createXorshift128Plus } from './prng-xorshift128p.ts';

export function sumEigenbasisLyapunovMinibatch(
  prng: ReturnType<typeof createXorshift128Plus>,
  packed: Float64Array,
  numSamples: number,
  nPair: number,
  batchB: number,
  invB: number,
  rows: Int32Array,
  sum: Float64Array,
): void {
  for (let b = 0; b < batchB; (b = (b + 1) | 0)) rows[b] = Math.imul(Math.floor(prng.nextFloat01() * numSamples) | 0, nPair);
  if (!(batchB > 0)) {
    for (let p = 0; p < nPair; (p = (p + 1) | 0)) sum[p] = 0.0 * invB;
    return;
  }
  const total = Math.ceil(batchB) | 0;
  for (let b = 0; b < total; b = (b + 4) | 0) {
    const count = total - b < 4 ? total - b : 4;
    const first = b === 0;
    const scale = b + count >= total ? invB : 1.0;
    const o0 = rows[b]!;
    const o1 = count > 1 ? rows[(b + 1) | 0]! : 0;
    const o2 = count > 2 ? rows[(b + 2) | 0]! : 0;
    const o3 = count > 3 ? rows[(b + 3) | 0]! : 0;
    if (count === 4) {
      for (let p = 0; p < nPair; (p = (p + 1) | 0)) {
        sum[p] = ((((first ? 0.0 : sum[p]!) + packed[(o0 + p) | 0]!) + packed[(o1 + p) | 0]!) + packed[(o2 + p) | 0]! + packed[(o3 + p) | 0]!) * scale;
      }
    } else if (count === 3) {
      for (let p = 0; p < nPair; (p = (p + 1) | 0)) {
        sum[p] = (((first ? 0.0 : sum[p]!) + packed[(o0 + p) | 0]!) + packed[(o1 + p) | 0]! + packed[(o2 + p) | 0]!) * scale;
      }
    } else if (count === 2) {
      for (let p = 0; p < nPair; (p = (p + 1) | 0)) sum[p] = ((first ? 0.0 : sum[p]!) + packed[(o0 + p) | 0]! + packed[(o1 + p) | 0]!) * scale;
    } else {
      for (let p = 0; p < nPair; (p = (p + 1) | 0)) sum[p] = ((first ? 0.0 : sum[p]!) + packed[(o0 + p) | 0]!) * scale;
    }
  }
}
