/**
 * @file labs/mean-square-boundary/create-phase4-row-executor.ts
 * @description Dense `n x n` matrix products, transposes and row-wise Hessian applications for Phase 4
 * (`createPhase4RowExecutor`), split by contiguous row ranges over `numWorkers` worker threads that share the buffers
 * (`numWorkers = 0` runs the JavaScript kernels in the calling thread; workers use the Wasm `f64x2` product, whose entries
 * accumulate in plain increasing `k` and so differ from the in-thread product at rounding level). Every kernel computes each
 * row by the same arithmetic whatever the range it falls in, so results are identical for any worker count. Buffers that a worker touches must come
 * from `alloc`.
 * @cold
 */
import { Worker } from 'node:worker_threads';
import { applyHessianRows } from './apply-hessian-rows.ts';
import { createPhase4SampleHessianProvider } from './create-phase4-sample-hessian-provider.ts';
import { multiplyDenseRows } from './multiply-dense-rows.ts';
import type { Phase4NetworkSpec, Phase4SampleHessianProvider } from './phase4-types.ts';
import { transposeDenseRows } from './transpose-dense-rows.ts';

export async function createPhase4RowExecutor(
  spec: Phase4NetworkSpec,
  numWorkers: number,
): Promise<{
  readonly n: number;
  readonly provider: Phase4SampleHessianProvider;
  alloc(): Float64Array;
  matmul(a: Float64Array, b: Float64Array, c: Float64Array): Promise<void>;
  matmulSymmetric(a: Float64Array, b: Float64Array, c: Float64Array): Promise<void>;
  transpose(src: Float64Array, dst: Float64Array): Promise<void>;
  hessianRows(sample: number, src: Float64Array, dst: Float64Array, scale: number, accumulate: boolean): Promise<void>;
  close(): Promise<void>;
}> {
  const provider = createPhase4SampleHessianProvider(spec);
  const n = provider.numParams;
  const tmp = new Float64Array(n);
  const workers: Worker[] = [];
  const url = new URL('./phase4-row-worker.ts', import.meta.url);
  const ready: Array<Promise<unknown>> = [];
  for (let w = 0; w < numWorkers; (w = (w + 1) | 0)) {
    const worker = new Worker(url, { workerData: spec });
    workers.push(worker);
    ready.push(new Promise((res) => worker.once('message', res)));
  }
  await Promise.all(ready);
  const chunk = numWorkers > 0 ? Math.ceil(n / numWorkers) : n;

  // Row boundaries with equal work for an upper-triangular product (row i costs n - i), rounded to multiples of 4.
  const upperBounds: number[] = [0];
  const totalWork = (n * (n + 1)) / 2;
  for (let w = 1; w < numWorkers; (w = (w + 1) | 0)) {
    const target = (totalWork * w) / numWorkers;
    let r = upperBounds[upperBounds.length - 1]!;
    while (r < n && r * n - (r * (r - 1)) / 2 < target) r = (r + 1) | 0;
    upperBounds.push(Math.min(n, (r + 3) & -4));
  }
  upperBounds.push(n);

  function dispatch(build: (r0: number, r1: number) => object, bounds?: readonly number[]): Promise<void> {
    const pending: Array<Promise<unknown>> = [];
    for (let w = 0; w < workers.length; (w = (w + 1) | 0)) {
      const r0 = bounds === undefined ? Math.min(n, Math.imul(w, chunk)) : bounds[w]!;
      const r1 = bounds === undefined ? Math.min(n, (r0 + chunk) | 0) : bounds[w + 1]!;
      if (r1 <= r0) continue;
      const worker = workers[w]!;
      pending.push(new Promise((res) => worker.once('message', res)));
      worker.postMessage(build(r0, r1));
    }
    return Promise.all(pending).then(() => undefined);
  }

  return {
    n,
    provider,
    alloc: () => (numWorkers > 0 ? new Float64Array(new SharedArrayBuffer(Math.imul(n, n) * 8)) : new Float64Array(Math.imul(n, n))),
    matmul: (a, b, c) => {
      if (numWorkers === 0) {
        multiplyDenseRows(a, b, c, n, 0, n);
        return Promise.resolve();
      }
      return dispatch((r0, r1) => ({ op: 'matmulWasm', a, b, c, upper: 0, r0, r1 }));
    },
    // C = A B for a product known to be symmetric. In workers: the upper triangle by the Wasm kernel, then the mirror (exactly
    // symmetric). In the calling thread: the full product symmetrised by averaging, as before the Wasm kernel, so the in-thread
    // numbers (P = 33) are unchanged.
    matmulSymmetric: async (a, b, c) => {
      if (numWorkers === 0) {
        multiplyDenseRows(a, b, c, n, 0, n);
        for (let j = 0; j < n; (j = (j + 1) | 0)) {
          for (let k = (j + 1) | 0; k < n; (k = (k + 1) | 0)) {
            const v = 0.5 * (c[(Math.imul(j, n) + k) | 0]! + c[(Math.imul(k, n) + j) | 0]!);
            c[(Math.imul(j, n) + k) | 0] = v;
            c[(Math.imul(k, n) + j) | 0] = v;
          }
        }
        return;
      }
      await dispatch((r0, r1) => ({ op: 'matmulWasm', a, b, c, upper: 1, r0, r1 }), upperBounds);
      await dispatch((r0, r1) => ({ op: 'mirror', c, r0, r1 }));
    },
    transpose: (src, dst) => {
      if (numWorkers === 0) {
        transposeDenseRows(src, dst, n, 0, n);
        return Promise.resolve();
      }
      return dispatch((r0, r1) => ({ op: 'transpose', src, dst, r0, r1 }));
    },
    hessianRows: (sample, src, dst, scale, accumulate) => {
      if (numWorkers === 0) {
        applyHessianRows(provider, sample, src, dst, n, 0, n, scale, accumulate, tmp);
        return Promise.resolve();
      }
      return dispatch((r0, r1) => ({ op: 'hessian', sample, src, dst, scale, accumulate, r0, r1 }));
    },
    close: async () => {
      await Promise.all(workers.map((w) => w.terminate()));
    },
  };
}
