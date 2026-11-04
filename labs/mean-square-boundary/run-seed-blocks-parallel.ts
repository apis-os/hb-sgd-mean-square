/**
 * @file labs/mean-square-boundary/run-seed-blocks-parallel.ts
 * @description Runs seed-block jobs of one worker entry file on a pool of `node:worker_threads` workers
 * (`runSeedBlocksParallel`). Results come back in job order, so the aggregation and the raw file do not depend on
 * the worker count; each job is posted as `workerData` and the worker answers with one block.
 * @cold
 */
import { Worker } from 'node:worker_threads';

export function runSeedBlocksParallel<TJob, TBlock>(
  workerFile: string,
  jobs: readonly TJob[],
  maxWorkers: number,
): Promise<Array<TBlock>> {
  const workerUrl = new URL(workerFile, import.meta.url);
  const results: Array<TBlock> = new Array(jobs.length);
  let next = 0;
  let done = 0;
  return new Promise((resolve, reject) => {
    if (jobs.length === 0) {
      resolve(results);
      return;
    }
    function launch(): void {
      if (next >= jobs.length) return;
      const idx = next;
      next = (next + 1) | 0;
      const w = new Worker(workerUrl, { workerData: jobs[idx] });
      w.once('message', (msg: TBlock) => {
        results[idx] = msg;
        done = (done + 1) | 0;
        void w.terminate();
        if (done === jobs.length) resolve(results);
        else launch();
      });
      w.once('error', reject);
    }
    const n = Math.max(1, Math.min(maxWorkers, jobs.length));
    for (let i = 0; i < n; (i = (i + 1) | 0)) launch();
  });
}
