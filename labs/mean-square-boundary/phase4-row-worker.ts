/**
 * @file labs/mean-square-boundary/phase4-row-worker.ts
 * @description Worker-thread entry of `createPhase4RowExecutor`: builds the sample-Hessian provider of its `workerData`
 * spec, then runs the row kernels (`multiplyDenseRows`, the Wasm `f64x2` product of `compileDenseF64MatmulKernel`,
 * `transposeDenseRows`, `applyHessianRows`, and the mirror of an upper triangle) on the row range of each message over shared
 * buffers and answers once the range is done.
 * @cold
 */
import { parentPort, workerData } from 'node:worker_threads';
import { compileDenseF64MatmulKernel } from './compile-dense-f64-matmul-kernel.ts';
import { applyHessianRows } from './apply-hessian-rows.ts';
import { createPhase4SampleHessianProvider } from './create-phase4-sample-hessian-provider.ts';
import { multiplyDenseRows } from './multiply-dense-rows.ts';
import type { Phase4NetworkSpec } from './phase4-types.ts';
import { transposeDenseRows } from './transpose-dense-rows.ts';

type RowMessage =
  | { readonly op: 'matmul'; readonly a: Float64Array; readonly b: Float64Array; readonly c: Float64Array; readonly r0: number; readonly r1: number }
  | { readonly op: 'transpose'; readonly src: Float64Array; readonly dst: Float64Array; readonly r0: number; readonly r1: number }
  | { readonly op: 'matmulWasm'; readonly a: Float64Array; readonly b: Float64Array; readonly c: Float64Array; readonly upper: number; readonly r0: number; readonly r1: number }
  | { readonly op: 'mirror'; readonly c: Float64Array; readonly r0: number; readonly r1: number }
  | {
      readonly op: 'hessian';
      readonly sample: number;
      readonly src: Float64Array;
      readonly dst: Float64Array;
      readonly scale: number;
      readonly accumulate: boolean;
      readonly r0: number;
      readonly r1: number;
    };

const spec = workerData as Phase4NetworkSpec;
const provider = createPhase4SampleHessianProvider(spec);
const n = provider.numParams;
const tmp = new Float64Array(n);
// Wasm f64x2 product: B, a block of rows of A and of C, and the two packing buffers in one memory, laid out for up to n rows.
// TurboFan is forced by starting the process with `--no-liftoff` (a single long call would otherwise stay in Liftoff, about
// 2.5x slower). The flag is no longer changed at run time from this thread: a process-wide flag change while other isolates
// compile is the suspected cause (not reproduced) of the V8 check failure in the JS compiler that ended the sixth Phase 4 launch.
let wasm: { readonly mem: Float64Array; readonly fn: (...args: number[]) => number } | null = null;
const nn = Math.imul(n, n);
const offB = 0;
const offA = nn;
const offC = 2 * nn;
const offPackB = 3 * nn;
const offPackA = offPackB + 131072;
function getWasm(): { readonly mem: Float64Array; readonly fn: (...args: number[]) => number } {
  if (wasm !== null) return wasm;
  const bytes = (offPackA + 256 * n + 16) * 8;
  const memory = new WebAssembly.Memory({ initial: Math.ceil(bytes / 65536) });
  const inst = new WebAssembly.Instance(new WebAssembly.Module(compileDenseF64MatmulKernel()), { env: { memory } });
  const name = Object.keys(inst.exports).find((key) => typeof inst.exports[key] === 'function');
  if (name === undefined) throw new Error('matmul kernel exports no function');
  wasm = { mem: new Float64Array(memory.buffer), fn: inst.exports[name] as (...args: number[]) => number };
  return wasm;
}
parentPort?.on('message', (m: RowMessage) => {
  if (m.op === 'matmul') multiplyDenseRows(m.a, m.b, m.c, n, m.r0, m.r1);
  else if (m.op === 'matmulWasm') {
    const w = getWasm();
    const rows = (m.r1 - m.r0) | 0;
    w.mem.set(m.b, offB);
    w.mem.set(m.a.subarray(Math.imul(m.r0, n), Math.imul(m.r1, n)), offA);
    w.fn(offA * 8, offB * 8, offC * 8, n, rows, m.r0, m.upper, offPackB * 8, offPackA * 8);
    m.c.set(w.mem.subarray(offC, offC + Math.imul(rows, n)), Math.imul(m.r0, n));
  } else if (m.op === 'mirror') {
    const tile = 32;
    for (let jb = m.r0; jb < m.r1; jb = (jb + tile) | 0) {
      const jEnd = jb + tile < m.r1 ? (jb + tile) | 0 : m.r1;
      for (let kb = 0; kb < jEnd; kb = (kb + tile) | 0) {
        for (let j = jb > kb ? jb : kb; j < jEnd; j = (j + 1) | 0) {
          const row = Math.imul(j, n);
          const kEnd = kb + tile < j ? (kb + tile) | 0 : j;
          let srcIdx = (Math.imul(kb, n) + j) | 0;
          for (let k = kb; k < kEnd; k = (k + 1) | 0) {
            m.c[(row + k) | 0] = m.c[srcIdx]!;
            srcIdx = (srcIdx + n) | 0;
          }
        }
      }
    }
  }
  else if (m.op === 'transpose') transposeDenseRows(m.src, m.dst, n, m.r0, m.r1);
  else applyHessianRows(provider, m.sample, m.src, m.dst, n, m.r0, m.r1, m.scale, m.accumulate, tmp);
  parentPort?.postMessage(0);
});
parentPort?.postMessage(0);
