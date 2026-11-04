/**
 * @file labs/mean-square-boundary/compile-dense-f64-matmul-kernel.ts
 * @description The WebAssembly SIMD `f64x2` 4x4 register-blocked product kernel used by the Phase 4 row workers
 * (`compileDenseF64MatmulKernel`), returned as the bytes of `dense-f64-matmul.wasm` (SHA-256
 * `17fd94985b6c7f80a3a28266adac6685b25c3731382a2c20d9756e33eb934714`). The module was compiled ahead of time by the code
 * generator of the companion lab, which is not part of this repository; `tests/phase4-tethered-thm-g6.test.ts` checks it
 * against the sequential product bit for bit.
 * @cold
 */
import fs from 'node:fs';

export function compileDenseF64MatmulKernel(): Uint8Array<ArrayBuffer> {
  return new Uint8Array(fs.readFileSync(new URL('./dense-f64-matmul.wasm', import.meta.url)));
}
