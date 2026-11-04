/**
 * @file labs/mean-square-boundary/restrict-eigenbasis-deviations.ts
 * @description The eigenbasis deviations of `buildEigenbasisSampleHessianDeviations` restricted to the modes of a 0/1 mask
 * (`restrictEigenbasisDeviations`): the kept eigenvalues in their original order, the deviations of the pairs inside the kept
 * modes re-packed with the same pair layout on `n' = |mask|` modes, and `modes` mapping a restricted index to the original one.
 * Used to remove decoupled null modes, and to restrict a threshold to the top modes; `qt` keeps only the rows of the kept modes.
 * @cold
 */
import type { buildEigenbasisSampleHessianDeviations } from './build-eigenbasis-sample-hessian-deviations.ts';

type Deviations = ReturnType<typeof buildEigenbasisSampleHessianDeviations>;

export function restrictEigenbasisDeviations(dev: Deviations, mask: Uint8Array): Deviations & { readonly modes: Int32Array } {
  const kept: number[] = [];
  for (let k = 0; k < dev.n; (k = (k + 1) | 0)) if (mask[k] === 1) kept.push(k);
  const n = kept.length | 0;
  const modes = Int32Array.from(kept);
  const nPair = ((n * (n + 1)) / 2) | 0;
  const pairIndex = new Int32Array(Math.imul(n, n));
  const pairA = new Int32Array(nPair);
  const pairB = new Int32Array(nPair);
  const source = new Int32Array(nPair);
  for (let a = 0; a < n; (a = (a + 1) | 0)) {
    for (let b = a; b < n; (b = (b + 1) | 0)) {
      const p = (Math.imul(a, n) - ((a * (a - 1)) / 2) + b - a) | 0;
      pairIndex[(Math.imul(a, n) + b) | 0] = p;
      pairIndex[(Math.imul(b, n) + a) | 0] = p;
      pairA[p] = a;
      pairB[p] = b;
      source[p] = dev.pairIndex[(Math.imul(modes[a]!, dev.n) + modes[b]!) | 0]!;
    }
  }
  const nS = dev.numSamples;
  const packed = new Float64Array(Math.imul(nS, nPair));
  for (let i = 0; i < nS; (i = (i + 1) | 0)) {
    const bIn = Math.imul(i, dev.nPair);
    const bOut = Math.imul(i, nPair);
    for (let p = 0; p < nPair; (p = (p + 1) | 0)) packed[(bOut + p) | 0] = dev.packed[(bIn + source[p]!) | 0]!;
  }
  const values = Float64Array.from(modes, (k) => dev.values[k]!);
  const qt = new Float64Array(Math.imul(n, dev.n));
  for (let a = 0; a < n; (a = (a + 1) | 0)) qt.set(dev.qt.subarray(Math.imul(modes[a]!, dev.n), Math.imul((modes[a]! + 1) | 0, dev.n)), Math.imul(a, dev.n));
  return { n, numSamples: nS, nPair, hSym: dev.hSym, values, qt, packed, pairIndex, pairA, pairB, modes };
}
