/**
 * @file labs/mean-square-boundary/copy-float64-view.ts
 * @description A private copy of the first `length` entries of a Phase 6 input (`copyFloat64View`), or `null` when
 * `readFloat64View` refuses it. The copy is taken by indexed reads, which on a genuine typed array go to its elements without calling
 * any user code, into a fresh `Float64Array`; every later read uses the copy, so no other thread, alias or callback can change the
 * input between validation and use (ninth audit round).
 * @cold
 */
import { readFloat64View } from './read-float64-view.ts';

export function copyFloat64View(view: Float64Array, length: number): Float64Array | null {
  if (readFloat64View(view, length) === null) return null;
  const out = new Float64Array(length);
  for (let i = 0; i < length; i++) out[i] = view[i]!;
  return out;
}
