/**
 * @file labs/mean-square-boundary/fill-float64-view.ts
 * @description Fill the first `maxLength` entries of a genuine `Float64Array` (all of them if it is shorter) with `value` by indexed
 * writes (`fillFloat64View`): the zeroing and the refusal signal (`NaN` in the value, `Infinity` in the bound) of the Phase 6 noise
 * operators, which write only the `n^2` entries they validated (eleventh audit round: filling a longer view wrote its unchecked tail,
 * which could overlap the input). An argument that `readFloat64View` does not accept cannot carry the signal, so it is a contract
 * violation and throws a `TypeError` (ninth audit round: a `Proxy` or an `Int32Array` passed as output would otherwise receive a refusal
 * that reads as zeros or as nothing). An operator built with a non-finite dimension throws as well.
 * @cold
 */
import { readFloat64View } from './read-float64-view.ts';

export function fillFloat64View(view: Float64Array, value: number, maxLength: number): void {
  const r = readFloat64View(view, 0);
  if (r === null) throw new TypeError('Phase 6 noise operator: output must be a non-shared Float64Array');
  // A non-finite length comes from an operator built with a non-finite dimension: there is no validated prefix to signal on
  // (thirteenth audit round: NaN made the loop run zero times and left the outputs as they were).
  if (!(maxLength >= 0) || !Number.isFinite(maxLength)) throw new TypeError('Phase 6 noise operator: built with a non-finite dimension');
  const len = Math.min(r.length, maxLength);
  for (let i = 0; i < len; i++) view[i] = value;
}
