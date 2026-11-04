/**
 * @file labs/mean-square-boundary/bisect-cone-lower-threshold.ts
 * @description Certified lower bound of the threshold of Theorem G.6 by bisection on `certifyConeCollatzWielandtAtRate`
 * (`bisectConeLowerThreshold`): on `(start, hi)` (`start` 0 by default, or a rate already certified with another test matrix),
 * `steps` halvings, keeping the largest rate at which the certificate passed. Each pass
 * proves `eta* > eta` on its own (the certificate does not need monotonicity), so the result is a lower bound whatever the pattern
 * of passes; `etaLow` is `null` when no tested rate passed (the caller keeps its own earlier bound). `w`, `wErr` are `fl(V(x))` and its bound, computed once by the caller.
 * @cold
 */
import { certifyConeCollatzWielandtAtRate } from './certify-cone-collatz-wielandt-at-rate.ts';

export function bisectConeLowerThreshold(
  lt: Float64Array,
  n: number,
  beta: number,
  x: Float64Array,
  w: Float64Array,
  wErr: Float64Array,
  hi: number,
  steps: number,
  start = 0,
): { readonly etaLow: number | null; readonly passes: number } {
  const m = new Float64Array(n * n);
  const work = new Float64Array(n * n);
  let lo = start;
  let top = hi;
  let best: number | null = null;
  let passes = 0;
  for (let s = 0; s < steps; s++) {
    const mid = 0.5 * (lo + top);
    if (certifyConeCollatzWielandtAtRate(lt, n, beta, x, w, wErr, mid, m, work)) {
      lo = mid;
      best = mid;
      passes++;
    } else {
      top = mid;
    }
  }
  return { etaLow: best, passes };
}
