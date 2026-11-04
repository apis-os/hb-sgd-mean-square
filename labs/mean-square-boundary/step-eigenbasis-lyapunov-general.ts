/**
 * @file labs/mean-square-boundary/step-eigenbasis-lyapunov-general.ts
 * @description One renormalised linearised heavy-ball step for any number `G` of step sizes sharing one scaled minibatch deviation
 * (`stepEigenbasisLyapunovGeneral`): for each `g`, `r = l x_g + D_B x_g`, `x_g <- x_g + beta (x_g - xPrev_g) - etas[g] r`, the
 * log norm of `(x_g, xPrev_g)` added to `logs[g]` when `recordLog`, and renormalisation. `x`, `xPrev` hold the `G` states one
 * after the other (`n` entries each), `r` is scratch of length `n`, `sum` the output of `sumEigenbasisLyapunovMinibatch`. With
 * `rowMajor` (`isRowMajorPairLayout`) the product runs row by row with the row entry in a register; otherwise it is the pair loop
 * of `simulateEigenbasisLyapunovExponentReference`. Both add the same terms to every `r[k]` in the same order, so each state is
 * bit-identical to a reference run with that step size alone. Used for `G >= 4` and for other pair layouts; `G = 1, 2, 3` on the
 * row-major layout use the generated kernels (`stepEigenbasisLyapunovG1`, `stepEigenbasisLyapunovG2`, `stepEigenbasisLyapunovG3`).
 * @cold
 */
import { fdlibmLog } from '../../src/math/index.ts';

export function stepEigenbasisLyapunovGeneral(
  sum: Float64Array,
  curvature: Float64Array,
  pairA: Int32Array,
  pairB: Int32Array,
  rowMajor: boolean,
  x: Float64Array,
  xPrev: Float64Array,
  r: Float64Array,
  n: number,
  nPair: number,
  beta: number,
  etas: Float64Array,
  logs: Float64Array,
  recordLog: boolean,
): void {
  for (let g = 0; g < etas.length; (g = (g + 1) | 0)) {
    const off = Math.imul(g, n);
    for (let k = 0; k < n; (k = (k + 1) | 0)) r[k] = curvature[k]! * x[(off + k) | 0]!;
    if (rowMajor) {
      let p = 0;
      for (let a = 0; a < n; (a = (a + 1) | 0)) {
        const xa = x[(off + a) | 0]!;
        let ra = r[a]! + sum[p]! * xa;
        p = (p + 1) | 0;
        for (let c = (a + 1) | 0; c < n; (c = (c + 1) | 0)) {
          const v = sum[p]!;
          p = (p + 1) | 0;
          ra += v * x[(off + c) | 0]!;
          r[c] = r[c]! + v * xa;
        }
        r[a] = ra;
      }
    } else {
      for (let p = 0; p < nPair; (p = (p + 1) | 0)) {
        const a = pairA[p]!;
        const c = pairB[p]!;
        const v = sum[p]!;
        r[a] = r[a]! + v * x[(off + c) | 0]!;
        if (a !== c) r[c] = r[c]! + v * x[(off + a) | 0]!;
      }
    }
    const eta = etas[g]!;
    let nrm = 0.0;
    for (let k = 0; k < n; (k = (k + 1) | 0)) {
      const cur = x[(off + k) | 0]!;
      const next = cur + beta * (cur - xPrev[(off + k) | 0]!) - eta * r[k]!;
      xPrev[(off + k) | 0] = cur;
      x[(off + k) | 0] = next;
      nrm += next * next + cur * cur;
    }
    const nu = Math.sqrt(nrm);
    if (recordLog) logs[g] = logs[g]! + fdlibmLog(nu);
    const inv = 1.0 / nu;
    for (let k = 0; k < n; (k = (k + 1) | 0)) {
      x[(off + k) | 0] = x[(off + k) | 0]! * inv;
      xPrev[(off + k) | 0] = xPrev[(off + k) | 0]! * inv;
    }
  }
}
