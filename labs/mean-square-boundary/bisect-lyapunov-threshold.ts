/**
 * @file labs/mean-square-boundary/bisect-lyapunov-threshold.ts
 * @description The almost-sure threshold of Phase 5 (h) (`bisectLyapunovThreshold`): with `gamma(eta)` the top Lyapunov exponent
 * of a linearised iteration (mean and standard error, from `simulateEigenbasisLyapunovExponent` with common random numbers
 * across `eta`), evaluate the two ends of `[lo, hi]`; if `gamma(lo) < 0 < gamma(hi)` (by the mean), bisect on `log eta` for
 * `steps` steps and return the geometric midpoint of the last bracket; otherwise return `null` and the sign profile of `gamma`
 * on `profileGrid`. Every evaluation is returned. With `gammaBatch` (the exponent at several step sizes from one lockstep
 * simulation) the same result is computed by `bisectLyapunovThresholdBatched`, two bisection steps per call.
 * @cold
 */
import { fdlibmExp, fdlibmLog } from '../../src/math/index.ts';
import { bisectLyapunovThresholdBatched } from './bisect-lyapunov-threshold-batched.ts';

export function bisectLyapunovThreshold(
  gamma: (eta: number) => { readonly mean: number; readonly standardError: number },
  lo: number,
  hi: number,
  steps: number,
  profileGrid: readonly number[],
  gammaBatch?: (etas: readonly number[]) => ReadonlyArray<{ readonly mean: number; readonly standardError: number }>,
): {
  readonly etaAs: number | null;
  readonly bracket: readonly [number, number] | null;
  readonly evaluations: ReadonlyArray<{ readonly eta: number; readonly mean: number; readonly standardError: number }>;
  readonly signProfile: ReadonlyArray<{ readonly eta: number; readonly mean: number; readonly standardError: number }> | null;
} {
  if (gammaBatch !== undefined) return bisectLyapunovThresholdBatched(gammaBatch, lo, hi, steps, profileGrid);
  const evaluations: Array<{ eta: number; mean: number; standardError: number }> = [];
  const at = (eta: number): number => {
    const g = gamma(eta);
    evaluations.push({ eta, mean: g.mean, standardError: g.standardError });
    return g.mean;
  };
  const gLo = at(lo);
  const gHi = at(hi);
  if (!(gLo < 0.0 && gHi > 0.0)) {
    const signProfile = profileGrid.map((eta) => {
      const g = gamma(eta);
      return { eta, mean: g.mean, standardError: g.standardError };
    });
    return { etaAs: null, bracket: null, evaluations, signProfile };
  }
  let a = fdlibmLog(lo);
  let b = fdlibmLog(hi);
  for (let s = 0; s < steps; (s = (s + 1) | 0)) {
    const mid = 0.5 * (a + b);
    if (at(fdlibmExp(mid)) > 0.0) b = mid;
    else a = mid;
  }
  return { etaAs: fdlibmExp(0.5 * (a + b)), bracket: [fdlibmExp(a), fdlibmExp(b)], evaluations, signProfile: null };
}
