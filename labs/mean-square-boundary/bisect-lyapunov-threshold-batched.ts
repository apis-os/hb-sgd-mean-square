/**
 * @file labs/mean-square-boundary/bisect-lyapunov-threshold-batched.ts
 * @description `bisectLyapunovThreshold` with a batched exponent (`bisectLyapunovThresholdBatched`): `gammaBatch(etas)` returns the
 * exponent at each of several step sizes (one lockstep simulation, `simulateEigenbasisLyapunovExponentsBatch`). The ends are one
 * call; without a sign change the profile grid is one call; with a sign change two bisection steps are one call of three points,
 * `[q1, mid, q3]` with `mid = (a + b) / 2`, `q1 = (a + mid) / 2`, `q3 = (mid + b) / 2` on `log eta`, the two possible second
 * midpoints, exactly the numbers the sequential bisection forms; an odd last step is one call of one point. Since the exponent at
 * a step size does not depend on the others in the call, `etaAs`, `bracket`, `evaluations` (same entries, same order) and
 * `signProfile` are bit-identical to the sequential function with the same exponent.
 * @cold
 */
import { fdlibmExp, fdlibmLog } from '../../src/math/index.ts';

type Gamma = { readonly mean: number; readonly standardError: number };

export function bisectLyapunovThresholdBatched(
  gammaBatch: (etas: readonly number[]) => ReadonlyArray<Gamma>,
  lo: number,
  hi: number,
  steps: number,
  profileGrid: readonly number[],
): {
  readonly etaAs: number | null;
  readonly bracket: readonly [number, number] | null;
  readonly evaluations: ReadonlyArray<{ readonly eta: number; readonly mean: number; readonly standardError: number }>;
  readonly signProfile: ReadonlyArray<{ readonly eta: number; readonly mean: number; readonly standardError: number }> | null;
} {
  const evaluations: Array<{ eta: number; mean: number; standardError: number }> = [];
  const ends = gammaBatch([lo, hi]);
  evaluations.push({ eta: lo, mean: ends[0]!.mean, standardError: ends[0]!.standardError });
  evaluations.push({ eta: hi, mean: ends[1]!.mean, standardError: ends[1]!.standardError });
  if (!(ends[0]!.mean < 0.0 && ends[1]!.mean > 0.0)) {
    const prof = gammaBatch(profileGrid);
    const signProfile = profileGrid.map((eta, i) => ({ eta, mean: prof[i]!.mean, standardError: prof[i]!.standardError }));
    return { etaAs: null, bracket: null, evaluations, signProfile };
  }
  let a = fdlibmLog(lo);
  let b = fdlibmLog(hi);
  let s = 0;
  while (s < steps) {
    const mid = 0.5 * (a + b);
    if (s + 1 < steps) {
      const q1 = 0.5 * (a + mid);
      const q3 = 0.5 * (mid + b);
      const e1 = fdlibmExp(q1);
      const e2 = fdlibmExp(mid);
      const e3 = fdlibmExp(q3);
      const gs = gammaBatch([e1, e2, e3]);
      evaluations.push({ eta: e2, mean: gs[1]!.mean, standardError: gs[1]!.standardError });
      const second = gs[1]!.mean > 0.0 ? 0 : 2;
      if (second === 0) b = mid;
      else a = mid;
      const eSecond = second === 0 ? e1 : e3;
      const qSecond = second === 0 ? q1 : q3;
      evaluations.push({ eta: eSecond, mean: gs[second]!.mean, standardError: gs[second]!.standardError });
      if (gs[second]!.mean > 0.0) b = qSecond;
      else a = qSecond;
      s = (s + 2) | 0;
    } else {
      const e2 = fdlibmExp(mid);
      const g = gammaBatch([e2])[0]!;
      evaluations.push({ eta: e2, mean: g.mean, standardError: g.standardError });
      if (g.mean > 0.0) b = mid;
      else a = mid;
      s = (s + 1) | 0;
    }
  }
  return { etaAs: fdlibmExp(0.5 * (a + b)), bracket: [fdlibmExp(a), fdlibmExp(b)], evaluations, signProfile: null };
}
