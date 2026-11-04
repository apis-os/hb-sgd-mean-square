/**
 * @file labs/mean-square-boundary/compute-adaptive-certified-threshold-bracket.ts
 * @description The Lanczos-length policy of Phase 6 around `computeCertifiedThresholdBracket`
 * (`computeAdaptiveCertifiedThresholdBracket`): 30 steps; 60 when the Ritz residual is above `1e-3` of the Ritz value or
 * `eta_low / eta_up < 0.98`; 120 when the bracket is still below 0.98. The plan's rule (section 4.1) was 30, then 60 on the residual
 * alone; the bracket criterion and the third length were added after stage 1, where four Gauss–Newton checkpoints with relative
 * residual about `2e-4` after 30 steps gave `eta_low / eta_up` between 0.70 and 0.94 (0.9994 at 60 steps on the worst). Every
 * length gives valid bounds; the policy only spends more steps where the bracket is wide. Returns the last bracket and the lengths
 * tried. A length that Lanczos does not use up stops the policy. That happens when its Krylov space is exhausted (as in dimension 1) and
 * also when the Ritz residual falls below the tolerance or the next vector is numerically small; in the last two cases more steps could
 * still change the Ritz pair, so the early stop is a policy choice, not a proof (thirteenth audit round). The 120-step arm has no test (ninth audit round): no Phase 5b checkpoint tried reaches it (the widest `|H|` checkpoints of
 * the first stage 1 run close at 60 steps with the `eps` ladder); it may be reached at the row-space sizes of stage 2.
 * @cold
 */
import { computeCertifiedThresholdBracket } from './compute-certified-threshold-bracket.ts';

export function computeAdaptiveCertifiedThresholdBracket(
  lt: Float64Array,
  n: number,
  beta: number,
  noise: { readonly n?: number; apply(x: Float64Array, out: Float64Array, errOut: Float64Array | null): void },
): { readonly bracket: ReturnType<typeof computeCertifiedThresholdBracket>; readonly lengthsTried: number[] } {
  const lengths = [30, 60, 120];
  const tried: number[] = [];
  let br: ReturnType<typeof computeCertifiedThresholdBracket> = null;
  for (let i = 0; i < lengths.length; i++) {
    br = computeCertifiedThresholdBracket(lt, n, beta, noise, lengths[i]!, 1e-12);
    tried.push(lengths[i]!);
    if (br === null) break;
    if (br.lanczosSteps < lengths[i]!) break;
    const narrow = br.etaLow !== null && br.etaLow / br.etaUp >= 0.98;
    const converged = br.ritzResidual <= 1e-3 / br.ritzEta;
    if (i === 0 ? narrow && converged : narrow) break;
  }
  return { bracket: br, lengthsTried: tried };
}
