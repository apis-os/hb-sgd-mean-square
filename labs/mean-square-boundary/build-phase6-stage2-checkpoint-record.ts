/**
 * @file labs/mean-square-boundary/build-phase6-stage2-checkpoint-record.ts
 * @description The record of one stage 2 checkpoint of Phase 6 (`buildPhase6Stage2CheckpointRecord`), pure over its arguments so
 * that its rules are tested without a pilot run (fourth audit round: the rules lived inside `runPhase6Stage2Run`, which no test
 * calls, and deleting the compressed-rank rule survived every test).
 * - Compressed row space (`rank < m`): the bracket is of a Rayleigh compression of the iteration on the full row space, which bounds
 *   `eta*` from above only, so every lower bound is `null`, in the top-level fields and inside `bracket` and `reference` (fourth
 *   audit round: the nested summaries kept the compressed `etaLow`).
 * - `validAgainstReference`: both brackets are certified for the same operator, so `low < reference up` and
 *   `reference low < up` must hold with no tolerance (fourth audit round: a `1e-8` slack could hide a wrong-side bracket);
 *   `null` when no reference was requested, `false` when one was requested and a bracket or a lower bound is missing.
 * - `validAgainstTop64`: `low <= top64 (1 + 1e-8)`, against the threshold of the top-64 modes stored by the pilot; that value is an
 *   uncertified Lanczos estimate of an upper bound of `eta*` (a compression), so the tolerance stays, and the flag is a consistency
 *   check, not part of the proof.
 * - `reading` of rule 3 against `[0.8 eta, 1.25 eta]`: `inside`, `below`, `above` or `undecided`.
 * - `meanResidualFrobeniusOverMinCurvature` (`||fl(Hbar) - diag(lambda)||_F / min lt`) and `dualResidualOverTop`
 *   (`max_k ||G v_k - lambda_k v_k||_2 / lambda_0`, computed in floating point) are diagnostics of the distance from the certified
 *   factors to the Gauss–Newton operator of the checkpoint; the bracket is certified for the stored factors and `fl(lambda + mu)`.
 * @cold
 */
type Bracket = {
  readonly etaLow: number | null;
  readonly etaUp: number;
  readonly etaDet: number;
  readonly ritzEta: number;
  readonly ritzResidual: number;
  readonly lanczosSteps: number;
  readonly witnessPositiveDefinite: boolean;
  readonly bestEps: number | null;
};

function summarise(br: Bracket | null, compressed: boolean, seconds: number): Record<string, unknown> | null {
  if (br === null) return null;
  const low = compressed ? null : br.etaLow;
  return { etaLow: low, etaUp: br.etaUp, etaDet: br.etaDet, ritzEta: br.ritzEta, ritzResidual: br.ritzResidual, lanczosSteps: br.lanczosSteps, witnessPositiveDefinite: br.witnessPositiveDefinite, lowOverUp: low === null ? null : low / br.etaUp, bestEps: compressed ? null : br.bestEps, seconds };
}

export function buildPhase6Stage2CheckpointRecord(a: {
  readonly step: number;
  readonly eta: number;
  readonly rank: number;
  readonly m: number;
  readonly bracket: Bracket | null;
  readonly bracketSeconds: number;
  readonly lengthsTried: readonly number[];
  readonly reference: Bracket | null;
  readonly referenceSeconds: number;
  readonly referenceRequested: boolean;
  readonly top64: number | null;
  readonly meanResidualOverMinCurvature: number;
  readonly dualResidualOverTop: number;
  readonly spectrumSeconds: number;
}): Record<string, unknown> {
  const compressed = a.rank < a.m;
  const br = a.bracket;
  const refBr = a.reference;
  const lo = compressed ? null : br?.etaLow ?? null;
  const refLo = compressed ? null : refBr?.etaLow ?? null;
  const up = br?.etaUp ?? Infinity;
  const reading = lo !== null && 0.8 * a.eta <= lo && up <= 1.25 * a.eta ? 'inside' : up < 0.8 * a.eta ? 'below' : lo !== null && lo > 1.25 * a.eta ? 'above' : 'undecided';
  const validAgainstReference = !a.referenceRequested ? null : br !== null && refBr !== null && lo !== null && refLo !== null && lo < refBr.etaUp && refLo < br.etaUp;
  // null, not false, on a compressed row space, which has no lower bound to compare (ninth audit round).
  const validAgainstTop64 = a.top64 === null || compressed ? null : lo !== null && lo <= a.top64 * (1 + 1e-8);
  return {
    step: a.step,
    rank: a.rank,
    rankCompressed: compressed,
    meanResidualFrobeniusOverMinCurvature: a.meanResidualOverMinCurvature,
    dualResidualOverTop: a.dualResidualOverTop,
    spectrumSeconds: a.spectrumSeconds,
    bracket: br === null ? null : { ...summarise(br, compressed, a.bracketSeconds), lengthsTried: a.lengthsTried.slice() },
    reference: a.referenceRequested ? summarise(refBr, compressed, a.referenceSeconds) : null,
    top64Stored: a.top64,
    upOverTop64: a.top64 === null || br === null ? null : br.etaUp / a.top64,
    lowOverEta: lo === null ? null : lo / a.eta,
    upOverEta: br === null ? null : br.etaUp / a.eta,
    reading,
    validAgainstReference,
    validAgainstTop64,
  };
}
