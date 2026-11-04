/**
 * @file labs/mean-square-boundary/decide-phase6-regeneration-mismatch.ts
 * @description The final regeneration check of the Phase 6 stage runners (`decidePhase6RegenerationMismatch`), pure so that it is
 * tested without a training run (tenth audit round: neither runner was exercised by any test). Given the mismatch already found during
 * regeneration (`null` if none), the divergence step of the regeneration and of the stored run, and pairs `[seen, expected]` of counts
 * (checkpoints compared against stored, records produced against requested), it returns the step of the first mismatch, `-1` when the
 * regeneration stayed finite where the stored run diverged, or `null` when everything agrees.
 * @cold
 */
export function decidePhase6RegenerationMismatch(
  found: number | null,
  regeneratedDivergedAt: number | null,
  storedDivergedAt: number | null,
  counts: ReadonlyArray<readonly [number, number]>,
): number | null {
  if (found !== null) return found;
  let agree = regeneratedDivergedAt === storedDivergedAt;
  for (let i = 0; i < counts.length; i++) if (counts[i]![0] !== counts[i]![1]) agree = false;
  return agree ? null : regeneratedDivergedAt ?? -1;
}
