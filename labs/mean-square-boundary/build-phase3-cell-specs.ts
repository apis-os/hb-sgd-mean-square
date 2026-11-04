/**
 * @file labs/mean-square-boundary/build-phase3-cell-specs.ts
 * @description The Phase 3 cells (`buildPhase3CellSpecs`): every size in `sizes`, `beta` in {0, 0.9}, `B` in {4, 16, 64}, Gaussian (`kappa = 0`) and Rademacher (`kappa = -2`) i.i.d. features, with the power-iteration settings.
 * @cold
 */
import type { Phase3CellSpec } from './phase3-types.ts';

export function buildPhase3CellSpecs(
  sizes: readonly number[],
  maxIterations = 200000,
  relTol = 1e-13,
  relativeOffset = 1e-3,
  antisymmetricHorizon = 4096,
): Phase3CellSpec[] {
  const out: Phase3CellSpec[] = [];
  const betas = [0.0, 0.9];
  const batches = [4, 16, 64];
  const families: readonly ('gaussian-iid' | 'rademacher-iid')[] = ['gaussian-iid', 'rademacher-iid'];
  for (let si = 0; si < sizes.length; (si = (si + 1) | 0)) {
    for (let bi = 0; bi < betas.length; (bi = (bi + 1) | 0)) {
      for (let ki = 0; ki < batches.length; (ki = (ki + 1) | 0)) {
        for (let fi = 0; fi < families.length; (fi = (fi + 1) | 0)) {
          out.push({
            numParams: sizes[si]!,
            beta: betas[bi]!,
            betaIndex: bi,
            batchB: batches[ki]!,
            featureFamily: families[fi]!,
            familyIndex: fi,
            maxIterations,
            relTol,
            relativeOffset,
            antisymmetricHorizon,
          });
        }
      }
    }
  }
  return out;
}
