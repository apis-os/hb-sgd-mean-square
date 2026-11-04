/**
 * @file labs/mean-square-boundary/assemble-phase3-report.ts
 * @description Assembles the Phase 3 report (`assemblePhase3Report`) from the cell rows in spec order: the prediction texts `q1`–`q3`, written before the full run, their outcomes, and the provenance note on what was run before the predictions.
 * @cold
 */
import type { Phase3CellRow, Phase3CellSpec, Phase3Report } from './phase3-types.ts';

const PREDICTION_TEXTS: Readonly<Record<string, string>> = {
  q1: 'In every cell the radius of the second-moment operator is below 1 at (1 - 1e-3) eta* and above 1 at (1 + 1e-3) eta*, eta* the secular root of Theorem G.',
  q2: 'In every cell the crossing rho = 1, located from rho(eta*) - 1 and the local slope, is within 1e-8 (relative) of eta*.',
  q3: 'In every cell the radius on the antisymmetric matrices at eta* is strictly below 1, the radius on the symmetric part there (Lemma: the antisymmetric part never dominates).',
};

const PROVENANCE_NOTE =
  'A 4-cell prototype (P = 64 and 256, beta = 0 and 0.9) was run before q1-q3 were written; it gave rho(eta*) = 1 to 12 digits, so q1 and q2 were informed by it. After the predictions, a P = 16 debugging run showed that the ratio of norms does not converge on the antisymmetric part at beta = 0.9 (a cluster of eigenvalues of equal modulus), and the antisymmetric estimator was changed to the Gelfand estimate over a fixed horizon before the full run; that run gave antisymmetric radii of 0.44-0.61 at beta = 0 and about 0.906 at beta = 0.9, so q3 was seen at P = 16 before the full run. No P = 64, 256 or 1024 cell was run with the final code before the full run.';

export function assemblePhase3Report(cells: readonly Phase3CellRow[], spec: Phase3CellSpec): Phase3Report {
  let q1 = true;
  let q2 = true;
  let q3 = true;
  for (let i = 0; i < cells.length; (i = (i + 1) | 0)) {
    const c = cells[i]!;
    if (!(c.radiusBelow < 1.0 && c.radiusAbove > 1.0)) q1 = false;
    if (!(c.crossingRelErr < 1e-8)) q2 = false;
    if (!(c.radiusAntisymmetricAtRoot < 1.0)) q3 = false;
  }
  return {
    labId: 'mean-square-boundary-phase3',
    date: new Date().toISOString(),
    nodeVersion: process.version,
    relativeOffset: spec.relativeOffset,
    relTol: spec.relTol,
    maxIterations: spec.maxIterations,
    cells,
    provenanceNote: PROVENANCE_NOTE,
    predictionTexts: PREDICTION_TEXTS,
    predictions: {
      q1RadiusCrossesOneAtSecularRootEveryCell: q1,
      q2CrossingWithin1e8RelativeEveryCell: q2,
      q3AntisymmetricRadiusBelowOneAtRootEveryCell: q3,
    },
  };
}
