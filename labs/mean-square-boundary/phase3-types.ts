/**
 * @file labs/mean-square-boundary/phase3-types.ts
 * @description Type definitions for Phase 3 (`Phase3Report`): the spectral radius of the exact heavy-ball second-moment operator, by matrix-free power iteration from the identity, against the secular root of Theorem G at `P` up to 1024, with the radius of the antisymmetric part.
 * @cold
 */

export interface Phase3CellSpec {
  readonly numParams: number;
  readonly beta: number;
  readonly betaIndex: number;
  readonly batchB: number;
  readonly featureFamily: 'gaussian-iid' | 'rademacher-iid';
  readonly familyIndex: number;
  readonly maxIterations: number;
  readonly relTol: number;
  readonly relativeOffset: number;
  readonly antisymmetricHorizon: number;
}

export interface Phase3CellRow {
  readonly cellId: string;
  readonly numParams: number;
  readonly beta: number;
  readonly batchB: number;
  readonly featureFamily: 'gaussian-iid' | 'rademacher-iid';
  readonly kappaFeat: number;
  readonly etaCritSecular: number;
  readonly radiusBelow: number;
  readonly radiusAtRoot: number;
  readonly radiusAbove: number;
  readonly radiusAntisymmetricAtRoot: number;
  readonly crossingRelErr: number;
  readonly iterationsAtRoot: number;
  readonly allConverged: boolean;
}

export interface Phase3Report {
  readonly labId: 'mean-square-boundary-phase3';
  readonly date: string;
  readonly nodeVersion: string;
  readonly relativeOffset: number;
  readonly relTol: number;
  readonly maxIterations: number;
  readonly cells: readonly Phase3CellRow[];
  readonly provenanceNote: string;
  readonly predictionTexts: Readonly<Record<string, string>>;
  readonly predictions: {
    readonly q1RadiusCrossesOneAtSecularRootEveryCell: boolean;
    readonly q2CrossingWithin1e8RelativeEveryCell: boolean;
    readonly q3AntisymmetricRadiusBelowOneAtRootEveryCell: boolean;
  };
}
