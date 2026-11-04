/**
 * @file labs/mean-square-boundary/phase2-types.ts
 * @description Row and report types of Phase 2 (`Phase2GrowthGridRow`, `Phase2NetworkCell`, `Phase2Report`): exact Hessian trace and noise-rank ratio of the scale networks, the random-feature secular bracket calibrated on `(lambda_1, lambda_2, tr H, nu_11)`, the destructive minibatch heavy-ball ceiling in `R^P`, the finite-difference HVP power-iteration baseline, and the U1 rerun on held-out seeds without the perpendicular damping.
 * @cold
 */

export interface Phase2GrowthGridRow {
  readonly etaOverQuadratic: number;
  readonly eta: number;
  readonly steps: number;
  readonly logGrowth: number;
  readonly guardTriggered: boolean;
  readonly ratePerStep: number;
}

export interface Phase2NetworkCell {
  readonly cellId: string;
  readonly numParams: number;
  readonly beta: number;
  readonly batchB: number;
  readonly lambda1: number;
  readonly lambda2: number;
  readonly lambda1AnalyticPowerIter40: number;
  readonly lambdaMinShiftedPowerIter100: number;
  readonly traceH: number;
  readonly traceOverLambda1: number;
  readonly noiseRankRatio: number;
  readonly nuB: number;
  readonly kappaFeatEffective: number;
  readonly alphaDiag: number;
  readonly etaCritQuadratic: number;
  readonly etaCritDestructiveRay: number;
  readonly quadraticVsDestructiveGapPct: number;
  readonly etaCritOneDimEoss: number;
  readonly etaSandwichLower: number;
  readonly etaSandwichUpper: number;
  readonly etaTraceBracketLower: number;
  readonly etaTraceBracketUpper: number;
  readonly empiricalCeilingMedian: number;
  readonly empiricalCeilingP05: number;
  readonly empiricalCeilingP95: number;
  readonly bisectionHalfWidthPct: number;
  readonly upperWasStableCount: number;
  readonly oneDimOverEmpiricalRatio: number;
  readonly quadraticOverEmpiricalRatio: number;
  readonly empiricalInsideTraceBracket: boolean;
  readonly growthGrid: readonly Phase2GrowthGridRow[];
  readonly fdHvpIterations: number;
  readonly fdHvpSampleGradEquivalents: number;
  readonly fdHvpLambda1Hat: number;
  readonly fdHvpLambda1RelErr: number;
  readonly fdHvpCosWithAnalyticU1: number;
  readonly fdHvpEtaCritHat: number;
  readonly fdHvpErrVsQuadraticPct: number;
  readonly fdHvpErrVsDestructivePct: number;
  readonly traceSeconds: number;
  readonly ceilingSeconds: number;
}

export interface Phase2Report {
  readonly labId: string;
  readonly date: string;
  readonly nodeVersion: string;
  readonly numSeeds: number;
  readonly streams: number;
  readonly steps: number;
  readonly bisectionIterations: number;
  readonly growthRatioThreshold: number;
  readonly noiseScaleTau: number;
  readonly predictionTexts: Readonly<Record<string, string>>;
  readonly predictions: Readonly<Record<string, boolean>>;
  readonly networks: readonly Phase2NetworkCell[];
}
