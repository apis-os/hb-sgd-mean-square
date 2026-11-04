/**
 * @file labs/mean-square-boundary/phase1-types.ts
 * @description Type definitions for Phase 1 (`Phase1Report`): the empirical mean-square ceiling of heavy-ball SGD on a quadratic with random-feature minibatches (Gaussian and Rademacher, i.i.d. and finite-pool without replacement) against the secular equation of Theorem G, its sandwich bounds and the one-dimensional ceiling of Theorem E(3).
 * @cold
 */

export interface Phase1CellRow {
  readonly cellId: string;
  readonly numParams: number;
  readonly beta: number;
  readonly batchB: number;
  readonly featureFamily: 'gaussian-iid' | 'rademacher-iid' | 'gaussian-pool';
  readonly kappaFeat: number;
  readonly poolSize: number;
  readonly zeta: number;
  readonly lambda1: number;
  readonly traceH: number;
  readonly noiseRankRatio: number;
  readonly etaCritDeterministic: number;
  readonly etaCritOneDim: number;
  readonly etaCritSecular: number;
  readonly etaCritSecularLeeLimit: number;
  readonly etaCritSandwichLower: number;
  readonly etaCritSandwichUpper: number;
  readonly empiricalCeilingMedian: number;
  readonly empiricalCeilingP05: number;
  readonly empiricalCeilingP95: number;
  readonly bisectionHalfWidthPct: number;
  readonly secularMedianErrPct: number;
  readonly secularLeeLimitMedianErrPct: number;
  readonly oneDimMedianErrPct: number;
  readonly oneDimOverEmpiricalRatio: number;
  readonly insideSandwich: boolean;
}

export interface Phase1Report {
  readonly labId: 'mean-square-boundary-phase1';
  readonly date: string;
  readonly nodeVersion: string;
  readonly numSeeds: number;
  readonly streams: number;
  readonly steps: number;
  readonly bisectionIterations: number;
  readonly growthRatioThreshold: number;
  readonly cells: readonly Phase1CellRow[];
  readonly predictionTexts: Readonly<Record<string, string>>;
  readonly predictions: {
    readonly p1IidCeilingsInsideSandwich: boolean;
    readonly p2SecularRootWithin5PctMedianIid: boolean;
    readonly p3RademacherOneDimOverestimatesBy2xWhileSecularWithin5Pct: boolean;
    readonly p4PoolArmLeeRootWithin8PctAtSmallZeta: boolean;
  };
}
