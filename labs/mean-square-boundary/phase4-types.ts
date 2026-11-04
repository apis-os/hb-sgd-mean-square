/**
 * @file labs/mean-square-boundary/phase4-types.ts
 * @description Types of Phase 4 (`Phase4NetworkSpec`, `Phase4SampleHessianProvider`, `Phase4ThresholdCell`,
 * `Phase4ResumedLanczosCell`, `Phase4OperatorCheck`, `Phase4TrajectoryJob`, `Phase4TrajectoryResult`, `Phase4NetworkRow`, `Phase4Report`): the
 * exact mean-square threshold of Theorem G.6 on tethered network Hessians, the second-moment operator at that threshold,
 * and sample-path ceilings of the linearised and of the network iteration.
 * @cold
 */

/** A network checkpoint (the `P = 33` network of Phases 2-8, or a scale network of lab Phase 4 by input width and seeds), or a quadratic with given per-sample Hessians (tests); `theta` replaces the checkpoint (Phase 5 evaluates along a training run); `gaussNewton` keeps only the Gauss-Newton part J_i^T J_i of each per-sample Hessian. */
export type Phase4NetworkSpec =
  | { readonly kind: 'mlp33'; readonly theta?: Float64Array; readonly gaussNewton?: boolean }
  | { readonly kind: 'scale'; readonly dIn: number; readonly seedLo: number; readonly seedHi: number; readonly theta?: Float64Array; readonly gaussNewton?: boolean }
  | { readonly kind: 'quadratic'; readonly numParams: number; readonly numSamples: number; readonly hessians: Float64Array };

/** Per-sample Hessians `H_i` of the loss of sample `i` at the checkpoint, their mean `H`, and minibatch gradients. */
export interface Phase4SampleHessianProvider {
  readonly numParams: number;
  readonly numSamples: number;
  readonly checkpointTheta0: Float64Array;
  applySampleHessian(sampleIdx: number, vec: Float64Array, out: Float64Array): void;
  applyMeanHessian(vec: Float64Array, out: Float64Array): void;
  computeMinibatchGradient(theta: Float64Array, batchIndices: Int32Array, batchSize: number, outGrad: Float64Array): number;
}

export interface Phase4ThresholdCell {
  readonly beta: number;
  readonly batchB: number;
  /** Grid value g; the tether is mu = g + max(0, -lambda_min(H)), so the smallest tethered eigenvalue is g + max(0, lambda_min). */
  readonly tetherGrid: number;
  readonly mu: number;
  readonly lambdaMinTethered: number;
  readonly lambda1Tethered: number;
  readonly etaDeterministic: number;
  /** One-dimensional mean-square ceiling along u_1 (Theorem E(3)) at this tether. */
  readonly etaOneDim: number;
  /** 1 / lambda_max(L_0^{-1/2} B_beta L_0^{-1/2}) of Theorem G.6(3), by Lanczos. */
  readonly etaThmG6: number;
  readonly lanczosIterations: number;
  readonly lanczosConverged: boolean;
  readonly lanczosRelResidual: number;
  readonly oneDimOverThmG6: number;
  /** f* = eta* / eta_det(mu) = eta* (lambda_1 + mu) / (2 (1 + beta)); non-decreasing in mu at fixed beta (weights of Theorem G.6(2) grow in mu at fixed f). */
  readonly fractionOfDeterministic: number;
  /** Second smallest eigenvalue of H + mu I: only the most negative direction of H reaches lambda_min + mu. */
  readonly lambdaSecondTethered: number;
  /**
   * The Krein-Rutman eigenmatrix Sigma_0* at eta* (from the top Ritz vector X of the Lanczos run, Sigma_0* = L_0^{-1/2} X in the
   * eigenbasis): the share of its trace on u_1, on the other rigid modes (lambda >= 2 |lambda_min|), on the positive near-flat block
   * and on the negative sector, and the lag-1 ratio tr(Sigma_1) / tr(Sigma_0) with (Sigma_1)_kk = (1 - eta* lambdaT_k / (1 + beta))
   * (Sigma_0)_kk: near -1 for a flip (period-2) mode, near +1 for a slow (flat or saddle) mode. Added after the registration.
   */
  readonly criticalMode?: {
    readonly shareTop: number;
    readonly shareRigid: number;
    readonly sharePositiveFlat: number;
    readonly shareNegative: number;
    readonly lagOneRatio: number;
    readonly lanczosRelResidual: number;
  };
  /** Set when the Lanczos outcome was read from the progress log of an interrupted launch (`resumeCells`). */
  readonly resumedFromLog?: true;
}

/** The Lanczos outcome of one cell as printed by an interrupted launch (full-precision `Number` strings round-trip exactly). */
export interface Phase4ResumedLanczosCell {
  readonly beta: number;
  readonly tetherGrid: number;
  readonly etaThmG6: number;
  readonly etaOneDim: number;
  readonly lanczosIterations: number;
  readonly lanczosConverged: boolean;
  readonly lanczosRelResidual: number;
  readonly criticalMode?: Phase4ThresholdCell['criticalMode'];
}

export interface Phase4OperatorCheck {
  readonly beta: number;
  readonly tetherGrid: number;
  readonly etaThmG6: number;
  readonly relOffset: number;
  readonly radiusBelow: number;
  readonly radiusAbove: number;
  readonly iterationsBelow: number;
  readonly iterationsAbove: number;
  readonly convergedBelow: boolean;
  readonly convergedAbove: boolean;
  readonly crossesAtThmG6: boolean;
  /** Collatz–Wielandt certificates at the last iterate of each probe (added 2026-10-27 after the registration, see README). */
  readonly certificateBelow: Phase4Certificate;
  readonly certificateAbove: Phase4Certificate;
  /** Radius of T at multiples of eta* above 1 + relOffset, and the steps a second moment needs to grow by 2.5 (post hoc). */
  /** When the certificates at relOffset do not decide both sides: the same probes at fallbackOffset (rule fixed before the P = 3,610 run). */
  readonly fallback: null | {
    readonly relOffset: number;
    readonly radiusBelow: number;
    readonly radiusAbove: number;
    readonly certificateBelow: Phase4Certificate;
    readonly certificateAbove: Phase4Certificate;
  };
  readonly horizon: ReadonlyArray<{ readonly multiplier: number; readonly radius: number; readonly iterations: number; readonly converged: boolean; readonly stepsToGrow2p5: number }>;
}

export interface Phase4Certificate {
  readonly iteratePositiveDefinite: boolean;
  readonly certifiedBelowOne: boolean;
  readonly certifiedAboveOne: boolean;
  readonly lowerBound: number | null;
  readonly upperBound: number | null;
}

export interface Phase4TrajectoryJob {
  readonly network: Phase4NetworkSpec;
  /** 0: linearised iteration with the per-sample Hessians at the checkpoint; 1: the network iteration. */
  readonly mode: 0 | 1;
  readonly mu: number;
  readonly beta: number;
  readonly batchB: number;
  readonly etaUpper: number;
  readonly streams: number;
  readonly steps: number;
  readonly iterations: number;
  readonly logGrowthThreshold: number;
  readonly startNorm: number;
  readonly divergenceEnergy: number;
  readonly seedLo: number;
  readonly seedHi: number;
}

export interface Phase4TrajectoryResult {
  readonly etaCeiling: number;
  readonly bracketHalfWidth: number;
  readonly upperWasStable: boolean;
}

export interface Phase4TrajectoryCell {
  readonly beta: number;
  readonly tetherGrid: number;
  readonly mode: 'linearised' | 'network';
  readonly ceilings: readonly number[];
  readonly medianCeiling: number;
  readonly maxBracketHalfWidth: number;
  readonly medianOverThmG6: number;
  /** 500 * 2 eta* m / (1 - beta) >= log 2.5 with m = lambda_min(H + mu I): the slowest noiseless mode can move the 1 000-step window statistic past the threshold near eta* (rule fixed before the P = 3,610 run). */
  readonly horizonInformative: boolean;
}

export interface Phase4NetworkRow {
  readonly networkId: string;
  readonly numParams: number;
  readonly numSamples: number;
  readonly hessianSource: string;
  readonly lambda1: number;
  readonly lambda2: number;
  readonly lambdaMin: number;
  readonly negativeEigenvalues: number;
  readonly lambdaSecond: number;
  /** Eigenvalues of H below 2 |lambda_min|: the near-flat block that the tether lifts only to about |lambda_min| + g. */
  readonly eigenvaluesBelowTwiceAbsMin: number;
  readonly traceH: number;
  readonly nuB11Sq: number;
  readonly eigendecompositionSeconds: number;
  readonly thresholds: readonly Phase4ThresholdCell[];
  readonly operatorChecks: readonly Phase4OperatorCheck[];
  readonly trajectories: readonly Phase4TrajectoryCell[];
  /** Log-log slope of etaThmG6 against lambdaMinTethered between the two smallest grid values, per beta. */
  readonly collapseSlopes: ReadonlyArray<{ readonly beta: number; readonly slope: number }>;
  /** Grid values of mu at which eta* is a strict local maximum along the grid (interior points only), per beta. */
  readonly interiorMaxima: ReadonlyArray<{ readonly beta: number; readonly tetherGrid: readonly number[] }>;
}

export interface Phase4Report {
  readonly labId: string;
  readonly date: string;
  readonly nodeVersion: string;
  readonly batchB: number;
  readonly tetherGrid: readonly number[];
  readonly predictionTexts: Readonly<Record<string, string>>;
  readonly predictions: Readonly<Record<string, boolean>>;
  readonly implementationCheckTexts: Readonly<Record<string, string>>;
  readonly implementationChecks: Readonly<Record<string, boolean>>;
  readonly networks: readonly Phase4NetworkRow[];
}
