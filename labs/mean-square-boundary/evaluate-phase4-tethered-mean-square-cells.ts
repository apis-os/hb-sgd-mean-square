/**
 * @file labs/mean-square-boundary/evaluate-phase4-tethered-mean-square-cells.ts
 * @description Phase 4 (`evaluatePhase4TetheredMeanSquareCells`): the exact mean-square threshold of Theorem G.6 on the
 * tethered Hessians of the `P = 33` network (dense per-sample Hessians from central differences; `beta` in `{0, 0.9}`) and the
 * `P = 3 610` network (exact per-sample Pearlmutter products; `beta = 0.9`), `B = 16`, tether grid `g` in
 * `{128, 32, 8, 2, 1/2, 1/8, 1/32, 1/128}` with `mu = g + max(0, -lambda_min)`. Per network `evaluatePhase4NetworkRow`; the predictions
 * `t1`-`t6`, whose texts were committed before the first run, are evaluated from the rows.
 * @cold
 */
import { evaluatePhase4NetworkRow } from './evaluate-phase4-network-row.ts';
import type { Phase4NetworkRow, Phase4Report, Phase4ResumedLanczosCell } from './phase4-types.ts';

const PHASE4_PREDICTION_TEXTS: Readonly<Record<string, string>> = {
  t1: 'Theorem G.6 is exact on network Hessians: in every operator cell (P = 33: beta in {0, 0.9} and every tether; P = 3,610: beta = 0.9, g = 2) the spectral radius of the exact second-moment operator T is below 1 at (1 - 1e-3) eta* and above 1 at (1 + 1e-3) eta*, with eta* = 1 / lambda_max(L_0^{-1/2} B_beta L_0^{-1/2}) by Lanczos.',
  t2: 'The mean Hessian of the P = 3,610 network is indefinite: the smallest eigenvalue of its full eigendecomposition is negative (the shifted power iteration of Phase 2 gave only the upper bound 0.012; the Gauss-Newton part has rank at most 640 and the residual part of the squared loss is indefinite on its null space).',
  t3: 'Mean-square stability implies almost-sure stability: in every cell the median sample-path ceiling of the linearised iteration is at least (1 - 0.02) eta* minus the largest bisection half-width.',
  t4: 'Linearisation: in every cell the median sample-path ceiling of the network iteration (start 1e-6 from the checkpoint) is within 2 % plus the two largest bisection half-widths of the median ceiling of the linearised iteration.',
  t5: 'Flat-mode collapse at P = 3,610: eta* decreases from g = 1/32 to g = 1/128, and the log-log slope of eta* against the smallest tethered eigenvalue between these two grid values lies in [0.4, 1.1] (the flat-mode bounds of Theorem G.5 give O(m) when the T_flat bound applies and O(sqrt(m)) from the pairwise bound otherwise; the interval allows for a grid that is not yet asymptotic).',
  t6: 'The fraction f* = eta* (lambda_1 + mu) / (2 (1 + beta)) of the deterministic threshold is non-decreasing in mu along the grid, on both networks and for every beta (at fixed f every weight of the Rayleigh quotient of Theorem G.6(2) grows with mu and the numerator does not depend on mu). eta* itself need not be monotone; its interior maxima are reported, not predicted.',
};

// Added 2026-10-27 after the predictions were committed and after the first launch had computed the P = 33 operator cells;
// they are checks of the implementation (consequences of the theorem), not predictions.
const IMPLEMENTATION_CHECK_TEXTS: Readonly<Record<string, string>> = {
  c1: 'For consecutive grid values mu1 < mu2 with eta*(mu1) <= eta_det(mu2) / 2, eta*(mu2) >= eta*(mu1) (below half the deterministic threshold every weight of Theorem G.6(2) grows with mu at fixed eta).',
  c3: 'Every operator cell left undecided by its certificates at 1e-3 is decided at the fallback offset 1e-2 (rule fixed before the P = 3,610 run).',
  c2: 'Every operator cell is decided by Collatz-Wielandt certificates at the last iterate: Q - T(Q) > 0 at (1 - 1e-3) eta* and T(Q) - Q > 0 at (1 + 1e-3) eta* (P = 33: from the full bracket [lambda_min, lambda_max] of Q^{-1/2} T(Q) Q^{-1/2}; P = 3,610: by Cholesky).',
};

export async function evaluatePhase4TetheredMeanSquareCells(
  cfg: {
    readonly matrixWorkers: number;
    readonly trajectoryWorkers: number;
    readonly trajectorySeeds: number;
    readonly operatorMaxIterations3610: number;
    /** Lanczos outcomes of `X1_P3610` cells from an interrupted launch (see `evaluatePhase4NetworkRow`). */
    readonly resumeCells3610?: readonly Phase4ResumedLanczosCell[];
  },
  log: (line: string) => void,
): Promise<Phase4Report> {
  const batchB = 16;
  const tetherGrid = [128, 32, 8, 2, 0.5, 0.125, 0.03125, 0.0078125] as const;
  const shared = {
    batchB,
    lanczosRelTol: 1e-10,
    relOffset: 1e-3,
    fallbackOffset: 1e-2,
    trajectorySeeds: cfg.trajectorySeeds,
    trajectoryWorkers: cfg.trajectoryWorkers,
    streams: 8,
    steps: 1000,
    bisectionIterations: 20,
    logGrowthThreshold: 0.9162907318741551, // log(2.5), the Phase 1 threshold
    startNorm: 1e-6,
    divergenceEnergy: 2.5e-5,
  };
  const cells33 = [];
  for (const beta of [0, 0.9]) for (const g of tetherGrid) cells33.push({ beta, tetherGrid: g });
  const net33 = await evaluatePhase4NetworkRow(
    'N33_mlp', { kind: 'mlp33' }, 'dense, central differences of per-sample gradients (eps 1e-5), symmetrised',
    [0, 0.9], tetherGrid, cells33,
    { ...shared, matrixWorkers: 0, classifyCriticalMode: true, certificateBracket: true, horizonMultipliers: [1.0, 1.1, 1.5], lanczosMaxIterations: 400, operatorMaxIterations: 200000, operatorRelTol: 1e-13, seedTag: 33 },
    log,
  );
  const net3610 = await evaluatePhase4NetworkRow(
    'X1_P3610', { kind: 'scale', dIn: 64, seedLo: 0x67400001, seedHi: 0x9e3779b9 }, 'exact per-sample Pearlmutter products',
    [0.9], tetherGrid, [{ beta: 0.9, tetherGrid: 2 }],
    { ...shared, matrixWorkers: cfg.matrixWorkers, certificateBracket: false, horizonMultipliers: [], processingOrder: [2, 0.0078125, 128, 32, 8, 0.5, 0.125, 0.03125], warmStart: true, logLanczosSteps: true, classifyCriticalMode: true, lanczosMaxIterations: 150, lanczosRelTol: 1e-7, operatorMaxIterations: cfg.operatorMaxIterations3610, operatorRelTol: 1e-12, seedTag: 3610, ...(cfg.resumeCells3610 === undefined ? {} : { resumeCells: cfg.resumeCells3610 }) },
    log,
  );
  const networks: Phase4NetworkRow[] = [net33, net3610];

  let t1 = true;
  let t3 = true;
  let t4 = true;
  for (const net of networks) {
    for (const oc of net.operatorChecks) if (!oc.crossesAtThmG6) t1 = false;
    for (const th of net.thresholds) {
      const lin = net.trajectories.find((t) => t.beta === th.beta && t.tetherGrid === th.tetherGrid && t.mode === 'linearised');
      const nw = net.trajectories.find((t) => t.beta === th.beta && t.tetherGrid === th.tetherGrid && t.mode === 'network');
      if (lin === undefined || nw === undefined) throw new Error('missing trajectory cell');
      if (lin.medianCeiling < 0.98 * th.etaThmG6 - lin.maxBracketHalfWidth) t3 = false;
      const tol = 0.02 * lin.medianCeiling + lin.maxBracketHalfWidth + nw.maxBracketHalfWidth;
      if (Math.abs(nw.medianCeiling - lin.medianCeiling) > tol) t4 = false;
    }
  }
  const t2 = net3610.lambdaMin < 0.0;
  const a = net3610.thresholds.find((t) => t.tetherGrid === 0.03125);
  const b = net3610.thresholds.find((t) => t.tetherGrid === 0.0078125);
  const slope = net3610.collapseSlopes.find((s) => s.beta === 0.9)?.slope ?? Number.NaN;
  const t5 = a !== undefined && b !== undefined && b.etaThmG6 < a.etaThmG6 && slope >= 0.4 && slope <= 1.1;
  // c1: for consecutive grid values mu1 < mu2 with eta*(mu1) <= eta_det(mu2) / 2 every weight of Theorem G.6(2) grows on
  // [mu1, mu2] at that eta, so eta*(mu2) >= eta*(mu1). c2: every operator cell is decided by its certificates.
  let c1 = true;
  let c2 = true;
  let c3 = true;
  for (const net of networks) {
    for (const oc of net.operatorChecks) {
      if (!(oc.certificateBelow.certifiedBelowOne && oc.certificateAbove.certifiedAboveOne)) c2 = false;
      if (oc.fallback !== null && !(oc.fallback.certificateBelow.certifiedBelowOne && oc.fallback.certificateAbove.certifiedAboveOne)) c3 = false;
    }
    for (const beta of new Set(net.thresholds.map((t) => t.beta))) {
      const row = net.thresholds.filter((t) => t.beta === beta).sort((u, v) => u.mu - v.mu);
      for (let i = 1; i < row.length; (i = (i + 1) | 0)) {
        if (row[i - 1]!.etaThmG6 <= 0.5 * row[i]!.etaDeterministic && row[i]!.etaThmG6 < row[i - 1]!.etaThmG6) c1 = false;
      }
    }
  }
  let t6 = true;
  for (const net of networks) {
    for (const beta of new Set(net.thresholds.map((t) => t.beta))) {
      const row = net.thresholds.filter((t) => t.beta === beta).sort((u, v) => u.mu - v.mu);
      for (let i = 1; i < row.length; (i = (i + 1) | 0)) {
        if (row[i]!.fractionOfDeterministic < row[i - 1]!.fractionOfDeterministic) t6 = false;
      }
    }
  }

  return {
    labId: 'mean-square-boundary-phase4',
    date: new Date().toISOString(),
    nodeVersion: process.version,
    batchB,
    tetherGrid,
    predictionTexts: PHASE4_PREDICTION_TEXTS,
    predictions: { t1, t2, t3, t4, t5, t6 },
    implementationCheckTexts: IMPLEMENTATION_CHECK_TEXTS,
    implementationChecks: { c1, c2, c3 },
    networks,
  };
}
