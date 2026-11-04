/**
 * @file labs/mean-square-boundary/evaluate-phase4-network-row.ts
 * @description One network of Phase 4 (`evaluatePhase4NetworkRow`). (1) The dense mean Hessian from `P` products, its full
 * eigendecomposition (`lambda_min`, the number of negative eigenvalues, `u_1`) and `nu_{11,B}^2 = (1/B) Var_i(u_1^T H_i u_1)`.
 * (2) For each `beta` and tether grid value `g` (tether `mu = g + max(0, -lambda_min)`): `eta*` of Theorem G.6(3) by Lanczos on
 * `applyThmG6RayleighOperator`, the deterministic threshold and the one-dimensional ceiling of Theorem E(3) at that tether.
 * (3) For the requested `(beta, g)` cells: the spectral radius of the exact second-moment operator at `(1 -+ relOffset) eta*`
 * by `measureNetworkSecondMomentSpectralRadius` from the identity, the second probe warm-started from the first. (4) The
 * sample-path ceilings of the linearised and of the network iteration (`measureTetheredHeavyBallCeiling`, one job per seed
 * and mode, on a worker pool after the matrix workers have exited), bisected from `1.1 x` the deterministic threshold.
 * `log` receives progress lines, and one `CELL` / `OPCHECK` line with the full JSON record of each cell and operator check as
 * it finishes; a cell listed in `resumeCells` takes its Lanczos outcome from an interrupted launch instead of recomputing it.
 * `eigendecompositionSeconds` is wall-clock and is the only field that is not reproducible.
 * @cold
 */
import { fdlibmLog } from '../../src/math/index.ts';
import { applyNetworkHeavyBallSecondMomentOperator } from './apply-network-heavy-ball-second-moment-operator.ts';
import { applyThmG6RayleighOperator } from './apply-thm-g6-rayleigh-operator.ts';
import { certifySecondMomentRadius } from './certify-second-moment-radius.ts';
import { computeFloat64Quantile } from './compute-float64-quantile.ts';
import { computeLanczosTopEigenvalue } from './compute-lanczos-top-eigenvalue.ts';
import { computeSymmetricEigendecompositionHouseholderQl } from './compute-symmetric-eigendecomposition-householder-ql.ts';
import { createPhase4RowExecutor } from './create-phase4-row-executor.ts';
import { measureNetworkSecondMomentSpectralRadius } from './measure-network-second-moment-spectral-radius.ts';
import type {
  Phase4NetworkRow,
  Phase4NetworkSpec,
  Phase4OperatorCheck,
  Phase4ResumedLanczosCell,
  Phase4ThresholdCell,
  Phase4TrajectoryCell,
  Phase4TrajectoryJob,
  Phase4TrajectoryResult,
} from './phase4-types.ts';
import { runSeedBlocksParallel } from './run-seed-blocks-parallel.ts';

export async function evaluatePhase4NetworkRow(
  networkId: string,
  spec: Phase4NetworkSpec,
  hessianSource: string,
  betas: readonly number[],
  tetherGrid: readonly number[],
  operatorCells: ReadonlyArray<{ readonly beta: number; readonly tetherGrid: number }>,
  cfg: {
    readonly batchB: number;
    readonly matrixWorkers: number;
    readonly lanczosMaxIterations: number;
    readonly lanczosRelTol: number;
    readonly operatorMaxIterations: number;
    readonly operatorRelTol: number;
    readonly relOffset: number;
    readonly fallbackOffset: number;
    /** Order in which the grid values are computed (results are reported in `tetherGrid` order). */
    readonly processingOrder?: readonly number[];
    /** Start each Lanczos run from the top Ritz vector of the previous cell of the same beta (identity for the first). */
    readonly warmStart?: boolean;
    /** Log the step count, residual and elapsed time of every 5th Lanczos step (never the Ritz value). */
    readonly logLanczosSteps?: boolean;
    /** Classify the critical second moment of each eta* from the top Ritz vector (sector shares, lag-1 ratio). */
    readonly classifyCriticalMode?: boolean;
    /** Lanczos outcomes of cells already computed by an interrupted launch; those cells skip the Lanczos run and the operator
     * check is still run. The recomputed one-dimensional ceiling must equal the recorded one bit for bit. */
    readonly resumeCells?: readonly Phase4ResumedLanczosCell[];
    readonly certificateBracket: boolean;
    readonly horizonMultipliers: readonly number[];
    readonly trajectorySeeds: number;
    readonly trajectoryWorkers: number;
    readonly streams: number;
    readonly steps: number;
    readonly bisectionIterations: number;
    readonly logGrowthThreshold: number;
    readonly startNorm: number;
    readonly divergenceEnergy: number;
    readonly seedTag: number;
  },
  log: (line: string) => void,
): Promise<Phase4NetworkRow> {
  const batchB = cfg.batchB;
  const exec = await createPhase4RowExecutor(spec, cfg.matrixWorkers);
  const n = exec.n;
  const provider = exec.provider;
  const total = Math.imul(n, n);

  // (1) Dense mean Hessian and its eigendecomposition.
  const eye = exec.alloc();
  for (let j = 0; j < n; (j = (j + 1) | 0)) eye[(Math.imul(j, n) + j) | 0] = 1.0;
  const hm = exec.alloc();
  await exec.hessianRows(-1, eye, hm, 1.0, false);
  const hSym = new Float64Array(total);
  for (let j = 0; j < n; (j = (j + 1) | 0)) {
    for (let k = 0; k < n; (k = (k + 1) | 0)) hSym[(Math.imul(j, n) + k) | 0] = 0.5 * (hm[(Math.imul(j, n) + k) | 0]! + hm[(Math.imul(k, n) + j) | 0]!);
  }
  const t0 = Date.now();
  const eig = computeSymmetricEigendecompositionHouseholderQl(hSym, n);
  const eigendecompositionSeconds = (Date.now() - t0) / 1000;
  const values = eig.values;
  const lambdaMin = values[0]!;
  const lambda1 = values[(n - 1) | 0]!;
  const lambda2 = values[(n - 2) | 0]!;
  let negativeEigenvalues = 0;
  let eigenvaluesBelowTwiceAbsMin = 0;
  let traceH = 0.0;
  for (let k = 0; k < n; (k = (k + 1) | 0)) {
    if (values[k]! < 0.0) negativeEigenvalues = (negativeEigenvalues + 1) | 0;
    if (values[k]! < 2.0 * Math.abs(lambdaMin)) eigenvaluesBelowTwiceAbsMin = (eigenvaluesBelowTwiceAbsMin + 1) | 0;
    traceH += hSym[(Math.imul(k, n) + k) | 0]!;
  }
  log(`${networkId}: eigendecomposition ${eigendecompositionSeconds.toFixed(1)} s, lambda_1 ${lambda1}, lambda_min ${lambdaMin}, ${negativeEigenvalues} negative`);
  const qt = exec.alloc();
  qt.set(eig.qt);
  const q = exec.alloc();
  await exec.transpose(qt, q);
  const u1 = qt.subarray(Math.imul((n - 1) | 0, n), total);
  const hu = new Float64Array(n);
  let sumSq = 0.0;
  for (let i = 0; i < provider.numSamples; (i = (i + 1) | 0)) {
    provider.applySampleHessian(i, u1, hu);
    let r = 0.0;
    for (let a = 0; a < n; (a = (a + 1) | 0)) r += u1[a]! * hu[a]!;
    sumSq += r * r;
  }
  const nuB11Sq = (sumSq / provider.numSamples - lambda1 * lambda1) / batchB;

  // (2) eta* of Theorem G.6 on each tether.
  const sA: [Float64Array, Float64Array, Float64Array, Float64Array, Float64Array, Float64Array] = [
    exec.alloc(), exec.alloc(), exec.alloc(), exec.alloc(), exec.alloc(), exec.alloc(),
  ];
  const x0 = exec.alloc();
  const vPrev = exec.alloc();
  const vCur = exec.alloc();
  const w = exec.alloc();
  const lambdaT = new Float64Array(n);
  const shift = lambdaMin < 0.0 ? -lambdaMin : 0.0;
  const thresholds: Phase4ThresholdCell[] = [];
  // (3) The exact second-moment operator at (1 -+ relOffset) eta*, run right after the Lanczos value of its cell.
  const operatorChecks: Phase4OperatorCheck[] = [];
  const oBlocks: [Float64Array, Float64Array, Float64Array] = [exec.alloc(), exec.alloc(), exec.alloc()];
  const q11 = x0;
  const q12 = vPrev;
  const q22 = vCur;
  const runOperatorCheck = async (cell: { readonly beta: number; readonly tetherGrid: number }, th: Phase4ThresholdCell): Promise<void> => {
    const inv = 1.0 / Math.sqrt(2.0 * n);
    q11.fill(0.0);
    q12.fill(0.0);
    q22.fill(0.0);
    for (let j = 0; j < n; (j = (j + 1) | 0)) {
      q11[(Math.imul(j, n) + j) | 0] = inv;
      q22[(Math.imul(j, n) + j) | 0] = inv;
    }
    const certify = async (eta: number): Promise<ReturnType<typeof certifySecondMomentRadius>> => {
      await applyNetworkHeavyBallSecondMomentOperator(exec, cell.beta, eta, th.mu, batchB, q11, q12, q22, oBlocks[0], oBlocks[1], oBlocks[2], sA);
      return certifySecondMomentRadius([q11, q12, q22], oBlocks, n, cfg.certificateBracket);
    };
    const etaBelow = (1.0 - cfg.relOffset) * th.etaThmG6;
    const etaAbove = (1.0 + cfg.relOffset) * th.etaThmG6;
    const below = await measureNetworkSecondMomentSpectralRadius(
      exec, cell.beta, etaBelow, th.mu, batchB, q11, q12, q22, oBlocks, sA, cfg.operatorMaxIterations, cfg.operatorRelTol,
    );
    const certificateBelow = await certify(etaBelow);
    const above = await measureNetworkSecondMomentSpectralRadius(
      exec, cell.beta, etaAbove, th.mu, batchB, q11, q12, q22, oBlocks, sA, cfg.operatorMaxIterations, cfg.operatorRelTol,
    );
    const certificateAbove = await certify(etaAbove);
    let fallback: Phase4OperatorCheck['fallback'] = null;
    if (!(certificateBelow.certifiedBelowOne && certificateAbove.certifiedAboveOne)) {
      const fb = (1.0 - cfg.fallbackOffset) * th.etaThmG6;
      const fa = (1.0 + cfg.fallbackOffset) * th.etaThmG6;
      const rb = await measureNetworkSecondMomentSpectralRadius(exec, cell.beta, fb, th.mu, batchB, q11, q12, q22, oBlocks, sA, cfg.operatorMaxIterations, cfg.operatorRelTol);
      const cb = await certify(fb);
      const ra = await measureNetworkSecondMomentSpectralRadius(exec, cell.beta, fa, th.mu, batchB, q11, q12, q22, oBlocks, sA, cfg.operatorMaxIterations, cfg.operatorRelTol);
      const ca = await certify(fa);
      fallback = { relOffset: cfg.fallbackOffset, radiusBelow: rb.radius, radiusAbove: ra.radius, certificateBelow: cb, certificateAbove: ca };
    }
    const horizon: Array<{ readonly multiplier: number; readonly radius: number; readonly iterations: number; readonly converged: boolean; readonly stepsToGrow2p5: number }> = [];
    for (const mult of cfg.horizonMultipliers) {
      const hz = await measureNetworkSecondMomentSpectralRadius(
        exec, cell.beta, mult * th.etaThmG6, th.mu, batchB, q11, q12, q22, oBlocks, sA, cfg.operatorMaxIterations, cfg.operatorRelTol,
      );
      horizon.push({ multiplier: mult, radius: hz.radius, iterations: hz.iterations, converged: hz.converged, stepsToGrow2p5: hz.radius > 1.0 ? fdlibmLog(2.5) / fdlibmLog(hz.radius) : Number.POSITIVE_INFINITY });
    }
    operatorChecks.push({
      beta: cell.beta,
      tetherGrid: cell.tetherGrid,
      etaThmG6: th.etaThmG6,
      relOffset: cfg.relOffset,
      radiusBelow: below.radius,
      radiusAbove: above.radius,
      iterationsBelow: below.iterations,
      iterationsAbove: above.iterations,
      convergedBelow: below.converged,
      convergedAbove: above.converged,
      crossesAtThmG6: below.radius < 1.0 && above.radius > 1.0,
      certificateBelow,
      certificateAbove,
      fallback,
      horizon,
    });
    log(`OPCHECK ${JSON.stringify({ networkId, ...operatorChecks[operatorChecks.length - 1]! })}`);
    log(`${networkId}: operator beta ${cell.beta} g ${cell.tetherGrid}: rho ${below.radius} / ${above.radius} (${below.iterations}, ${above.iterations} steps), certified below ${certificateBelow.certifiedBelowOne} [${certificateBelow.lowerBound}, ${certificateBelow.upperBound}], above ${certificateAbove.certifiedAboveOne} [${certificateAbove.lowerBound}, ${certificateAbove.upperBound}], horizon ${horizon.map((h) => `${h.multiplier}: ${h.radius}`).join(' ')}`);
  };

  const ritz = cfg.warmStart === true || cfg.classifyCriticalMode === true ? new Float64Array(total) : undefined;
  // Sigma_0*_kk = X_kk / sqrt(2 lambdaT_k) (L_0^{-1/2} on the diagonal of the eigenbasis); sign fixed by a positive trace.
  const classify = (x: Float64Array, eta: number, beta: number, relResidual: number): NonNullable<Phase4ThresholdCell['criticalMode']> => {
    let tot = 0.0;
    let top = 0.0;
    let rigid = 0.0;
    let posFlat = 0.0;
    let neg = 0.0;
    let lag = 0.0;
    const cut = 2.0 * Math.abs(lambdaMin);
    for (let k = 0; k < n; (k = (k + 1) | 0)) {
      const d = x[(Math.imul(k, n) + k) | 0]! / Math.sqrt(2.0 * lambdaT[k]!);
      tot += d;
      lag += (1.0 - (eta * lambdaT[k]!) / (1.0 + beta)) * d;
      if (k === n - 1) top += d;
      else if (values[k]! >= cut) rigid += d;
      else if (values[k]! >= 0.0) posFlat += d;
      else neg += d;
    }
    return { shareTop: top / tot, shareRigid: rigid / tot, sharePositiveFlat: posFlat / tot, shareNegative: neg / tot, lagOneRatio: lag / tot, lanczosRelResidual: relResidual };
  };
  for (const beta of betas) {
    const kappaBeta = (1.0 + beta) / (1.0 - beta);
    let haveRitz = false;
    for (const g of cfg.processingOrder ?? tetherGrid) {
      const mu = g + shift;
      for (let k = 0; k < n; (k = (k + 1) | 0)) lambdaT[k] = values[k]! + mu;
      const l1 = lambda1 + mu;
      const etaOneDim = (2.0 * (1.0 + beta) * l1) / (l1 * l1 + kappaBeta * nuB11Sq);
      const resumed = cfg.resumeCells?.find((c) => c.beta === beta && c.tetherGrid === g);
      if (resumed !== undefined && resumed.etaOneDim !== etaOneDim) throw new Error(`resumed cell beta ${beta} g ${g}: 1D ceiling ${etaOneDim} differs from the recorded ${resumed.etaOneDim}`);
      if (ritz !== undefined && haveRitz && cfg.warmStart === true) x0.set(ritz);
      else {
        x0.fill(0.0);
        for (let j = 0; j < n; (j = (j + 1) | 0)) x0[(Math.imul(j, n) + j) | 0] = 1.0;
      }
      const tStart = Date.now();
      const lz = resumed !== undefined ? null : await computeLanczosTopEigenvalue(
        (x, out) => applyThmG6RayleighOperator(exec, lambdaT, q, qt, beta, batchB, x, out, sA),
        x0, vPrev, vCur, w, cfg.lanczosMaxIterations, cfg.lanczosRelTol,
        {
          ...(ritz !== undefined ? { ritzOut: ritz } : {}),
          ...(cfg.logLanczosSteps === true
            ? { onStep: (step: number, r: number) => { if (step % 5 === 0) log(`${networkId}: beta ${beta} g ${g}: Lanczos step ${step}, rel residual ${r.toExponential(2)}, ${((Date.now() - tStart) / 1000).toFixed(0)} s`); } }
            : {}),
        },
      );
      haveRitz = ritz !== undefined && lz !== null;
      const etaThmG6 = lz !== null ? 1.0 / lz.value : resumed!.etaThmG6;
      const lanczosRelResidual = lz !== null ? lz.relResidual : resumed!.lanczosRelResidual;
      thresholds.push({
        beta,
        batchB,
        tetherGrid: g,
        mu,
        lambdaMinTethered: lambdaMin + mu,
        lambda1Tethered: l1,
        etaDeterministic: (2.0 * (1.0 + beta)) / l1,
        etaOneDim,
        etaThmG6,
        lanczosIterations: lz !== null ? lz.iterations : resumed!.lanczosIterations,
        lanczosConverged: lz !== null ? lz.converged : resumed!.lanczosConverged,
        lanczosRelResidual,
        oneDimOverThmG6: etaOneDim / etaThmG6,
        fractionOfDeterministic: (etaThmG6 * l1) / (2.0 * (1.0 + beta)),
        lambdaSecondTethered: values[1]! + mu,
        ...(lz === null
          ? (resumed!.criticalMode === undefined ? {} : { criticalMode: resumed!.criticalMode })
          : cfg.classifyCriticalMode === true && ritz !== undefined ? { criticalMode: classify(ritz, etaThmG6, beta, lz.relResidual) } : {}),
        ...(lz === null ? { resumedFromLog: true } : {}),
      });
      log(`CELL ${JSON.stringify({ networkId, ...thresholds[thresholds.length - 1]! })}`);
      log(`${networkId}: beta ${beta} g ${g}: eta* ${etaThmG6} (${thresholds[thresholds.length - 1]!.lanczosIterations} Lanczos steps, converged ${thresholds[thresholds.length - 1]!.lanczosConverged}${lz === null ? ', resumed' : ''}), 1D ${etaOneDim}${thresholds[thresholds.length - 1]!.criticalMode === undefined ? '' : `, critical mode ${JSON.stringify(thresholds[thresholds.length - 1]!.criticalMode)}`}`);
      const cell = operatorCells.find((c) => c.beta === beta && c.tetherGrid === g);
      if (cell !== undefined) await runOperatorCheck(cell, thresholds[thresholds.length - 1]!);
    }
  }
  // Report in the registered grid order whatever the processing order.
  const rank = (t: { readonly beta: number; readonly tetherGrid: number }): number => betas.indexOf(t.beta) * 1000 + tetherGrid.indexOf(t.tetherGrid);
  thresholds.sort((a, b) => rank(a) - rank(b));
  operatorChecks.sort((a, b) => rank(a) - rank(b));

  await exec.close();

  // (4) Sample-path ceilings, linearised and network iteration.
  const jobs: Phase4TrajectoryJob[] = [];
  for (const th of thresholds) {
    for (const mode of [0, 1] as const) {
      for (let s = 0; s < cfg.trajectorySeeds; (s = (s + 1) | 0)) {
        jobs.push({
          network: spec,
          mode,
          mu: th.mu,
          beta: th.beta,
          batchB,
          etaUpper: 1.1 * th.etaDeterministic,
          streams: cfg.streams,
          steps: cfg.steps,
          iterations: cfg.bisectionIterations,
          logGrowthThreshold: cfg.logGrowthThreshold,
          startNorm: cfg.startNorm,
          divergenceEnergy: cfg.divergenceEnergy,
          seedLo: (0x13a00000 + Math.imul(s, 7919) + Math.imul(cfg.seedTag, 0x101)) | 0,
          seedHi: (0x3c6ef372 ^ Math.imul(s + 1, 0x9e3779b9) ^ Math.imul(cfg.seedTag + 1, 0x85ebca6b)) | 0,
        });
      }
    }
  }
  const results = await runSeedBlocksParallel<Phase4TrajectoryJob, Phase4TrajectoryResult>('./phase4-trajectory-worker.ts', jobs, cfg.trajectoryWorkers);
  const trajectories: Phase4TrajectoryCell[] = [];
  let jobIdx = 0;
  for (const th of thresholds) {
    for (const mode of [0, 1] as const) {
      const ceilings: number[] = [];
      let maxHalf = 0.0;
      for (let s = 0; s < cfg.trajectorySeeds; (s = (s + 1) | 0)) {
        const r = results[jobIdx]!;
        jobIdx = (jobIdx + 1) | 0;
        ceilings.push(r.etaCeiling);
        if (r.bracketHalfWidth > maxHalf) maxHalf = r.bracketHalfWidth;
      }
      const median = computeFloat64Quantile(Float64Array.from(ceilings), 0.5);
      trajectories.push({
        beta: th.beta,
        tetherGrid: th.tetherGrid,
        mode: mode === 0 ? 'linearised' : 'network',
        ceilings,
        medianCeiling: median,
        maxBracketHalfWidth: maxHalf,
        medianOverThmG6: median / th.etaThmG6,
        horizonInformative: (500 * 2.0 * th.etaThmG6 * th.lambdaMinTethered) / (1.0 - th.beta) >= fdlibmLog(2.5),
      });
    }
  }
  log(`${networkId}: ${jobs.length} trajectory jobs done`);

  const collapseSlopes: Array<{ readonly beta: number; readonly slope: number }> = [];
  const sortedGrid = [...tetherGrid].sort((a, b) => a - b);
  for (const beta of betas) {
    const a = thresholds.find((t) => t.beta === beta && t.tetherGrid === sortedGrid[0]);
    const b = thresholds.find((t) => t.beta === beta && t.tetherGrid === sortedGrid[1]);
    if (a === undefined || b === undefined) continue;
    const slope = fdlibmLog(b.etaThmG6 / a.etaThmG6) / fdlibmLog(b.lambdaMinTethered / a.lambdaMinTethered);
    collapseSlopes.push({ beta, slope });
  }

  const interiorMaxima: Array<{ readonly beta: number; readonly tetherGrid: readonly number[] }> = [];
  for (const beta of betas) {
    const row = thresholds.filter((t) => t.beta === beta).sort((u, v) => u.mu - v.mu);
    const maxima: number[] = [];
    for (let i = 1; i + 1 < row.length; (i = (i + 1) | 0)) {
      if (row[i]!.etaThmG6 > row[i - 1]!.etaThmG6 && row[i]!.etaThmG6 > row[i + 1]!.etaThmG6) maxima.push(row[i]!.tetherGrid);
    }
    interiorMaxima.push({ beta, tetherGrid: maxima });
  }

  return {
    networkId,
    numParams: n,
    numSamples: provider.numSamples,
    hessianSource,
    lambda1,
    lambda2,
    lambdaMin,
    negativeEigenvalues,
    lambdaSecond: values[1]!,
    eigenvaluesBelowTwiceAbsMin,
    traceH,
    nuB11Sq,
    eigendecompositionSeconds,
    thresholds,
    operatorChecks,
    trajectories,
    collapseSlopes,
    interiorMaxima,
  };
}
