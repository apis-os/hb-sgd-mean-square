/**
 * @file labs/mean-square-boundary/analyze-phase5-final-point.ts
 * @description The final-point analysis of a Phase 5 run (`analyzePhase5FinalPoint`; registration items (g)-(i), amendments
 * 1-2). (g) The exact second-moment operator of the `|H|` iteration (`measureEigenbasisSecondMomentRadius` on the kept modes) at
 * `(1 -+ 1e-3) eta*_|H|` and at `eta`, each with the Collatz–Wielandt bracket at the last iterate and the witness certificate of
 * Theorem splitting; at `eta >= eta_det,|H|` the radius exceeds 1 without a certificate (`rho(T) >= rho(F)^2 > 1`). (h) Top
 * Lyapunov exponents (4 streams, `2e5` steps, burn-in 1 000, common random numbers across `eta`) of the real iteration (curvature
 * `lambda + mu_train`, all modes) and of the `|H|` iteration, at `eta`, and their almost-sure thresholds by
 * `bisectLyapunovThreshold` (12 steps; real on `[1e-3, 4 eta]`, `|H|` on `[1e-3, max(4 eta, 1.5 eta_det,|H|)]`; sign profiles on
 * `{1/4, 1/2, 1, 2, 4} eta`, plus `1.5 eta_det,|H|` for `|H|`); the real sign profile on that grid always (`p5`), and `eta*` of
 * the real iteration where `H + mu_train I > 0` (`c4`). (i) `eta*_g(theta_T)` for `g` in `{1/8, 1/32, 1/128, 1/512}`.
 * `almostSureOnly` restricts the call to the `|H|` almost-sure threshold (the window checkpoints of amendment 1).
 * The exponents come from `createMemoizedLyapunovGamma` (lockstep over step sizes, each step size simulated once): the first
 * `|H|` call holds both bracket ends and `eta`, the first real call the bracket ends and the whole sign grid, and the bisections
 * run two steps per simulation (`bisectLyapunovThresholdBatched`). The returned object is bit-identical to
 * `analyzePhase5FinalPointReference` (the test oracle), which simulates every evaluation separately.
 * @cold
 */
import { assembleCurvatureNoiseOperatorMatrix } from './assemble-curvature-noise-operator-matrix.ts';
import { assembleThmG6OperatorMatrix } from './assemble-thm-g6-operator-matrix.ts';
import { bisectLyapunovThreshold } from './bisect-lyapunov-threshold.ts';
import type { buildEigenbasisSampleHessianDeviations } from './build-eigenbasis-sample-hessian-deviations.ts';
import { applyEigenbasisSecondMomentOperator } from './apply-eigenbasis-second-moment-operator.ts';
import { certifySecondMomentRadius } from './certify-second-moment-radius.ts';
import { computeDenseSymmetricTopEigenpair } from './compute-dense-symmetric-top-eigenpair.ts';
import { computeSplittingWitnessCertificate } from './compute-splitting-witness-certificate.ts';
import { createMemoizedLyapunovGamma } from './create-memoized-lyapunov-gamma.ts';
import { measureEigenbasisSecondMomentRadius } from './measure-eigenbasis-second-moment-radius.ts';

type Dev = ReturnType<typeof buildEigenbasisSampleHessianDeviations>;

export async function analyzePhase5FinalPoint(
  full: { readonly dev: Dev; readonly gram: Float64Array; readonly values: Float64Array },
  abs: { readonly dev: Dev; readonly gram: Float64Array; readonly curvature: Float64Array },
  etaStarAbs: number,
  eta: number,
  beta: number,
  seed: number,
  almostSureOnly: boolean,
  lyapunovSteps: number,
): Promise<Record<string, unknown>> {
  const batchB = 16;
  const streams = 4;
  const burnIn = 1000;
  let maxCurv = 0.0;
  for (let k = 0; k < abs.dev.n; (k = (k + 1) | 0)) if (abs.curvature[k]! > maxCurv) maxCurv = abs.curvature[k]!;
  const etaDetAbs = (2.0 * (1.0 + beta)) / maxCurv;
  const seedAbs = (0x14c00001 + Math.imul(seed, 7919)) | 0;
  const seedReal = (0x14d00001 + Math.imul(seed, 7919)) | 0;
  const gammaAbs = createMemoizedLyapunovGamma(abs.dev, abs.curvature, beta, batchB, streams, lyapunovSteps, burnIn, seedAbs, 0x2545f491);
  const hiAbs = Math.max(4.0 * eta, 1.5 * etaDetAbs);
  const profileAbs = [0.25 * eta, 0.5 * eta, eta, 2.0 * eta, 4.0 * eta, 1.5 * etaDetAbs];
  gammaAbs.batch(almostSureOnly ? [1e-3, hiAbs] : [1e-3, hiAbs, eta]);
  const almostSureAbs = bisectLyapunovThreshold(gammaAbs.one, 1e-3, hiAbs, 12, profileAbs, gammaAbs.batch);
  if (almostSureOnly) return { etaDetAbs, almostSureAbs };
  // (g) exact operator of the |H| iteration.
  const vmat = assembleCurvatureNoiseOperatorMatrix(abs.gram, abs.dev.n, abs.dev.nPair, abs.dev.pairIndex, abs.dev.pairA, abs.dev.pairB);
  const n = abs.dev.n;
  const nn = Math.imul(n, n);
  const operatorAt = async (e: number): Promise<Record<string, unknown>> => {
    const q: [Float64Array, Float64Array, Float64Array] = [new Float64Array(nn), new Float64Array(nn), new Float64Array(nn)];
    const o: [Float64Array, Float64Array, Float64Array] = [new Float64Array(nn), new Float64Array(nn), new Float64Array(nn)];
    const inv = 1.0 / Math.sqrt(2.0 * n);
    for (let j = 0; j < n; (j = (j + 1) | 0)) {
      q[0][(Math.imul(j, n) + j) | 0] = inv;
      q[2][(Math.imul(j, n) + j) | 0] = inv;
    }
    const m = measureEigenbasisSecondMomentRadius(vmat, n, abs.dev.nPair, abs.dev.pairA, abs.dev.pairB, abs.curvature, beta, e, q[0], q[1], q[2], o[0], o[1], o[2], 200000, 1e-13);
    applyEigenbasisSecondMomentOperator(vmat, n, abs.dev.nPair, abs.dev.pairA, abs.dev.pairB, abs.curvature, beta, e, q[0], q[1], q[2], o[0], o[1], o[2], new Float64Array(abs.dev.nPair), new Float64Array(abs.dev.nPair));
    const iterate = certifySecondMomentRadius(q, o, n, true);
    const witness = await computeSplittingWitnessCertificate(vmat, n, abs.dev.nPair, abs.dev.pairA, abs.dev.pairB, abs.curvature, beta, e);
    const aboveDet = e >= etaDetAbs;
    const certifiedAbove = iterate.certifiedAboveOne || (witness !== null && witness.certificate.certifiedAboveOne) || aboveDet;
    const certifiedBelow = iterate.certifiedBelowOne || (witness !== null && witness.certificate.certifiedBelowOne);
    return { eta: e, radius: m.radius, iterations: m.iterations, converged: m.converged, iterateCertificate: iterate, witness, aboveDeterministicThreshold: aboveDet, verdict: certifiedAbove ? 'rho>1' : certifiedBelow ? 'rho<1' : 'unresolved' };
  };
  const operatorBelow = await operatorAt((1.0 - 1e-3) * etaStarAbs);
  const operatorAbove = await operatorAt((1.0 + 1e-3) * etaStarAbs);
  const operatorAtEta = await operatorAt(eta);
  // (h) Lyapunov exponents at eta and the almost-sure thresholds.
  const gammaReal = createMemoizedLyapunovGamma(full.dev, full.values, beta, batchB, streams, lyapunovSteps, burnIn, seedReal, 0x2545f491);
  const realGrid = [0.25 * eta, 0.5 * eta, eta, 2.0 * eta, 4.0 * eta];
  gammaReal.batch([1e-3, 4.0 * eta, ...realGrid]);
  const gammaAbsAtEta = gammaAbs.one(eta);
  const gammaRealAtEta = gammaReal.one(eta);
  const almostSureReal = bisectLyapunovThreshold(gammaReal.one, 1e-3, 4.0 * eta, 12, realGrid, gammaReal.batch);
  const signProfileReal = gammaReal.batch(realGrid).map((g, i) => ({ eta: realGrid[i]!, ...g }));
  // eta* of the real iteration exists only where H + mu_train I is positive definite (for c4).
  let etaStarReal: number | null = null;
  if (full.values[0]! > 0.0) {
    const opR = assembleThmG6OperatorMatrix(full.gram, full.dev.n, full.dev.nPair, full.dev.pairIndex, full.dev.pairA, full.dev.pairB, full.values, beta);
    etaStarReal = 1.0 / (await computeDenseSymmetricTopEigenpair(opR.a, opR.dim, 1e-10, 400)).value;
  }
  // (i) eta*_g at theta_T.
  const lambdaMin = full.values[0]!;
  const etaStarTether: Array<{ g: number; etaStar: number }> = [];
  for (const g of [1 / 8, 1 / 32, 1 / 128, 1 / 512]) {
    const muE = g + (lambdaMin < 0.0 ? -lambdaMin : 0.0);
    const curv = Float64Array.from(full.values, (v) => v + muE);
    const op = assembleThmG6OperatorMatrix(full.gram, full.dev.n, full.dev.nPair, full.dev.pairIndex, full.dev.pairA, full.dev.pairB, curv, beta);
    etaStarTether.push({ g, etaStar: 1.0 / (await computeDenseSymmetricTopEigenpair(op.a, op.dim, 1e-10, 400)).value });
  }
  return { etaDetAbs, operatorBelow, operatorAbove, operatorAtEta, gammaAbsAtEta, gammaRealAtEta, almostSureAbs, almostSureReal, signProfileReal, etaStarReal, etaStarTether };
}
