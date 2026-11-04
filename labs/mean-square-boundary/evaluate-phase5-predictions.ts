/**
 * @file labs/mean-square-boundary/evaluate-phase5-predictions.ts
 * @description The registered decisions of Phase 5 from the run records of `runPhase5Run` (`evaluatePhase5Predictions`).
 * Window: checkpoints at steps 10 000-20 000 inclusive; statistics are window medians per run; seeds are never pooled. Saturated
 * regime: the `(eta, mu_train)`, `eta != 0.01`, for which every non-divergent seed has window median `lambda_1` below 0.8 times
 * that of the control `eta = 0.01` of the same seed and `mu_train`. With `r_t = eta*_|H|(theta_t) / eta`: H1 (per `mu_train`,
 * every saturated `eta`, every seed: window median of `r_t` in `[0.8, 1.25]`); H2 (`r_t < 0.5` at every window checkpoint of every
 * saturated run); otherwise "intermediate"; H3 per run (`lambda_min(H + mu I) < 0` at more than half of the window checkpoints),
 * at an `(eta, mu)` "holds" if it holds in every non-divergent seed and "fails" if it fails in every one; H4 per saturated run
 * at `theta_T` (`|H|` operator verdict `rho > 1` at `eta` and `gamma_|H|(eta) + 3 SE < 0`); H5 per `mu_train`
 * (`eta_a.s.,|H|(theta_T) / eta` in `[0.8, 1.25]` in every saturated run). Predictions `p1`-`p6` and checks `c4`-`c6` as written
 * in the registration; the paired comparison of amendment 1 (item 2). Missing quantities (no crossing, divergence) are counted
 * and make a universally quantified statement false only where the registration requires the quantity. Amendment 3: the last
 * clause of `p5` (real iteration, `gamma > 0` on the sign grid) is read on the saturated runs, as the rest of `p5`, with the
 * reading on every run reported beside it; the window filter of `p3` names both ends; the reported tables of
 * `evaluatePhase5Amendment3Diagnostics` under `amendment3`.
 * @cold
 */
import { computeFloat64Quantile } from './compute-float64-quantile.ts';
import { evaluatePhase5Amendment3Diagnostics } from './evaluate-phase5-amendment3-diagnostics.ts';
import { createTwoLayerMlpDatasetAndOps } from './mlp-network.ts';

type Run = Record<string, unknown> & { eta: number; muTrain: number; seed: number; divergedAt: number | null; checkpoints: Array<Record<string, unknown>>; window: Array<Record<string, unknown>>; finalPoint: Record<string, unknown> | null; c6: Record<string, number> | null };

const median = (xs: number[]): number | null => (xs.length === 0 ? null : computeFloat64Quantile(Float64Array.from(xs), 0.5));
const num = (o: unknown, ...keys: string[]): number => {
  let x = o as Record<string, unknown>;
  for (const k of keys) x = x[k] as Record<string, unknown>;
  return x as unknown as number;
};

export function evaluatePhase5Predictions(runs: readonly Run[]): Record<string, unknown> {
  const beta = 0.9;
  const etas = [...new Set(runs.map((r) => r.eta))].sort((a, b) => a - b);
  const mus = [...new Set(runs.map((r) => r.muTrain))].sort((a, b) => a - b);
  const summaries = runs.map((r) => {
    const win = r.checkpoints.filter((c) => (c.step as number) >= 10000 && (c.step as number) <= 20000);
    const ok = r.divergedAt === null && win.length > 0;
    const rt = win.map((c) => num(c, 'absH', 'etaStar') / r.eta);
    const r32 = win.map((c) => num(c, 'tethered', String(1 / 32), 'etaStar') / r.eta);
    const r128 = win.map((c) => num(c, 'tethered', String(1 / 128), 'etaStar') / r.eta);
    const fp = r.finalPoint;
    const etaAs = fp === null ? null : ((fp.almostSureAbs as { etaAs: number | null }).etaAs);
    const gammaAbs = fp === null ? null : (fp.gammaAbsAtEta as { mean: number; standardError: number });
    const verdictAtEta = fp === null ? null : ((fp.operatorAtEta as { verdict: string }).verdict);
    return {
      eta: r.eta, muTrain: r.muTrain, seed: r.seed, divergedAt: r.divergedAt, nonDivergent: ok, windowCheckpoints: win.length,
      lambda1Median: median(win.map((c) => c.lambda1 as number)),
      rMedian: median(rt), rMax: rt.length > 0 ? Math.max(...rt) : null,
      allBelowHalf: rt.length > 0 && rt.every((x) => x < 0.5),
      r32Median: median(r32), r128Median: median(r128),
      indefiniteFraction: win.length > 0 ? win.filter((c) => (c.lambdaMin as number) < 0.0).length / win.length : null,
      bsScaledMedian: median(win.map((c) => num(c, 'batchSharpness', 'expectationOfRatioScaled'))),
      lambda1ScaledMedian: median(win.map((c) => c.lambda1Scaled as number)),
      p6RatioMedian: median(win.map((c) => num(c, 'batchSharpness', 'ratioOfExpectations') / (c.p6Formula as number))),
      e13Median: median(win.filter((c) => c.e13 !== null).map((c) => num(c, 'e13', 'rHat') - num(c, 'e13', 'predicted'))),
      etaStarAbsFinal: fp === null ? null : (r.checkpoints[r.checkpoints.length - 1]!.absH as { etaStar: number }).etaStar,
      etaAsAbsFinal: etaAs,
      verdictAtEta,
      gammaAbsAtEta: gammaAbs,
      h4: fp !== null && verdictAtEta === 'rho>1' && gammaAbs !== null && gammaAbs.mean + 3.0 * gammaAbs.standardError < 0.0,
      realProfileAllPositive: fp === null ? null : (fp.signProfileReal as Array<{ mean: number }>).every((g) => g.mean > 0.0),
    };
  });
  const find = (eta: number, mu: number, seed: number) => summaries.find((s) => s.eta === eta && s.muTrain === mu && s.seed === seed);
  const saturated: Array<{ eta: number; muTrain: number }> = [];
  for (const mu of mus) {
    for (const eta of etas) {
      if (eta === 0.01) continue;
      const cell = summaries.filter((s) => s.eta === eta && s.muTrain === mu && s.nonDivergent);
      if (cell.length === 0) continue;
      if (cell.every((s) => { const c = find(0.01, mu, s.seed); return c !== undefined && c.lambda1Median !== null && s.lambda1Median !== null && s.lambda1Median < 0.8 * c.lambda1Median; })) saturated.push({ eta, muTrain: mu });
    }
  }
  const isSat = (s: { eta: number; muTrain: number }): boolean => saturated.some((x) => x.eta === s.eta && x.muTrain === s.muTrain);
  const satRuns = (mu: number) => summaries.filter((s) => s.muTrain === mu && s.nonDivergent && isSat(s));
  const perMu = mus.map((mu) => {
    const sr = satRuns(mu);
    const h1 = sr.length > 0 && sr.every((s) => s.rMedian !== null && s.rMedian >= 0.8 && s.rMedian <= 1.25);
    const h2 = sr.length > 0 && sr.every((s) => s.allBelowHalf);
    const h4Runs = sr.filter((s) => s.h4).length;
    const h5 = sr.length > 0 && sr.every((s) => s.etaAsAbsFinal !== null && s.etaAsAbsFinal / s.eta >= 0.8 && s.etaAsAbsFinal / s.eta <= 1.25);
    const outcome = h1 ? 'H1' : h2 ? (h4Runs > 0 ? 'H2 with H4' : 'H2 without H4') : 'intermediate';
    return { muTrain: mu, saturatedRuns: sr.length, h1, h2, h4Runs, h5, outcome, rMedians: sr.map((s) => ({ eta: s.eta, seed: s.seed, rMedian: s.rMedian })) };
  });
  const h3At = (eta: number, mu: number): 'holds' | 'fails' | 'mixed' | null => {
    const cell = summaries.filter((s) => s.eta === eta && s.muTrain === mu && s.nonDivergent && s.indefiniteFraction !== null);
    if (cell.length === 0) return null;
    if (cell.every((s) => s.indefiniteFraction! > 0.5)) return 'holds';
    if (cell.every((s) => s.indefiniteFraction! <= 0.5)) return 'fails';
    return 'mixed';
  };
  const p1 = mus.every((mu) => etas.filter((e) => e >= 0.2).every((e) => h3At(e, mu) === 'holds') && etas.filter((e) => e <= 0.03).every((e) => h3At(e, mu) === 'fails'));
  const controlOk = summaries.filter((s) => s.eta === 0.01 && s.nonDivergent).every((s) => s.rMedian !== null && s.rMedian > 5);
  const p2 = perMu.every((m) => m.h1) && mus.every((mu) => satRuns(mu).every((s) => s.rMedian! >= 0.8 && s.rMedian! <= 1.0)) && controlOk;
  const verdictOf = (rs: Array<{ med: number | null; below: boolean }>): string => (rs.every((x) => x.med !== null && x.med >= 0.8 && x.med <= 1.25) ? 'H1' : rs.every((x) => x.below) ? 'H2' : 'intermediate');
  const p3Cells = saturated.map((c) => {
    const rs = summaries.filter((s) => s.eta === c.eta && s.muTrain === c.muTrain && s.nonDivergent);
    const vAbs = verdictOf(rs.map((s) => ({ med: s.rMedian, below: s.allBelowHalf })));
    const runsC = runs.filter((r) => r.eta === c.eta && r.muTrain === c.muTrain && r.divergedAt === null);
    const v32 = verdictOf(runsC.map((r) => {
      const w = r.checkpoints.filter((x) => (x.step as number) >= 10000 && (x.step as number) <= 20000).map((x) => num(x, 'tethered', String(1 / 32), 'etaStar') / r.eta);
      return { med: median(w), below: w.length > 0 && w.every((v) => v < 0.5) };
    }));
    const close = rs.every((s) => s.r32Median !== null && s.r128Median !== null && Math.abs(s.r32Median - s.r128Median) / s.r128Median < 0.05);
    return { ...c, verdictAbs: vAbs, verdictTether32: v32, sameVerdict: vAbs === v32, tetherInsensitive: close };
  });
  const p3 = p3Cells.length > 0 && p3Cells.every((c) => c.sameVerdict && c.tetherInsensitive);
  const allSat = summaries.filter((s) => s.nonDivergent && isSat(s));
  const p4 = allSat.length > 0 && allSat.every((s) => s.bsScaledMedian !== null && s.bsScaledMedian < 0.5 && s.lambda1ScaledMedian !== null && s.lambda1ScaledMedian < 0.5);
  const p5Order = allSat.length > 0 && allSat.every((s) => s.etaStarAbsFinal !== null && s.etaAsAbsFinal !== null && s.etaStarAbsFinal <= s.eta && s.eta <= s.etaAsAbsFinal);
  const p5H4 = perMu.every((m) => m.h4Runs >= 1);
  const p5H5Fails = allSat.length > 0 && allSat.filter((s) => s.etaAsAbsFinal === null || s.etaAsAbsFinal / s.eta > 1.25).length >= allSat.length / 2;
  // Amendment 3: the clause on the real iteration belongs to `p5` "in every saturated run" and is read on those runs; the reading
  // on every run is reported beside it.
  const p5Real = allSat.filter((s) => s.realProfileAllPositive !== null).every((s) => s.realProfileAllPositive === true);
  const p5RealAllRuns = summaries.filter((s) => s.nonDivergent && s.realProfileAllPositive !== null).every((s) => s.realProfileAllPositive === true);
  const p5 = p5Order && p5H4 && p5H5Fails && p5Real;
  const lowEta = summaries.filter((s) => s.nonDivergent && s.eta <= 0.1);
  const p6 = lowEta.length > 0 && lowEta.every((s) => s.p6RatioMedian !== null && s.p6RatioMedian >= 0.85 && s.p6RatioMedian <= 1.15);
  // c4: eta_a.s. >= eta* for the same iteration wherever both exist (|H| at theta_T and at the window checkpoints; real where H > 0).
  let c4 = true;
  const c4Cases: Array<Record<string, unknown>> = [];
  for (const r of runs) {
    if (r.divergedAt !== null || r.finalPoint === null) continue;
    const fp = r.finalPoint;
    const pairs: Array<{ where: string; etaStar: number | null; bracket: readonly [number, number] | null }> = [
      { where: 'final |H|', etaStar: (r.checkpoints[r.checkpoints.length - 1]!.absH as { etaStar: number }).etaStar, bracket: (fp.almostSureAbs as { bracket: readonly [number, number] | null }).bracket },
      { where: 'final real', etaStar: fp.etaStarReal as number | null, bracket: (fp.almostSureReal as { bracket: readonly [number, number] | null }).bracket },
      ...r.window.map((w) => ({ where: `window ${String(w.step)}`, etaStar: w.etaStarAbs as number, bracket: (w.almostSureAbs as { bracket: readonly [number, number] | null }).bracket })),
    ];
    for (const pr of pairs) {
      if (pr.etaStar === null || pr.bracket === null) continue;
      const okPair = pr.bracket[1] >= pr.etaStar * (1.0 - 1e-3);
      if (!okPair) { c4 = false; c4Cases.push({ eta: r.eta, muTrain: r.muTrain, seed: r.seed, where: pr.where, etaStar: pr.etaStar, bracket: pr.bracket }); }
    }
  }
  const c5Runs = runs.filter((r) => r.divergedAt === null && r.finalPoint !== null).map((r) => {
    const fp = r.finalPoint!;
    const b = fp.operatorBelow as { iterateCertificate: { certifiedBelowOne: boolean } };
    const a = fp.operatorAbove as { iterateCertificate: { certifiedAboveOne: boolean }; aboveDeterministicThreshold: boolean };
    return { eta: r.eta, muTrain: r.muTrain, seed: r.seed, below: b.iterateCertificate.certifiedBelowOne, above: a.iterateCertificate.certifiedAboveOne || a.aboveDeterministicThreshold };
  });
  const c5 = c5Runs.length > 0 && c5Runs.every((x) => x.below && x.above);
  // Amendment 4 (post hoc, written after the results): c5 also read with a side certified by either certificate of amendment 2,
  // item 3 (the Collatz-Wielandt bracket at the last power iterate or the witness of Theorem splitting); the registered c5 stays.
  const c5EitherRuns = runs.filter((r) => r.divergedAt === null && r.finalPoint !== null).map((r) => {
    const fp = r.finalPoint!;
    type Side = { iterateCertificate: { certifiedBelowOne: boolean; certifiedAboveOne: boolean }; witness: { certificate: { certifiedBelowOne: boolean; certifiedAboveOne: boolean } } | null; aboveDeterministicThreshold: boolean };
    const b = fp.operatorBelow as Side;
    const a = fp.operatorAbove as Side;
    return {
      eta: r.eta, muTrain: r.muTrain, seed: r.seed,
      below: b.iterateCertificate.certifiedBelowOne || (b.witness !== null && b.witness.certificate.certifiedBelowOne),
      above: a.iterateCertificate.certifiedAboveOne || a.aboveDeterministicThreshold || (a.witness !== null && a.witness.certificate.certifiedAboveOne),
    };
  });
  const c5EitherCertificate = c5EitherRuns.length > 0 && c5EitherRuns.every((x) => x.below && x.above);
  const c6Rec = runs.find((r) => r.c6 !== null)?.c6 ?? null;
  const c6 = c6Rec !== null && (c6Rec.relativeDifference ?? Number.POSITIVE_INFINITY) < 1e-10;
  // Amendment 1, item 2: paired errors at the window checkpoints of the runs with eta >= 0.1.
  const paired = runs.filter((r) => r.divergedAt === null && r.window.length > 0).map((r) => {
    const rows = r.window.map((w) => {
      const c = r.checkpoints.find((x) => x.step === w.step)!;
      const etaAs = (w.almostSureAbs as { etaAs: number | null }).etaAs;
      const maxAbs = Math.max(Math.abs(c.lambda1 as number), Math.abs(c.lambdaMin as number));
      return {
        eAs: etaAs === null ? null : Math.abs(etaAs / r.eta - 1.0),
        eMs: Math.abs(num(c, 'absH', 'etaStar') / r.eta - 1.0),
        eBs: Math.abs(num(c, 'batchSharpness', 'expectationOfRatioScaled') - 1.0),
        eLambda: Math.abs((maxAbs * r.eta) / (2.0 * (1.0 + beta)) - 1.0),
      };
    });
    const asVals = rows.filter((x) => x.eAs !== null).map((x) => x.eAs!);
    return { eta: r.eta, muTrain: r.muTrain, seed: r.seed, missingAlmostSure: rows.length - asVals.length, eAs: median(asVals), eMs: median(rows.map((x) => x.eMs)), eBs: median(rows.map((x) => x.eBs)), eLambda: median(rows.map((x) => x.eLambda)) };
  });
  const pairedVerdict = mus.map((mu) => {
    const sr = paired.filter((p) => p.muTrain === mu && isSat(p));
    const asCloser = sr.length > 0 && sr.every((p) => p.eAs !== null && p.eMs !== null && p.eAs < p.eMs);
    const msCloser = sr.length > 0 && sr.every((p) => p.eAs !== null && p.eMs !== null && p.eMs < p.eAs);
    return { muTrain: mu, runs: sr.length, verdict: asCloser ? 'almost-sure closer' : msCloser ? 'mean-square closer' : 'mixed' };
  });
  return {
    saturated,
    perMuTrain: perMu,
    h3: mus.flatMap((mu) => etas.map((eta) => ({ eta, muTrain: mu, h3: h3At(eta, mu) }))),
    predictions: { p1, p2, p3, p4, p5, p6 },
    predictionDetails: { p2ControlAbove5: controlOk, p3Cells, p5: { order: p5Order, h4PerMu: p5H4, h5FailsInHalf: p5H5Fails, realProfileAllPositive: p5Real, realProfileAllPositiveAllRuns: p5RealAllRuns } },
    checks: { c4, c5, c6 },
    checkDetails: { c4Failures: c4Cases, c5Runs, c6: c6Rec, amendment4: { c5EitherCertificate, undecidedSides: c5EitherRuns.flatMap((x) => [...(x.below ? [] : [{ eta: x.eta, muTrain: x.muTrain, seed: x.seed, side: 'below' }]), ...(x.above ? [] : [{ eta: x.eta, muTrain: x.muTrain, seed: x.seed, side: 'above' }])]), c5EitherRuns } },
    paired,
    pairedVerdict,
    summaries,
    amendment3: evaluatePhase5Amendment3Diagnostics(runs, saturated, trivialLossOf(createTwoLayerMlpDatasetAndOps().yData)),
  };
}

// Loss of the constant predictor `mean(y)` under the training loss `mean_i (f_i - y_i)^2 / 2`: half the population variance.
function trivialLossOf(y: Float64Array): number {
  let mean = 0.0;
  for (let i = 0; i < y.length; (i = (i + 1) | 0)) mean += y[i]! / y.length;
  let v = 0.0;
  for (let i = 0; i < y.length; (i = (i + 1) | 0)) v += ((y[i]! - mean) * (y[i]! - mean)) / y.length;
  return 0.5 * v;
}
