/**
 * @file labs/mean-square-boundary/evaluate-phase5-amendment3-diagnostics.ts
 * @description The reported tables of amendment 3 over the run records (`evaluatePhase5Amendment3Diagnostics`; no decision rule,
 * H1-H5, `p1`-`p6` and `c4`-`c6` are decided as registered by `evaluatePhase5Predictions`). Window and statistics as registered
 * (steps 10 000-20 000, window medians per run, seeds never pooled). Per run: window medians of `eta* / eta` for the added criteria
 * (`gaussNewton`, `hybridGnCurvatureFullNoise`, `rigidPositive` 0.05 and 0.02, `absHRestricted1e2`), of the hidden-unit counts
 * and of the data loss; `indefiniteRel1e2Fraction` (checkpoints with `lambda_min(H + mu I) < -1e-2 lambda_1`, beside the strict
 * fraction of `p1`); the `p6` ratios against `p6FormulaWithMu` and `p6FormulaResidualWeighted` and the decomposition `a0`-`a4`
 * (the `e13` stream quantities of the companion lab are not computed here). `perMuTrainByCriterion` applies the H1/H2 reading of the registration to each added criterion on
 * the saturated set (labelled "no decision"). `saturatedRunsByActivationRegime` splits the saturated runs by the window median
 * of `saturatedHiddenUnits1e4` (0 against at least 1); `lossStratification` uses `trivialLoss = Var(y) / 2` (population variance,
 * the loss of the constant predictor) with the cut fixed in amendment 3 (productive: data-loss median below `0.25 trivialLoss`
 * and at most one saturated unit). `p3Contrast` reports, beside `p3`, the relative differences of the two tether-free cuts and of
 * Gauss–Newton against `rigidPositive` 0.05. `pairedExtended` adds to the paired table of amendment 1 the median of
 * `|eta_c/eta - 1|` for each added criterion, for the Gauss–Newton `eta_D` and restricted `eta_D` (upper bounds of their `eta*`:
 * the value measures the slack of a bound, not the error of a predictor) and for the one-dimensional ceiling `absH.etaOneDim`.
 * @cold
 */
import { computeFloat64Quantile } from './compute-float64-quantile.ts';

type Run = Record<string, unknown> & { eta: number; muTrain: number; seed: number; divergedAt: number | null; checkpoints: Array<Record<string, unknown>>; window: Array<Record<string, unknown>> };
type Cp = Record<string, unknown>;

const CRITERIA: ReadonlyArray<readonly [string, readonly string[]]> = [
  ['gaussNewton', ['gaussNewton', 'etaStar']],
  ['hybridGnCurvatureFullNoise', ['hybridGnCurvatureFullNoise', 'etaStar']],
  ['rigidPositive0.05', ['rigidPositive', '0.05', 'etaStar']],
  ['rigidPositive0.02', ['rigidPositive', '0.02', 'etaStar']],
  ['absHRestricted1e2', ['absHRestricted1e2', 'etaStar']],
];

const median = (xs: number[]): number | null => (xs.length === 0 ? null : computeFloat64Quantile(Float64Array.from(xs), 0.5));

// Nested number at `keys`, or null when any level is missing or null.
function pick(o: unknown, keys: readonly string[]): number | null {
  let x = o;
  for (const k of keys) {
    if (x === null || x === undefined || typeof x !== 'object') return null;
    x = (x as Record<string, unknown>)[k];
  }
  return typeof x === 'number' ? x : null;
}

const values = (win: Cp[], keys: readonly string[]): number[] => win.map((c) => pick(c, keys)).filter((v): v is number => v !== null);
const restrictedAt = (c: Cp, list: string, r: number, field: string): number | null => {
  const arr = (c.gaussNewton as Record<string, unknown> | undefined)?.[list] as Array<Record<string, number>> | undefined;
  const hit = arr?.find((x) => x.r === r);
  return hit === undefined ? null : hit[field]!;
};

function summariseRun(r: Run): Record<string, unknown> {
  const win = r.checkpoints.filter((c) => (c.step as number) >= 10000 && (c.step as number) <= 20000);
  const ratios: Record<string, number | null> = {};
  for (const [name, keys] of CRITERIA) ratios[name] = median(values(win, keys).map((v) => v / r.eta));
  const rel = (num: readonly string[]): number | null => median(win.map((c) => { const a = pick(c, num); const b = pick(c, ['batchSharpness', 'ratioOfExpectations']); return a === null || b === null ? null : b / a; }).filter((v): v is number => v !== null));
  return {
    eta: r.eta, muTrain: r.muTrain, seed: r.seed, nonDivergent: r.divergedAt === null && win.length > 0, ratioMedians: ratios,
    absHRatioMedian: median(values(win, ['absH', 'etaStar']).map((v) => v / r.eta)),
    saturatedHiddenUnits1e4Median: median(values(win, ['hiddenUnits', 'saturatedHiddenUnits1e4'])),
    activeTangentDimMedian: median(values(win, ['hiddenUnits', 'activeTangentDim'])),
    dataLossMedian: median(values(win, ['dataLoss'])),
    indefiniteRel1e2Fraction: win.length > 0 ? win.filter((c) => (c.lambdaMin as number) < -1e-2 * (c.lambda1 as number)).length / win.length : null,
    p6: {
      ratioOverFormulaWithMu: rel(['p6FormulaWithMu']), ratioOverResidualWeighted: rel(['p6FormulaResidualWeighted']),
      decomposition: Object.fromEntries(['a0MonteCarlo', 'a1ExactFullHessian', 'a2ExactGaussNewton', 'a3GaussNewtonMeanDropped', 'a4RegisteredFormula'].map((k) => [k, median(values(win, ['decomposition', k]))])),
    },
    p3Contrast: {
      rigidCutsRelative: median(win.map((c) => { const a = pick(c, ['rigidPositive', '0.05', 'etaStar']); const b = pick(c, ['rigidPositive', '0.02', 'etaStar']); return a === null || b === null ? null : Math.abs(a - b) / a; }).filter((v): v is number => v !== null)),
      gaussNewtonVsRigid05Relative: median(win.map((c) => { const a = pick(c, ['rigidPositive', '0.05', 'etaStar']); const g = pick(c, ['gaussNewton', 'etaStar']); return a === null || g === null ? null : Math.abs(g - a) / a; }).filter((v): v is number => v !== null)),
    },
  };
}

function pairedRow(r: Run): Record<string, unknown> {
  const rows = r.window.map((w) => r.checkpoints.find((x) => x.step === w.step)!);
  const err = (get: (c: Cp) => number | null): number | null => median(rows.map(get).filter((v): v is number => v !== null).map((v) => Math.abs(v / r.eta - 1.0)));
  const out: Record<string, unknown> = { eta: r.eta, muTrain: r.muTrain, seed: r.seed };
  for (const [name, keys] of CRITERIA) out[`e_${name}`] = err((c) => pick(c, keys));
  out.e_gaussNewtonEtaD_boundSlack = err((c) => pick(c, ['gaussNewton', 'etaD']));
  for (const rr of [1, 4, 8, 16]) {
    out[`e_gaussNewtonRestricted${rr}`] = err((c) => restrictedAt(c, 'restricted', rr, 'eta'));
    out[`e_gaussNewtonRestrictedEtaD${rr}_boundSlack`] = err((c) => restrictedAt(c, 'restrictedEtaD', rr, 'etaD'));
  }
  out.eOneDim = err((c) => pick(c, ['absH', 'etaOneDim']));
  return out;
}

export function evaluatePhase5Amendment3Diagnostics(runs: readonly Run[], saturated: ReadonlyArray<{ eta: number; muTrain: number }>, trivialLoss: number): Record<string, unknown> {
  const summaries = runs.map(summariseRun);
  const isSat = (s: { eta: number; muTrain: number }): boolean => saturated.some((x) => x.eta === s.eta && x.muTrain === s.muTrain);
  const mus = [...new Set(runs.map((r) => r.muTrain))].sort((a, b) => a - b);
  const satRuns = (mu: number) => summaries.filter((s) => s.muTrain === mu && s.nonDivergent === true && isSat(s as { eta: number; muTrain: number }));
  const perMuTrainByCriterion = mus.map((mu) => {
    const sr = satRuns(mu);
    const byCriterion: Record<string, unknown> = {};
    for (const [name] of CRITERIA) {
      const meds = sr.map((s) => (s.ratioMedians as Record<string, number | null>)[name] ?? null);
      byCriterion[name] = { allMediansIn08To125: sr.length > 0 && meds.every((m) => m !== null && m >= 0.8 && m <= 1.25), medians: meds };
    }
    return { muTrain: mu, saturatedRuns: sr.length, label: 'no decision', byCriterion };
  });
  const allBelowHalf = (name: string, keys: readonly string[]) => runs.filter((r) => r.divergedAt === null && isSat(r)).map((r) => ({ eta: r.eta, muTrain: r.muTrain, seed: r.seed, criterion: name, allBelowHalf: values(r.checkpoints.filter((c) => (c.step as number) >= 10000 && (c.step as number) <= 20000), keys).every((v) => v / r.eta < 0.5) }));
  const sat = summaries.filter((s) => s.nonDivergent === true && isSat(s as { eta: number; muTrain: number }));
  const regime = (s: Record<string, unknown>): string => ((s.saturatedHiddenUnits1e4Median as number | null) === 0 ? 'unsaturated' : 'unitSaturated');
  const productive = (s: Record<string, unknown>): boolean => (s.dataLossMedian as number | null) !== null && (s.dataLossMedian as number) < 0.25 * trivialLoss && (s.saturatedHiddenUnits1e4Median as number | null) !== null && (s.saturatedHiddenUnits1e4Median as number) <= 1;
  return {
    label: 'reported quantities of amendment 3; no decision rule',
    trivialLoss,
    perMuTrainByCriterion,
    allBelowHalfByCriterion: CRITERIA.flatMap(([name, keys]) => allBelowHalf(name, keys)),
    saturatedRunsByActivationRegime: sat.map((s) => ({ eta: s.eta, muTrain: s.muTrain, seed: s.seed, regime: regime(s), absHRatioMedian: s.absHRatioMedian })),
    lossStratification: sat.map((s) => ({ eta: s.eta, muTrain: s.muTrain, seed: s.seed, productive: productive(s), dataLossMedian: s.dataLossMedian, saturatedHiddenUnits1e4Median: s.saturatedHiddenUnits1e4Median })),
    pairedExtended: runs.filter((r) => r.divergedAt === null && r.window.length > 0).map(pairedRow),
    summaries,
  };
}
