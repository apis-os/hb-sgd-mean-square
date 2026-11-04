/**
 * @file labs/mean-square-boundary/evaluate-phase5b-predictions.ts
 * @description The registered decisions of Phase 5b (`evaluatePhase5bPredictions`; README, "Phase 5b"), from its 48 run
 * records and the checks `c4`, `c6` of `evaluatePhase5Predictions`. The record set must be exactly the grid (8 rates, 2 values
 * of `mu_train`, seeds 1401-1403), else nothing is evaluated. A run read by a `q` needs `divergedAt === null` and checkpoints at
 * exactly 200, 400, ..., 20 000; on the 51 window checkpoints (10 000-20 000) every threshold it reads is finite and positive and
 * every other field present and finite, else the `q` fails; a read threshold with `lanczosRelResidual > 1e-10` suspends the `q`
 * (solver). Medians are `computeFloat64Quantile` at 0.5. `q1`-`q6` as registered; `c5b` (either certificate per side, a run
 * without `finalPoint` undecided on both sides); a side certified in the wrong direction is a contradiction. Suspensions: `c4`
 * false or a contradiction suspends `q1`-`q4`, `c6` false suspends `q5`. `replication` holds only if all six hold and none is
 * suspended.
 * @cold
 */
import { computeFloat64Quantile } from './compute-float64-quantile.ts';

type Cp = Record<string, unknown>;
type Run = { eta: number; muTrain: number; seed: number; divergedAt: number | null; checkpoints: Cp[]; finalPoint: Record<string, unknown> | null };
type Q = { holds: boolean; suspended: boolean; reasons: string[]; details: Record<string, unknown> };

const ETAS = [0.01, 0.03, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5];
const MUS = [0, 1e-3];
const SEEDS = [1401, 1402, 1403];
const med = (xs: number[]): number => computeFloat64Quantile(Float64Array.from(xs), 0.5);

function field(o: unknown, keys: readonly string[]): unknown {
  let x = o;
  for (const k of keys) {
    if (x === null || x === undefined || typeof x !== 'object') return undefined;
    x = (x as Record<string, unknown>)[k];
  }
  return x;
}

// Window values of one numeric field of one run; `threshold` adds the positivity and residual rules. Records failures in `q`.
function windowValues(r: Run, keys: readonly string[], threshold: boolean, q: Q): number[] {
  const steps = r.checkpoints.map((c) => c.step as number);
  const exact = steps.length === 100 && steps.every((s, i) => s === 200 * (i + 1));
  if (r.divergedAt !== null || !exact) { q.holds = false; q.reasons.push(`run ${r.eta}/${r.muTrain}/${r.seed} diverged or incomplete`); return []; }
  const out: number[] = [];
  for (const c of r.checkpoints) {
    if ((c.step as number) < 10000) continue;
    const v = field(c, keys);
    if (typeof v !== 'number' || !Number.isFinite(v) || (threshold && !(v > 0))) { q.holds = false; q.reasons.push(`${keys.join('.')} invalid at ${r.eta}/${r.muTrain}/${r.seed}/${String(c.step)}`); continue; }
    if (threshold) {
      const res = field(c, [...keys.slice(0, -1), 'lanczosRelResidual']);
      if (typeof res !== 'number' || !(res <= 1e-10)) { q.suspended = true; q.reasons.push(`solver residual ${String(res)} of ${keys[0]} at ${r.eta}/${r.muTrain}/${r.seed}/${String(c.step)}`); }
    }
    out.push(v);
  }
  return out;
}

const newQ = (): Q => ({ holds: true, suspended: false, reasons: [], details: {} });
const ABS = ['absH', 'etaStar'] as const;
const GN = ['gaussNewton', 'etaStar'] as const;
const NSAT = ['hiddenUnits', 'saturatedHiddenUnits1e4'] as const;

function evaluateQ1(at: (eta: number, mu: number, s: number) => Run): Q {
  const q = newQ();
  const ms: Record<string, number> = {};
  for (const s of SEEDS) {
    const m = ETAS.map((e) => med(windowValues(at(e, 1e-3, s), ABS, true, q)));
    m.forEach((v, i) => { ms[`${ETAS[i]}/${s}`] = v; if (!(v >= 0.07 && v <= 0.18)) q.holds = false; });
    const ratio = Math.max(...m) / Math.min(...m);
    ms[`maxOverMin/${s}`] = ratio;
    if (!(ratio <= 2.5)) q.holds = false;
  }
  q.details = ms;
  return q;
}

function evaluateQ2(at: (eta: number, mu: number, s: number) => Run): Q {
  const q = newQ();
  const det: Record<string, number> = {};
  for (const s of SEEDS) {
    const r = (e: number): number => med(windowValues(at(e, 1e-3, s), ABS, true, q)) / e;
    const [r1, r15, r2, r3, r4] = [r(0.1), r(0.15), r(0.2), r(0.3), r(0.4)];
    Object.assign(det, { [`r0.1/${s}`]: r1, [`r0.15/${s}`]: r15, [`r0.2/${s}`]: r2, [`r0.3/${s}`]: r3, [`r0.4/${s}`]: r4 });
    if (!(r1 > 1.15 && r15 >= 0.8 && r15 <= 1.15 && r3 < 0.45 && r15 > r2 && r2 > r3 && r3 > r4)) q.holds = false;
  }
  q.details = det;
  return q;
}

function evaluateQ3(at: (eta: number, mu: number, s: number) => Run): Q {
  const q = newQ();
  const det: Record<string, number> = {};
  for (const s of SEEDS) {
    const top = at(0.5, 0, s);
    const nTop = med(windowValues(top, NSAT, false, q));
    const rTop = med(windowValues(top, ABS, true, q)) / 0.5;
    Object.assign(det, { [`nSatMedian0.5/${s}`]: nTop, [`r0.5/${s}`]: rTop });
    if (!(nTop >= 1 && rTop >= 1.2 && rTop <= 1.5)) q.holds = false;
    for (const e of ETAS.filter((x) => x <= 0.2)) if (!windowValues(at(e, 0, s), NSAT, false, q).every((v) => v === 0)) q.holds = false;
    for (const e of ETAS) if (med(windowValues(at(e, 1e-3, s), NSAT, false, q)) !== 0) q.holds = false;
  }
  q.details = det;
  return q;
}

function evaluateQ4(at: (eta: number, mu: number, s: number) => Run): Q {
  const q = newQ();
  const det: Record<string, number> = {};
  const pair = (r: Run): { a: number[]; g: number[] } => ({ a: windowValues(r, ABS, true, q), g: windowValues(r, GN, true, q) });
  for (const s of SEEDS) {
    for (const mu of MUS) {
      for (const e of [0.01, 0.03, 0.1]) {
        const { a, g } = pair(at(e, mu, s));
        const eGn = a.length === g.length && a.length > 0 ? med(a.map((x, i) => Math.abs(g[i]! / x - 1))) : Number.NaN;
        det[`eGN/${e}/${mu}/${s}`] = eGn;
        if (!(eGn < (e === 0.1 ? 0.12 : 0.03))) q.holds = false;
      }
    }
    for (const e of [0.3, 0.4, 0.5]) {
      const { a, g } = pair(at(e, 1e-3, s));
      const gGn = a.length === g.length && a.length > 0 ? med(a.map((x, i) => g[i]! / x)) : Number.NaN;
      det[`gGN/${e}/${s}`] = gGn;
      if (!(gGn >= 1.5)) q.holds = false;
    }
  }
  q.details = det;
  return q;
}

function evaluateQ5Q6(at: (eta: number, mu: number, s: number) => Run): { q5: Q; q6: Q } {
  const q5 = newQ();
  const q6 = newQ();
  const d5: Record<string, number> = {};
  const d6: Record<string, number> = {};
  for (const s of SEEDS) for (const mu of MUS) for (const e of ETAS.filter((x) => x >= 0.15)) {
    const r = at(e, mu, s);
    const b = Math.max(
      med(windowValues(r, ['batchSharpness', 'expectationOfRatioScaled'], false, q5)),
      med(windowValues(r, ['batchSharpness', 'ratioOfExpectationsScaled'], false, q5)),
      med(windowValues(r, ['lambda1Scaled'], false, q5)),
    );
    d5[`${e}/${mu}/${s}`] = b;
    if (!(b < 0.38) || (e <= 0.3 && !(b < 0.25))) q5.holds = false;
    const lm = windowValues(r, ['lambdaMin'], false, q6);
    const f = lm.length === 51 ? lm.filter((v) => v < 0).length / 51 : Number.NaN;
    d6[`${e}/${mu}/${s}`] = f;
    if (!(f >= 0.85)) q6.holds = false;
  }
  q5.details = d5;
  q6.details = d6;
  return { q5, q6 };
}

type Side = { iterateCertificate?: { certifiedBelowOne?: boolean; certifiedAboveOne?: boolean }; witness?: { certificate?: { certifiedBelowOne?: boolean; certifiedAboveOne?: boolean } } | null; aboveDeterministicThreshold?: boolean };

function evaluateC5b(runs: readonly Run[]): { c5b: boolean; undecided: unknown[]; contradictions: unknown[] } {
  const undecided: unknown[] = [];
  const contradictions: unknown[] = [];
  const live = runs.filter((r) => r.divergedAt === null);
  for (const r of live) {
    const id = { eta: r.eta, muTrain: r.muTrain, seed: r.seed };
    const b = (r.finalPoint?.operatorBelow ?? null) as Side | null;
    const a = (r.finalPoint?.operatorAbove ?? null) as Side | null;
    const below = b !== null && (b.iterateCertificate?.certifiedBelowOne === true || b.witness?.certificate?.certifiedBelowOne === true);
    const above = a !== null && (a.iterateCertificate?.certifiedAboveOne === true || a.aboveDeterministicThreshold === true || a.witness?.certificate?.certifiedAboveOne === true);
    if (!below) undecided.push({ ...id, side: 'below' });
    if (!above) undecided.push({ ...id, side: 'above' });
    if (b !== null && (b.iterateCertificate?.certifiedAboveOne === true || b.witness?.certificate?.certifiedAboveOne === true)) contradictions.push({ ...id, side: 'below' });
    if (a !== null && (a.iterateCertificate?.certifiedBelowOne === true || a.witness?.certificate?.certifiedBelowOne === true)) contradictions.push({ ...id, side: 'above' });
  }
  return { c5b: live.length > 0 && undecided.length === 0, undecided, contradictions };
}

export function evaluatePhase5bPredictions(runs: readonly Run[], checks: { readonly c4: boolean; readonly c6: boolean }): Record<string, unknown> {
  const key = (e: number, mu: number, s: number): string => `${e}/${mu}/${s}`;
  const byKey = new Map(runs.map((r) => [key(r.eta, r.muTrain, r.seed), r]));
  const complete = runs.length === 48 && byKey.size === 48 && ETAS.every((e) => MUS.every((mu) => SEEDS.every((s) => byKey.has(key(e, mu, s)))));
  if (!complete) return { gridComplete: false, label: 'record set is not the 48-run grid of Phase 5b; nothing evaluated' };
  const at = (e: number, mu: number, s: number): Run => byKey.get(key(e, mu, s))!;
  const c5 = evaluateC5b(runs);
  const qs: Record<string, Q> = { q1: evaluateQ1(at), q2: evaluateQ2(at), q3: evaluateQ3(at), q4: evaluateQ4(at), ...evaluateQ5Q6(at) };
  const suspendAbs = !checks.c4 || c5.contradictions.length > 0;
  for (const k of ['q1', 'q2', 'q3', 'q4']) if (suspendAbs) { qs[k]!.suspended = true; qs[k]!.reasons.push(!checks.c4 ? 'c4 failed' : 'certificate contradiction'); }
  if (!checks.c6) { qs.q5!.suspended = true; qs.q5!.reasons.push('c6 failed'); }
  const replication = Object.values(qs).every((q) => q.holds && !q.suspended);
  return { gridComplete: true, ...qs, c5b: c5.c5b, c5bUndecidedSides: c5.undecided, certificateContradictions: c5.contradictions, checks, replication };
}
