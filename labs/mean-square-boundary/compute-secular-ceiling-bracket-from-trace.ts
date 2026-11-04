/**
 * @file labs/mean-square-boundary/compute-secular-ceiling-bracket-from-trace.ts
 * @description Two-sided bracket on the root of the secular equation of Theorem G from `lambda_1`, `lambda_2` and `tr H` alone (`computeSecularCeilingBracketFromTrace`), for a Hessian with `0 <= lambda_k <= lambda_2` for `k >= 2`: with `f(x) = x / (1 - c x)` convex and `f(x) / x` increasing, `tr H - lambda_1 <= sum_{k >= 2} f(lambda_k) <= (tr H - lambda_1) f(lambda_2) / lambda_2`, so the root with the upper sum is a lower bound on the ceiling and the root with the lower sum an upper bound. Also returns the sandwich of Theorem G(3), the one-dimensional ceiling of Theorem E(3) and the noise-rank ratio `kappa_beta tr H / (B lambda_1)`. The bracket is `NaN` when `alpha_diag <= 0`.
 * @cold
 */
export function computeSecularCeilingBracketFromTrace(
  lambda1: number,
  lambda2: number,
  traceH: number,
  beta: number,
  batchB: number,
  kappaFeat: number,
): {
  readonly alphaDiag: number;
  readonly noiseRankRatio: number;
  readonly etaCritDeterministic: number;
  readonly etaCritOneDim: number;
  readonly etaSandwichLower: number;
  readonly etaSandwichUpper: number;
  readonly etaTraceBracketLower: number;
  readonly etaTraceBracketUpper: number;
} {
  const kappaBeta = (1.0 + beta) / (1.0 - beta);
  const twoOnePlusBeta = 2.0 * (1.0 + beta);
  const alphaDiag = 1.0 + (kappaBeta * (1.0 + kappaFeat)) / batchB;
  const nu11Sq = (lambda1 * lambda1 * (2.0 + kappaFeat)) / batchB;
  const etaCritOneDim = (twoOnePlusBeta * lambda1) / (lambda1 * lambda1 + kappaBeta * nu11Sq);
  const kbTraceOverB = (kappaBeta * traceH) / batchB;
  const sandwichUpperMu = Math.max(alphaDiag, 0.0) * lambda1 + kbTraceOverB;
  const sandwichLowerMu = Math.max(
    alphaDiag * lambda1 + (kappaBeta * lambda1) / batchB,
    Math.min(alphaDiag, 0.0) * lambda1 + kbTraceOverB,
  );
  const result = {
    alphaDiag,
    noiseRankRatio: kbTraceOverB / lambda1,
    etaCritDeterministic: twoOnePlusBeta / lambda1,
    etaCritOneDim,
    etaSandwichLower: twoOnePlusBeta / sandwichUpperMu,
    etaSandwichUpper: twoOnePlusBeta / sandwichLowerMu,
    etaTraceBracketLower: Number.NaN,
    etaTraceBracketUpper: Number.NaN,
  };
  if (alphaDiag <= 0.0 || lambda2 <= 0.0 || traceH <= lambda1) return result;
  const etaMax = twoOnePlusBeta / (alphaDiag * lambda1);
  const rest = traceH - lambda1;
  const scale = 1.0 / (2.0 * (1.0 - beta) * batchB);
  // residual(eta, useUpperSum): eta * S(eta) / (2 (1 - beta) B) - 1, increasing in eta on (0, etaMax)
  const roots = new Float64Array(2);
  for (let which = 0; which < 2; (which = (which + 1) | 0)) {
    let lo = 0.0;
    let hi = etaMax;
    for (let it = 0; it < 200; (it = (it + 1) | 0)) {
      const mid = 0.5 * (lo + hi);
      const c = (mid * alphaDiag) / twoOnePlusBeta;
      const f1 = lambda1 / (1.0 - c * lambda1);
      const restSum = which === 0 ? rest / (1.0 - c * lambda2) : rest;
      const residual = mid * scale * (f1 + restSum) - 1.0;
      if (residual > 0.0) hi = mid;
      else lo = mid;
    }
    roots[which] = 0.5 * (lo + hi);
  }
  return {
    ...result,
    etaTraceBracketLower: roots[0]!,
    etaTraceBracketUpper: roots[1]!,
  };
}
