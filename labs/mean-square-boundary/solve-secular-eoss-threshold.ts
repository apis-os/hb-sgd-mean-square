/**
 * @file labs/mean-square-boundary/solve-secular-eoss-threshold.ts
 * @description Root of the rank-one secular equation of Theorem G (`solveSecularEossThreshold`): the mean-square stability ceiling of heavy-ball SGD on a quadratic with random-feature minibatches, with the sandwich bounds, the Lee–Cheng–Paquette–Paquette limit (no kurtosis term) and the one-dimensional ceiling of Theorem E(3) along `u_1`, all for a without-replacement factor `a = 1 - zeta`.
 * @cold
 */

export function solveSecularEossThreshold(
  lambdas: Float64Array,
  beta: number,
  batchB: number,
  kappaFeat: number,
  zeta = 0.0,
): {
  readonly etaCritSecular: number;
  readonly etaCritSecularLeeLimit: number;
  readonly etaCritSandwichLower: number;
  readonly etaCritSandwichUpper: number;
  readonly etaCritOneDim: number;
  readonly etaCritDeterministic: number;
  readonly muStar: number;
  readonly traceH: number;
  readonly lambda1: number;
} {
  const p = lambdas.length | 0;
  let lambda1 = 0.0;
  let traceH = 0.0;
  for (let k = 0; k < p; (k = (k + 1) | 0)) {
    const l = lambdas[k]!;
    traceH += l;
    if (l > lambda1) lambda1 = l;
  }
  const a = 1.0 - zeta;
  const kappaBeta = (1.0 + beta) / (1.0 - beta);
  const twoOnePlusBeta = 2.0 * (1.0 + beta);
  const alphaDiag = 1.0 + (kappaBeta * (1.0 + kappaFeat) * a) / batchB;
  const etaCritDeterministic = twoOnePlusBeta / lambda1;

  // Secular equation in eta: (eta a / (2 (1 - beta) B)) sum_k lambda_k / (1 - eta lambda_k alpha / (2 (1 + beta))) = 1,
  // unique root in (0, 2 (1 + beta) / (lambda_1 max(alpha, 1))); the left side increases on that interval (alpha may be
  // negative for kappaFeat < -1, in which case there is no pole and the root is below the deterministic ceiling).
  const solveRoot = (alpha: number): number => {
    const etaPole = twoOnePlusBeta / (Math.max(alpha, 1.0) * lambda1);
    let lo = 0.0;
    let hi = etaPole;
    for (let it = 0; it < 200; (it = (it + 1) | 0)) {
      const mid = 0.5 * (lo + hi);
      let sum = 0.0;
      for (let k = 0; k < p; (k = (k + 1) | 0)) {
        const l = lambdas[k]!;
        const denom = 1.0 - (mid * l * alpha) / twoOnePlusBeta;
        sum += l / Math.max(1e-300, denom);
      }
      const lhs = (mid * a * sum) / (2.0 * (1.0 - beta) * batchB);
      if (lhs < 1.0) lo = mid;
      else hi = mid;
      if (hi - lo <= 1e-15 * etaPole) break;
    }
    return 0.5 * (lo + hi);
  };

  const etaCritSecular = solveRoot(alphaDiag);
  const etaCritSecularLeeLimit = solveRoot(1.0);
  const muStar = twoOnePlusBeta / etaCritSecular;

  // Sandwich of Theorem G(3), valid for either sign of alphaDiag:
  //   max(alpha lambda_1 + kb lambda_1 / B, min(alpha, 0) lambda_1 + kb tr(H) / B) <= mu* <= max(alpha, 0) lambda_1 + kb tr(H) / B
  const kbTraceOverB = (kappaBeta * a * traceH) / batchB;
  const sandwichUpperMu = Math.max(alphaDiag, 0.0) * lambda1 + kbTraceOverB;
  const sandwichLowerMu = Math.max(
    alphaDiag * lambda1 + (kappaBeta * a * lambda1) / batchB,
    Math.min(alphaDiag, 0.0) * lambda1 + kbTraceOverB,
  );
  const etaCritSandwichLower = twoOnePlusBeta / sandwichUpperMu;
  const etaCritSandwichUpper = twoOnePlusBeta / sandwichLowerMu;

  const nu11Sq = (lambda1 * lambda1 * (2.0 + kappaFeat) * a) / batchB;
  const etaCritOneDim = (twoOnePlusBeta * lambda1) / (lambda1 * lambda1 + kappaBeta * nu11Sq);

  return {
    etaCritSecular,
    etaCritSecularLeeLimit,
    etaCritSandwichLower,
    etaCritSandwichUpper,
    etaCritOneDim,
    etaCritDeterministic,
    muStar,
    traceH,
    lambda1,
  };
}
