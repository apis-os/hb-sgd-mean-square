/**
 * @file labs/mean-square-boundary/hessian-ground-truth.ts
 * @description Computes exact top Hessian eigenpair `(lambda_1, u_1)` via central-difference Pearlmutter Hessian-vector product power iteration and empirical critical learning rate `eta_crit^destr` via destructive binary search (`computeMlpGroundTruthStability`).
 * @cold
 */
import type { createTwoLayerMlpDatasetAndOps } from './mlp-network.ts';
import { createXorshift128Plus } from './prng-xorshift128p.ts';

export function computeMlpGroundTruthStability(
  mlp: ReturnType<typeof createTwoLayerMlpDatasetAndOps>,
  theta: Float64Array,
): {
  readonly lambda1: number;
  readonly lambda2: number;
  readonly spectralGapRatio: number;
  readonly topEigenvectorU1: Float64Array;
  readonly etaCritQuadraticSgd: number;
  readonly etaCritQuadraticHb09: number;
  readonly etaCritDestructiveSgd: number;
  readonly etaCritDestructiveHb09: number;
} {
  const p = mlp.numParams;
  const eps = 1e-5;
  const thetaPlus = new Float64Array(p);
  const thetaMinus = new Float64Array(p);
  const gradPlus = new Float64Array(p);
  const gradMinus = new Float64Array(p);

  function applyHessianVec(vec: Float64Array, outHv: Float64Array): void {
    for (let i = 0; i < p; i = (i + 1) | 0) {
      thetaPlus[i] = theta[i]! + eps * vec[i]!;
      thetaMinus[i] = theta[i]! - eps * vec[i]!;
    }
    mlp.computeFullGradient(thetaPlus, gradPlus);
    mlp.computeFullGradient(thetaMinus, gradMinus);
    const inv2Eps = 0.5 / eps;
    for (let i = 0; i < p; i = (i + 1) | 0) {
      outHv[i] = (gradPlus[i]! - gradMinus[i]!) * inv2Eps;
    }
  }

  const prng = createXorshift128Plus(0x67210001, 0x3c6ef372);
  const u1 = new Float64Array(p);
  const hv = new Float64Array(p);
  let norm = 0.0;
  for (let i = 0; i < p; i = (i + 1) | 0) {
    u1[i] = prng.nextGaussian();
    norm += u1[i]! * u1[i]!;
  }
  const invNorm0 = 1.0 / Math.sqrt(Math.max(1e-30, norm));
  for (let i = 0; i < p; i = (i + 1) | 0) {
    u1[i] = u1[i]! * invNorm0;
  }

  let lambda1 = 0.0;
  for (let iter = 0; iter < 80; iter = (iter + 1) | 0) {
    applyHessianVec(u1, hv);
    let rayleigh = 0.0;
    let sq = 0.0;
    for (let i = 0; i < p; i = (i + 1) | 0) {
      rayleigh += u1[i]! * hv[i]!;
      sq += hv[i]! * hv[i]!;
    }
    lambda1 = rayleigh;
    const invLen = 1.0 / Math.sqrt(Math.max(1e-30, sq));
    for (let i = 0; i < p; i = (i + 1) | 0) {
      u1[i] = hv[i]! * invLen;
    }
  }

  // Deflated power iteration for lambda_2
  const u2 = new Float64Array(p);
  for (let i = 0; i < p; i = (i + 1) | 0) {
    u2[i] = prng.nextGaussian();
  }
  let lambda2 = 0.0;
  for (let iter = 0; iter < 60; iter = (iter + 1) | 0) {
    let dot1 = 0.0;
    for (let i = 0; i < p; i = (i + 1) | 0) {
      dot1 += u2[i]! * u1[i]!;
    }
    let n2 = 0.0;
    for (let i = 0; i < p; i = (i + 1) | 0) {
      u2[i] = u2[i]! - dot1 * u1[i]!;
      n2 += u2[i]! * u2[i]!;
    }
    const invN2 = 1.0 / Math.sqrt(Math.max(1e-30, n2));
    for (let i = 0; i < p; i = (i + 1) | 0) {
      u2[i] = u2[i]! * invN2;
    }
    applyHessianVec(u2, hv);
    let dotHv = 0.0;
    for (let i = 0; i < p; i = (i + 1) | 0) {
      dotHv += u1[i]! * hv[i]!;
    }
    let ray2 = 0.0;
    let sq2 = 0.0;
    for (let i = 0; i < p; i = (i + 1) | 0) {
      const defl = hv[i]! - dotHv * u1[i]!;
      ray2 += u2[i]! * defl;
      sq2 += defl * defl;
      u2[i] = defl;
    }
    lambda2 = Math.abs(ray2);
    const invLen2 = 1.0 / Math.sqrt(Math.max(1e-30, sq2));
    for (let i = 0; i < p; i = (i + 1) | 0) {
      u2[i] = u2[i]! * invLen2;
    }
  }

  // Destructive binary search for ground-truth linear/local stability boundary along u_1 on the real MLP
  const wCur = new Float64Array(p);
  const wPrev = new Float64Array(p);
  const gradCur = new Float64Array(p);
  const gradBase = new Float64Array(p);
  mlp.computeFullGradient(theta, gradBase);

  function isDivergentAtEta(eta: number, beta: number): boolean {
    const initAmp = 1e-4;
    for (let i = 0; i < p; i = (i + 1) | 0) {
      wPrev[i] = theta[i]! + initAmp * u1[i]!;
      wCur[i] = theta[i]! + initAmp * u1[i]!;
    }
    let prevProjSq = initAmp * initAmp;
    for (let t = 0; t < 160; t = (t + 1) | 0) {
      mlp.computeFullGradient(wCur, gradCur);
      let proj = 0.0;
      for (let i = 0; i < p; i = (i + 1) | 0) {
        // Local restoring force relative to theta equilibrium + momentum
        const gRel = gradCur[i]! - gradBase[i]!;
        const dx = -eta * gRel + beta * (wCur[i]! - wPrev[i]!);
        const wNext = wCur[i]! + dx;
        wPrev[i] = wCur[i]!;
        wCur[i] = wNext;
        proj += (wCur[i]! - theta[i]!) * u1[i]!;
      }
      if (!Number.isFinite(proj) || Math.abs(proj) > 50.0 * initAmp) {
        return true;
      }
      prevProjSq = proj * proj;
    }
    return prevProjSq > initAmp * initAmp * 1.5;
  }

  function binarySearchCriticalEta(beta: number, guess: number): number {
    let lo = guess * 0.25;
    let hi = guess * 2.0;
    for (let step = 0; step < 28; step = (step + 1) | 0) {
      const mid = 0.5 * (lo + hi);
      if (isDivergentAtEta(mid, beta)) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    return 0.5 * (lo + hi);
  }

  const etaCritQuadraticSgd = 2.0 / lambda1;
  const etaCritQuadraticHb09 = (2.0 * 1.9) / lambda1;
  const etaCritDestructiveSgd = binarySearchCriticalEta(0.0, etaCritQuadraticSgd);
  const etaCritDestructiveHb09 = binarySearchCriticalEta(0.9, etaCritQuadraticHb09);

  return {
    lambda1,
    lambda2,
    spectralGapRatio: lambda1 / Math.max(1e-12, lambda2),
    topEigenvectorU1: u1,
    etaCritQuadraticSgd,
    etaCritQuadraticHb09,
    etaCritDestructiveSgd,
    etaCritDestructiveHb09,
  };
}
