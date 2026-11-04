/**
 * @file labs/mean-square-boundary/compute-lanczos-top-eigenvalue.ts
 * @description Largest eigenvalue of a symmetric linear operator on `n x n` symmetric matrices with the Frobenius inner product
 * (`computeLanczosTopEigenvalue`), by the three-term Lanczos recurrence from the start `v0` (normalised here). Without
 * reorthogonalisation; loss of orthogonality only adds copies of converged Ritz values and does not move the largest one.
 * After each step the largest eigenvalue `theta` of the tridiagonal matrix and the last component `s` of its eigenvector give
 * the residual norm `beta_{k+1} |s|` of the Ritz pair; the iteration stops when it is below `relTol * |theta|` or after
 * `maxIterations`. `w`, `vPrev`, `vCur` are caller buffers (from the executor, so that the operator can hand them to workers). Optionally a
 * per-step callback (step and residual only) and the top Ritz vector, for a warm start of a neighbouring problem.
 * @cold
 */
import { computeSymmetricEigendecompositionHouseholderQl } from './compute-symmetric-eigendecomposition-householder-ql.ts';

function orthogonalizeLanczosStep(
  w: Float64Array,
  vCur: Float64Array,
  vPrev: Float64Array,
  betaPrev: number,
  total: number,
): { readonly alpha: number; readonly beta: number } {
  const total4 = total & ~3;
  let alpha = 0.0;
  let e = 0;
  for (; e < total4; (e = (e + 4) | 0)) {
    alpha =
      alpha +
      w[e]! * vCur[e]! +
      w[(e + 1) | 0]! * vCur[(e + 1) | 0]! +
      w[(e + 2) | 0]! * vCur[(e + 2) | 0]! +
      w[(e + 3) | 0]! * vCur[(e + 3) | 0]!;
  }
  for (; e < total; (e = (e + 1) | 0)) alpha += w[e]! * vCur[e]!;
  let betaSq = 0.0;
  e = 0;
  for (; e < total4; (e = (e + 4) | 0)) {
    const r0 = w[e]! - alpha * vCur[e]! - betaPrev * vPrev[e]!;
    const r1 = w[(e + 1) | 0]! - alpha * vCur[(e + 1) | 0]! - betaPrev * vPrev[(e + 1) | 0]!;
    const r2 = w[(e + 2) | 0]! - alpha * vCur[(e + 2) | 0]! - betaPrev * vPrev[(e + 2) | 0]!;
    const r3 = w[(e + 3) | 0]! - alpha * vCur[(e + 3) | 0]! - betaPrev * vPrev[(e + 3) | 0]!;
    w[e] = r0;
    w[(e + 1) | 0] = r1;
    w[(e + 2) | 0] = r2;
    w[(e + 3) | 0] = r3;
    betaSq = betaSq + r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3;
  }
  for (; e < total; (e = (e + 1) | 0)) {
    const r = w[e]! - alpha * vCur[e]! - betaPrev * vPrev[e]!;
    w[e] = r;
    betaSq += r * r;
  }
  return { alpha, beta: Math.sqrt(betaSq) };
}

export async function computeLanczosTopEigenvalue(
  apply: (x: Float64Array, out: Float64Array) => Promise<void>,
  v0: Float64Array,
  vPrev: Float64Array,
  vCur: Float64Array,
  w: Float64Array,
  maxIterations: number,
  relTol: number,
  options?: {
    /** Called after each step with the step count and the relative residual (not the Ritz value). */
    readonly onStep?: (step: number, relResidual: number) => void;
    /** When given, the Lanczos vectors are kept (in single precision) and the top Ritz vector is written here on return. */
    readonly ritzOut?: Float64Array;
  },
): Promise<{ readonly value: number; readonly iterations: number; readonly converged: boolean; readonly relResidual: number }> {
  const total = v0.length;
  let norm = 0.0;
  for (let e = 0; e < total; (e = (e + 1) | 0)) norm += v0[e]! * v0[e]!;
  const inv0 = 1.0 / Math.sqrt(norm);
  for (let e = 0; e < total; (e = (e + 1) | 0)) {
    vCur[e] = v0[e]! * inv0;
    vPrev[e] = 0.0;
  }
  const alphas: number[] = [];
  const betas: number[] = [];
  let betaPrev = 0.0;
  let theta = 0.0;
  let relResidual = Number.POSITIVE_INFINITY;
  const basis: Float32Array[] = [];
  let lastEigvec: Float64Array | null = null;
  const writeRitz = (): void => {
    const out = options?.ritzOut;
    if (out === undefined || lastEigvec === null) return;
    out.fill(0.0);
    const m = basis.length;
    for (let i = 0; i < m; (i = (i + 1) | 0)) {
      const c = lastEigvec[i]!;
      const v = basis[i]!;
      for (let e = 0; e < total; (e = (e + 1) | 0)) out[e] = out[e]! + c * v[e]!;
    }
    let nr = 0.0;
    for (let e = 0; e < total; (e = (e + 1) | 0)) nr += out[e]! * out[e]!;
    const inv = 1.0 / Math.sqrt(nr);
    for (let e = 0; e < total; (e = (e + 1) | 0)) out[e] = out[e]! * inv;
  };
  for (let k = 0; k < maxIterations; (k = (k + 1) | 0)) {
    if (options?.ritzOut !== undefined) basis.push(Float32Array.from(vCur));
    await apply(vCur, w);
    const { alpha, beta } = orthogonalizeLanczosStep(w, vCur, vPrev, betaPrev, total);
    alphas.push(alpha);
    const m = alphas.length;
    const tri = new Float64Array(Math.imul(m, m));
    for (let i = 0; i < m; (i = (i + 1) | 0)) {
      tri[(Math.imul(i, m) + i) | 0] = alphas[i]!;
      if (i + 1 < m) {
        tri[(Math.imul(i, m) + i + 1) | 0] = betas[i]!;
        tri[(Math.imul(i + 1, m) + i) | 0] = betas[i]!;
      }
    }
    const eig = computeSymmetricEigendecompositionHouseholderQl(tri, m);
    theta = eig.values[(m - 1) | 0]!;
    const lastComponent = eig.qt[(Math.imul((m - 1) | 0, m) + m - 1) | 0]!;
    relResidual = (beta * Math.abs(lastComponent)) / Math.abs(theta);
    if (options?.ritzOut !== undefined) lastEigvec = Float64Array.from(eig.qt.subarray(Math.imul((m - 1) | 0, m), Math.imul(m, m)));
    options?.onStep?.(m, relResidual);
    if (relResidual <= relTol || !(beta > 0.0)) {
      writeRitz();
      return { value: theta, iterations: m, converged: true, relResidual };
    }
    betas.push(beta);
    const invBeta = 1.0 / beta;
    for (let e = 0; e < total; (e = (e + 1) | 0)) {
      vPrev[e] = vCur[e]!;
      vCur[e] = w[e]! * invBeta;
    }
    betaPrev = beta;
  }
  writeRitz();
  return { value: theta, iterations: maxIterations, converged: false, relResidual };
}
