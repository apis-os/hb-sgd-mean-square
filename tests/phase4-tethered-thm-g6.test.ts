/**
 * @file tests/phase4-tethered-thm-g6.test.ts
 * @description Phase 4 of lab L67: the Householder-QL eigendecomposition against cyclic Jacobi and its own reconstruction;
 * the tiled row product against a naive triple loop and against a different row split (bit for bit); `eta*` of Theorem G.6 by
 * Lanczos on `applyThmG6RayleighOperator` against an independent dense construction of `L_0^{-1/2} B_beta L_0^{-1/2}` on an
 * orthonormal basis of `S^P` from explicit matrices; the exact second-moment operator crossing 1 at that `eta*`; and the
 * linearised and network sample-path modes agreeing on a quadratic, whose gradient is the linear system; and a launch resumed
 * from the progress-log lines of an earlier one giving the same cells bit for bit.
 * @cold
 */
import { describe, expect, it } from 'vitest';
import { applyNetworkHeavyBallSecondMomentOperator } from '../labs/mean-square-boundary/apply-network-heavy-ball-second-moment-operator.ts';
import { applyThmG6RayleighOperator } from '../labs/mean-square-boundary/apply-thm-g6-rayleigh-operator.ts';
import { certifySecondMomentRadius } from '../labs/mean-square-boundary/certify-second-moment-radius.ts';
import { compileDenseF64MatmulKernel } from '../labs/mean-square-boundary/compile-dense-f64-matmul-kernel.ts';
import { computeLanczosTopEigenvalue } from '../labs/mean-square-boundary/compute-lanczos-top-eigenvalue.ts';
import { computeSymmetricEigendecompositionHouseholderQl } from '../labs/mean-square-boundary/compute-symmetric-eigendecomposition-householder-ql.ts';
import { computeSymmetricEigenvaluesJacobi } from '../labs/mean-square-boundary/compute-symmetric-eigenvalues-jacobi.ts';
import { createPhase4RowExecutor } from '../labs/mean-square-boundary/create-phase4-row-executor.ts';
import { createPhase4SampleHessianProvider } from '../labs/mean-square-boundary/create-phase4-sample-hessian-provider.ts';
import { evaluatePhase4NetworkRow } from '../labs/mean-square-boundary/evaluate-phase4-network-row.ts';
import { measureNetworkSecondMomentSpectralRadius } from '../labs/mean-square-boundary/measure-network-second-moment-spectral-radius.ts';
import { multiplyDenseRows } from '../labs/mean-square-boundary/multiply-dense-rows.ts';
import { parsePhase4CellLogLine } from '../labs/mean-square-boundary/parse-phase4-cell-log-line.ts';
import type { Phase4NetworkSpec } from '../labs/mean-square-boundary/phase4-types.ts';
import { createXorshift128Plus } from '../labs/mean-square-boundary/prng-xorshift128p.ts';
import { simulateTetheredHeavyBallEnergy } from '../labs/mean-square-boundary/simulate-tethered-heavy-ball-energy.ts';

function randomSymmetric(n: number, seed: number): Float64Array {
  const prng = createXorshift128Plus(seed, 0x2545f491);
  const a = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) {
    const v = prng.nextGaussian();
    a[i * n + j] = v;
    a[j * n + i] = v;
  }
  return a;
}

function quadraticSpec(p: number, nS: number, seed: number): Phase4NetworkSpec & { readonly kind: 'quadratic' } {
  const prng = createXorshift128Plus(seed, 678);
  const hs = new Float64Array(nS * p * p);
  for (let i = 0; i < nS; i++) {
    const g = Array.from({ length: 2 * p }, () => prng.nextGaussian());
    for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) hs[i * p * p + a * p + b] = g[a]! * g[b]! + g[p + a]! * g[p + b]! + 0.3 * prng.nextGaussian();
    for (let a = 0; a < p; a++) for (let b = a + 1; b < p; b++) {
      const v = 0.5 * (hs[i * p * p + a * p + b]! + hs[i * p * p + b * p + a]!);
      hs[i * p * p + a * p + b] = v;
      hs[i * p * p + b * p + a] = v;
    }
  }
  return { kind: 'quadratic', numParams: p, numSamples: nS, hessians: hs };
}

function matmul(a: Float64Array, b: Float64Array, n: number): Float64Array {
  const c = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) c[i * n + j] = c[i * n + j]! + a[i * n + k]! * b[k * n + j]!;
  return c;
}

// eta* of Theorem G.6(3) from explicit matrices: the operator A = L0^{-1/2} B L0^{-1/2} on the orthonormal basis
// {E_kk} U {(E_jk + E_kj)/sqrt 2} of S^P in the eigenbasis of H + mu I, its largest eigenvalue by Jacobi.
function denseThmG6Eta(spec: ReturnType<typeof quadraticSpec>, beta: number, batchB: number, mu: number): number {
  const p = spec.numParams;
  const nS = spec.numSamples;
  const pp = p * p;
  const hMean = new Float64Array(pp);
  for (let i = 0; i < nS; i++) for (let e = 0; e < pp; e++) hMean[e] = hMean[e]! + spec.hessians[i * pp + e]! / nS;
  const eig = computeSymmetricEigendecompositionHouseholderQl(hMean, p);
  const lam = Array.from(eig.values, (v) => v + mu);
  const qt = eig.qt;
  const q = new Float64Array(pp);
  for (let a = 0; a < p; a++) for (let k = 0; k < p; k++) q[a * p + k] = qt[k * p + a]!;
  const rotIn = (x: Float64Array): Float64Array => matmul(matmul(q, x, p), qt, p);
  const rotOut = (y: Float64Array): Float64Array => matmul(matmul(qt, y, p), q, p);
  const dH = Array.from({ length: nS }, (_, i) => {
    const m = new Float64Array(pp);
    for (let e = 0; e < pp; e++) m[e] = spec.hessians[i * pp + e]! - hMean[e]!;
    return m;
  });
  const basis: Float64Array[] = [];
  for (let j = 0; j < p; j++) for (let k = j; k < p; k++) {
    const e = new Float64Array(pp);
    if (j === k) e[j * p + j] = 1;
    else {
      e[j * p + k] = Math.SQRT1_2;
      e[k * p + j] = Math.SQRT1_2;
    }
    basis.push(e);
  }
  const applyA = (x: Float64Array): Float64Array => {
    const y = new Float64Array(pp);
    for (let j = 0; j < p; j++) for (let k = 0; k < p; k++) y[j * p + k] = x[j * p + k]! / Math.sqrt(lam[j]! + lam[k]!);
    const yOrig = rotIn(y);
    const v = new Float64Array(pp);
    for (const d of dH) {
      const t = matmul(matmul(d, yOrig, p), d, p);
      for (let e = 0; e < pp; e++) v[e] = v[e]! + t[e]! / (nS * batchB);
    }
    const vEig = rotOut(v);
    const out = new Float64Array(pp);
    for (let j = 0; j < p; j++) for (let k = 0; k < p; k++) {
      const diff = lam[j]! - lam[k]!;
      const mult = (lam[j]! * lam[k]!) / (1 + beta) - (beta / ((1 - beta) * (1 - beta * beta))) * diff * diff;
      out[j * p + k] = (mult * y[j * p + k]! + vEig[j * p + k]! / (1 - beta)) / Math.sqrt(lam[j]! + lam[k]!);
    }
    return out;
  };
  const dim = basis.length;
  const mat = new Float64Array(dim * dim);
  for (let c = 0; c < dim; c++) {
    const col = applyA(basis[c]!);
    for (let r = 0; r < dim; r++) {
      let s = 0;
      for (let e = 0; e < pp; e++) s += basis[r]![e]! * col[e]!;
      mat[r * dim + c] = s;
    }
  }
  for (let r = 0; r < dim; r++) for (let c = r + 1; c < dim; c++) {
    const v = 0.5 * (mat[r * dim + c]! + mat[c * dim + r]!);
    mat[r * dim + c] = v;
    mat[c * dim + r] = v;
  }
  return 1 / computeSymmetricEigenvaluesJacobi(mat, dim)[0]!;
}

describe('Phase 4 — Theorem G.6 on tethered per-sample Hessians', () => {
  it('Householder-QL eigendecomposition reconstructs the matrix and matches cyclic Jacobi', () => {
    const n = 30;
    const a = randomSymmetric(n, 0x13000001);
    const { values, qt } = computeSymmetricEigendecompositionHouseholderQl(a, n);
    const jac = computeSymmetricEigenvaluesJacobi(a, n);
    for (let k = 0; k < n; k++) expect(Math.abs(values[k]! - jac[n - 1 - k]!)).toBeLessThan(1e-11);
    for (let k = 1; k < n; k++) expect(values[k]!).toBeGreaterThanOrEqual(values[k - 1]!);
    let maxErr = 0;
    let maxOrth = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      let s = 0;
      let o = 0;
      for (let k = 0; k < n; k++) {
        s += qt[k * n + i]! * values[k]! * qt[k * n + j]!;
        o += qt[i * n + k]! * qt[j * n + k]!;
      }
      maxErr = Math.max(maxErr, Math.abs(s - a[i * n + j]!));
      maxOrth = Math.max(maxOrth, Math.abs(o - (i === j ? 1 : 0)));
    }
    expect(maxErr).toBeLessThan(1e-12);
    expect(maxOrth).toBeLessThan(1e-13);
  });

  it('tiled row product equals the naive product and does not depend on the row split', () => {
    const n = 137;
    const a = randomSymmetric(n, 0x13000002);
    const b = randomSymmetric(n, 0x13000003);
    const naive = matmul(a, b, n);
    const whole = new Float64Array(n * n);
    multiplyDenseRows(a, b, whole, n, 0, n);
    const split = new Float64Array(n * n);
    multiplyDenseRows(a, b, split, n, 0, 41);
    multiplyDenseRows(a, b, split, n, 41, 100);
    multiplyDenseRows(a, b, split, n, 100, n);
    for (let e = 0; e < n * n; e++) {
      expect(Math.abs(whole[e]! - naive[e]!)).toBeLessThan(1e-11);
      expect(split[e]).toBe(whole[e]);
    }
  });

  it('Lanczos eta* equals the dense construction, and the exact second-moment operator crosses 1 there', async () => {
    const spec = quadraticSpec(4, 6, 12345);
    const batchB = 2;
    const exec = await createPhase4RowExecutor(spec, 0);
    const n = exec.n;
    const hm = new Float64Array(n * n);
    const eye = new Float64Array(n * n);
    for (let j = 0; j < n; j++) eye[j * n + j] = 1;
    await exec.hessianRows(-1, eye, hm, 1, false);
    const eig = computeSymmetricEigendecompositionHouseholderQl(hm, n);
    const qt = new Float64Array(eig.qt);
    const q = new Float64Array(n * n);
    await exec.transpose(qt, q);
    const s = [exec.alloc(), exec.alloc(), exec.alloc(), exec.alloc(), exec.alloc(), exec.alloc()] as const;
    for (const [beta, mu] of [[0, 0.5], [0.9, 2], [0.9, 0.0078125]] as const) {
      const lambdaT = Float64Array.from(eig.values, (v) => v + mu);
      const x0 = new Float64Array(n * n);
      for (let j = 0; j < n; j++) x0[j * n + j] = 1;
      const lz = await computeLanczosTopEigenvalue(
        (x, out) => applyThmG6RayleighOperator(exec, lambdaT, q, qt, beta, batchB, x, out, s),
        x0, exec.alloc(), exec.alloc(), exec.alloc(), 100, 1e-12,
      );
      expect(lz.converged).toBe(true);
      const eta = 1 / lz.value;
      expect(Math.abs(eta / denseThmG6Eta(spec, beta, batchB, mu) - 1)).toBeLessThan(1e-10);
      const q11 = new Float64Array(n * n);
      const q12 = new Float64Array(n * n);
      const q22 = new Float64Array(n * n);
      const inv = 1 / Math.sqrt(2 * n);
      for (let j = 0; j < n; j++) {
        q11[j * n + j] = inv;
        q22[j * n + j] = inv;
      }
      const o = [exec.alloc(), exec.alloc(), exec.alloc()] as const;
      const below = await measureNetworkSecondMomentSpectralRadius(exec, beta, (1 - 1e-3) * eta, mu, batchB, q11, q12, q22, o, s, 100000, 1e-13);
      const above = await measureNetworkSecondMomentSpectralRadius(exec, beta, (1 + 1e-3) * eta, mu, batchB, q11, q12, q22, o, s, 100000, 1e-13);
      expect(below.converged && above.converged).toBe(true);
      expect(below.radius).toBeLessThan(1);
      expect(above.radius).toBeGreaterThan(1);
      // Collatz-Wielandt: at the last iterate of the upper probe the bracket contains its radius and is above 1.
      await applyNetworkHeavyBallSecondMomentOperator(exec, beta, (1 + 1e-3) * eta, mu, batchB, q11, q12, q22, o[0], o[1], o[2], s);
      const cert = certifySecondMomentRadius([q11, q12, q22], o, n, true);
      const signs = certifySecondMomentRadius([q11, q12, q22], o, n, false);
      expect(cert.lowerBound!).toBeLessThanOrEqual(above.radius * (1 + 1e-9));
      expect(cert.upperBound!).toBeGreaterThanOrEqual(above.radius * (1 - 1e-9));
      expect(cert.certifiedAboveOne).toBe(true);
      expect(signs.certifiedAboveOne).toBe(true);
      expect(signs.certifiedBelowOne).toBe(false);
    }
    await exec.close();
  });

  it('the fraction f* = eta* (lambda_1 + mu) / (2 (1 + beta)) is non-decreasing in mu while eta* need not be', async () => {
    const spec = quadraticSpec(4, 6, 12345);
    const exec = await createPhase4RowExecutor(spec, 0);
    const n = exec.n;
    const hm = new Float64Array(n * n);
    const eye = new Float64Array(n * n);
    for (let j = 0; j < n; j++) eye[j * n + j] = 1;
    await exec.hessianRows(-1, eye, hm, 1, false);
    const eig = computeSymmetricEigendecompositionHouseholderQl(hm, n);
    const qt = new Float64Array(eig.qt);
    const q = new Float64Array(n * n);
    await exec.transpose(qt, q);
    const s = [exec.alloc(), exec.alloc(), exec.alloc(), exec.alloc(), exec.alloc(), exec.alloc()] as const;
    const lambda1 = eig.values[n - 1]!;
    for (const beta of [0, 0.9]) {
      const fs: number[] = [];
      const etas: number[] = [];
      for (const mu of [0.0078125, 0.125, 0.5, 2, 8, 32, 128]) {
        const lambdaT = Float64Array.from(eig.values, (v) => v + mu);
        const x0 = new Float64Array(n * n);
        for (let j = 0; j < n; j++) x0[j * n + j] = 1;
        const lz = await computeLanczosTopEigenvalue(
          (x, out) => applyThmG6RayleighOperator(exec, lambdaT, q, qt, beta, 2, x, out, s),
          x0, exec.alloc(), exec.alloc(), exec.alloc(), 100, 1e-12,
        );
        etas.push(1 / lz.value);
        fs.push((lambda1 + mu) / (lz.value * 2 * (1 + beta)));
      }
      for (let i = 1; i < fs.length; i++) expect(fs[i]!).toBeGreaterThanOrEqual(fs[i - 1]!);
      expect(fs[fs.length - 1]!).toBeLessThan(1);
      if (beta === 0) expect(etas[3]!).toBeLessThan(etas[2]!);
    }
    await exec.close();
  });

  it('Wasm f64x2 product equals the sequential-k product bit for bit, also on an upper triangle with a row offset', () => {
    // n = 45 fits one block; n = 523 crosses the k blocks at 256 and 512 and the column block at 512, and r0 = 300 starts the
    // upper triangle inside a block, so the accumulator carried from one k block to the next is exercised.
    for (const [n, cases] of [[45, [[0, 45, 0], [7, 13, 0], [9, 22, 1]]], [523, [[0, 523, 0], [7, 131, 0], [300, 101, 1]]]] as const) {
    const a = randomSymmetric(n, 0x13000011 + n);
    const b = randomSymmetric(n, 0x13000012 + n);
    const seq = (i: number, j: number): number => {
      let c = 0;
      for (let k = 0; k < n; k++) c = c + a[i * n + k]! * b[k * n + j]!;
      return c;
    };
    const memory = new WebAssembly.Memory({ initial: Math.ceil(((4 * n * n + 256 * n + 131072) * 8) / 65536) + 1 });
    const inst = new WebAssembly.Instance(new WebAssembly.Module(compileDenseF64MatmulKernel()), { env: { memory } });
    const fn = Object.values(inst.exports).find((v) => typeof v === 'function') as (...x: number[]) => number;
    const f = new Float64Array(memory.buffer);
    for (const [r0, rows, upper] of cases) {
      f.set(b, 0);
      f.set(a.subarray(r0 * n, (r0 + rows) * n), n * n);
      fn(n * n * 8, 0, 2 * n * n * 8, n, rows, r0, upper, 3 * n * n * 8, (3 * n * n + 131072) * 8);
      let mismatches = 0;
      for (let i = 0; i < rows; i++) for (let j = 0; j < n; j++) {
        if (upper === 1 && j < r0 + i) continue;
        if (f[2 * n * n + i * n + j] !== seq(r0 + i, j)) mismatches++;
      }
      expect(mismatches).toBe(0);
    }
    }
  });

  it('worker products give the same bits for 2 and 3 workers, and matmulSymmetric is exactly symmetric', async () => {
    const spec = quadraticSpec(37, 3, 4242);
    const outs: Float64Array[][] = [];
    for (const workers of [2, 3]) {
      const exec = await createPhase4RowExecutor(spec, workers);
      const n = exec.n;
      const a = exec.alloc();
      const b = exec.alloc();
      a.set(randomSymmetric(n, 0x13000021));
      b.set(randomSymmetric(n, 0x13000022));
      const c = exec.alloc();
      const cs = exec.alloc();
      await exec.matmul(a, b, c);
      await exec.matmulSymmetric(a, a, cs);
      // Against the sequential-k products: the full product, and the upper triangle of A A mirrored (what the lower part must be).
      const seq = (x: Float64Array, y: Float64Array, i: number, j: number): number => {
        let t = 0;
        for (let k = 0; k < n; k++) t = t + x[i * n + k]! * y[k * n + j]!;
        return t;
      };
      for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) {
        expect(c[j * n + k]).toBe(seq(a, b, j, k));
        expect(cs[j * n + k]).toBe(j <= k ? seq(a, a, j, k) : seq(a, a, k, j));
      }
      outs.push([Float64Array.from(c), Float64Array.from(cs)]);
      await exec.close();
    }
    for (let e = 0; e < outs[0]![0]!.length; e++) {
      expect(outs[1]![0]![e]).toBe(outs[0]![0]![e]);
      expect(outs[1]![1]![e]).toBe(outs[0]![1]![e]);
    }
  });

  it('a Lanczos warm start from the previous Ritz vector gives the same eta* in fewer steps', async () => {
    const spec = quadraticSpec(4, 6, 12345);
    const exec = await createPhase4RowExecutor(spec, 0);
    const n = exec.n;
    const hm = new Float64Array(n * n);
    const eye = new Float64Array(n * n);
    for (let j = 0; j < n; j++) eye[j * n + j] = 1;
    await exec.hessianRows(-1, eye, hm, 1, false);
    const eig = computeSymmetricEigendecompositionHouseholderQl(hm, n);
    const qt = new Float64Array(eig.qt);
    const q = new Float64Array(n * n);
    await exec.transpose(qt, q);
    const s = [exec.alloc(), exec.alloc(), exec.alloc(), exec.alloc(), exec.alloc(), exec.alloc()] as const;
    const run = async (mu: number, start: Float64Array, ritz?: Float64Array): Promise<{ eta: number; steps: number }> => {
      const lambdaT = Float64Array.from(eig.values, (v) => v + mu);
      const lz = await computeLanczosTopEigenvalue(
        (x, out) => applyThmG6RayleighOperator(exec, lambdaT, q, qt, 0.9, 2, x, out, s),
        start, exec.alloc(), exec.alloc(), exec.alloc(), 100, 1e-12, ritz === undefined ? undefined : { ritzOut: ritz },
      );
      return { eta: 1 / lz.value, steps: lz.iterations };
    };
    const id = new Float64Array(n * n);
    for (let j = 0; j < n; j++) id[j * n + j] = 1;
    const ritz = new Float64Array(n * n);
    await run(2, id, ritz);
    const cold = await run(1.5, id);
    const warm = await run(1.5, ritz);
    expect(Math.abs(warm.eta / cold.eta - 1)).toBeLessThan(1e-10);
    expect(warm.steps).toBeLessThanOrEqual(cold.steps);
    await exec.close();
  });

  it('linearised and network sample paths agree on a quadratic, whose gradient is the linear system', () => {
    const provider = createPhase4SampleHessianProvider(quadraticSpec(4, 6, 777));
    const lin = simulateTetheredHeavyBallEnergy(provider, 0, 0.5, 0.9, 0.04, 2, 4, 400, 1e-6, 2.5e-5, 11, 13);
    const net = simulateTetheredHeavyBallEnergy(provider, 1, 0.5, 0.9, 0.04, 2, 4, 400, 1e-6, 2.5e-5, 11, 13);
    expect(Math.abs(lin - net)).toBeLessThan(1e-8);
  });
  it('a row resumed from the logged cells of an earlier run gives the same cells, and a mismatched record is refused', async () => {
    const spec = quadraticSpec(4, 6, 12345);
    const cfg = {
      batchB: 2, matrixWorkers: 0, lanczosMaxIterations: 60, lanczosRelTol: 1e-12, operatorMaxIterations: 10, operatorRelTol: 1e-12,
      relOffset: 1e-3, fallbackOffset: 1e-2, classifyCriticalMode: true, certificateBracket: true, horizonMultipliers: [],
      trajectorySeeds: 0, trajectoryWorkers: 1, streams: 1, steps: 10, bisectionIterations: 1, logGrowthThreshold: 0.9162907318741551,
      startNorm: 1e-6, divergenceEnergy: 2.5e-5, seedTag: 1,
    };
    const lines: string[] = [];
    const stamp = (l: string): void => { lines.push(`2026-10-28T00:00:00.000Z ${l}`); };
    const first = await evaluatePhase4NetworkRow('Q4', spec, 'test', [0.9], [2, 0.5], [], cfg, stamp);
    const parsed = lines.map(parsePhase4CellLogLine).filter((c) => c !== null);
    const fromText = lines.filter((l) => !l.includes(' CELL ')).map(parsePhase4CellLogLine).filter((c) => c !== null);
    const fromJson = lines.filter((l) => l.includes(' CELL ')).map(parsePhase4CellLogLine).filter((c) => c !== null);
    expect(parsed.length).toBe(4);
    expect(fromText).toEqual(fromJson);
    const resumeCells = fromText.filter((c) => c.tetherGrid === 2);
    const second = await evaluatePhase4NetworkRow('Q4', spec, 'test', [0.9], [2, 0.5], [], { ...cfg, resumeCells }, () => {});
    // Every number round-trips through its printed form except the sign of a zero (`String(-0)` is "0"), which only a share of
    // the critical mode can carry; that record is compared as printed text.
    expect(second.thresholds.map(({ resumedFromLog: _r, criticalMode: _m, ...t }) => t)).toEqual(first.thresholds.map(({ criticalMode: _m, ...t }) => t));
    expect(second.thresholds.map((t) => JSON.stringify(t.criticalMode))).toEqual(first.thresholds.map((t) => JSON.stringify(t.criticalMode)));
    expect(second.thresholds[0]!.resumedFromLog).toBe(true);
    expect(second.thresholds[1]!.resumedFromLog).toBeUndefined();
    const bad = resumeCells.map((c) => ({ ...c, etaOneDim: c.etaOneDim * (1 + 1e-15) }));
    await expect(evaluatePhase4NetworkRow('Q4', spec, 'test', [0.9], [2, 0.5], [], { ...cfg, resumeCells: bad }, () => {})).rejects.toThrow(/differs from the recorded/);
  });
});
