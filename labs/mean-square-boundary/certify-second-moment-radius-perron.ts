/**
 * @file labs/mean-square-boundary/certify-second-moment-radius-perron.ts
 * @description Perron certificate for the spectral radius of the second-moment operator at small `P`
 * (`certifySecondMomentRadiusPerron`; amendment 4). `T` maps the positive semidefinite cone into itself, so its spectral radius
 * is an eigenvalue with a positive semidefinite eigenmatrix (Krein–Rutman), and for real `sigma > rho(T)` it is the eigenvalue
 * nearest to `sigma` (every eigenvalue has real part at most `rho`). The dense matrix of `T`
 * (`assembleSecondMomentOperatorDenseMatrix`) shifted by `sigma` is factorised once (`factorDenseLuPartialPivoting`) and inverse
 * iteration from the identity runs until the iterate stops changing (relative `1e-14`) or `maxIterations`; the iterate, mapped
 * back to blocks and symmetrised, is the candidate `V`. The decision is the Collatz–Wielandt bracket of
 * `V^{-1/2} T(V) V^{-1/2}` with `T(V)` applied exactly (`applyEigenbasisSecondMomentOperator`, `certifySecondMomentRadius` with
 * `bracket = true`); it holds for any positive definite `V`, so the way `V` was obtained does not enter the certificate. When
 * `sigma <= rho(T)` the iteration finds another eigenvalue and the bracket simply fails to decide.
 * @cold
 */
import { applyEigenbasisSecondMomentOperator } from './apply-eigenbasis-second-moment-operator.ts';
import { assembleSecondMomentOperatorDenseMatrix } from './assemble-second-moment-operator-dense-matrix.ts';
import { certifySecondMomentRadius } from './certify-second-moment-radius.ts';
import { factorDenseLuPartialPivoting } from './factor-dense-lu-partial-pivoting.ts';

function coordinatesToBlocks(v: Float64Array, n: number, nPair: number, pairA: Int32Array, pairB: Int32Array): [Float64Array, Float64Array, Float64Array] {
  const nn = Math.imul(n, n);
  const b11 = new Float64Array(nn);
  const b12 = new Float64Array(nn);
  const b22 = new Float64Array(nn);
  for (let p = 0; p < nPair; (p = (p + 1) | 0)) {
    const a = pairA[p]!;
    const b = pairB[p]!;
    b11[(Math.imul(a, n) + b) | 0] = v[p]!;
    b11[(Math.imul(b, n) + a) | 0] = v[p]!;
    b22[(Math.imul(a, n) + b) | 0] = v[(nPair + nn + p) | 0]!;
    b22[(Math.imul(b, n) + a) | 0] = v[(nPair + nn + p) | 0]!;
  }
  for (let e = 0; e < nn; (e = (e + 1) | 0)) b12[e] = v[(nPair + e) | 0]!;
  let tr = 0.0;
  for (let k = 0; k < n; (k = (k + 1) | 0)) tr += b11[(Math.imul(k, n) + k) | 0]! + b22[(Math.imul(k, n) + k) | 0]!;
  if (tr < 0.0) for (let e = 0; e < nn; (e = (e + 1) | 0)) { b11[e] = -b11[e]!; b12[e] = -b12[e]!; b22[e] = -b22[e]!; }
  return [b11, b12, b22];
}

export function certifySecondMomentRadiusPerron(
  vmat: Float64Array,
  n: number,
  nPair: number,
  pairA: Int32Array,
  pairB: Int32Array,
  curvature: Float64Array,
  beta: number,
  eta: number,
  sigma: number,
  maxIterations: number,
): Record<string, unknown> {
  const { m, dim } = assembleSecondMomentOperatorDenseMatrix(vmat, n, nPair, pairA, pairB, curvature, beta, eta);
  for (let i = 0; i < dim; (i = (i + 1) | 0)) m[(Math.imul(i, dim) + i) | 0] = m[(Math.imul(i, dim) + i) | 0]! - sigma;
  const lu = factorDenseLuPartialPivoting(m, dim);
  if (lu === null) return { sigma, dim, factorised: false, iterations: 0, certificate: null };
  const nn = Math.imul(n, n);
  let v = new Float64Array(dim);
  for (let p = 0; p < nPair; (p = (p + 1) | 0)) if (pairA[p] === pairB[p]) { v[p] = 1.0; v[(nPair + nn + p) | 0] = 1.0; }
  let it = 0;
  let change = Number.POSITIVE_INFINITY;
  let estimate = Number.NaN;
  while (it < maxIterations && !(change < 1e-14)) {
    const w = Float64Array.from(v);
    lu.solve(w);
    let wv = 0.0;
    let vv = 0.0;
    let ww = 0.0;
    for (let i = 0; i < dim; (i = (i + 1) | 0)) { wv += w[i]! * v[i]!; vv += v[i]! * v[i]!; ww += w[i]! * w[i]!; }
    estimate = sigma + vv / wv;
    const inv = (wv < 0.0 ? -1.0 : 1.0) / Math.sqrt(ww);
    change = 0.0;
    const vn = Math.sqrt(vv);
    for (let i = 0; i < dim; (i = (i + 1) | 0)) { const x = w[i]! * inv; change = Math.max(change, Math.abs(x - v[i]! / vn)); w[i] = x; }
    v = w;
    it = (it + 1) | 0;
  }
  const q = coordinatesToBlocks(v, n, nPair, pairA, pairB);
  const o: [Float64Array, Float64Array, Float64Array] = [new Float64Array(nn), new Float64Array(nn), new Float64Array(nn)];
  applyEigenbasisSecondMomentOperator(vmat, n, nPair, pairA, pairB, curvature, beta, eta, q[0], q[1], q[2], o[0], o[1], o[2], new Float64Array(nPair), new Float64Array(nPair));
  return { sigma, dim, factorised: true, iterations: it, lastChange: change, eigenvalueEstimate: estimate, certificate: certifySecondMomentRadius(q, o, n, true) };
}
