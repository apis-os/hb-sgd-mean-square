/**
 * @file labs/mean-square-boundary/compute-splitting-witness-certificate.ts
 * @description Certificate of `rho(T)` against 1 from the witness of Theorem splitting (`computeSplittingWitnessCertificate`),
 * for small `P` in the eigenbasis. At `eta < eta_det`: (a) the Krein–Rutman pair `(r, X*)` of `K = (eta / (1 - beta)) L_eta^{-1} V`
 * from the symmetric form `(eta / (1 - beta)) L^{-1/2} V L^{-1/2}` by full Householder QL (`L_eta` acts entry by entry with
 * `s_p - eta (l_a l_b / (1 + beta) - beta (l_a - l_b)^2 / ((1 - beta)(1 - beta^2)))`); (b) `Y* = blkdiag(eta^2 V(X*), 0)`
 * and `W = (I - T_0)^{-1} Y*` in closed form per entry: `W11 = W22 = u`, `W12_jk = (a_j - beta a_k) u / (1 - beta^2)`,
 * `W12_kj = (a_k - beta a_j) u / (1 - beta^2)`, `u = Y_jk / phi_jk`,
 * `phi_jk = 1 - a_j a_k - beta^2 + beta (a_j (a_j - beta a_k) + a_k (a_k - beta a_j)) / (1 - beta^2)`, `a = 1 + beta - eta l`;
 * (c) the identity residuals `|T(W) - W + (1 - r) Y*| / |W|` and `|T^2(W) - W + (1 - r) Z| / |W|`, `Z = Y* + T(Y*)`;
 * (d) the Collatz–Wielandt bracket of `W^{-1/2} T^2(W) W^{-1/2}`, which brackets `rho(T)^2` whatever the accuracy of `X*`, and
 * the Cholesky decisions `W - T^2(W) > 0` (`rho < 1`) and `T^2(W) - W > 0` (`rho > 1`), by `certifySecondMomentRadius`;
 * (e) `s_low = lambda_min(W^{-1/2} T_0(W) W^{-1/2})` and `m_2 = lambda_min(W^{-1/2} Z W^{-1/2})` of parts (4) and (6).
 * Returns `null` when `L_eta` is not positive (`eta` at or above the deterministic threshold, where `rho(T) >= 1` anyway).
 * @cold
 */
import { applyEigenbasisSecondMomentOperator } from './apply-eigenbasis-second-moment-operator.ts';
import { certifySecondMomentRadius } from './certify-second-moment-radius.ts';
import { computeSymmetricEigendecompositionHouseholderQl } from './compute-symmetric-eigendecomposition-householder-ql.ts';

export async function computeSplittingWitnessCertificate(
  vmat: Float64Array,
  n: number,
  nPair: number,
  pairA: Int32Array,
  pairB: Int32Array,
  curvature: Float64Array,
  beta: number,
  eta: number,
): Promise<{
  readonly r: number;
  readonly identityResidual1: number;
  readonly identityResidual2: number;
  readonly certificate: ReturnType<typeof certifySecondMomentRadius>;
  readonly sLow: number | null;
  readonly m2: number | null;
} | null> {
  const nn = Math.imul(n, n);
  const invOnePlusBeta = 1.0 / (1.0 + beta);
  const commutator = beta / ((1.0 - beta) * (1.0 - beta * beta));
  const ell = new Float64Array(nPair);
  for (let p = 0; p < nPair; (p = (p + 1) | 0)) {
    const la = curvature[pairA[p]!]!;
    const lb = curvature[pairB[p]!]!;
    const diff = la - lb;
    ell[p] = la + lb - eta * (la * lb * invOnePlusBeta - commutator * diff * diff);
    if (!(ell[p]! > 0.0)) return null;
  }
  const scale = eta / (1.0 - beta);
  const s = new Float64Array(Math.imul(nPair, nPair));
  for (let p = 0; p < nPair; (p = (p + 1) | 0)) {
    for (let q = 0; q < nPair; (q = (q + 1) | 0)) s[(Math.imul(p, nPair) + q) | 0] = (scale * vmat[(Math.imul(p, nPair) + q) | 0]!) / Math.sqrt(ell[p]! * ell[q]!);
  }
  const eig = computeSymmetricEigendecompositionHouseholderQl(s, nPair);
  const r = eig.values[(nPair - 1) | 0]!;
  const y = eig.qt.subarray(Math.imul((nPair - 1) | 0, nPair), Math.imul(nPair, nPair));
  const xMat = new Float64Array(nn);
  for (let p = 0; p < nPair; (p = (p + 1) | 0)) {
    const a = pairA[p]!;
    const b = pairB[p]!;
    const c = y[p]! / Math.sqrt(ell[p]!);
    const e = a === b ? c : c * Math.SQRT1_2;
    xMat[(Math.imul(a, n) + b) | 0] = e;
    xMat[(Math.imul(b, n) + a) | 0] = e;
  }
  // Y* = blkdiag(eta^2 V(X*), 0), V applied to the orthonormal coordinates of X*.
  const xc = new Float64Array(nPair);
  const vc = new Float64Array(nPair);
  for (let p = 0; p < nPair; (p = (p + 1) | 0)) {
    const a = pairA[p]!;
    const b = pairB[p]!;
    const e = xMat[(Math.imul(a, n) + b) | 0]!;
    xc[p] = a === b ? e : Math.SQRT2 * e;
  }
  for (let p = 0; p < nPair; (p = (p + 1) | 0)) {
    let acc = 0.0;
    for (let q = 0; q < nPair; (q = (q + 1) | 0)) acc += vmat[(Math.imul(p, nPair) + q) | 0]! * xc[q]!;
    vc[p] = acc;
  }
  const y11 = new Float64Array(nn);
  for (let p = 0; p < nPair; (p = (p + 1) | 0)) {
    const a = pairA[p]!;
    const b = pairB[p]!;
    const e = eta * eta * (a === b ? vc[p]! : vc[p]! * Math.SQRT1_2);
    y11[(Math.imul(a, n) + b) | 0] = e;
    y11[(Math.imul(b, n) + a) | 0] = e;
  }
  const w11 = new Float64Array(nn);
  const w12 = new Float64Array(nn);
  const w22 = new Float64Array(nn);
  const b2 = 1.0 - beta * beta;
  for (let j = 0; j < n; (j = (j + 1) | 0)) {
    const aj = 1.0 + beta - eta * curvature[j]!;
    for (let k = 0; k < n; (k = (k + 1) | 0)) {
      const ak = 1.0 + beta - eta * curvature[k]!;
      const phi = 1.0 - aj * ak - beta * beta + (beta * (aj * (aj - beta * ak) + ak * (ak - beta * aj))) / b2;
      const u = y11[(Math.imul(j, n) + k) | 0]! / phi;
      w11[(Math.imul(j, n) + k) | 0] = u;
      w22[(Math.imul(j, n) + k) | 0] = u;
      w12[(Math.imul(j, n) + k) | 0] = ((aj - beta * ak) * u) / b2;
    }
  }
  const zero12 = new Float64Array(nn);
  const xs = new Float64Array(nPair);
  const ys = new Float64Array(nPair);
  const t1: [Float64Array, Float64Array, Float64Array] = [new Float64Array(nn), new Float64Array(nn), new Float64Array(nn)];
  const t2: [Float64Array, Float64Array, Float64Array] = [new Float64Array(nn), new Float64Array(nn), new Float64Array(nn)];
  const ty: [Float64Array, Float64Array, Float64Array] = [new Float64Array(nn), new Float64Array(nn), new Float64Array(nn)];
  const t0w: [Float64Array, Float64Array, Float64Array] = [new Float64Array(nn), new Float64Array(nn), new Float64Array(nn)];
  applyEigenbasisSecondMomentOperator(vmat, n, nPair, pairA, pairB, curvature, beta, eta, w11, w12, w22, t1[0], t1[1], t1[2], xs, ys);
  applyEigenbasisSecondMomentOperator(vmat, n, nPair, pairA, pairB, curvature, beta, eta, t1[0], t1[1], t1[2], t2[0], t2[1], t2[2], xs, ys);
  const zeroY = new Float64Array(nn);
  applyEigenbasisSecondMomentOperator(vmat, n, nPair, pairA, pairB, curvature, beta, eta, y11, zero12, zeroY, ty[0], ty[1], ty[2], xs, ys);
  applyEigenbasisSecondMomentOperator(new Float64Array(Math.imul(nPair, nPair)), n, nPair, pairA, pairB, curvature, beta, eta, w11, w12, w22, t0w[0], t0w[1], t0w[2], xs, ys);
  let wNorm = 0.0;
  let res1 = 0.0;
  let res2 = 0.0;
  for (let e = 0; e < nn; (e = (e + 1) | 0)) {
    wNorm += w11[e]! * w11[e]! + 2.0 * w12[e]! * w12[e]! + w22[e]! * w22[e]!;
    const r11 = t1[0][e]! - w11[e]! + (1.0 - r) * y11[e]!;
    const r12 = t1[1][e]! - w12[e]!;
    const r22 = t1[2][e]! - w22[e]!;
    res1 += r11 * r11 + 2.0 * r12 * r12 + r22 * r22;
    const z11 = y11[e]! + ty[0][e]!;
    const z12 = ty[1][e]!;
    const z22 = ty[2][e]!;
    const d11 = t2[0][e]! - w11[e]! + (1.0 - r) * z11;
    const d12 = t2[1][e]! - w12[e]! + (1.0 - r) * z12;
    const d22 = t2[2][e]! - w22[e]! + (1.0 - r) * z22;
    res2 += d11 * d11 + 2.0 * d12 * d12 + d22 * d22;
  }
  const certificate = certifySecondMomentRadius([w11, w12, w22], t2, n, true);
  const sLowCert = certifySecondMomentRadius([w11, w12, w22], t0w, n, true);
  const z: [Float64Array, Float64Array, Float64Array] = [Float64Array.from(y11, (v, e) => v + ty[0][e]!), Float64Array.from(ty[1]), Float64Array.from(ty[2])];
  const m2Cert = certifySecondMomentRadius([w11, w12, w22], z, n, true);
  return {
    r,
    identityResidual1: Math.sqrt(res1 / wNorm),
    identityResidual2: Math.sqrt(res2 / wNorm),
    certificate,
    sLow: sLowCert.lowerBound,
    m2: m2Cert.lowerBound,
  };
}
