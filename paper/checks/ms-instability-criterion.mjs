// Checks Corollary (mean-square instability criterion): on heavy-ball SGD
//   w_{t+1} = ((1+beta)I - eta H_t) w_t - beta w_{t-1} - eta xi_t
// with i.i.d. minibatch Hessians H_t that do not commute with each other or with H = E H_t,
// and xi_t independent of H_t with Sigma_xi > 0:
//  (a) the exact second-moment recursion Q_{t+1} = T(Q_t) + G stays bounded for rho(T) < 1 and
//      grows at least linearly (<W,Q_t> >= t <W,G>) at rho(T) = 1 and beyond;
//  (b) the Perron eigenvector W >= 0 of the adjoint T* has a non-zero upper-left block;
//  (c) a Monte Carlo run of the recursion matches tr Q_t.
// Run: node paper/checks/ms-instability-criterion.mjs   (exits non-zero on the first failed check)
const fail = (m) => { console.log('FAIL ' + m); process.exit(1); };
const ok = (m) => console.log('ok   ' + m);

const P = 2, n = 2 * P, beta = 0.9;
const rot = (a) => [[Math.cos(a), -Math.sin(a)], [Math.sin(a), Math.cos(a)]];
const mul = (A, B) => A.map((r, i) => B[0].map((_, j) => r.reduce((s, _, k) => s + A[i][k] * B[k][j], 0)));
const tr = (A) => A.reduce((s, r, i) => s + r[i], 0);
const T_ = (A) => A[0].map((_, j) => A.map((r) => r[j]));
// three equiprobable minibatch Hessians R diag(d) R^T with different eigenbases
const Hs = [[0.0, [2.6, 0.2]], [0.7, [0.5, 1.4]], [-1.1, [1.9, 0.1]]].map(([a, d]) => { const R = rot(a); return mul(mul(R, [[d[0], 0], [0, d[1]]]), T_(R)); });
const Hbar = [[0, 0], [0, 0]].map((r, i) => r.map((_, j) => Hs.reduce((s, H) => s + H[i][j], 0) / Hs.length));
const comm = mul(Hs[0], Hs[1]).map((r, i) => r.map((x, j) => x - mul(Hs[1], Hs[0])[i][j]));
if (!(Math.abs(comm[0][1]) > 1e-3)) fail('minibatch Hessians commute');
const Sxi = [[0.3, 0.1], [0.1, 0.2]];
const F = (H, eta) => { const M = Array.from({ length: n }, () => new Array(n).fill(0)); for (let i = 0; i < P; i++) { for (let j = 0; j < P; j++) M[i][j] = (i === j ? 1 + beta : 0) - eta * H[i][j]; M[i][P + i] = -beta; M[P + i][i] = 1; } return M; };
const T = (Q, eta) => { const out = Array.from({ length: n }, () => new Array(n).fill(0)); for (const H of Hs) { const Ft = F(H, eta), R = mul(mul(Ft, Q), T_(Ft)); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i][j] += R[i][j] / Hs.length; } return out; };
const Tadj = (Y, eta) => { const out = Array.from({ length: n }, () => new Array(n).fill(0)); for (const H of Hs) { const Ft = F(H, eta), R = mul(mul(T_(Ft), Y), Ft); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i][j] += R[i][j] / Hs.length; } return out; };
const G = (eta) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i < P && j < P ? eta * eta * Sxi[i][j] : 0)));
const norm = (A) => Math.sqrt(A.flat().reduce((s, x) => s + x * x, 0));
const scale = (A, c) => A.map((r) => r.map((x) => x * c));
function perron(map, eta) { let Y = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))), r = 0; for (let k = 0; k < 4000; k++) { const Z = map(Y, eta); r = norm(Z) / norm(Y); Y = scale(Z, 1 / norm(Z)); } return { r, Y }; }
const rho = (eta) => perron(T, eta).r;
// bisect the mean-square ceiling rho(T(eta)) = 1
let lo = 1e-3, hi = 2 * (1 + beta) / 2.6;
if (!(rho(lo) < 1 && rho(hi) > 1)) fail('bracket');
for (let i = 0; i < 50; i++) { const m = (lo + hi) / 2; if (rho(m) < 1) lo = m; else hi = m; }
const etaStar = (lo + hi) / 2;
ok(`non-commuting minibatch Hessians, |[H_1,H_2]_12| = ${Math.abs(comm[0][1]).toFixed(3)}; mean-square ceiling eta* = ${etaStar.toFixed(6)}, deterministic 2(1+beta)/lambda_1(H) = ${(2 * (1 + beta) / ((tr(Hbar) + Math.sqrt((Hbar[0][0] - Hbar[1][1]) ** 2 + 4 * Hbar[0][1] ** 2)) / 2)).toFixed(6)}`);
for (const f of [0.9, 1.0, 1.02]) {
  const eta = f * etaStar, { r, Y: W } = perron(Tadj, eta);
  // W >= 0 (Krein-Rutman) and W_11 block non-zero
  const W11 = [[W[0][0], W[0][1]], [W[1][0], W[1][1]]], minDiag = Math.min(...W.map((rw, i) => rw[i]));
  if (!(minDiag > -1e-12 && norm(W11) > 1e-6)) fail(`W at f=${f}`);
  const g = tr(mul(W11, scale(Sxi, eta * eta)));
  let Q = Array.from({ length: n }, () => new Array(n).fill(0)), minRatio = Infinity;
  const steps = 3000;
  for (let t = 1; t <= steps; t++) { Q = T(Q, eta).map((rw, i) => rw.map((x, j) => x + G(eta)[i][j])); const wq = W.flat().reduce((s, x, k) => s + x * Q.flat()[k], 0); minRatio = Math.min(minRatio, wq / (t * g)); }
  const m2 = Q[0][0] + Q[1][1];
  if (f < 1 && !(m2 < 1e6)) fail(`bounded below the ceiling: ${m2}`);
  if (f >= 1 && !(minRatio >= 1 - 1e-9)) fail(`<W,Q_t> >= t g fails at f=${f}: ${minRatio}`);
  ok(`eta = ${f.toFixed(2)} eta*: rho(T) = ${r.toFixed(6)}, ||W_11||/||W|| = ${norm(W11).toFixed(3)}, E|w_3000|^2 = ${m2.toExponential(3)}, min_t <W,Q_t>/(t g) = ${minRatio.toFixed(3)}`);
}
// (c) Monte Carlo at 0.5 eta*: tr Q_t against simulated E|w_t|^2 (seeded LCG, Box-Muller)
{
  let s = 12345; const U = () => { s = (Math.imul(1664525, s) + 1013904223) >>> 0; return (s + 0.5) / 4294967296; };
  const gauss = () => Math.sqrt(-2 * Math.log(U())) * Math.cos(2 * Math.PI * U());
  const L = [[Math.sqrt(Sxi[0][0]), 0], [Sxi[0][1] / Math.sqrt(Sxi[0][0]), 0]]; L[1][1] = Math.sqrt(Sxi[1][1] - L[1][0] ** 2);
  const eta = 0.5 * etaStar, t1 = 200, N = 40000; let acc = 0;
  for (let rep = 0; rep < N; rep++) { let w = [0, 0], wp = [0, 0]; for (let t = 0; t < t1; t++) { const H = Hs[Math.floor(U() * 3)], z = [gauss(), gauss()], xi = [L[0][0] * z[0], L[1][0] * z[0] + L[1][1] * z[1]]; const nw = [0, 1].map((i) => (1 + beta) * w[i] - eta * (H[i][0] * w[0] + H[i][1] * w[1]) - beta * wp[i] - eta * xi[i]); wp = w; w = nw; } acc += w[0] * w[0] + w[1] * w[1]; }
  let Q = Array.from({ length: n }, () => new Array(n).fill(0)); for (let t = 0; t < t1; t++) Q = T(Q, eta).map((rw, i) => rw.map((x, j) => x + G(eta)[i][j]));
  const exact = Q[0][0] + Q[1][1], mc = acc / N;
  if (!(Math.abs(mc / exact - 1) < 0.03)) fail(`Monte Carlo ${mc} against exact ${exact}`);
  ok(`Monte Carlo E|w_200|^2 at 0.5 eta*: ${mc.toFixed(5)} (40000 runs) against exact recursion ${exact.toFixed(5)}`);
}
