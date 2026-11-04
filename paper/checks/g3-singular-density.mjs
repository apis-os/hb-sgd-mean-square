// Checks the unbounded-noise form of Proposition G3(1) at beta = 0 (lambda_1 = 1):
// with u = 2 delta + 2 (1 + delta) zeta and M_sing the supremum of the density of 1 + u on [-1/2, 1/2],
//   | E log|1+u| - (E u - E u^2 / 2) | <= 11 E|u|^3 + (1 + log 2) M_sing,
// for zeta ~ N(0, 2/B) (Gaussian zeta) and zeta = chi2_B / B - 1 (Gaussian features), |delta| <= 1/8,
// and the explicit envelopes of M_sing used in the text:
//   Gaussian: M_sing <= (2 pi (49/16) eps)^(-1/2) exp(-1/(162 eps)), eps = Var(zeta) = 2/B;
//   chi2:     density of w = chi2_B/B on [2/9, 6/7] <= sqrt(B / (2 pi w^2)) exp(-(B/2)(w - 1 - log w)).
// Run: node paper/checks/g3-singular-density.mjs   (exits non-zero on the first failed check)
const fail = (m) => { console.log('FAIL ' + m); process.exit(1); };
const ok = (m) => console.log('ok   ' + m);
function lgamma(z) {
  const C = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.138571095836524, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * z))) - lgamma(1 - z);
  z -= 1; let x = C[0]; for (let i = 1; i < 9; i++) x += C[i] / (z + i);
  const t = z + 7.5; return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}
// integrate g(zeta) against a density on [lo, hi] by the midpoint rule
function integrate(dens, g, lo, hi, n) { const h = (hi - lo) / n; let s = 0; for (let i = 0; i < n; i++) { const z = lo + (i + 0.5) * h; s += dens(z) * g(z) * h; } return s; }
const laws = (B) => {
  const v = 2 / B, sd = Math.sqrt(v);
  const gauss = { name: 'Gaussian zeta', dens: (z) => Math.exp(-z * z / (2 * v)) / Math.sqrt(2 * Math.PI * v), lo: -14 * sd, hi: 14 * sd };
  const k = B / 2, lc = -lgamma(k) - k * Math.log(2);
  // zeta = w - 1, w = chi2_B / B, density of w: B * f_chi2(B w)
  const chi = { name: 'chi2 features', dens: (z) => { const w = z + 1; if (w <= 0) return 0; const x = B * w; return B * Math.exp(lc + (k - 1) * Math.log(x) - x / 2); }, lo: -1, hi: 14 * sd + 2 };
  return [gauss, chi];
};
let worst = 0, n = 0;
for (const B of [16, 64, 256, 1024, 4096]) {
  for (const law of laws(B)) {
    for (const delta of [-0.125, 0, 0.125]) {
      const u = (z) => 2 * delta + 2 * (1 + delta) * z, N = 4000000;
      const L = integrate(law.dens, (z) => Math.log(Math.abs(1 + u(z))), law.lo, law.hi, N);
      const m1 = integrate(law.dens, (z) => u(z), law.lo, law.hi, N);
      const m2 = integrate(law.dens, (z) => u(z) ** 2, law.lo, law.hi, N);
      const m3 = integrate(law.dens, (z) => Math.abs(u(z)) ** 3, law.lo, law.hi, N);
      // density of v = 1 + u at v is dens(zeta) / (2 (1 + delta)) with zeta = (v - 1 - 2 delta) / (2 (1 + delta))
      let Ms = 0; for (let i = 0; i <= 2000; i++) { const vv = -0.5 + i / 2000; Ms = Math.max(Ms, law.dens((vv - 1 - 2 * delta) / (2 * (1 + delta))) / (2 * (1 + delta))); }
      const gap = Math.abs(L - (m1 - m2 / 2)), bound = 11 * m3 + (1 + Math.log(2)) * Ms;
      if (!(gap <= bound)) fail(`${law.name} B=${B} delta=${delta}: gap ${gap} > bound ${bound}`);
      if (law.name === 'Gaussian zeta') {
        const eps = 2 / B, env = Math.exp(-1 / (162 * eps)) / Math.sqrt(2 * Math.PI * (49 / 16) * eps);
        if (!(Ms <= env * (1 + 1e-9))) fail(`Gaussian M_sing envelope B=${B} delta=${delta}: ${Ms} > ${env}`);
      }
      worst = Math.max(worst, gap / bound); n++;
      if (delta === 0) ok(`${law.name.padEnd(13)} B=${String(B).padStart(4)}: |E log|1+u| - (Eu - Eu^2/2)| = ${gap.toExponential(3)}, 11 E|u|^3 = ${(11 * m3).toExponential(3)}, M_sing = ${Ms.toExponential(3)}`);
    }
  }
}
// chi2 density envelope on [2/9, 6/7]
for (const B of [16, 64, 256, 1024, 4096]) {
  const k = B / 2, lc = -lgamma(k) - k * Math.log(2);
  for (let i = 0; i <= 400; i++) {
    const w = 2 / 9 + (6 / 7 - 2 / 9) * i / 400, x = B * w;
    const d = B * Math.exp(lc + (k - 1) * Math.log(x) - x / 2), env = Math.sqrt(B / (2 * Math.PI * w * w)) * Math.exp(-(B / 2) * (w - 1 - Math.log(w)));
    if (!(d <= env * (1 + 1e-9))) fail(`chi2 envelope B=${B} w=${w}: ${d} > ${env}`);
  }
}
ok(`${n} cells (|delta| <= 1/8): gap / (11 E|u|^3 + (1 + log 2) M_sing) at most ${worst.toFixed(4)}; Gaussian and chi2 density envelopes hold`);
