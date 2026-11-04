#!/usr/bin/env node
// Numerical checks for Proposition G3 at beta = 0, lambda1 = 1, delta = 0.
// The multiplier is |1 + 2*zeta| and the quadratic centre is -2*nu^2.
// The script prints the quadrature thresholds quoted in the proposition,
// checks the Lagrange remainder under |u| <= 1/2, and checks that a
// two-point family approaching the singular multiplier has an unbounded ratio.

function lgamma(z) {
  const g = 7;
  const C = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.138571095836524, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (z < 0.5) return Math.PI / (Math.sin(Math.PI * z) * Math.exp(lgamma(1 - z)));
  z -= 1;
  let x = C[0];
  for (let i = 1; i < g + 2; i++) x += C[i] / (z + i);
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

// (a) Gaussian quadrature for zeta ~ N(0, 1/B)
function evalGaussian(B) {
  const sigma = 1 / Math.sqrt(B);
  const m3 = Math.sqrt(8 / Math.PI) * Math.pow(sigma, 3);
  const nu4 = 1 / (B * B);

  // Quadrature over [-10, 10] with 500,000 subintervals (Simpson's rule)
  const a = -10, b = 10;
  const N = 500000;
  const h = (b - a) / N;
  let sum = 0;
  for (let i = 0; i <= N; i++) {
    const x = a + i * h;
    const w = (i === 0 || i === N) ? 1 : (i % 2 === 1 ? 4 : 2);
    const arg = Math.abs(1 + 2 * sigma * x);
    if (arg < 1e-15) continue;
    const f = Math.log(arg) * Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
    sum += w * f;
  }
  const elog = sum * (h / 3);
  return { elog, m3, nu4 };
}

// (b) Three-point law zeta in {0, +-a}, a = 0.4, P(+-a) = nu^2 / (2 a^2), nu^2 = 1/B
function evalThreePoint(B, a = 0.4) {
  const nu2 = 1 / B;
  const probA = nu2 / (2 * a * a); // prob for +a and for -a
  const prob0 = 1 - 2 * probA;
  const m3 = 2 * probA * Math.pow(a, 3); // = a / B
  const nu4 = 1 / (B * B);

  const elog = prob0 * Math.log(1) + probA * Math.log(Math.abs(1 + 2 * a)) + probA * Math.log(Math.abs(1 - 2 * a));
  return { elog, m3, nu4 };
}

// (c1) Scaled Rademacher sign zeta in {+- 1/sqrt(B)} with equal probability
function evalRademacherScaled(B) {
  const z = 1 / Math.sqrt(B);
  const m3 = Math.pow(z, 3);
  const nu4 = 1 / (B * B);
  const elog = 0.5 * Math.log(Math.abs(1 + 2 * z)) + 0.5 * Math.log(Math.abs(1 - 2 * z));
  return { elog, m3, nu4 };
}

// (c2) Rademacher average zeta = (1/B) * sum_{b=1}^B s_b
function evalRademacherAverage(B) {
  let elog = 0;
  let m3 = 0;
  let totalProb = 0;
  const nu4 = 1 / (B * B);

  for (let k = 0; k <= B; k++) {
    if (k === B / 4) continue; // singularity 1 + 2*zeta = 0 has prob <= 2.6e-5 at B=64
    const lbinom = lgamma(B + 1) - lgamma(k + 1) - lgamma(B - k + 1);
    const prob = Math.exp(lbinom - B * Math.log(2));
    const zeta = (2 * k - B) / B;
    totalProb += prob;
    elog += prob * Math.log(Math.abs(1 + 2 * zeta));
    m3 += prob * Math.pow(Math.abs(zeta), 3);
  }
  elog /= totalProb;
  m3 /= totalProb;
  return { elog, m3, nu4 };
}

const BValues = [64, 256, 1024, 4096];

console.log('=== Proposition G3: Lyapunov Exponent Remainder Scaling ===\n');

const cases = [
  { name: '(a) Gaussian zeta ~ N(0, 1/B)', fn: evalGaussian },
  { name: '(b) Three-point law zeta in {0, +-0.4}, P(+-0.4) = 1/(0.32 B)', fn: evalThreePoint },
  { name: '(c1) Scaled Rademacher zeta in {+-1/sqrt(B)}', fn: evalRademacherScaled },
  { name: '(c2) Rademacher average zeta = (1/B) sum_{b=1}^B s_b (conditioned zeta != -0.5)', fn: evalRademacherAverage },
];

for (const c of cases) {
  console.log(`--- ${c.name} ---`);
  console.log('   B   | Quadratic centre |      E log|M|     |       Gap        |   Gap / B^{-3/2}   |   Gap / (m_3+nu^4) ');
  console.log('-------+------------------+-------------------+------------------+--------------------+--------------------');

  for (const B of BValues) {
    const { elog, m3, nu4 } = c.fn(B);
    const centre = -2 / B;
    const gap = elog - centre;
    const oldRemainder = Math.pow(B, -1.5);
    const newRemainder = m3 + nu4;
    const ratioOld = gap / oldRemainder;
    const ratioNew = gap / newRemainder;

    console.log(
      `${B.toString().padStart(6)} | ` +
      `${centre.toFixed(10).padStart(16)} | ` +
      `${elog.toFixed(10).padStart(17)} | ` +
      `${gap.toExponential(6).padStart(16)} | ` +
      `${ratioOld.toFixed(6).padStart(18)} | ` +
      `${ratioNew.toFixed(6).padStart(18)}`
    );
  }
  console.log('');
}

// Support bound |u| = 2|zeta| <= 1/2: Lagrange remainder |R| <= (8/3) E|u|^3.
console.log('Support bound |u| <= 1/2, symmetric two-point law:');
for (const a of [0.05, 0.1, 0.15, 0.2, 0.25]) {
  const elog = 0.5 * Math.log(1 - 4 * a * a);
  const centre = -2 * a * a;
  const gap = elog - centre;
  const Eu3 = 8 * a * a * a;
  const bound = (8 / 3) * Eu3;
  if (!(Math.abs(gap) <= bound + 1e-12)) {
    throw new Error(`support-bound remainder failed at a=${a}: |gap|=${Math.abs(gap)} > ${bound}`);
  }
  console.log(`  a=${a.toFixed(2)}: gap=${gap.toExponential(6)}, (8/3)E|u|^3=${bound.toExponential(6)}`);
}

// Singular family zeta = ±(1/2)(1-eps). The ratio of the gap to m_3 tends to -infinity.
console.log('\nSingular two-point family zeta = ±(1/2)(1-eps), delta = 0:');
let previous = 0;
for (const eps of [1e-2, 1e-4, 1e-6, 1e-8]) {
  const a = 0.5 * (1 - eps);
  const elog = 0.5 * Math.log(eps * (2 - eps));
  const centre = -0.5 * (1 - eps) * (1 - eps);
  const gap = elog - centre;
  const m3 = a * a * a;
  const ratio = gap / m3;
  console.log(`  eps=${eps.toExponential(0)}: gap=${gap.toExponential(6)}, gap/m_3=${ratio.toFixed(4)}`);
  if (!(ratio < previous)) throw new Error(`singular ratio did not decrease at eps=${eps}`);
  previous = ratio;
}
if (!(previous < -40)) throw new Error(`singular ratio ${previous} is not large`);

// Verification of Proposition G3(1) and line 638 threshold values for B=64, beta=0, lambda1=1
console.log('\nVerification of Proposition G3(1) threshold values (B=64, beta=0, lambda1=1):');
const B64 = 64;
const nu2_64 = 2 / B64; // = 1/32 for Gaussian features (chi^2_B / B - 1)
const eta_ms_first_order = 2 * (1 - nu2_64);
const eta_ms_exact = 2 / (1 + nu2_64);
const eta_as_first_order = 2 * (1 + nu2_64);

const ms1 = eta_ms_first_order.toFixed(4);
const msE = eta_ms_exact.toFixed(4);
const as1 = eta_as_first_order.toFixed(4);
if (ms1 !== '1.9375' || msE !== '1.9394' || as1 !== '2.0625') {
  throw new Error(`threshold strings ${ms1}, ${msE}, ${as1}`);
}
console.log(`  First-order mean-square threshold: ${ms1}`);
console.log(`  Exact mean-square threshold:       ${msE}`);
console.log(`  First-order almost-sure threshold: ${as1}`);

function elogChi2(eta, B = 64) {
  const k = B / 2;
  const lnorm = -k * Math.log(2) - lgamma(k);
  const N = 500000;
  const a = 1, b = 250;
  const h = (b - a) / N;
  let sum = 0;
  for (let i = 0; i <= N; i++) {
    const x = a + i * h;
    const w = (i === 0 || i === N) ? 1 : (i % 2 === 1 ? 4 : 2);
    const arg = Math.abs(1 - eta * x / B);
    if (arg < 1e-15) continue;
    const logf = (k - 1) * Math.log(x) - 0.5 * x + lnorm;
    sum += w * Math.log(arg) * Math.exp(logf);
  }
  return sum * (h / 3);
}

function findRoot(fn, l, r) {
  for (let i = 0; i < 50; i++) {
    const m = 0.5 * (l + r);
    if (fn(m) > 0) r = m; else l = m;
  }
  return 0.5 * (l + r);
}

const rootChi2 = findRoot(eta => elogChi2(eta, 64), 2.05, 2.08);
console.log(`  Quadrature Chi^2_64 root:          ${rootChi2.toFixed(4)}`);

function elogGaussEta(eta, sigma = Math.sqrt(1 / 32)) {
  const a = -12, b = 12;
  const N = 500000;
  const h = (b - a) / N;
  let sum = 0;
  for (let i = 0; i <= N; i++) {
    const x = a + i * h;
    const w = (i === 0 || i === N) ? 1 : (i % 2 === 1 ? 4 : 2);
    const arg = Math.abs(1 - eta * (1 + sigma * x));
    if (arg < 1e-15) continue;
    const f = Math.log(arg) * Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
    sum += w * f;
  }
  return sum * (h / 3);
}

const rootGauss = findRoot(elogGaussEta, 2.06, 2.09);
console.log(`  Quadrature Gaussian root:          ${rootGauss.toFixed(4)} (3 dec: ${rootGauss.toFixed(3)})`);

console.log('\nAll numerical checks in g3-lyapunov-remainder.mjs passed.');
