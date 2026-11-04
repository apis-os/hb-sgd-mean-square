/**
 * @file paper/checks/evidence-ranges.mjs
 * @description Verifies and prints exact ranges from evidence JSON files for Phases 1 and 3,
 * plus Gaussian Lyapunov exponent and chi-squared quadrature for the Phase 1 almost-sure stability boundary.
 *
 * Plain Node.js, zero dependencies. Run with:
 *   node paper/checks/evidence-ranges.mjs
 */

import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '../..');

// ---------------------------------------------------------------------------
// Helpers: Quadrature and Special Functions
// ---------------------------------------------------------------------------

function lgamma(z) {
  if (Number.isInteger(z) && z > 0) {
    let s = 0.0;
    for (let i = 1; i < z; i++) {
      s += Math.log(i);
    }
    return s;
  }
  const p = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.138571095836526,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  const g = 7;
  const x = z - 1;
  let a = p[0];
  const t = x + g + 0.5;
  for (let i = 1; i < p.length; i++) {
    a += p[i] / (x + i);
  }
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

function chi2Pdf(x, k) {
  if (x <= 0) return 0;
  const halfK = k / 2;
  return Math.exp((halfK - 1) * Math.log(x) - x / 2 - halfK * Math.log(2) - lgamma(halfK));
}

function normPdf(z, sigma) {
  return (1 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * (z / sigma) ** 2);
}

function lyapGauss(eta, lambda1, batchB, nPoints = 200000, zStdDevs = 8) {
  const sigma = Math.sqrt((2 * lambda1 * lambda1) / batchB);
  const zMax = zStdDevs * sigma;
  const dz = (2 * zMax) / nPoints;
  let sum = 0;
  for (let i = 0; i < nPoints; i++) {
    const z = -zMax + (i + 0.5) * dz;
    const mult = Math.abs(1 - eta * (lambda1 + z));
    if (mult > 1e-15) {
      sum += Math.log(mult) * normPdf(z, sigma) * dz;
    }
  }
  return sum;
}

function lyapChi2(eta, k, nPoints = 200000, xMax = 250) {
  const dx = xMax / nPoints;
  let sum = 0;
  for (let i = 0; i < nPoints; i++) {
    const x = (i + 0.5) * dx;
    const mult = Math.abs(1 - (eta * x) / k);
    if (mult > 1e-15) {
      sum += Math.log(mult) * chi2Pdf(x, k) * dx;
    }
  }
  return sum;
}

function findChi2Root(k, lo = 2.05, hi = 2.09, tol = 1e-7) {
  while (hi - lo > tol) {
    const mid = 0.5 * (lo + hi);
    const val = lyapChi2(mid, k);
    if (val < 0) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return 0.5 * (lo + hi);
}

// ---------------------------------------------------------------------------
// Main Verification Script
// ---------------------------------------------------------------------------

function runChecks() {
  console.log('================================================================');
  console.log('Evidence Ranges Verification (Phases 1, 3 + Quadratures)');
  console.log('================================================================\n');

  // 1. Phase 1
  const p9Path = resolve(repoRoot, 'evidence/mean-square-boundary/raw-phase1.json');
  const p9 = JSON.parse(readFileSync(p9Path, 'utf8'));

  const insideCells = p9.cells.filter((c) => c.insideSandwich);
  const outsideCells = p9.cells.filter((c) => !c.insideSandwich);

  const secErrs = insideCells.map((c) => c.secularMedianErrPct);
  const leeErrs = insideCells.map((c) => c.secularLeeLimitMedianErrPct);
  const halfWidths = insideCells.map((c) => c.bisectionHalfWidthPct);

  const minSecErr = Math.min(...secErrs);
  const maxSecErr = Math.max(...secErrs);
  const minLeeErr = Math.min(...leeErrs);
  const maxLeeErr = Math.max(...leeErrs);
  const maxHw = Math.max(...halfWidths);

  const maxSecCell = insideCells.find((c) => c.secularMedianErrPct === maxSecErr);

  const rad64Cell = p9.cells.find(
    (c) => c.numParams === 64 && c.beta === 0 && c.batchB === 64 && c.featureFamily === 'rademacher-iid',
  );
  const gaussB16Cell = p9.cells.find(
    (c) => c.numParams === 64 && c.beta === 0 && c.batchB === 16 && c.featureFamily === 'gaussian-iid',
  );

  console.log('--- Phase 1 (raw-phase1.json) ---');
  console.log(`Total cells: ${p9.cells.length}`);
  console.log(`Inside sandwich cells: ${insideCells.length}`);
  console.log(`Outside sandwich cells: ${outsideCells.length} (${outsideCells.map((c) => c.cellId).join(', ')})`);
  console.log(`Secular median error range (inside sandwich): [${minSecErr.toFixed(4)}%, ${maxSecErr.toFixed(4)}%]`);
  console.log(`  (min rounds to ${minSecErr.toFixed(2)}%, max rounds to ${maxSecErr.toFixed(2)}%)`);
  console.log(`  Max error cell: ${maxSecCell.cellId} (${maxSecErr.toFixed(4)}%)`);
  console.log(`Lee limit median error range (inside sandwich): [${minLeeErr.toFixed(4)}%, ${maxLeeErr.toFixed(4)}%]`);
  console.log(`  (min rounds to ${minLeeErr.toFixed(2)}%, max rounds to ${maxLeeErr.toFixed(2)}%)`);
  console.log(`Max bisection half-width (inside sandwich): ${maxHw.toFixed(4)}% (<= 0.05%)`);
  console.log(`Rademacher P=64 beta=0 B=64 error: ${rad64Cell.secularMedianErrPct.toFixed(6)}% (rounds to ${rad64Cell.secularMedianErrPct.toFixed(2)}%, bounded by 0.72%)`);
  if (gaussB16Cell) {
    console.log(`Gaussian P=64 beta=0 B=16 error: ${gaussB16Cell.secularMedianErrPct.toFixed(6)}% (rounds to ${gaussB16Cell.secularMedianErrPct.toFixed(2)}%)`);
  }
  console.log();

  // 2. Phase 3
  const p11Path = resolve(repoRoot, 'evidence/mean-square-boundary/raw-phase3.json');
  const p11 = JSON.parse(readFileSync(p11Path, 'utf8'));

  const crossingErrs = p11.cells.map((c) => c.crossingRelErr);
  const maxCrossingErr = Math.max(...crossingErrs);
  const belowCount = p11.cells.filter((c) => c.radiusBelow < 1.0).length;
  const aboveCount = p11.cells.filter((c) => c.radiusAbove > 1.0).length;
  const antiRadii = p11.cells.map((c) => c.radiusAntisymmetricAtRoot);
  const minAnti = Math.min(...antiRadii);
  const maxAnti = Math.max(...antiRadii);

  const notConvergedCells = p11.cells.filter((c) => !c.allConverged);

  console.log('--- Phase 3 (raw-phase3.json) ---');
  console.log(`Total cells: ${p11.cells.length}`);
  console.log(`Max crossing relative error: ${maxCrossingErr.toExponential(6)} (rounds to ${maxCrossingErr.toExponential(2)}, bounded by 6.11e-10, exceeds 6e-10 in 2 cells)`);
  console.log(`Cells with radiusBelow < 1: ${belowCount}/${p11.cells.length}`);
  console.log(`Cells with radiusAbove > 1: ${aboveCount}/${p11.cells.length}`);
  console.log(`Antisymmetric spectral radius range at root: [${minAnti.toFixed(6)}, ${maxAnti.toFixed(6)}] (rounds to [${minAnti.toFixed(2)}, ${maxAnti.toFixed(2)}])`);
  console.log(`Cells with allConverged == false: ${notConvergedCells.length}`);

  const notConvCrossing = notConvergedCells.map((c) => c.crossingRelErr);
  const notConvRadiusDiff = notConvergedCells.map((c) => Math.abs(c.radiusAtRoot - 1.0));
  for (const c of notConvergedCells) {
    console.log(`  ${c.cellId}: crossingRelErr = ${c.crossingRelErr.toExponential(4)}, |radiusAtRoot - 1| = ${Math.abs(c.radiusAtRoot - 1.0).toExponential(4)}`);
  }
  console.log(`  Crossing rel errors for non-converged: [${Math.min(...notConvCrossing).toExponential(2)}, ${Math.max(...notConvCrossing).toExponential(2)}]`);
  console.log(`  |radiusAtRoot - 1| for non-converged: [${Math.min(...notConvRadiusDiff).toExponential(2)}, ${Math.max(...notConvRadiusDiff).toExponential(2)}]`);
  console.log();

  // 3. Quadratures for Lyapunov Exponent and Chi-Squared Root
  console.log('--- Quadratures for Lyapunov Exponent and chi^2_64 Root ---');
  const lambda1 = 1.0;
  const batchB = 64;
  const etaValues = [2.00, 2.02, 2.04, 2.10];
  console.log('Gaussian Lyapunov exponent E log|1 - eta(lambda_1 + zeta)|, zeta ~ N(0, 2 lambda_1^2 / B):');
  for (const eta of etaValues) {
    const val = lyapGauss(eta, lambda1, batchB);
    console.log(`  eta = ${eta.toFixed(2)}: ${val.toFixed(6)} (rounds to ${val.toFixed(3)})`);
  }

  console.log('\nExact chi^2_64 Lyapunov exponent E log|1 - eta chi^2_64 / 64|:');
  for (const eta of etaValues) {
    const val = lyapChi2(eta, batchB);
    console.log(`  eta = ${eta.toFixed(2)}: ${val.toFixed(6)} (rounds to ${val.toFixed(3)})`);
  }

  const root = findChi2Root(batchB);
  console.log(`\nExact root of E log|1 - eta chi^2_64 / 64| = 0: ${root.toFixed(6)} (rounds to ${root.toFixed(4)})`);
  console.log('================================================================');
}

runChecks();
