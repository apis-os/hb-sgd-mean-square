/**
 * @file labs/mean-square-boundary/compute-scaled-hypot.ts
 * @description `sqrt(x^2 + y^2)` with the scaling of the implicit QL of `computeSymmetricEigendecompositionHouseholderQl`
 * (`computeScaledHypot`): the same operations in the same order, so a QL that calls it reproduces that solver bit for bit.
 * Native `Math.hypot` is not correctly rounded across engines.
 * @cold
 */
export function computeScaledHypot(x: number, y: number): number {
  const ax = Math.abs(x);
  const ay = Math.abs(y);
  const m = ax > ay ? ax : ay;
  if (m === 0.0) return 0.0;
  const rx = ax / m;
  const ry = ay / m;
  return m * Math.sqrt(rx * rx + ry * ry);
}
