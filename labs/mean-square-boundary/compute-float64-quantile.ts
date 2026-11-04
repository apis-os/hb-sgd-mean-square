/**
 * @file labs/mean-square-boundary/compute-float64-quantile.ts
 * @description Linear-interpolated quantile of a `Float64Array` (`computeFloat64Quantile`), sorting a copy by numeric value;
 * the value at position `q * (n - 1)` is interpolated between its two neighbours.
 * @cold
 */
export function computeFloat64Quantile(values: Float64Array, q: number): number {
  const n = values.length;
  if (n === 0) return 0.0;
  const sorted = new Float64Array(values);
  sorted.sort();
  const pos = q * (n - 1);
  const lo = Math.floor(pos) | 0;
  const hi = Math.ceil(pos) | 0;
  const frac = pos - lo;
  return sorted[lo]! * (1.0 - frac) + sorted[hi]! * frac;
}
