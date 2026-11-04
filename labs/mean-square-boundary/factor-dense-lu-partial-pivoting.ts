/**
 * @file labs/mean-square-boundary/factor-dense-lu-partial-pivoting.ts
 * @description LU factorisation with partial pivoting of a dense row-major `dim x dim` matrix, in place
 * (`factorDenseLuPartialPivoting`; amendment 4, inverse iteration of the Perron certificate), and the solve with the factors.
 * `a` holds `L` (unit lower, below the diagonal) and `U` on return; `piv[k]` is the row swapped into `k`. Returns `null` when a
 * pivot is exactly zero. `solve(b)` overwrites `b` with `A^{-1} b`.
 * @cold
 */
export function factorDenseLuPartialPivoting(a: Float64Array, dim: number): { readonly solve: (b: Float64Array) => void } | null {
  const piv = new Int32Array(dim);
  for (let k = 0; k < dim; (k = (k + 1) | 0)) {
    let p = k;
    let best = Math.abs(a[(Math.imul(k, dim) + k) | 0]!);
    for (let r = (k + 1) | 0; r < dim; (r = (r + 1) | 0)) {
      const v = Math.abs(a[(Math.imul(r, dim) + k) | 0]!);
      if (v > best) { best = v; p = r; }
    }
    piv[k] = p;
    if (best === 0.0) return null;
    if (p !== k) {
      const rk = Math.imul(k, dim);
      const rp = Math.imul(p, dim);
      for (let c = 0; c < dim; (c = (c + 1) | 0)) { const t = a[(rk + c) | 0]!; a[(rk + c) | 0] = a[(rp + c) | 0]!; a[(rp + c) | 0] = t; }
    }
    const rk = Math.imul(k, dim);
    const inv = 1.0 / a[(rk + k) | 0]!;
    for (let r = (k + 1) | 0; r < dim; (r = (r + 1) | 0)) {
      const rr = Math.imul(r, dim);
      const l = a[(rr + k) | 0]! * inv;
      a[(rr + k) | 0] = l;
      if (l === 0.0) continue;
      for (let c = (k + 1) | 0; c < dim; (c = (c + 1) | 0)) a[(rr + c) | 0] = a[(rr + c) | 0]! - l * a[(rk + c) | 0]!;
    }
  }
  const solve = (b: Float64Array): void => {
    for (let k = 0; k < dim; (k = (k + 1) | 0)) { const p = piv[k]!; if (p !== k) { const t = b[k]!; b[k] = b[p]!; b[p] = t; } }
    for (let r = 1; r < dim; (r = (r + 1) | 0)) {
      const rr = Math.imul(r, dim);
      let s = b[r]!;
      for (let c = 0; c < r; (c = (c + 1) | 0)) s -= a[(rr + c) | 0]! * b[c]!;
      b[r] = s;
    }
    for (let r = (dim - 1) | 0; r >= 0; (r = (r - 1) | 0)) {
      const rr = Math.imul(r, dim);
      let s = b[r]!;
      for (let c = (r + 1) | 0; c < dim; (c = (c + 1) | 0)) s -= a[(rr + c) | 0]! * b[c]!;
      b[r] = s / a[(rr + r) | 0]!;
    }
  };
  return { solve };
}
