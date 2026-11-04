/**
 * @file labs/mean-square-boundary/multiply-dense-matrix-vector-six-rows.ts
 * @description `y = A x` for a dense row-major `dim x dim` matrix (`multiplyDenseMatrixVectorSixRows`), six rows per pass over
 * `x` so that each `x[c]` is loaded once for six rows. Every row is still summed from `0.0` over `c = 0 .. dim - 1` in order, so
 * each `y[r]` is bit-identical to the scalar loop `s += A[r][c] * x[c]`. `x` and `y` must not alias.
 * @cold
 */
export function multiplyDenseMatrixVectorSixRows(a: Float64Array, x: Float64Array, y: Float64Array, dim: number): void {
  const d = dim | 0;
  let r = 0;
  for (; ((r + 5) | 0) < d; r = (r + 6) | 0) {
    const r0 = Math.imul(r, d);
    const r1 = (r0 + d) | 0;
    const r2 = (r1 + d) | 0;
    const r3 = (r2 + d) | 0;
    const r4 = (r3 + d) | 0;
    const r5 = (r4 + d) | 0;
    let s0 = 0.0;
    let s1 = 0.0;
    let s2 = 0.0;
    let s3 = 0.0;
    let s4 = 0.0;
    let s5 = 0.0;
    for (let c = 0; c < d; (c = (c + 1) | 0)) {
      const xc = x[c]!;
      s0 += a[(r0 + c) | 0]! * xc;
      s1 += a[(r1 + c) | 0]! * xc;
      s2 += a[(r2 + c) | 0]! * xc;
      s3 += a[(r3 + c) | 0]! * xc;
      s4 += a[(r4 + c) | 0]! * xc;
      s5 += a[(r5 + c) | 0]! * xc;
    }
    y[r] = s0;
    y[(r + 1) | 0] = s1;
    y[(r + 2) | 0] = s2;
    y[(r + 3) | 0] = s3;
    y[(r + 4) | 0] = s4;
    y[(r + 5) | 0] = s5;
  }
  for (; r < d; (r = (r + 1) | 0)) {
    const row = Math.imul(r, d);
    let s = 0.0;
    for (let c = 0; c < d; (c = (c + 1) | 0)) s += a[(row + c) | 0]! * x[c]!;
    y[r] = s;
  }
}
