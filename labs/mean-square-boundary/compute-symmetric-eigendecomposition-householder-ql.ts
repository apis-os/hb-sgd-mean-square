/**
 * @file labs/mean-square-boundary/compute-symmetric-eigendecomposition-householder-ql.ts
 * @description All eigenvalues and eigenvectors of a dense symmetric matrix (`computeSymmetricEigendecompositionHouseholderQl`):
 * Householder reduction to tridiagonal form and the implicit QL algorithm with eigenvector accumulation (EISPACK `tred2` and
 * `tql2`, in the form of the JAMA port). The vectors are kept transposed, `qt[k * n + a] = (u_k)_a`, so that the `O(n^3)` loops
 * of both stages run over contiguous memory; this is what makes `n = 3 610` take minutes rather than an hour. Eigenvalues are
 * returned in increasing order with `qt` permuted to match. `hypot` is computed with a scaled `Math.sqrt` (native `Math.hypot`
 * is not correctly rounded across engines).
 * @cold
 */

function scaledHypot(x: number, y: number): number {
  const ax = Math.abs(x);
  const ay = Math.abs(y);
  const m = ax > ay ? ax : ay;
  if (m === 0.0) return 0.0;
  const rx = ax / m;
  const ry = ay / m;
  return m * Math.sqrt(rx * rx + ry * ry);
}

function accumulateHouseholderVectors(vt: Float64Array, d: Float64Array, e: Float64Array, n: number): void {
  for (let i = 0; i < ((n - 1) | 0); (i = (i + 1) | 0)) {
    const rowI = Math.imul(i, n);
    const rowI1 = Math.imul((i + 1) | 0, n);
    vt[(rowI + n - 1) | 0] = vt[(rowI + i) | 0]!;
    vt[(rowI + i) | 0] = 1.0;
    const h = d[(i + 1) | 0]!;
    if (h !== 0.0) {
      for (let k = 0; k <= i; (k = (k + 1) | 0)) d[k] = vt[(rowI1 + k) | 0]! / h;
      let j = 0;
      for (; ((j + 3) | 0) <= i; j = (j + 4) | 0) {
        const r0 = Math.imul(j, n);
        const r1 = (r0 + n) | 0;
        const r2 = (r1 + n) | 0;
        const r3 = (r2 + n) | 0;
        let g0 = 0.0;
        let g1 = 0.0;
        let g2 = 0.0;
        let g3 = 0.0;
        for (let k = 0; k <= i; (k = (k + 1) | 0)) {
          const vik = vt[(rowI1 + k) | 0]!;
          g0 += vik * vt[(r0 + k) | 0]!;
          g1 += vik * vt[(r1 + k) | 0]!;
          g2 += vik * vt[(r2 + k) | 0]!;
          g3 += vik * vt[(r3 + k) | 0]!;
        }
        for (let k = 0; k <= i; (k = (k + 1) | 0)) {
          const dk = d[k]!;
          vt[(r0 + k) | 0] = vt[(r0 + k) | 0]! - g0 * dk;
          vt[(r1 + k) | 0] = vt[(r1 + k) | 0]! - g1 * dk;
          vt[(r2 + k) | 0] = vt[(r2 + k) | 0]! - g2 * dk;
          vt[(r3 + k) | 0] = vt[(r3 + k) | 0]! - g3 * dk;
        }
      }
      for (; j <= i; (j = (j + 1) | 0)) {
        const rowJ = Math.imul(j, n);
        let g = 0.0;
        for (let k = 0; k <= i; (k = (k + 1) | 0)) g += vt[(rowI1 + k) | 0]! * vt[(rowJ + k) | 0]!;
        for (let k = 0; k <= i; (k = (k + 1) | 0)) vt[(rowJ + k) | 0] = vt[(rowJ + k) | 0]! - g * d[k]!;
      }
    }
    for (let k = 0; k <= i; (k = (k + 1) | 0)) vt[(rowI1 + k) | 0] = 0.0;
  }
  const lastRow = Math.imul((n - 1) | 0, n);
  for (let j = 0; j < n; (j = (j + 1) | 0)) {
    d[j] = vt[(Math.imul(j, n) + n - 1) | 0]!;
    vt[(Math.imul(j, n) + n - 1) | 0] = 0.0;
  }
  vt[(lastRow + n - 1) | 0] = 1.0;
  e[0] = 0.0;
}

function runImplicitQlAndSort(vt: Float64Array, d: Float64Array, e: Float64Array, n: number): void {
  for (let i = 1; i < n; (i = (i + 1) | 0)) e[(i - 1) | 0] = e[i]!;
  e[(n - 1) | 0] = 0.0;
  let f = 0.0;
  let tst1 = 0.0;
  const eps = 2.220446049250313e-16;
  const n3 = (n - 3) | 0;
  for (let l = 0; l < n; (l = (l + 1) | 0)) {
    const t = Math.abs(d[l]!) + Math.abs(e[l]!);
    if (t > tst1) tst1 = t;
    let m = l;
    while (m < n) {
      if (Math.abs(e[m]!) <= eps * tst1) break;
      m = (m + 1) | 0;
    }
    if (m > l) {
      do {
        let g = d[l]!;
        let p = (d[(l + 1) | 0]! - g) / (2.0 * e[l]!);
        let r = scaledHypot(p, 1.0);
        if (p < 0) r = -r;
        d[l] = e[l]! / (p + r);
        d[(l + 1) | 0] = e[l]! * (p + r);
        const dl1 = d[(l + 1) | 0]!;
        let h = g - d[l]!;
        for (let i = (l + 2) | 0; i < n; (i = (i + 1) | 0)) d[i] = d[i]! - h;
        f = f + h;
        p = d[m]!;
        let c = 1.0;
        let c2 = c;
        let c3 = c;
        const el1 = e[(l + 1) | 0]!;
        let s = 0.0;
        let s2 = 0.0;
        for (let i = (m - 1) | 0; i >= l; (i = (i - 1) | 0)) {
          c3 = c2;
          c2 = c;
          s2 = s;
          g = c * e[i]!;
          h = c * p;
          r = scaledHypot(p, e[i]!);
          e[(i + 1) | 0] = s * r;
          s = e[i]! / r;
          c = p / r;
          p = c * d[i]! - s * g;
          d[(i + 1) | 0] = h + s * (c * g + s * d[i]!);
          const rowI = Math.imul(i, n);
          const rowI1 = (rowI + n) | 0;
          let k = 0;
          for (; k < n3; k = (k + 4) | 0) {
            const idx1 = (rowI1 + k) | 0;
            const idx0 = (rowI + k) | 0;
            const h0 = vt[idx1]!, v0 = vt[idx0]!;
            const h1 = vt[(idx1 + 1) | 0]!, v1 = vt[(idx0 + 1) | 0]!;
            const h2 = vt[(idx1 + 2) | 0]!, v2 = vt[(idx0 + 2) | 0]!;
            const h3 = vt[(idx1 + 3) | 0]!, v3 = vt[(idx0 + 3) | 0]!;
            vt[idx1] = s * v0 + c * h0;
            vt[idx0] = c * v0 - s * h0;
            vt[(idx1 + 1) | 0] = s * v1 + c * h1;
            vt[(idx0 + 1) | 0] = c * v1 - s * h1;
            vt[(idx1 + 2) | 0] = s * v2 + c * h2;
            vt[(idx0 + 2) | 0] = c * v2 - s * h2;
            vt[(idx1 + 3) | 0] = s * v3 + c * h3;
            vt[(idx0 + 3) | 0] = c * v3 - s * h3;
          }
          for (; k < n; (k = (k + 1) | 0)) {
            const hk = vt[(rowI1 + k) | 0]!;
            const vk = vt[(rowI + k) | 0]!;
            vt[(rowI1 + k) | 0] = s * vk + c * hk;
            vt[(rowI + k) | 0] = c * vk - s * hk;
          }
        }
        p = (-s * s2 * c3 * el1 * e[l]!) / dl1;
        e[l] = s * p;
        d[l] = c * p;
      } while (Math.abs(e[l]!) > eps * tst1);
    }
    d[l] = d[l]! + f;
    e[l] = 0.0;
  }

  for (let i = 0; i < ((n - 1) | 0); (i = (i + 1) | 0)) {
    let k = i;
    let p = d[i]!;
    for (let j = (i + 1) | 0; j < n; (j = (j + 1) | 0)) {
      if (d[j]! < p) {
        k = j;
        p = d[j]!;
      }
    }
    if (k !== i) {
      d[k] = d[i]!;
      d[i] = p;
      const rowI = Math.imul(i, n);
      const rowK = Math.imul(k, n);
      for (let j = 0; j < n; (j = (j + 1) | 0)) {
        const tmp = vt[(rowI + j) | 0]!;
        vt[(rowI + j) | 0] = vt[(rowK + j) | 0]!;
        vt[(rowK + j) | 0] = tmp;
      }
    }
  }
}

export function computeSymmetricEigendecompositionHouseholderQl(
  matrixRowMajor: Float64Array,
  n: number,
): { readonly values: Float64Array; readonly qt: Float64Array } {
  const vt = new Float64Array(matrixRowMajor);
  const d = new Float64Array(n);
  const e = new Float64Array(n);
  for (let j = 0; j < n; (j = (j + 1) | 0)) d[j] = vt[(Math.imul(j, n) + n - 1) | 0]!;

  for (let i = (n - 1) | 0; i > 0; (i = (i - 1) | 0)) {
    let scale = 0.0;
    let h = 0.0;
    for (let k = 0; k < i; (k = (k + 1) | 0)) scale += Math.abs(d[k]!);
    if (scale === 0.0) {
      e[i] = d[(i - 1) | 0]!;
      for (let j = 0; j < i; (j = (j + 1) | 0)) {
        d[j] = vt[(Math.imul(j, n) + i - 1) | 0]!;
        vt[(Math.imul(j, n) + i) | 0] = 0.0;
        vt[(Math.imul(i, n) + j) | 0] = 0.0;
      }
    } else {
      for (let k = 0; k < i; (k = (k + 1) | 0)) {
        d[k] = d[k]! / scale;
        h += d[k]! * d[k]!;
      }
      let f = d[(i - 1) | 0]!;
      let g = Math.sqrt(h);
      if (f > 0) g = -g;
      e[i] = scale * g;
      h = h - f * g;
      d[(i - 1) | 0] = f - g;
      for (let j = 0; j < i; (j = (j + 1) | 0)) e[j] = 0.0;
      for (let j = 0; j < i; (j = (j + 1) | 0)) {
        const rowJ = Math.imul(j, n);
        f = d[j]!;
        vt[(Math.imul(i, n) + j) | 0] = f;
        g = e[j]! + vt[(rowJ + j) | 0]! * f;
        for (let k = (j + 1) | 0; k <= ((i - 1) | 0); (k = (k + 1) | 0)) {
          const vkj = vt[(rowJ + k) | 0]!;
          g += vkj * d[k]!;
          e[k] = e[k]! + vkj * f;
        }
        e[j] = g;
      }
      f = 0.0;
      for (let j = 0; j < i; (j = (j + 1) | 0)) {
        e[j] = e[j]! / h;
        f += e[j]! * d[j]!;
      }
      const hh = f / (h + h);
      for (let j = 0; j < i; (j = (j + 1) | 0)) e[j] = e[j]! - hh * d[j]!;
      for (let j = 0; j < i; (j = (j + 1) | 0)) {
        const rowJ = Math.imul(j, n);
        f = d[j]!;
        g = e[j]!;
        for (let k = j; k <= ((i - 1) | 0); (k = (k + 1) | 0)) {
          vt[(rowJ + k) | 0] = vt[(rowJ + k) | 0]! - (f * e[k]! + g * d[k]!);
        }
        d[j] = vt[(rowJ + i - 1) | 0]!;
        vt[(rowJ + i) | 0] = 0.0;
      }
    }
    d[i] = h;
  }

  accumulateHouseholderVectors(vt, d, e, n);
  runImplicitQlAndSort(vt, d, e, n);
  return { values: d, qt: vt };
}
