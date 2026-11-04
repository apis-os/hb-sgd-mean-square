# Phase 1: the stochastic ceiling in R^P, Gaussian against Rademacher features (Theorem G)

- **Date**: `2026-10-27T17:08:57.456Z`
- **Node**: `v26.5.0`
- **Seeds per cell**: `8`; streams `8`, steps `1000`, bisection iterations `12`, growth-ratio threshold `2.5`

## 1. Pre-registered predictions (`p1`–`p4`)

- **`p1`**: p1 (Theorem G(3)): in every i.i.d. cell (Gaussian and Rademacher features), the median empirical mean-square ceiling lies inside the sandwich [2(1+beta)/(alpha lambda_1 + kappa_beta tr(H)/B), 2(1+beta)/max(alpha lambda_1 + kappa_beta lambda_1/B, kappa_beta tr(H)/B)] within 3 % plus the bisection half-width.
- **`p2`**: p2 (Theorem G(2)): in every i.i.d. cell the secular root predicts the empirical ceiling with median relative error below 5 %.
- **`p3`**: p3 (Theorem G(4), Rademacher separation): in every Rademacher cell with kappa_beta tr(H)/(B lambda_1) >= 1, the one-dimensional ceiling 2(1+beta)/lambda_1 of Theorem E(3) (nu_11 = 0) exceeds the empirical ceiling by more than a factor 2 while the secular root stays within 5 %.
- **`p4`**: p4 (Remark G.1, Lee–Cheng–Paquette–Paquette 2022): in the finite-pool arm (Gaussian features sampled without replacement, zeta = B/n <= 1/8), the root of the kernel-norm equation ||K|| = 1 with the factor (1 - zeta) and no kurtosis term predicts the empirical ceiling with median relative error below 8 %.

### Prediction outcomes

- `p1IidCeilingsInsideSandwich`: `false`
- `p2SecularRootWithin5PctMedianIid`: `false`
- `p3RademacherOneDimOverestimatesBy2xWhileSecularWithin5Pct`: `true`
- `p4PoolArmLeeRootWithin8PctAtSmallZeta`: `true`

## 2. Cells

| Cell | `P` | `beta` | `B` | Family | `zeta` | `kappa_beta tr(H)/(B lambda_1)` | `eta_det` | 1D (Thm E(3)) | Secular (Thm G) | Lee limit | Sandwich lower | Sandwich upper | Empirical median | Emp. p05 | Emp. p95 | Bisection ±% | Secular err % | Lee err % | 1D err % | 1D / emp | Inside sandwich |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `P64_beta0_B4_gaussian-iid` | 64 | 0 | 4 | gaussian-iid | 0.0000 | 3.65 | 2.0000 | 1.3333 | 0.4897 | 0.5009 | 0.4081 | 0.5479 | 0.4894 | 0.4875 | 0.4957 | 0.05 | 0.34 | 2.35 | 172.44 | 2.72 | yes |
| `P64_beta0_B4_rademacher-iid` | 64 | 0 | 4 | rademacher-iid | 0.0000 | 3.65 | 2.0000 | 2.0000 | 0.5123 | 0.5009 | 0.4545 | 0.5479 | 0.5135 | 0.5121 | 0.5162 | 0.05 | 0.22 | 2.45 | 289.49 | 3.89 | yes |
| `P64_beta0_B4_gaussian-pool` | 64 | 0 | 4 | gaussian-pool | 0.0078 | 3.44 | 1.8918 | 1.2645 | 0.4887 | 0.5011 | 0.4052 | 0.5533 | 0.4896 | 0.4816 | 0.4958 | 0.05 | 0.33 | 2.38 | 157.59 | 2.58 | yes |
| `P64_beta0_B16_gaussian-iid` | 64 | 0 | 16 | gaussian-iid | 0.0000 | 0.91 | 2.0000 | 1.7778 | 1.4468 | 1.4882 | 1.0126 | 1.7778 | 1.4862 | 1.4497 | 1.4969 | 0.02 | 2.65 | 0.47 | 19.62 | 1.20 | yes |
| `P64_beta0_B16_rademacher-iid` | 64 | 0 | 16 | rademacher-iid | 0.0000 | 0.91 | 2.0000 | 2.0000 | 1.5299 | 1.4882 | 1.0810 | 2.0000 | 1.5544 | 1.5298 | 1.5646 | 0.02 | 1.57 | 4.26 | 28.67 | 1.29 | yes |
| `P64_beta0_B16_gaussian-pool` | 64 | 0 | 16 | gaussian-pool | 0.0313 | 0.89 | 1.9532 | 1.7422 | 1.4341 | 1.4770 | 1.0156 | 1.7422 | 1.4421 | 1.4034 | 1.5150 | 0.02 | 1.00 | 1.97 | 18.69 | 1.20 | yes |
| `P64_beta0_B64_gaussian-iid` | 64 | 0 | 64 | gaussian-iid | 0.0000 | 0.23 | 2.0000 | 1.9394 | 1.9268 | 1.9560 | 1.6080 | 1.9394 | 2.0399 | 1.9852 | 2.0400 | 0.01 | 5.54 | 4.11 | 4.93 | 0.96 | no |
| `P64_beta0_B64_rademacher-iid` | 64 | 0 | 64 | rademacher-iid | 0.0000 | 0.23 | 2.0000 | 2.0000 | 1.9861 | 1.9560 | 1.6494 | 2.0000 | 2.0004 | 1.9865 | 2.0146 | 0.01 | 0.71 | 2.22 | 0.47 | 1.00 | yes |
| `P64_beta0_B64_gaussian-pool` | 64 | 0 | 64 | gaussian-pool | 0.1250 | 0.22 | 1.9386 | 1.8870 | 1.8778 | 1.9028 | 1.6050 | 1.8870 | 1.9408 | 1.7940 | 2.1111 | 0.01 | 3.82 | 2.53 | 3.41 | 0.97 | yes |
| `P64_beta0.9_B4_gaussian-iid` | 64 | 0.9 | 4 | gaussian-iid | 0.0000 | 69.36 | 3.8000 | 0.3619 | 0.0533 | 0.0545 | 0.0506 | 0.0548 | 0.0532 | 0.0529 | 0.0537 | 0.05 | 0.44 | 2.45 | 579.95 | 6.80 | yes |
| `P64_beta0.9_B4_rademacher-iid` | 64 | 0.9 | 4 | rademacher-iid | 0.0000 | 69.36 | 3.8000 | 3.8000 | 0.0558 | 0.0545 | 0.0548 | 0.0579 | 0.0559 | 0.0552 | 0.0564 | 0.05 | 0.39 | 2.49 | 6695.36 | 68.04 | yes |
| `P64_beta0.9_B4_gaussian-pool` | 64 | 0.9 | 4 | gaussian-pool | 0.0078 | 67.42 | 3.7173 | 0.3565 | 0.0536 | 0.0549 | 0.0509 | 0.0552 | 0.0538 | 0.0529 | 0.0552 | 0.05 | 0.97 | 1.54 | 551.14 | 6.60 | yes |
| `P64_beta0.9_B16_gaussian-iid` | 64 | 0.9 | 16 | gaussian-iid | 0.0000 | 17.34 | 3.8000 | 1.1259 | 0.2103 | 0.2151 | 0.1946 | 0.2191 | 0.2113 | 0.2093 | 0.2167 | 0.05 | 0.59 | 1.81 | 432.98 | 5.30 | yes |
| `P64_beta0.9_B16_rademacher-iid` | 64 | 0.9 | 16 | rademacher-iid | 0.0000 | 17.34 | 3.8000 | 3.8000 | 0.2199 | 0.2151 | 0.2191 | 0.2215 | 0.2196 | 0.2178 | 0.2232 | 0.05 | 0.78 | 2.06 | 1630.45 | 17.26 | yes |
| `P64_beta0.9_B16_gaussian-pool` | 64 | 0.9 | 16 | gaussian-pool | 0.0313 | 16.57 | 3.6315 | 1.1002 | 0.2155 | 0.2208 | 0.1989 | 0.2256 | 0.2170 | 0.2140 | 0.2212 | 0.05 | 1.69 | 1.55 | 401.70 | 5.07 | yes |
| `P64_beta0.9_B64_gaussian-iid` | 64 | 0.9 | 64 | gaussian-iid | 0.0000 | 4.33 | 3.8000 | 2.3843 | 0.7948 | 0.8130 | 0.6747 | 0.8766 | 0.7971 | 0.7858 | 0.8086 | 0.05 | 0.77 | 1.99 | 199.12 | 2.99 | yes |
| `P64_beta0.9_B64_rademacher-iid` | 64 | 0.9 | 64 | rademacher-iid | 0.0000 | 4.33 | 3.8000 | 3.8000 | 0.8315 | 0.8130 | 0.7543 | 0.8766 | 0.8428 | 0.8203 | 0.8502 | 0.05 | 1.44 | 3.54 | 350.86 | 4.53 | yes |
| `P64_beta0.9_B64_gaussian-pool` | 64 | 0.9 | 64 | gaussian-pool | 0.1250 | 4.19 | 3.6777 | 2.4203 | 0.8899 | 0.9120 | 0.7454 | 1.0012 | 0.9027 | 0.8931 | 0.9167 | 0.05 | 1.44 | 1.05 | 164.18 | 2.68 | yes |
| `P256_beta0_B4_gaussian-iid` | 256 | 0 | 4 | gaussian-iid | 0.0000 | 7.64 | 2.0000 | 1.3333 | 0.2531 | 0.2548 | 0.2249 | 0.2617 | 0.2544 | 0.2540 | 0.2552 | 0.05 | 0.53 | 0.15 | 424.01 | 5.24 | yes |
| `P256_beta0_B4_rademacher-iid` | 256 | 0 | 4 | rademacher-iid | 0.0000 | 7.64 | 2.0000 | 2.0000 | 0.2565 | 0.2548 | 0.2383 | 0.2617 | 0.2578 | 0.2570 | 0.2583 | 0.05 | 0.49 | 1.15 | 675.83 | 7.76 | yes |
| `P256_beta0_B4_gaussian-pool` | 256 | 0 | 4 | gaussian-pool | 0.0039 | 7.28 | 1.9077 | 1.2734 | 0.2528 | 0.2548 | 0.2241 | 0.2627 | 0.2534 | 0.2517 | 0.2546 | 0.05 | 0.19 | 0.68 | 401.90 | 5.03 | yes |
| `P256_beta0_B16_gaussian-iid` | 256 | 0 | 16 | gaussian-iid | 0.0000 | 1.91 | 2.0000 | 1.7778 | 0.9268 | 0.9342 | 0.6727 | 1.0467 | 0.9269 | 0.9233 | 0.9312 | 0.03 | 0.31 | 0.79 | 91.81 | 1.92 | yes |
| `P256_beta0_B16_rademacher-iid` | 256 | 0 | 16 | rademacher-iid | 0.0000 | 1.91 | 2.0000 | 2.0000 | 0.9416 | 0.9342 | 0.7022 | 1.0467 | 0.9450 | 0.9414 | 0.9462 | 0.03 | 0.37 | 1.15 | 111.63 | 2.12 | yes |
| `P256_beta0_B16_gaussian-pool` | 256 | 0 | 16 | gaussian-pool | 0.0156 | 1.85 | 1.9337 | 1.7219 | 0.9232 | 0.9315 | 0.6708 | 1.0624 | 0.9220 | 0.9183 | 0.9329 | 0.03 | 0.25 | 0.77 | 88.95 | 1.86 | yes |
| `P256_beta0.9_B4_gaussian-iid` | 256 | 0.9 | 4 | gaussian-iid | 0.0000 | 145.21 | 3.8000 | 0.3619 | 0.0260 | 0.0261 | 0.0252 | 0.0262 | 0.0261 | 0.0259 | 0.0263 | 0.05 | 0.68 | 0.26 | 1284.57 | 13.86 | yes |
| `P256_beta0.9_B4_rademacher-iid` | 256 | 0.9 | 4 | rademacher-iid | 0.0000 | 145.21 | 3.8000 | 3.8000 | 0.0263 | 0.0261 | 0.0262 | 0.0269 | 0.0264 | 0.0263 | 0.0265 | 0.05 | 0.49 | 1.13 | 14276.25 | 143.68 | yes |
| `P256_beta0.9_B4_gaussian-pool` | 256 | 0.9 | 4 | gaussian-pool | 0.0039 | 136.89 | 3.5805 | 0.3422 | 0.0260 | 0.0262 | 0.0252 | 0.0262 | 0.0262 | 0.0259 | 0.0265 | 0.05 | 0.82 | 0.49 | 1211.52 | 13.06 | yes |
| `P256_beta0.9_B16_gaussian-iid` | 256 | 0.9 | 16 | gaussian-iid | 0.0000 | 36.30 | 3.8000 | 1.1259 | 0.1034 | 0.1041 | 0.0987 | 0.1047 | 0.1039 | 0.1031 | 0.1043 | 0.05 | 0.44 | 0.22 | 984.03 | 10.85 | yes |
| `P256_beta0.9_B16_rademacher-iid` | 256 | 0.9 | 16 | rademacher-iid | 0.0000 | 36.30 | 3.8000 | 3.8000 | 0.1048 | 0.1041 | 0.1047 | 0.1052 | 0.1051 | 0.1041 | 0.1062 | 0.05 | 0.64 | 0.99 | 3514.19 | 36.15 | yes |
| `P256_beta0.9_B16_gaussian-pool` | 256 | 0.9 | 16 | gaussian-pool | 0.0156 | 35.15 | 3.6774 | 1.1017 | 0.1047 | 0.1055 | 0.0999 | 0.1062 | 0.1049 | 0.1043 | 0.1058 | 0.05 | 0.39 | 0.43 | 945.85 | 10.49 | yes |

