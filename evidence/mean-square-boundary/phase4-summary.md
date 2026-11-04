# Phase 4: the exact mean-square threshold of Theorem G.6 on tethered network Hessians

- **Date**: `2026-10-28T09:25:09.041Z`
- **Node**: `v26.5.0`
- **Batch**: `B = 16`, tether grid `g` = 128, 32, 8, 2, 0.5, 0.125, 0.03125, 0.0078125, `mu = g + max(0, -lambda_min)`

## 1. Predictions (`t1`–`t6`, written before the run)

- **`t1`**: Theorem G.6 is exact on network Hessians: in every operator cell (P = 33: beta in {0, 0.9} and every tether; P = 3,610: beta = 0.9, g = 2) the spectral radius of the exact second-moment operator T is below 1 at (1 - 1e-3) eta* and above 1 at (1 + 1e-3) eta*, with eta* = 1 / lambda_max(L_0^{-1/2} B_beta L_0^{-1/2}) by Lanczos.
- **`t2`**: The mean Hessian of the P = 3,610 network is indefinite: the smallest eigenvalue of its full eigendecomposition is negative (the shifted power iteration of Phase 2 gave only the upper bound 0.012; the Gauss-Newton part has rank at most 640 and the residual part of the squared loss is indefinite on its null space).
- **`t3`**: Mean-square stability implies almost-sure stability: in every cell the median sample-path ceiling of the linearised iteration is at least (1 - 0.02) eta* minus the largest bisection half-width.
- **`t4`**: Linearisation: in every cell the median sample-path ceiling of the network iteration (start 1e-6 from the checkpoint) is within 2 % plus the two largest bisection half-widths of the median ceiling of the linearised iteration.
- **`t5`**: Flat-mode collapse at P = 3,610: eta* decreases from g = 1/32 to g = 1/128, and the log-log slope of eta* against the smallest tethered eigenvalue between these two grid values lies in [0.4, 1.1] (the flat-mode bounds of Theorem G.5 give O(m) when the T_flat bound applies and O(sqrt(m)) from the pairwise bound otherwise; the interval allows for a grid that is not yet asymptotic).
- **`t6`**: The fraction f* = eta* (lambda_1 + mu) / (2 (1 + beta)) of the deterministic threshold is non-decreasing in mu along the grid, on both networks and for every beta (at fixed f every weight of the Rayleigh quotient of Theorem G.6(2) grows with mu and the numerator does not depend on mu). eta* itself need not be monotone; its interior maxima are reported, not predicted.

### Outcomes

- `t1`: `true`
- `t2`: `true`
- `t3`: `true`
- `t4`: `true`
- `t5`: `false`
- `t6`: `true`

### Implementation checks (added after the registration; consequences of the theorem, not predictions)

- **`c1`** (`true`): For consecutive grid values mu1 < mu2 with eta*(mu1) <= eta_det(mu2) / 2, eta*(mu2) >= eta*(mu1) (below half the deterministic threshold every weight of Theorem G.6(2) grows with mu at fixed eta).
- **`c3`** (`false`): Every operator cell left undecided by its certificates at 1e-3 is decided at the fallback offset 1e-2 (rule fixed before the P = 3,610 run).
- **`c2`** (`false`): Every operator cell is decided by Collatz-Wielandt certificates at the last iterate: Q - T(Q) > 0 at (1 - 1e-3) eta* and T(Q) - Q > 0 at (1 + 1e-3) eta* (P = 33: from the full bracket [lambda_min, lambda_max] of Q^{-1/2} T(Q) Q^{-1/2}; P = 3,610: by Cholesky).

## N33_mlp (P = 33, N = 64)

- Hessians: dense, central differences of per-sample gradients (eps 1e-5), symmetrised
- `lambda_1 = 5.762792099753815`, `lambda_2 = 2.9966395652833664`, `lambda_min = -0.008014736055870856`, 2 negative eigenvalues, second smallest `-0.006346927629161653`, 9 eigenvalues below `2 |lambda_min|`, `tr H = 17.523406916813038`, `nu_11,B^2 = 3.1045450128019456`

| beta | g | mu | lambda_min(H + mu) | eta_det | 1D ceiling | eta* (Thm G.6) | f* = eta* / eta_det | 1D / eta* | Lanczos steps | linearised / eta* | network / eta* | horizon informative |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 128 | 128.008 | 128.000 | 0.0149509 | 0.0149484 | 0.014948331 | 0.999825 | 1.000 | 16 | 1.001 | 1.001 | yes |
| 0 | 32 | 32.0080 | 32.0000 | 0.0529509 | 0.0528360 | 0.052832710 | 0.997767 | 1.000 | 17 | 1.004 | 1.004 | yes |
| 0 | 8 | 8.00801 | 8.00000 | 0.145235 | 0.142895 | 0.14270913 | 0.982610 | 1.001 | 17 | 1.030 | 1.030 | yes |
| 0 | 2 | 2.00801 | 2.00000 | 0.257374 | 0.244788 | 0.24286101 | 0.943613 | 1.008 | 17 | 1.107 | 1.107 | yes |
| 0 | 0.5 | 0.508015 | 0.500000 | 0.318938 | 0.295601 | 0.29055073 | 0.910994 | 1.017 | 17 | 1.162 | 1.162 | yes |
| 0 | 0.125 | 0.133015 | 0.125000 | 0.339224 | 0.311411 | 0.30429984 | 0.897047 | 1.023 | 17 | 1.156 | 1.156 | yes |
| 0 | 0.03125 | 0.0392647 | 0.0312500 | 0.344705 | 0.315600 | 0.30763346 | 0.892453 | 1.026 | 17 | 1.135 | 1.135 | yes |
| 0 | 0.0078125 | 0.0158272 | 0.00781250 | 0.346103 | 0.316663 | 0.30841041 | 0.891093 | 1.027 | 17 | 1.124 | 1.124 | yes |
| 0.9 | 128 | 128.008 | 128.000 | 0.0284068 | 0.0283135 | 0.028307600 | 0.996508 | 1.000 | 26 | 1.003 | 1.003 | yes |
| 0.9 | 32 | 32.0080 | 32.0000 | 0.100607 | 0.0966122 | 0.096014899 | 0.954358 | 1.006 | 42 | 1.021 | 1.021 | yes |
| 0.9 | 8 | 8.00801 | 8.00000 | 0.275946 | 0.210477 | 0.19406782 | 0.703282 | 1.085 | 59 | 1.104 | 1.104 | yes |
| 0.9 | 2 | 2.00801 | 2.00000 | 0.489010 | 0.247371 | 0.18081066 | 0.369749 | 1.368 | 70 | 1.190 | 1.190 | yes |
| 0.9 | 0.5 | 0.508015 | 0.500000 | 0.605983 | 0.242389 | 0.13958924 | 0.230352 | 1.736 | 74 | 1.143 | 1.143 | yes |
| 0.9 | 0.125 | 0.133015 | 0.125000 | 0.644526 | 0.238985 | 0.12098939 | 0.187718 | 1.975 | 73 | 1.145 | 1.145 | yes |
| 0.9 | 0.03125 | 0.0392647 | 0.0312500 | 0.654940 | 0.237968 | 0.11450515 | 0.174833 | 2.078 | 73 | 1.166 | 1.166 | yes |
| 0.9 | 0.0078125 | 0.0158272 | 0.00781250 | 0.657597 | 0.237703 | 0.11242063 | 0.170957 | 2.114 | 73 | 1.183 | 1.183 | yes |

| beta | g | eta* | rho(T) at (1 - d) eta* | rho(T) at (1 + d) eta* | 1 - rho below | rho above - 1 | steps | converged | crosses | certificate below | certificate above | horizon (multiplier: rho, steps to 2.5x) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 128 | 0.014948331 | 0.9960047042 | 1.0040032944 | 4.00e-3 | 4.00e-3 | 264, 117 | yes | yes | decided [0.9960045596, 0.9960047928] | decided [1.004003197, 1.004003401] | 1: 1.0000000000, inf; 1.1: 1.4399225113, 3; 1.5: 3.9994723608, 1 |
| 0 | 32 | 0.052832710 | 0.9960131722 | 1.0039948099 | 3.99e-3 | 3.99e-3 | 124, 85 | yes | yes | decided [0.9960131721, 0.9960131723] | decided [1.003994810, 1.003994810] | 1: 1.0000000000, inf; 1.1: 1.4389923902, 3; 1.5: 3.9931613872, 1 |
| 0 | 8 | 0.14270913 | 0.9960790855 | 1.0039287767 | 3.92e-3 | 3.93e-3 | 62, 50 | yes | yes | decided [0.9960790855, 0.9960790855] | decided [1.003928777, 1.003928777] | 1: 1.0000000000, 1031651649657872; 1.1: 1.4317898169, 3; 1.5: 3.9447693406, 1 |
| 0 | 2 | 0.24286101 | 0.9962720453 | 1.0037355634 | 3.73e-3 | 3.74e-3 | 45, 39 | yes | yes | decided [0.9962720453, 0.9962720453] | decided [1.003735563, 1.003735563] | 1: 1.0000000000, inf; 1.1: 1.4111094827, 3; 1.5: 3.8099448903, 1 |
| 0 | 0.5 | 0.29055073 | 0.9965029446 | 1.0035051113 | 3.50e-3 | 3.51e-3 | 83, 54 | yes | yes | decided [0.9965029303, 0.9965029446] | decided [1.003505110, 1.003505111] | 1: 1.0000000000, 1031651649657872; 1.1: 1.3887893228, 3; 1.5: 3.6790819625, 1 |
| 0 | 0.125 | 0.30429984 | 0.9967123004 | 1.0032990288 | 3.29e-3 | 3.30e-3 | 265, 172 | yes | yes | decided [0.9967121519, 0.9967123004] | decided [1.003299006, 1.003299029] | 1: 1.0000000000, 4197972124753; 1.1: 1.3741401248, 3; 1.5: 3.6098984393, 1 |
| 0 | 0.03125 | 0.30763346 | 0.9968824153 | 1.0031393510 | 3.12e-3 | 3.14e-3 | 986, 554 | yes | yes | decided [0.9968822794, 0.9968824153] | decided [1.003139223, 1.003139351] | 1: 1.0000000000, 293062040951; 1.1: 1.3670978747, 3; 1.5: 3.5825895188, 1 |
| 0 | 0.0078125 | 0.30841041 | 0.9970692703 | 1.0030174109 | 2.93e-3 | 3.02e-3 | 6532, 1751 | yes | yes | decided [0.9970692576, 0.9970692710] | decided [1.003017341, 1.003017411] | 1: 1.0000000000, 49439977460; 1.1: 1.3644475121, 3; 1.5: 3.5734980572, 1 |
| 0.9 | 128 | 0.028307600 | 0.9814577790 | 1.0239395833 | 1.85e-2 | 2.39e-2 | 315, 203 | yes | yes | decided [0.9814577559, 0.9814577961] | decided [1.023939583, 1.023939583] | 1: 1.0000000000, inf; 1.1: 3.0741390175, 1; 1.5: 12.4304768845, 0 |
| 0.9 | 32 | 0.096014899 | 0.9979396184 | 1.0021458265 | 2.06e-3 | 2.15e-3 | 311, 247 | yes | yes | decided [0.9979396183, 0.9979396185] | decided [1.002145826, 1.002145826] | 1: 1.0000000000, inf; 1.1: 2.2956040975, 1; 1.5: 10.7331225673, 0 |
| 0.9 | 8 | 0.19406782 | 0.9996836805 | 1.0003177831 | 3.16e-4 | 3.18e-4 | 383, 286 | yes | yes | decided [0.9996836805, 0.9996836805] | decided [1.000317783, 1.000317783] | 1: 1.0000000000, inf; 1.1: 1.0413797271, 23; 1.5: 2.9251910502, 1 |
| 0.9 | 2 | 0.18081066 | 0.9998504067 | 1.0001497432 | 1.50e-4 | 1.50e-4 | 419, 302 | yes | yes | decided [0.9998504067, 0.9998504067] | decided [1.000149743, 1.000149743] | 1: 1.0000000000, 1242953794769; 1.1: 1.0157592764, 59; 1.5: 1.1008827039, 10 |
| 0.9 | 0.5 | 0.13958924 | 0.9998741874 | 1.0001258758 | 1.26e-4 | 1.26e-4 | 433, 275 | yes | yes | decided [0.9998741874, 0.9998741874] | decided [1.000125876, 1.000125876] | 1: 1.0000000000, inf; 1.1: 1.0129075123, 71; 1.5: 1.0718239772, 13 |
| 0.9 | 0.125 | 0.12098939 | 0.9998804772 | 1.0001195695 | 1.20e-4 | 1.20e-4 | 408, 253 | yes | yes | decided [0.9998804772, 0.9998804772] | decided [1.000119569, 1.000119570] | 1: 1.0000000000, inf; 1.1: 1.0121919231, 76; 1.5: 1.0661113877, 14 |
| 0.9 | 0.03125 | 0.11450515 | 0.9998832573 | 1.0001167845 | 1.17e-4 | 1.17e-4 | 432, 275 | yes | yes | decided [0.9998832573, 0.9998832573] | decided [1.000116785, 1.000116785] | 1: 1.0000000000, 1339807337218; 1.1: 1.0118881718, 78; 1.5: 1.0639795559, 15 |
| 0.9 | 0.0078125 | 0.11242063 | 0.9998851534 | 1.0001148925 | 1.15e-4 | 1.15e-4 | 1203, 782 | yes | yes | decided [0.9998851532, 0.9998851534] | decided [1.000114892, 1.000114893] | 1: 1.0000000000, 252731908295; 1.1: 1.0117097011, 79; 1.5: 1.0630137843, 15 |

- Interior maxima of eta* along the grid, beta 0: none
- Interior maxima of eta* along the grid, beta 0.9: g = 8
- Collapse slope (log eta* against log lambda_min(H + mu), two smallest g), beta 0: `-0.001819521517942065`
- Collapse slope (log eta* against log lambda_min(H + mu), two smallest g), beta 0.9: `0.013252859494244734`

## X1_P3610 (P = 3610, N = 64)

- Hessians: exact per-sample Pearlmutter products
- `lambda_1 = 98.67701845430254`, `lambda_2 = 66.19049703571315`, `lambda_min = -0.08333001843178042`, 1528 negative eigenvalues, second smallest `-0.07498385337392427`, 3435 eigenvalues below `2 |lambda_min|`, `tr H = 601.7390949925793`, `nu_11,B^2 = 511.2370809878354`

| beta | g | mu | lambda_min(H + mu) | eta_det | 1D ceiling | eta* (Thm G.6) | f* = eta* / eta_det | 1D / eta* | Lanczos steps | linearised / eta* | network / eta* | horizon informative |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.9 | 128 | 128.083 | 128.000 | 0.0167578 | 0.0140951 | 0.013993390 | 0.835038 | 1.007 | 55 | 1.081 | 1.081 | yes |
| 0.9 | 32 | 32.0833 | 32.0000 | 0.0290608 | 0.0185325 | 0.017880637 | 0.615284 | 1.036 | 73 | 1.196 | 1.202 | yes |
| 0.9 | 8 | 8.08333 | 8.00000 | 0.0355937 | 0.0192167 | 0.017637855 | 0.495533 | 1.090 | 86 | 1.251 | 1.251 | yes |
| 0.9 | 2 | 2.08333 | 2.00000 | 0.0377132 | 0.0192735 | 0.016781922 | 0.444987 | 1.148 | 104 | 1.257 | 1.257 | yes |
| 0.9 | 0.5 | 0.583330 | 0.500000 | 0.0382832 | 0.0192777 | 0.016155850 | 0.422009 | 1.193 | 90 | 1.235 | 1.235 | yes |
| 0.9 | 0.125 | 0.208330 | 0.125000 | 0.0384283 | 0.0192781 | 0.015823464 | 0.411765 | 1.218 | 78 | 1.202 | 1.202 | yes |
| 0.9 | 0.03125 | 0.114580 | 0.0312500 | 0.0384648 | 0.0192781 | 0.015689409 | 0.407890 | 1.229 | 72 | 1.222 | 1.222 | yes |
| 0.9 | 0.0078125 | 0.0911425 | 0.00781250 | 0.0384739 | 0.0192781 | 0.015640812 | 0.406530 | 1.233 | 86 | 1.204 | 1.204 | yes |

| beta | g | eta* | rho(T) at (1 - d) eta* | rho(T) at (1 + d) eta* | 1 - rho below | rho above - 1 | steps | converged | crosses | certificate below | certificate above | horizon (multiplier: rho, steps to 2.5x) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|
| 0.9 | 2 | 0.016781922 | 0.9998242732 | 1.0001760008 | 1.76e-4 | 1.76e-4 | 439, 237 | yes | yes | Q not PD; not decided | Q not PD; not decided | ; fallback ±0.01: 0.99825495 / 1.00177245, Q not PD; not decided / Q not PD; not decided |

- Interior maxima of eta* along the grid, beta 0.9: g = 32
- Collapse slope (log eta* against log lambda_min(H + mu), two smallest g), beta 0.9: `0.002237794145845624`
