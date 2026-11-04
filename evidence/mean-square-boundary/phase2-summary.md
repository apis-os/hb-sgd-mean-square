# Phase 2: Hessian trace, the ceiling in R^P on the networks, and the finite-difference HVP baseline

- **Date**: `2026-10-27T16:44:16.289Z`
- **Node**: `v26.5.0`
- **Seeds per network cell**: `4`; streams `8`, steps `1000`, bisection iterations `14`, growth-ratio threshold `2.5`, additive-noise scale `tau = 0`

## 1. Pre-registered predictions (`q1`–`q4`)

- **`q1`**: Exact trace: tr(H) / lambda_1 >= 4 on both networks, so the noise-rank ratio kappa_beta tr(H) / (B lambda_1) at beta = 0.9, B = 16 is >= 4.75 and the bulk carries most of the curvature noise that Theorem G feeds into u_1.
- **`q2`**: The destructive ceiling of heavy-ball SGD with real minibatches in R^P (beta = 0.9, B = 16, homogeneous run from a start in the range of H) lies below the one-dimensional EoSS ceiling of Theorem E(3) along u_1 by a factor >= 1.5 on both networks.
- **`q3`**: The median destructive ceiling lies inside the trace bracket of the random-feature model (Theorem G with kappa_feat calibrated from nu_11, lambda_2 and tr H), widened by 10 % on each side plus the bisection half-width, on both networks.
- **`q4`**: A power iteration with central-difference Hessian-vector products from two full gradients each (15 products, 1,920 sample-gradient equivalents, no second-order code) estimates lambda_1 to a relative error below 1e-6 on both networks, so its error against the quadratic threshold is below 0.01 % and against the destructive threshold equals the quadratic-destructive gap within 0.01 points.

### Prediction outcomes

- `q1`: `true`
- `q2`: `true`
- `q3`: `false`
- `q4`: `true`

## 2. Networks (beta = 0.9, B = 16)

| Cell | `P` | `lambda_1` | `lambda_2` | `lambda_min` (shifted PI) | `tr H` | `tr H / lambda_1` | `rho_B` | `nu_B` | `kappa_feat,eff` | `alpha_diag` | `eta_quad` | `eta_destr` (ray) | gap % | 1D EoSS | sandwich lower | trace lower | trace upper | sandwich upper | empirical median | p05 | p95 | bisection ±% | upper stable | 1D / emp | quad / emp | inside |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `X1_P3610_hb09_B16` | 3610 | 98.677 | 66.190 | 0.0118 | 601.74 | 6.10 | 7.24 | 22.61 | -1.160 | 0.810 | 0.038509 | 0.038510 | 0.000 | 0.019278 | 0.004783 | 0.004918 | 0.005213 | 0.005318 | 0.002707 | 0.002665 | 0.002809 | 0.048 | 0 | 7.12 | 14.23 | no |
| `X2_P12826_hb09_B16` | 12826 | 148.353 | 77.366 | -0.0721 | 901.72 | 6.08 | 7.22 | 48.02 | -0.323 | 1.804 | 0.025615 | 0.025761 | 0.567 | 0.008564 | 0.002839 | 0.003080 | 0.003376 | 0.003549 | 0.000049 | 0.000048 | 0.000050 | 1.754 | 0 | 174.73 | 522.62 | no |

### Post-hoc growth grid (log energy growth over the window ratio; rate per step = growth / (steps / 2); no prediction registered)

| Cell | `eta / eta_quad` | `eta` | steps | log growth | rate per step | guard |
|---|---:|---:|---:|---:|---:|---|
| `X1_P3610_hb09_B16` | 0.002 | 7.702e-5 | 1000 | -0.993 | -1.987e-3 | no |
| `X1_P3610_hb09_B16` | 0.002 | 7.702e-5 | 4000 | -0.912 | -4.561e-4 | no |
| `X1_P3610_hb09_B16` | 0.005 | 1.925e-4 | 1000 | -0.774 | -1.547e-3 | no |
| `X1_P3610_hb09_B16` | 0.005 | 1.925e-4 | 4000 | -0.788 | -3.942e-4 | no |
| `X1_P3610_hb09_B16` | 0.01 | 3.851e-4 | 1000 | -0.858 | -1.716e-3 | no |
| `X1_P3610_hb09_B16` | 0.01 | 3.851e-4 | 4000 | -0.223 | -1.114e-4 | no |
| `X1_P3610_hb09_B16` | 0.02 | 7.702e-4 | 1000 | -0.878 | -1.755e-3 | no |
| `X1_P3610_hb09_B16` | 0.02 | 7.702e-4 | 4000 | 0.695 | 3.473e-4 | no |
| `X1_P3610_hb09_B16` | 0.05 | 1.925e-3 | 1000 | -0.419 | -8.372e-4 | no |
| `X1_P3610_hb09_B16` | 0.05 | 1.925e-3 | 4000 | 5.774 | 2.887e-3 | no |
| `X1_P3610_hb09_B16` | 0.1 | 3.851e-3 | 1000 | 2.695 | 5.389e-3 | no |
| `X1_P3610_hb09_B16` | 0.1 | 3.851e-3 | 4000 | guard | guard | yes |
| `X1_P3610_hb09_B16` | 0.2 | 7.702e-3 | 1000 | 5.649 | 1.130e-2 | no |
| `X1_P3610_hb09_B16` | 0.2 | 7.702e-3 | 4000 | guard | guard | yes |
| `X1_P3610_hb09_B16` | 0.5 | 1.925e-2 | 1000 | guard | guard | yes |
| `X1_P3610_hb09_B16` | 0.5 | 1.925e-2 | 4000 | guard | guard | yes |
| `X2_P12826_hb09_B16` | 0.002 | 5.123e-5 | 1000 | 1.011 | 2.022e-3 | no |
| `X2_P12826_hb09_B16` | 0.002 | 5.123e-5 | 4000 | 10.195 | 5.098e-3 | no |
| `X2_P12826_hb09_B16` | 0.005 | 1.281e-4 | 1000 | 5.989 | 1.198e-2 | no |
| `X2_P12826_hb09_B16` | 0.005 | 1.281e-4 | 4000 | guard | guard | yes |
| `X2_P12826_hb09_B16` | 0.01 | 2.561e-4 | 1000 | 11.544 | 2.309e-2 | no |
| `X2_P12826_hb09_B16` | 0.01 | 2.561e-4 | 4000 | guard | guard | yes |
| `X2_P12826_hb09_B16` | 0.02 | 5.123e-4 | 1000 | guard | guard | yes |
| `X2_P12826_hb09_B16` | 0.02 | 5.123e-4 | 4000 | guard | guard | yes |
| `X2_P12826_hb09_B16` | 0.05 | 1.281e-3 | 1000 | guard | guard | yes |
| `X2_P12826_hb09_B16` | 0.05 | 1.281e-3 | 4000 | guard | guard | yes |
| `X2_P12826_hb09_B16` | 0.1 | 2.561e-3 | 1000 | guard | guard | yes |
| `X2_P12826_hb09_B16` | 0.1 | 2.561e-3 | 4000 | guard | guard | yes |
| `X2_P12826_hb09_B16` | 0.2 | 5.123e-3 | 1000 | guard | guard | yes |
| `X2_P12826_hb09_B16` | 0.2 | 5.123e-3 | 4000 | guard | guard | yes |
| `X2_P12826_hb09_B16` | 0.5 | 1.281e-2 | 1000 | guard | guard | yes |
| `X2_P12826_hb09_B16` | 0.5 | 1.281e-2 | 4000 | guard | guard | yes |

## 3. Finite-difference HVP power iteration (two full gradients per product, no second-order code)

| Cell | `P` | products | sample-grad equiv. | `lambda_1` hat | rel. err | cos with analytic `u_1` | `eta_crit` hat (0.9) | err vs quadratic % | err vs destructive % | trace s | ceiling s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `X1_P3610_hb09_B16` | 3610 | 15 | 1920 | 98.676935 | 8.43e-7 | 0.99999942 | 0.038510 | 0.0001 | 0.0002 | 14.3 | 561.9 |
| `X2_P12826_hb09_B16` | 12826 | 15 | 1920 | 148.353132 | 1.60e-8 | 1.00000000 | 0.025615 | 0.0000 | 0.5668 | 160.8 | 741.5 |


