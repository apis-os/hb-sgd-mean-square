# Phase 3: the secular root of Theorem G against the exact second-moment operator

- **Date**: `2026-10-27T16:42:13.033Z`
- **Node**: `v24.18.0`
- **Power iteration**: from the identity on the symmetric part, relative tolerance `1e-13`, at most `200000` steps; probes at `(1 ± 0.001) eta*`.
- **Provenance**: A 4-cell prototype (P = 64 and 256, beta = 0 and 0.9) was run before q1-q3 were written; it gave rho(eta*) = 1 to 12 digits, so q1 and q2 were informed by it. After the predictions, a P = 16 debugging run showed that the ratio of norms does not converge on the antisymmetric part at beta = 0.9 (a cluster of eigenvalues of equal modulus), and the antisymmetric estimator was changed to the Gelfand estimate over a fixed horizon before the full run; that run gave antisymmetric radii of 0.44-0.61 at beta = 0 and about 0.906 at beta = 0.9, so q3 was seen at P = 16 before the full run. No P = 64, 256 or 1024 cell was run with the final code before the full run.

## 1. Predictions (`q1`–`q3`)

- **`q1`**: In every cell the radius of the second-moment operator is below 1 at (1 - 1e-3) eta* and above 1 at (1 + 1e-3) eta*, eta* the secular root of Theorem G.
- **`q2`**: In every cell the crossing rho = 1, located from rho(eta*) - 1 and the local slope, is within 1e-8 (relative) of eta*.
- **`q3`**: In every cell the radius on the antisymmetric matrices at eta* is strictly below 1, the radius on the symmetric part there (Lemma: the antisymmetric part never dominates).

### Outcomes

- `q1RadiusCrossesOneAtSecularRootEveryCell`: `true`
- `q2CrossingWithin1e8RelativeEveryCell`: `true`
- `q3AntisymmetricRadiusBelowOneAtRootEveryCell`: `true`

## 2. Cells

| Cell | `P` | `beta` | `B` | Family | `eta*` (secular) | `rho` at `(1-d) eta*` | `rho` at `eta*` | `rho` at `(1+d) eta*` | antisymmetric `rho` at `eta*` | crossing rel. err | iterations | converged |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `P64_b0_B4_gaussian-iid` | 64 | 0 | 4 | gaussian-iid | 0.4896505550 | 0.999758343 | 1.000000000000 | 1.000242305 | 0.879803 | 1.85e-13 | 117 | yes |
| `P64_b0_B4_rademacher-iid` | 64 | 0 | 4 | rademacher-iid | 0.5123439917 | 0.999755961 | 1.000000000000 | 1.000244668 | 0.873901 | 1.54e-13 | 111 | yes |
| `P64_b0_B16_gaussian-iid` | 64 | 0 | 16 | gaussian-iid | 1.446768523 | 0.998912645 | 1.000000000000 | 1.001092332 | 0.667777 | 8.15e-16 | 62 | yes |
| `P64_b0_B16_rademacher-iid` | 64 | 0 | 16 | rademacher-iid | 1.529890476 | 0.998963535 | 1.000000000000 | 1.001040652 | 0.650568 | 8.55e-16 | 58 | yes |
| `P64_b0_B64_gaussian-iid` | 64 | 0 | 64 | gaussian-iid | 1.926840974 | 0.996289680 | 1.000000000000 | 1.003719488 | 0.710858 | 4.18e-16 | 57 | yes |
| `P64_b0_B64_rademacher-iid` | 64 | 0 | 64 | rademacher-iid | 1.986074929 | 0.996187812 | 1.000000000000 | 1.003821771 | 0.748973 | 1.16e-16 | 57 | yes |
| `P64_b0.9_B4_gaussian-iid` | 64 | 0.9 | 4 | gaussian-iid | 0.05332913348 | 0.999925767 | 1.000000000000 | 1.000074240 | 0.901909 | 5.11e-13 | 175 | no |
| `P64_b0.9_B4_rademacher-iid` | 64 | 0.9 | 4 | rademacher-iid | 0.05575715604 | 0.999925376 | 1.000000000000 | 1.000074630 | 0.901701 | 1.70e-13 | 185 | no |
| `P64_b0.9_B16_gaussian-iid` | 64 | 0.9 | 16 | gaussian-iid | 0.2103284051 | 0.999903014 | 1.000000000000 | 1.000096992 | 0.901765 | 8.93e-14 | 185 | no |
| `P64_b0.9_B16_rademacher-iid` | 64 | 0.9 | 16 | rademacher-iid | 0.2199182232 | 0.999902867 | 1.000000000000 | 1.000097140 | 0.902232 | 6.86e-14 | 187 | no |
| `P64_b0.9_B64_gaussian-iid` | 64 | 0.9 | 64 | gaussian-iid | 0.7947906867 | 0.999888984 | 1.000000000000 | 1.000111044 | 0.902384 | 3.20e-14 | 205 | no |
| `P64_b0.9_B64_rademacher-iid` | 64 | 0.9 | 64 | rademacher-iid | 0.8314581528 | 0.999889064 | 1.000000000000 | 1.000110964 | 0.902514 | 1.07e-13 | 208 | no |
| `P256_b0_B4_gaussian-iid` | 256 | 0 | 4 | gaussian-iid | 0.2530896285 | 0.999937848 | 0.999999999999 | 1.000062316 | 0.968195 | 2.09e-11 | 282 | yes |
| `P256_b0_B4_rademacher-iid` | 256 | 0 | 4 | rademacher-iid | 0.2565363702 | 0.999937773 | 0.999999999999 | 1.000062389 | 0.967735 | 2.00e-11 | 278 | yes |
| `P256_b0_B16_gaussian-iid` | 256 | 0 | 16 | gaussian-iid | 0.9267511681 | 0.999746848 | 1.000000000000 | 1.000253925 | 0.886973 | 2.30e-13 | 121 | yes |
| `P256_b0_B16_rademacher-iid` | 256 | 0 | 16 | rademacher-iid | 0.9415852230 | 0.999748723 | 1.000000000000 | 1.000252018 | 0.885169 | 2.15e-13 | 119 | yes |
| `P256_b0_B64_gaussian-iid` | 256 | 0 | 64 | gaussian-iid | 1.903564779 | 0.997093887 | 1.000000000000 | 1.002931072 | 0.799565 | 1.22e-15 | 99 | yes |
| `P256_b0_B64_rademacher-iid` | 256 | 0 | 64 | rademacher-iid | 1.959448970 | 0.997097194 | 1.000000000000 | 1.002929365 | 0.845189 | 6.09e-16 | 99 | yes |
| `P256_b0.9_B4_gaussian-iid` | 256 | 0.9 | 4 | gaussian-iid | 0.02596103679 | 0.999960775 | 0.999999999999 | 1.000039258 | 0.960142 | 2.29e-11 | 308 | yes |
| `P256_b0.9_B4_rademacher-iid` | 256 | 0.9 | 4 | rademacher-iid | 0.02630405832 | 0.999960714 | 0.999999999999 | 1.000039318 | 0.959448 | 2.29e-11 | 300 | yes |
| `P256_b0.9_B16_gaussian-iid` | 256 | 0.9 | 16 | gaussian-iid | 0.1034104751 | 0.999925538 | 1.000000000000 | 1.000074470 | 0.900164 | 2.30e-13 | 177 | yes |
| `P256_b0.9_B16_rademacher-iid` | 256 | 0.9 | 16 | rademacher-iid | 0.1047830605 | 0.999925492 | 1.000000000000 | 1.000074516 | 0.900172 | 3.67e-13 | 177 | yes |
| `P256_b0.9_B64_gaussian-iid` | 256 | 0.9 | 64 | gaussian-iid | 0.4066875541 | 0.999902549 | 1.000000000000 | 1.000097459 | 0.900123 | 1.00e-13 | 188 | yes |
| `P256_b0.9_B64_rademacher-iid` | 256 | 0.9 | 64 | rademacher-iid | 0.4121953925 | 0.999902570 | 1.000000000000 | 1.000097437 | 0.900131 | 1.80e-13 | 192 | yes |
| `P1024_b0_B4_gaussian-iid` | 1024 | 0 | 4 | gaussian-iid | 0.1266493841 | 0.999984393 | 0.999999999992 | 1.000015648 | 0.991712 | 5.16e-10 | 638 | yes |
| `P1024_b0_B4_rademacher-iid` | 1024 | 0 | 4 | rademacher-iid | 0.1271470163 | 0.999984395 | 0.999999999992 | 1.000015646 | 0.991687 | 5.19e-10 | 634 | yes |
| `P1024_b0_B16_gaussian-iid` | 1024 | 0 | 16 | gaussian-iid | 0.4942770596 | 0.999937344 | 0.999999999999 | 1.000062827 | 0.968921 | 2.17e-11 | 287 | yes |
| `P1024_b0_B16_rademacher-iid` | 1024 | 0 | 16 | rademacher-iid | 0.4963837962 | 0.999937409 | 0.999999999999 | 1.000062761 | 0.968783 | 2.05e-11 | 287 | yes |
| `P1024_b0_B64_gaussian-iid` | 1024 | 0 | 64 | gaussian-iid | 1.684763773 | 0.999641402 | 1.000000000000 | 1.000361469 | 0.897046 | 2.25e-13 | 134 | yes |
| `P1024_b0_B64_rademacher-iid` | 1024 | 0 | 64 | rademacher-iid | 1.705176184 | 0.999661435 | 1.000000000000 | 1.000340928 | 0.895841 | 2.21e-13 | 132 | yes |
| `P1024_b0.9_B4_gaussian-iid` | 1024 | 0.9 | 4 | gaussian-iid | 0.01275892528 | 0.999986402 | 0.999999999992 | 1.000013624 | 0.991342 | 6.04e-10 | 912 | yes |
| `P1024_b0.9_B4_rademacher-iid` | 1024 | 0.9 | 4 | rademacher-iid | 0.01280804120 | 0.999986401 | 0.999999999992 | 1.000013624 | 0.991307 | 6.10e-10 | 906 | yes |
| `P1024_b0.9_B16_gaussian-iid` | 1024 | 0.9 | 16 | gaussian-iid | 0.05097339203 | 0.999960737 | 0.999999999999 | 1.000039296 | 0.961199 | 2.37e-11 | 313 | yes |
| `P1024_b0.9_B16_rademacher-iid` | 1024 | 0.9 | 16 | rademacher-iid | 0.05117025909 | 0.999960737 | 0.999999999999 | 1.000039296 | 0.960972 | 2.45e-11 | 310 | yes |
| `P1024_b0.9_B64_gaussian-iid` | 1024 | 0.9 | 64 | gaussian-iid | 0.2028910920 | 0.999925436 | 1.000000000000 | 1.000074572 | 0.900007 | 7.25e-13 | 165 | yes |
| `P1024_b0.9_B64_rademacher-iid` | 1024 | 0.9 | 64 | rademacher-iid | 0.2036854605 | 0.999925442 | 1.000000000000 | 1.000074566 | 0.900007 | 2.89e-13 | 178 | yes |

