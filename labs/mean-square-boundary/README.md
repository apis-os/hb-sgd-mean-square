# Phases 1–6: the mean-square stability boundary of heavy-ball SGD

The six phases that the extended version `paper/excerpt-second-moment.tex` reports (Phase 5b is the replication of Phase 5); the shorter
note `paper/note-secular-heavy-ball.tex` reports Phases 1 and 3. Code comments and prediction texts use the theorem letters of a
companion manuscript; the two papers number them as follows.

| Code and raw files | Extended version (`excerpt-second-moment`) | Shorter note (`note-secular-heavy-ball`) |
|---|---|---|
| Theorem E(3) (one-dimensional ceiling) | Proposition 1 | equation (11) |
| Theorem F | Theorem 2 | Theorem 1 |
| the second-moment operator is decided on the symmetric matrices | Lemma 3 | Lemma 2 (stated on symmetric matrices) |
| mean-square instability criterion | Corollary 4 | Lemma 2 |
| Theorem G | Theorem 5 | Theorem 5 |
| renewal equation of the expected loss | Proposition 11 | Proposition 6 |
| Proposition G.3 | Proposition 7 | not in the note |
| Theorem G.5 | Theorem 8 | not in the note |
| Theorem G.6 | Theorem 9 | Theorem 3 |
| effective step at the threshold increasing in beta | Corollary 10 | Corollary 4 |
| Theorem diag-threshold, Proposition diag-expansion | Theorem 20, Proposition 21 | not in the note |
| Theorem splitting (Krein–Rutman witness) | Theorem 24 | not in the note |

Phases 4, 5, 5b and 6 were run in the companion lab as its Phases 13, 14, 14b and 16; their code and raw files were ported here
(see *Provenance of Phases 4–6* at the end). "lab Phase N" in a comment refers to a phase of that lab that is not
part of the extended version. Proposition E13 is a statement of the companion manuscript; the increment-stream block `e13` that tests it is
not computed in this repository.

## Phase 1 — the stochastic ceiling in $\mathbb{R}^P$: Gaussian against Rademacher features (Theorem G; predictions written before the run)

Raw: `evidence/mean-square-boundary/raw-phase1.json`, `phase1-summary.md` (written by `run-phase1.ts`). Theorem G gives the exact mean-square ceiling of heavy-ball SGD on a quadratic with random-feature minibatches $H_B = \frac{1}{B}\sum z z^\top$, $z = H^{1/2}\omega$, as the root of a rank-one secular equation, with sandwich bounds, and says that for Rademacher features ($\omega \in \{\pm 1\}$, $\nu_{11}^2 = 0$) the one-dimensional ceiling of Theorem E(3) along $u_1$ is $2(1+\beta)/\lambda_1$ for every $B$ while the true ceiling is bounded by the sandwich: at most $2(1+\beta)/(\lambda_1+\kappa_\beta\operatorname{tr}(H_\perp)/B)$ for $B \le \kappa_\beta$ and $2(1+\beta)/\max(\lambda_1, \kappa_\beta\operatorname{tr}(H)/B)$ for $B > \kappa_\beta$, with $\kappa_\beta = (1+\beta)/(1-\beta)$ and $\operatorname{tr}(H_\perp) = \operatorname{tr}(H) - \lambda_1$ (Corollary 6 of the extended version). (The registered text had $2(1-\beta)B/\operatorname{tr}(H)$, which is not an upper bound when $B < \kappa_\beta$: at $P = 64$, $\beta = 0.9$, $B = 4$ the root 0.0558 exceeds it, 0.0548.)

Design. Spectrum $\lambda_k = k^{-1/2}$, $P \in \{64, 256\}$, $\beta \in \{0, 0.9\}$, $B \in \{4, 16, 64\}$ ($B \le 16$ at $P = 256$), additive noise $\sigma_\xi = 0.01$ isotropic. Three families: Gaussian i.i.d. features ($\kappa_{\text{feat}} = 0$), Rademacher i.i.d. features ($\kappa_{\text{feat}} = -2$, $\omega^4 \equiv 1$), and a finite pool of $n$ Gaussian features ($n = 512$ at $P = 64$, $1024$ at $P = 256$) sampled without replacement, the model of Lee, Cheng, Paquette and Paquette (2022), whose kernel-norm root carries the factor $1 - \zeta$, $\zeta = B/n$; in the pool arm the prediction uses the eigenvalues of the pool covariance (cyclic Jacobi). The empirical ceiling is found by bisection on $\eta$ (12 iterations, bracket expanded upward from $4\times$ the predicted root until unstable, capped at $1.02\times$ the deterministic ceiling); a learning rate is unstable when the mean-square energy over 8 streams grows by more than $2.5\times$ between the second and the last quarter of 1000 steps. 8 seeds per cell; the bisection half-width is recorded per cell.

Predictions (registered before the run):
- `p1` (Theorem G(3)): in every i.i.d. cell the median empirical ceiling lies inside the sandwich $\big[\frac{2(1+\beta)}{\alpha\lambda_1 + \kappa_\beta\operatorname{tr}(H)/B},\ \frac{2(1+\beta)}{\max(\alpha\lambda_1 + \kappa_\beta\lambda_1/B,\ \kappa_\beta\operatorname{tr}(H)/B)}\big]$, within 3 % plus the bisection half-width.
- `p2` (Theorem G(2)): in every i.i.d. cell the secular root predicts the empirical ceiling with median relative error below 5 %.
- `p3` (Theorem G(4)): in every Rademacher cell with $\kappa_\beta\operatorname{tr}(H)/(B\lambda_1) \ge 1$ the one-dimensional ceiling exceeds the empirical one by more than a factor 2 while the secular root stays within 5 %.
- `p4` (Remark G.1): in the pool arm with $\zeta \le 1/8$ the Lee et al. root (factor $1-\zeta$, no kurtosis term) predicts the empirical ceiling with median relative error below 8 %.

Results (`raw-phase1.json`, 2026-10-27, 43 min single thread, Node 26.5). `p3 = true`, `p4 = true`, `p1 = false`, `p2 = false`, the last two on one cell. In 29 of 30 cells the median empirical ceiling is inside the sandwich within 3 % plus the bisection half-width, strictly inside in 27 (the Rademacher cell at $P = 64$, $\beta = 0$, $B = 64$ is 0.02 % above its upper bound 2 and the pool cell at the same $(P, \beta, B)$ 2.85 % above its seed-averaged upper bound) (`p1` is scored with the sandwich of Theorem G, which has the terms $\max(\alpha_{\mathrm{diag}},0)$ and $\min(\alpha_{\mathrm{diag}},0)$ that the registered text lacks; a deviation written after the registration, and `p1` fails under either form) and the secular root predicts it with median error 0.19–3.82 %, the upper end from the pool cell at $P = 64$, $\beta = 0$, $B = 64$ (0.19–2.65 % in the other 28) (Lee root without the kurtosis term: 0.15–4.26 %); median bisection half-width over seeds ≤ 0.05 % in every cell. Rademacher separation at $\beta = 0.9$: the one-dimensional ceiling $2(1+\beta)/\lambda_1 = 3.8$ ($\nu_{11}^2 = 0$) exceeds the measured ceiling by 68×, 17×, 4.5× ($P = 64$, $B = 4, 16, 64$) and 144×, 36× ($P = 256$, $B = 4, 16$), with the secular root within 1.5 % in all five cells. The failed cell is $P = 64$, $\beta = 0$, $B = 64$, Gaussian: the measured ceiling (2.0399, at the bisection cap $1.02 \times 2/\lambda_1$) is above the deterministic ceiling, which no mean-square ceiling can be; the diagnosis (`p9-diag`, log energy ratio $-2.5$ to $-4.6$ at $\eta = 2.00, 2.02, 2.04$ and $+24$ at $2.10$) is the gap between mean-square and almost-sure stability of the random-coefficient recursion along $u_1$ when the multiplicative noise there dominates: the typical path decays while the second moment grows through rare paths that 8 streams do not sample. The Rademacher cell at the same $(P, \beta, B)$ has $\nu_{11}^2 = 0$ and matches within 0.7 %. Post-hoc diagnostic (`evidence/mean-square-boundary/phase1-diag-as-vs-ms.json`, not a pre-registered prediction): the same cell rerun with the bisection cap raised to $2.3$ gives a sample-path threshold of median 2.0032 (range 1.9628–2.0667) for Gaussian features and 1.9999 (1.9763–2.0128) for Rademacher, against the one-dimensional almost-sure root $2.0688$ (exact, quadrature) and one-dimensional mean-square ceiling $1.9394$ (secular root $1.9268$) for Gaussian, and $2.0$ for both in the Rademacher case (Proposition G.3): the Gaussian median sits between the mean-square and the one-dimensional almost-sure thresholds, near the middle of that interval rather than at the almost-sure root, and the eight seeds spread over most of it. Written by `run-phase1-cap-diagnostic.ts` (Phase 1 streams, steps and threshold; 12 bisection steps on $[0, 2.3]$; seeds 0–7). A harness that estimates $\mathbb{E}\|w_t\|^2$ rather than the sample-path energy would be needed to measure Theorem G in that regime.


## Phase 2 — Hessian trace, the ceiling in $\mathbb{R}^P$ on the networks, and a first-order Hessian baseline (predictions written before the run)

Raw: `evidence/mean-square-boundary/raw-phase2.json`, `phase2-summary.md` (written by `run-phase2.ts`). The questions: the trace and noise-rank ratio $\rho_B = \kappa_\beta\operatorname{tr}H/(B\lambda_1)$ of the two networks (Theorem G says the one-dimensional ceiling is not a cap when $\rho_B$ is not small); a destructive measurement of the heavy-ball minibatch ceiling in $\mathbb{R}^P$; and a Hessian baseline that needs no second-order code.

Design. Two network checkpoints, whose records ($\lambda_1$ and $\lambda_2$ from analytical HVPs, the quadratic and destructive thresholds, $\nu_B$) are in `networks.json` ($P = 3\,610$: `64-48-10`; $P = 12\,826$: `256-48-10`; $N = 64$), $\beta = 0.9$, $B = 16$. Exact trace from $P$ analytical HVPs with the canonical basis (`compute-exact-hessian-trace.ts`); $\lambda_{\min}$ from 100 shifted power iterations; secular bracket from $(\lambda_1, \lambda_2, \operatorname{tr}H)$ with $\kappa_{\text{feat}}$ read from $\nu_{11}$ (`compute-secular-ceiling-bracket-from-trace.ts`). Ceiling: homogeneous linearised iteration $w_{t+1} = w_t + \beta(w_t - w_{t-1}) - \eta\,(g_{B_t}(w_t) - g_{B_t}(\Theta_0))$ with real minibatches of 16 of the 64 samples drawn with replacement, every stream started at distance $10^{-6}$ in the range of $H$, Phase 1 statistic (log ratio of fourth- to second-quarter energy over 8 streams of 1000 steps, threshold $\log 2.5$), bisection of 14 steps from $1.1 \times$ the quadratic threshold, 4 seeds (`simulate-network-minibatch-heavy-ball-energy.ts`, `measure-network-minibatch-heavy-ball-ceiling.ts`). First-order baseline: power iteration with central-difference HVPs from two full gradients ($\epsilon = 10^{-5}$), 15 products, 1 920 sample-gradient equivalents (`compute-top-eigenpair-power-iteration.ts`).

Harness note. A first smoke run used the additive minibatch gradient noise scaled by $\tau = 10^{-3}$ and a start at $\Theta_0$, as in Phase 1. On the networks the Hessian has rank at most $N \cdot d_{\text{out}} = 640 \ll P$ and the additive noise diffuses freely in its null space, where the energy grows linearly in $t$ at every learning rate and the window ratio sits at $7/3$, next to the threshold. The registered run therefore uses $\tau = 0$ and a start in the range of $H$; the predictions were not changed.

Predictions (registered before the run):
- `q1`: $\operatorname{tr}H/\lambda_1 \ge 4$ on both networks, so $\rho_B \ge 4.75$ at $\beta = 0.9$, $B = 16$.
- `q2`: the measured ceiling in $\mathbb{R}^P$ lies below the one-dimensional EoSS ceiling of Theorem E(3) by a factor $\ge 1.5$ on both networks.
- `q3`: the median measured ceiling lies inside the trace bracket of the random-feature model, widened by 10 % plus the bisection half-width, on both networks.
- `q4`: the finite-difference power iteration estimates $\lambda_1$ to a relative error below $10^{-6}$ on both networks, so its error against the quadratic threshold is below 0.01 % and against the destructive one equals the quadratic–destructive gap within 0.01 points.

Results (`raw-phase2.json`, 2026-10-27, about 8 min single thread, Node 26.5). `q1 = true`, `q2 = true`, `q4 = true`; `q3 = false`.
- Trace: $\operatorname{tr}H = 601.74$ and $901.72$, $\operatorname{tr}H/\lambda_1 = 6.10$ and $6.08$, $\rho_B = 7.24$ and $7.22$; $\lambda_{\min} = 0.012$ and $-0.072$ ($H \succeq 0$ to $5 \times 10^{-4}\lambda_1$); $\kappa_{\text{feat,eff}} = -1.16$ and $-0.32$, $\alpha_{\text{diag}} = 0.81$ and $1.80$; trace bracket $[0.00492, 0.00521]$ and $[0.00308, 0.00338]$ against one-dimensional ceilings $0.01928$ and $0.008564$.
- Ceiling in $\mathbb{R}^P$: medians $0.002707$ ($P = 3\,610$; $7.1\times$ below the one-dimensional ceiling, $14\times$ below the quadratic threshold, half-width 0.05 %) and $4.90 \times 10^{-5}$ ($P = 12\,826$; $175\times$ and $523\times$, half-width 1.8 %). `q2` holds; `q3` fails by $1.8\times$ and $63\times$ below the bracket.
- Growth grid (post hoc, 4 streams, 1000 and 4000 steps): at $P = 3\,610$ the log growth is negative at both lengths for $0.002$–$0.01\times\eta_{\text{quad}}$ ($-0.99$ to $-0.22$), $-0.88$ then $+0.70$ at $0.02\times$, $-0.42$ then $+5.8$ at $0.05\times$, guard at $0.1\times$ (4000 steps); at $P = 12\,826$ $+1.0$ then $+10.2$ at $0.002\times$, $+6.0$ then guard at $0.005\times$. Diffusion would give the same window ratio at both lengths; the growth increases with length, so it is exponential, and the bisection value is where the growth reaches the ratio 2.5 within 1000 steps. The linearly stable regime ends between $0.01\times$ and $0.02\times$ the quadratic threshold at $P = 3\,610$ (25–50× below the one-dimensional ceiling) and below $0.002\times$ at $P = 12\,826$ (more than 167× below).
- Reading. The ceilings of Theorem E(3) (along $u_1$) and Theorem G (feature model) tie the curvature noise entering a mode to that mode's curvature; the per-sample Hessians of the networks put curvature noise into directions where $H$ is nearly flat, which have no restoring force. The linearly stable regime at these checkpoints lies one to two orders of magnitude below the one-dimensional ceiling; the bisection value is the learning rate at which the growth reaches the detection ratio within the run. The excursions of actual training are bounded by the nonlinearity, not by a linear ceiling.
- First-order baseline: $\hat\lambda_1$ within $8.4 \times 10^{-7}$ and $1.6 \times 10^{-8}$ of the analytical value, $|\cos| = 0.9999994$ and $1.0000000$, error on $\eta_{\text{crit}}(0.9)$ of 0.0001 % and 0.0000 % against the quadratic threshold and 0.0002 % and 0.5668 % against the destructive one (the quadratic–destructive gap of the networks), 1 920 sample-gradient equivalents and no second-order code (`q4`).

## Phase 3 — the secular root of Theorem G against the exact second-moment operator

Raw: `evidence/mean-square-boundary/raw-phase3.json`, `phase3-summary.md` (written by `run-phase3.ts`). The spectral radius of the second-moment operator, by power iteration from the identity on the symmetric matrices, at $(1 \pm 10^{-3})\,\eta^\star$ and at $\eta^\star$, the secular root; spectrum $\lambda_k = k^{-1/2}$, $P \in \{64, 256, 1024\}$, $\beta \in \{0, 0.9\}$, $B \in \{4, 16, 64\}$, Gaussian and Rademacher features (36 cells).

Provenance: A 4-cell prototype (P = 64 and 256, beta = 0 and 0.9) was run before q1-q3 were written; it gave rho(eta*) = 1 to 12 digits, so q1 and q2 were informed by it. After the predictions, a P = 16 debugging run showed that the ratio of norms does not converge on the antisymmetric part at beta = 0.9 (a cluster of eigenvalues of equal modulus), and the antisymmetric estimator was changed to the Gelfand estimate over a fixed horizon before the full run; that run gave antisymmetric radii of 0.44-0.61 at beta = 0 and about 0.906 at beta = 0.9, so q3 was seen at P = 16 before the full run. No P = 64, 256 or 1024 cell was run with the final code before the full run.

Predictions:
- **`q1`**: In every cell the radius of the second-moment operator is below 1 at (1 - 1e-3) eta* and above 1 at (1 + 1e-3) eta*, eta* the secular root of Theorem G.
- **`q2`**: In every cell the crossing rho = 1, located from rho(eta*) - 1 and the local slope, is within 1e-8 (relative) of eta*.
- **`q3`**: In every cell the radius on the antisymmetric matrices at eta* is strictly below 1, the radius on the symmetric part there (Lemma: the antisymmetric part never dominates).

Results (`raw-phase3.json`, 2026-10-27, Node 24.18): `q1`, `q2` and `q3` hold in all 36 cells; the crossing is within $6.11 \times 10^{-10}$ (relative) of $\eta^\star$ in every cell. Six cells at $P = 64$, $\beta = 0.9$ have `allConverged` false, the conjunction of four convergence flags, with crossing errors between $3.2 \times 10^{-14}$ and $5.1 \times 10^{-13}$. `check-phase3-convergence.ts` shows that the flag that fails is the agreement test of the Gelfand estimate on the antisymmetric matrices; the three symmetric iterations, which locate the crossing, converge (248–275 steps from a cold start; the run, warm-started from the iterate at $(1-10^{-3})\eta^\star$, recorded 175–208 at the root).


## Phase 4 — the exact threshold of Theorem G.6 on tethered network Hessians (predictions written before the run)

Raw: `evidence/mean-square-boundary/raw-phase4.json`, `phase4-summary.md` (written by `run-phase4.ts`; cells evaluated by
`evaluate-phase4-tethered-mean-square-cells.ts` and `evaluate-phase4-network-row.ts`, per-sample Hessians from
`create-phase4-sample-hessian-provider.ts`). The mean Hessians of the two network checkpoints are indefinite ($P = 33$:
$\lambda_{\min} = -0.0080$, two negative eigenvalues; $P = 3\,610$: $\lambda_{\min} = -0.0833$, 1 528 negative of 3 610). A tether
$\mu = g + \max(0, -\lambda_{\min})$ makes $H + \mu I$ positive definite with smallest eigenvalue $g$ and leaves the curvature
noise unchanged, so Theorem G.6(3) gives one finite threshold $\eta^\star(\mu)$, computed by Lanczos from the real per-sample
Hessians: at $P = 33$ dense central differences of per-sample gradients ($\epsilon = 10^{-5}$, symmetrised), $\beta \in \{0, 0.9\}$;
at $P = 3\,610$ exact per-sample Hessian-vector products, $\beta = 0.9$; $B = 16$ with replacement,
$g \in \{128, 32, 8, 2, \frac12, \frac18, \frac1{32}, \frac1{128}\}$. Predictions `t1`–`t6` and checks `c1`–`c3` are in the raw file.

Results (2026-10-28). `t1`–`t4` and `t6` hold, `t5` fails; `c2` and `c3` fail at $P = 3\,610$. At $P = 33$ the spectral radius
of the exact second-moment operator is below 1 at $(1 - 10^{-3})\eta^\star$ and above 1 at $(1 + 10^{-3})\eta^\star$ in all 16
cells, each side certified by a Collatz–Wielandt bracket; at $P = 3\,610$, $g = 2$, power iteration gives 0.99982427 and
1.00017600 but the last iterate is not positive definite in floating point, so the crossing rests on the power iteration alone.
The median sample-path ceiling lies at 1.0008–1.190 $\eta^\star$ ($P = 33$) and 1.081–1.257 $\eta^\star$ ($P = 3\,610$). The
predicted flat-mode collapse (log–log slope in $[0.4, 1.1]$ between $g = \frac1{32}$ and $\frac1{128}$) did not occur: slope
0.0022 (`t5`). At the smallest tether the threshold is $1.23\times$ below the one-dimensional ceiling at $P = 3\,610$. The
$P = 3\,610$ cells were resumed from the logged cells of an earlier launch that crashed (`run-phase4-recover-resume-cells.ts`,
`phase4-progress*.log`, `phase4-resume-x1-p3610.json`). The dense products of the row workers use the WebAssembly `f64x2`
kernel `dense-f64-matmul.wasm` (`compile-dense-f64-matmul-kernel.ts` reads it; the test checks it bit for bit against the
sequential product).

## Phase 5 — does training saturate the threshold? (registered; read as exploratory)

Raw: `evidence/mean-square-boundary/phase5-runs/` (48 run records), `raw-phase5.json` (`run-phase5-aggregate.ts`),
`phase5-c5-perron.json` (`run-phase5-c5-perron.ts`). The $P = 33$ network is trained from a fresh initialisation by heavy-ball SGD
($\beta = 0.9$, $B = 16$ with replacement, 20 000 steps, $\eta \in \{0.01, 0.03, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5\}$, weight decay
$\mu_{\mathrm{train}} \in \{0, 10^{-3}\}$, seeds 1–3; `run-phase5.ts`, one run per process). Every 200 steps the Hessian and the 64
per-sample Hessians at $\theta_t$ give $\eta^\star_{|H|}(\theta_t)$ (Theorem G.6 with the eigenvalues of $H$ replaced by their
absolute values) and the other criteria of `evaluate-phase5-checkpoint.ts`; the statistic is $r = m_\star / \eta$, $m_\star$ the median
over the checkpoints of steps 10 000–20 000. Hypotheses H1–H5, predictions `p1`–`p6` and checks `c4`–`c6` were registered before any
threshold was computed; a pre-commitment (written at 23:25:29 UTC on 28 October 2026, after the 48 runs
were launched at 23:15:35 and when 22 had finished; the author declares that no output had been read) hashes the registration, the
evaluator, the runner, the package manifest and the job list; it and the manifest are kept in the companion lab, and the job list is
`phase5-jobs.json`.

Results (2026-10-28). No run diverged. As computed, H1 and H2 both fail at both $\mu_{\mathrm{train}}$, `p4` holds and `p1`, `p2`,
`p3`, `p5`, `p6` fail; check `c5` (Collatz–Wielandt bracket of the second-moment operator at $(1 \mp 10^{-3})\eta^\star_{|H|}(\theta_T)$)
decides neither side in 12 of 48 runs (21 of 96 sides). The diagnosis, written after the runs, is a resolution limit: near-flat modes
put the Perron eigenvalue $1.5\times10^{-9}$–$1.1\times10^{-7}$ below 1 on the failing lower sides and its eigenmatrix is too
ill-conditioned for a double-precision test; no certificate contradicts a side. The registration stops the reading of the
predictions at a failed check, so Phase 5 is reported as run and serves as the exploratory run of Phase 5b.

## Phase 5b — a within-dataset replication (predictions written from Phase 5, before the run)

Raw: `evidence/mean-square-boundary/phase5b-runs/`, `raw-phase5b.json` (field `phase5b`, `evaluate-phase5b-predictions.ts`). Phase 5
without any change to the network, data, training, grid or criteria, on seeds 1401–1403; predictions `q1`–`q6` were written after
reading Phase 5 and restricted to the cells where its three seeds agreed. The registration, the evaluator and the hashes of the code
package and job list (the pre-commitment and the package manifest, kept in the companion lab, and `phase5b-jobs.json`) were committed at 02:22 UTC on
29 October, before the first run started (02:33).

Results (2026-10-29). All 48 runs finished, no check suspends a prediction, and `q1`–`q6` hold. With weight decay $10^{-3}$ the
window median $m_\star$ stays at 0.088–0.154 over a fifty-fold range of $\eta$; the runs cross it between $\eta = 0.1$
($r \approx 1.29$) and $0.15$ ($r = 0.84$–$0.96$) and train at $2.6$–$2.9\times$ ($\eta = 0.3$) and $4.0$–$4.5\times$ ($\eta = 0.4$) the
threshold of their own iterates without diverging, at a higher data loss. The certificate check at the final point fails on one run
($\mu_{\mathrm{train}} = 0$, $\eta = 0.4$, seed 1402), the resolution limit of Phase 5; no prediction reads it.

## Phase 6 — a certified two-sided bracket of the Gauss–Newton threshold (exploratory)

Plan and rules: `phase6-plan-certified-mean-square-bracket.md` (stage rules in its section 5, outcomes in section 12). The object is
the heavy-ball iteration whose per-sample curvature is the Gauss–Newton matrix plus the tether $\mu_{\mathrm{train}} I$, at a frozen
iterate. Upper bound: the Rayleigh quotient of a short Lanczos vector minus its forward error bound
(`compute-certified-rayleigh-upper-threshold.ts`); lower bound: a positive-definite witness $X$ with $X - K(\eta)(X) \succ 0$ for the
cone-preserving operator $K(\eta)$ of Theorem G.6 (`certify-cone-collatz-wielandt-at-rate.ts`), proved by a floating-point Cholesky
with Rump's shift (`verify-positive-definite-shifted-cholesky.ts`), bisected below the upper bound
(`compute-certified-threshold-bracket.ts`, `compute-adaptive-certified-threshold-bracket.ts`). Checkpoints are regenerated by rerunning
training (`regenerate-phase5-checkpoint-iterates.ts`, `regenerate-phase6-pilot-checkpoint-iterates.ts`) and the stored loss trace is
compared bit for bit before any bracket is computed. The checkpoint lists are in `phase6-checkpoints.json`, committed before the runs.

- Stage 1 ($P = 33$, every window checkpoint of the 48 Phase 5b runs, Gauss–Newton, its quotient by the kernel at
  $\mu_{\mathrm{train}} = 0$ and $|H|$: 9 600 brackets; `run-phase6-stage1.ts`, `summarize-phase6-stage1.ts`,
  `phase6-stage1/`, `phase6-stage1-summary.json`; reference: a dense exact value, `phase6-stage1-dense-reference/`). All
  regenerations match; no bracket excludes the dense exact threshold; every bracket has $\eta_{\mathrm{low}}/\eta_{\mathrm{up}} \ge 0.98$
  (smallest 0.9934 Gauss–Newton, 0.9802 $|H|$). A first run of the same code before the audit rounds is kept as `phase6-stage1-v1`.
- Stage 2 ($P = 3\,610$ and $12\,826$, row space $R = 640$; seed 0 of every pilot cell with $\mu_{\mathrm{train}} = 10^{-3}$ and a finite
  run, 30 runs, 11 window checkpoints each; `run-phase6-stage2.ts`, `summarize-phase6-stage2.ts`, `phase6-stage2/`,
  `phase6-stage2-summary.json`). The pilot runs are heavy-ball training of the two networks of Phase 2 on pilot datasets with `tanh`,
  softsign and leaky ReLU (slope 0.1) units (`create-phase6-student-network.ts`, `phase6-network-configs.ts`); their records, which
  the regeneration compares against, are in `phase6-pilot/`. All 30 regenerations match; every bracket overlaps its long-Lanczos
  reference with no tolerance; $\eta_{\mathrm{low}}/\eta_{\mathrm{up}} \ge 0.982$ (median 0.9994), relative width at most 1.78 %
  (median 0.061 %). Against the band $[0.8\eta, 1.25\eta]$, at $\eta \ge 0.01$: 161 checkpoints inside, 23 above, 14 below; window
  medians of $\eta_{\mathrm{up}}/\eta$ 0.81–1.10 at $\eta \ge 0.02$ for `tanh` and softsign. Expectations written before the batch:
  `e16-2` and `e16-4` hold, `e16-3` fails (two runs at $\eta = 0.01$, $P = 3\,610$, read above).

## Provenance of Phases 4–6

Code: copied from the companion lab on 2026-11-01 with the phase numbers and folder names of this repository, with three changes.
(1) `compile-dense-f64-matmul-kernel.ts` returns the bytes of `dense-f64-matmul.wasm` (SHA-256 `17fd9498…eb934714`), the module the
lab's code generator emits; the generator is not part of this repository. (2) The increment-stream block `e13` of the Phase 5
checkpoint records (an estimator of the companion manuscript) is not computed: `evaluate-phase5-checkpoint.ts` writes `e13: null`, and
the amendment-3 diagnostics omit its summaries. (3) Comments name auditors generically.

Raw files: copied byte for byte (`evidence/mean-square-boundary/PORTED-FROM-LAB.txt` lists them), except that the `e13` blocks of the
Phase 5 and 5b run records are `null` and `raw-phase5.json` and `raw-phase5b.json` were rewritten by `run-phase5-aggregate.ts` from those
records (identical to the lab aggregates apart from `date`, `labId`, `nodeVersion`, the `e13` summaries and the SHA-256 of the run
files). The run records keep the lab's `labId` values. Checked on 2026-11-01 with the code of this repository: a full Phase 5b run
(`eta = 0.03`, `mu_train = 0`, seed 1401) reproduces its record except `date`, `elapsedSeconds` and `nodeVersion`, and both Phase 6
summaries regenerate from the raw records (also a test).

Pre-commitments: the pre-commitments and package manifests of Phases 5 and 5b, and the package manifest of the Phase 6 pilots, list
files of the companion lab that are not part of this repository; they are kept there and are not copied here. The job lists are copied
unchanged (they name the lab's paths), with the OpenTimestamps receipts made at the time as `*.ots` files; no statement of the papers
rests on them.

