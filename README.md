# Minibatch Heavy-Ball SGD at Finite Dimension: Exact Threshold, Rate and Critical Slowing Down

Code, raw files and checks for:

- **Note (5 November 2026):** [*Minibatch Heavy-Ball SGD at Finite Dimension: Exact Threshold, Rate and Critical Slowing Down*](paper/note-short.pdf) ([LaTeX](paper/note-short.tex)): five pages of results, then the proofs. Every number is printed by `python3 paper/checks/note-v2-checks.py` (numpy; parts A, C, D and H–K); [VERIFY.md](VERIFY.md) lists, for each result, the independent computation it is checked against, and `python3 paper/checks/crossover-figure.py` checks the momentum crossover at $`B-1=B_\times`$ against a search on the $`3\times3`$ matrix $`M(q)`$ alone.
- **Longer note (4 November 2026):** [*Minibatch Heavy-Ball SGD with Mode-Dependent Kurtosis*](paper/note-v2.pdf) ([LaTeX](paper/note-v2.tex)): the general setting, the EMA form of momentum and batch, the rate at a fixed loss floor.
- **Earlier note (3 November 2026):** [*Secular Equation and Exact Loss Convolution for Minibatch Heavy-Ball SGD on Quadratics*](paper/note-secular-heavy-ball.pdf) ([LaTeX](paper/note-secular-heavy-ball.tex)).
- **Extended draft:** [*The Exact Mean-Square Stability Boundary of Heavy-Ball SGD: a Lyapunov Equation and a Secular Equation*](paper/excerpt-second-moment.pdf) ([LaTeX](paper/excerpt-second-moment.tex)): the theory and the measurements of Phases 1–6. Its Theorem G.6 is also the criterion of Cai et al. (2026, Theorem F.1), obtained concurrently and independently; the draft cites them.
- **Interactive lab:** [*The Critical Step*](https://apis-os.github.io/hb-sgd-mean-square/), with a [plain-English version](https://apis-os.github.io/hb-sgd-mean-square/plain.html): six scenes with narrated tours, the last one on the vocabulary model (Theorems 2, 3 and 6 of the note). Served from the [`gh-pages`](https://github.com/apis-os/hb-sgd-mean-square/tree/gh-pages) branch.

**In short.** For heavy-ball SGD on a quadratic with minibatches of random features whose coordinates in the eigenbasis of the Hessian are independent, with coordinate-dependent fourth moments, the note gives at finite dimension and batch the mean-square threshold as the root of a rank-one secular equation and the exact convergence rate as $`1/z_\star`$, with $`z_\star`$ the smallest positive root of $`z\widehat K(z)=1`$ for the generating function of the kernel of an exact convolution equation for the loss (a renewal equation when the kernel is nonnegative), together with the prefactor and the slope of critical slowing down. Applied to the vocabulary model of [Everett and Paquette (2026)](https://arxiv.org/abs/2605.28961), it proves at every finite batch the three vocabulary statements that their Appendix A.4 lists as not proved, and the observations of their Remark H.17: the slowest row is the most or the least frequent one; no step and momentum make a gated feature decay faster than the best SGD step, so momentum needs as many samples per decay time as SGD; at a fixed momentum the samples per decay time increase strictly with the batch, so their minimum is at $`B=1`$ and has a closed form; momentum takes exactly as many steps as SGD for $`B-1\le B_\times`$ (their batch size $`B_\times`$), where the tuned momentum is zero, and fewer for $`B-1>B_\times`$; and the fewest steps approach their large-batch limit as $`C_{\mathrm{mom}}B^{-1/2}`$ with an explicit constant. These results concern the least-squares vocabulary model; they do not address the logistic-regression model or the samples needed to reach a prescribed loss with label noise. Prior and related work (Liu, Ziyin and Ueda; Ziyin et al.; Wang et al.; Velikanov et al.; Bordelon and Pehlevan; Morwani et al.; Cai et al.; Everett and Paquette) is credited in the note.

## Results of the earlier note (3 November 2026)

Heavy-ball SGD on the quadratic $`L(w)=\frac12 w^\top Hw`$ in $`\mathbb{R}^P`$, $`H = \sum_k \lambda_k u_k u_k^\top \succ 0`$:

```math
w_{t+1} = \big((1+\beta)I - \eta H_{B_t}\big)w_t - \beta w_{t-1} - \eta \xi_t, \qquad \beta \in [0,1),\ \eta > 0,
```

with i.i.d. minibatch Hessians $`H_{B_t}`$ (mean $`H`$, curvature-noise operator $`\mathcal{V}(X) = \mathbb{E}[(H_{B_t}-H)X(H_{B_t}-H)]`$) that need not commute with each other or with $`H`$.

**1. Exact threshold for non-commuting Hessians (Theorem 1, Lemma 2, Theorem 3, Corollary 4).** Theorem 3 is also [Theorem F.1 of Cai et al. (2026)](https://arxiv.org/abs/2610.36778), obtained independently; their eq. (67) gives Corollary 4 without strictness. Eliminating the lag-one covariance gives the stationary equation on $`\Sigma_0 = \mathbb{E}[w_t w_t^\top]`$,

```math
(H\Sigma_0 + \Sigma_0 H) - \frac{\eta}{1+\beta}H\Sigma_0 H + \frac{\eta\beta}{(1-\beta)(1-\beta^2)}[H,[H,\Sigma_0]] - \frac{\eta}{1-\beta}\mathcal{V}(\Sigma_0) = \frac{\eta}{1-\beta}\Sigma_\xi .
```

With $`\mathcal{L}_0(X) = HX+XH`$ and $`\mathcal{B}_\beta(X) = \frac{1}{1+\beta}HXH - \frac{\beta}{(1-\beta)(1-\beta^2)}[H,[H,X]] + \frac{1}{1-\beta}\mathcal{V}(X)`$, the mean-square threshold is

```math
\eta^\star(\beta) = 1\big/\lambda_{\max}\big(\mathcal{L}_0^{-1/2}\mathcal{B}_\beta\mathcal{L}_0^{-1/2}\big).
```

If $`\mathcal V`$ does not depend on $`\beta`$, the effective threshold $`\eta^\star(\beta)/(1-\beta)`$ increases strictly with $`\beta`$. With non-degenerate additive noise ($`\Sigma_\xi\succ0`$) as in Lemma 2, bounded second moments are equivalent to $`\rho(\mathbb E[F_t\otimes F_t])<1`$, a standard fact (Costa et al. 2005); Lemma 2 adds the linear growth beyond the threshold.

**2. Random features (Theorem 5, Proposition 6).** For $`H_{B_t} = \frac1B\sum_b z_{t,b}z_{t,b}^\top`$ with fresh features $`z = H^{1/2}\omega`$, whose coordinates are independent with $`\mathbb{E}[\omega_k^4] = 3 + \kappa_{\mathrm{feat}}`$, and $`p = 1/B`$, $`\eta^\star(\beta)`$ is the unique root with $`\eta\lambda_1\max(\alpha_{\mathrm{diag}},0) < 2(1+\beta)`$ of

```math
S(\eta)=\frac{\eta p}{2(1-\beta)}\sum_{k=1}^P \frac{\lambda_k}{1 - \frac{\eta\lambda_k}{2(1+\beta)}\alpha_{\mathrm{diag}}} = 1, \qquad \alpha_{\mathrm{diag}} = 1 + p\frac{1+\beta}{1-\beta}(1+\kappa_{\mathrm{feat}}).
```

The quantity $`\ell_t = \mathbb E\langle w_t, Hw_t\rangle`$ (twice the expected loss) obeys an exact discrete convolution equation $`\ell_t = f_t + \sum_{s<t}K_{t-1-s}\ell_s`$. Its kernel sum equals $`S(\eta)`$ when $`\eta\lambda_k\max(\alpha_{\mathrm{diag}}, 2-\alpha_{\mathrm{diag}}) < 2(1+\beta)`$ for every $`k`$, which holds up to $`\eta^\star`$ when $`\kappa_{\mathrm{feat}} \ge -1`$. Take $`S(\eta)`$ at $`\alpha_{\mathrm{diag}} = 1`$, with $`p`$ replaced by $`(1-\zeta)/B`$, the large-$`n`$ form of the without-replacement factor $`\frac{n-B}{B(n-1)}`$, and their notation $`\eta=\gamma B`$, $`\beta=\Delta`$, $`\lambda_j=\sigma_j^2/n`$, $`\zeta=B/n`$. That has the closed form of the Volterra kernel norm in the proof of Proposition 2 of Lee, Cheng, Paquette and Paquette (2022), a high-dimensional result for orthogonally invariant data; the two formulas coincide, the two settings do not. At $`B=1`$ with Gaussian features, the secular equation with $`\eta = 2\mu`$ and $`\beta = \alpha`$ (their notation) is the necessary condition $`Q(1)>0`$ of Roy and Shynk (1990) for momentum LMS; Theorem 5 shows that it is also sufficient below the first pole.

**3. Mode coupling (Section 4).** For Rademacher features ($`\kappa_{\mathrm{feat}} = -2`$) the curvature along $`u_1`$ is deterministic, so the one-dimensional threshold along $`u_1`$ is $`2(1+\beta)/\lambda_1`$ for every $`B`$. The coupling of $`u_1`$ to the other modes, $`\mathbb E[(u_1^\top H_{B_t}u_k)^2] = \lambda_1\lambda_k/B`$, lowers $`\eta^\star(\beta)`$ to order $`(1-\beta)B/\mathrm{tr} H`$ at high effective rank. At $`P=256`$, $`\lambda_k = k^{-1/2}`$, $`\beta=0.9`$, $`B=4`$: one-dimensional $`3.8`$, secular root $`0.0263`$, a factor $`144.5`$; the sample-path median is $`0.0264`$.

### A two-dimensional example

Not in the note; a check of Theorem 3. Take $`H_+=\begin{pmatrix}3&2\\2&2\end{pmatrix}`$ and $`H_-=\begin{pmatrix}3&-2\\-2&2\end{pmatrix}`$, each drawn with probability $`1/2`$ at every step, $`\beta=0.9`$. Both are positive definite and neither commutes with the other or with their mean $`H=\mathrm{diag}(3,2)`$. The operator of Theorem 3 on the $`2\times2`$ symmetric matrices gives $`\eta^\star=0.113328`$; bisection on the spectral radius of the $`16\times16`$ operator $`\mathbb E[F_t\otimes F_t]`$ gives the same value to a relative $`2\times10^{-14}`$, with $`\rho=0.99445`$, $`1.00000`$, $`1.00562`$ at $`0.95`$, $`1`$, $`1.05`$ times $`\eta^\star`$. The one-dimensional threshold $`2(1+\beta)/\lambda_1=1.2667`$ is $`11.18`$ times larger. `python3 paper/checks/noncommuting-2x2.py` (numpy) prints these numbers.

### Where each number is checked

| In the note | Raw file or script |
|---|---|
| Table 1 and Figure 1 (sample-path medians) | Phase 1: `evidence/mean-square-boundary/raw-phase1.json`, `phase1-summary.md` |
| Operator crossing $`\rho(\mathcal T)=1`$ against the secular root, 36 cells, at most $`6.11\times10^{-10}`$ (six cells at $`P=64`$, $`\beta=0.9`$ have `allConverged` false: the flag that fails is the Gelfand estimate on the antisymmetric matrices, not the symmetric iterations that locate the crossing; see the Phase 3 results in `labs/mean-square-boundary/README.md`) | Phase 3: `evidence/mean-square-boundary/raw-phase3.json`, `phase3-summary.md`, `labs/mean-square-boundary/check-phase3-convergence.ts`, `paper/checks/evidence-ranges.mjs` |
| Theorem 1 (the script also checks the relation to the Schur complement of Andreyev et al. (2026, App. D) stated in the extended draft) | `paper/check-schur-factor.py` |
| Lemma 2 (linear growth for $`\rho(\mathcal T)\ge1`$) | `paper/checks/ms-instability-criterion.mjs` |
| Theorem 3 (operator of the threshold, against an independent dense construction) | `tests/phase4-tethered-thm-g6.test.ts` |
| Corollary 4 (effective threshold increasing in $`\beta`$, and its bound) | `paper/checks/second-moment-renewal.py` (i) |
| Theorem 5 (secular root against the second-moment operator on $`2P\times2P`$ matrices) | Phase 3 (row above) |
| Proposition 6 (convolution equation; closed form of the kernel sum) | `paper/checks/second-moment-renewal.py` (c) |
| Importance-resampling estimate, $`0.32\%`$ | `paper/checks/phase1-resampling-estimator.py`, `evidence/mean-square-boundary/phase1-resampling-estimator.json` |

The other check scripts serve the extended draft. `labs/mean-square-boundary/README.md` maps the theorem letters used in the code to the numbering of the drafts.

## Layout

- `paper/`: the two drafts (`excerpt-second-moment.tex`, `note-secular-heavy-ball.tex`, their PDFs, `refs.bib`) and the numerical checks they cite
  (`checks/*.mjs`, `checks/*.py`, `check-schur-factor.py`).
- `labs/mean-square-boundary/`: the code of the six phases (1, 2, 3, 4, 5 with its replication 5b, and 6). `README.md` there gives the design, the predictions written
  before each run, the results, and the map from the theorem letters used in the code to the numbering of the draft.
- `evidence/mean-square-boundary/`: raw files (`raw-phase1.json` to `raw-phase5b.json`, `phase1-diag-as-vs-ms.json`, the run
  records `phase5-runs/` and `phase5b-runs/`, the Phase 6 brackets `phase6-stage1/` and `phase6-stage2/` with their summaries and
  checkpoint list `phase6-checkpoints.json`, the pilot records `phase6-pilot/` that Phase 6 regenerates) and their summaries;
  `networks.json` holds the records of the two network checkpoints that Phases 2 and 4 read. The job lists of Phases 5 and 5b are kept unchanged
  (with the `*.ots` receipts made at the time); their pre-commitments and package manifests list files of the companion lab and are kept there.
- `src/math/`: the FDLIBM 5.3 ports of `exp`, `log`, `sin`, `cos` and `tanh` (and their helpers; the same files hold `tan`,
  `sinh` and `cosh`, listed in `NOTICE`) used instead of the engine's `Math` functions, so that the raw files do not depend on the JavaScript engine.
- `scripts/`: the generator of the three Lyapunov step kernels `step-eigenbasis-lyapunov-g*.gen.ts` (`npm run generate:check`).
- `tests/`: one test per phase (Phase 5 has none of its own: the Phase 5b and 6 tests read its records); each re-executes a reduced version of the phase and checks the committed raw file, including
  the predictions that failed.

## Reproduce

Node 22.18 or later (23.6 or later on the 23 line) runs the TypeScript sources directly; the tests load a `.ts` file in a
worker thread, which needs type stripping on by default. The raw files of Phases 1 and 2 and
of the diagnostic were written under Node 26.5, the one of Phase 3 under Node 24.18.

```sh
npm install                 # TypeScript, Vitest and Node types, for the checks below
npm run typecheck
npm test                    # the phase tests
npm run checks:note         # the standalone checks except the resampling estimator and bridge-theorems.py; standard-library Python only, about 2 min
npm run checks:resampling   # the resampling estimator of the note (Phase 1), about 2 min; rewrites phase1-resampling-estimator.json bit for bit
npm run checks              # checks:note plus bridge-theorems.py (cited in the longer draft) and noncommuting-2x2.py (needs Python 3 with numpy)
npm run phase1              # 43 min on one thread; rewrites raw-phase1.json
npm run phase1:diagnostic   # about 7 min; rewrites phase1-diag-as-vs-ms.json
npm run phase2              # about 8 min; rewrites raw-phase2.json
npm run phase3              # worker pool; rewrites raw-phase3.json
npm run phase4              # hours on 4-8 cores; rewrites raw-phase4.json
npm run phase5:run -- 0.03 0 1401    # one Phase 5/5b run (eta, mu_train, seed), about 3 min; 48 per phase
npm run phase5:aggregate    # rewrites raw-phase5.json from phase5-runs/
npm run phase5b:aggregate   # rewrites raw-phase5b.json from phase5b-runs/ (with the q1-q6 decisions)
npm run phase6:stage1 -- eta0.03-mu0-seed1401   # one stage 1 run record (Phase 5b run name)
npm run phase6:stage2 -- p3610-tanh-eta0.05-mu0.001-seed0 10000 200 out.json   # one stage 2 checkpoint, 4-13 min
npm run phase6:summaries    # rewrites both Phase 6 summaries from the records
```

A rerun reproduces a raw file bit for bit except its `date` and `nodeVersion` fields and, in Phase 2, the wall-clock fields
`traceSeconds` and `ceilingSeconds` (Phases 5, 5b, 6: also `elapsedSeconds` and the per-bracket `seconds`). Phases 4–6 were run in
the companion lab and ported; `labs/mean-square-boundary/README.md` (*Provenance of Phases 4–6*) states what was changed and what
was checked here.

Where a README or a draft says that predictions were written before a run, the order is the author's statement: the history of
the companion lab in which they were written is not part of this repository.

## License

Code, raw files, and `paper/refs.bib`: MIT (`LICENSE`); FDLIBM ports in `src/math/` retain the Sun Microsystems notice (`NOTICE`). Drafts, READMEs, and figures: CC BY 4.0 (`LICENSE-TEXT.md`).
