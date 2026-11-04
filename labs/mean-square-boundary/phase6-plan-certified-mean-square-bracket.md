# Phase 6 plan: a certified two-sided bracket of the exact mean-square threshold along training

Status: plan, revision 2 (2026-10-29). Nothing implemented or run. Revision 1 (commit `13724cf7`) was reviewed blind by four
reviewers, read-only, with their own scripts; section 9 summarises what changed.

## 1. Question and scope

Andreyev et al. (ICML 2026, arXiv:2604.14108) state that the precise instability criterion of SGD with momentum, and a proof that
training saturates it, remain open. The companion paper of this lab proves the exact mean-square criterion of heavy-ball SGD on a
quadratic with arbitrary non-commuting per-sample Hessians (Theorem `thm:G6`: `eta* = 1 / lambda_max(L0^{-1/2} B_beta L0^{-1/2})`).

At a training checkpoint the loss Hessian is indefinite (Phase 5, and the Phase 6 pilots), so the exact threshold of the loss
linearisation is zero, and the Phase 6 pilots measured its second moment growing (`g2` from 0.002 to 0.25 per step on `tanh` and `softsign`, `mu_train = 1e-3`,
`eta >= 0.01`) while training continued. Phase 6 therefore does not answer Andreyev's question for the loss linearisation. It asks
it for one named surrogate:

> **Object.** The heavy-ball iteration whose per-sample curvature is the Gauss–Newton matrix `J_i^T J_i` plus the tether
> `mu_train I`, at the frozen iterate `theta_t`, restricted to the row space of the stacked Jacobian (section 2).
>
> **Question.** Is `eta`, the training rate, inside `[0.8 eta*, 1.25 eta*]` along the window of a run, with `eta*` the exact
> mean-square threshold of that object, certified from both sides?

The band `[0.8, 1.25]` is the H1 band of the Phase 5 registration (`eta*/eta` in `[0.8, 1.25]`), kept so that the two phases
read on one scale. Every table that reports the bracket also reports, at the same checkpoint, `lambda_min(H + mu_train I)`, the
second-moment growth rate `g2` of the loss linearisation (the probe of the Phase 6 pilots), `lambda_max` and batch sharpness against their edges. A closed
bracket supports the surrogate and leaves the loss linearisation where the pilots put it.

What is already known about this object, from data on disk:

- At `P = 33` the Gauss–Newton row space is all of `R^33` and Phase 5b computed its exact `eta*` at every checkpoint (dense
  Lanczos, `gaussNewton.etaStar` in `evidence/mean-square-boundary/phase5b-runs/`). Window medians (steps 10 000–20 000) of
  `eta*_GN / eta` per seed, at `mu_train = 1e-3`: 11.9–12.0 (`eta = 0.01`), 4.0 (0.03), 1.29–1.30 (0.1), 0.99–1.44 (0.15),
  0.85–0.92 (0.2), 0.68–0.86 (0.3), 0.48–0.59 (0.4), 0.57–0.76 (0.5). The ratio falls through 1 and keeps falling: at `P = 33`
  the Gauss–Newton threshold does not cap training at `eta >= 0.3`. (Read by a reviewer's script, `audit_existing_evidence.py`
  section A0; to be recomputed by the stage 1 code before anything is written from it.)
- At `P = 3 610` and `12 826` (pilot) and `34 186` (pilot B), only the top-64 Rayleigh compression was computed; its threshold is
  an upper bound of `eta*`, and its ratio to `eta` was 0.84–1.13 over a range of rates (pilot, `mu_train = 1e-3`) and 0.88–1.1 (pilot B, `eta >= 0.006`).
  On the tethered Phase 4 Hessians the top-64 compression was within 1.035 of the exact value at `P = 3 610`; at training
  checkpoints that gap is not measured.

So the open part is whether the larger networks behave like `P = 33` (threshold below `eta` at large rates) or sit inside the band.

Order with respect to Phase 6. Phase 6 itself has not run: nothing of it is registered, and every number above from the larger
networks comes from its two exploratory pilots (two seeds each, pilot dataset seeds chosen so that the lab Phase 4 datasets stay unseen).
Phase 6 is validated only on those pilot checkpoints and on the Phase 5b runs; it never reads a dataset or seed reserved for the
Phase 6 registration. If the method is adopted (rule 2 of section 5), the bracket is written into the Phase 6 registration as one
of its measurements before Phase 6 runs, and the band question of this section is decided there, on the registered seeds; the
expectations `e16-*` below are about the pilot checkpoints only.

## 2. Objects

Linearised heavy-ball at a frozen point, `z_{t+1} = F_t z_t`, `z_t = (d_t, d_{t-1})`,
`F_t = [[(1 + beta) I - eta (H_{B_t} + mu I), -beta I], [I, 0]]`, `H_{B_t}` the mean of `B` per-sample curvatures drawn with
replacement from `N`. Second-moment operator `T(Q) = E[F Q F^T]` on `S^{2R}`; mean-square stable iff `rho(T) < 1`.

Theorem `thm:G6`, for `H_it = H + mu I > 0` (eigenvalues `lt_1 >= ... >= lt_R > 0`, eigenbasis `U`) and
`eta in (0, eta_det)`, `eta_det = 2(1 + beta)/lt_1`:

    L(eta)(X)_jk = l_jk(eta) X_jk,  l_jk(eta) = (lt_j + lt_k) + eta b_jk,
    b_jk = - lt_j lt_k / (1 + beta) + c_beta (lt_j - lt_k)^2,      c_beta = beta / ((1 - beta)(1 - beta^2)),
    V(X) = (1/B) E[dH_i X dH_i],  dH_i = H_i - H,
    K(eta)(X) = (eta / (1 - beta)) L(eta)^{-1}(V(X)),

`[1/l_jk] >= 0` (positive semidefinite, singular when two `lt` coincide; sixth audit round: revision 2 wrote `> 0`, and the cone
argument needs only semidefiniteness), `K(eta)` preserves `S_+^R`, `rho(T) < 1 iff rho(K(eta)) < 1`, `rho(K(eta))` is continuous in
`eta` and strictly increasing when `V != 0` (with `V = 0` it is 0 below `eta_det`), and `eta* = 1 / lambda_max(G)` with `G = L0^{-1/2} B_beta L0^{-1/2}`,
`B_beta(X) = H X H/(1 + beta) - c_beta [H, [H, X]] + V(X)/(1 - beta)`, `L0(X) = H X + X H` (all with `H_it`).

`B` enters only through `V`, as `1/B` times the per-sample variance. Averaging `F_i Q F_i^T` over the `N` samples is the `B = 1`
operator (revision 1 wrote that; at `B = 16` it underestimates `eta*` by a factor of about 15 on a reviewer's instance).

Reduction to the row space. Each `J_i^T J_i` leaves `range(J^T)` invariant and vanishes on its orthogonal complement. With
`mu_train > 0` the complement is a scalar heavy ball of curvature `mu_train` with no noise, threshold `2(1 + beta)/mu_train`
(3 800 at `1e-3`, `beta = 0.9`), so the row-space threshold is the threshold of the whole Gauss–Newton-plus-tether iteration. With
`mu_train = 0` the complement has companion eigenvalue 1 and the unreduced threshold is 0; the row-space threshold is then the
threshold of the quotient by the kernel, and is reported under that name, separately from `mu_train = 1e-3`.

Coordinates (the convention of `compute-gauss-newton-dual-spectrum.ts`). The dual matrix `(1/N) J J^T` (`N dOut` rows: 640 for
the lab Phase 4 networks, 2 560 for pilot B) has eigenpairs `(lambda_k, v_k)`; the retained basis is `u_k = J^T v_k / sqrt(N lambda_k)`
for `lambda_k > 1e-10 lambda_1` (it is the whole row space only when every positive mode is kept; otherwise the problem is a
compression, which bounds `eta*` from above only, and the record drops every lower bound; tenth audit round) (the cut and the number `R` of kept modes are recorded; at `P = 3 610` Phase 4 found rank 640 at
cuts `1e-10` and `1e-6`). In that basis the per-sample curvature is `A_i^T A_i` with `A_i in R^{dOut x R}`,
`(A_i)_{c,k} = (J_i u_k)_c = sqrt(N lambda_k) v_k[(i, c)]`, and the mean is `diag(lambda_k)`. Rounding makes `(1/N) sum_i A_i^T A_i`
diagonal only up to a residual `E = fl(Hbar) - diag(lambda)`, `Hbar = (1/N) sum_i A_i^T A_i`; the certificate is stated for the
iteration whose per-sample matrices are `A_i^T A_i - Hbar + diag(lt)` (same deviations `dH_i`, mean exactly `diag(lt)` with
`lt = fl(lambda_k + mu_train)`; seventh audit round: the earlier sentence dropped the tether; the code names the `dOut x R` factor
`M_i` and the curvature `A_i = M_i^T M_i`), and `||E||_F / min lt` is reported beside every
bracket (Frobenius, at least the spectral norm; revision 2 named `||E||_2 / lt_R`), with the dual eigen-residual
`max_k ||G v_k - lambda_k v_k||_2 / lambda_1` (round 4).

`|H|` (Phase 5) is a second surrogate. It is used only as a method check at `P = 33`: it does not enter the Gauss–Newton adoption
statistic, but an invalid `|H|` bracket rejects the method under rule 1 (thirteenth audit round: this sentence said never in a
decision).

## 3. What exists

- Upper bounds of `eta*` from any test matrix (Rayleigh): top-`r` compression (`compute-gauss-newton-restricted-threshold.ts`),
  diagonal threshold `eta_D`, one-dimensional ceiling.
- Lower bound `eta_tr(M)` (`computeTraceWeightedLowerBound`), with `M = I` equal to revision 1's Route A,
  `2 / lambda_max(H/(1 + beta) + H^{-1/2} S_B H^{-1/2}/(1 - beta))`, `S_B = E[dH_i^2]/B` (the identity was checked to `1.3e-15`
  relative by a reviewer). Measured on the 4 800 Phase 5b checkpoints: `eta_tr(I)/eta* >= 0.8` at 5.5 % (`|H|`) and 6.8 %
  (Gauss–Newton); at `P = 3 610`, `beta = 0.9`, in 1 of 8 Phase 4 cells; on a reviewer's random non-commuting instances its
  median is 0.37–0.94 at `beta in {0.5, 0.9}`, depending on the ensemble. It cannot close a bracket on these spectra and is kept only as a logged comparison.
- Exact `eta*` at `P = 33` (dense, Phases 4–5b).
- Gauss–Newton dual spectrum and per-sample factors in the row space (`compute-gauss-newton-dual-spectrum.ts`,
  `compute-student-gauss-newton-dual-spectrum.ts`).
- The sampled second-moment probe (`compute-linearised-second-moment-growth-matrix-free.ts`), not a certificate.

## 4. The bracket

Both sides come from one coarse Lanczos run on `G` and one evaluation of `V` at one matrix.

### 4.1 Upper bound: Rayleigh quotient of a Lanczos vector

`G` acts on `S^R` (dimension `R(R+1)/2`: 205 120 at `R = 640`, 3 277 120 at `R = 2 560`). In the eigenbasis `L0^{-1/2}` is
entrywise division by `sqrt(lt_j + lt_k)`, `H X H` and `[H, [H, X]]` are entrywise, and `V(X)` costs about `3 N R^2 dOut`
multiply-adds with the rank-`dOut` factors (`E[A_i^T (A_i X A_i^T) A_i] - H X H`, scaled by `1/B`). Lanczos with full
reorthogonalisation, started at `L0^{1/2}(I_R)`, for `k` steps (`k = 30`, raised to 60 if the Ritz residual is above `1e-3`).
For the top Ritz vector `y` (unit Frobenius norm), recompute `q = <y, G y>` with a forward error bound `delta_q` (the
dot-product and product rounding bounds of Higham, ch. 3, with the dimensions above), and set

    eta_up = 1 / (q - delta_q)      (q - delta_q <= lambda_max(G), hence eta_up >= eta*).

This needs no convergence: any unit vector gives an upper bound. The Ritz residual and the Kato–Temple bound (when its gap
hypothesis holds) are recorded beside it, as diagnostics.

### 4.2 Lower bound: cone Collatz–Wielandt certificate on `K(eta)`

For `X > 0`, `X - K(eta)(X) > 0` implies `K(eta)(X) <= c X` for some `c < 1`, hence `rho(K(eta)) <= c < 1` (Collatz–Wielandt for a
map that preserves the proper cone `S_+^R`; Vandergraft 1968, Thm 4.2), hence `eta* > eta` by Theorem `thm:G6`(1) and the final equivalence of its proof (`rho(K) < 1` iff `L0 - eta B_beta > 0`,
pointwise in `eta`; with `V = 0` the same chain gives `eta* = eta_det`, outside the hypothesis `V != 0`; eighth audit round).

Witness. `X_raw = L0^{-1/2}(y)` (sign chosen so that `tr X_raw >= 0`), its positive part `X_+` (eigendecomposition of size `R`),
`X = X_+ / ||X_+||_2 + eps I`, `eps = 1e-3` (so `lambda_min(X) >= eps`, `kappa(X) <= 1 + 1/eps`). `W = V(X)` is computed once:
`V` does not depend on `eta`.

Bisection. For `eta in (0, eta_det)`, `K(eta)(X)_jk = (eta/(1 - beta)) W_jk / l_jk(eta)` costs `O(R^2)`; the test is a Cholesky of
`M(eta) - alpha(eta) I`, `M(eta) = fl(X - K(eta)(X))`, with `alpha(eta)` the sum of (i) the forward error of forming `M(eta)`
(entrywise, from the error bound of `W` and of the divisions) and (ii) Rump's shift `(R + 2) u / (1 - (R + 2) u) tr(M(eta))`
(Rump 2006, *BIT* 46:433–452); success of the floating-point Cholesky then proves `X - K(eta)(X) > 0` in exact arithmetic. Bisection
on `eta` (40 steps, relative tolerance `1e-6`) gives `eta_low`, the largest rate at which the test passed. A failure at every
`eta` returns "undefined", never a number.

Cost per checkpoint (estimates, measured on the first checkpoint of each stage before any batch run):

| | `R = 640`, `N = 64` | `R = 2 560`, `N = 256` |
|---|---|---|
| one `V(X)` | ~8e8 flops | ~5e10 flops |
| 30 Lanczos products + reorthogonalisation | ~3e10 | ~1.5e12 |
| Krylov memory (30 vectors) | 49 MB | 790 MB |
| Cholesky per bisection step | 9e7 | 5.6e9 |
| bisection (40 steps) | ~4e9 | ~2.2e11 |

Correction (eighth audit round): `V(X)` also costs `2 R^3` for `S X S`, which equals the sample term at full rank `R = N dOut`, so
one `V(X)` is about `4 N R^2 dOut + 2 N R dOut^2` multiply-adds (1.1e9 at `R = 640`, 6.7e10 at `R = 2 560`), twice with the bound;
the Lanczos code stores full `R^2` vectors, so 30 steps take 98 MB / 1.57 GB and the 120 steps of the adaptive policy 393 MB /
6.3 GB. The last figure bears on stage 3 at `R = 2 560`. The lower bound costs up to 9 witnesses times about 52 factorisations
(468 per bracket, up to 1 404 per adaptive call), not 40: about 7.9e12 flops at `R = 2 560` (ninth audit round).

At `R = 2 560` that is tens of minutes per checkpoint in single-thread JavaScript; stage 3 uses a fixed small set of checkpoints
(section 5).

Reviewer measurements of this construction (not yet ours): on 600 random non-commuting instances (three ensembles, near-degenerate
spectra included, `beta in {0, 0.5, 0.9, 0.99}`, dense exact `eta*` as reference) no violation and `eta_low/eta* >= 0.993` on all;
on an `R = 36` Gauss–Newton instance with eigenvalues across nine decades, 15 Lanczos steps gave `eta_low/eta_up = 0.99994` at
`mu = 0` and `mu = 1e-3`. Reproduced on 2026-10-29 by rerunning `test_routes_and_alternatives.py` (45 s); stage 0 replaces it with
the lab's own code.

### 4.2b As implemented after the audit rounds (section 11)

The construction above is the one written before the code; the code differs where stage 1 or an audit round required it, and the
docblocks state each change:
- upper bound: the Rayleigh quotient of `z = L0^{-1/2}(y)` with forward error bounds on both forms
  (`computeCertifiedRayleighUpperThreshold`), clipped by `eta_det` rounded upwards (`fl(2(1 + beta)/max lt)(1 + 8u)`);
- Lanczos lengths 30, 60, 120: 60 when the Ritz residual is above `1e-3` of the Ritz value or `eta_low / eta_up < 0.98`, 120 when the
  bracket is still below 0.98 (`computeAdaptiveCertifiedThresholdBracket`);
- witness: the ladder `eps = 1e-1, 1e-2, ..., 1e-9`, each witness verified positive definite by the certificate itself, `eta_low`
  the largest rate certified by any of them;
- bisection: 50 steps on `(best so far, eta_up)`;
- verified Cholesky: shift `c = (2 gamma_{n+2} tr+(a)(1 + gamma_n) + rho)(1 + 8u)` and the comparison of
  `verifyPositiveDefiniteShiftedCholesky`, with `rho` the Frobenius norm of the entrywise error bounds of `M(eta)`;
- certified domain: `n <= 4096`, `beta = 0` or `beta in [2^-20, 1 - 2^-20]`, `lt in [2^-60, 2^60]`, `N, B <= 2^20`, `dOut <= 1024`,
  `|x|, |w|, wErr <= 2^300`, `eta >= 2^-200`; outside it, on aliased or shared memory, or on invalid noise output, the functions refuse.

### 4.3 Comparisons logged beside the bracket (no decision rests on them)

- `eta_tr(I)` from the existing function; a new implementation that disagrees with it is a bug.
- Theorem `thm:diag-threshold`(1), the eigenvector split, `(1/eta_D + c''_beta)^{-1}`, after folding clusters closer than
  `(1 - beta)` times the noise scale into the diagonal block (Remark `rem:diag-scope`(iii)); "undefined" when its side condition
  fails. It needs no Krylov iteration on `S^R`, so it is the fallback at sizes where 4.1 is too slow.
- Russo–Dye: `lambda_max(K(eta)^m(I))^{1/m} < 1` for `m = 14` (Lemma `lem:russodye`), inversion-free; `eta_low/eta* >= 0.90` on
  the reviewer's instances. A cross-check of 4.2 that does not use the witness.

## 5. Stages, checkpoint lists and decision rules

Checkpoints are regenerated by rerunning training (deterministic, bit for bit on one platform; the run records do not store
`theta`). The regeneration records the provenance file of the original runs and the commit of the bracket code (ninth audit
round: revision 2 promised the original commit, which the runners did not write) and checks the stored loss trace bit for bit
before computing anything; a mismatch stops that run. The checkpoint lists below are written into `phase6-checkpoints.json` and committed before
any bracket is computed.

**Stage 0, synthetic.** The tests of section 6, plus the reviewer ensembles regenerated in TypeScript.

**Stage 1, `P = 33` (method check against the dense exact value).** Every window checkpoint of the Phase 5b runs (Gauss–Newton
and `|H|`, all `mu_train`, `eta`, seeds; 4 800 checkpoints). Reference: the dense `eta*` of the Phase 5b record.

**Stage 2, `R = 640` (pilot networks `3610` and `12826`).** Seed 0 of every pilot cell with `mu_train = 1e-3` and a finite run, the
window checkpoints of the Gauss–Newton block (every 1 000 steps, 11 per run). Reference: Lanczos on `G` run to Ritz residual
`1e-10` (the same code as 4.1, longer).

**Stage 3, `R = 2 560` (pilot B, `34186`).** Seed 0, `mu_train = 1e-3`, the three activations, `eta in {0.006, 0.025, 0.1}`, the
Gauss–Newton window checkpoints at steps 10 000, 16 000 and 20 000 (27 checkpoints). No reference; the diagnostics of 4.1 (Ritz residual, the
gap `eta_up/eta_low`) and the logged comparisons of 4.3 are reported.

`mu_train = 0` is run in stages 1–2 on the quotient object and reported in a separate table; it enters no adoption decision, and
an invalid bracket there still rejects the method under rule 1 (eleventh audit round: the code has always done so; the sentence said
it entered no decision).

Rules, fixed now:

1. **Validity (any stage, any checkpoint).** If `eta_low > eta*_ref` or `eta*_ref > eta_up` by more than `1e-8` relative, the
   method is rejected and the cause is found before any further run. "Undefined" is counted, never scored as a pass.
2. **Method adoption.** Separately in stage 1 (Gauss–Newton) and stage 2: `eta_low / eta_up >= 0.98` at 90 % of that stage's listed
   checkpoints. Stage 3 runs only if stage 2 adopts.
3. **Per checkpoint, the reading.** *Inside* if `0.8 eta <= eta_low` and `eta_up <= 1.25 eta`; *below* if `eta_up < 0.8 eta`
   (training above the surrogate's threshold, as at `P = 33`, `eta >= 0.3`); *above* if `eta_low > 1.25 eta`; *undecided*
   otherwise. With `eta_low/eta_up >= 0.98` the undecided set is a thin shell around the band edges.
4. **Per run.** The share of window checkpoints in each reading. A run is "at the surrogate's edge" if at least 80 % are *inside*.

Expectations (written before any bracket; not registered; the registration is later, on new seeds):

- `e16-1`: stage 1 adopts (the dense reference makes this a code check).
- `e16-2`: at `R = 640`, `eta*_GN / eta*_GN,64 >= 0.95` at 90 % of stage 2 checkpoints (the top-64 compression was within 1.035 on
  the tethered Phase 4 Hessians).
- `e16-3`: if `e16-2` holds, the stage 2 runs with `eta >= 0.01` and `mu_train = 1e-3` read *inside* at most of their checkpoints,
  since their top-64 ratios were 0.84–1.13; the runs at `eta <= 0.003`, where training is near a minimum, read *above*.
- `e16-4`: unlike `P = 33`, no stage 2 run reads *below* at most of its checkpoints. This is the expectation most likely to fail,
  and its failure would say the larger networks behave like `P = 33`.

## 6. Code and tests (lab rules)

One export per file, generic names (no network or model names), FDLIBM math, docblocks whose backticked identifiers exist,
reference oracles in `.reference.ts`, golden fixtures. Planned files (names may change at review):
`compute-row-space-curvature-factors.ts` (the `A_i` and the residual `E`), `apply-curvature-noise-operator.ts` (`V(X)`),
`apply-mean-square-pencil-operator.ts` (`G`), `compute-rayleigh-upper-threshold.ts` (4.1 with `delta_q`),
`build-cone-witness-from-ritz-vector.ts`, `verify-positive-definite-shifted-cholesky.ts` (Rump), `bisect-cone-lower-threshold.ts`
(4.2), `compute-certified-threshold-bracket.ts`, `regenerate-checkpoint-iterates.ts`, `run-phase6-bracket.ts`.

Tests, each against the dense exact `eta*` (`P <= 6`, `R <= 12`, non-commuting per-sample matrices, near-degenerate spectra,
eigenvalues across eight decades, `beta in {0, 0.5, 0.9, 0.99}`, `B in {1, 16}`):

- `eta_low <= eta* <= eta_up` on every instance, and `eta_low / eta_up >= 0.95` on at least 95 % of them.
- `B` scaling: the `B = 16` threshold from the code equals the dense `B = 16` threshold; a mutation that drops `1/B` fails.
- Mutations that must be caught by finding `eta_low > eta*` on some instance: the sign of the commutator term in `b_jk`; Rump's
  shift set to zero on an instance built with `lambda_min(X - K(X))` below the rounding level; `V` built from `H_i` instead of
  `dH_i`.
- Undefined cases return "undefined": `eta >= eta_det`, `H_it` not positive definite, the witness not positive definite.
- `V(X)` from the rank-`dOut` factors equals the dense `(1/B) mean_i dH_i X dH_i` to `1e-12` relative.
- `eta_tr(I)` from the new code equals `computeTraceWeightedLowerBound` with `M = I`.

Local tests only of the added files, once at the end; the global suite before merge.

## 7. Risks

- The bracket may be tight while the surrogate is the wrong object: the Phase 6 pilots already show the loss linearisation growing where the
  top-64 Gauss–Newton ratio is near 1. The claim stays attached to the surrogate.
- `P = 33` already answers the question for its network in the negative at `eta >= 0.3`; the larger networks may do the same.
- At `R = 2 560` the Lanczos cost may force fewer checkpoints than listed; the list is then cut by a rule written before the cut
  (keep the first checkpoint of each run).
- The witness may fail (no `eta` passes) on spectra with very small `lt_R` at `mu_train = 0`; the reviewer instance passed at
  `lt_R / lt_1 ~ 1e-9`, the network spectra are not tested.
- Everything is on two-layer MLPs with `N <= 256`. A negative outcome (bracket closed with `eta` outside the band, or the method not
  adopted) is reported as such.

## 8. Order of work

Stage 0 code and tests; stage 1; then a decision with the owner before stages 2–3 (Modal cost). Nothing is registered before
stage 2 is read; the registration uses new seeds.

## 9. Review of revision 1 (2026-10-29)

Four blind reviews of revision 1 (reports and scripts kept outside this repository) led to revision 2: the cone Collatz–Wielandt
lower bound of section 4.2 replaced two routes that bounded the threshold from the wrong side or needed infeasible memory, one band
(the H1 band of the Phase 5 registration), one adoption bar per stage, a hard validity rule and fixed checkpoint lists.

## 10. Stage 0 outcome (2026-10-29)

Code: `computeCertifiedThresholdBracket` and its parts (`createDenseDeviationNoiseOperator`, `applyEigenbasisThresholdOperator`,
`computeReorthogonalisedLanczosTopPair`, `computeCertifiedRayleighUpperThreshold`, `buildConeWitnessFromRitzVector`,
`certifyConeCollatzWielandtAtRate`, `bisectConeLowerThreshold`, `verifyPositiveDefiniteShiftedCholesky`, `computeRoundoffGamma`),
the oracle `computeEigenbasisThresholdDenseReference`; tests `tests/phase6-certified-bracket.test.ts` (6, all pass, 2 s).

- The oracle agrees with the spectral radius of the second-moment operator on `S^{2n}` (power iteration): `rho < 1` at
  `0.95 eta*`, `rho > 1` at `1.05 eta*`, for `B` in {1, 16} and `beta` in {0, 0.9}.
- 240 random instances (three ensembles: general non-commuting, near-degenerate pairs `1e-7` apart, rank-1 Gauss–Newton with
  eigenvalues over eight decades and tether `1e-3`; `beta` in {0, 0.5, 0.9, 0.99}; `B` in {1, 16}; `n` from 2 to 6; `eta*` from
  0.002 to 0.97 of `eta_det`): no violation of `eta_low <= eta* <= eta_up`; `eta_low` defined on all 240;
  `eta_low / eta*` from 0.998 to 1.0 in every cell; `eta_up / eta*` equal to 1 to seven digits (the Lanczos runs to the full
  dimension at these sizes).
- With Lanczos cut at 3 steps (unconverged on more than 10 of 60 instances with `n >= 5`) both sides still hold.
- Dimension 1: the bracket encloses the closed form with `eta_low / eta_up > 0.999` for every `beta`.
- Detected by the comparisons: a noise operator built with `B = 1` for a `B = 16` truth (the upper bound falls below `eta*`), and a
  reference with the commutator sign flipped (outside the bracket); `verifyPositiveDefiniteShiftedCholesky` refuses a matrix with
  `lambda_min = 1e-6` when the perturbation radius is `1e-5`.

Not yet tested: sizes where Lanczos stops far from convergence (`R = 640`); the rank-factor noise operator of the Gauss–Newton row
space; the regeneration of checkpoints. Next: stage 1 (`P = 33`, Phase 5b checkpoints, dense reference).

## 11. Audit rounds of the code (2026-10-29 to 2026-10-30)

The stage 0–2 code was run on stage 1 before any audit; that run is kept as `phase6-stage1-v1` and is not evidence of the method.
The code then went through fourteen audit rounds by independent reviewers (reports kept outside this repository). Each round that
found a wrong-side bracket, a false statement or an untested rounding term changed the code or the text, and every change landed
with a test: the `round N` cases of `tests/phase6-certified-bracket.test.ts` are those inputs, judged against exact rational or
dyadic arithmetic, together with the mutations they kill. Sections 4.2b and 6 describe the code as it stands after the last round;
the certified domain (`n <= 4096`, `beta` in `{0}` or `[2^-20, 1 - 2^-20]`, `lt` in `[2^-60, 2^60]`, `N, B <= 2^20`) and the
contract that the certificate accepts only noise operators built by the two library factories (`phase6NoiseOperatorBrand`) are
results of those rounds.

## 12. Stage 1 and stage 2 outcomes (exploratory)

**Stage 1** (`P = 33`, the 48 Phase 5b runs, code `d8101384`, commit `3879bdcc`, `evidence/mean-square-boundary/phase6-stage1-summary.json`).
All 48 regenerations matched the stored loss traces bit for bit. Validity: no violation against the dense reference on any of the
9 600 brackets (2 400 Gauss–Newton at `mu_train = 1e-3`, 2 400 quotient objects at `mu_train = 0`, 4 800 `|H|`); 94 quotient-object
checkpoints lie outside the bracket of the stored Lanczos value, all where that value had not converged, as in v1. Adoption: every
bracket has `eta_low / eta_up >= 0.98` (smallest 0.9934 Gauss–Newton, 0.9802 `|H|`). `e16-1` holds.

**Stage 2** (`R = 640`, 30 pilot runs, 330 Gauss–Newton window checkpoints, bracket code `fb56d61d`, list committed in `fb56d61d`
before the batch, raw records `bc90e7a4`, Modal app `ap-7lpJ2nasOtet9h8BVDCYHh`; summary `evidence/mean-square-boundary/phase6-stage2-summary.json`
by `summarize-phase6-stage2.ts`). The records match the committed list; 0 regeneration mismatches, 0 divergences; rank 640 at every
checkpoint. Validity: every bracket overlaps the reference bracket with no tolerance (0 invalid, 0 missing); the two upper bounds
differ by at most `7.0e-6` relative (median `1.2e-9`). Adoption: `eta_low / eta_up >= 0.98` at every checkpoint (smallest 0.9822,
median 0.9994); relative width at most 1.78 % (median 0.061 %). Factor residuals: `meanResidualFrobeniusOverMinCurvature <= 8.9e-10`,
`dualResidualOverTop <= 1.4e-14`. Bracket time 202–801 s per checkpoint on 1 CPU (median 364 s), reference not included.
Readings (rule 3): all 330, 161 inside, 155 above, 14 below, 0 undecided; at `eta >= 0.01` (198), 161 inside, 23 above, 14 below.
Rule 4: 13 of 30 runs at the surrogate's edge (at least 80 % inside); no run reads below at more than 4 of its 11 checkpoints.
Window medians of `eta_up / eta`: 1.67–23.8 at `eta <= 0.003` (per checkpoint 1.60–26.7, all above); 0.81–1.10 at `eta >= 0.02`
for `tanh` and `softsign` on both networks; `tanh` and `softsign` at `P = 3 610`, `eta = 0.01` read above at every checkpoint
(1.46, 1.81); smallest single checkpoint 0.46 (`P = 12 826`, `softsign`, `eta = 0.05`).
Expectations written before the batch: `e16-2` holds narrowly (`eta_up / eta*_GN,64 >= 0.95` at 90.6 % of the checkpoints, range
0.919–0.983; the top-64 compression overstates the threshold by 2–9 %); `e16-3` fails (two runs with `eta >= 0.01`, `tanh` and
`softsign` at `P = 3 610`, `eta = 0.01`, read above, not inside; the `eta <= 0.003` half holds); `e16-4` holds. Stage 3 may run
under rule 2 (stage 2 adopts); it has not run.
