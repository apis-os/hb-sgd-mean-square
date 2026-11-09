# Checking the numbers of the focused note

The focused note is `paper/note-short.pdf`. Every number it prints comes from two scripts. They need Python 3 and numpy.

```
python3 paper/checks/note-v2-checks.py      # about 4 minutes, all parts
python3 paper/checks/crossover-figure.py    # a few minutes
```

## What each check compares

Each check sets a closed form of the note against a second computation that does not use it.

| Note | Closed form | Independent computation | Script, part |
|---|---|---|---|
| Lemma 8, Theorem 1 | block-diagonal plus rank-one restriction of the second-moment operator; `rho = 1/z*` | the operator `E[F Q F^T]` built by exact enumeration of discrete feature laws; iterated second moments for the prefactor | `note-v2-checks.py` A, C |
| Theorem 2 | slowest row is an extreme row | `rho(M(q))` on a grid of `q` for 2163 random stable settings; `M(q)` against exact enumeration of the row operator | `note-v2-checks.py` I |
| Theorem 3 | `rho(M(q)) >= (d+2-q)/(d+2-q+Bq)`, closed form `z_beta(q,B)` | eigenvalues of `M(q)` at 200000 random `(d, q, B, eta, beta)` | `note-v2-checks.py` J |
| Corollary 4 | `E_beta = 1/ln z_beta(q_min, 1)`, attained at `B = 1` | minimum of `B S_beta(B)` over `B <= 30` from a direct search over `eta` | `note-v2-checks.py` K |
| Corollary 5, Theorem 6(b) | tuned `beta` is 0 for `B - 1 <= B_x` and positive for `B - 1 > B_x` | search over `(eta, beta)` that uses only the eigenvalues of `M(q)` | `crossover-figure.py` |
| Theorem 6(a) | balanced-branch equation for `S_beta(B)` | direct search over `eta` | `note-v2-checks.py` K |
| Theorem 6(c) | `S_mom(B) - S_mom = C_mom B^{-1/2} + O(1/B)` | `sqrt(B)(S_mom(B) - S_mom)` at `B = 10^4 ... 10^7` | `note-v2-checks.py` K |
| Corollary 7, Table 1 | bounds on `mu*` and the threshold at high effective rank | `lambda_max` of the secular matrix | `note-v2-checks.py` H |

Near `beta = 1` the 3x3 matrix `M(q)` is close to a Jordan block, and floating-point eigenvalues lose accuracy there. The searches therefore stay at `beta <= 0.9`, where the tuned `beta` of these examples lies.

## Expected output (excerpt)

```
I. note-short Theorem 2: the slowest row of the Everett-Paquette vocabulary is an extreme row
   M(p) against exact enumeration of T (d=2, B<=2): max rel error 2.0e-14
   2163 random stable settings: 0 with a row slower than both extremes of a frequency interval
J. note-short Theorem 3: rho(M(q)) >= (d+2-q)/(d+2-q+Bq) for every (eta, beta); equality at the best SGD step
   200000 random (d, q, B, eta, beta): min rho/bound = 1.000000000001 (>= 1 up to rounding)
K. note-short Corollary 4 and Theorem 6: E_beta, the curve S_beta(B), and S_mom(B) - S_mom
   Theorem 6(a): 12 random vocabularies, max rel error of ln z (direct search vs formula) = 1.1e-12
   Corollary 4: 8 vocabularies, B <= 30: max rel |min_B B S_beta(B) - 1/ln z_beta(q_min,1)| = 3.7e-12
   Theorem 6(c), d=8, q=(10/11, 1/11): B=10000000  sqrt(B)(S_mom(B)-S_mom) = 6.6708  (C_mom = 6.6601)
```

`crossover-figure.py`, two-row vocabulary `d = 8`, `q = (10/11, 1/11)`, `B_x = 110/9`:

```
B= 13  S_SGD formula    8.8752  search    8.8752  S_mom search    8.8752  tuned beta 0.000
B= 14  S_SGD formula    8.2824  search    8.2824  S_mom search    8.2763  tuned beta 0.023
max rel |S_SGD formula - search| = 1.2e-14
B with tuned beta > 1e-3: [14, 15, 16] ...
```

The numerical checks test the statements on many cases. They are not proofs; the proofs are in Appendix A of the note.
