"""Theorem 3 on a two-dimensional example with non-commuting minibatch Hessians (numpy only).

H_+ = [[3, 2], [2, 2]] and H_- = [[3, -2], [-2, 2]], each drawn with probability 1/2, independently at every step; both are
positive definite, they do not commute with each other or with their mean H = diag(3, 2); beta = 0.9, no additive noise.
The threshold is computed twice:
(a) Theorem 3: 1 / lambda_max(L0^{-1/2} B_beta L0^{-1/2}) on the symmetric 2 x 2 matrices (orthonormal basis, dimension 3);
(b) independently: bisection on the spectral radius of E[F_t (x) F_t] (16 x 16) for the lagged state (w_t, w_{t-1}).
Prints both thresholds, the one-dimensional threshold 2(1 + beta)/lambda_1 and rho(E[F (x) F]) at 0.95, 1 and 1.05 times
the threshold; exits with status 1 if (a) and (b) differ by more than 1e-10 relative or the radii are on the wrong side
of 1. Run: python3 paper/checks/noncommuting-2x2.py
"""
import sys

import numpy as np

H_PLUS = np.array([[3.0, 2.0], [2.0, 2.0]])
H_MINUS = np.array([[3.0, -2.0], [-2.0, 2.0]])
HESSIANS = (H_PLUS, H_MINUS)
H = (H_PLUS + H_MINUS) / 2
BETA = 0.9
S = 1 / np.sqrt(2)
BASIS = [np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, S], [S, 0.0]]), np.array([[0.0, 0.0], [0.0, 1.0]])]


def l0(x):
    return H @ x + x @ H


def b_beta(x):
    comm = H @ x - x @ H
    double_comm = H @ comm - comm @ H
    noise = sum((hb - H) @ x @ (hb - H) for hb in HESSIANS) / len(HESSIANS)
    return H @ x @ H / (1 + BETA) - BETA / ((1 - BETA) * (1 - BETA**2)) * double_comm + noise / (1 - BETA)


def matrix_of(op):
    return np.array([[np.sum(ei * op(ej)) for ej in BASIS] for ei in BASIS])


def threshold_theorem3():
    w, u = np.linalg.eigh(matrix_of(l0))
    l0_inv_half = u @ np.diag(w**-0.5) @ u.T
    return 1 / np.linalg.eigvalsh(l0_inv_half @ matrix_of(b_beta) @ l0_inv_half).max()


def rho_full(eta):
    eye, zero = np.eye(2), np.zeros((2, 2))
    t = np.zeros((16, 16))
    for hb in HESSIANS:
        f = np.block([[(1 + BETA) * eye - eta * hb, -BETA * eye], [eye, zero]])
        t += np.kron(f, f) / len(HESSIANS)
    return np.abs(np.linalg.eigvals(t)).max()


def threshold_full(lo=1e-6, hi=1.0):
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if rho_full(mid) < 1 else (lo, mid)
    return lo


eta_thm3 = threshold_theorem3()
eta_full = threshold_full()
eta_1d = 2 * (1 + BETA) / np.linalg.eigvalsh(H).max()
radii = [rho_full(c * eta_thm3) for c in (0.95, 1.0, 1.05)]
rel = abs(eta_thm3 - eta_full) / eta_full
print(f"Theorem 3 threshold        {eta_thm3:.12f}")
print(f"rho(E[F (x) F]) = 1 at     {eta_full:.12f}  (relative difference {rel:.1e})")
print(f"one-dimensional threshold  {eta_1d:.6f}  ({eta_1d / eta_thm3:.2f} times larger)")
print(f"rho at 0.95, 1, 1.05 x     {radii[0]:.6f} {radii[1]:.6f} {radii[2]:.6f}")
ok = rel < 1e-10 and radii[0] < 1 < radii[2] and abs(radii[1] - 1) < 1e-9
print("ok" if ok else "FAIL")
sys.exit(0 if ok else 1)
