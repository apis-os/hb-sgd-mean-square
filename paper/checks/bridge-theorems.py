"""Numerical checks of Theorems modal-bridge and slow-fast-saturation (numpy only, exact enumeration where possible).

Prints each check and exits with status 1 if a residual exceeds its tolerance. Run: python3 paper/checks/bridge-theorems.py
(1) modal interpolation identity, C_mod sandwich, variational bound at S = I and S = u1 u1^T, by exact enumeration of
    Rademacher and three-point feature laws (P = 3, B = 1..3, beta in {0, 0.5, 0.9});
(2) pole decomposition of Sigma_0(eta) and the bound on R_perp; d_star against a finite difference of rho(K_eta);
(3) the closed form of BS* in the random-feature model against exact enumeration, and its two plateau limits on
    lambda_k = k^{-1/2}, P = 4096, Gaussian features;
(4) part (3): the cubic moment of Delta with the (2 + kappa) s_m sum s_q^2 term, the bound on |BS*| and the
    O(sqrt(lambda_k)) collapse;
(5) part (1)(b): E||G_Z - I||_F^2 = B (B + 1 + kappa) tr(H^2)/(tr H)^2 and |BS(theta) - tr H / B| <= (tr H / B) delta_Gram;
(6) the saturation law of Theorem slow-fast-saturation on a scalar-curvature family: the root of the slow drift against
    eta (1 + c_bar sigma^2 / v) to first order.
"""
import itertools
import sys

import numpy as np

FAIL = []


def check(name, value, tol):
    ok = abs(value) <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {value:.3e} (tol {tol:.0e})")
    if not ok:
        FAIL.append(name)


def sym_basis(P):
    out = []
    for j in range(P):
        for k in range(j, P):
            E = np.zeros((P, P))
            if j == k:
                E[j, j] = 1.0
            else:
                E[j, k] = E[k, j] = 1 / np.sqrt(2)
            out.append(E)
    return out


def matrix_of(op, basis):
    return np.array([[np.sum(b1 * op(b2)) for b2 in basis] for b1 in basis])


def law_hessians(H, sz, vals, pv, B, centred):
    """Exact minibatch Hessians and probabilities: H_B = H + (1/B) sum (z z^T - Sz) with z = Sz^{1/2} w (centred), or
    H_B = (1/B) sum z z^T with z = H^{1/2} w (random features)."""
    P = H.shape[0]
    feats, probs = [], []
    for t in itertools.product(range(len(vals)), repeat=P):
        feats.append(np.array([vals[i] for i in t]))
        probs.append(np.prod([pv[i] for i in t]))
    scale = np.sqrt(np.diag(sz)) if centred else np.sqrt(np.diag(H))
    singles = [np.outer(scale * w, scale * w) - (sz if centred else 0) for w in feats]
    HB, PB = [], []
    for idx in itertools.product(range(len(singles)), repeat=B):
        m = sum(singles[i] for i in idx) / B
        HB.append(H + m if centred else m)
        PB.append(np.prod([probs[i] for i in idx]))
    return HB, PB


def threshold(H, HB, PB, beta):
    P = H.shape[0]
    V = lambda X: sum(p * (h - H) @ X @ (h - H) for p, h in zip(PB, HB))
    L0 = lambda X: H @ X + X @ H
    comm = lambda X: H @ (H @ X - X @ H) - (H @ X - X @ H) @ H
    Bb = lambda X: H @ X @ H / (1 + beta) - beta / ((1 - beta) * (1 - beta ** 2)) * comm(X) + V(X) / (1 - beta)
    basis = sym_basis(P)
    ML, MB = matrix_of(L0, basis), matrix_of(Bb, basis)
    w, U = np.linalg.eigh(ML)
    Lh = U @ np.diag(w ** -0.5) @ U.T
    ev, EV = np.linalg.eigh(Lh @ MB @ Lh)
    x = Lh @ EV[:, -1]
    S = sum(c * b for c, b in zip(x, basis))
    S = S / np.trace(S)
    return 1 / ev[-1], S, V, Bb, L0, basis, ML, MB, ev


def part1_and_3():
    lam = np.array([1.0, 0.6, 0.25])
    H = np.diag(lam)
    laws = {'rademacher': ([-1.0, 1.0], [0.5, 0.5], -2.0, 1.0),
            'three-point': ([-np.sqrt(2), 0.0, np.sqrt(2)], [0.25, 0.5, 0.25], -1.0, 4.0)}
    for lname, (vals, pv, kf, m6) in laws.items():
        for B in (1, 2, 3) if lname == 'rademacher' else (1, 2):
            for beta in (0.0, 0.5, 0.9):
                HB, PB = law_hessians(H, None, vals, pv, B, centred=False)
                eta, S, V, Bb, L0, basis, *_ = threshold(H, HB, PB, beta)
                tHS = np.trace(H @ S)
                a = np.trace(H @ H @ S) / tHS
                v = np.trace(V(S)) / tHS
                EH2 = sum(p * h @ h for p, h in zip(PB, HB))
                EH3 = sum(p * h @ h @ h for p, h in zip(PB, HB))
                tag = f"{lname} B={B} beta={beta}"
                check(f"(1) modal identity {tag}", 1 / eta - (a / (2 * (1 + beta)) + v / (2 * (1 - beta))), 1e-12)
                Cmod = np.trace(EH2 @ S) / tHS
                check(f"(1) C_mod in sandwich {tag}", max(0.0, 2 * (1 - beta) / eta - Cmod - 1e-12, Cmod - 2 * (1 + beta) / eta - 1e-12), 0)
                for Sname, Sv in (('I', np.eye(3)), ('u1u1', np.diag([1.0, 0, 0]))):
                    X = np.sqrt(Sv)
                    bound = (np.trace(H @ H @ Sv) / np.trace(H @ Sv)) / (2 * (1 + beta)) + (np.sum(X * V(X)) / np.trace(H @ Sv)) / (2 * (1 - beta))
                    check(f"(1) variational bound S={Sname} {tag} (1/eta* - bound, must be >= 0)", min(0.0, 1 / eta - bound + 1e-12), 0)
                BS = np.trace(EH3 @ S) / np.trace(EH2 @ S)
                check(f"(2) BS* >= C_mod {tag}", min(0.0, BS - Cmod + 1e-12), 0)
                q = 1 + (1 + kf) / B
                r = 1 + 3 * (1 + kf) / B + (m6 - 11 - 6 * kf) / B ** 2
                trH, trH2 = lam.sum(), (lam ** 2).sum()
                m2 = np.trace(H @ H @ H @ S) / tHS
                kb = (1 + beta) / (1 - beta)
                ad = 1 + kb * (1 + kf) / B
                check(f"(2) moment identity {tag}", ad * (m2 - a ** 2) - kb / B * (trH * a - trH2), 1e-12)
                BSf = Cmod + (q / B * trH2 + q ** 2 * (m2 - a ** 2) + (r - q ** 2) * m2) / (trH / B + q * a)
                check(f"(2) closed form of BS* {tag}", BSf - BS, 1e-12)


def part2_pole_and_dstar():
    lam = np.array([1.0, 0.6, 0.25])
    H = np.diag(lam)
    HB, PB = law_hessians(H, None, [-np.sqrt(2), 0.0, np.sqrt(2)], [0.25, 0.5, 0.25], 2, centred=False)
    beta = 0.5
    eta_s, S, V, Bb, L0, basis, ML, MB, ev = threshold(H, HB, PB, beta)
    Sxi = np.diag([1.0, 0.5, 0.2])
    sx = np.array([np.sum(b * Sxi) for b in basis])
    gap = ev[-1] - ev[-2]
    for frac in (0.5, 0.9, 0.99):
        eta = frac * eta_s
        x = np.linalg.solve(ML - eta * MB, eta / (1 - beta) * sx)
        Sig = sum(c * b for c, b in zip(x, basis))
        coef = eta * np.trace(S @ Sxi) / (2 * (1 - beta) * np.trace(H @ S @ S) * (1 - eta / eta_s))
        R = Sig - coef * S
        Rv = np.array([np.sum(b * R) for b in basis])
        nR = np.sqrt(Rv @ ML @ Rv)
        w, U = np.linalg.eigh(ML)
        nL = np.linalg.norm(U @ np.diag(w ** -0.5) @ U.T @ sx)
        check(f"(2) R_perp orthogonal to Sigma* in <.,.>_L0, eta={frac}eta*", Rv @ ML @ np.array([np.sum(b * S) for b in basis]), 1e-10)
        check(f"(2) ||R_perp|| bound, eta={frac}eta* (excess, must be <= 0)", max(0.0, nR - nL / ((1 - beta) * gap)), 0)

    def rhoK(eta):
        P = 3
        Lmat = lambda X: H @ X + X @ H - eta / (1 + beta) * H @ X @ H + eta * beta / ((1 - beta) * (1 - beta ** 2)) * (H @ (H @ X - X @ H) - (H @ X - X @ H) @ H)
        ML_ = matrix_of(Lmat, basis)
        MV = matrix_of(V, basis)
        return max(np.linalg.eigvals(eta / (1 - beta) * np.linalg.solve(ML_, MV)).real)
    h = 1e-6 * eta_s
    d_fd = eta_s * (rhoK(eta_s + h) - rhoK(eta_s - h)) / (2 * h)
    d_formula = 2 * (1 - beta) * np.trace(H @ S @ S) / (eta_s * np.sum(S * V(S)))
    check("(2) d_star formula against finite difference (relative)", (d_fd - d_formula) / d_formula, 1e-6)


def plateaus():
    P = 4096
    lam = np.arange(1, P + 1) ** -0.5
    trH, trH2 = lam.sum(), (lam ** 2).sum()
    out = {}
    for B in (1, 10 ** 7):
        beta, kf, m6 = 0.9, 0.0, 15.0
        kb = (1 + beta) / (1 - beta)
        ad = 1 + kb * (1 + kf) / B
        f = lambda e: e / (2 * (1 - beta) * B) * np.sum(lam / (1 - e * lam * ad / (2 * (1 + beta)))) - 1
        lo, hi = 1e-12, 2 * (1 + beta) / (lam[0] * max(ad, 1)) * (1 - 1e-14)
        for _ in range(300):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if f(mid) > 0 else (mid, hi)
        eta = 0.5 * (lo + hi)
        mu = 2 * (1 + beta) / eta
        g = 1 / (mu - ad * lam)
        g /= g.sum()
        tHS = (lam * g).sum()
        a, m2 = (lam ** 2 * g).sum() / tHS, (lam ** 3 * g).sum() / tHS
        Cmod = a + trH / B + (1 + kf) / B * a
        q, r = 1 + (1 + kf) / B, 1 + 3 * (1 + kf) / B + (m6 - 11 - 6 * kf) / B ** 2
        BS = Cmod + (q / B * trH2 + q ** 2 * (m2 - a ** 2) + (r - q ** 2) * m2) / (trH / B + q * a)
        out[B] = (BS * eta / (2 * (1 - beta)), BS * eta / (2 * (1 + beta)))
    print(f"     plateaus (P=4096, beta=0.9, Gaussian): B=1 BS*/(2(1-b)/eta*)={out[1][0]:.4f}; B=1e7 BS*/(2(1+b)/eta*)={out[10**7][1]:.4f}")
    check("(2) small-batch plateau B=1 (BS*/(2(1-b)/eta*) - 1)", out[1][0] - 1, 3e-3)
    check("(2) large-batch plateau B=1e7 (BS*/(2(1+b)/eta*) - 1)", out[10 ** 7][1] - 1, 1e-3)


def part3():
    p = 1 / 9
    a = np.sqrt(1 / p)
    vals, pv = [-a, 0.0, a], [p / 2, 1 - p, p / 2]
    m4, m6 = 1 / p, 1 / p ** 2
    kf = m4 - 3
    c3 = m6 - 3 * m4 + 2
    s = np.array([0.3, 0.7, 1.1])
    P = 3
    E3 = np.zeros((P, P))
    for t in itertools.product(range(3), repeat=P):
        w = np.array([vals[i] for i in t])
        pr = np.prod([pv[i] for i in t])
        D = np.outer(np.sqrt(s) * w, np.sqrt(s) * w) - np.diag(s)
        E3 += pr * D @ D @ D
    for m in range(P):
        o = [q for q in range(P) if q != m]
        formula = s[m] * sum(s[x] * s[y] for x in o for y in o if x != y) + (2 + kf) * s[m] * sum(s[q] ** 2 for q in o) + 2 * (2 + kf) * s[m] ** 2 * sum(s[q] for q in o) + c3 * s[m] ** 3
        check(f"(3) cubic moment of Delta, mode {m}, kappa={kf:.0f}", E3[m, m] - formula, 1e-10)
    k6t = abs(c3) / (2 + kf)
    ratios = []
    for lk in (1e-2, 1e-3, 1e-4):
        lam = np.array([1.0, 0.5, lk])
        sz = np.diag([0.5, 0.5, 0.5])
        HB, PB = law_hessians(np.diag(lam), sz, [-1.0, 1.0], [0.5, 0.5], 1, centred=True)
        eta, S, *_ = threshold(np.diag(lam), HB, PB, 0.5)
        EH2 = sum(pp * h @ h for pp, h in zip(PB, HB))
        EH3 = sum(pp * h @ h @ h for pp, h in zip(PB, HB))
        BS = np.trace(EH3 @ S) / np.trace(EH2 @ S)
        M0 = max(lam + 0.5)
        bound = (3 + max(P - 2.0, 2 * (P - 1) + 0.0) / 1) * M0
        check(f"(3) |BS*| bound, lambda_k={lk:g} (excess, must be <= 0)", max(0.0, abs(BS) - bound), 0)
        ratios.append(abs(BS) * eta / (2 * (1 - 0.5)))
    print(f"     collapse |BS*|/(2(1-b)/eta*) at lambda_k = 1e-2, 1e-3, 1e-4: {ratios[0]:.4f} {ratios[1]:.4f} {ratios[2]:.4f}")
    check("(3) collapse rate ratio(1e-3)/ratio(1e-2) - 1/sqrt(10)", ratios[1] / ratios[0] - 10 ** -0.5, 0.03)
    check("(3) collapse rate ratio(1e-4)/ratio(1e-3) - 1/sqrt(10)", ratios[2] / ratios[1] - 10 ** -0.5, 0.03)


def gram():
    rng = np.random.default_rng(0)
    P = 200
    lam = np.arange(1, P + 1) ** -0.5
    trH, trH2 = lam.sum(), (lam ** 2).sum()
    for B, kind, kf in ((4, 'gauss', 0.0), (4, 'rad', -2.0)):
        n = 20000
        dev, bs = [], []
        r = rng.standard_normal(B)
        for _ in range(n):
            w = rng.standard_normal((P, B)) if kind == 'gauss' else rng.choice([-1.0, 1.0], (P, B))
            Z = np.sqrt(lam)[:, None] * w
            G = Z.T @ Z / trH
            dev.append(np.sum((G - np.eye(B)) ** 2))
            g = Z @ r / B
            bs.append(g @ (Z @ Z.T / B) @ g / (g @ g))
        d2 = B * (B + 1 + kf) * trH2 / trH ** 2
        check(f"(1b) E||G_Z - I||_F^2 against formula, B={B} {kind} (relative, Monte Carlo)", (np.mean(dev) - d2) / d2, 0.03)
        check(f"(1b) |BS - trH/B| <= (trH/B) delta, B={B} {kind} (excess)", max(0.0, abs(np.mean(bs) - trH / B) - trH / B * np.sqrt(d2)), 0)


def covariance_identity():
    """Theorem modal-bridge(1)(a): ratio of expectations minus expectation of the ratio equals Cov(R_B, W_B), bounded by
    sigma_BS * CV_{g^2}; arbitrary symmetric minibatch Hessians and gradients, exact finite law."""
    rng = np.random.default_rng(1)
    P, n = 5, 40
    Hs = [(lambda a: (a + a.T) / 2)(rng.standard_normal((P, P))) for _ in range(n)]
    gs = [rng.standard_normal(P) * (1 + 2 * rng.random()) for _ in range(n)]
    pr = rng.random(n); pr /= pr.sum()
    num = sum(p * g @ h @ g for p, g, h in zip(pr, gs, Hs)); den = sum(p * g @ g for p, g in zip(pr, gs))
    R = np.array([g @ h @ g / (g @ g) for g, h in zip(gs, Hs)]); W = np.array([g @ g for g in gs]) / den
    cov = np.sum(pr * R * W) - np.sum(pr * R) * np.sum(pr * W)
    check("(1a) ratio of expectations - expectation of ratio = Cov(R, W)", num / den - np.sum(pr * R) - cov, 1e-12)
    sd = np.sqrt(np.sum(pr * (R - np.sum(pr * R)) ** 2)); cv = np.sqrt(np.sum(pr * (W - 1) ** 2))
    check("(1a) |Cov(R, W)| <= sigma_BS * CV (excess)", max(0.0, abs(cov) - sd * cv), 0)


def saturation():
    """Scalar curvature family H(y) = h(y), one mode, curvature noise of variance nu^2: the stationary variance is
    explicit, so the root of G(y) = -v + 0.5 * Phi * Sigma_0(y) can be compared with the first-order law."""
    beta, eta, nu2, B = 0.5, 0.6, 0.3, 1
    kb = (1 + beta) / (1 - beta)
    h = lambda y: 1.0 + y            # curvature rises with y (progressive sharpening)
    Phi = 1.0                         # dH/dy
    eta_star = lambda y: 2 * (1 + beta) * h(y) / (h(y) ** 2 + kb * nu2 / B)

    def sigma0(y, sxi):
        hh = h(y)
        f = eta * hh / (2 * (1 + beta))
        a = 1 - 4 * (1 - beta ** 2) * f * (1 - f) + eta ** 2 * nu2 / B
        return eta ** 2 * sxi / (1 - a)
    lo, hi = 0.0, 10.0               # eta_star decreases in y on [0, 10]; y* solves eta_star(y*) = eta
    assert eta_star(lo) > eta > eta_star(hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if eta_star(mid) > eta else (lo, mid)
    ystar = 0.5 * (lo + hi)
    for sxi, v in ((1e-4, 1.0), (1e-5, 1.0)):
        G = lambda y: -v + 0.5 * Phi * sigma0(y, sxi)
        lo, hi = 0.0, ystar - 1e-12
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if G(mid) < 0 else (lo, mid)
        yeq = 0.5 * (lo + hi)
        gap = (eta_star(yeq) - eta) / eta
        S0 = 1.0
        cbar = eta * Phi * S0 / (4 * (1 - beta) * h(ystar) * S0 ** 2)
        law = cbar * sxi / v
        check(f"(6) saturation law, sigma^2/v={sxi / v:g}: (gap - c_bar sigma^2/v)/(c_bar sigma^2/v)", (gap - law) / law, 0.02)


part1_and_3()
part2_pole_and_dstar()
plateaus()
part3()
gram()
covariance_identity()
saturation()
if FAIL:
    print(f"{len(FAIL)} check(s) failed")
    sys.exit(1)
print("all checks passed")
