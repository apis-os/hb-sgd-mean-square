"""Numerical checks for paper/note-v2.tex (run: python3 paper/checks/note-v2-checks.py).

Every number quoted in note-v2.tex is printed by this script. The operator T(Q) = E[F Q F^T] is built
either exactly, by enumerating a discrete feature law (A, B), or from the formula of Lemma 4 (C, D, E).
Requires numpy only.
"""
import itertools
import numpy as np

# ---------------------------------------------------------------- helpers
def three_point(m4):
    """Symmetric law on {-a, 0, a} with mean 0, variance 1, fourth moment m4 >= 1."""
    a = np.sqrt(m4)
    return np.array([-a, 0.0, a]), np.array([1 / (2 * m4), 1 - 1 / m4, 1 / (2 * m4)])

def kron_T_exact(lam, eta, beta, B, sample_law):
    """T as a matrix on vec(R^{2P x 2P}); sample_law = list of (z, prob) for one feature vector z."""
    P = len(lam); n = 2 * P; M = np.zeros((n * n, n * n))
    for combo in itertools.product(range(len(sample_law)), repeat=B):
        HB = np.zeros((P, P)); pr = 1.0
        for c in combo:
            z, q = sample_law[c]; HB += np.outer(z, z) / B; pr *= q
        F = np.zeros((n, n)); F[:P, :P] = (1 + beta) * np.eye(P) - eta * HB
        F[:P, P:] = -beta * np.eye(P); F[P:, :P] = np.eye(P)
        M += pr * np.kron(F, F)
    return M

def law_independent(lam, m4s, gate_q=None):
    """z = xi H^{1/2} omega, omega_k independent three-point laws, xi = s/sqrt(q) with s ~ Bernoulli(q) (or xi = 1)."""
    laws = [three_point(m) for m in m4s]
    out = []
    for o in itertools.product(range(3), repeat=len(lam)):
        w = np.array([laws[k][0][o[k]] for k in range(len(lam))])
        pr = np.prod([laws[k][1][o[k]] for k in range(len(lam))])
        z = np.sqrt(lam) * w
        if gate_q is None:
            out.append((z, pr))
        else:
            out.append((z / np.sqrt(gate_q), pr * gate_q))
            out.append((0 * z, pr * (1 - gate_q)))
    return out

def T_diag(lam, eta, beta, p, gam, kaps):
    """Restriction of T to the per-mode 2x2 blocks (coordinates E w_k^2, E w_k w'_k, E w'_k^2), Lemma 4."""
    lam = np.asarray(lam, float); kaps = np.asarray(kaps, float); P = len(lam)
    M = np.zeros((3 * P, 3 * P)); a = 1 + beta - eta * lam; I = 3 * np.arange(P)
    M[I, I] = a * a + eta**2 * p * (gam * (2 + kaps) - 1) * lam**2
    M[I, I + 1] = -2 * a * beta; M[I, I + 2] = beta**2
    M[I + 1, I] = a; M[I + 1, I + 1] = -beta; M[I + 2, I] = 1
    M[np.ix_(I, I)] += eta**2 * p * gam * np.outer(lam, lam)
    return M

rad = lambda A: max(abs(np.linalg.eigvals(A)))

def secular_S(eta, lam, beta, p, gam, kaps):
    kb = (1 + beta) / (1 - beta); al = 1 + p * kb * (gam * (2 + np.asarray(kaps)) - 1)
    return eta * p * gam / (2 * (1 - beta)) * np.sum(lam / (1 - eta * lam * al / (2 * (1 + beta))))

def bisect(f, lo, hi, it=80):
    for _ in range(it):
        m = (lo + hi) / 2; lo, hi = (m, hi) if f(m) else (lo, m)
    return (lo + hi) / 2

def psi_hat(z, l, eta, beta):
    a = 1 + beta - eta * l
    return eta**2 * (1 + beta * z) / ((1 - beta * z) * (1 - (a * a - 2 * beta) * z + beta**2 * z * z))

def h_hat(z, l, eta, beta, p, gam, k):
    ps = psi_hat(z, l, eta, beta); return ps / (1 - p * (gam * (2 + k) - 1) * l * l * z * ps)

def K_hat(z, lam, eta, beta, p, gam, kaps):
    return p * gam * sum(l * l * h_hat(z, l, eta, beta, p, gam, k) for l, k in zip(lam, kaps))

def smallest_root(f, zmax, n=400001):
    zs = np.linspace(1e-6, zmax, n); v = f(zs) - 1
    i = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]
    return bisect(lambda z: f(z) < 1, zs[i[0]], zs[i[0] + 1], 60)

# ---------------------------------------------------------------- A. Lemma 4 by exact enumeration
print("A. Lemma 4 / Theorem 5(a,b): exact enumeration vs. formulas")
rng = np.random.default_rng(0); worst = [0.0, 0.0]
for case in range(8):
    P = 2; lam = np.sort(rng.uniform(0.2, 1, P))[::-1]; beta = float(rng.uniform(0, 0.8)); B = int(rng.integers(1, 3))
    gate = None if case % 2 == 0 else float(rng.choice([0.3, 0.6]))
    m4s = rng.choice([1.0, 2.0, 3.0, 6.0], P)            # kappa_k = m4 - 3 in {-2,-1,0,3}
    gam = 1.0 if gate is None else 1 / gate; kaps = m4s - 3
    law = law_independent(lam, m4s, gate)
    eta = float(rng.uniform(0.1, 1.2))
    rT = rad(kron_T_exact(lam, eta, beta, B, law)); rD = rad(T_diag(lam, eta, beta, 1 / B, gam, kaps))
    worst[0] = max(worst[0], abs(rT - rD) / rT)
    stable = lambda e: rad(kron_T_exact(lam, e, beta, B, law)) < 1
    es = bisect(stable, 1e-4, 2 * (1 + beta) / lam[0], 40)
    worst[1] = max(worst[1], abs(secular_S(es, lam, beta, 1 / B, gam, kaps) - 1))
print(f"   max rel |rho(T) - rho(T_diag)| = {worst[0]:.1e};  max |S(eta*) - 1| at the exact crossing = {worst[1]:.1e}")

# ---------------------------------------------------------------- B. Corollary 6: Everett-Paquette eq. (6)
print("B. Corollary 6: isotropic gated features recover Everett & Paquette (2026), eq. (6)")
for (d, B, q, beta) in [(2, 1, 0.5, 0.6), (3, 2, 0.3, 0.8), (2, 2, 0.7, 0.0)]:
    lam = q * np.ones(d); law = law_independent(np.ones(d), 3 * np.ones(d), q)   # H = q I, Gaussian-moment omega
    law = [(z * np.sqrt(q) * 1.0, pr) for z, pr in law]                              # z = s * omega (unnormalised gate)
    stable = lambda e: rad(kron_T_exact(lam, e, beta, B, law)) < 1
    eta_hb = bisect(stable, 1e-4, 2 * (1 + beta) / q, 45)
    eta_ema_eq6 = 1 / ((d + 2 - q) / (2 * B) + q / 2 * (1 - beta) / (1 + beta))
    print(f"   d={d} B={B} q={q} beta={beta}: exact eta_EMA* = {eta_hb / (1 - beta):.6f}   eq.(6) = {eta_ema_eq6:.6f}")

# ---------------------------------------------------------------- C. Theorem 5(c,d,e): rate, prefactor, slowing down
print("C. Theorem 5: rho(T) = 1/z*, prefactor C, critical slowing down")
cases = [([1, 0.4, 0.1], 0.5, 0.25, 1.0, [0, 0, 0]), ([1, 0.6, 0.3], 0.9, 0.5, 1.0, [2, -1, 0]),
         ([1, 0.5], 0.0, 1.0, 1.0, [-1, -1]), ([1, 0.7, 0.2], 0.7, 0.5, 2.0, [0, 0, 0])]
for lam, beta, p, gam, kaps in cases:
    lam = np.array(lam, float); kaps = np.array(kaps, float); P = len(lam)
    es = bisect(lambda e: rad(T_diag(lam, e, beta, p, gam, kaps)) < 1, 1e-4, 2 * (1 + beta) / lam[0])
    eta = 0.7 * es; M = T_diag(lam, eta, beta, p, gam, kaps); rho = rad(M)
    f = lambda z: z * K_hat(z, lam, eta, beta, p, gam, kaps)
    zs = smallest_root(f, 1.5 / rho)
    sig = np.linspace(0.5, 1.5, P); g = np.zeros(3 * P); g[3 * np.arange(P)] = eta**2 * sig
    x = np.zeros(3 * P); ls = []
    for t in range(3000):
        ls.append(lam @ x[3 * np.arange(P)]); x = M @ x + g
    linf = lam @ np.linalg.solve(np.eye(3 * P) - M, g)[3 * np.arange(P)]
    T0 = int(np.log(1e-6) / np.log(rho)); Cm = np.median([(linf - ls[t]) / rho**t for t in range(T0, T0 + 20)])
    phi = lambda z: sum(l * s * h_hat(z, l, eta, beta, p, gam, k) for l, s, k in zip(lam, sig, kaps))
    dz = 1e-6; dzK = (f(zs + dz) - f(zs - dz)) / (2 * dz)
    Cp = phi(zs) / ((zs - 1) * dzK)
    S1 = lambda e: secular_S(e, lam, beta, p, gam, kaps)
    num = (S1(es + 1e-6) - S1(es - 1e-6)) / 2e-6
    den = (1 * K_hat(1 + 1e-6, lam, es, beta, p, gam, kaps) * (1 + 1e-6) - (1 - 1e-6) * K_hat(1 - 1e-6, lam, es, beta, p, gam, kaps)) / 2e-6
    d = 1e-4 * es; slope = (1 - rad(T_diag(lam, es - d, beta, p, gam, kaps))) / d
    print(f"   rho={rho:.6f} 1/z*={1/zs:.6f} | C measured={Cm:.6f} predicted={Cp:.6f} | slope measured={slope:.5f} predicted={num/den:.5f}")

# ---------------------------------------------------------------- D. Proposition 7: one-direction test vs. true threshold
print("D. Proposition 7: P=256, lam_k=k^-1/2, beta=0.9, B=4")
P = 256; lam = np.arange(1, P + 1) ** -0.5; beta = 0.9; B = 4; p = 1 / B; kb = (1 + beta) / (1 - beta); reff = lam.sum() / lam[0]
etas = {}
for kap in (-2.0, -1.0, 0.0, 2.0, 10.0):
    al = 1 + p * kb * (1 + kap)
    ok = lambda e: np.all(e * lam * max(al, 0) < 2 * (1 + beta)) and secular_S(e, lam, beta, p, 1.0, kap * np.ones(P)) < 1
    es = bisect(ok, 1e-9, 2 * (1 + beta) / (lam[0] * max(al, 1e-12)) if al > 0 else 10.0, 200)
    e1 = 2 * (1 + beta) / (lam[0] * (1 + p * kb * (2 + kap))); etas[kap] = es
    mu, lo_, hi_ = 2 * (1 + beta) / es, p * kb * lam.sum() - abs(al) * lam[0], p * kb * lam.sum() + abs(al) * lam[0]
    print(f"   kappa={kap:5.1f}: eta*={es:.6f}  eta_1D={e1:.6f}  ratio={e1/es:8.2f}  bound check {lo_:.3f} <= mu*={mu:.3f} <= {hi_:.3f}")
print(f"   r_eff = {reff:.4f};  Rademacher/Gaussian eta* = {etas[-2.0]/etas[0.0]:.6f};  Rademacher/Gaussian eta_1D = {(1 + p * kb * 2) / (1 + p * kb * 0):.4f}")

# ---------------------------------------------------------------- E. Proposition 8: fixed effective step
print("E. Proposition 8: H=diag(1,0.3), H_pm = H pm N, Sigma_xi = v v^T + 1e-3 I, eta_eff = 0.52")
def sigma0(H, Hs, Sxi, eta, beta):
    P = len(H); basis = []
    for i in range(P):
        for j in range(i, P):
            E = np.zeros((P, P)); E[i, j] = E[j, i] = 1; basis.append(E)
    def L(S):
        V = sum((Hi - H) @ S @ (Hi - H) for Hi in Hs) / len(Hs); C = H @ (H @ S - S @ H) - (H @ S - S @ H) @ H
        return H @ S + S @ H - eta / (1 + beta) * H @ S @ H + eta * beta / ((1 - beta) * (1 - beta**2)) * C - eta / (1 - beta) * V
    A = np.array([[np.sum(L(Bj) * Bi) for Bj in basis] for Bi in basis]); rhs = np.array([np.sum(eta / (1 - beta) * Sxi * Bi) for Bi in basis])
    return sum(c * E for c, E in zip(np.linalg.solve(A, rhs), basis))
def stable_full(H, Hs, eta, beta):
    P = len(H); n = 2 * P; M = np.zeros((n * n, n * n))
    for Hi in Hs:
        F = np.zeros((n, n)); F[:P, :P] = (1 + beta) * np.eye(P) - eta * Hi; F[:P, P:] = -beta * np.eye(P); F[P:, :P] = np.eye(P)
        M += np.kron(F, F) / len(Hs)
    return rad(M)
H = np.diag([1.0, 0.3]); N = np.array([[-0.6, -0.7], [-0.7, -0.2]]); Hs = [H + N, H - N]
print(f"   eig(H+N) = {np.round(np.linalg.eigvalsh(H + N), 4)}  eig(H-N) = {np.round(np.linalg.eigvalsh(H - N), 4)}")
v = np.array([-0.8, 0.6]); Sxi = np.outer(v, v) + 1e-3 * np.eye(2); eeff = 0.52
for beta in (0.5, 0.95):
    S0 = sigma0(H, Hs, Sxi, eeff * (1 - beta), beta)
    print(f"   beta={beta}: rho(T)={stable_full(H, Hs, eeff * (1 - beta), beta):.4f}  loss tr(H S0)={np.trace(H @ S0):.5f}  tr(Sxi S0)={np.trace(Sxi @ S0):.5f}")

# ---------------------------------------------------------------- F. Corollary 6(iv): EMA form, momentum and batch
print("F. Corollary 6(iv): level sets of S in EMA form, beta -> 1, batch ratio")
def S_ema(x, lam, theta, p, gam, kaps):
    c = p * (gam * (2 + kaps) - 1) * lam**2; d = (theta * lam + c / lam) / 2
    return x * p * gam / 2 * np.sum(lam / (1 - x * d))
def ema_root(s, lam, beta, p, gam, kaps):
    theta = (1 - beta) / (1 + beta); c = p * (gam * (2 + kaps) - 1) * lam**2; d = (theta * lam + c / lam) / 2
    hi = 1 / max(d.max(), 1e-300) if d.max() > 0 else 1e12
    return bisect(lambda x: S_ema(x, lam, theta, p, gam, kaps) < s, 0.0, hi * (1 - 1e-15), 200)
def M_inf(lam, gam, kaps):
    return np.diag((gam * (2 + kaps) - 1) * lam) + gam * np.outer(np.sqrt(lam), np.sqrt(lam))
rng = np.random.default_rng(2026); viol = [0, 0, 0, 0]; n = 0
for _ in range(200):
    P = int(rng.integers(1, 7)); lam = np.sort(rng.uniform(0.02, 1, P))[::-1]; kaps = rng.uniform(-2, 8, P)
    gam = rng.uniform(1, 3); B = int(rng.integers(1, 64)); p = 1 / B; s = rng.uniform(0.05, 1.0); n += 1
    y0 = 1 / ema_root(s, lam, 0.0, p, gam, kaps); ys = []
    for beta in (0.3, 0.6, 0.9, 0.99):
        y = 1 / ema_root(s, lam, beta, p, gam, kaps); ys.append(y)
        lo, hi = beta / (1 + beta) * lam.min(), beta / (1 + beta) * lam.max()
        if not (lo * (1 - 1e-9) - 1e-12 <= y0 - y <= hi * (1 + 1e-9) + 1e-12): viol[0] += 1
    if np.any(np.diff([y0] + ys) >= 0): viol[1] += 1
    # threshold: 1/eta*_EMA = lambda_max(diag(d) + p gam/2 sqrt(lam) sqrt(lam)^T), and the eta_det ratio
    beta = rng.uniform(0, 0.99); kb = (1 + beta) / (1 - beta); eta_s = 1 / lam[0] if False else None
    Mt = np.diag(lam) + p * kb * M_inf(lam, gam, kaps); mu = np.linalg.eigvalsh(Mt).max()
    if abs(2 * kb / mu - ema_root(1.0, lam, beta, p, gam, kaps)) > 1e-7 * 2 * kb / mu: viol[2] += 1
    if P >= 2:
        l = np.sort(lam)[::-1]
        if np.linalg.eigvalsh(M_inf(lam, gam, kaps)).max() < np.sqrt(l[0] * l[1]) * (1 - 1e-12): viol[3] += 1
print(f"   {n} random models: shift-bound violations {viol[0]}, non-monotone {viol[1]}, threshold mismatch {viol[2]}, lambda_max(M_inf) < sqrt(l1 l2): {viol[3]}")
lam = np.array([1.0, 0.5, 0.2]); kaps = np.array([0.0, 4.0, -1.5]); gam = 2.0; B = 4; p = 1 / B
for beta in (0.0, 0.9, 0.99, 0.9999):
    print(f"   beta={beta}: eta*_EMA = {ema_root(1.0, lam, beta, p, gam, kaps):.6f}")
print(f"   limit 2B/lambda_max(M_inf) = {2 * B / np.linalg.eigvalsh(M_inf(lam, gam, kaps)).max():.6f}")
G = 2 * lam.max() - lam; Mi = M_inf(lam, gam, kaps); Bh0 = np.linalg.eigvalsh(Mi / np.sqrt(np.outer(G, G))).max()
for beta in (0.0, 0.5, 0.9):
    kb = (1 + beta) / (1 - beta); Bh = kb * Bh0
    mu = np.linalg.eigvalsh(np.diag(lam) + kb / Bh * Mi).max()
    print(f"   beta={beta}: B_1/2 = {Bh:.6f}, eta*/eta_det there = {lam.max() / mu:.12f}")
lm = np.linalg.eigvalsh(Mi).max(); print(f"   B_1/2(0) = {Bh0:.6f} in [{lm / (2 * lam.max()):.6f}, {lm / lam.max():.6f}]")

# ---------------------------------------------------------------- G. Proposition 9: rate at a fixed loss floor
print("G. Proposition 9: label noise Sigma_xi = sigma^2 H / B, fixed floor = fixed S")
def floor_exact(lam, eta, beta, p, gam, kaps, sig2):
    # stationary point of T_diag with additive noise sigma_k^2 = sig2 * p * lam_k
    P = len(lam); M = T_diag(lam, eta, beta, p, gam, kaps); g = np.zeros(3 * P); g[3 * np.arange(P)] = eta**2 * sig2 * p * lam
    x = np.linalg.solve(np.eye(3 * P) - M, g); return np.sum(lam * x[3 * np.arange(P)])
lam = np.array([1.0, 0.5, 0.2]); kaps = np.array([0.0, 4.0, -1.0]); gam = 2.0; p = 0.25; beta = 0.7; sig2 = 0.8
eta = 0.6 * bisect(lambda e: np.all(e * lam * np.maximum(1 + p * (1 + beta) / (1 - beta) * (gam * (2 + kaps) - 1), 0) < 2 * (1 + beta)) and secular_S(e, lam, beta, p, gam, kaps) < 1, 1e-9, 10, 200)
S = secular_S(eta, lam, beta, p, gam, kaps)
print(f"   floor: exact {floor_exact(lam, eta, beta, p, gam, kaps, sig2):.10f}  (sigma^2/gamma) S/(1-S) = {sig2 / gam * S / (1 - S):.10f}")
def K_series(lam, eta, beta, p, gam, kaps, T=4000):
    K = np.zeros(T)
    for l, k in zip(lam, kaps):
        a = 1 + beta - eta * l; g = np.zeros(T); g[0] = 1; g[1] = a
        for t in range(1, T - 1): g[t + 1] = a * g[t] - beta * g[t - 1]
        psi = eta**2 * g * g; c = p * (gam * (2 + k) - 1) * l * l; h = np.zeros(T); h[0] = psi[0]
        for t in range(1, T): h[t] = psi[t] + c * np.dot(psi[t - 1::-1][:t], h[:t])
        K += p * gam * l * l * h
    return K
K = K_series(lam, eta, beta, p, gam, kaps); tb = np.sum((np.arange(len(K)) + 1) * K) / K.sum()
rho = rad(T_diag(lam, eta, beta, p, gam, kaps))
print(f"   sum K = {K.sum():.8f} = S = {S:.8f};  rho = {rho:.6f} >= S^(1/taubar) = {S**(1 / tb):.6f} >= S")
def eta_hb_for(s, lam, beta, p, gam, kaps): return (1 - beta) * ema_root(s, lam, beta, p, gam, kaps)
betas = np.unique(np.concatenate([np.linspace(0, 0.99, 199), 1 - np.logspace(-2, -4.5, 120)]))
cases = [("P=1, kappa=0, B=4", np.array([1.0]), np.array([0.0]), 1.0, 0.25, 0.3),
         ("P=1, kappa=0, B=4", np.array([1.0]), np.array([0.0]), 1.0, 0.25, 0.01),
         ("P=1, kappa=0, B=4", np.array([1.0]), np.array([0.0]), 1.0, 0.25, 0.001),
         ("P=5 Gaussian, B=8", np.array([1, 0.5, 0.2, 0.05, 0.01]), np.zeros(5), 1.0, 0.125, 0.3),
         ("P=5 Gaussian, B=8", np.array([1, 0.5, 0.2, 0.05, 0.01]), np.zeros(5), 1.0, 0.125, 0.01),
         ("P=5 mixed kappa_k, gamma=2, B=8", np.array([1, 0.5, 0.2, 0.05, 0.01]), np.array([5.0, 0, 2, 8, 1]), 2.0, 0.125, 0.3)]
for name, lam, kaps, gam, p, s in cases:
    rate = lambda b: 1 - rad(T_diag(lam, eta_hb_for(s, lam, b, p, gam, kaps), b, p, gam, kaps))
    r = np.array([rate(b) for b in betas]); i = int(np.argmax(r)); bb, rb = betas[i], r[i]
    if i > 0:  # golden-section refinement between the grid neighbours
        lo, hi = betas[i - 1], betas[min(i + 1, len(betas) - 1)]; g = (np.sqrt(5) - 1) / 2
        for _ in range(80):
            m1, m2 = hi - g * (hi - lo), lo + g * (hi - lo)
            if rate(m1) >= rate(m2): hi = m2
            else: lo = m1
        bb = (lo + hi) / 2; rb = rate(bb)
    print(f"   {name}, S={s}: 1-rho at beta=0 {r[0]:.6f}; best beta {bb:.6f}, 1-rho {rb:.6f}, speedup {rb / r[0]:.3f}")
lam = np.array([1, 0.5, 0.2, 0.05, 0.01]); kaps = np.zeros(5); gam = 1.0; p = 0.125; s = 0.3
for b in (0.999, 0.9999):
    e = eta_hb_for(s, lam, b, p, gam, kaps); r = 1 - rad(T_diag(lam, e, b, p, gam, kaps))
    print(f"   beta={b}: (1-rho)/(1-beta) = {r / (1 - b):.5f}, 1 - eta_EMA(beta;S)/eta*_EMA(beta) = {1 - ema_root(s, lam, b, p, gam, kaps) / ema_root(1.0, lam, b, p, gam, kaps):.5f}")

# ---------------------------------------------------------------- H. Proposition 7, weighted lower bound
print("H. Proposition 7: weighted lower bound on mu*")
P = 256; lam = np.arange(1, P + 1) ** -0.5; beta = 0.9; B = 4; p = 1 / B; kb = (1 + beta) / (1 - beta)
for kap in (-2.0, -1.0, 0.0, 2.0, 10.0):
    al = 1 + p * kb * (1 + kap); mu = np.linalg.eigvalsh(np.diag(al * lam) + p * kb * np.outer(np.sqrt(lam), np.sqrt(lam))).max()
    lo = al * np.sum(lam**2) / lam.sum() + p * kb * lam.sum(); hi = max(al, 0) * lam[0] + p * kb * lam.sum()
    print(f"   kappa={kap:5.1f}: {lo:.3f} <= mu* = {mu:.3f} <= {hi:.3f};  ratio bound {lo / (lam[0] * (1 + p * kb * (2 + kap))):.2f}")
