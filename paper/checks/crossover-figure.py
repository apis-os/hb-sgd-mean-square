# Crossover of tuned momentum at B - 1 = B_x for the two-row vocabulary d = 8, q = (10/11, 1/11).
# Closed forms of the note (Theorem 3, Lemma 9(ii), Theorem 6(a)) against an independent search that uses only
# the eigenvalues of the 3x3 matrix M(q) of Everett and Paquette (2026, Section H.3).
# Usage: python3 paper/checks/crossover-figure.py   (numpy; matplotlib only for the figure)
import math, numpy as np
d, q1, q2 = 8, 10 / 11, 1 / 11
D1, D2 = d + 2 - q1, d + 2 - q2
Bx = (d + 2) / (q1 - q2)

def M(q, eta, beta, B):
    a = 1 + beta - eta * q
    return np.array([[a * a + eta * eta * q * (d + 2 - q) / B, -2 * a * beta, beta * beta], [a, -beta, 0], [1, 0, 0]])

def rho_voc(eta, beta, B):
    return max(max(abs(np.linalg.eigvals(M(q, eta, beta, B)))) for q in (q1, q2))

def best_eta(beta, B):
    # golden-section search on log(eta) after a coarse scan; independent of the note's formulas
    le = np.linspace(math.log(1e-3), math.log(4 / q1), 400)
    v = [rho_voc(math.exp(x), beta, B) for x in le]; i = int(np.argmin(v))
    lo, hi = le[max(i - 1, 0)], le[min(i + 1, len(le) - 1)]
    g = (math.sqrt(5) - 1) / 2
    for _ in range(80):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if rho_voc(math.exp(a), beta, B) < rho_voc(math.exp(b), beta, B): hi = b
        else: lo = a
    return rho_voc(math.exp((lo + hi) / 2), beta, B)

def search(B):
    betas = np.linspace(0, 0.9, 91)
    v = [best_eta(b, B) for b in betas]; i = int(np.argmin(v))
    lo, hi = betas[max(i - 1, 0)], betas[min(i + 1, len(betas) - 1)]
    g = (math.sqrt(5) - 1) / 2
    for _ in range(50):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if best_eta(a, B) < best_eta(b, B): hi = b
        else: lo = a
    bb = (lo + hi) / 2; r = best_eta(bb, B)
    if v[0] <= r: return v[0], 0.0
    return r, bb

def sgd_formula(B):
    # Lemma 9(ii): rho_SGD = W / (W + B)
    W = D2 / q2 if B <= Bx + 1 else D2 / q2 + (q1 - q2) ** 2 / (4 * q1 * q2) * (B - 1 - Bx) ** 2 / (B - 1)
    return W / (W + B)

print(f"d = {d}, q = (10/11, 1/11), B_x = {Bx:.4f}: tuned beta is 0 for B <= {math.floor(Bx + 1)}, positive from B = {math.floor(Bx + 1) + 1}")
rows = []
for B in range(1, 41):
    r_s = best_eta(0.0, B); r_m, b_m = search(B)
    S_s, S_m, S_f = -1 / math.log(r_s), -1 / math.log(r_m), -1 / math.log(sgd_formula(B))
    rows.append((B, S_f, S_s, S_m, b_m))
    print(f"B={B:3d}  S_SGD formula {S_f:9.4f}  search {S_s:9.4f}  S_mom search {S_m:9.4f}  tuned beta {b_m:.3f}")
mx = max(abs(r[1] - r[2]) / r[1] for r in rows)
print(f"max rel |S_SGD formula - search| = {mx:.1e}")
print("B with tuned beta > 1e-3:", [r[0] for r in rows if r[4] > 1e-3][:3], "...")
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Bs = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=150)
    ax.plot(Bs, [r[1] for r in rows], "-", color="0.4", label=r"$S_{\mathrm{SGD}}(B)$, Lemma 9(ii)")
    ax.plot(Bs, [r[2] for r in rows], "o", ms=3, mfc="none", color="0.4", label="SGD, search on $M(q)$")
    ax.plot(Bs, [r[3] for r in rows], "s", ms=3, color="C0", label=r"$S_{\mathrm{mom}}(B)$, search on $M(q)$")
    ax.axvline(Bx + 1, ls="--", lw=0.8, color="C3"); ax.text(Bx + 1.4, ax.get_ylim()[1] * 0.92, r"$B-1=B_\times$", color="C3", fontsize=8)
    ax.set_xlabel("batch size $B$"); ax.set_ylabel("steps per decay time"); ax.set_yscale("log"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig("paper/figures/crossover.png"); print("wrote paper/figures/crossover.png")
except ImportError:
    print("matplotlib not installed; figure skipped")
