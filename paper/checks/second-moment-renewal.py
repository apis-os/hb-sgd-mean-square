"""Checks for the without-replacement noise operator, the renewal equation of the expected loss and the moment ratio.

(a) Sampling B of N fixed matrices without replacement: E[(H_B - H) X (H_B - H)] = p (1/N) sum_i dH_i X dH_i with
    p = (N - B) / (B (N - 1)), by exact enumeration of the subsets.
(b) Pool average: for a pool of n i.i.d. matrices centred at its own mean, E[(H_i - Hhat) X (H_i - Hhat)] = (n - 1)/n V_1(X),
    by exact enumeration of a three-point law (rational arithmetic); and for the pool of all 2^P sign vectors (Rademacher),
    Hhat = H and V = p_wor V_1 exactly.
(c) Random features with kappa_feat in {-2, -1, 0, 2}: the exact diagonal second-moment recursion of heavy-ball SGD gives a
    trace tr(H Sigma_t) equal to the solution of the scalar renewal equation with kernel K_tau = p sum_k lambda_k^2 h_k(tau),
    h_k the dressed per-mode propagator, and sum_tau K_tau equals the left side of the secular equation with alpha_diag.
(e) The resampling estimator: E[Qhat_t] = T^t(Q_0) by exact enumeration (rational arithmetic) for a two-matrix law, S = 2,
    t = 2.
(f) Random reshuffling, N = 6, B = 2 (m = 3), over the 720 permutations: E[Delta_r X Delta_s] = (m delta_rs - 1)/(m - 1) V_wor(X);
    for m = 2, beta = 0: F_2 F_1 = (I - eta H)^2 + eta^2 [H, Delta_1] - eta^2 Delta_1^2.
(h) Phase 1 cell P = 64, beta = 0, B = 64, Gaussian: the exact ratio tr T^{2m}(u1 u1^T) / tr T^m(u1 u1^T) crosses 1 at 1.926417
    for m = 8 and approaches the secular root 1.926841 as m grows.
(i) eta_eff*(beta) = eta*(beta)/(1-beta) from the operator of Theorem G6 on a non-commuting model (P = 3, N = 5, B = 2):
    strictly increasing in beta, and 1/eta*(0) - (1-beta)/eta*(beta) >= beta/(1+beta) lambda_min(H); the Phase 1 Gaussian cells
    at P = 64, B = 64: eta_eff* = 1.927 at beta = 0 and 7.948 at beta = 0.9.
(j) Rademacher cells of Phase 1: mu* > (2 - alpha_diag) lambda_1 in 9 of 10 (fails at P = 64, beta = 0, B = 64).
(g) Along u_1 at beta = 0, a scalar pool of 8 values with B = 2: the mean-square threshold of one reshuffled epoch against
    2(1 + nu^2) and of independent steps without replacement against 2(1 - nu^2), by exact enumeration.
(d) Along u_1 at beta = 0 with zeta = chi^2_64/64 - 1 and lambda_1 = 1: m4/m2^2 of the multiplier 1 - eta(1 + zeta) at eta = 1.94.

Pure Python; exit code 1 on any failure.
"""
import itertools
import math
import random
import sys

fails = []
random.seed(1)


def sym(P):
    m = [[random.gauss(0, 1) for _ in range(P)] for _ in range(P)]
    return [[(m[i][j] + m[j][i]) / 2 for j in range(P)] for i in range(P)]


def mm(a, b):
    n = len(a)
    return [[sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def lin(*terms):
    n = len(terms[0][1])
    return [[sum(c * m[i][j] for c, m in terms) for j in range(n)] for i in range(n)]


# (a)
P, N, B = 3, 6, 2
Hs = [sym(P) for _ in range(N)]
H = lin(*[(1 / N, h) for h in Hs])
X = sym(P)
subsets = list(itertools.combinations(range(N), B))
V = lin((0.0, X))
for s in subsets:
    d = lin(*[(1 / B, Hs[i]) for i in s], (-1.0, H))
    V = lin((1.0, V), (1 / len(subsets), mm(mm(d, X), d)))
p = (N - B) / (B * (N - 1))
R = lin((0.0, X))
for h in Hs:
    d = lin((1.0, h), (-1.0, H))
    R = lin((1.0, R), (p / N, mm(mm(d, X), d)))
err_a = max(abs(V[i][j] - R[i][j]) for i in range(P) for j in range(P))
print(f"(a) without replacement, N={N} B={B}: max |V - p (1/N) sum dH X dH| = {err_a:.1e}")
if err_a > 1e-12:
    fails.append("a")

# (b)
from fractions import Fraction as Fr
vals = [Fr(0), Fr(1), Fr(4)]
mean_v = sum(vals) / 3
v1 = sum((v - mean_v) ** 2 for v in vals) / 3
npool = 3
acc = Fr(0)
for pool in itertools.product(vals, repeat=npool):
    hh = sum(pool) / npool
    acc += (pool[0] - hh) ** 2 / len(vals) ** npool
print(f"(b) three-point law, n={npool}: E_pool[(H_1 - Hhat)^2] / V_1 = {acc / v1} against (n-1)/n = {Fr(npool - 1, npool)}")
if acc / v1 != Fr(npool - 1, npool):
    fails.append("b")
cube = []
for sgn in itertools.product([1, -1], repeat=2):
    z = [sgn[0] * 2, sgn[1]]
    cube.append([[Fr(z[i] * z[j]) for j in range(2)] for i in range(2)])
ncube, bcube = 4, 2
hcube = [[sum(m[i][j] for m in cube) / ncube for j in range(2)] for i in range(2)]
xq = [[Fr(1), Fr(1, 3)], [Fr(1, 3), Fr(2)]]


def mmq(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]


def subq(a, b):
    return [[a[i][j] - b[i][j] for j in range(2)] for i in range(2)]


subsq = list(itertools.combinations(range(ncube), bcube))
vw = [[Fr(0)] * 2 for _ in range(2)]
for sub in subsq:
    hb = [[sum(cube[i][r][c] for i in sub) / bcube for c in range(2)] for r in range(2)]
    d = subq(hb, hcube)
    mq = mmq(mmq(d, xq), d)
    vw = [[vw[i][j] + mq[i][j] / len(subsq) for j in range(2)] for i in range(2)]
v1q = [[Fr(0)] * 2 for _ in range(2)]
for m_ in cube:
    d = subq(m_, hcube)
    mq = mmq(mmq(d, xq), d)
    v1q = [[v1q[i][j] + mq[i][j] / ncube for j in range(2)] for i in range(2)]
pw = Fr(ncube - bcube, bcube * (ncube - 1))
cube_ok = hcube == [[Fr(4), Fr(0)], [Fr(0), Fr(1)]] and vw == [[pw * v1q[i][j] for j in range(2)] for i in range(2)]
print(f"(b) hypercube pool (P=2, n=4, B=2): Hhat = H and V = p_wor V_1 exactly: {cube_ok}")
if not cube_ok:
    fails.append("b-cube")

# (c)
P, beta, p, s2, T = 7, 0.7, 0.11, 1e-4, 500
lam = [1 / math.sqrt(k + 1) for k in range(P)]


def impulse(l, eta, steps):
    out, xm, x = [], 0.0, 0.0
    for t in range(steps):
        xn = (1 + beta - eta * l) * x - beta * xm + (1.0 if t == 0 else 0.0)
        out.append(xn)
        xm, x = x, xn
    return out


for kf in (-2.0, -1.0, 0.0, 2.0):
    ad = 1 + p * (1 + beta) / (1 - beta) * (1 + kf)

    def secular(eta):
        return eta * p / (2 * (1 - beta)) * sum(l / (1 - eta * l * ad / (2 * (1 + beta))) for l in lam)

    lo, hi = 1e-9, 2 * (1 + beta) / (lam[0] * max(ad, 1.0)) * (1 - 1e-12)
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if secular(mid) < 1 else (lo, mid)
    eta = 0.8 * lo
    psi = [[eta ** 2 * v * v for v in impulse(l, eta, T)] for l in lam]
    hk = []
    for k in range(P):
        c = p * (1 + kf) * lam[k] ** 2
        h = []
        for tau in range(T):
            h.append(psi[k][tau] + c * sum(psi[k][tau - 1 - u] * h[u] for u in range(tau)))
        hk.append(h)
    C, Cm, D, trace = [0.0] * P, [0.0] * P, [0.0] * P, []
    for t in range(T):
        tr = sum(lam[k] * C[k] for k in range(P))
        trace.append(tr)
        Cn, Dn = [], []
        for k in range(P):
            a = 1 + beta - eta * lam[k]
            v = p * (lam[k] * tr + (1 + kf) * lam[k] ** 2 * C[k])
            Cn.append(a * a * C[k] + beta * beta * Cm[k] - 2 * a * beta * D[k] + eta ** 2 * v + eta ** 2 * s2)
            Dn.append(a * C[k] - beta * D[k])
        Cm, C, D = C, Cn, Dn
    K = [p * sum(lam[k] ** 2 * hk[k][t] for k in range(P)) for t in range(T)]
    f = [sum(lam[k] * s2 * sum(hk[k][u] for u in range(t)) for k in range(P)) for t in range(T)]
    ren = []
    for t in range(T):
        ren.append(f[t] + sum(K[t - 1 - s_] * ren[s_] for s_ in range(t)))
    err_ren = max(abs(x - y) / abs(x) for x, y in zip(trace[1:], ren[1:]))
    big_psi = [eta / (2 * (1 - beta) * l * (1 - eta * l / (2 * (1 + beta)))) for l in lam]
    ksum = sum(p * l * l * q / (1 - p * (1 + kf) * l * l * q) for l, q in zip(lam, big_psi))
    err_sum = abs(ksum - secular(eta))
    print(f"(c) kappa_feat={kf:+.0f}: renewal vs exact recursion over {T} steps, max relative difference {err_ren:.1e}; |sum K - secular| = {err_sum:.1e}; min h = {min(min(h) for h in hk):.2e}")
    if err_ren > 1e-12 or err_sum > 1e-12:
        fails.append(f"c{kf}")

# (d)
k = 64


def dens(y):
    x = k * y
    return k * math.exp((k / 2 - 1) * math.log(x) - x / 2 - (k / 2) * math.log(2) - math.lgamma(k / 2))


ys = [i * 0.0005 for i in range(1, 8000)]
w = [dens(y) * 0.0005 for y in ys]
total = sum(w)
w = [x / total for x in w]
m2 = sum(wi * (1 - 1.94 * y) ** 2 for wi, y in zip(w, ys))
m4 = sum(wi * (1 - 1.94 * y) ** 4 for wi, y in zip(w, ys))
r = m4 / m2 ** 2
print(f"(d) eta=1.94: m4/m2^2 = {r:.4f}, log of its 1000th power = {1000 * math.log(r):.1f}")
if abs(r - 1.498) > 0.001:
    fails.append("d")

# (e)
beta_q = Fr(1, 2)
fam = [[[Fr(3, 2) - Fr(1, 3), -beta_q], [Fr(1), Fr(0)]], [[Fr(3, 2) - Fr(2), -beta_q], [Fr(1), Fr(0)]]]


def apply_t(qm):
    out = [[Fr(0)] * 2 for _ in range(2)]
    for fm in fam:
        fq = mmq(fm, qm)
        fqf = [[sum(fq[i][k] * fm[j][k] for k in range(2)) for j in range(2)] for i in range(2)]
        out = [[out[i][j] + fqf[i][j] / 2 for j in range(2)] for i in range(2)]
    return out


def n2(v):
    return v[0] * v[0] + v[1] * v[1]


ss = 2
start = [[Fr(1), Fr(0)], [Fr(3, 5), Fr(4, 5)]]
q0 = [[sum(u[i] * u[j] for u in start) / ss for j in range(2)] for i in range(2)]
states = [(Fr(1), Fr(1), start)]
for _ in range(2):
    nxt = []
    for prob, mhat, parts in states:
        for choice in itertools.product(range(2), repeat=ss):
            ys = [[fam[c][0][0] * u[0] + fam[c][0][1] * u[1], fam[c][1][0] * u[0] + fam[c][1][1] * u[1]] for c, u in zip(choice, parts)]
            r = [n2(y) / n2(u) for y, u in zip(ys, parts)]
            mu = sum(r) / ss
            for idx in itertools.product(range(ss), repeat=ss):
                pr = Fr(1)
                for i in idx:
                    pr *= r[i] / (ss * mu)
                nxt.append((prob * Fr(1, 2 ** ss) * pr, mhat * mu, [ys[i] for i in idx]))
    states = nxt
est = [[Fr(0)] * 2 for _ in range(2)]
for prob, mhat, parts in states:
    for v in parts:
        est = [[est[i][j] + prob * mhat * v[i] * v[j] / n2(v) / ss for j in range(2)] for i in range(2)]
unbiased = est == apply_t(apply_t(q0))
print(f"(e) resampling estimator: E[Qhat_2] = T^2(Q_0) exactly: {unbiased}")
if not unbiased:
    fails.append("e")

# (f)
rnd = random.Random(5)


def rs2():
    a, b, c = [Fr(rnd.randint(-5, 5), rnd.randint(1, 4)) for _ in range(3)]
    return [[a, b], [b, c]]


def addq(a, b, sgn=1):
    return [[a[i][j] + sgn * b[i][j] for j in range(2)] for i in range(2)]


def scq(c, a):
    return [[c * a[i][j] for j in range(2)] for i in range(2)]


zq = [[Fr(0)] * 2 for _ in range(2)]
nrr, brr = 6, 2
mrr = nrr // brr
hrr = [rs2() for _ in range(nrr)]
hbar = zq
for h_ in hrr:
    hbar = addq(hbar, scq(Fr(1, nrr), h_))
xr = rs2()
vwor = zq
for h_ in hrr:
    d = addq(h_, hbar, -1)
    vwor = addq(vwor, scq(Fr(nrr - brr, brr * (nrr - 1)) * Fr(1, nrr), mmq(mmq(d, xr), d)))
perms = list(itertools.permutations(range(nrr)))
cov = {(r, q): zq for r in range(mrr) for q in range(mrr)}
for pi in perms:
    deltas = []
    for r in range(mrr):
        hb = zq
        for i in pi[r * brr:(r + 1) * brr]:
            hb = addq(hb, scq(Fr(1, brr), hrr[i]))
        deltas.append(addq(hb, hbar, -1))
    for r in range(mrr):
        for q in range(mrr):
            cov[(r, q)] = addq(cov[(r, q)], scq(Fr(1, len(perms)), mmq(mmq(deltas[r], xr), deltas[q])))
cov_ok = all(cov[(r, q)] == scq(Fr(mrr * (1 if r == q else 0) - 1, mrr - 1), vwor) for r in range(mrr) for q in range(mrr))
eye = [[Fr(1), Fr(0)], [Fr(0), Fr(1)]]
eta_q = Fr(1, 3)
h4 = hrr[:4]
hb4 = zq
for h_ in h4:
    hb4 = addq(hb4, scq(Fr(1, 4), h_))
b1 = scq(Fr(1, 2), addq(h4[0], h4[1]))
b2 = scq(Fr(1, 2), addq(h4[2], h4[3]))
d1 = addq(b1, hb4, -1)
fep = mmq(addq(eye, scq(eta_q, b2), -1), addq(eye, scq(eta_q, b1), -1))
aq = addq(eye, scq(eta_q, hb4), -1)
closed = addq(addq(mmq(aq, aq), scq(eta_q * eta_q, addq(mmq(hb4, d1), mmq(d1, hb4), -1))), scq(eta_q * eta_q, mmq(d1, d1)), -1)
print(f"(f) reshuffling covariance identity over 720 permutations: {cov_ok}; m=2 epoch product closed form: {fep == closed}")
if not (cov_ok and fep == closed):
    fails.append("f")

# (g)
lam1 = 1.0
pool8 = [lam1 + 0.06 * x for x in (-3, -1, -1, 0, 0, 1, 1, 3)]
avg8 = sum(pool8) / 8
pool8 = [v - avg8 + lam1 for v in pool8]
b8, m8 = 2, 4
nu2 = sum((v - lam1) ** 2 for v in pool8) / 8 * (8 - b8) / (b8 * 7)


def epoch_m2(eta):
    tot, cnt = 0.0, 0
    for pi in itertools.permutations(range(8)):
        prod = 1.0
        for r in range(m8):
            prod *= 1 - eta * sum(pool8[i] for i in pi[r * b8:(r + 1) * b8]) / b8
        tot += prod * prod
        cnt += 1
    return tot / cnt


def step_m2(eta):
    pairs = list(itertools.permutations(pool8, b8))
    return sum((1 - eta * sum(c) / b8) ** 2 for c in pairs) / len(pairs)


def bisect(fn, lo, hi, it=30):
    for _ in range(it):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fn(mid) < 1 else (lo, mid)
    return lo


thr_rr = bisect(epoch_m2, 1.9, 2.1)
thr_iid = bisect(step_m2, 1.9, 2.1)
print(f"(g) nu^2 = {nu2:.5f}: reshuffled epoch threshold {thr_rr:.5f} against 2(1+nu^2) = {2 * (1 + nu2):.5f}; independent steps {thr_iid:.5f} against 2(1-nu^2) = {2 * (1 - nu2):.5f}")
if abs(thr_rr - 2 * (1 + nu2)) > 1e-3 or abs(thr_iid - 2 * (1 - nu2)) > 1e-3:
    fails.append("g")

# (h)
lam64 = [1 / math.sqrt(k + 1) for k in range(64)]
p64 = 1 / 64


def sec64(eta):
    return eta * p64 / 2 * sum(l / (1 - eta * l * (1 + p64) / 2) for l in lam64)


def tr_ratio(eta, mm):
    c = [0.0] * 64
    c[0] = 1.0
    traces = []
    for _ in range(2 * mm + 1):
        traces.append(sum(c))
        tr = sum(l * x for l, x in zip(lam64, c))
        c = [(1 - eta * l) ** 2 * x + eta * eta * p64 * (l * tr + l * l * x) for l, x in zip(lam64, c)]
    return traces[2 * mm] / traces[mm]


root64 = bisect(lambda e: sec64(e), 1.0, 2 / (1 + p64) * 0.999999, 100)
cross8 = bisect(lambda e: tr_ratio(e, 8), 1.85, 2.0, 60)
cross16 = bisect(lambda e: tr_ratio(e, 16), 1.85, 2.0, 60)
print(f"(h) secular root {root64:.6f}; trace-ratio crossing m=8: {cross8:.6f}, m=16: {cross16:.6f}")
if abs(cross8 - 1.926417) > 1e-6 or abs(root64 - 1.926841) > 1e-6 or abs(cross16 - root64) > 1e-5:
    fails.append("h")

# (i)
def jacobi_max(a):
    n_ = len(a)
    a = [row[:] for row in a]
    for _ in range(100):
        off = sum(a[i][j] ** 2 for i in range(n_) for j in range(n_) if i != j)
        if off < 1e-26:
            break
        for p_ in range(n_):
            for q in range(p_ + 1, n_):
                if abs(a[p_][q]) < 1e-300:
                    continue
                th = (a[q][q] - a[p_][p_]) / (2 * a[p_][q])
                t_ = (1 if th >= 0 else -1) / (abs(th) + math.sqrt(th * th + 1))
                c_ = 1 / math.sqrt(t_ * t_ + 1)
                s_ = t_ * c_
                for k in range(n_):
                    akp, akq = a[k][p_], a[k][q]
                    a[k][p_], a[k][q] = c_ * akp - s_ * akq, s_ * akp + c_ * akq
                for k in range(n_):
                    apk, aqk = a[p_][k], a[q][k]
                    a[p_][k], a[q][k] = c_ * apk - s_ * aqk, s_ * apk + c_ * aqk
    return max(a[i][i] for i in range(n_))


rg = random.Random(11)
pg, ng, bg = 3, 5, 2
lamg = [2.0, 1.0, 0.4]


def rsg():
    mt = [[rg.gauss(0, 0.6) for _ in range(pg)] for _ in range(pg)]
    return [[(mt[i][j] + mt[j][i]) / 2 for j in range(pg)] for i in range(pg)]


dhs = [rsg() for _ in range(ng)]
mg = [[sum(d[i][j] for d in dhs) / ng for j in range(pg)] for i in range(pg)]
dhs = [[[d[i][j] - mg[i][j] for j in range(pg)] for i in range(pg)] for d in dhs]
bas = []
for j in range(pg):
    for k in range(j, pg):
        e_ = [[0.0] * pg for _ in range(pg)]
        if j == k:
            e_[j][j] = 1.0
        else:
            e_[j][k] = e_[k][j] = 1 / math.sqrt(2)
        bas.append((j, k, e_))


def mm3(a, b):
    return [[sum(a[i][t] * b[t][j] for t in range(pg)) for j in range(pg)] for i in range(pg)]


def vop(x):
    out = [[0.0] * pg for _ in range(pg)]
    for d in dhs:
        mt = mm3(mm3(d, x), d)
        out = [[out[i][j] + mt[i][j] / (ng * bg) for j in range(pg)] for i in range(pg)]
    return out


vmat = [[sum(bas[a][2][i][j] * vop(bas[b][2])[i][j] for i in range(pg) for j in range(pg)) for b in range(len(bas))] for a in range(len(bas))]


def eta_g6(beta):
    nb = len(bas)
    g = [[0.0] * nb for _ in range(nb)]
    for a, (j, k, _) in enumerate(bas):
        for b, (j2, k2, _) in enumerate(bas):
            v = vmat[a][b] / (1 - beta)
            if a == b:
                v += lamg[j] * lamg[k] / (1 + beta) - beta / ((1 - beta) * (1 - beta ** 2)) * (lamg[j] - lamg[k]) ** 2
            g[a][b] = v / math.sqrt((lamg[j] + lamg[k]) * (lamg[j2] + lamg[k2]))
    return 1 / jacobi_max(g)


betas = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
etag = {b: eta_g6(b) for b in betas}
effg = [etag[b] / (1 - b) for b in betas]
mono = all(effg[i] < effg[i + 1] for i in range(len(effg) - 1))
bound = all(1 / etag[0.0] - (1 - b) / etag[b] >= b / (1 + b) * min(lamg) for b in betas[1:])


def root_sec(lams, beta, bsz, kf):
    pp = 1 / bsz
    ad = 1 + pp * (1 + beta) / (1 - beta) * (1 + kf)
    fn = lambda e: e / (2 * (1 - beta) * bsz) * sum(l / (1 - e * l * ad / (2 * (1 + beta))) for l in lams)
    return bisect(fn, 1e-9, 2 * (1 + beta) / (lams[0] * max(ad, 1)) * (1 - 1e-12), 200), ad


r0, _ = root_sec(lam64, 0.0, 64, 0.0)
r9, _ = root_sec(lam64, 0.9, 64, 0.0)
print(f"(i) eta_eff* on the non-commuting model: {[round(x, 4) for x in effg]}, strictly increasing: {mono}, bound holds: {bound}; Phase 1 P=64 B=64 Gaussian: {r0:.3f} and {r9 / 0.1:.3f}, factor {r9 / 0.1 / r0:.2f}")
if not (mono and bound and abs(r9 / 0.1 - 7.948) < 1e-3 and abs(r0 - 1.927) < 1e-3):
    fails.append("i")

# (j)
held = []
for pp_, beta_, bsz_ in [(64, 0.0, 4), (64, 0.0, 16), (64, 0.0, 64), (64, 0.9, 4), (64, 0.9, 16), (64, 0.9, 64), (256, 0.0, 4), (256, 0.0, 16), (256, 0.9, 4), (256, 0.9, 16)]:
    lams = [1 / math.sqrt(k + 1) for k in range(pp_)]
    rt, ad = root_sec(lams, beta_, bsz_, -2.0)
    held.append(2 * (1 + beta_) / rt > (2 - ad) * lams[0])
print(f"(j) Rademacher cells with mu* > (2 - alpha_diag) lambda_1: {sum(held)} of {len(held)}; the failing cell is index {[i for i, h in enumerate(held) if not h]}")
if sum(held) != 9 or held[2]:
    fails.append("j")

if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("all checks passed")
