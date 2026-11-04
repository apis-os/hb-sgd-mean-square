#!/usr/bin/env python3
"""Numerical check of Results 1 and 2 (diag draft). Pure Python 3 stdlib.

Result 1: block bounds rho(K_D) <= rho(K) <= block bound; (1-beta)^2 scaling at fixed f;
          threshold eta* vs diagonal threshold eta_D; repeated eigenvalue; Rademacher reduction.
Result 2: (a) Loewner burn-in Q_inf - Q_t = T^t(Q_inf - Q_0) and the Perron / Q_theta envelopes;
          (b) rho(R) = rho(K), identity T(W) = W - (1-r) blkdiag(eta^2 V(X*), 0), the one-step
              constant s = 1, and the two-step bound (1-r) m2 / 2 <= 1 - rho(T) <= (1-r)(1-s_low).
T is built in Kraus form by enumerating every batch (sampling with replacement).
"""
import itertools
import math
import random
import sys
import time

# ---------------------------------------------------------------- dense linear algebra

def zeros(n, m):
    return [[0.0] * m for _ in range(n)]

def eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def mm(A, B):
    Bt = list(zip(*B))
    return [[sum(a * b for a, b in zip(r, c)) for c in Bt] for r in A]

def tr_(A):
    return [list(r) for r in zip(*A)]

def add(A, B, a=1.0, b=1.0):
    return [[a * x + b * y for x, y in zip(r, s)] for r, s in zip(A, B)]

def scal(A, a):
    return [[a * x for x in r] for r in A]

def fro(A):
    return math.sqrt(sum(x * x for r in A for x in r))

def mv(A, v):
    return [sum(a * b for a, b in zip(r, v)) for r in A]

def jacobi(Ain, maxsweep=200):
    """Symmetric eigen-decomposition, cyclic Jacobi. Returns (eigs, V) with A V = V diag(eigs)."""
    n = len(Ain)
    A = [list(map(float, r)) for r in Ain]
    for i in range(n):
        for j in range(i + 1, n):
            m = 0.5 * (A[i][j] + A[j][i]); A[i][j] = m; A[j][i] = m
    V = eye(n)
    scale = max(1e-300, fro(A))
    for _ in range(maxsweep):
        off = math.sqrt(sum(A[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
        if off <= 1e-15 * scale:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p][q]
                if abs(apq) <= 1e-300:
                    continue
                theta = (A[q][q] - A[p][p]) / (2.0 * apq)
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0); s = t * c
                for k in range(n):
                    akp = A[k][p]; akq = A[k][q]
                    A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq
                for k in range(n):
                    apk = A[p][k]; aqk = A[q][k]
                    A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp = V[k][p]; vkq = V[k][q]
                    V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq
    return [A[i][i] for i in range(n)], V

def lmax(S):
    e, _ = jacobi(S); return max(e)

def lmin(S):
    e, _ = jacobi(S); return min(e)

def opnorm(M):
    """Spectral norm of a rectangular matrix."""
    if len(M) == 0 or len(M[0]) == 0:
        return 0.0
    G = mm(tr_(M), M) if len(M[0]) <= len(M) else mm(M, tr_(M))
    return math.sqrt(max(0.0, lmax(G)))

def solve(A, Bm):
    """Solve A X = Bm (Bm: n x m) by Gaussian elimination with partial pivoting."""
    n = len(A); m = len(Bm[0])
    M = [list(A[i]) + list(Bm[i]) for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        inv = 1.0 / piv
        rowc = M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] * inv
                if f != 0.0:
                    rr = M[r]
                    for k in range(c, n + m):
                        rr[k] -= f * rowc[k]
    return [[M[i][n + k] / M[i][i] for k in range(m)] for i in range(n)]

def specrad(M, K=46):
    """rho(M) = lim ||M^n||^(1/n) by repeated squaring (n = 2^K). Returns (rho, last normalized power)."""
    nrm = fro(M)
    if nrm == 0.0:
        return 0.0, M
    X = scal(M, 1.0 / nrm); L = math.log(nrm)
    for _ in range(K):
        X = mm(X, X); s = fro(X)
        if s == 0.0:
            return 0.0, X
        X = scal(X, 1.0 / s); L = 2.0 * L + math.log(s)
    return math.exp(L / 2.0 ** K), X

def chol(A):
    n = len(A); L = zeros(n, n)
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0:
                    raise ValueError("not PD")
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L

def congr_inv(Wm, Z):
    """L^{-1} Z L^{-T} with W = L L^T (symmetric, same spectrum as W^{-1/2} Z W^{-1/2})."""
    L = chol(Wm)
    Y = solve(L, Z)            # L^{-1} Z
    Y2 = solve(L, tr_(Y))      # L^{-1} (L^{-1} Z)^T = L^{-1} Z^T L^{-T}
    return Y2

# ---------------------------------------------------------------- symmetric basis of S^n
R2 = math.sqrt(2.0)

def sym_index(n):
    return [(i, j) for i in range(n) for j in range(i, n)]

def vec(X, idx):
    return [X[i][j] if i == j else R2 * X[i][j] for (i, j) in idx]

def unvec(v, idx, n):
    X = zeros(n, n)
    for a, (i, j) in enumerate(idx):
        if i == j:
            X[i][i] = v[a]
        else:
            X[i][j] = v[a] / R2; X[j][i] = v[a] / R2
    return X

def op_matrix(op, n):
    idx = sym_index(n); d = len(idx)
    cols = []
    for b in range(d):
        e = [0.0] * d; e[b] = 1.0
        cols.append(vec(op(unvec(e, idx, n)), idx))
    return [[cols[b][a] for b in range(d)] for a in range(d)]

def blk(A, B, C, D):
    return [ra + rb for ra, rb in zip(A, B)] + [rc + rd for rc, rd in zip(C, D)]

def sub(M, r0, r1, c0, c1):
    return [row[c0:c1] for row in M[r0:r1]]

# ---------------------------------------------------------------- cases

def gram_case(P, N, r, seed, flat=None):
    rng = random.Random(seed)
    Hs = []
    for _ in range(N):
        G = [[rng.gauss(0, 1) for _ in range(r)] for _ in range(P)]
        if flat is not None:
            G[P - 1] = [flat * x for x in G[P - 1]]
        Hs.append(mm(G, tr_(G)))
    H = scal([[sum(Hs[i][a][b] for i in range(N)) for b in range(P)] for a in range(P)], 1.0 / N)
    e, U = jacobi(H)
    order = sorted(range(P), key=lambda k: -e[k])
    U = [[U[a][k] for k in order] for a in range(P)]
    lam = [e[k] for k in order]
    Hs_e = [mm(tr_(U), mm(Hi, U)) for Hi in Hs]          # eigenbasis
    dH = [add(Hi, [[lam[a] if a == b else 0.0 for b in range(P)] for a in range(P)], 1.0, -1.0) for Hi in Hs_e]
    return lam, Hs_e, dH

def centered_case(lam, N, seed, frac=0.9):
    rng = random.Random(seed); P = len(lam)
    D = []
    for _ in range(N):
        A = [[rng.gauss(0, 1) for _ in range(P)] for _ in range(P)]
        D.append([[0.5 * (A[a][b] + A[b][a]) for b in range(P)] for a in range(P)])
    mean = scal([[sum(D[i][a][b] for i in range(N)) for b in range(P)] for a in range(P)], 1.0 / N)
    D = [add(Di, mean, 1.0, -1.0) for Di in D]
    mx = max(max(abs(x) for x in jacobi(Di)[0]) for Di in D)
    s = frac * min(lam) / mx
    D = [scal(Di, s) for Di in D]
    H = [[lam[a] if a == b else 0.0 for b in range(P)] for a in range(P)]
    Hs = [add(H, Di) for Di in D]
    return list(lam), Hs, D

def rademacher_case(lam):
    P = len(lam)
    Hs = []
    for om in itertools.product([-1.0, 1.0], repeat=P):
        z = [math.sqrt(lam[k]) * om[k] for k in range(P)]
        Hs.append([[z[a] * z[b] for b in range(P)] for a in range(P)])
    H = [[lam[a] if a == b else 0.0 for b in range(P)] for a in range(P)]
    dH = [add(Hi, H, 1.0, -1.0) for Hi in Hs]
    return list(lam), Hs, dH

# ---------------------------------------------------------------- operators

class Model:
    def __init__(self, name, lam, Hs, dH, B, tol=1e-9):
        self.name = name; self.lam = lam; self.Hs = Hs; self.dH = dH; self.B = B
        self.P = P = len(lam); self.N = len(Hs)
        self.idxP = sym_index(P); self.idx2 = sym_index(2 * P)
        # V on S^P (formula, sampling with replacement)
        def Vop(X):
            acc = zeros(P, P)
            for Di in dH:
                acc = add(acc, mm(Di, mm(X, Di)))
            return scal(acc, 1.0 / (B * self.N))
        self.Vop = Vop
        self.V = op_matrix(Vop, P)
        # partition: D = entries inside an eigenspace, O = the rest
        self.Didx = [a for a, (j, k) in enumerate(self.idxP) if abs(lam[j] - lam[k]) <= tol]
        self.Oidx = [a for a, (j, k) in enumerate(self.idxP) if abs(lam[j] - lam[k]) > tol]
        gaps = [abs(lam[j] - lam[k]) for (j, k) in self.idxP if abs(lam[j] - lam[k]) > tol]
        self.dmin = min(gaps) if gaps else float('inf')
        # Kraus pieces on S^{2P}: T(eta) = Ta - eta Tb + eta^2 Tc   (Tc by batch enumeration)
        self.batches = list(itertools.product(range(self.N), repeat=B))
        HB = []
        for bt in self.batches:
            S = zeros(P, P)
            for i in bt:
                S = add(S, Hs[i])
            HB.append(scal(S, 1.0 / B))
        self.HB = HB
        Hm = [[lam[a] if a == b else 0.0 for b in range(P)] for a in range(P)]
        self.Hm = Hm
        n2 = 2 * P
        def Tc_op(Q):
            Q11 = sub(Q, 0, P, 0, P); acc = zeros(P, P)
            for Hb in HB:
                acc = add(acc, mm(Hb, mm(Q11, Hb)))
            acc = scal(acc, 1.0 / len(HB))
            return blk(acc, zeros(P, P), zeros(P, P), zeros(P, P))
        def Tc0_op(Q):
            Q11 = sub(Q, 0, P, 0, P)
            return blk(mm(Hm, mm(Q11, Hm)), zeros(P, P), zeros(P, P), zeros(P, P))
        self.Tc = op_matrix(Tc_op, n2)
        self.Tc0 = op_matrix(Tc0_op, n2)
        self.Ebar = blk(Hm, zeros(P, P), zeros(P, P), zeros(P, P))

    def pieces(self, beta):
        P = self.P; n2 = 2 * P
        I = eye(P)
        F0 = blk(scal(I, 1 + beta), scal(I, -beta), I, zeros(P, P))
        Ta = op_matrix(lambda Q: mm(F0, mm(Q, tr_(F0))), n2)
        Tb = op_matrix(lambda Q: add(mm(self.Ebar, mm(Q, tr_(F0))), mm(F0, mm(Q, tr_(self.Ebar)))), n2)
        return Ta, Tb

    def T(self, beta, eta, pcs=None):
        Ta, Tb = pcs if pcs else self.pieces(beta)
        return [[a - eta * b + eta * eta * c for a, b, c in zip(ra, rb, rc)] for ra, rb, rc in zip(Ta, Tb, self.Tc)]

    def T0(self, beta, eta, pcs=None):
        Ta, Tb = pcs if pcs else self.pieces(beta)
        return [[a - eta * b + eta * eta * c for a, b, c in zip(ra, rb, rc)] for ra, rb, rc in zip(Ta, Tb, self.Tc0)]

    def ell(self, beta, eta):
        lam = self.lam
        return [(lam[j] + lam[k]) - eta * lam[j] * lam[k] / (1 + beta)
                + eta * beta * (lam[j] - lam[k]) ** 2 / ((1 - beta) ** 2 * (1 + beta)) for (j, k) in self.idxP]

    def Ktil(self, beta, eta):
        l = self.ell(beta, eta); d = len(l)
        sq = [math.sqrt(eta / x) for x in l]
        return [[sq[a] * self.V[a][b] * sq[b] / (1 - beta) for b in range(d)] for a in range(d)], l

    def Gmat(self, beta):
        """G_beta = L0^{-1/2} B_beta L0^{-1/2} on S^P; eta* = 1/lmax(G)."""
        lam = self.lam; d = len(self.idxP)
        s = [lam[j] + lam[k] for (j, k) in self.idxP]
        G = [[self.V[a][b] / ((1 - beta) * math.sqrt(s[a] * s[b])) for b in range(d)] for a in range(d)]
        for a, (j, k) in enumerate(self.idxP):
            G[a][a] += (lam[j] * lam[k] / (1 + beta) - beta * (lam[j] - lam[k]) ** 2 / ((1 - beta) ** 2 * (1 + beta))) / s[a]
        return G

def pick(M, rows, cols):
    return [[M[r][c] for c in cols] for r in rows]

# ---------------------------------------------------------------- Result 1 quantities

def result1(m, beta, eta):
    Kt, l = m.Ktil(beta, eta)
    e, U = jacobi(Kt)
    rhoK = max(e); negK = min(e)
    D, O = m.Didx, m.Oidx
    A = pick(Kt, D, D); Bm = pick(Kt, D, O); C = pick(Kt, O, O)
    a = lmax(A)
    b = opnorm(Bm) if O else 0.0
    if O:
        eC = jacobi(C)[0]; cmax = max(eC); cn = max(abs(x) for x in eC)
    else:
        cmax = cn = 0.0
    gen = 0.5 * (a + cmax) + math.sqrt(0.25 * (a - cmax) ** 2 + b * b)
    req = a + b * b / (a - cn) if a > cn else float('nan')
    # analytic: omega_hat = max_O 1/ell, eps, r_D, ||Dhat^{1/2} V_DO||, ||V_OO||
    om = max(1.0 / l[o] for o in O) if O else 0.0
    eps = (1 - beta) ** 2 * (1 + beta) / (eta * beta * m.dmin ** 2) if O else 0.0
    rD = a * (1 - beta) / eta
    dh = [math.sqrt(1.0 / l[x]) for x in D]
    VDO = pick(m.V, D, O); VOO = pick(m.V, O, O)
    bD = opnorm([[dh[i] * VDO[i][j] for j in range(len(O))] for i in range(len(D))]) if O else 0.0
    vOO = max(abs(x) for x in jacobi(VOO)[0]) if O else 0.0
    relbound = om * bD ** 2 / (rD * (rD - om * vOO)) if (O and rD > om * vOO) else float('nan')
    # fully explicit version (Theorem D.2(3)): rho_V = lmax(V_DD), f = eta/eta_det
    lam = m.lam; f = eta * lam[0] / (2 * (1 + beta))
    rhoV = lmax(pick(m.V, D, D)); nVDO = opnorm(VDO) if O else 0.0
    trI = sum(m.V[a][b] for a, (i, j) in enumerate(m.idxP) if i == j for b, (k, l) in enumerate(m.idxP) if k == l)  # <I,V(I)>
    assert rhoV >= trI / m.P * (1 - 1e-12), "rho_V lower bound"
    den = rhoV - 2 * lam[0] * eps * vOO
    relexp = 2 * lam[0] ** 2 * eps * nVDO ** 2 / ((1 - f) * lam[-1] * rhoV * den) if (O and den > 0) else float('nan')
    # commutator-energy coordinates (Theorem D.2(3')): N_O,o <= q / Delta_o^2, q = (1-b)^2 (1+b)/b
    relDel = float('nan')
    if O:
        Dl = [abs(lam[m.idxP[o][0]] - lam[m.idxP[o][1]]) for o in O]
        bDl = opnorm([[dh[i] * VDO[i][j] / Dl[j] for j in range(len(O))] for i in range(len(D))])
        vDl = max(abs(x) for x in jacobi([[VOO[i][j] / (Dl[i] * Dl[j]) for j in range(len(O))] for i in range(len(O))])[0])
        q = (1 - beta) ** 2 * (1 + beta) / beta
        den2 = rD - q * vDl / eta
        if den2 > 0:
            relDel = (1 + beta) * (1 - beta) ** 2 * bDl ** 2 / (beta * eta * rD * den2)
    return dict(rhoK=rhoK, negK=negK, rhoD=a, b=b, cmax=cmax, cn=cn, gen=gen, req=req,
                om=om, eps=eps, relbound=relbound, relexp=relexp, relDel=relDel, rel=(rhoK - a) / a)

def thresholds(m, beta):
    G = m.Gmat(beta)
    D, O = m.Didx, m.Oidx
    eG, UG = jacobi(G)
    gmax = max(eG)
    GD = pick(G, D, D); eD, UD = jacobi(GD); gD = max(eD)
    eta_star = 1.0 / gmax; eta_D = 1.0 / gD
    out = dict(eta_star=eta_star, eta_D=eta_D)
    if O:
        GDO = pick(G, D, O); GOO = pick(G, O, O)
        gO = lmax(GOO)
        nb = opnorm(GDO)
        out['cor_bound'] = nb ** 2 / (gD - gO) if gD > gO else float('nan')   # bound on 1/eta* - 1/eta_D
        out['gap_inv'] = gmax - gD
        lam = m.lam
        nVDO = opnorm(pick(m.V, D, O)); nVOO = max(abs(x) for x in jacobi(pick(m.V, O, O))[0])
        den = beta * m.dmin ** 2 - (1 - beta * beta) * lam[0] * nVOO / lam[-1]
        out['c_beta'] = lam[0] * (1 + beta) * nVDO ** 2 / (2 * lam[-1] ** 2 * den) if den > 0 else float('nan')
        # lmax(Vtilde) >= E||dH||_F^2 / (B tr H)
        s0 = [lam[j] + lam[k] for (j, k) in m.idxP]
        Vt = [[2 * m.V[a][b] / math.sqrt(s0[a] * s0[b]) for b in D] for a in D]
        lV = lmax(Vt)
        EdH = sum(fro(Di) ** 2 for Di in m.dH) / m.N
        out['lVt_ok'] = lV >= EdH / (m.B * sum(lam)) * (1 - 1e-12)
        out['lVt'] = lV
        # first-order (second-order perturbation) estimate of lmax(G)
        y = [UD[i][eD.index(gD)] for i in range(len(D))]
        z = [sum(GDO[i][j] * y[i] for i in range(len(D))) for j in range(len(O))]   # G_OD y
        Mx = [[(gD if i == j else 0.0) - GOO[i][j] for j in range(len(O))] for i in range(len(O))]
        sol = solve(Mx, [[v] for v in z])
        out['pert'] = sum(z[j] * sol[j][0] for j in range(len(O)))
        # beta -> 1 limit of the correction: c_inf = 2 sum_O (V_OD L0D^{-1/2} y_inf)_o^2 / Delta_o^2,
        # y_inf = top eigenvector of L0D^{-1/2} V_DD L0D^{-1/2}
        lam = m.lam
        s0 = [lam[j] + lam[k] for (j, k) in m.idxP]
        VDDs = [[m.V[a][b] / math.sqrt(s0[a] * s0[b]) for b in D] for a in D]
        ev, Uv = jacobi(VDDs); top = ev.index(max(ev))
        yi = [Uv[i][top] / math.sqrt(s0[D[i]]) for i in range(len(D))]
        zz = [sum(m.V[o][D[i]] * yi[i] for i in range(len(D))) for o in O]
        out['c_inf'] = 2.0 * sum((zz[q] / (lam[m.idxP[o][0]] - lam[m.idxP[o][1]])) ** 2 for q, o in enumerate(O))
        # commutator-energy coordinates: M1 = D_Delta^{-1} V_OD Lambda_D^{-1/2}, v_Delta = lmax(D_Delta^{-1} V_OO D_Delta^{-1})
        Dl = [abs(lam[m.idxP[o][0]] - lam[m.idxP[o][1]]) for o in O]
        lamD = [lam[m.idxP[d][0]] for d in D]
        M1 = [[m.V[o][d] / (Dl[q] * math.sqrt(lamD[i])) for i, d in enumerate(D)] for q, o in enumerate(O)]
        nM1 = opnorm(M1)
        vD = lmax([[m.V[o][p2] / (Dl[q] * Dl[r]) for r, p2 in enumerate(O)] for q, o in enumerate(O)])
        th = (1 - beta * beta) * vD / beta
        out['c_new'] = (1 + beta) / (2 * beta) * nM1 ** 2 / (1 - th) if th < 1 else float('nan')
        out['c_lim'] = nM1 ** 2
        xinf = [Uv[i][top] for i in range(len(D))]
        out['c_inf_alt'] = sum(sum(M1[q][i] * xinf[i] for i in range(len(D))) ** 2 for q in range(len(O)))
        # weighted (operator) version with cbar_o = c_o + g_D - l_j l_k/((1+b) s_o)
        sO = [s0[o] for o in O]
        cbar = [beta * Dl[q] ** 2 / ((1 - beta) ** 2 * (1 + beta) * sO[q]) + gD
                - lam[m.idxP[o][0]] * lam[m.idxP[o][1]] / ((1 + beta) * sO[q]) for q, o in enumerate(O)]
        Eb = [[m.V[o][p2] / ((1 - beta) * math.sqrt(sO[q] * sO[r] * cbar[q] * cbar[r])) for r, p2 in enumerate(O)] for q, o in enumerate(O)]
        thb = lmax(Eb)
        Tbar = Eb  # T = Cbar^{-1/2} E_OO Cbar^{-1/2}, kept for the first-order majorant below
        GODc = [[GDO[i][q] / math.sqrt(cbar[q]) for i in range(len(D))] for q in range(len(O))]
        out['c_bar'] = opnorm(GODc) ** 2 / (1 - thb) if thb < 1 else float('nan')
        # eigenvector split (Theorem diag-threshold(2'')): E >= S(mu) >= 0, u = top eigenvector of G_DD,
        # 1/eta* - 1/eta_D <= u'Eu + ||P_perp E u||^2 / (gamma_D + u'Eu - ||P_perp E P_perp||)
        nD = len(D)
        top_i = eD.index(gD)
        u = [UD[i][top_i] for i in range(nD)]
        gam = gD - (sorted(eD)[-2] if nD >= 2 else -float('inf'))
        out['gamma_D'] = gam
        def eig_split(Em):
            Eu = [sum(Em[i][k] * u[k] for k in range(nD)) for i in range(nD)]
            uEu = sum(u[i] * Eu[i] for i in range(nD))
            pEu = [Eu[i] - uEu * u[i] for i in range(nD)]
            npEu2 = sum(x * x for x in pEu)
            Pp = [[(1.0 if i == k else 0.0) - u[i] * u[k] for k in range(nD)] for i in range(nD)]
            PEP = mm(Pp, mm(Em, Pp))
            nPEP = max(abs(x) for x in jacobi(PEP)[0]) if nD >= 2 else 0.0
            den = gam + uEu - nPEP
            return (uEu + npEu2 / den if den > 0 else float('nan')), uEu, den
        if th < 1:
            al = (1 + beta) / (2 * beta * (1 - th))
            Em = [[al * sum(M1[q][i] * M1[q][k] for q in range(len(O))) for k in range(nD)] for i in range(nD)]
            out['c_eig'], out['uEu'], out['den_eig'] = eig_split(Em)
        else:
            out['c_eig'], out['uEu'], out['den_eig'] = float('nan'), float('nan'), float('nan')
        if thb < 1:
            Eb = [[sum(GDO[i][q] * GDO[k][q] / cbar[q] for q in range(len(O))) / (1 - thb) for k in range(nD)] for i in range(nD)]
            out['c_eigbar'], out['uEbu'], _ = eig_split(Eb)
        else:
            out['c_eigbar'] = out['uEbu'] = float('nan')
        # S8 quantities. (a) First-order majorant: 1/(1-t) <= 1 + t_+/(1-thb) for t <= thb < 1, so
        # (Cbar - E_OO)^{-1} <= Cbar^{-1} + Cbar^{-1/2} T_+ Cbar^{-1/2}/(1-thb) and
        # E1 = G_DO Cbar^{-1} G_OD + G_DO Cbar^{-1/2} T_+ Cbar^{-1/2} G_OD/(1-thb) majorizes S(mu*); T_+ = T only if T >= 0.
        nO = len(O)
        eT, UT = jacobi(Tbar)
        out['T_min_eig'] = min(eT)
        if thb < 1:
            Tp = [[sum(UT[i][l] * max(eT[l], 0.0) * UT[k][l] for l in range(nO)) for k in range(nO)] for i in range(nO)]
            Tr = [[sum(UT[i][l] * eT[l] * UT[k][l] for l in range(nO)) for k in range(nO)] for i in range(nO)]
            Gc = [[GDO[i][q] / math.sqrt(cbar[q]) for q in range(nO)] for i in range(nD)]  # G_DO Cbar^{-1/2}
            def maj(TT):
                GT = mm(Gc, TT)
                return [[sum(Gc[i][q] * Gc[k][q] for q in range(nO)) + sum(GT[i][q] * Gc[k][q] for q in range(nO)) / (1 - thb)
                         for k in range(nD)] for i in range(nD)]
            E1 = maj(Tp); E1raw = maj(Tr)
            out['c_eig1'] = eig_split(E1)[0]
            out['c_bar1'] = lmax(E1)
            out['c_eig1raw'] = eig_split(E1raw)[0]  # with T in place of T_+: not a valid majorant when T has negative eigenvalues
        else:
            out['c_eig1'] = out['c_bar1'] = out['c_eig1raw'] = float('nan')
        # (b) Resolvent at g_D: mu* >= g_D > lmax(G_OO) gives (mu* - G_OO)^{-1} <= (g_D - G_OO)^{-1}, so E_op = G_DO (g_D - G_OO)^{-1} G_OD >= S(mu*).
        if gD > gO:
            sol2 = solve([[(gD if i == j else 0.0) - GOO[i][j] for j in range(nO)] for i in range(nO)], [[GDO[k][q] for k in range(nD)] for q in range(nO)])
            Eop = [[sum(GDO[i][q] * sol2[q][k] for q in range(nO)) for k in range(nD)] for i in range(nD)]
            Eop = [[0.5 * (Eop[i][k] + Eop[k][i]) for k in range(nD)] for i in range(nD)]
            out['c_eigop'] = eig_split(Eop)[0]
        else:
            out['c_eigop'] = float('nan')
    # explicit diagonal formula (simple spectrum): lmax(Lambda + kappa * Gamma~)
    if len(D) == m.P:
        lam = m.lam; kap = (1 + beta) / (1 - beta)
        diagpos = [m.idxP.index((k, k)) for k in range(m.P)]
        Gam = [[m.V[diagpos[k]][diagpos[l]] for l in range(m.P)] for k in range(m.P)]
        Mt = [[(lam[k] if k == l else 0.0) + kap * Gam[k][l] / math.sqrt(lam[k] * lam[l]) for l in range(m.P)] for k in range(m.P)]
        out['eta_D_formula'] = 2 * (1 + beta) / lmax(Mt)
        Gt = [[Gam[k][l] / math.sqrt(lam[k] * lam[l]) for l in range(m.P)] for k in range(m.P)]
        out['lead'] = 2 * (1 - beta) / lmax(Gt)
    return out

# ---------------------------------------------------------------- main

def fmt(x, p=6):
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.{p}g}"
    return str(x)

ALLREL = []
ALLTH = []

def ratio_range(rows, num_idx, den_idx):
    rs = [r[num_idx] / r[den_idx] for r in rows if not math.isnan(r[num_idx]) and not math.isnan(r[den_idx]) and r[den_idx] > 0]
    nn = sum(1 for r in rows if not math.isnan(r[num_idx]))
    return (min(rs), max(rs), nn, len(rows)) if rs else (float('nan'), float('nan'), nn, len(rows))

def main():
    t0 = time.time()
    fails = []
    out = print

    # jacobi self-test
    rng = random.Random(1)
    A = [[rng.gauss(0, 1) for _ in range(7)] for _ in range(7)]
    S = add(A, tr_(A))
    e, U = jacobi(S)
    res = fro(add(mm(S, U), mm(U, [[e[i] if i == j else 0 for j in range(7)] for i in range(7)]), 1, -1))
    out(f"[selftest] jacobi residual {res:.2e}")

    models = []
    models.append(Model("gram P=3 N=6 B=2 r=1", *gram_case(3, 6, 1, 11), B=2))
    models.append(Model("gram P=4 N=6 B=1 r=2", *gram_case(4, 6, 2, 12), B=1))
    models.append(Model("gram P=5 N=6 B=2 r=2", *gram_case(5, 6, 2, 13), B=2))
    models.append(Model("centered P=4 lam=(3,1.5,1.5,0.7) N=5 B=2 [repeated]", *centered_case([3.0, 1.5, 1.5, 0.7], 5, 14), B=2))
    models.append(Model("centered P=5 lam=(2,1,1,1,0.5) N=4 B=1 [triple]", *centered_case([2.0, 1.0, 1.0, 1.0, 0.5], 4, 15), B=1))
    models.append(Model("gram P=3 N=6 B=1 r=2 flat(0.1)", *gram_case(3, 6, 2, 16, flat=0.1), B=1))
    rad = Model("rademacher P=3 lam=(2,1,0.5) B=2", *rademacher_case([2.0, 1.0, 0.5]), B=2)
    models.append(rad)
    out(f"[setup] {len(models)} models built in {time.time()-t0:.1f}s")

    # sanity: Delta = Tc - Tc0 equals blkdiag(V(Q11),0) and per-sample Hessians do not commute
    for m in models:
        P = m.P
        Vb = op_matrix(lambda Q: blk(m.Vop(sub(Q, 0, P, 0, P)), zeros(P, P), zeros(P, P), zeros(P, P)), 2 * P)
        err = fro(add(add(m.Tc, m.Tc0, 1, -1), Vb, 1, -1))
        comm = max(fro(add(mm(m.Hs[i], m.Hs[j]), mm(m.Hs[j], m.Hs[i]), 1, -1)) for i in range(m.N) for j in range(m.N))
        psd = min(lmin(Hi) for Hi in m.Hs)
        out(f"[sanity] {m.name}: lam={[round(x,4) for x in m.lam]} |Tc-Tc0-V|={err:.1e} max||[Hi,Hj]||={comm:.3f} min eig Hi={psd:.2e} Delta_min={fmt(m.dmin,4)} |D|={len(m.Didx)} |O|={len(m.Oidx)}")
        if err > 1e-10:
            fails.append(f"Kraus/V mismatch {m.name}")

    # ---------------- crossing: rho(T)=1 <=> rho(K)=1
    out("\n=== Crossing check: eta* = 1/lmax(G_beta); rho(K(eta*)); rho(T) at eta*(1-d), eta*, eta*(1+d), d=1e-3")
    for m in models:
        for beta in (0.5, 0.9):
            th = thresholds(m, beta); es = th['eta_star']
            pcs = m.pieces(beta)
            rk = result1(m, beta, es)['rhoK']
            rts = [specrad(m.T(beta, es * f, pcs))[0] for f in (1 - 1e-3, 1.0, 1 + 1e-3)]
            ok = rts[0] < 1 < rts[2] and abs(rts[1] - 1) < 1e-6 and abs(rk - 1) < 1e-9
            if not ok:
                fails.append(f"crossing {m.name} beta={beta}")
            out(f"  {m.name:52s} beta={beta}: eta*={es:.6g} rhoK={rk:.12f} rhoT=[{rts[0]:.8f}, {rts[1]:.10f}, {rts[2]:.8f}] |rhoT(eta*)-1|={abs(rts[1]-1):.1e} {'OK' if ok else 'FAIL'}")

    # ---------------- Rademacher reduction to Theorem G
    out("\n=== Rademacher features: V_DO = 0, eta_D = secular root of Theorem G, eta* = eta_D")
    for beta in (0.5, 0.9):
        lam = rad.lam; B = rad.B; kap = (1 + beta) / (1 - beta); kf = -2.0
        alpha = 1 + kap * (1 + kf) / B
        def sec(eta):
            return eta / (2 * (1 - beta) * B) * sum(l / (1 - eta * l * alpha / (2 * (1 + beta))) for l in lam) - 1
        lo, hi = 1e-12, 2 * (1 + beta) / (lam[0] * max(alpha, 1)) * (1 - 1e-14)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if sec(mid) < 0: lo = mid
            else: hi = mid
        th = thresholds(rad, beta)
        VDO = opnorm(pick(rad.V, rad.Didx, rad.Oidx))
        ok = abs(th['eta_D'] / lo - 1) < 1e-9 and abs(th['eta_star'] / lo - 1) < 1e-9 and VDO < 1e-12
        if not ok:
            fails.append(f"rademacher beta={beta}")
        out(f"  beta={beta}: secular={lo:.12g} eta_D={th['eta_D']:.12g} eta_D(formula)={th['eta_D_formula']:.12g} eta*={th['eta_star']:.12g} ||V_DO||={VDO:.1e} {'OK' if ok else 'FAIL'}")

    # ---------------- Result 1 bounds and scaling
    betas = (0.5, 0.8, 0.9, 0.95, 0.99)
    f = 0.5
    out(f"\n=== Result 1 at fixed f = eta/eta_det = {f}: rho_D <= rho(K) <= bounds; rel=(rho(K)-rho_D)/rho_D; eps=(1-b)^2(1+b)/(eta b Dmin^2)")
    for m in models:
        if m is rad:
            continue
        out(f"  -- {m.name}")
        rels = []
        for beta in betas:
            eta = f * 2 * (1 + beta) / m.lam[0]
            r = result1(m, beta, eta)
            lower_ok = r['rhoD'] <= r['rhoK'] * (1 + 1e-12)
            gen_ok = r['rhoK'] <= r['gen'] * (1 + 1e-12)
            req_ok = math.isnan(r['req']) or r['rhoK'] <= r['req'] * (1 + 1e-12)
            rb_ok = math.isnan(r['relbound']) or r['rel'] <= r['relbound'] * (1 + 1e-9) + 1e-14
            rb_ok = rb_ok and (math.isnan(r['relexp']) or r['rel'] <= r['relexp'] * (1 + 1e-9) + 1e-14)
            rb_ok = rb_ok and (math.isnan(r['relDel']) or r['rel'] <= r['relDel'] * (1 + 1e-9) + 1e-14)
            ALLREL.append((m.name, beta, r['rel'], r['relbound'], r['relexp'], r['relDel']))
            kr_ok = r['rhoK'] >= abs(r['negK'])
            ok = lower_ok and gen_ok and req_ok and rb_ok and kr_ok and r['om'] <= r['eps'] * (1 + 1e-12)
            if not ok:
                fails.append(f"R1 bounds {m.name} beta={beta}")
            rels.append(r['rel'])
            out(f"    beta={beta:<5} rhoK={r['rhoK']:.6g} rhoD={r['rhoD']:.6g} ||B||={r['b']:.3g} ||C||={r['cn']:.3g} "
                f"gen={r['gen']:.6g} req={fmt(r['req'])} rel={r['rel']:.3e} relbound={fmt(r['relbound'],3)} relexp={fmt(r['relexp'],3)} relDel={fmt(r['relDel'],3)} "
                f"eps={r['eps']:.3e} rel/(1-b)^2={r['rel']/(1-beta)**2:.4g} {'OK' if ok else 'FAIL'}")
        sl = math.log(rels[-2] / rels[-1]) / math.log((1 - betas[-2]) / (1 - betas[-1])) if rels[-1] > 0 else float('nan')
        out(f"    log-slope of rel in (1-beta) between beta=0.95 and 0.99: {fmt(sl,4)} (predicted 2 when V_DO != 0)")

    betas = (0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999)
    out("\n=== Threshold: eta* <= eta_D; (eta_D/eta* - 1)/(1-beta); corollary bound 1/eta* - 1/eta_D <= ||G_DO||^2/(g_D - lmax G_OO); leading 2(1-b)/lmax(Gamma~)")
    for m in models:
        if m is rad:
            continue
        out(f"  -- {m.name}")
        gaps = []
        for beta in betas:
            th = thresholds(m, beta)
            g = th['eta_D'] / th['eta_star'] - 1
            gaps.append(g)
            ok = th['eta_star'] <= th['eta_D'] * (1 + 1e-12) and (math.isnan(th['cor_bound']) or th['gap_inv'] <= th['cor_bound'] * (1 + 1e-9))
            ok = ok and th['lVt_ok'] and (math.isnan(th['c_beta']) or th['gap_inv'] <= th['c_beta'] * (1 + 1e-9))
            ok = ok and (math.isnan(th['c_new']) or th['gap_inv'] <= th['c_new'] * (1 + 1e-9))
            ok = ok and (math.isnan(th['c_bar']) or th['gap_inv'] <= th['c_bar'] * (1 + 1e-9))
            ok = ok and (math.isnan(th['c_eig']) or th['gap_inv'] <= th['c_eig'] * (1 + 1e-9))
            ok = ok and (math.isnan(th['c_eigbar']) or th['gap_inv'] <= th['c_eigbar'] * (1 + 1e-9))
            ok = ok and abs(th['c_inf_alt'] - th['c_inf']) <= 1e-10 * max(1.0, th['c_inf']) and th['c_inf'] <= th['c_lim'] * (1 + 1e-12)
            ALLTH.append((m.name, beta, th['gap_inv'], th['cor_bound'], th['c_beta'], th['c_new'], th['c_bar'], th['c_eig'], th['c_eigbar']))
            # leading order: 0 <= 1/eta* - lmax(Vt)/(2(1-b)) <= lam1/(2(1+b)) + (1/eta* - 1/eta_D)
            lead_dev = 1 / th['eta_star'] - th['lVt'] / (2 * (1 - beta))
            ok = ok and (-1e-9 <= lead_dev <= m.lam[0] / (2 * (1 + beta)) + th['gap_inv'] + 1e-9)
            if 'eta_D_formula' in th:
                ok = ok and abs(th['eta_D_formula'] / th['eta_D'] - 1) < 1e-10
            if not ok:
                fails.append(f"threshold {m.name} beta={beta}")
            extra = f" lead/eta*={th['lead']/th['eta_star']:.6f}" if 'lead' in th else ""
            out(f"    beta={beta:<5} eta*={th['eta_star']:.6g} eta_D={th['eta_D']:.6g} (eta_D/eta*-1)={g:.3e} /(1-b)={g/(1-beta):.4g} "
                f"1/eta*-1/eta_D={th['gap_inv']:.4g} cor_bound={fmt(th['cor_bound'],4)} c_beta={fmt(th['c_beta'],4)} c_new={fmt(th['c_new'],4)} c_bar={fmt(th['c_bar'],4)} c_eig={fmt(th['c_eig'],5)} c_eigbar={fmt(th['c_eigbar'],5)} uEu={fmt(th['uEu'],5)} gammaD={th['gamma_D']:.4g} den={fmt(th['den_eig'],4)} pert={th['pert']:.4g} c_inf={th['c_inf']:.4g} c_lim={th['c_lim']:.4g}{extra} {'OK' if ok else 'FAIL'}")
        sl = math.log(gaps[-2] / gaps[-1]) / math.log((1 - betas[-2]) / (1 - betas[-1]))
        out(f"    log-slope of (eta_D/eta*-1) between beta=0.999 and 0.9999: {sl:.4f} (predicted 1 when V_DO != 0)")

    out("\n=== Looseness of the scalar bounds (ratio to the operator form; defined cells / all cells)")
    out("  threshold, operator form = ||G_DO||^2/(g_D - lmax G_OO):")
    for nm, i in (("c_beta (Dmin, lambda_P)", 4), ("c_new (commutator energy)", 5), ("c_bar (weighted)", 6), ("c_eig (eigvec split, comm.)", 7), ("c_eigbar (eigvec split, wtd)", 8)):
        lo, hi, nn, tot = ratio_range(ALLTH, i, 3)
        out(f"    {nm:28s}: ratio {fmt(lo,3)} .. {fmt(hi,3)}  defined in {nn}/{tot}")
    for nm in sorted(set(r[0] for r in ALLTH)):
        rows = [r for r in ALLTH if r[0] == nm]
        a = ratio_range(rows, 4, 3); b = ratio_range(rows, 5, 3); c = ratio_range(rows, 6, 3); d = ratio_range(rows, 7, 3); e = ratio_range(rows, 7, 2)
        out(f"    {nm:52s} c_beta/op {fmt(a[0],3)}..{fmt(a[1],3)} ({a[2]}/{a[3]})  c_new/op {fmt(b[0],3)}..{fmt(b[1],3)} ({b[2]}/{b[3]})  c_bar/op {fmt(c[0],3)}..{fmt(c[1],3)} ({c[2]}/{c[3]})  c_eig/op {fmt(d[0],3)}..{fmt(d[1],3)} ({d[2]}/{d[3]})  c_eig/true {fmt(e[0],4)}..{fmt(e[1],4)}")
        out("      c_eig/true by beta: " + "  ".join(f"{r[1]}:{fmt(r[7]/r[2],4) if not math.isnan(r[7]) else 'undef'}" for r in rows))
    out("  threshold, against the actual gap 1/eta* - 1/eta_D:")
    for nm, i in (("operator form", 3), ("c_beta", 4), ("c_new", 5), ("c_bar", 6), ("c_eig", 7), ("c_eigbar", 8)):
        lo, hi, nn, tot = ratio_range(ALLTH, i, 2)
        out(f"    {nm:28s}: ratio {fmt(lo,3)} .. {fmt(hi,3)}  defined in {nn}/{tot}")
    out("  fixed f, operator form = relbound:")
    for nm, i in (("relexp (Dmin, lambda_P)", 4), ("relDel (commutator energy)", 5)):
        lo, hi, nn, tot = ratio_range(ALLREL, i, 3)
        out(f"    {nm:28s}: ratio {fmt(lo,3)} .. {fmt(hi,3)}  defined in {nn}/{tot}")
    out("  fixed f, against the actual relative error:")
    for nm, i in (("relbound", 3), ("relexp", 4), ("relDel", 5)):
        lo, hi, nn, tot = ratio_range(ALLREL, i, 2)
        out(f"    {nm:28s}: ratio {fmt(lo,3)} .. {fmt(hi,3)}  defined in {nn}/{tot}")

    # ---------------- S8: first-order Neumann majorant and resolvent at g_D in the eigenvector split
    out("\n=== S8. Eigenvector split with E1 (first-order Neumann, T_+) and E_op = G_DO (g_D - G_OO)^{-1} G_OD, against c_eigbar and the true gap")
    s8rows = []
    for m in models:
        if m is rad:
            continue
        for beta in betas:
            th = thresholds(m, beta)
            gp = th['gap_inv']
            for k in ('c_eigbar', 'c_eig1', 'c_bar1', 'c_eigop'):
                if not math.isnan(th[k]) and gp > th[k] * (1 + 1e-9) + 1e-14:
                    fails.append(f"S8 {k} {m.name} beta={beta}")
            raw_viol = (not math.isnan(th['c_eig1raw'])) and gp > th['c_eig1raw'] * (1 + 1e-9) + 1e-14
            s8rows.append((m.name, beta, gp, th['c_eigbar'], th['c_eig1'], th['c_bar1'], th['c_eigop'], th['T_min_eig']))
            rr = lambda k: fmt(th[k] / gp, 4) if (not math.isnan(th[k]) and gp > 0) else 'undef'
            out(f"  {m.name:52s} beta={beta:<6} gap={gp:.4g}  c_eigbar/gap={rr('c_eigbar')}  c_eig1/gap={rr('c_eig1')}  ||E1||/gap={rr('c_bar1')}  c_eigop/gap={rr('c_eigop')}  min eig T={th['T_min_eig']:.3g}{'  (T in place of T_+ violates)' if raw_viol else ''}")
    for nm, i in (("c_eigbar", 3), ("c_eig1", 4), ("||E1||", 5), ("c_eigop", 6)):
        lo, hi, nn, tot = ratio_range(s8rows, i, 2)
        out(f"  {nm:10s}/gap: {fmt(lo,4)} .. {fmt(hi,4)}  defined in {nn}/{tot}")

    out("\n=== rho(T) sign agreement at fixed f (rho(T)>1 iff rho(K)>1), beta in {0.5,0.9,0.99}")
    for m in models[:4]:
        for beta in (0.5, 0.9, 0.99):
            eta = f * 2 * (1 + beta) / m.lam[0]
            rk = result1(m, beta, eta)['rhoK']
            rt = specrad(m.T(beta, eta))[0]
            ok = (rt > 1) == (rk > 1)
            if not ok:
                fails.append(f"sign {m.name} {beta}")
            out(f"  {m.name:52s} beta={beta}: rho(K)={rk:.5g} rho(T)={rt:.8f} {'OK' if ok else 'FAIL'}")

    # ---------------- Result 2
    out("\n=== Result 2")
    for (m, beta, frac) in [(models[0], 0.5, 0.7), (models[0], 0.9, 0.9), (models[1], 0.9, 0.7), (models[3], 0.8, 0.8), (models[2], 0.95, 0.8)]:
        P = m.P; n2 = 2 * P; idx2 = m.idx2; d2 = len(idx2)
        es = thresholds(m, beta)['eta_star']; eta = frac * es
        pcs = m.pieces(beta)
        T = m.T(beta, eta, pcs); T0 = m.T0(beta, eta, pcs)
        Dl = add(T, T0, 1, -1)
        rhoT, Mpow = specrad(T)
        out(f"  -- {m.name} beta={beta} eta={frac}*eta* = {eta:.6g}: rho(T)={rhoT:.10f}")
        # (b) R = (I-T0)^{-1} Delta, r = rho(R) vs rho(K)
        IT0 = add(eye(d2), T0, 1, -1)
        R = solve(IT0, Dl)
        rR, _ = specrad(R)
        Kt, l = m.Ktil(beta, eta)
        eK, UK = jacobi(Kt); rK = max(eK); y = [UK[i][eK.index(rK)] for i in range(len(eK))]
        xs = [math.sqrt(eta / l[a]) * y[a] for a in range(len(l))]
        Xs = unvec(xs, m.idxP, P)
        if sum(Xs[i][i] for i in range(P)) < 0:
            Xs = scal(Xs, -1.0)
        okb = abs(rR - rK) < 1e-9
        out(f"     rho(R)={rR:.12f} rho(K)={rK:.12f} |diff|={abs(rR-rK):.1e}  X* min eig={lmin(Xs):.3e}")
        Y = blk(scal(m.Vop(Xs), eta * eta), zeros(P, P), zeros(P, P), zeros(P, P))
        Wv = [r[0] for r in solve(IT0, [[v] for v in vec(Y, idx2)])]
        W = unvec(Wv, idx2, n2)
        RW = mv(R, Wv); errRW = math.sqrt(sum((a - rK * b) ** 2 for a, b in zip(RW, Wv))) / math.sqrt(sum(b * b for b in Wv))
        TW = mv(T, Wv); idv = [w - (1 - rK) * yv for w, yv in zip(Wv, vec(Y, idx2))]
        errId = math.sqrt(sum((a - b) ** 2 for a, b in zip(TW, idv))) / math.sqrt(sum(b * b for b in Wv))
        W11 = sub(W, 0, P, 0, P)
        err11 = fro(add(W11, scal(Xs, rK), 1, -1)) / fro(W11)
        wmin = lmin(W)
        out(f"     W=J(X*): ||RW-rW||/||W||={errRW:.1e}  ||TW-(W-(1-r)Y)||/||W||={errId:.1e}  ||W11-rX*||/||W11||={err11:.1e}  min eig W={wmin:.3e}")
        T0W = unvec(mv(T0, Wv), idx2, n2)
        Sm = congr_inv(W, T0W); eS = jacobi(Sm)[0]; s_up = max(eS); s_low = min(eS)
        TY = unvec(mv(T, vec(Y, idx2)), idx2, n2)
        m2 = lmin(congr_inv(W, add(Y, TY)))
        lo_b = (1 - rK) * m2 / 2; hi_b = (1 - rK) * (1 - s_low)
        gapT = 1 - rhoT
        ok2 = okb and errRW < 1e-8 and errId < 1e-8 and abs(s_up - 1) < 1e-7 and lo_b <= gapT * (1 + 1e-9) and gapT <= hi_b * (1 + 1e-9) \
            and rhoT ** 2 <= 1 - (1 - rK) * m2 + 1e-12 and rhoT >= rK + (1 - rK) * s_low - 1e-12
        if not ok2:
            fails.append(f"R2b {m.name} beta={beta}")
        out(f"     one-step s=lmax(W^-1/2 T0(W) W^-1/2)={s_up:.12f} (predicted exactly 1)  s_low={s_low:.4g}  m2={m2:.4g}")
        out(f"     (1-r)m2/2={lo_b:.4g} <= 1-rho(T)={gapT:.4g} <= (1-r)(1-s_low)={hi_b:.4g};  1-r={1-rK:.4g}  {'OK' if ok2 else 'FAIL'}")
        # (a) burn-in. Q_* Perron of T from the last normalized power applied to I
        Iv = vec(eye(n2), idx2)
        qv = mv(Mpow, Iv)
        for _ in range(200):
            qv2 = mv(T, qv); nq = math.sqrt(sum(x * x for x in qv2)); qv = [x / nq for x in qv2]
        Qs = unvec(qv, idx2, n2)
        if sum(Qs[i][i] for i in range(n2)) < 0:
            Qs = scal(Qs, -1.0); qv = [-x for x in qv]
        resP = math.sqrt(sum((a - rhoT * b) ** 2 for a, b in zip(mv(T, qv), qv)))
        qmin = lmin(Qs)
        rng = random.Random(7)
        Am = [[rng.gauss(0, 1) for _ in range(P)] for _ in range(P)]
        Sxi = add(mm(Am, tr_(Am)), eye(P), 1.0, 0.5)
        Gm = blk(scal(Sxi, eta * eta), zeros(P, P), zeros(P, P), zeros(P, P))
        Gv = vec(Gm, idx2)
        Qinf_v = [r[0] for r in solve(add(eye(d2), T, 1, -1), [[g] for g in Gv])]
        Qinf = unvec(Qinf_v, idx2, n2)
        theta = 0.5 * (1 + rhoT)
        Qth_v = [r[0] for r in solve(add(eye(d2), scal(T, 1 / theta), 1, -1), [[x] for x in Iv])]
        Qth = unvec(Qth_v, idx2, n2)
        TQth = unvec(mv(T, Qth_v), idx2, n2)
        thchk = lmin(add(scal(Qth, theta), TQth, 1, -1))
        Bm_ = [[rng.gauss(0, 1) for _ in range(n2)] for _ in range(n2)]
        starts = {"Q0=0": zeros(n2, n2), "Q0=3Qinf+R R^T (Qinf-Q0 indefinite)": add(scal(Qinf, 3.0), mm(Bm_, tr_(Bm_)), 1.0, 0.05)}
        out(f"     Q*: ||T(Q*)-rho Q*||={resP:.1e} min eig Q*={qmin:.3e} (Q*>0: {qmin>0});  Q_theta (theta=(1+rho)/2): min eig(theta Q_th - T(Q_th))={thchk:.3e}")
        for lab, Q0 in starts.items():
            Dlt = add(Qinf, Q0, 1, -1)
            cp = max(0.0, lmax(congr_inv(Qs, Dlt))); cm = max(0.0, lmax(congr_inv(Qs, scal(Dlt, -1))))
            tp = max(0.0, lmax(congr_inv(Qth, Dlt))); tm = max(0.0, lmax(congr_inv(Qth, scal(Dlt, -1))))
            qv_t = vec(Q0, idx2)
            worst = float('inf'); worst_th = float('inf'); worst_id = 0.0
            Dv0 = vec(Dlt, idx2); Tt_D = Dv0[:]
            for t in range(0, 301):
                if t > 0:
                    qv_t = [a + b for a, b in zip(mv(T, qv_t), Gv)]
                    Tt_D = mv(T, Tt_D)
                if t % 20 == 0:
                    Ddiff = unvec([a - b for a, b in zip(Qinf_v, qv_t)], idx2, n2)
                    worst_id = max(worst_id, fro(add(Ddiff, unvec(Tt_D, idx2, n2), 1, -1)) / max(1e-300, fro(Dlt)))
                    # envelope tested on the propagated difference T^t(Qinf - Q0): the difference Qinf - Qt itself
                    # loses all relative accuracy once rho^t < 1e-16 (cancellation), which is not a property of T
                    Dt = unvec(Tt_D, idx2, n2)
                    sc = max(1e-300, rhoT ** t * fro(Qs))
                    up = lmin(add(scal(Qs, cp * rhoT ** t), Dt, 1, -1)) / sc
                    dn = lmin(add(Dt, scal(Qs, cm * rhoT ** t), 1, 1)) / sc
                    worst = min(worst, up, dn)
                    sc2 = max(1e-300, theta ** t * fro(Qth))
                    up2 = lmin(add(scal(Qth, tp * theta ** t), Dt, 1, -1)) / sc2
                    dn2 = lmin(add(Dt, scal(Qth, tm * theta ** t), 1, 1)) / sc2
                    worst_th = min(worst_th, up2, dn2)
            ok3 = worst > -1e-8 and worst_th > -1e-8 and worst_id < 1e-8 and thchk > 0
            if not ok3:
                fails.append(f"R2a {m.name} beta={beta} {lab}")
            out(f"     [{lab}] c+={cp:.4g} c-={cm:.4g}: min over t<=300 of normalized lmin of envelope gaps: Perron {worst:.2e}, Q_theta {worst_th:.2e}; ||(Qinf-Qt)-T^t(Qinf-Q0)|| rel {worst_id:.1e} {'OK' if ok3 else 'FAIL'}")

    out(f"\nTotal time {time.time()-t0:.1f}s")
    if fails:
        out("FAILURES: " + "; ".join(fails))
    else:
        out("ALL CHECKS PASSED")

if __name__ == "__main__":
    main()
