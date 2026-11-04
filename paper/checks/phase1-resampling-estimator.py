"""Resampling estimator of the second moment on the full P = 64 operator of the failed Phase 1 cell.

Cell: P = 64, lambda_k = k^{-1/2}, beta = 0, B = 64, i.i.d. Gaussian features (kappa_feat = 0), secular root 1.926841, one-
dimensional mean-square ceiling along u_1 1.9394; the sample-path median of Phase 1 sat at the cap (2.0399), and with the cap
raised at 2.003. Estimator: S = 8 unit vectors, each step an independent minibatch per particle, y = (I - eta H_B) u, mu_hat =
mean |y|^2, resampling with probabilities |y|^2 / (S mu_hat); K = 64 independent blocks of 2m = 16 steps (1 024 steps x 8
particles, the budget of one Phase 1 cell: 1 000 steps x 8 streams); R_hat(eta) = mean_r M_hat_{2m}^{(r)} / mean_r M_hat_m^{(r)}
with M_hat the product of mu_hat; bisection of R_hat = 1 on [1.80, 2.04] in 12 steps with common random numbers; seeds 1..8.
H_B u is sampled exactly in O(P): with v = H^{1/2} u and e = v/|v|, W v = |v| ((c^2/B) e + (c/B) g_perp), c^2 ~ chi^2_B,
g_perp the projection of a standard normal vector off e (first column of a Wishart matrix in a frame where v is e_1).

Deterministic target (exact recursion, no sampling): tr T^{2m}(u1 u1^T) / tr T^m(u1 u1^T) crosses 1 at 1.926417 for m = 8.

Predictions, written before the run (2026-11-01):
  r1  the median over the 8 seeds lies within 1 % of the secular root, in [1.90757, 1.94611];
  r2  every seed lies within 3 % of the secular root;
  r3  the median lies below the median sample-path threshold 2.003 of the raised-cap diagnostic.
Writes the 8 crossings and the decisions to the JSON path given as the first argument.

Usage: python3 phase1-resampling-estimator.py <out.json>
"""
import json
import math
import random
import sys

P, B, BETA = 64, 64, 0.0
LAM = [1 / math.sqrt(k + 1) for k in range(P)]
SQ = [math.sqrt(x) for x in LAM]
ROOT, ONE_D, SAMPLE_PATH = 1.926841, 1.9394, 2.003
S, K, M = 8, 64, 8


def step(u, eta, rng):
    v = [SQ[k] * u[k] for k in range(P)]
    nv = math.sqrt(sum(x * x for x in v))
    e = [x / nv for x in v]
    c2 = sum(rng.gauss(0, 1) ** 2 for _ in range(B))
    c = math.sqrt(c2)
    g = [rng.gauss(0, 1) for _ in range(P)]
    ge = sum(a * b for a, b in zip(g, e))
    wv = [nv * ((c2 / B) * e[k] + (c / B) * (g[k] - ge * e[k])) for k in range(P)]
    return [u[k] - eta * SQ[k] * wv[k] for k in range(P)]


def ratio(eta, seed):
    rng = random.Random(seed)
    num = den = 0.0
    for _ in range(K):
        parts = [[1.0] + [0.0] * (P - 1) for _ in range(S)]
        prod = 1.0
        for t in range(2 * M):
            ys = [step(u, eta, rng) for u in parts]
            n2 = [sum(x * x for x in y) for y in ys]
            mu = sum(n2) / S
            prod *= mu
            if t == M - 1:
                den += prod
            total = sum(n2)
            cum, acc = [], 0.0
            for q in n2:
                acc += q / total
                cum.append(acc)
            new = []
            for _ in range(S):
                r = rng.random()
                i = next((j for j, cv in enumerate(cum) if r <= cv), S - 1)
                ni = math.sqrt(n2[i])
                new.append([x / ni for x in ys[i]])
            parts = new
        num += prod
    return num / den


def crossing(seed):
    lo, hi = 1.80, 2.04
    for _ in range(12):
        mid = (lo + hi) / 2
        if ratio(mid, seed) < 1.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


crossings = [crossing(seed) for seed in range(1, 9)]
med = sorted(crossings)[3] / 2 + sorted(crossings)[4] / 2
r1 = abs(med - ROOT) / ROOT <= 0.01
r2 = all(abs(x - ROOT) / ROOT <= 0.03 for x in crossings)
r3 = med < SAMPLE_PATH
out = {"cell": "P64_beta0_B64_gaussian-iid", "S": S, "K": K, "m": M, "bisection": [1.80, 2.04, 12],
       "secularRoot": ROOT, "oneDimCeiling": ONE_D, "samplePathMedianRaisedCap": SAMPLE_PATH,
       "crossings": crossings, "median": med, "medianRelErr": (med - ROOT) / ROOT,
       "predictions": {"r1": r1, "r2": r2, "r3": r3}}
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(out, indent=1) + "\n")
print(json.dumps({"crossings": [round(x, 5) for x in crossings], "median": round(med, 5), "relErr": (med - ROOT) / ROOT,
                  "r1": r1, "r2": r2, "r3": r3}))
