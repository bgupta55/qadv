#!/usr/bin/env python
"""Numerical verification of the theorem register (blueprint section 5). Each check prints PASS/FAIL with numbers.
  python scripts/verify_theorems.py            # T2-T6 (T5b uses the noisy Aer emulation, ~1 min)
  python scripts/verify_theorems.py --skip-noisy
"""
import argparse, os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qadv import kernels as K, diagnostics as D
from qadv import hardware as H

ap = argparse.ArgumentParser(); ap.add_argument("--skip-noisy", action="store_true"); a = ap.parse_args()
rng = np.random.default_rng(0); res = {}
def rep(name, ok, **kw):
    res[name] = {"pass": bool(ok), **kw}; print(("PASS " if ok else "FAIL ") + name, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in kw.items()})

def g2(Kc, Kq, lam):    # g^2 = lambda_max( A^{1/2} Kq A^{1/2} ),  A = (Kc + lam I)^{-1}
    A = np.linalg.inv(Kc + lam * np.eye(len(Kc))); w, V = np.linalg.eigh(A); Ah = (V * np.sqrt(w)) @ V.T
    return float(np.linalg.eigvalsh(Ah @ Kq @ Ah).max())

# ---- T2 (cited, Tekeli): small-bandwidth metric = I + pi^2 Q for 1-rep shifted ZZ
n = 5; h = 1e-3; x0 = np.zeros((1, n)); G = np.zeros((n, n))
F = lambda x: K.fidelity_gram(x0, x.reshape(1, -1), reps=1, edges="linear", shifted=True)[0, 0]
for i in range(n):
    for j in range(n):
        ei, ej = np.eye(n)[i] * h, np.eye(n)[j] * h
        G[i, j] = -0.5 * (F(ei + ej) - F(ei - ej) - F(-ei + ej) + F(-ei - ej)) / (4 * h * h)
M = np.eye(n) + np.pi ** 2 * K.signless_laplacian(n, K.make_edges(n, "linear"))
rep("T2 small-bandwidth metric (1 rep)", np.linalg.norm(G - M) / np.linalg.norm(M) < 1e-3, rel_err=float(np.linalg.norm(G - M) / np.linalg.norm(M)))
G2 = np.zeros((n, n)); F2 = lambda x: K.fidelity_gram(x0, x.reshape(1, -1), reps=2, edges="linear", shifted=True)[0, 0]
for i in range(n):
    for j in range(n):
        ei, ej = np.eye(n)[i] * h, np.eye(n)[j] * h
        G2[i, j] = -0.5 * (F2(ei + ej) - F2(ei - ej) - F2(-ei + ej) + F2(-ei - ej)) / (4 * h * h)
A_ = np.stack([np.eye(n).ravel(), K.signless_laplacian(n, K.make_edges(n, "linear")).ravel()], 1); c, *_ = np.linalg.lstsq(A_, G2.ravel(), rcond=None)
rep("T2b 2-rep metric is NOT exactly I+aQ (expected failure of exactness)", np.linalg.norm(A_ @ c - G2.ravel()) / np.linalg.norm(G2) > 0.05, fit_rel_err=float(np.linalg.norm(A_ @ c - G2.ravel()) / np.linalg.norm(G2)))

# ---- T3 concentration inflation:  g^2(Kq) >= (1 - ||Kq - I||_2) / (lmin(Kc)+lam)   and   |g^2(Kq)-g^2(I)| <= ||Kq-I||/(lmin(Kc)+lam)
viol, trials, worst = 0, 0, 0.0
for t in range(300):
    N = int(rng.integers(10, 40)); nq = int(rng.integers(3, 11)); X = rng.uniform(-1, 1, (N, nq)); lam = 10 ** rng.uniform(-4, -1)
    Kc = K.rbf_gram(X, X, 10 ** rng.uniform(-2, 0.5) / nq); Kq = K.fidelity_gram(X, reps=int(rng.integers(1, 3)), bandwidth=10 ** rng.uniform(-1, 0.3))
    lmin = np.linalg.eigvalsh(Kc).min() + lam; eps = np.linalg.norm(Kq - np.eye(N), 2)
    lhs, rhs = g2(Kc, Kq, lam), (1 - eps) / lmin; gI = 1.0 / lmin
    trials += 1; ok = (lhs >= rhs - 1e-8 * max(1, abs(rhs))) and (abs(lhs - gI) <= eps / lmin + 1e-8 * gI)
    viol += (not ok); worst = max(worst, (rhs - lhs) / max(1e-12, abs(rhs)))
rep("T3 concentration-inflation bound (300 random trials)", viol == 0, violations=viol, trials=trials, worst_rel_violation=float(worst))
# the practical consequence: spurious g/sqrt(N) > 1 from concentration alone
N = 30; nq = 12; X = rng.uniform(-1, 1, (N, nq)); Kq = K.fidelity_gram(X, reps=2, bandwidth=0.8); Kc = K.rbf_gram(X, X, 0.5 / nq)
Kqn, Kcn = D.trace_normalize(Kq), D.trace_normalize(Kc)
gval = D.geometric_difference(Kc, Kq, lam=1e-3)
rep("T3 consequence: concentrated 12-qubit kernel passes g/sqrtN>1 (spurious)", gval / np.sqrt(N) > 1, g_over_sqrtN=gval / np.sqrt(N), norm_Kq_minus_I=float(np.linalg.norm(Kq - np.eye(N), 2)))

# ---- T4 norm identity and vacuity at g = sqrt(N)
N = 60; X = rng.uniform(-1, 1, (N, 8)); Kq = K.fidelity_gram(X, reps=1, bandwidth=0.6); Kc = K.rbf_gram(X, X, 0.3 / 8) + K.laplacian_gram(X, X, 0.3 / 8)
Kqn, Kcn = D.trace_normalize(Kq), D.trace_normalize(Kc); lam = 1e-3
w, V = np.linalg.eigh(Kqn); S = (V * np.sqrt(np.clip(w, 0, None))) @ V.T; A = np.linalg.inv(Kcn + lam * np.eye(N))
Mx = S @ A @ S; ev, EV = np.linalg.eigh((Mx + Mx.T) / 2); v = EV[:, -1]; f = S @ v
qn = float(f @ np.linalg.pinv(Kqn, rcond=1e-10) @ f); cn = float(f @ A @ f)
rep("T4a norm identity: quantum norm^2 = |v|^2 = 1, classical norm^2 = g^2", abs(qn - 1) < 1e-6 and abs(cn - ev[-1]) < 1e-8 * max(1, ev[-1]), quantum_norm2=qn, classical_norm2=cn, g2=float(ev[-1]))
def emp_rad(Kmat, B, reps=4000):
    L = np.linalg.cholesky(Kmat + 1e-10 * np.eye(len(Kmat))); sig = rng.choice([-1.0, 1.0], (reps, len(Kmat)))
    nr = np.linalg.norm(sig @ L, axis=1)
    return float(B * nr.mean() / len(Kmat)), float(B * nr.std() / np.sqrt(reps) / len(Kmat))
bound = lambda B: B * np.sqrt(np.trace(Kmat_) ) / N
(rq, seq), bq = emp_rad(Kqn, 1.0), 1.0 * np.sqrt(np.trace(Kqn)) / N
(rc, sec), bc = emp_rad(Kcn, np.sqrt(cn)), np.sqrt(cn) * np.sqrt(np.trace(Kcn)) / N
rep("T4b Rademacher: empirical <= B*sqrt(tr K)/N (Monte-Carlo, 4 s.e. slack); bound ratio = g", rq <= bq + 4 * seq and rc <= bc + 4 * sec and abs(bc / bq - np.sqrt(ev[-1])) < 1e-6 * np.sqrt(ev[-1]), emp_q=rq, bound_q=bq, emp_c=rc, bound_c=bc, ratio=float(bc / bq), g=float(np.sqrt(ev[-1])))
rep("T4c vacuity: classical bound >= 1 iff g >= sqrt(N)  (identity check)", (bc >= 1) == (np.sqrt(ev[-1]) >= np.sqrt(N)), classical_bound=float(bc), g_over_sqrtN=float(np.sqrt(ev[-1]) / np.sqrt(N)))

# ---- T6 shot complexity (Hoeffding + union bound)
eps, delta, Nn = 0.02, 0.05, 30; P = Nn * (Nn - 1) // 2; s = int(np.ceil(np.log(2 * P / delta) / (2 * eps ** 2)))
p = rng.uniform(0.01, 0.99, P); fails = 0; T = 400
for _ in range(T): fails += np.any(np.abs(rng.binomial(s, p) / s - p) > eps)
rep("T6 Hoeffding+union: all Gram entries within eps w.p. >= 1-delta", fails / T <= delta + 0.03, shots_bound=s, empirical_failure=float(fails / T), delta=delta)

def krr(Ktr, Kte, y, lam, center=True):
    n = len(Ktr)
    if not center: return Kte @ np.linalg.solve(Ktr + lam * np.eye(n), y)
    one = np.ones((n, n)) / n; ot = np.ones((Kte.shape[0], n)) / n
    Kc_tr = Ktr - one @ Ktr - Ktr @ one + one @ Ktr @ one; Kc_te = Kte - ot @ Ktr - Kte @ one + ot @ Ktr @ one
    return Kc_te @ np.linalg.solve(Kc_tr + lam * np.eye(n), y - y.mean()) + y.mean()

# ---- T5 global depolarising  p0 = f K + (1-f)/2^n   and  noise-as-ridge
nq = 6; f_true = 0.7; Xs = rng.uniform(-1, 1, (12, nq)); Kx = K.fidelity_gram(Xs, reps=2, bandwidth=0.6)
Kdep = f_true * Kx + (1 - f_true) / 2 ** nq; np.fill_diagonal(Kdep, 1.0)
fhat = D.fit_survival(Kx, Kdep - (1 - f_true) / 2 ** nq * 0)  # fit on affine-mapped off-diagonals
y = np.sin(3 * Xs[:, 0]) + 0.1 * rng.normal(size=12); lam = 0.05; Xt = rng.uniform(-1, 1, (30, nq)); Kt = K.fidelity_gram(Xt, Xs, reps=2, bandwidth=0.6)
Kt_dep = f_true * Kt + (1 - f_true) / 2 ** nq
lam_eff = (lam + 1 - f_true) / f_true
out = {}
for cen in (False, True):
    ph = krr(Kdep, Kt_dep, y, lam, cen); pe = krr(Kx, Kt, y, lam_eff, cen); pn = krr(Kx, Kt, y, lam, cen)
    out[cen] = (np.linalg.norm(ph - pe) / np.linalg.norm(ph), np.linalg.norm(ph - pn) / np.linalg.norm(ph))
rep("T5a noise-as-ridge, UNcentred (offset (1-f)/2^n not absorbed)", out[False][0] < out[False][1], rel_diff_ridge_model=float(out[False][0]), rel_diff_naive=float(out[False][1]), lam_eff=float(lam_eff))
rep("T5a' noise-as-ridge with intercept/centring (offset absorbed)", out[True][0] < 0.03 and out[True][0] < out[True][1], rel_diff_ridge_model=float(out[True][0]), rel_diff_naive=float(out[True][1]))

if not a.skip_noisy:
    t0 = time.time(); fm = dict(reps=2, edges="linear", shifted=True, bandwidth=0.6); nq = 6; Xs = rng.uniform(-1, 1, (10, nq)); Xt = rng.uniform(-1, 1, (8, nq))
    smp, be = H.get_sampler("aer_noisy", 3000)
    pt, ps = H.make_pairs(10), H.make_pairs(8, 10)
    e1, _ = H.estimate_entries_param(Xs, Xs, pt, smp, be, 3000, **fm); e2, _ = H.estimate_entries_param(Xt, Xs, ps, smp, be, 3000, **fm)
    Kh, Kth = H.assemble_gram(e1, 10), H.assemble_gram(e2, 8, 10); Kx = K.fidelity_gram(Xs, **fm); Kt = K.fidelity_gram(Xt, Xs, **fm)
    fh = D.fit_survival(Kx, Kh); y = np.sin(3 * Xs[:, 0]) + 0.1 * rng.normal(size=10); lam = 0.05
    le = (lam + 1 - fh) / fh
    ph = krr(Kh, Kth, y, lam, True); pr = krr(Kx, Kt, y, le, True); pn = krr(Kx, Kt, y, lam, True)
    c_r, c_n = float(np.corrcoef(ph, pr)[0, 1]), float(np.corrcoef(ph, pn)[0, 1]); er, en = np.linalg.norm(ph - pr) / np.linalg.norm(ph), np.linalg.norm(ph - pn) / np.linalg.norm(ph)
    rep("T5b noise-as-ridge predicts noisy-emulator KRR better than ignoring noise", er <= en, fitted_f=float(fh), rel_err_ridge_model=float(er), rel_err_ignore_noise=float(en), corr_ridge=c_r, corr_naive=c_n, seconds=float(time.time() - t0))

json.dump(res, open("results/theorem_checks.json", "w"), indent=1)
print("\nSUMMARY:", sum(v["pass"] for v in res.values()), "/", len(res), "checks passed")
