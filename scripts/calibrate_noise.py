#!/usr/bin/env python
"""Fit the depolarising survival factor f for your backend/circuit family from a SMALL calibration set,
then use it to project hardware kernels for larger N without running them.
  python scripts/calibrate_noise.py --mode aer_noisy --qubits 6 8 --n 8 --bandwidth 0.6
  python scripts/calibrate_noise.py --mode ibm --backend ibm_kingston --qubits 6 --n 6 --shots 4000"""
import argparse, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qadv import kernels as K, diagnostics as D, hardware as H
ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="aer_noisy"); ap.add_argument("--backend", default=None)
ap.add_argument("--qubits", type=int, nargs="+", default=[6, 8]); ap.add_argument("--n", type=int, default=8)
ap.add_argument("--bandwidth", type=float, default=0.6); ap.add_argument("--reps", type=int, default=2); ap.add_argument("--shots", type=int, default=4000)
a = ap.parse_args(); rng = np.random.default_rng(0)
print(f"{'qubits':>6} {'2q gates':>9} {'depth':>6} {'f_fit':>7} {'f_pred(e2q=3e-3)':>17} {'rel offdiag err raw':>17} {'rel offdiag err after depol fit':>25}")
for nq in a.qubits:
    X = rng.uniform(-1, 1, (a.n, nq)); fm = dict(reps=a.reps, edges="linear", shifted=True, bandwidth=a.bandwidth)
    s, be = H.get_sampler(a.mode, a.shots, a.backend)
    st = H.transpiled_stats(X[0], X[1], be, 1, **fm) if be is not None else {"depth": -1, "two_qubit_gates": 0}
    Kx = K.fidelity_gram(X, **fm)
    Kh = H.assemble_gram(H.estimate_entries(X, X, H.make_pairs(a.n), s, be, a.shots, 100, 1, None, None, **fm), a.n)
    f = D.fit_survival(Kx, Kh); Km = D.depolarized_gram(Kx, f, nq); np.fill_diagonal(Km, np.diag(Kh))
    fpred = float(np.exp(-3e-3 * st["two_qubit_gates"])) if st["two_qubit_gates"] else float("nan")  # circuit and its inverse ~2x gates already counted -> rough
    ox = D.offdiag(Kx); e_raw = np.linalg.norm(D.offdiag(Kh) - ox) / np.linalg.norm(ox); e_mod = np.linalg.norm(D.offdiag(Kh) - D.offdiag(Km)) / np.linalg.norm(ox)
    print(f"{nq:>6} {st['two_qubit_gates']:>9} {st['depth']:>6} {f:>7.3f} {fpred:>17.3f} {e_raw:>17.3f} {e_mod:>25.3f}")
