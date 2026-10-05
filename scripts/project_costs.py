#!/usr/bin/env python
"""Project kernel concentration, shot cost, circuit counts and QPU time BEFORE spending hardware budget.
  python scripts/project_costs.py --qubits 4 6 8 10 12 --n-train 30 --n-test 30 --sec-per-circuit 0.5
`--sec-per-circuit` must come from your own calibration job (run_hardware.py --dry-run, then a ~20-circuit test)."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qadv.kernels import fidelity_gram
from qadv import diagnostics as D
ap = argparse.ArgumentParser()
ap.add_argument("--qubits", type=int, nargs="+", default=[4, 6, 8, 10, 12]); ap.add_argument("--bandwidth", type=float, nargs="+", default=[0.1, 0.3, 0.6, 1.0])
ap.add_argument("--n-train", type=int, default=30); ap.add_argument("--n-test", type=int, default=30); ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--shots", type=int, default=4000); ap.add_argument("--sec-per-circuit", type=float, default=None); ap.add_argument("--open-plan-minutes", type=float, default=10)
a = ap.parse_args(); rng = np.random.default_rng(0)
print(f"{'qubits':>6} {'bw':>5} {'offdiag mean':>13} {'offdiag var':>12} {'shots/entry (SNR3)':>19} {'eff rank':>9}")
for n in a.qubits:
    X = rng.uniform(-1, 1, (40, n))
    for bw in a.bandwidth:
        K = fidelity_gram(X, reps=a.reps, edges="linear", bandwidth=bw); o = D.offdiag_stats(K)
        print(f"{n:>6} {bw:>5.2f} {o['mean']:>13.4f} {o['var']:>12.2e} {D.shots_to_resolve(K):>19,} {D.effective_rank(K):>9.1f}")
nc = a.n_train * (a.n_train - 1) // 2 + a.n_train * a.n_test
print(f"\ncircuits for N_train={a.n_train}, N_test={a.n_test}: {nc:,}   total shots @ {a.shots}: {nc * a.shots:,}")
if a.sec_per_circuit:
    t = nc * a.sec_per_circuit / 60
    print(f"projected QPU time: {t:.1f} min  ({t / a.open_plan_minutes:.1f} x the {a.open_plan_minutes:.0f}-minute Open Plan window)")
