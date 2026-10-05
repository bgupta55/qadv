#!/usr/bin/env python
"""Projection tools.
  python scripts/project_results.py curves results/grid.jsonl [--extrapolate-to 200]   # learning-curve projection (needs >=5 N levels to be trusted)
  python scripts/project_results.py hardware --qubits 10 --offdiag-mean 0.002 --offdiag-std 0.004 --f 0.7 --shots 4000
"""
import argparse, sys
import numpy as np, pandas as pd
from scipy.optimize import curve_fit

def powerlaw(N, a, b, c): return a - b * np.power(N, -c)

def curves(path, target):
    d = pd.read_json(path, lines=True); R = d[d.kind != "SUMMARY"].copy()
    R["score"] = R.roc_auc.where(R.task == "classification", R.spearman)
    rows = []
    for ds, g in R.groupby("dataset"):
        per = g.groupby(["N", "family", "kind"]).score.mean().reset_index()
        q = per[per.kind == "quantum"].groupby("N").score.max(); c = per[per.kind != "quantum"].groupby("N").score.max()
        Ns = np.array(sorted(q.index), float)
        if len(Ns) < 5: print(f"[{ds}] only {len(Ns)} N levels: projection would be over-fit; run N in {{10,20,30,50,75,100}}"); 
        out = {"dataset": ds}
        for name, s in (("quantum", q), ("classical", c)):
            try:
                p, _ = curve_fit(powerlaw, Ns, s.loc[Ns].values, p0=[s.max() + .05, 1, .5], bounds=([0, 0, 0.05], [1.0, 50, 3]), maxfev=20000)
                out[name] = (p, powerlaw(target, *p))
            except Exception as e: out[name] = (None, np.nan)
        out["gap_at_target"] = out["quantum"][1] - out["classical"][1]
        out["asymptote_gap"] = (out["quantum"][0][0] - out["classical"][0][0]) if out["quantum"][0] is not None and out["classical"][0] is not None else np.nan
        rows.append(out)
    print(f"{'dataset':<22}{'proj quantum@N':>16}{'proj classical@N':>18}{'gap@N':>9}{'asymptote gap':>15}   (N={target:.0f}; treat as indicative only)")
    for o in rows: print(f"{o['dataset']:<22}{o['quantum'][1]:>16.3f}{o['classical'][1]:>18.3f}{o['gap_at_target']:>9.3f}{o['asymptote_gap']:>15.3f}")

def hardware(a):
    p = float(np.clip(a.offdiag_mean, 1e-9, 1 - 1e-9)); shot_sd = np.sqrt(p * (1 - p) / a.shots)
    sig = a.f * a.offdiag_std
    snr = sig / shot_sd; need = 9 * p * (1 - p) / (a.f * a.offdiag_std) ** 2
    print(f"signal sd after noise: {sig:.2e} | shot-noise sd per entry: {shot_sd:.2e} | SNR = {snr:.2f}")
    print(f"shots per entry needed for SNR>=3: {need:,.0f}  ({'OK' if snr >= 3 else 'INSUFFICIENT at current shots'})")
    print(f"depolarising floor 2^-n = {2.0 ** -a.qubits:.2e}; mean entry should stay above floor to be distinguishable.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("curves"); c.add_argument("jsonl"); c.add_argument("--extrapolate-to", type=float, default=200)
    h = sp.add_parser("hardware"); h.add_argument("--qubits", type=int, required=True); h.add_argument("--offdiag-mean", type=float, required=True)
    h.add_argument("--offdiag-std", type=float, required=True); h.add_argument("--f", type=float, default=0.8); h.add_argument("--shots", type=int, default=4000)
    a = ap.parse_args()
    curves(a.jsonl, a.extrapolate_to) if a.cmd == "curves" else hardware(a)
