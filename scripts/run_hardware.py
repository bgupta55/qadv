#!/usr/bin/env python
"""Hardware (or emulated) validation of ONE pre-selected cell. Resumable. Saves raw counts.
  python scripts/run_hardware.py --dataset sk:breast_cancer --n-train 20 --n-test 20 --family zz_lin_r1 --mode aer_noisy
  python scripts/run_hardware.py ... --mode ibm --backend ibm_kingston --shots 4000 --dry-run   # preview only
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qadv.data import get_dataset
from qadv.models import FeaturePipe, quantum_families, tune_and_eval, _fit_predict, metrics, PRIMARY
from qadv import kernels as K, diagnostics as D, hardware as H
from run_grid import fewshot_split

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--family", default="zz_lin_r1")
ap.add_argument("--n-qubits", type=int, default=8); ap.add_argument("--n-train", type=int, default=20)
ap.add_argument("--n-test", type=int, default=20); ap.add_argument("--draw", type=int, default=0)
ap.add_argument("--mode", choices=["ideal", "aer_noisy", "ibm"], default="ideal")
ap.add_argument("--backend", default=None); ap.add_argument("--shots", type=int, default=4000)
ap.add_argument("--chunk", type=int, default=100); ap.add_argument("--opt-level", type=int, default=1)
ap.add_argument("--twirling", action="store_true"); ap.add_argument("--no-dd", action="store_true")
ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--out-dir", default="results/hardware")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
tag = f"{a.dataset.replace(':','_').replace('/','_')}_{a.family}_N{a.n_train}_M{a.n_test}_d{a.draw}_{a.mode}_s{a.shots}"
ds = get_dataset(a.dataset); rng = np.random.default_rng([0, a.draw, a.n_train, 7])
tr, te = fewshot_split(ds, a.n_train, a.n_test, rng)
pipe = FeaturePipe(a.n_qubits).fit(ds.X[tr]); Atr, Ate = pipe.transform(ds.X[tr]), pipe.transform(ds.X[te])
ytr, yte = ds.y[tr], ds.y[te]; task = ds.task
fam = [f for f in quantum_families(a.n_qubits) if f.name == a.family][0]
res, _, _ = tune_and_eval(fam, Atr, ytr, Ate, yte, task, np.random.default_rng(1))
th, C = res["theta"], res["C"]
fm = {"bandwidth": th, "reps": 2 if "r2" in a.family else 1,
      "edges": "full" if "full" in a.family else ("none" if "product" in a.family else "linear"), "shifted": "unshifted" not in a.family}
print(f"[{tag}] tuned theta={th:.4f} C={C:.2f}  exact test {PRIMARY[task]}={res[PRIMARY[task]]:.3f}")
Ktr_x = K.fidelity_gram(Atr, **fm); Kte_x = K.fidelity_gram(Ate, Atr, **fm)
pairs_tr, pairs_te = H.make_pairs(len(tr)), H.make_pairs(len(te), len(tr))
n_circ = len(pairs_tr) + len(pairs_te)
print(f"circuits: {len(pairs_tr)} (train Gram) + {len(pairs_te)} (test x train) = {n_circ};  total shots = {n_circ * a.shots:,}")
sampler, be = H.get_sampler(a.mode, a.shots, a.backend, use_dd=not a.no_dd, use_twirling=a.twirling)
if be is not None:
    st = H.transpiled_stats(Atr[0], Atr[1], be, a.opt_level, **fm); print("one transpiled circuit:", {k: st[k] for k in ("depth", "two_qubit_gates")})
    print(f"estimated QPU usage (2 PUBs): {H.usage_estimate_seconds(len(pairs_tr), a.shots, 6e-6) + H.usage_estimate_seconds(len(pairs_te), a.shots, 6e-6):.0f} s  [assumes rep_delay 250 us and ~6 us circuits; confirm with job.usage_estimation]")
if a.dry_run:
    print("dry-run: no jobs submitted"); sys.exit(0)
t0 = time.time(); log = lambda s: print(s, flush=True)
e_tr, info_tr = H.estimate_entries_param(Atr, Atr, pairs_tr, sampler, be, a.shots, a.opt_level, log, f"{a.out_dir}/{tag}_train_counts.json", **fm)
e_te, info_te = H.estimate_entries_param(Ate, Atr, pairs_te, sampler, be, a.shots, a.opt_level, log, f"{a.out_dir}/{tag}_test_counts.json", **fm)
Ktr_h, Kte_h = H.assemble_gram(e_tr, len(tr)), H.assemble_gram(e_te, len(te), len(tr))
rngs = np.random.default_rng(5)
Ktr_s = K.finite_shot_gram(Ktr_x, a.shots, rngs, True); Kte_s = K.finite_shot_gram(Kte_x, a.shots, rngs)
def ev(Ktr, Kte): return metrics(yte, _fit_predict(Ktr, ytr, Kte, task, C), task)
out = {"tag": tag, "mode": a.mode, "backend": a.backend, "shots": a.shots, "theta": th, "C": C, "n_circuits": n_circ,
       "wall_seconds": time.time() - t0, "exact": ev(Ktr_x, Kte_x), "shot_noise_only": ev(D.nearest_psd(Ktr_s), Kte_s),
       "hardware_raw": ev(Ktr_h, Kte_h), "hardware_psd": ev(D.nearest_psd(Ktr_h), Kte_h),
       "gram_frob_err_hw": float(np.linalg.norm(Ktr_h - Ktr_x) / np.linalg.norm(Ktr_x)),
       "offdiag_rel_err_hw": float(np.linalg.norm(D.offdiag(Ktr_h) - D.offdiag(Ktr_x)) / np.linalg.norm(D.offdiag(Ktr_x))),
       "survival_f_fit": D.fit_survival(Ktr_x, Ktr_h),
       "gram_frob_err_shot_only": float(np.linalg.norm(Ktr_s - Ktr_x) / np.linalg.norm(Ktr_x)),
       "offdiag_mean_exact": float(D.offdiag(Ktr_x).mean()), "offdiag_mean_hw": float(D.offdiag(Ktr_h).mean()),
       "offdiag_var_exact": float(D.offdiag(Ktr_x).var()), "offdiag_var_hw": float(D.offdiag(Ktr_h).var()),
       "min_eig_hw": float(np.linalg.eigvalsh((Ktr_h + Ktr_h.T) / 2).min()),
       "cka_hw_vs_exact": D.cka(Ktr_h, Ktr_x)}
json.dump(out, open(f"{a.out_dir}/{tag}_summary.json", "w"), indent=1); print(json.dumps(out, indent=1))
