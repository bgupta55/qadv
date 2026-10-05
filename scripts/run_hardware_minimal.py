#!/usr/bin/env python
"""Minimal-QPU-time hardware experiment: TRAIN GRAM ONLY (N(N-1)/2 rows in ONE parametrised PUB).
Evidence produced from a single small job: kernel fidelity (off-diagonal error, survival f, CKA vs exact),
centred kernel-target alignment (KTA), and for quantum-exact, quantum-hardware, RBF, Laplacian and metric-twin kernels on the same N points
(classical gamma chosen to MAXIMISE KTA on the same labels, i.e. an optimistic classical oracle).
  python scripts/run_hardware_minimal.py --dataset "eng:delaney@0.6@rbf" --n 12 --mode aer_noisy --shots 1000
  python scripts/run_hardware_minimal.py ... --mode ibm --backend ibm_kingston --dry-run
"""
import argparse, json, os, sys, time
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qadv.data import get_dataset
from qadv import kernels as K, diagnostics as D, hardware as H

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--n", type=int, default=12); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--reps", type=int, default=1); ap.add_argument("--edges", default="linear"); ap.add_argument("--bandwidth", type=float, default=0.6)
ap.add_argument("--mode", choices=["ideal", "aer_noisy", "ibm"], default="ideal"); ap.add_argument("--backend", default=None)
ap.add_argument("--shots", type=int, default=1000); ap.add_argument("--rep-delay", type=float, default=2.5e-4, help="ASSUMED; read the real value from your backend")
ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--out-dir", default="results/hardware")
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)

ds = get_dataset(a.dataset); rng = np.random.default_rng(a.seed)
for _ in range(500):   # class-balanced draw of n points
    idx = rng.choice(len(ds.y), a.n, replace=False)
    if ds.task == "classification" and min(np.bincount(ds.y[idx].astype(int), minlength=2)) >= max(3, a.n // 4): break
X, y = ds.X[idx], ds.y[idx].astype(float); ypm = 2 * y - 1 if ds.task == "classification" else y
fm = dict(reps=a.reps, edges=a.edges, shifted=True, bandwidth=a.bandwidth)
Kx = K.fidelity_gram(X, **fm); pairs = H.make_pairs(a.n)
sampler, be = H.get_sampler(a.mode, a.shots, a.backend)
qc, m = H.kernel_param_circuit(X.shape[1], a.reps, a.edges)
if be is not None:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    isa = generate_preset_pass_manager(optimization_level=1, backend=be, scheduling_method="alap").run(qc)
    circ_s = float(isa.duration * be.dt) if getattr(isa, "duration", None) else 1e-5
    two_q = int(sum(v for k, v in isa.count_ops().items() if k in ("cz", "ecr", "cx"))); depth = isa.depth()
else:
    circ_s, two_q, depth = 1e-5, 0, qc.depth()
est = H.usage_estimate_seconds(len(pairs), a.shots, circ_s, a.rep_delay)
print(f"rows (pairs): {len(pairs)} | shots/row: {a.shots} | transpiled depth {depth}, 2q gates {two_q}, circuit time {circ_s * 1e6:.1f} us")
print(f"ESTIMATED QPU usage = 2 s + (rep_delay + circuit) * rows * shots = {est:.1f} s   (rep_delay assumed {a.rep_delay * 1e6:.0f} us; confirm with job.usage_estimation)")
if a.dry_run: sys.exit(0)
t0 = time.time(); ent, info = H.estimate_entries_param(X, X, pairs, sampler, be, a.shots, 1, None, f"{a.out_dir}/min_counts.json", **fm)
Kh = H.assemble_gram(ent, a.n); wall = time.time() - t0

def center(Km): return D.center(Km)
def kta(Km, yv):
    Kc = center(Km); Y = np.outer(yv - yv.mean(), yv - yv.mean()); return float((Kc * Y).sum() / (np.linalg.norm(Kc) * np.linalg.norm(Y) + 1e-12))
bin_ = ds.task == "classification"
cands = {"quantum_exact": Kx, "quantum_hardware": Kh}
G = np.logspace(-2.5, 1, 40) / X.shape[1]     # classical kernels get the BEST gamma by KTA on the same labels (optimistic for classical)
for name, fn in (("rbf", lambda g: K.rbf_gram(X, X, g)), ("laplacian", lambda g: K.laplacian_gram(X, X, g)), ("twin", lambda g: K.twin_gram(X, X, g, edges=a.edges))):
    cands[f"{name}_best_gamma_KTA"] = fn(max(G, key=lambda g: kta(fn(g), ypm)))
out = {"dataset": ds.name, "mode": a.mode, "backend": a.backend, "n": a.n, "rows": len(pairs), "shots": a.shots, "depth": depth, "two_qubit_gates": two_q,
       "circuit_seconds": circ_s, "estimated_qpu_seconds": est, "wall_seconds_client": wall, "job_info": {k: v for k, v in info.items() if v is not None},
       "offdiag_rel_err": float(np.linalg.norm(D.offdiag(Kh) - D.offdiag(Kx)) / np.linalg.norm(D.offdiag(Kx))), "survival_f": D.fit_survival(Kx, Kh),
       "cka_hw_vs_exact": D.cka(Kh, Kx), "min_eig_hw": float(np.linalg.eigvalsh((Kh + Kh.T) / 2).min()),
       "kta": {k: kta(v, ypm) for k, v in cands.items()}}
json.dump(out, open(f"{a.out_dir}/min_{ds.name}_{a.mode}_n{a.n}.json", "w"), indent=1); print(json.dumps(out, indent=1))
