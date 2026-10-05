#!/usr/bin/env python
"""Few-shot grid: datasets x N_train x draws. Writes one JSON line per (draw, family) + one per draw summary."""
import argparse, json, os, sys, time, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qadv.data import get_dataset
from qadv.models import (FeaturePipe, quantum_families, classical_kernel_families, tune_and_eval,
                         sklearn_rivals, PRIMARY)
from qadv import diagnostics as D


def fewshot_split(ds, n_train, n_test, rng):
    n = len(ds.y)
    if ds.groups is not None:   # scaffold-aware: test scaffolds never appear in train
        g = rng.permutation(np.unique(ds.groups)); test = []
        sizes = {k: (ds.groups == k).sum() for k in g}
        chosen = set()
        for k in g:
            if len(test) >= n_test: break
            test += list(np.where(ds.groups == k)[0]); chosen.add(k)
        test = np.array(test[: max(n_test, 1)]); pool = np.setdiff1d(np.arange(n), np.where(np.isin(ds.groups, list(chosen)))[0])
    else:
        perm = rng.permutation(n); test, pool = perm[:n_test], perm[n_test:]
    for _ in range(200):
        tr = rng.choice(pool, size=min(n_train, len(pool)), replace=False)
        if ds.task == "regression" or min(np.bincount(ds.y[tr].astype(int), minlength=2)) >= 3:
            if ds.task == "regression" or len(np.unique(ds.y[test])) > 1:
                return tr, test
    raise RuntimeError("could not draw class-balanced few-shot split")


def run_draw(ds, n_qubits, n_train, n_test, rng, with_extra=True):
    task = ds.task; prim = PRIMARY[task]
    tr, te = fewshot_split(ds, n_train, n_test, rng)
    if getattr(ds, "prefeaturized", False):
        Aq, Bq = ds.X[tr], ds.X[te]
    else:
        pipe = FeaturePipe(n_qubits).fit(ds.X[tr])
        Aq, Bq = pipe.transform(ds.X[tr]), pipe.transform(ds.X[te])
    ytr, yte = ds.y[tr], ds.y[te]
    rows, qgrams, cgrams, q_sel = [], {}, [], {}
    for fam in classical_kernel_families(n_qubits, with_fp=ds.fp is not None):
        A, B = (Aq, Bq) if fam.view == "x" else (ds.fp[tr], ds.fp[te])
        r, grams, Ktr = tune_and_eval(fam, A, ytr, B, yte, task, rng); rows.append(r)
        if fam.view == "x": cgrams += list(grams.values()); r["_K"] = Ktr
    for fam in quantum_families(n_qubits):
        r, grams, Ktr = tune_and_eval(fam, Aq, ytr, Bq, yte, task, rng); r["_K"] = Ktr; rows.append(r)
    for r in sklearn_rivals(ds.X[tr], ytr, ds.X[te], yte, task, rng, extra=with_extra): rows.append(r)
    # ---- label-independent diagnostics for each quantum family
    best_cls_k = max([r for r in rows if r["kind"] == "classical_kernel"], key=lambda r: r["inner_score"])
    for r in rows:
        if r["kind"] != "quantum": continue
        Kq = r["_K"]; gmin, _ = D.min_geometric_difference(Kq, cgrams)
        o = D.offdiag_stats(Kq)
        r.update(g_min=gmin, g_over_sqrtN=gmin / np.sqrt(len(ytr)), offdiag_mean=o["mean"], offdiag_var=o["var"],
                 eff_rank=D.effective_rank(Kq), eps_identity=D.eps_identity(Kq), shots_to_resolve=D.shots_to_resolve(Kq),
                 cka_vs_best_classical=D.cka(Kq, best_cls_k["_K"]), frob_vs_best_classical=D.frob_dist(Kq, best_cls_k["_K"]))
    bq = max([r for r in rows if r["kind"] == "quantum"], key=lambda r: r["inner_score"])
    bc = best_cls_k
    allc = [r for r in rows if r["kind"] in ("classical_kernel", "classical_fp", "classical_ml") and not np.isnan(r["inner_score"])]
    bc_all = max(allc, key=lambda r: r["inner_score"])
    qmax = max([r for r in rows if r["kind"] == "quantum"], key=lambda r: r["g_over_sqrtN"])
    summary = {"kind": "SUMMARY", "metric": prim, "best_quantum": bq["family"], "q_metric": bq[prim],
               "best_classical_kernel": bc["family"], "ck_metric": bc[prim], "best_classical_any": bc_all["family"], "ca_metric": bc_all[prim],
               "gap_vs_kernel": bq[prim] - bc[prim], "gap_vs_any": bq[prim] - bc_all[prim],
               "sel_g_over_sqrtN": bq["g_over_sqrtN"], "max_g_over_sqrtN": qmax["g_over_sqrtN"],
               "sel_offdiag_var": bq["offdiag_var"], "sel_cka_vs_classical": bq["cka_vs_best_classical"],
               "sel_frob_vs_classical": bq["frob_vs_best_classical"], "sel_eff_rank": bq["eff_rank"], "sel_eps_identity": bq["eps_identity"], "sel_G1b_pass": bool(bq["eps_identity"] >= 1.0),
               "sel_shots_to_resolve": bq["shots_to_resolve"], "twin_gap": bq[prim] - [r for r in rows if r["family"] == "twin_matched"][0][prim]}
    for r in rows: r.pop("_K", None)
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--n-qubits", type=int, default=8)
    ap.add_argument("--n-train", type=int, nargs="+", default=[10, 20, 30, 50])
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-extra", action="store_true", help="skip optional TabPFN")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a") as f:
        for spec in a.datasets:
            ds = get_dataset(spec)
            print(f"[{ds.name}] n={len(ds.y)} task={ds.task} features={ds.X.shape[1]}", flush=True)
            for N in a.n_train:
                t0 = time.time()
                for d in range(a.draws):
                    rng = np.random.default_rng([a.seed, d, N, zlib.crc32(ds.name.encode()) % 10**6])
                    try:
                        rows, s = run_draw(ds, a.n_qubits, N, a.n_test, rng, not a.no_extra)
                    except Exception as e:
                        print("  draw failed:", repr(e)[:120]); continue
                    base = {"dataset": ds.name, "domain": ds.domain, "task": ds.task, "n_qubits": a.n_qubits, "N": N, "draw": d}
                    for r in rows + [s]: f.write(json.dumps({**base, **r}) + "\n")
                    f.flush()
                print(f"  N={N}: {a.draws} draws in {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
