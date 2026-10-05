#!/usr/bin/env python
"""Turn grid JSONL into the paper's tables/figures. Usage: python scripts/analyze.py results/grid.jsonl --out results/report"""
import argparse, os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon

ap = argparse.ArgumentParser(); ap.add_argument("jsonl"); ap.add_argument("--out", default="results/report")
ap.add_argument("--practical", type=float, default=0.02, help="practical-significance margin on the primary metric")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
d = pd.read_json(a.jsonl, lines=True)
S = d[d.kind == "SUMMARY"].copy(); R = d[d.kind != "SUMMARY"].copy()
rng = np.random.default_rng(0)

def boot(x, B=2000):
    x = np.asarray(x, float); m = rng.choice(x, (B, len(x))).mean(1); return np.percentile(m, [2.5, 97.5])

rows = []
for (ds, N), g in S.groupby(["dataset", "N"]):
    for ref, col in [("best classical kernel", "gap_vs_kernel"), ("best classical (any, incl. RF/GBT/linear)", "gap_vs_any")]:
        x = g[col].values; lo, hi = boot(x)
        try: p = wilcoxon(x).pvalue if np.any(x != 0) else 1.0
        except ValueError: p = 1.0
        rows.append(dict(dataset=ds, N=N, reference=ref, draws=len(x), mean_gap=x.mean(), ci_lo=lo, ci_hi=hi,
                         quantum_win_rate=(x > 0).mean(), practical_win_rate=(x > a.practical).mean(), p_raw=p))
T = pd.DataFrame(rows)
# Holm correction across all (dataset, N, reference) tests
order = T.p_raw.argsort().values; m = len(T); holm = np.empty(m); run = 0
for rank, idx in enumerate(order):
    run = max(run, min(1.0, (m - rank) * T.p_raw.iloc[idx])); holm[idx] = run
T["p_holm"] = holm
T["verdict"] = np.where((T.p_holm < 0.05) & (T.ci_lo > a.practical), "QUANTUM WIN (practical)",
                np.where((T.p_holm < 0.05) & (T.ci_hi < -a.practical), "CLASSICAL WIN", "no evidence of difference"))
T.round(4).to_csv(f"{a.out}/table_gaps.csv", index=False)

fam = R.groupby(["dataset", "N", "family"]).agg(metric=(R.columns[R.columns.get_loc("roc_auc")] if False else "roc_auc", "mean")).reset_index() if "roc_auc" in R else None
prim = R.copy(); prim["score"] = np.where(prim.task == "classification", prim.get("roc_auc"), prim.get("spearman"))
F = prim.groupby(["dataset", "N", "family", "kind"]).score.agg(["mean", "std"]).reset_index(); F.round(4).to_csv(f"{a.out}/table_family_scores.csv", index=False)
B = prim[prim.kind.isin(["quantum", "classical_kernel", "classical_fp"])].groupby(["kind", "family"]).at_grid_boundary.mean().reset_index()
B.round(3).to_csv(f"{a.out}/table_grid_boundary_rate.csv", index=False)

# ---- H1: do label-independent diagnostics predict the (label-dependent) gap?  leave-one-dataset-out
feats = ["sel_g_over_sqrtN", "max_g_over_sqrtN", "sel_offdiag_var", "sel_cka_vs_classical", "sel_frob_vs_classical", "sel_eff_rank"]
H = S.dropna(subset=feats + ["gap_vs_kernel"]).copy()
H["log_var"] = np.log10(H.sel_offdiag_var.clip(lower=1e-12))
use = ["sel_g_over_sqrtN", "max_g_over_sqrtN", "log_var", "sel_cka_vs_classical", "sel_frob_vs_classical", "sel_eff_rank"]
out = []
for f in use:
    rho, p = spearmanr(H[f], H.gap_vs_kernel); out.append(dict(feature=f, pooled_spearman=rho, p=p))
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
preds = []
for ds in H.dataset.unique():
    trn, tst = H[H.dataset != ds], H[H.dataset == ds]
    if len(trn) < 10 or tst.empty: continue
    sc = StandardScaler().fit(trn[use]); m = Ridge(alpha=5.0).fit(sc.transform(trn[use]), trn.gap_vs_kernel)
    preds.append(pd.DataFrame({"dataset": ds, "true": tst.gap_vs_kernel.values, "pred": m.predict(sc.transform(tst[use]))}))
if preds:
    P = pd.concat(preds); rho, p = spearmanr(P.true, P.pred)
    # per-dataset mean prediction vs per-dataset mean truth (cell-level aggregation is the honest unit for H1)
    pm = P.groupby("dataset").mean(); r2, p2 = spearmanr(pm.true, pm.pred) if len(pm) > 3 else (np.nan, np.nan)
    out.append(dict(feature="LODO ridge (draw level)", pooled_spearman=rho, p=p)); out.append(dict(feature="LODO ridge (dataset-mean level)", pooled_spearman=r2, p=p2))
pd.DataFrame(out).round(4).to_csv(f"{a.out}/table_H1_predictor.csv", index=False)

# ---- figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4.2))
for ds, g in T[T.reference == "best classical kernel"].groupby("dataset"):
    g = g.sort_values("N"); ax.errorbar(g.N, g.mean_gap, yerr=[g.mean_gap - g.ci_lo, g.ci_hi - g.mean_gap], marker="o", capsize=3, label=ds)
ax.axhline(0, c="k", lw=.8); ax.axhline(a.practical, c="gray", ls=":", lw=.8); ax.set_xlabel("N training compounds/samples"); ax.set_ylabel("quantum minus best classical kernel (primary metric)")
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{a.out}/fig_gap_vs_N.png", dpi=160); plt.close(fig)
fig, ax = plt.subplots(figsize=(5, 4.2)); ax.scatter(H.sel_g_over_sqrtN, H.gap_vs_kernel, c=pd.factorize(H.dataset)[0], s=14, cmap="tab10")
ax.axvline(1.0, c="r", ls="--", lw=.8); ax.axhline(0, c="k", lw=.8); ax.set_xlabel("g / sqrt(N)  (>1 necessary for advantage)"); ax.set_ylabel("gap vs best classical kernel")
fig.tight_layout(); fig.savefig(f"{a.out}/fig_g_vs_gap.png", dpi=160); plt.close(fig)
fig, ax = plt.subplots(figsize=(5, 4.2)); ax.scatter(H.sel_offdiag_var, H.sel_shots_to_resolve, s=14); ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("off-diagonal kernel variance"); ax.set_ylabel("shots/entry to resolve (heuristic)"); fig.tight_layout(); fig.savefig(f"{a.out}/fig_concentration_cost.png", dpi=160); plt.close(fig)

print("=== gap table (vs best classical kernel) ==="); print(T[T.reference == "best classical kernel"].round(3).to_string(index=False))
print("\n=== gap table (vs best classical of ANY type) ==="); print(T[T.reference != "best classical kernel"].round(3).to_string(index=False))
print("\n=== grid-boundary rate (should be low) ==="); print(B.to_string(index=False))
print("\n=== H1 predictor ==="); print(pd.DataFrame(out).round(3).to_string(index=False))
print("\nmax g/sqrtN observed:", float(S.max_g_over_sqrtN.max()), "| share of draws with selected g/sqrtN>1:", float((S.sel_g_over_sqrtN > 1).mean()))
