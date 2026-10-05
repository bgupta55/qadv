"""Kernel families, nested tuning, classical rivals, metrics."""
import numpy as np
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, cohen_kappa_score, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from . import kernels as K


class FeaturePipe:
    """Fit on TRAIN only (no leakage): standardise -> PCA(n_qubits) -> scale to [-1, 1] by train max."""
    def __init__(self, n_components):
        self.k = n_components
    def fit(self, X):
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        self.pca = PCA(n_components=min(self.k, Z.shape[0] - 1, Z.shape[1]), random_state=0).fit(Z)
        P = self.pca.transform(Z)
        self.scale = np.abs(P).max(0) + 1e-9
        return self
    def transform(self, X):
        P = self.pca.transform(self.sc.transform(X)) / self.scale
        if P.shape[1] < self.k:  # pad if fewer components than qubits
            P = np.hstack([P, np.zeros((P.shape[0], self.k - P.shape[1]))])
        return np.clip(P, -1.5, 1.5)


class Family:
    def __init__(self, name, kind, grid, gram, view="x"):
        self.name, self.kind, self.grid, self.gram, self.view = name, kind, np.asarray(grid, float), gram, view


def quantum_families(n_qubits):
    bw = np.logspace(-1.7, 0.35, 12)   # bandwidth multiplier on features in [-1, 1]
    mk = lambda name, **kw: Family(name, "quantum", bw, lambda A, B, th, kw=kw: K.fidelity_gram(A, B, bandwidth=th, **kw))
    return [mk("zz_lin_r1", reps=1, edges="linear"), mk("zz_lin_r2", reps=2, edges="linear"),
            mk("zz_full_r1", reps=1, edges="full"), mk("zz_unshifted_lin_r1", reps=1, edges="linear", shifted=False),
            mk("product_r2", reps=2, edges="none")]


def classical_kernel_families(n_qubits, with_fp=False):
    g = np.logspace(-2.5, 1.0, 12) / n_qubits
    fam = [Family("rbf_matched", "classical_kernel", g, lambda A, B, th: K.rbf_gram(A, B, th)),
           Family("laplacian_matched", "classical_kernel", g, lambda A, B, th: K.laplacian_gram(A, B, th)),
           Family("twin_matched", "classical_kernel", g, lambda A, B, th: K.twin_gram(A, B, th, edges="linear"))]
    if with_fp:
        fam.append(Family("tanimoto_fp", "classical_fp", [1.0, 2.0, 3.0], lambda A, B, th: K.tanimoto_gram(A, B, power=th), view="fp"))
    return fam


def _cv_splits(y, task, rng, k=3):
    n = len(y)
    if task == "classification":
        c = np.bincount(y.astype(int), minlength=2).min()
        if c >= k:
            return list(StratifiedKFold(k, shuffle=True, random_state=int(rng.integers(1e6))).split(np.zeros(n), y))
    return list(KFold(min(k, n), shuffle=True, random_state=int(rng.integers(1e6))).split(np.zeros(n)))


def _fit_predict(Ktr, ytr, Kte, task, c):
    if task == "classification":
        if len(np.unique(ytr)) < 2:
            return np.zeros(Kte.shape[0])
        m = SVC(kernel="precomputed", C=c).fit(Ktr, ytr); return m.decision_function(Kte)
    m = KernelRidge(kernel="precomputed", alpha=1.0 / c).fit(Ktr, ytr - ytr.mean()); return m.predict(Kte) + ytr.mean()


def _score(y, pred, task):
    if task == "classification":
        return roc_auc_score(y, pred) if len(np.unique(y)) > 1 else 0.5
    return -mean_absolute_error(y, pred)


def metrics(y, pred, task):
    if task == "classification":
        lab = (pred > 0).astype(int) if pred.min() < 0 else (pred > 0.5).astype(int)
        return {"roc_auc": float(roc_auc_score(y, pred)), "pr_auc": float(average_precision_score(y, pred)),
                "mcc": float(matthews_corrcoef(y, lab)), "kappa": float(cohen_kappa_score(y, lab))}
    return {"mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred)),
            "spearman": float(spearmanr(y, pred)[0])}


PRIMARY = {"classification": "roc_auc", "regression": "spearman"}


def tune_and_eval(fam, Atr, ytr, Ate, yte, task, rng, refine=True):
    """Nested selection of (kernel parameter, C) by inner CV on TRAIN; one test evaluation."""
    Cs = np.logspace(-1, 3, 5)
    splits = _cv_splits(ytr, task, rng)
    def inner(th):
        Kt = fam.gram(Atr, Atr, th); best = (-1e9, None)
        for c in Cs:
            sc = np.mean([_score(ytr[va], _fit_predict(Kt[np.ix_(tr, tr)], ytr[tr], Kt[np.ix_(va, tr)], task, c), task) for tr, va in splits])
            if sc > best[0]: best = (sc, c)
        return best[0], best[1], Kt
    res = {float(th): inner(th) for th in fam.grid}
    if fam.kind != "classical_fp":   # expand the grid outward (up to 4x3 points) until the optimum is interior
        ratio = float(fam.grid[1] / fam.grid[0]); expanded = 0
        for _ in range(4):
            g = np.sort(list(res)); t0 = max(res, key=lambda t: res[t][0])
            if t0 == g[0]: new = [g[0] / ratio ** k for k in (1, 2, 3)]
            elif t0 == g[-1]: new = [g[-1] * ratio ** k for k in (1, 2, 3)]
            else: break
            for th in new: res[float(th)] = inner(th)
            expanded += 1
    if refine and len(fam.grid) > 3:   # refine continuously around the coarse optimum (avoids grid-boundary artefacts)
        t0 = max(res, key=lambda t: res[t][0]); g = np.sort(list(res))
        k = int(np.where(g == t0)[0][0]); lo = g[max(k - 1, 0)]; hi = g[min(k + 1, len(g) - 1)]
        for th in np.exp(np.linspace(np.log(lo), np.log(hi), 5))[1:-1]:
            res[float(th)] = inner(th)
    th = max(res, key=lambda t: res[t][0]); sc, c, Ktr = res[th]
    Kte = fam.gram(Ate, Atr, th)
    pred = _fit_predict(Ktr, ytr, Kte, task, c)
    out = {"family": fam.name, "kind": fam.kind, "theta": th, "C": float(c), "inner_score": float(sc),
           "at_grid_boundary": bool(th <= min(res) or th >= max(res)), **metrics(yte, pred, task)}
    return out, {float(t): v[2] for t, v in res.items()}, Ktr


def sklearn_rivals(Xtr, ytr, Xte, yte, task, rng, extra=True):
    """Standard non-kernel classical frameworks on FULL features (strongest-available information)."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.model_selection import GridSearchCV
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
    cls = task == "classification"
    cv = _cv_splits(ytr, task, rng)
    models = {
        "linear_full": (LogisticRegression(max_iter=2000) if cls else Ridge(), {("C" if cls else "alpha"): np.logspace(-3, 2, 6)}),
        "rf_full": (RandomForestClassifier(n_estimators=200, random_state=0) if cls else RandomForestRegressor(n_estimators=200, random_state=0), {"min_samples_leaf": [1, 3]}),
        "gbt_full": (GradientBoostingClassifier(random_state=0) if cls else GradientBoostingRegressor(random_state=0), {"max_depth": [2, 3], "n_estimators": [50, 150]}),
    }
    out = []
    for name, (m, grid) in models.items():
        gs = GridSearchCV(m, grid, cv=cv, scoring="roc_auc" if cls else "neg_mean_absolute_error").fit(A, ytr)
        pred = gs.predict_proba(B)[:, 1] if cls else gs.predict(B)
        out.append({"family": name, "kind": "classical_ml", "inner_score": float(gs.best_score_), **metrics(yte, pred, task)})
    if extra:
        try:  # optional foundation-model baseline (needs `pip install tabpfn` + weights download)
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            m = (TabPFNClassifier() if cls else TabPFNRegressor()).fit(A, ytr)
            pred = m.predict_proba(B)[:, 1] if cls else m.predict(B)
            out.append({"family": "tabpfn_full", "kind": "classical_ml", "inner_score": float("nan"), **metrics(yte, pred, task)})
        except Exception:
            pass
    return out
