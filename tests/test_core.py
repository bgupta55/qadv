"""Run: python tests/test_core.py   (no pytest needed)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from qadv.models import FeaturePipe
from qadv.data import synthetic_control, Dataset
from qadv import kernels as K, diagnostics as D
from run_grid import fewshot_split

def test_pipeline_ignores_test_data():
    rng = np.random.default_rng(0); Xtr = rng.normal(size=(30, 12)); Xte1 = rng.normal(size=(20, 12)); Xte2 = Xte1 * 100 + 50
    p = FeaturePipe(6).fit(Xtr); s0 = p.scale.copy(); p.transform(Xte1); p.transform(Xte2)
    assert np.allclose(s0, p.scale), "scaler state changed after seeing test data"
    p2 = FeaturePipe(6).fit(Xtr); assert np.allclose(p.transform(Xtr), p2.transform(Xtr))

def test_scaffold_disjoint():
    rng = np.random.default_rng(1); n = 400
    ds = Dataset("t", "x", "classification", rng.normal(size=(n, 5)), rng.integers(0, 2, n), groups=rng.integers(0, 60, n))
    for s in range(10):
        tr, te = fewshot_split(ds, 20, 100, np.random.default_rng(s))
        assert not set(ds.groups[tr]) & set(ds.groups[te]), "scaffold leakage between train and test"
        assert not set(tr) & set(te)

def test_kernel_properties():
    X = np.random.default_rng(2).uniform(-1, 1, (12, 6)); G = K.fidelity_gram(X, reps=2, bandwidth=0.5)
    assert np.allclose(G, G.T) and np.allclose(np.diag(G), 1) and np.linalg.eigvalsh(G).min() > -1e-10 and G.max() <= 1 + 1e-12

def test_geometric_difference_identity():
    X = np.random.default_rng(3).uniform(-1, 1, (15, 4)); G = K.rbf_gram(X, X, 0.3)
    g = D.geometric_difference(G, G, lam=1e-6); assert 0.9 < g < 1.1, g

def test_labels_not_used_in_diagnostics():
    import inspect
    for name, fn in inspect.getmembers(D, inspect.isfunction):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"y", "labels", "target", "ytr", "yte"}, f"{name} takes labels: {params}"

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"): fn(); print("PASS", name)
