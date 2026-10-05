"""Label-independent diagnostics (no labels are ever used here)."""
import numpy as np


def trace_normalize(K):
    return K * (K.shape[0] / np.trace(K))


def _sqrtm_psd(K):
    w, V = np.linalg.eigh((K + K.T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def geometric_difference(Kc, Kq, lam=1e-3):
    """Ridge-stabilised variant: g = sqrt(|| sqrt(Kq) (Kc + lam I)^-1 sqrt(Kq) ||_2).
    Kernels are trace-normalised to N. Compare with sqrt(N). Verify the exact
    regularised definition in Huang et al., Nat. Commun. 12, 2631 (2021) before publishing."""
    N = Kc.shape[0]
    Kc, Kq = trace_normalize(Kc), trace_normalize(Kq)
    S = _sqrtm_psd(Kq)
    M = S @ np.linalg.inv(Kc + lam * np.eye(N)) @ S
    return float(np.sqrt(np.linalg.norm((M + M.T) / 2, 2)))


def min_geometric_difference(Kq, classical_grams, lam=1e-3):
    vals = [geometric_difference(Kc, Kq, lam) for Kc in classical_grams]
    return float(np.min(vals)), int(np.argmin(vals))


def offdiag(K):
    iu = np.triu_indices_from(K, 1)
    return K[iu]


def offdiag_stats(K):
    o = offdiag(K)
    return {"mean": float(o.mean()), "var": float(o.var()), "min": float(o.min()), "max": float(o.max())}


def shots_to_resolve(K, snr=3.0):
    """Shots per kernel entry so that binomial standard error is spread/snr (heuristic)."""
    o = offdiag(K)
    sd = max(o.std(), 1e-12)
    p = float(np.clip(o.mean(), 1e-6, 1 - 1e-6))
    return int(np.ceil(p * (1 - p) / (sd / snr) ** 2))


def effective_rank(K):
    w = np.clip(np.linalg.eigvalsh((K + K.T) / 2), 0, None)
    p = w / w.sum() if w.sum() > 0 else w
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def center(K):
    n = K.shape[0]; H = np.eye(n) - 1.0 / n
    return H @ K @ H


def cka(K1, K2):
    A, B = center(K1), center(K2)
    return float((A * B).sum() / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-12))


def frob_dist(K1, K2):
    A, B = trace_normalize(K1), trace_normalize(K2)
    return float(np.linalg.norm(A - B) / np.sqrt(A.shape[0] ** 2))


def nearest_psd(K):
    K = (K + K.T) / 2
    w, V = np.linalg.eigh(K)
    return (V * np.clip(w, 0, None)) @ V.T


def depolarized_gram(K, f, n_qubits):
    """Global-depolarising projection of a kernel measured on noisy hardware:
    p0_hw = f * K + (1 - f) / 2**n.  f is the circuit 'survival' factor; fit it from the
    variance ratio Var(K_hw)/Var(K_exact) ~ f**2 on a small calibration set."""
    Kh = f * K + (1 - f) / 2 ** n_qubits
    return Kh


def fit_survival(K_exact, K_hw):
    """Least-squares f from off-diagonal entries (ignores the 2^-n floor)."""
    a, b = offdiag(K_exact), offdiag(K_hw)
    return float(np.clip((a * b).sum() / (a * a).sum(), 0.0, 1.0))


def eps_identity(K):
    """epsilon_Q = || K - I ||_2 .  Gate G1b: epsilon_Q >= 1 is required, otherwise Theorem T3 shows that
    g/sqrt(N) > 1 can be produced by concentration alone (K ~ identity) and is NOT evidence of advantage."""
    return float(np.linalg.norm(K - np.eye(K.shape[0]), 2))
