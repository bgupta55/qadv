"""Kernels: exact numpy simulation of ZZ-type fidelity kernels + classical kernels.

The numpy simulator matches Qiskit's ZZFeatureMap convention (H, P(2x_i), then
CX-P(theta_ij)-CX with theta_ij = 2(pi-x_i)(pi-x_j) when `shifted=True`).
It is validated against Qiskit in scripts/validate_install.py.
"""
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel


def make_edges(n, kind="linear"):
    if kind == "none":
        return []
    if kind == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if kind == "circular":
        return [(i, (i + 1) % n) for i in range(n)] if n > 2 else [(0, 1)]
    if kind == "full":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    raise ValueError(kind)


def _bits(n):
    idx = np.arange(1 << n)
    return ((idx[:, None] >> np.arange(n)) & 1).astype(np.float64)  # little-endian, like Qiskit


def _hadamard_all(psi, n):
    N = psi.shape[0]
    for q in range(n):
        v = psi.reshape(N, psi.shape[1] >> (q + 1), 2, 1 << q)
        a, b = v[:, :, 0, :], v[:, :, 1, :]
        psi = np.stack([(a + b), (a - b)], axis=2).reshape(N, -1) / np.sqrt(2)
    return psi


def zz_states(X, reps=2, bandwidth=1.0, edges="linear", shifted=True):
    X = np.atleast_2d(np.asarray(X, float)) * bandwidth
    N, n = X.shape
    if n > 16:
        raise ValueError("numpy statevector simulation limited to 16 qubits")
    E = make_edges(n, edges) if isinstance(edges, str) else list(edges)
    B = _bits(n)
    phase = 2.0 * X @ B.T
    for (i, j) in E:
        th = 2.0 * (np.pi - X[:, i]) * (np.pi - X[:, j]) if shifted else 2.0 * X[:, i] * X[:, j]
        phase = phase + th[:, None] * (B[:, i] != B[:, j])[None, :]
    D = np.exp(1j * phase)
    psi = np.full((N, 1 << n), 1.0 / np.sqrt(1 << n), dtype=complex) * D
    for _ in range(reps - 1):
        psi = _hadamard_all(psi, n) * D
    return psi


def fidelity_gram(Xa, Xb=None, **kw):
    Sa = zz_states(Xa, **kw)
    Sb = Sa if Xb is None else zz_states(Xb, **kw)
    return np.abs(Sa.conj() @ Sb.T) ** 2  # rows: Xa, cols: Xb


def signless_laplacian(n, edges):
    A = np.zeros((n, n))
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    return np.diag(A.sum(1)) + A


def twin_gram(Xa, Xb, gamma, edges="linear", a=np.pi ** 2):
    """Metric-matched classical rival: anisotropic Gaussian with M = I + a*Q
    (Q = signless Laplacian of the entanglement graph), free scalar bandwidth."""
    n = Xa.shape[1]
    M = np.eye(n) + a * signless_laplacian(n, make_edges(n, edges) if isinstance(edges, str) else edges)
    L = np.linalg.cholesky(M)
    return rbf_kernel(Xa @ L, Xb @ L, gamma=gamma)


def rbf_gram(Xa, Xb, gamma):
    return rbf_kernel(Xa, Xb, gamma=gamma)


def laplacian_gram(Xa, Xb, gamma):
    return laplacian_kernel(Xa, Xb, gamma=gamma)


def tanimoto_gram(Fa, Fb, power=1.0):
    A = Fa.astype(np.float32); B = Fb.astype(np.float32)
    inter = A @ B.T
    s = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return (np.where(s > 0, inter / np.maximum(s, 1e-9), 1.0)) ** power


def finite_shot_gram(K, shots, rng, symmetric=False):
    """Emulate compute-uncompute estimation: each entry ~ Binomial(shots, K)/shots."""
    K = np.clip(K, 0.0, 1.0)
    if symmetric:
        iu = np.triu_indices_from(K, 1)
        est = K.copy()
        est[iu] = rng.binomial(shots, K[iu]) / shots
        est = np.triu(est, 1); est = est + est.T + np.eye(K.shape[0])
        return est
    return rng.binomial(shots, K) / shots
