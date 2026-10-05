#!/usr/bin/env python
"""Run FIRST. Verifies: (1) numpy simulator == Qiskit circuits, (2) compute-uncompute sampler path,
(3) small-bandwidth metric of the ZZ kernel ~ I + pi^2 Q (Tekeli-type twin), (4) optional IBM login."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qiskit.quantum_info import Statevector
from qadv.kernels import fidelity_gram, signless_laplacian, make_edges
from qadv.hardware import zz_circuit, get_sampler, estimate_entries, make_pairs, assemble_gram

def ok(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name, extra); return cond

rng = np.random.default_rng(0); n = 5; X = rng.uniform(-1, 1, (4, n)); fails = 0
for kw in [dict(reps=2, edges="linear", shifted=True, bandwidth=0.7), dict(reps=1, edges="full", shifted=False, bandwidth=1.3)]:
    K = fidelity_gram(X, **kw); S = [Statevector.from_instruction(zz_circuit(x, **kw)).data for x in X]
    K2 = np.array([[abs(np.vdot(a, b)) ** 2 for b in S] for a in S]); e = np.abs(K - K2).max()
    fails += not ok(f"numpy sim == qiskit statevector {kw}", e < 1e-10, f"max err {e:.1e}")
s, be = get_sampler("ideal", shots=20000); pairs = make_pairs(4)
K = assemble_gram(estimate_entries(X, X, pairs, s, None, 20000, **dict(reps=2, edges="linear", shifted=True, bandwidth=0.7)), 4)
Kx = fidelity_gram(X, reps=2, edges="linear", shifted=True, bandwidth=0.7); e = np.abs(K - Kx).max()
fails += not ok("sampler compute-uncompute within shot noise", e < 0.02, f"max err {e:.4f}")
x0 = np.zeros((1, n)); h = 1e-3; G = np.zeros((n, n))
F = lambda x: fidelity_gram(x0, x.reshape(1, -1), reps=1, edges="linear", shifted=True)[0, 0]
for a in range(n):
    for b in range(n):
        ea, eb = np.eye(n)[a] * h, np.eye(n)[b] * h
        G[a, b] = -0.5 * (F(ea + eb) - F(ea - eb) - F(-ea + eb) + F(-ea - eb)) / (4 * h * h)
M = np.eye(n) + np.pi ** 2 * signless_laplacian(n, make_edges(n, "linear"))
fails += not ok("small-bandwidth metric == I + pi^2 Q (reps=1, shifted)", np.linalg.norm(G - M) / np.linalg.norm(M) < 1e-3)
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService(); bes = svc.backends(simulator=False, operational=True)
    print("IBM login OK. Backends:", [(b.name, b.num_qubits) for b in bes])
except Exception as e:
    print("IBM login not configured (fine for local work):", repr(e)[:120])
print("ALL CORE CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED"); sys.exit(1 if fails else 0)
