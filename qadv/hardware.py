"""Hardware path: compute-uncompute kernel estimation on ideal / noisy-simulated / IBM backends.

Same circuit builder is used for all backends and matches qadv.kernels.zz_states exactly.
Qiskit/runtime APIs change often: all version-sensitive calls are isolated here.
"""
import json, time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from .kernels import make_edges


def zz_circuit(x, reps=2, edges="linear", shifted=True, bandwidth=1.0):
    x = np.asarray(x, float) * bandwidth
    n = len(x)
    E = make_edges(n, edges) if isinstance(edges, str) else list(edges)
    qc = QuantumCircuit(n)
    for _ in range(reps):
        qc.h(range(n))
        for i in range(n):
            qc.p(2.0 * x[i], i)
        for (i, j) in E:
            th = 2.0 * (np.pi - x[i]) * (np.pi - x[j]) if shifted else 2.0 * x[i] * x[j]
            qc.cx(i, j); qc.p(th, j); qc.cx(i, j)
    return qc


def kernel_circuit(x, y, **kw):
    n = len(x)
    qc = zz_circuit(x, **kw)
    qc.compose(zz_circuit(y, **kw).inverse(), inplace=True)
    qc.measure_all()
    return qc


def make_pairs(n_a, n_b=None):
    """Index pairs needed: upper triangle (symmetric Gram) or full rectangle (cross Gram)."""
    if n_b is None:
        return [(i, j) for i in range(n_a) for j in range(i + 1, n_a)]
    return [(i, j) for i in range(n_a) for j in range(n_b)]


def get_sampler(mode, shots=4096, backend_name=None, seed=0, use_dd=True, use_twirling=False, min_qubits=20, fake="FakeKingston"):
    """Returns (sampler, backend_or_None). mode in {'ideal','aer_noisy','ibm'}."""
    if mode == "ideal":
        from qiskit.primitives import StatevectorSampler
        return StatevectorSampler(default_shots=shots, seed=seed), None
    if mode == "aer_noisy":
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2 as AerSampler
        import qiskit_ibm_runtime.fake_provider as fp
        be = AerSimulator.from_backend(getattr(fp, fake)())   # FakeKingston = Heron r2 noise snapshot
        return AerSampler.from_backend(be, default_shots=shots, seed=seed), be
    if mode == "ibm":
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        svc = QiskitRuntimeService()  # run QiskitRuntimeService.save_account(...) once beforehand
        be = svc.backend(backend_name) if backend_name else svc.least_busy(operational=True, simulator=False, min_num_qubits=min_qubits)
        s = SamplerV2(mode=be)  # job mode (Open Plan: no sessions; check current docs)
        s.options.default_shots = shots
        if use_dd:
            s.options.dynamical_decoupling.enable = True
            s.options.dynamical_decoupling.sequence_type = "XpXm"
        if use_twirling:
            s.options.twirling.enable_gates = True
            s.options.twirling.enable_measure = True
        return s, be
    raise ValueError(mode)


def estimate_entries(X_left, X_right, pairs, sampler, backend=None, shots=4096, chunk=100,
                     opt_level=1, log=None, dump_path=None, **fm_kw):
    """Estimate K_ij = P(0...0) for each (i, j) in pairs. Returns dict pair -> (p0, counts_dict)."""
    n = X_left.shape[1]
    zero = "0" * n
    pm = generate_preset_pass_manager(optimization_level=opt_level, backend=backend) if backend is not None else None
    out, t0 = {}, time.time()
    for s in range(0, len(pairs), chunk):
        batch = pairs[s:s + chunk]
        circs = [kernel_circuit(X_left[i], X_right[j], **fm_kw) for (i, j) in batch]
        if pm is not None:
            circs = pm.run(circs)
        job = sampler.run([(c,) for c in circs], shots=shots)
        res = job.result()
        for (i, j), r in zip(batch, res):
            counts = r.data.meas.get_counts()
            tot = sum(counts.values())
            out[(i, j)] = (counts.get(zero, 0) / tot, counts)
        if log:
            log(f"  {min(s + chunk, len(pairs))}/{len(pairs)} circuits  ({time.time() - t0:.1f}s)")
    if dump_path:
        with open(dump_path, "w") as f:
            json.dump({f"{i},{j}": v[1] for (i, j), v in out.items()}, f)
    return out


def assemble_gram(entries, n_a, n_b=None):
    if n_b is None:
        K = np.eye(n_a)
        for (i, j), (p, _) in entries.items():
            K[i, j] = K[j, i] = p
        return K
    K = np.zeros((n_a, n_b))
    for (i, j), (p, _) in entries.items():
        K[i, j] = p
    return K


def transpiled_stats(x, y, backend, opt_level=1, **fm_kw):
    """Depth / two-qubit-gate count of one kernel circuit after transpilation (cost preview)."""
    pm = generate_preset_pass_manager(optimization_level=opt_level, backend=backend)
    c = pm.run(kernel_circuit(x, y, **fm_kw))
    ops = c.count_ops()
    two_q = sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx"))
    return {"depth": c.depth(), "two_qubit_gates": int(two_q), "ops": dict(ops)}


# ----------------------------------------------------------------------------------------------
# Parametrised single-PUB path (recommended for real hardware: ONE transpilation, ONE PUB).
# IBM documents a per-sub-job overhead of ~2 s in its usage formula, so many tiny PUBs are costly.
# Every gate angle is its own free parameter (linear in the parameters), so the runtime can bind them.
# ----------------------------------------------------------------------------------------------
from qiskit.circuit import ParameterVector


def n_angles(n, edges="linear"):
    E = make_edges(n, edges) if isinstance(edges, str) else list(edges)
    return n + len(E)


def angle_vector(x, reps=2, edges="linear", shifted=True, bandwidth=1.0):
    """Angles [2*x_i ... , theta_ij ...] for one data point (reps handled by re-using the same angles)."""
    x = np.asarray(x, float) * bandwidth
    n = len(x)
    E = make_edges(n, edges) if isinstance(edges, str) else list(edges)
    one = [2.0 * xi for xi in x]
    two = [(2.0 * (np.pi - x[i]) * (np.pi - x[j]) if shifted else 2.0 * x[i] * x[j]) for (i, j) in E]
    return np.array(one + two)


def _zz_from_params(phi, n, E, reps):
    qc = QuantumCircuit(n)
    for _ in range(reps):
        qc.h(range(n))
        for i in range(n):
            qc.p(phi[i], i)
        for k, (i, j) in enumerate(E):
            qc.cx(i, j); qc.p(phi[n + k], j); qc.cx(i, j)
    return qc


def kernel_param_circuit(n, reps=2, edges="linear"):
    E = make_edges(n, edges) if isinstance(edges, str) else list(edges)
    m = n + len(E)
    a, b = ParameterVector("a", m), ParameterVector("b", m)
    qc = _zz_from_params(a, n, E, reps)
    qc.compose(_zz_from_params(b, n, E, reps).inverse(), inplace=True)
    qc.measure_all()
    return qc, m


def estimate_entries_param(X_left, X_right, pairs, sampler, backend=None, shots=4096, opt_level=1,
                           log=None, dump_path=None, scheduling=True, **fm_kw):
    """Same output as estimate_entries but ONE parametrised circuit and ONE PUB (rows = pairs)."""
    n = X_left.shape[1]
    reps = fm_kw.get("reps", 2); edges = fm_kw.get("edges", "linear")
    shifted = fm_kw.get("shifted", True); bw = fm_kw.get("bandwidth", 1.0)
    qc, m = kernel_param_circuit(n, reps, edges)
    if backend is not None:
        kw = dict(optimization_level=opt_level, backend=backend)
        if scheduling:
            kw["scheduling_method"] = "alap"
        isa = generate_preset_pass_manager(**kw).run(qc)
    else:
        isa = qc
    AL = np.array([angle_vector(x, reps, edges, shifted, bw) for x in X_left])
    AR = np.array([angle_vector(x, reps, edges, shifted, bw) for x in X_right])
    vals = np.array([np.concatenate([AL[i], AR[j]]) for (i, j) in pairs])
    t0 = time.time()
    job = sampler.run([(isa, vals)], shots=shots)
    res = job.result()[0]
    ba = res.data.meas
    out = {}
    for k, (i, j) in enumerate(pairs):
        c = ba.get_int_counts(k)
        tot = sum(c.values())
        out[(i, j)] = (c.get(0, 0) / tot, {format(key, f"0{n}b"): v for key, v in c.items()})
    if log:
        log(f"  {len(pairs)} parameter rows in one PUB  ({time.time() - t0:.1f}s wall)")
    if dump_path:
        with open(dump_path, "w") as f:
            json.dump({f"{i},{j}": v[1] for (i, j), v in out.items()}, f)
    info = {"job_id": getattr(job, "job_id", lambda: None)() if callable(getattr(job, "job_id", None)) else None}
    try:
        info["usage"] = job.usage()
    except Exception:
        pass
    info["circuit_seconds"] = (isa.duration * backend.dt) if (backend is not None and getattr(isa, "duration", None) and getattr(backend, "dt", None)) else None
    info["depth"] = isa.depth()
    info["two_qubit_gates"] = int(sum(v for k, v in isa.count_ops().items() if k in ("cz", "ecr", "cx")))
    return out, info


def usage_estimate_seconds(n_rows, shots, circuit_seconds, rep_delay=2.5e-4, per_subjob_overhead=2.0, n_pubs=1):
    """IBM's documented estimate: overhead(~2 s per sub-job) + (rep_delay + circuit length) * executions.
    rep_delay default here is an ASSUMPTION; read the real value from the backend/sampler options and
    cross-check with job.usage_estimation before submitting."""
    return n_pubs * per_subjob_overhead + (rep_delay + circuit_seconds) * n_rows * shots
