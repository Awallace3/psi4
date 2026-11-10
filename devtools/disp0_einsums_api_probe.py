"""Small dependency/API probe, not a physical SAPT or GPU correctness gate.

Run with the selected coherent Python/native Einsums2 runtime. Import Psi4
first to expose library-order problems. All comparisons are FP64, with fixed
rtol=1e-11/atol=1e-12 declared before results are inspected.
"""

import gc
import json
import os

import psi4
import einsums as ein
import einsums.graph as cg
import numpy as np

RTOL, ATOL = 1e-11, 1e-12


def check(actual, reference):
    np.testing.assert_allclose(np.asarray(actual), reference, rtol=RTOL, atol=ATOL)


def main():
    rng = np.random.default_rng(73619)
    required = ("RuntimeTensorD", "RuntimeTensorViewD", "MemoryPool", "asarray")
    for symbol in required:
        assert hasattr(ein, symbol), f"Required Einsums2 API missing: {symbol}"
    for symbol in ("outer_sum", "direct_division"):
        assert hasattr(ein.linalg, symbol), f"Required linalg API missing: {symbol}"
    checks = []
    for nrb, nsb in ((1, 1), (3, 4), (2, 1)):
        na, nb, nq = 3, 2, 11
        m, n = na * nrb, nb * nsb
        a_np, b_np = rng.normal(size=(nq, m)), rng.normal(size=(nq, n))
        a, b = ein.asarray(a_np, name="A"), ein.asarray(b_np, name="B")
        v = ein.create_zero_tensor("V", [m, n])
        # Owning RuntimeTensors are column-major. Slices preserve owner
        # strides; reshapes must preserve occupied-fastest index semantics.
        assert np.asarray(a).flags.f_contiguous
        ein.einsum("ij <- ki ; kj", v, a, b, c_pf=0.0, ab_pf=1.0)
        reference = a_np.T @ b_np
        check(v, reference)

        ea, eb = -rng.uniform(0.3, 2.0, na), -rng.uniform(0.3, 2.0, nb)
        er, es = rng.uniform(0.1, 3.0, nrb), rng.uniform(0.1, 3.0, nsb)
        energies = [ein.asarray(x) for x in (ea, er, eb, es)]
        d = ein.create_zero_tensor("D", [m, n])
        dv = d.reshape_view([na, nrb, nb, nsb])
        ein.linalg.outer_sum(dv, [x[:] for x in energies], [1.0, -1.0, 1.0, -1.0])
        denominator = (ea[:, None, None, None] - er[None, :, None, None]
                       + eb[None, None, :, None] - es[None, None, None, :])
        check(dv, denominator)
        t = ein.create_zero_tensor("T", [m, n])
        ein.linalg.direct_division(1.0, v, d, 0.0, t)
        amplitude = reference / denominator.reshape((m, n), order="F")
        check(t, amplitude)
        energy = ein.create_zero_tensor("energy", [na, nb])
        ein.einsum("ab <- arbs ; arbs", energy,
                   t.reshape_view([na, nrb, nb, nsb]),
                   v.reshape_view([na, nrb, nb, nsb]), c_pf=0.0, ab_pf=4.0)
        expected = 4 * np.sum(amplitude.reshape(na, nrb, nb, nsb, order="F")
                              * reference.reshape(na, nrb, nb, nsb, order="F"), axis=(1, 3))
        check(energy, expected)
        checks.append(f"gemm-denominator-division-reduction-{nrb}x{nsb}")

    # Capture/replay through the default executor, including a strided
    # submatrix. Mutate an input between replays to detect stale device/host
    # cache state. This probe does NOT assert that any node runs on a GPU.
    a_np, b_np = rng.normal(size=(9, 5)), rng.normal(size=(9, 7))
    a, b = ein.asarray(a_np), ein.asarray(b_np)
    out = ein.create_zero_tensor("graph-output", [3, 4])
    g = cg.Graph("disp0-api-probe")
    with cg.capture(g):
        ein.einsum("ij <- ki ; kj", out, a[1:8, 1:4], b[1:8, 2:6],
                   c_pf=0.0, ab_pf=1.0)
    g.optimize()
    g.execute()
    check(out, a_np[1:8, 1:4].T @ b_np[1:8, 2:6])
    np.asarray(a)[...] *= -0.5
    g.execute()
    check(out, -0.5 * (a_np[1:8, 1:4].T @ b_np[1:8, 2:6]))
    checks.append("default-graph-replay-strided-view-mutation")

    # The view must keep a pool carve alive after its Python owner is gone.
    pool = ein.MemoryPool(4096, "disp0-api-probe")
    owner = pool.empty([3, 4], name="pooled")
    np.asarray(owner)[...] = np.arange(12).reshape(3, 4)
    view = owner.reshape_view([2, 6])
    expected = np.array(view)
    del owner
    gc.collect()
    check(view, expected)
    try:
        pool.reset()
    except RuntimeError:
        pass
    else:
        raise AssertionError("MemoryPool.reset accepted a live view")
    del view
    gc.collect()
    assert pool.bytes_used == 0
    pool.reset()
    checks.append("pool-view-keepalive-reset")

    observation = {}
    if os.environ.get("DISP0_API_RUNTIME_PROOF") == "1":
        from revised_physical_proof import observed
        observation["runtime_observation"] = observed()
    print(json.dumps({
        "status": "PASS",
        "scope": "CPU/API semantics only; not GPU execution or physical SAPT",
        "psi4": psi4.__file__, "psi4_version": psi4.__version__,
        "einsums": ein.__file__, "einsums_version": ein.__version__,
        "tolerances": {"rtol": RTOL, "atol": ATOL},
        "checks": checks,
        **observation,
    }, indent=2))


if __name__ == "__main__":
    main()
