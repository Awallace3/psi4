"""Read-only snapshots at the native RHF energy boundary; diagnostic, not timing."""
from contextlib import contextmanager
from unittest.mock import patch

from phoenix_timing import digest, save


@contextmanager
def energy_observer(core, observe):
    """Preserve the native call/return exactly; restore the descriptor on failure."""
    original = core.HF.compute_E

    def wrapped(wfn, *args, **kwargs):
        value = original(wfn, *args, **kwargs)
        observe(wfn, value)
        return value

    with patch.object(core.HF, "compute_E", wrapped):
        yield


def contraction(a, b):
    import numpy as np
    return np.sum(a.astype(np.longdouble) * b.T.astype(np.longdouble),
                  dtype=np.longdouble)


@contextmanager
def trace_energy(psi4, output, backend):
    import numpy as np
    selected = {20, 25, 40, 80, 99, 100}
    report = dict(scope=__doc__, extra_JK_calls=0, native_method_restored=False,
                  selected_iterations=sorted(selected), rows=[], snapshots={},
                  longdouble_mantissa_bits=int(np.finfo(np.longdouble).nmant))
    previous = None

    def observe(wfn, returned_energy):
        nonlocal previous
        iteration = int(wfn.iteration_)
        if iteration < 1:  # SAD does not yet have physical occupied orbitals.
            return
        jk = wfn.jk()
        expected = "cuESTJK" if backend == "gpu" else "MemDFJK"
        if jk.name() != expected:
            raise RuntimeError("Trace requires original SCF backend: " + expected)
        if len(jk.J()) != 1 or len(jk.K()) != 1 or len(jk.D()) != 1:
            raise RuntimeError("One full-dimer RHF density/J/K required for trace")
        arrays = dict(D=np.array(wfn.Da().np, copy=True),
                      H=np.array(wfn.H().np, copy=True),
                      S=np.array(wfn.S().np, copy=True),
                      J=np.array(jk.J()[0].np, copy=True),
                      K=np.array(jk.K()[0].np, copy=True),
                      JK_D=np.array(jk.D()[0].np, copy=True))
        if not all(a.shape == (922, 922) and np.isfinite(a).all() for a in arrays.values()):
            raise RuntimeError("Unexpected/nonfinite dimer trace arrays")
        d, h, s, j, k = [arrays[name] for name in ("D", "H", "S", "J", "K")]
        one, coulomb, exchange = 2 * contraction(d, h), 2 * contraction(d, j), -contraction(d, k)
        nuclear = wfn.molecule().nuclear_repulsion_energy()
        total = np.longdouble(nuclear) + one + coulomb + exchange
        row = dict(iteration=iteration, backend_name=jk.name(),
                   returned_energy_Eh=float(returned_energy), nuclear_Eh=float(nuclear),
                   one_electron_Eh=float(one), coulomb_Eh=float(coulomb),
                   exchange_Eh=float(exchange), reconstructed_energy_Eh=float(total),
                   returned_minus_reconstructed_Eh=float(np.longdouble(returned_energy) - total),
                   electron_count=float(2 * contraction(d, s)),
                   nominal_electron_count=int(wfn.nalpha() + wfn.nbeta()),
                   JK_D_vs_D_max_abs=float(np.max(np.abs(arrays["JK_D"] - d))),
                   H_asymmetry_max_abs=float(np.max(np.abs(h - h.T))),
                   S_asymmetry_max_abs=float(np.max(np.abs(s - s.T))),
                   J_asymmetry_max_abs=float(np.max(np.abs(j - j.T))))
        if previous is not None:
            row["changes"] = {name + "_max_abs": float(np.max(np.abs(arrays[name] - previous[name])))
                              for name in arrays}
            # Exact bilinear difference identities, not differences of totals.
            dd = d - previous["D"]
            row["delta_components_Eh"] = {
                "one_electron": float(2 * (contraction(dd, previous["H"])
                                          + contraction(d, h - previous["H"]))),
                "coulomb": float(2 * (contraction(dd, previous["J"])
                                      + contraction(d, j - previous["J"]))),
                "exchange": float(-contraction(dd, previous["K"])
                                  - contraction(d, k - previous["K"])),
            }
            row["delta_reconstructed_Eh"] = sum(row["delta_components_Eh"].values())
        if iteration in selected:
            co = np.array(wfn.Ca_subset("AO", "OCC").np, copy=True)
            row["occupied_orthonormality_max_abs"] = float(
                np.max(np.abs(co.T @ s @ co - np.eye(co.shape[1]))))
            row["D_vs_Cocc_CoccT_max_abs"] = float(np.max(np.abs(d - co @ co.T)))
            filename = f"iteration-{iteration:03d}.npz"
            np.savez(output / filename, Cocc=co, **arrays)
            report["snapshots"][filename] = digest(output / filename)
        report["rows"].append(row)
        previous = arrays
        save(output / "energy-trace.json", report)

    try:
        with energy_observer(psi4.core, observe):
            yield
    finally:
        report["native_method_restored"] = True
        if output.is_dir():
            save(output / "energy-trace.json", report)
