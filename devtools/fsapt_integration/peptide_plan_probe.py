"""Identical-orbital retained/fresh JK comparison, after (never during) SCF."""
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from dimer_fixed_jk import delta_summary
from dimer_scf_trace import contraction
from small_dimer_probe import INPUT_SHA, basis_receipt
from phoenix_timing import digest, save

STATUS = "EXECUTED_PLAN_REPLAY_DIAGNOSTIC_ONLY"


def density_reconstruction(co, stored):
    """C bytes are fixed; a separate BLAS multiplication need not be bitwise."""
    import numpy as np
    reconstructed = co @ co.T
    np.testing.assert_allclose(stored, reconstructed, atol=1e-13, rtol=0)
    difference = reconstructed - stored
    return dict(max_abs=float(np.max(np.abs(difference))),
                rms=float(np.sqrt(np.mean(difference**2))),
                unequal_elements=int(np.count_nonzero(difference)),
                bitwise_equal=stored.tobytes() == reconstructed.tobytes())


@contextmanager
def capture_orthogonalizer(core, captured):
    """One copy after iterations, before HF::finalize releases X."""
    original = core.HF.finalize_energy
    def wrapped(wfn, *args, **kwargs):
        import numpy as np
        x = wfn._diis_orthogonalizer()
        if x is None:
            raise RuntimeError("SCF orthogonalizer missing before finalization")
        captured["X"] = np.array(x.np, copy=True)
        return original(wfn, *args, **kwargs)
    with patch.object(core.HF, "finalize_energy", wrapped):
        yield


def option_receipt(core):
    keys = ("DF_FITTING_CONDITION", "INTS_TOLERANCE", "SCREENING",
            "CUEST_MIXED_PRECISION", "CUEST_DFK_SLICES", "CUEST_DFK_MODULI",
            "CUEST_DIRECTJ", "CUEST_GRIDK")
    options = core.get_options()
    module = options.get_current_module()
    try:
        values = {key: dict(effective=core.get_option("SCF", key),
                            global_value=core.get_global_option(key),
                            global_changed=core.has_global_option_changed(key),
                            local_changed=core.has_local_option_changed("SCF", key))
                  for key in keys}
        return dict(current_module=module, values=values)
    finally:
        # core.get_option itself switches module scope; observing it must not
        # silently turn this into a changed-scope experiment.
        options.set_current_module(module)


def jk_state(jk):
    return dict(name=jk.name(), memory_doubles=jk.memory(),
                omega=jk.get_omega(), alpha=jk.get_omega_alpha(),
                beta=jk.get_omega_beta(), wcombine=jk.get_wcombine())


def fresh_plan(core, primary, auxiliary, state, backend):
    memory = state["memory_doubles"]
    jk = core.JK.build(primary, auxiliary, jk_type="DF", do_wK=False, memory=memory)
    expected = "cuESTJK" if backend == "gpu" else "MemDFJK"
    if jk.name() != expected:
        raise RuntimeError("Unexpected fresh JK backend: " + jk.name())
    jk.set_memory(memory)
    jk.set_do_J(True)
    jk.set_do_K(True)
    jk.set_do_wK(False)
    jk.set_omega(state["omega"])
    jk.set_omega_alpha(state["alpha"])
    jk.set_omega_beta(state["beta"])
    jk.set_wcombine(state["wcombine"])
    jk.set_print(3)
    return jk


def replay_variants(state, backend):
    if backend == "cpu":
        return [dict(state), dict(state)]
    if backend != "gpu" or state["omega"] != 0 or state["alpha"] != 1:
        raise ValueError("Unused-beta control requires explicit zero-omega HF GPU replay")
    # Native HF actually supplied beta=1 even with do_wK=False and omega=0.
    # Interleave the actual setting with beta=0: do not silently normalize it.
    return [dict(state), {**state, "beta": 0.0},
            dict(state), {**state, "beta": 0.0}]


def rebuilt_state(d, h, s, x, j, k, nuclear):
    import numpy as np
    fock = h + 2 * j - k
    gradient = x.T @ (fock @ d @ s - s @ d @ fock) @ x
    return dict(total_rebuilt_Eh=float(np.longdouble(nuclear) + 2 * contraction(d, h)
                                      + 2 * contraction(d, j) - contraction(d, k)),
                orthonormal_commutator_rms=float(np.sqrt(np.mean(gradient**2))))


def fixed_samples(core, jk, co, d, label, arrays, rows, baseline=None, context=None):
    import numpy as np
    coefficient = core.Matrix.from_array(co.copy())
    original_bytes = co.tobytes()
    jk.C_clear()
    jk.C_left_add(coefficient)
    first = None
    for call in range(4):
        jk.compute()
        if coefficient.np.tobytes() != original_bytes:
            raise RuntimeError("JK modified supplied occupied coefficients")
        j, k, actual_d = [np.array(v[0].np, copy=True) for v in (jk.J(), jk.K(), jk.D())]
        if not all(v.shape == d.shape and np.isfinite(v).all() for v in (j, k, actual_d)):
            raise RuntimeError("Incomplete native JK arrays")
        np.testing.assert_allclose(actual_d, d, atol=1e-12, rtol=1e-12)
        if first is None:
            first = (j, k)
        key = f"{label}_{call}"
        arrays["J_" + key], arrays["K_" + key] = j, k
        arrays["D_" + key] = actual_d
        rows.append(dict(label=key, state=jk_state(jk),
                         D_max_abs=float(np.max(np.abs(actual_d - d))),
                         coulomb_Eh=float(2 * contraction(d, j)),
                         exchange_Eh=float(-contraction(d, k)),
                         vs_same_plan_first=delta_summary(*first, j, k, d),
                         vs_retained_first=delta_summary(*(baseline or first), j, k, d)))
        if context is not None:
            h, s, x, nuclear = context
            rows[-1].update(rebuilt_state(d, h, s, x, j, k, nuclear))
    return first


def retained_and_fresh(psi4, wfn, output, orthogonalizer=None):
    import numpy as np
    primary, auxiliary = wfn.basisset(), wfn.get_basisset("DF_BASIS_SCF")
    original = wfn.jk()
    state = jk_state(original)
    post_options = option_receipt(psi4.core)
    if state["name"] != "cuESTJK":
        raise RuntimeError("Producer requires retained cuEST SCF JK")
    co = np.array(wfn.Ca_subset("AO", "OCC").np, copy=True)
    d = co @ co.T
    s, h = [np.array(m.np, copy=True) for m in (wfn.S(), wfn.H())]
    if orthogonalizer is None:
        available = wfn._diis_orthogonalizer()
        if available is None:
            raise RuntimeError("No captured SCF orthogonalizer for fixed-state residual")
        orthogonalizer = available.np
    x = np.array(orthogonalizer, copy=True)
    nuclear = float(wfn.molecule().nuclear_repulsion_energy())
    context = (h, s, x, nuclear)
    np.testing.assert_allclose(co.T @ s @ co, np.eye(co.shape[1]), atol=1e-10, rtol=1e-10)
    arrays, rows = {}, []
    # These are the last evaluated inputs/outputs, NOT assumed synchronized
    # with the final orbitals returned after the last DIIS update.
    old_d, old_j, old_k = [np.array(v[0].np, copy=True)
                         for v in (original.D(), original.J(), original.K())]
    np.savez(output / "last-evaluated-jk.npz", D=old_d, J=old_j, K=old_k)
    original.set_print(3)  # Emits a creation count if compute silently rebuilds.
    psi4.core.print_out("\nFIXED_JK retained SCF plan (no reinitialization)\n")
    original.print_header()
    baseline = fixed_samples(psi4.core, original, co, d, "retained", arrays, rows, context=context)
    np.savez(output / "fixed-input.npz", Cocc=co, D=d, S=s, H=h, X=x,
             J=baseline[0], K=baseline[1])
    report = dict(status="RUNNING", state=state, samples=rows,
                  post_scf_options=post_options, nuclear_Eh=nuclear,
                  reported_scf_energy_Eh=float(wfn.energy()),
                  last_evaluated_state=rebuilt_state(old_d, h, s, x, old_j, old_k, nuclear),
                  last_evaluated_D_vs_final_C_D_max_abs=float(np.max(np.abs(old_d - d))),
                  last_evaluated_sha256=digest(output / "last-evaluated-jk.npz"),
                  input_sha256=digest(output / "fixed-input.npz"),
                  primary=basis_receipt(primary), auxiliary=basis_receipt(auxiliary),
                  scope="Reconstructed D=Cocc*Cocc.T, not the original SCF energy boundary")
    save(output / "fixed-jk.json", report)
    for plan in range(2):
        jk = fresh_plan(psi4.core, primary, auxiliary, state, "gpu")
        initialized = False
        try:
            psi4.core.print_out(f"\nFIXED_JK fresh same-object plan {plan}\n")
            jk.initialize()
            initialized = True
            jk.print_header()  # Query actual initialized mode.
            fixed_samples(psi4.core, jk, co, d, f"fresh{plan}", arrays, rows, baseline, context)
        finally:
            if initialized:
                jk.finalize()
        save(output / "fixed-jk.json", report)
    # Revisit the original plan after creating/destroying fresh plans.
    psi4.core.print_out("\nFIXED_JK retained SCF plan after fresh plans\n")
    fixed_samples(psi4.core, original, co, d, "retained_after", arrays, rows, baseline, context)
    np.savez(output / "fixed-jk.npz", **arrays)
    report.update(status=STATUS, matrices_sha256=digest(output / "fixed-jk.npz"))
    psi4.core.flush_outfile()
    text = (output / "psi4.out").read_text()
    sections = text.split("FIXED_JK retained SCF plan")
    if len(sections) != 3:
        raise RuntimeError("Missing retained-plan provenance markers")
    first_section = sections[1].split("FIXED_JK fresh same-object plan", 1)[0]
    report["retained_plan_rebuilt"] = any(
        "cuESTJK plan created" in part for part in (first_section, sections[2]))
    save(output / "fixed-jk.json", report)
    return report


def run(psi4, inputs, output, backend, reference=None, reference_sha=None):
    import numpy as np
    if reference is None:
        if backend != "gpu":
            raise ValueError("Retained-plan producer must be GPU")
        from small_dimer_probe import run as scf_run
        initializations = []
        captured = {}
        original_init = psi4.core.HF.initialize_jk
        def observed_init(wfn, *args, **kwargs):
            value = original_init(wfn, *args, **kwargs)
            initializations.append(dict(options=option_receipt(psi4.core),
                                        state=jk_state(wfn.jk()), nbf=wfn.basisset().nbf()))
            return value
        def post_scf(psi4, wfn, output):
            return retained_and_fresh(psi4, wfn, output, captured.get("X"))
        with patch.object(psi4.core.HF, "initialize_jk", observed_init), \
                capture_orthogonalizer(psi4.core, captured):
            report = scf_run(psi4, inputs, output, backend, "energy", "peptide",
                             post_scf=post_scf)
        report["initializations"] = initializations
        report["initialization_observer"] = "Post-initialize scalar receipts only; no iteration observer"
        report.update(scf_status=report["status"], status=STATUS)
        save(output / "result.json", report)
        return report
    if digest(inputs) != INPUT_SHA or digest(reference) != reference_sha:
        raise RuntimeError("Replay inputs/producer receipt changed")
    producer = json.loads(reference.read_text())
    proof = producer["fixed_jk"]
    if producer["status"] != STATUS or proof["status"] != STATUS:
        raise RuntimeError("Incomplete retained-plan producer")
    fixed_input = reference.parent / "fixed-input.npz"
    if digest(fixed_input) != proof["input_sha256"]:
        raise RuntimeError("Fixed coefficient artifact changed")
    with np.load(fixed_input, allow_pickle=False) as data:
        co, d, s, j0, k0 = [np.array(data[k], copy=True) for k in ("Cocc", "D", "S", "J", "K")]
        h, x = [np.array(data[k], copy=True) for k in ("H", "X")]
    if not all(np.isfinite(x).all() for x in (co, d, s, j0, k0)):
        raise RuntimeError("Nonfinite fixed input")
    density_check = density_reconstruction(co, d)
    np.testing.assert_allclose(co.T @ s @ co, np.eye(co.shape[1]), atol=1e-10, rtol=1e-10)
    output.mkdir(parents=True, exist_ok=False)
    psi4.core.set_output_file(str(output / "psi4.out"), False)
    psi4.set_memory("6 GiB")
    psi4.set_num_threads(4)
    psi4.set_options({**producer["options"], "use_cuest": backend == "gpu"})
    mol = psi4.geometry(producer["geometry"])
    mol.update_geometry()
    primary = psi4.core.BasisSet.build(mol, "BASIS", "jun-cc-pvdz", puream=1)
    auxiliary = psi4.core.BasisSet.build(
        mol, "DF_BASIS_SCF", "aug-cc-pvdz-jkfit", "JKFIT", "jun-cc-pvdz", puream=1)
    if (basis_receipt(primary) != proof["primary"]
            or basis_receipt(auxiliary) != proof["auxiliary"]):
        raise RuntimeError("Rebuilt basis differs from actual SCF basis fingerprint")
    rows, arrays = [], {}
    report = dict(status="RUNNING", backend=backend, scf_executed=False,
                  reference_sha256=reference_sha, input_sha256=proof["input_sha256"],
                  host_density_reconstruction=density_check,
                  samples=rows, jk_name="cuESTJK" if backend == "gpu" else "MemDFJK",
                  variants=replay_variants(proof["state"], backend))
    try:
        for plan, state in enumerate(report["variants"]):
            jk = fresh_plan(psi4.core, primary, auxiliary, state, backend)
            initialized = False
            try:
                label = f"beta{state['beta']:g}_plan{plan}"
                psi4.core.print_out(f"\nFIXED_JK independent {backend} {label}\n")
                jk.initialize()
                initialized = True
                jk.print_header()
                fixed_samples(psi4.core, jk, co, d, label, arrays, rows,
                              (j0, k0), (h, s, x, proof["nuclear_Eh"]))
            finally:
                if initialized:
                    jk.finalize()
            save(output / "result.json", report)
        np.savez(output / "fixed-jk.npz", **arrays)
        report.update(status=STATUS, matrices_sha256=digest(output / "fixed-jk.npz"))
    except BaseException as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        psi4.core.flush_outfile()
        save(output / "result.json", report)
    return report
