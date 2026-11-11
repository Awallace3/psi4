"""Fixed CPU-dimer coefficient controls, with no SCF or SAPT execution."""
import json
import os
from pathlib import Path
import time

from dimer_energy_probe import INPUT_SHA, scf_options
from phoenix_timing import digest, save


def delta_summary(j0, k0, j1, k1, density):
    import numpy as np
    dj, dk = j1 - j0, k1 - k0
    # Contract differences directly, rather than subtracting large totals.
    # Extended-precision host accumulation helps expose, rather than add,
    # cancellation noise in this diagnostic.
    contraction = lambda x: np.sum(
        density.astype(np.longdouble) * x.T.astype(np.longdouble), dtype=np.longdouble)
    ej, ek = 2 * contraction(dj), -contraction(dk)
    return {
        "J_bitwise_equal": j0.tobytes(order="C") == j1.tobytes(order="C"),
        "K_bitwise_equal": k0.tobytes(order="C") == k1.tobytes(order="C"),
        "J_max_abs": float(np.max(np.abs(dj))),
        "K_max_abs": float(np.max(np.abs(dk))),
        "J_rms": float(np.sqrt(np.mean(dj * dj))),
        "K_rms": float(np.sqrt(np.mean(dk * dk))),
        "two_Tr_D_deltaJ_Eh": float(ej), "minus_Tr_D_deltaK_Eh": float(ek),
        "closed_shell_JK_delta_Eh": float(ej + ek),
    }


def reference(path, expected_sha):
    if digest(path) != expected_sha:
        raise RuntimeError("Frozen CPU reference result changed")
    doc = json.loads(path.read_text())
    if (doc["status"] != "CONVERGED_DIAGNOSTIC_ONLY" or doc["backend"] != "cpu"
            or doc["options"] != scf_options("cpu") or doc["input_sha256"] != INPUT_SHA
            or doc["nbf"] != 922 or doc["job_id"] != "15482205"):
        raise RuntimeError("Exact converged protein83 CPU dimer required")
    snapshot = path.parent / "dimer-wavefunction.npz"
    if digest(snapshot) != doc["snapshot_sha256"]:
        raise RuntimeError("Frozen full-dimer coefficients changed")
    return doc, snapshot


def run(psi4, inputs_path, output, backend, threads, reference_path, reference_sha):
    import numpy as np
    from scf_repeat_probe import sample_plan, PLANS, CALLS

    doc, snapshot = reference(reference_path, reference_sha)
    if digest(inputs_path) != INPUT_SHA:
        raise RuntimeError("Frozen input manifest changed")
    inputs = json.loads(inputs_path.read_text())
    point = inputs["systems"]["protein83"]
    if point["geometry"] != doc["geometry"]:
        raise RuntimeError("Reference geometry differs from frozen input")
    for item in inputs["basis_files"].values():
        if digest(Path(psi4.core.get_datadir()) / "basis" / item["filename"]) != item["sha256"]:
            raise RuntimeError("Installed basis changed")
    if backend == "cpu" and (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("SLURM_JOB_GPUS")
    ):
        raise RuntimeError("CPU fixed-density control must allocate no GPU")
    output.mkdir(parents=True, exist_ok=False)
    psi4.core.set_output_file(str(output / "psi4.out"), False)
    psi4.set_memory("96 GiB")
    psi4.set_num_threads(threads)
    psi4.set_options(scf_options(backend))
    molecule = psi4.geometry(doc["geometry"])
    molecule.update_geometry()
    primary = psi4.core.BasisSet.build(molecule, "BASIS", "jun-cc-pvdz", puream=1)
    auxiliary = psi4.core.BasisSet.build(
        molecule, "DF_BASIS_SCF", "aug-cc-pvdz-jkfit", "JKFIT", "jun-cc-pvdz", puream=1)
    if (primary.nbf(), auxiliary.nbf()) != (922, 4762):
        raise RuntimeError("Unexpected primary/JKFIT dimensions")
    with np.load(snapshot, allow_pickle=False) as data:
        co, density, s, h = [np.array(data[k], copy=True) for k in ("Cocc", "D", "S", "H")]
    np.testing.assert_allclose(co.T @ s @ co, np.eye(co.shape[1]), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(density, co @ co.T, atol=1e-12, rtol=1e-12)
    stored = dict(Cocc=co, density=density, S=s, H=h)
    report = dict(
        status="STARTED", scope=__doc__, backend=backend, threads=threads,
        scf_executed=False, sapt_executed=False, accepted_speedup_count=0,
        reference_result_sha256=reference_sha, snapshot_sha256=doc["snapshot_sha256"],
        source_pin=os.environ["DISP0_PIN"], options=scf_options(backend),
        pid=os.getpid(), job_id=os.environ.get("SLURM_JOB_ID"),
        psi4_file=psi4.__file__, core_file=psi4.core.__file__,
        plans=PLANS, calls_per_plan=CALLS, samples=[],
        difference_direction="current_minus_baseline",
        fixed_density_source="converged_CPU_dimer_Da",
        longdouble_mantissa_bits=int(np.finfo(np.longdouble).nmant),
    )
    started = time.perf_counter()
    first = None
    try:
        for plan in range(PLANS):
            samples = sample_plan(psi4.core, primary, auxiliary, co, gpu=backend == "gpu")
            for call, (j, k, elapsed, actual_d, name) in enumerate(samples):
                if first is None:
                    first = (j, k, actual_d)
                stored[f"J_{plan}_{call}"] = j
                stored[f"K_{plan}_{call}"] = k
                row = dict(plan=plan, call=call, backend_name=name, compute_seconds=elapsed,
                           vs_first=delta_summary(first[0], first[1], j, k, density),
                           vs_plan_first=delta_summary(samples[0][0], samples[0][1], j, k, density),
                           actual_D_vs_first_max_abs=float(np.max(np.abs(actual_d - first[2]))),
                           actual_D_vs_reference_max_abs=float(np.max(np.abs(actual_d - density))),
                           J_asymmetry_max_abs=float(np.max(np.abs(j - j.T))),
                           K_asymmetry_max_abs=float(np.max(np.abs(k - k.T))))
                report["samples"].append(row)
            save(output / "result.json", report)
        stored["actual_D_first"] = first[2]
        np.savez(output / "fixed-jk.npz", **stored)
        report.update(status="EXECUTED_FIXED_INPUT_DIAGNOSTIC_ONLY",
                      matrices_sha256=digest(output / "fixed-jk.npz"))
    except BaseException as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["timers"] = psi4.core.get_timer_dict()
        save(output / "result.json", report)
        psi4.core.flush_outfile()
