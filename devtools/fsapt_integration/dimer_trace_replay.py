"""Replay identical GPU-SCF snapshots through native MemDFJK and cuESTJK."""
import json
import os
from pathlib import Path
import time

from dimer_energy_probe import INPUT_SHA, scf_options
from dimer_fixed_jk import delta_summary
from dimer_scf_trace import contraction
from phoenix_timing import digest, save

ITERATIONS = (20, 25, 40)


def validate_reference(path, result_sha, trace_sha):
    if digest(path) != result_sha:
        raise RuntimeError("GPU trajectory result changed")
    doc = json.loads(path.read_text())
    trace_path = path.parent / "energy-trace.json"
    if digest(trace_path) != trace_sha:
        raise RuntimeError("GPU trajectory trace changed")
    trace = json.loads(trace_path.read_text())
    if (doc["status"] != "CONVERGED_DIAGNOSTIC_ONLY" or doc["backend"] != "gpu"
            or doc["job_id"] != "15503335" or doc["nbf"] != 922
            or doc["options"] != scf_options("gpu") or doc["input_sha256"] != INPUT_SHA
            or not trace["native_method_restored"] or trace["extra_JK_calls"] != 0):
        raise RuntimeError("Exact original GPU trajectory required")
    for iteration in ITERATIONS:
        name = f"iteration-{iteration:03d}.npz"
        if name not in trace["snapshots"]:
            raise RuntimeError("Missing selected trajectory snapshot")
    return doc, trace


def build_plan(core, primary, auxiliary, backend):
    """Supply memory BEFORE factory selection; refuse a DiskDFJK substitution."""
    if backend not in ("cpu", "gpu"):
        raise ValueError("Explicit backend required")
    memory = int(core.get_memory() * 0.75 / 8)  # doubles; 72 GiB of a 96 GiB setting
    jk = core.JK.build(primary, auxiliary, jk_type="DF", memory=memory)
    expected = "MemDFJK" if backend == "cpu" else "cuESTJK"
    if jk.name() != expected:
        raise RuntimeError(f"Expected {expected}, got {jk.name()}")
    jk.set_memory(memory)
    jk.set_omega_alpha(1.)
    jk.set_omega_beta(0.)
    jk.set_do_J(True)
    jk.set_do_K(True)
    return jk, memory


def run(psi4, inputs_path, output, backend, threads, reference_path, reference_sha, trace_sha):
    import numpy as np
    doc, trace = validate_reference(reference_path, reference_sha, trace_sha)
    if digest(inputs_path) != INPUT_SHA:
        raise RuntimeError("Frozen input manifest changed")
    inputs = json.loads(inputs_path.read_text())
    if inputs["systems"]["protein83"]["geometry"] != doc["geometry"]:
        raise RuntimeError("Trajectory geometry differs")
    for item in inputs["basis_files"].values():
        if digest(Path(psi4.core.get_datadir()) / "basis" / item["filename"]) != item["sha256"]:
            raise RuntimeError("Installed basis changed")
    if backend == "cpu" and (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("SLURM_JOB_GPUS")
    ):
        raise RuntimeError("CPU replay must allocate no GPU")
    frames = {}
    for iteration in ITERATIONS:
        name = f"iteration-{iteration:03d}.npz"
        path = reference_path.parent / name
        if digest(path) != trace["snapshots"][name]:
            raise RuntimeError("Trajectory snapshot changed")
        with np.load(path, allow_pickle=False) as data:
            frame = {key: np.array(data[key], copy=True) for key in ("Cocc", "D", "S", "H", "J", "K")}
        if not all(np.isfinite(value).all() for value in frame.values()):
            raise RuntimeError("Nonfinite reference arrays")
        co, d, s = [frame[key] for key in ("Cocc", "D", "S")]
        if d.shape != (922, 922) or co.shape != (922, 146):
            raise RuntimeError("Unexpected reference dimensions")
        np.testing.assert_allclose(co.T @ s @ co, np.eye(146), atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(d, co @ co.T, atol=1e-12, rtol=1e-12)
        frames[iteration] = frame
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
        raise RuntimeError("Replay basis dimensions differ")
    report = dict(status="STARTED", scope=__doc__, backend=backend,
                  source_pin=os.environ["DISP0_PIN"], threads=threads,
                  reference_result_sha256=reference_sha, trace_sha256=trace_sha,
                  snapshot_sha256={str(i): trace["snapshots"][f"iteration-{i:03d}.npz"] for i in ITERATIONS},
                  options=scf_options(backend), scf_executed=False, sapt_executed=False,
                  accepted_speedup_count=0, samples=[], reciprocity=[],
                  job_id=os.environ.get("SLURM_JOB_ID"), pid=os.getpid(),
                  difference_direction="replay_minus_original_GPU_trace")
    arrays = {}
    started = time.perf_counter()
    first_overall = {}
    try:
        for plan, order in enumerate((ITERATIONS * 2, tuple(reversed(ITERATIONS)) * 2)):
            jk, memory = build_plan(psi4.core, primary, auxiliary, backend)
            report["factory_memory_doubles"] = memory
            initialized = False
            first = {}
            try:
                jk.print_header()
                jk.initialize()
                initialized = True
                for call, iteration in enumerate(order):
                    frame = frames[iteration]
                    co, d = frame["Cocc"], frame["D"]
                    coefficient = psi4.core.Matrix.from_array(co.copy())
                    jk.C_clear()
                    jk.C_left_add(coefficient)
                    jk.compute()
                    if coefficient.np.tobytes() != co.tobytes():
                        raise RuntimeError("Native JK changed input coefficients")
                    j, k, actual_d = [np.array(v[0].np, copy=True) for v in (jk.J(), jk.K(), jk.D())]
                    if not all(v.shape == (922, 922) and np.isfinite(v).all() for v in (j, k, actual_d)):
                        raise RuntimeError("Incomplete native replay")
                    np.testing.assert_allclose(actual_d, d, atol=1e-12, rtol=1e-12)
                    first.setdefault(iteration, (j, k))
                    first_overall.setdefault(iteration, (j, k))
                    label = f"p{plan}_c{call}_i{iteration}"
                    arrays["J_" + label], arrays["K_" + label] = j, k
                    arrays["actual_D_" + label] = actual_d
                    row = dict(plan=plan, call=call, iteration=iteration, array_label=label,
                               backend_name=jk.name(), actual_D_vs_snapshot_max_abs=float(np.max(np.abs(actual_d-d))),
                               vs_original_trace=delta_summary(frame["J"], frame["K"], j, k, d),
                               vs_plan_first_same_input=delta_summary(*first[iteration], j, k, d),
                               vs_first_plan_same_input=delta_summary(*first_overall[iteration], j, k, d),
                               replay_coulomb_Eh=float(2 * contraction(d, j)),
                               replay_exchange_Eh=float(-contraction(d, k)),
                               J_asymmetry_max_abs=float(np.max(np.abs(j-j.T))))
                    report["samples"].append(row)
                    save(output / "result.json", report)
                # DF Coulomb should represent a self-adjoint linear map.
                for left, right in ((20, 25), (25, 40), (20, 40)):
                    defect = (contraction(frames[left]["D"], first[right][0])
                              - contraction(frames[right]["D"], first[left][0]))
                    report["reciprocity"].append(dict(
                        plan=plan, left_iteration=left, right_iteration=right,
                        Tr_Dleft_Jright_minus_Tr_Dright_Jleft_Eh=float(defect)))
            finally:
                if initialized:
                    jk.finalize()
        np.savez(output / "replay-jk.npz", **arrays)
        report.update(status="EXECUTED_TRACE_REPLAY_DIAGNOSTIC_ONLY",
                      matrices_sha256=digest(output / "replay-jk.npz"))
    except BaseException as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        save(output / "result.json", report)
        psi4.core.flush_outfile()
