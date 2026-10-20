"""One fresh-process HF-E2 suite timing, only after an attested GPU gate.

No oracle, profiling, interposer, geometry edits, or cross-method comparison.
The CPU arm changes placement locally; all other scientific options agree.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import time
import traceback
from unittest.mock import patch

if not __debug__:
    raise RuntimeError("Dispersion validation requires assertions; optimized Python is forbidden")

SYSTEMS = ("peptide", "nanotube", "protein83", "protein157")
NBF = dict(zip(SYSTEMS, (250, 548, 922, 1786)))
SUITE_SHA256 = "72668390d24f134316c8e779931a2f65efcd9289ecea8f1cf20a72733262e3bb"
SCIENCE = {
    "basis": "6-31+g**", "puream": True, "scf_type": "df", "scf_subtype": "AUTO",
    "reference": "rhf", "freeze_core": False, "DF_BASIS_SCF": "def2-universal-jkfit",
    "DF_BASIS_MP2": "aug-cc-pvdz-ri", "D_CONVERGENCE": 8, "E_CONVERGENCE": 9,
    "MAXITER": 150, "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL",
    "SAPT_DFT_FUNCTIONAL": "HF", "SAPT_DFT_DO_DHF": True,
    "SAPT_DFT_INDUCTION_TYPE": "CPHF", "SAPT_DFT_USE_EINSUMS": True,
    "SAPT_DFT_MP2_DISP_ALG": "EINSUMS", "SAPT_DFT_EINSUMS_DISP_BLOCK": 24,
}
COMPONENTS = tuple(f"SAPT {label} ENERGY" for label in
                   ("ELST", "EXCH", "IND", "DISP", "TOTAL")) + ("Disp20,u", "Exch-Disp20,u")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def require_gate(gate, build):
    """Reject stale/partial/different-runtime proofs before importing science."""
    assert gate["status"] == "PASS" and not gate["failures"], "GPU numerical gate did not pass"
    assert gate["comparisons"] and all(row["pass"] for row in gate["comparisons"])
    assert gate["native_cublas_trace_evidence"]["actual_production_cublas_calls"] > 0
    assert gate["profile_dispatch_evidence"]["observed_cpu_edges"]
    logger = gate["native_logger_preflight"]
    assert logger["cpu"]["probe"]["dgemm_mentions"] == 0
    assert logger["cpu"]["policy_device"] == "cpu" and logger["gpu"]["policy_device"] == "cuda"
    assert all(logger[arm]["status"] == "PASS" for arm in ("cpu", "gpu"))
    assert all(logger[arm]["positive_control"]["dgemm_mentions"] > 0 for arm in ("cpu", "gpu"))
    assert gate["science_build"]["cpu_preparation_job"] == "14196330"
    assert gate["science_build"]["pin"] == "587e172a33afaadb00eb7f574a5089533f7fc755"
    binaries = gate["loaded_binary_sha256"]
    assert len(binaries) == 3
    for name, checksum in binaries.items():
        path = Path(name).resolve()
        assert path.is_relative_to(build)
        assert digest(path) == checksum, f"Changed gate-tested binary: {path}"
    modules = gate["imported_scientific_modules"]
    expected_names = {"sapt_jk_terms_ein.py", "sapt_mp2_terms_ein.py",
                      "sapt_dispersion_gpu.py", "sapt_proc.py"}
    assert len(modules) == 4 and {Path(name).name for name in modules} == expected_names
    for name, checksum in modules.items():
        path = Path(name).resolve()
        assert path.parent == build / "psi4-build/stage/lib/psi4/driver/procrouting/sapt"
        assert digest(path) == checksum, f"Changed gate-tested science: {path}"
    assert gate["cuda_identity"]["header_version"] == gate["cuda_identity"]["runtime_version"] == 13030


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--arm", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--repeat", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--suite", type=Path, required=True, help="read-only geometries directory")
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--gate-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.gate_result = args.gate_result.resolve()
    runner_path = Path(__file__).resolve()
    assert not os.environ.get("LD_PRELOAD"), "Do not time with a tracing interposer"
    assert not os.environ.get("DISP0_CUBLAS_TRACE"), "Diagnostic trace environment is forbidden"
    assert not any(key.startswith("CUBLAS_LOG") for key in os.environ), \
        "Native logger must remain default-OFF throughout the fresh timed call"
    build = args.build.resolve()
    gate = json.loads(args.gate_result.read_text())
    require_gate(gate, build)
    suite_file = args.suite.resolve() / "systems.py"
    assert digest(suite_file) == SUITE_SHA256, "Fixed suite changed"
    spec = importlib.util.spec_from_file_location("disp0_fixed_suite", suite_file)
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    geometry = suite.SYSTEMS[args.system]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(out)  # fresh timer.dat, no source-package shadowing
    record = {
        "status": "RUNNING", "method": "SAPT(DFT)/HF canonical EINSUMS", "arm": args.arm,
        "system": args.system, "repeat": args.repeat, "threads": 8, "memory": "128 GiB",
        "suite_sha256": SUITE_SHA256, "geometry_sha256": hashlib.sha256(geometry.encode()).hexdigest(),
        "runner_sha256": digest(runner_path), "gate_result_sha256": digest(args.gate_result),
        "science_options": SCIENCE, "cpu_graph_placement_override": args.arm == "cpu",
        "scope": "policy-controlled 8-CPU versus hybrid timing; timed dispatch is not traced",
        "actual_timed_gpu_dispatch_verified": False,
        "hostname": os.uname().nodename, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "kernel_only_seconds": None,
        "protocol": "24-fresh-timed-full-calls-plus-bounded-post-timing-synthetic-probes",
        "planned_process_work": {"timed_full_calls": 1, "untimed_full_calls": 0,
                                 "small_synthetic_kernel_probes": 1, "separate_positive_controls": 1},
    }
    try:
        from benchmark_evidence import NativeLogger, cuda_identity, check_context, synthetic_policy_probe
        import benchmark_resources

        record["resource_preflight"] = benchmark_resources.check(args.system)
        import psi4  # core first, then the same native/Python Einsums world
        import einsums as ein
        import einsums.rc
        import einsums.sealed as sealed
        assert not any(arg.startswith("--einsums:profile:") for arg in ein.cli_options)
        if einsums.rc.profile_disable is None:
            einsums.rc.profile_disable = True
        if einsums.rc.profile_report is None:
            einsums.rc.profile_report = False
        assert einsums.rc.profile_disable is True and einsums.rc.profile_report is False
        from psi4.driver.procrouting.sapt import sapt_jk_terms_ein as terms
        from psi4.driver.procrouting.sapt import sapt_dispersion_gpu as gpu
        actual = {str(Path(p).resolve()) for p in
                  (psi4.core.__file__, ein._core.__file__, sealed.world()["library_path"])}
        assert actual == set(gate["loaded_binary_sha256"]), "Timing imported another native world"
        assert {str(Path(p).resolve()) for p in sealed.mapped_einsums_libraries()} == {
            str(Path(sealed.world()["library_path"]).resolve())}
        for path in gate["imported_scientific_modules"]:
            name = "psi4.driver.procrouting.sapt." + Path(path).stem
            module = __import__(name, fromlist=["__file__"])
            assert str(Path(module.__file__).resolve()) == path
        assert not any("disp0_cublas" in line for line in Path("/proc/self/maps").read_text().splitlines())
        record["native_world"] = sealed.world()
        record["loaded_binary_sha256"] = gate["loaded_binary_sha256"]
        record["imported_scientific_modules"] = gate["imported_scientific_modules"]
        record["library_maps"] = sorted({line.split()[-1] for line in
                                         Path("/proc/self/maps").read_text().splitlines()
                                         if any(token in line for token in
                                                ("libcublas", "libcudart", "libEinsums", "libmkl"))})
        psi4.set_memory(record["memory"])
        psi4.set_num_threads(record["threads"])
        psi4.core.set_output_file(str(out / "psi4.out"), False)
        psi4.set_options({**SCIENCE, "SAPT_DFT_EINSUMS_GPU": args.arm == "gpu"})
        molecule = psi4.geometry(geometry)
        molecule.reset_point_group("c1")
        molecule.update_geometry()
        basis = psi4.core.BasisSet.build(molecule, "BASIS", SCIENCE["basis"])
        assert basis.nbf() == NBF[args.system]
        record["nbf"] = basis.nbf()
        record["naux"] = {
            role: psi4.core.BasisSet.build(molecule, role, value, kind, SCIENCE["basis"]).nbf()
            for role, value, kind in (
                ("DF_BASIS_SCF", SCIENCE["DF_BASIS_SCF"], "JKFIT"),
                ("DF_BASIS_MP2", SCIENCE["DF_BASIS_MP2"], "RIFIT"))}
        original = terms.dispersion0
        dispersion = []

        def measured_dispersion(*a, **kw):
            if args.arm == "cpu":
                assert not kw.get("gpu", False)
                kw["_graph_device"] = "cpu"  # local default graph, never global executor mutation
            started = time.perf_counter()
            value = original(*a, **kw)
            if args.arm == "gpu":
                ein.gpu.device_synchronize()
            dispersion.append({"end_to_end_seconds": time.perf_counter() - started,
                               "signed_Eh": {key: float(value[key]) for key in
                                             ("Disp20,u", "Exch-Disp20,u")},
                               "execution": value["Disp0 execution"]})
            return value

        psi4.core.clean_timers()
        # Do not retain adapters/cache tensors: normal destruction is part of
        # production whole-call/dispersion time. Capture only scalar metrics.
        with patch.object(terms, "dispersion0", measured_dispersion):
            start = time.perf_counter()
            energy = psi4.energy("sapt(dft)", molecule=molecule)
            if args.arm == "gpu":
                ein.gpu.device_synchronize()
            record["wall_s"] = time.perf_counter() - start
        record["timing_process_peak_rss_KiB_before_untimed_proof"] = \
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record["timer_records"] = psi4.core.get_timer_records()
        record["returned_energy_Eh"] = float(energy)
        record["components_Eh"] = {key: float(psi4.variable(key)) for key in COMPONENTS}
        record["dispersion"] = list(dispersion)
        assert len(dispersion) == 1
        components = record["components_Eh"]
        assert abs(sum(components[f"SAPT {key} ENERGY"] for key in ("ELST", "EXCH", "IND", "DISP"))
                   - components["SAPT TOTAL ENERGY"]) < 1e-8
        assert abs(float(energy) - components["SAPT TOTAL ENERGY"]) < 1e-8
        assert abs(sum(dispersion[0]["signed_Eh"].values()) - components["SAPT DISP ENERGY"]) < 1e-8
        # Preserve completed measurements even if later diagnostic work is
        # interrupted. This checkpoint is NOT an accepted timing.
        record.update(status="MEASURED_PROOF_PENDING", measurement_complete=True,
                      untimed_proof_complete=False, completed_timed_full_calls=1)
        save(out / "result.json", record)
        # No synthetic GPU warmup, CUDA identity call, or forced GPU sync
        # before timing. First-use initialization stays in the production call.
        # Native logging is default-OFF in the fresh process (env checked above).
        record["runtime"] = gpu.gpu_runtime()
        logger = NativeLogger()
        record["cuda_identity"] = cuda_identity()

        # Fresh tiny synthetic state only. No timed tensor/graph/cache/DF
        # scratch is retained or reused, and no second full HF call is made.
        assert SCIENCE["freeze_core"] is False
        if args.arm == "cpu":
            assert dispersion[0]["execution"]["planned_gpu_contractions"] == 0
        diagnostic_out = out / "post-timing-probe"
        diagnostic_out.mkdir()
        policy = {"compute_block_extents": dispersion[0]["execution"]["compute_block_extents"],
                  "occupied_A": suite.FRAGMENTS[args.system]["A"]["electrons"] // 2,
                  "occupied_B": suite.FRAGMENTS[args.system]["B"]["electrons"] // 2,
                  "naux": record["naux"]["DF_BASIS_MP2"]}
        record["untimed_diagnostic_dispatch"] = synthetic_policy_probe(
            logger, args.arm, diagnostic_out, ein, gpu, policy)
        record["exclusive_cuda_context"] = check_context(record["cuda_identity"])
        record["whole_process_peak_rss_KiB_including_untimed_proof"] = \
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record["untimed_proof_complete"] = True
        record["status"] = "PASS"
    except BaseException:
        record["status"] = "FAIL"
        record["exception"] = traceback.format_exc()
        raise
    finally:
        save(out / "result.json", record)


if __name__ == "__main__":
    main()
