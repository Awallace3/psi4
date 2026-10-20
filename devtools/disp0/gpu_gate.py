"""Authorized scalar-dispersion GPU correctness gate, never a scaling benchmark.

Import only the immutable scientific stage; this standalone validation harness
does not overlay Python science on an older native core. CPU distributions are
oracles, not claims of implemented GPU F-SAPT distributions.
"""

import argparse
import ctypes
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback
from unittest.mock import patch

import psi4
from psi4 import core
import einsums as ein
import einsums.profile as prof
import einsums.sealed as sealed
import numpy as np

terms = importlib.import_module("psi4.driver.procrouting.sapt.sapt_jk_terms_ein")
mp2 = importlib.import_module("psi4.driver.procrouting.sapt.sapt_mp2_terms_ein")
gpu = importlib.import_module("psi4.driver.procrouting.sapt.sapt_dispersion_gpu")

KERNEL_RTOL, KERNEL_ATOL = 1e-11, 1e-12
PHYSICAL_RTOL, PHYSICAL_ATOL = 1e-8, 1e-8  # Eh; declared before interpreting results
LABELS = ("Disp20,u", "Exch-Disp20,u")

WATER_NE = """
0 1
O 0.0 0.0 0.0
H 0.0 0.0 0.96
H 0.91 0.0 -0.24
--
0 1
Ne 2.8 1.1 1.5
symmetry c1
no_reorient
no_com
"""
NE_WATER = """
0 1
Ne 2.8 1.1 1.5
--
0 1
O 0.0 0.0 0.0
H 0.0 0.0 0.96
H 0.91 0.0 -0.24
symmetry c1
no_reorient
no_com
"""
ETHANE_WATER = """
0 1
C 0.000 0.000 0.000
H 1.090 0.000 0.000
C -0.770 1.330 0.000
H -1.860 1.330 0.000
H -0.390 1.870 0.890
H -0.390 1.870 -0.890
H -0.390 -0.540 0.890
H -0.390 -0.540 -0.890
--
0 1
O 0.200 -0.300 3.400
H 0.900 -0.100 4.000
H -0.500 -0.700 3.950
symmetry c1
no_reorient
no_com
"""


def benzene_dimer():
    fragments = []
    for shift, z in ((0.0, 0.0), (0.4, 3.7)):
        atoms = ["0 1"]
        for atom, radius in (("C", 1.39), ("H", 2.48)):
            for i in range(6):
                angle = i * math.pi / 3
                atoms.append(f"{atom} {shift + radius * math.cos(angle):.12f} "
                             f"{radius * math.sin(angle):.12f} {z:.12f}")
        fragments.append("\n".join(atoms))
    return "\n--\n".join(fragments) + "\nsymmetry c1\nno_reorient\nno_com\n"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def trace_evidence(path, pid):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    physical = [r for r in records if r.get("phase", "").startswith("physical/")]
    assert physical, "No actual cuBLAS calls occurred in physical GPU dispersion"
    assert all(r["api"] == "cublasDgemm_v2" and r["pid"] == pid and r["status"] == 0
               and min(r["m"], r["n"], r["k"]) > 0 for r in records), records
    production = [r for r in physical if r["phase"].endswith("/production")]
    assert production, "No actual cuBLAS call in the unchanged production wrapper"
    return {"actual_physical_cublas_calls": len(physical),
            "actual_production_cublas_calls": len(production),
            "all_calls": len(records), "trace_sha256": digest(path)}


def profile_evidence(path, *, require_cpu_edge=False):
    data = json.loads(path.read_text())
    evidence = []
    cpu_edges = []

    def visit(value, ancestors=()):
        if isinstance(value, dict):
            names = (*ancestors, value.get("name", ""))
            annotations = value.get("annotations", {})
            dispatch = annotations.get("gpu_dispatch")
            in_products_replay = "ComputeGraph::execute(sapt0-dispersion-products)" in names
            if dispatch and in_products_replay and any(name.startswith("physical/") for name in names):
                evidence.append({"path": list(names), "dispatch": dispatch})
            if (any(name.startswith("physical/") and name.endswith("/edge-1") for name in names)
                    and in_products_replay
                    and annotations.get("op_kind", "").lower() == "einsum"
                    and (annotations.get("device") == "CPU" or dispatch == "cpu_fallback")):
                cpu_edges.append({"path": list(names), "annotations": annotations})
            for child in value.get("children", []):
                visit(child, names)
            if "children" not in value:
                for child in value.values():
                    if isinstance(child, (dict, list)):
                        visit(child, names)
        elif isinstance(value, list):
            for child in value:
                visit(child, ancestors)

    visit(data)
    assert any(r["dispatch"] == "gemm" for r in evidence), "No successful physical GPU graph dispatch"
    if require_cpu_edge:
        assert cpu_edges, "No observed CPU execution/fallback in physical edge-1 replay"
    # Aggregated profile annotations are evidence of observed dispatch, not an
    # exact kernel count. Native API trace supplies actual invocation counts.
    return {"gpu_dispatches": evidence, "observed_cpu_edges": cpu_edges}


def cuda_identity():
    library = ctypes.CDLL(os.environ["DISP0_TRACE_LIB"])
    function = library.disp0_cuda_identity
    function.argtypes = (ctypes.c_char_p, ctypes.c_size_t)
    function.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(8192)
    status = function(buffer, len(buffer))
    result = json.loads(buffer.value.decode())
    assert status == 0 and result["status"] == 0, result
    assert result["count"] == 1 and result["logical_device"] == 0, result
    assert "H200" in result["name"], result
    assert result["header_version"] == result["runtime_version"] == 13030, result
    return result


def forwarding_identity():
    library = ctypes.CDLL(os.environ["DISP0_TRACE_LIB"])
    function = library.disp0_cublas_forwarding_identity
    function.argtypes = (ctypes.c_char_p, ctypes.c_size_t)
    function.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(8192)
    assert function(buffer, len(buffer)) == 0, buffer.value
    result = json.loads(buffer.value)
    assert result["status"] == 0 and "libcublas.so.13" in Path(result["library"]).resolve().name
    return result


def mapped_paths():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            if fields[5].endswith(" (deleted)"):
                continue  # anonymous/JIT mappings are irrelevant; required binaries must match below
            paths.add(Path(fields[5]).resolve())
    return paths


def nvml_identity(identity, require_pid=False):
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"], text=True)
    assert any(row.split(",")[0].strip() == identity["uuid"] for row in hardware.splitlines()), hardware
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
         "--format=csv,noheader,nounits"], text=True)
    on_device = [row.split(",") for row in processes.splitlines()
                 if len(row.split(",")) >= 3 and row.split(",")[1].strip() == identity["uuid"]]
    foreign = [row for row in on_device if row[0].strip() != str(os.getpid())]
    assert not foreign, f"Foreign compute contexts on allocated CUDA UUID: {foreign}"
    if require_pid:
        assert any(row[0].strip() == str(os.getpid()) for row in on_device), processes
    return {"cuda_uuid": identity["uuid"], "matching_process_rows": on_device}


def replay_probe(report):
    """Real CUDA/default replay: strided packs, cache hits and live pool borrows."""
    pool = ein.MemoryPool(128 * 1024**2, "gpu-gate-replay")
    checks = report.setdefault("replay_probe", {"comparisons": []})

    def exercise():
        rng = np.random.default_rng(7219)
        replay = gpu.BlockContractions()
        buffers = {}
        main_phases = []
        for iteration, (nr, ns) in enumerate(((128, 128), (1, 2), (128, 128), (5, 3), (128, 128))):
            na, nb, nq = 15, 15, 1024
            nk = 2 * nq + 2
            key = nr, ns
            if key not in buffers:
                parents = [ein.create_zero_tensor("strided-parent", [nk + 4, width + 3])
                           for width in (nr * na, ns * nb, ns * na, nr * nb)]
                factors = [p[2:2 + nk, 1:1 + width] for p, width in
                           zip(parents, (nr * na, ns * nb, ns * na, nr * nb))]
                outputs = [pool.empty(list(shape), name="replay-output") for shape in
                           ((nr * na, ns * nb), (nr * na, ns * nb), (ns * na, nr * nb))]
                buffers[key] = parents, factors, outputs
            parents, factors, outputs = buffers[key]
            for parent in parents:
                np.asarray(parent)[...] = rng.normal(size=parent.shape)
            for output in outputs:
                np.asarray(output)[...] = 3139.0
            phase = f"synthetic/replay-{iteration}-{nr}x{ns}"
            if key == (128, 128):
                main_phases.append(phase)
            os.environ["DISP0_TRACE_PHASE"] = phase
            with prof.section(phase):
                replay.products(*factors, nq, *outputs)
            ein.gpu.device_synchronize()
            af, fa, bc, cb = [np.asarray(factor) for factor in factors]
            for index, (actual, expected) in enumerate(zip(
                    outputs, (af[:nq].T @ fa[nq:2 * nq], af.T @ fa, bc.T @ cb))):
                passed = bool(np.allclose(np.asarray(actual), expected, rtol=KERNEL_RTOL, atol=KERNEL_ATOL))
                checks["comparisons"].append({"shape": list(key), "product": index, "pass": passed,
                                               "max_abs_delta": float(np.max(np.abs(np.asarray(actual) - expected)))})
                assert passed, checks["comparisons"][-1]
        records = [json.loads(line) for line in Path(os.environ["DISP0_CUBLAS_TRACE"]).read_text().splitlines()
                   if line.strip()]
        counts = {phase: sum(r["phase"] == phase and r["pid"] == os.getpid()
                             and r["status"] == 0 for r in records) for phase in main_phases}
        assert len(counts) == 3 and all(count > 0 for count in counts.values()), \
            f"Repeated main replays did not actually reach cuBLAS: {counts}"
        checks["actual_main_replay_cublas_calls"] = counts
        # Leave the cached replay as the ONLY owner of pool-backed outputs,
        # not the test's buffers/list/loop variables.
        del buffers, outputs, output, actual
        gc.collect()
        try:
            pool.reset()
        except RuntimeError:
            checks["reset_rejected_live_replay_owner"] = True
        else:
            raise AssertionError("Pool reset destroyed storage borrowed by live GPU replay")
        return replay.report(0.0)

    checks["execution"] = exercise()
    gc.collect()
    assert pool.bytes_used == 0, f"GPU replay retained pool borrows: {pool.bytes_used}"
    pool.reset()
    checks["reset_after_release"] = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "scope": "scalar HF GPU contractions; CPU elementwise/reductions",
              "gpu_fragment_distribution_implemented": False, "comparisons": [], "failures": [],
              "tolerances": {"physical_rtol": PHYSICAL_RTOL, "physical_atol_Eh": PHYSICAL_ATOL,
                             "kernel_rtol": KERNEL_RTOL, "kernel_atol": KERNEL_ATOL}}
    result_path = out / "gpu-gate-result.json"
    started = time.perf_counter()

    def checkpoint():
        save(result_path, report)

    def compare(case, mode, actual, reference, **metadata):
        for label in LABELS:
            a, r = float(actual[label]), float(reference[label])
            passed = bool(np.isclose(a, r, rtol=PHYSICAL_RTOL, atol=PHYSICAL_ATOL))
            row = {"case": case, "mode": mode, "label": label, "actual_Eh": a,
                   "reference_Eh": r, "delta_Eh": a - r, "pass": passed, **metadata}
            report["comparisons"].append(row)
            if not passed:
                report["failures"].append(row)
        checkpoint()

    try:
        build = Path(os.environ["DISP0_BUILD_ROOT"]).resolve()
        prep = json.loads((build / "runtime-provenance.json").read_text())
        assert prep["status"] == "PASS"
        assert Path(psi4.__file__).resolve().is_relative_to(build / "psi4-build/stage")
        assert Path(ein.__file__).resolve().is_relative_to(build / "einsums-stage")
        expected_binaries = {Path(p).resolve(): h for p, h in prep["binary_sha256"].items()}
        actual_binaries = {Path(p).resolve() for p in
                           (core.__file__, ein._core.__file__, sealed.world()["library_path"])}
        assert actual_binaries == set(expected_binaries), (actual_binaries, expected_binaries)
        actual_maps = mapped_paths()
        assert actual_binaries <= actual_maps, f"Prepared binaries not actually mapped: {actual_binaries - actual_maps}"
        report["loaded_binary_sha256"] = {}
        for binary in actual_binaries:
            actual_hash = digest(binary)
            assert actual_hash == expected_binaries[binary], f"Changed loaded prepared binary: {binary}"
            report["loaded_binary_sha256"][str(binary)] = actual_hash
        assert {Path(p).resolve() for p in sealed.mapped_einsums_libraries()} == {
            Path(sealed.world()["library_path"]).resolve()}
        science_source = Path(os.environ["DISP0_SCIENCE_SOURCE"]).resolve()
        science_pin = subprocess.check_output(["git", "-C", str(science_source), "rev-parse", "HEAD"], text=True).strip()
        assert science_pin == os.environ["DISP0_BUILD_PIN"]
        report["science_build"] = {"pin": science_pin, "source": str(science_source),
                                   "cpu_preparation_job": os.environ["DISP0_BUILD_JOB_ID"]}
        subprocess.run(["git", "-C", str(science_source), "diff", "--exit-code"], check=True)
        report["imported_scientific_modules"] = {}
        for module in (terms, mp2, gpu, importlib.import_module("psi4.driver.procrouting.sapt.sapt_proc")):
            path = Path(module.__file__).resolve()
            assert path.is_relative_to(build / "psi4-build/stage/lib/psi4") and path.suffix == ".py", path
            expected = science_source / "psi4/driver/procrouting/sapt" / path.name
            assert digest(path) == digest(expected), f"Scientific source overlay/mismatch: {path}"
            report["imported_scientific_modules"][str(path)] = digest(path)
        assert digest(os.environ["DISP0_TRACE_LIB"]) == os.environ["DISP0_TRACE_SHA256"]
        assert prof.available(), "Einsums profiler must be compiled ON"
        assert any(arg.startswith("--einsums:profile:") for arg in ein.cli_options), \
            "Explicit profile CLI flags required; driver otherwise disables profiling"
        report["runtime"] = gpu.gpu_runtime()
        assert sealed.world()["git_commit"] == "d17bb76e4ed3a07fbbc470e6813a5c134dfe0e62"
        identity = cuda_identity()
        report["forwarding_identity"] = forwarding_identity()
        assert Path(report["forwarding_identity"]["library"]).resolve() in mapped_paths()
        report["cuda_identity"] = identity
        report["nvml_before"] = nvml_identity(identity)
        report["environment"] = {k: os.environ.get(k) for k in
                                 ("SLURM_JOB_ID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES")}
        report["device_free_bytes_before"] = ein.gpu.available_device_memory()
        checkpoint()
        replay_probe(report)
        checkpoint()
        core.set_num_threads(4)
        # Still well below the 32G host request; permits a naturally profitable
        # larger benzene block without overriding the placement cost model.
        psi4.set_memory("8 GB")
        core.set_output_file(str(out / "psi4-gate.out"), False)

        cases = [("water-ne", WATER_NE, "cc-pvdz", False, True),
                 ("water-ne-shifted", WATER_NE.replace("2.8 1.1 1.5", "3.1 0.9 1.7"), "cc-pvdz", False, True),
                 ("ne-water-reversed", NE_WATER, "cc-pvdz", False, True),
                 ("ethane-water-cart-fc", ETHANE_WATER, "6-31g*", True, False),
                 ("benzene-dimer-fc", benzene_dimer(), "cc-pvdz", True, True),
                 ("water-ne-repeat", WATER_NE, "cc-pvdz", False, True)]
        production_values = {}
        execution = []
        original_build, original_entry = terms.build_sapt_jk_cache, mp2.df_mp2_einsums_dispersion
        for name, geometry, basis, frozen, puream in cases:
            core.clean()
            core.clean_options()
            captured = {}

            def cache_builder(dimer, a, b, *values, **keywords):
                captured["wfn_a"] = a
                return original_build(dimer, a, b, *values, **keywords)

            def numerical_entry(dimer, cache, nfrozen_a=0, nfrozen_b=0, do_print=True):
                auxiliary = core.BasisSet.build(dimer.molecule(), "DF_BASIS_MP2", "cc-pvdz-ri",
                                                 "RIFIT", core.get_global_option("BASIS"))
                a = captured["wfn_a"]
                os.environ["DISP0_TRACE_PHASE"] = f"cpu-oracle/{name}/native"
                t = time.perf_counter()
                native = mp2.df_mp2_fisapt_dispersion(
                    a, a.basisset(), auxiliary, cache, nfrozen_a, nfrozen_b, False)
                report.setdefault("native_oracle_seconds", {})[name] = time.perf_counter() - t
                for mode in (None, "cpu"):
                    os.environ["DISP0_TRACE_PHASE"] = f"cpu-oracle/{name}/{mode}"
                    cpu = terms.dispersion0(cache, dimer, nfrozen_a, nfrozen_b, block_size=24,
                                            _graph_device=mode, do_print=False)
                    compare(name, f"cpu-{mode}", cpu, native)
                assert not report["failures"], "CPU baseline diverged before GPU comparison"
                for edge in (1, 5, 24, 500):
                    phase = f"physical/{name}/edge-{edge}"
                    os.environ["DISP0_TRACE_PHASE"] = phase
                    t = time.perf_counter()
                    with prof.section(phase):
                        value = terms.dispersion0(
                            cache, dimer, nfrozen_a, nfrozen_b, block_size=edge, gpu=True, do_print=False)
                    ein.gpu.device_synchronize()
                    metric = value["Disp0 execution"]
                    execution.append({"case": name, "edge": edge, "calling_seconds": time.perf_counter() - t,
                                      **metric})
                    assert metric["cached_work_shapes"] <= 4 and metric["cached_graph_shapes"] <= 4
                    compare(name, "cuda-request", value, native, edge=edge)
                    gc.collect()
                    assert terms._FDISP_POOL.bytes_used == 0, "Device graph retained host scratch across calls"
                phase = f"physical/{name}/nonaligned-df-cap"
                os.environ["DISP0_TRACE_PHASE"] = phase
                with prof.section(phase):
                    value = terms.dispersion0(
                        cache, dimer, nfrozen_a, nfrozen_b, block_size=5,
                        _df_block_size=7, gpu=True, do_print=False)
                ein.gpu.device_synchronize()
                assert value["Disp0 execution"]["df_block_extents"] == [5, 5]
                assert value["Disp0 execution"]["cached_graph_shapes"] <= 4
                compare(name, "cuda-df-cap7", value, native)
                os.environ["DISP0_TRACE_PHASE"] = f"cpu-oracle/{name}/asymmetric-freeze"
                asym_ref = mp2.df_mp2_fisapt_dispersion(a, a.basisset(), auxiliary, cache, 1, 0, False)
                phase = f"physical/{name}/asymmetric-freeze"
                os.environ["DISP0_TRACE_PHASE"] = phase
                with prof.section(phase):
                    asymmetric = terms.dispersion0(cache, dimer, 1, 0, block_size=24, gpu=True, do_print=False)
                ein.gpu.device_synchronize()
                compare(name, "cuda-asymmetric-freeze", asymmetric, asym_ref, frozen=[1, 0])
                phase = f"physical/{name}/production"
                os.environ["DISP0_TRACE_PHASE"] = phase
                with prof.section(phase):
                    production = original_entry(dimer, cache, nfrozen_a, nfrozen_b, do_print)
                ein.gpu.device_synchronize()
                os.environ["DISP0_TRACE_PHASE"] = f"post-production/{name}"
                compare(name, "production", production, native)
                production_values[name] = {k: float(production[k]) for k in LABELS}
                return production

            psi4.set_options({
                "basis": basis, "puream": puream, "scf_type": "df", "freeze_core": frozen,
                "DF_BASIS_MP2": "cc-pvdz-ri", "DF_BASIS_SAPT": "cc-pvdz-ri",
                "SAPT_DFT_FUNCTIONAL": "HF", "SAPT_DFT_MP2_DISP_ALG": "EINSUMS",
                "SAPT_DFT_EINSUMS_GPU": True,
                "SAPT_DFT_EINSUMS_DISP_BLOCK": 500 if name == "benzene-dimer-fc" else 24,
                "SAPT_DFT_USE_EINSUMS": True, "SAPT_DFT_DO_DHF": False,
                "SAPT_DFT_INDUCTION_TYPE": "NONE", "SAPT_DFT_DO_FSAPT": "NONE",
                "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL", "FISAPT_FSAPT_FILEPATH": "none",
            })
            t = time.perf_counter()
            with patch.object(terms, "build_sapt_jk_cache", cache_builder), \
                    patch.object(mp2, "df_mp2_einsums_dispersion", numerical_entry):
                energy, wfn = psi4.energy("sapt(dft)", molecule=psi4.geometry(geometry), return_wfn=True)
            published = {label: float(core.variable(label)) for label in LABELS}
            published_wfn = {label: float(wfn.variable(label)) for label in LABELS}
            compare(name, "published-global-signed", published, production_values[name])
            compare(name, "published-wavefunction-signed", published_wfn, production_values[name])
            expected_dispersion = sum(production_values[name].values())
            global_dispersion = float(core.variable("SAPT DISP ENERGY"))
            wfn_dispersion = float(wfn.variable("SAPT DISP ENERGY"))
            global_total = float(core.variable("SAPT TOTAL ENERGY"))
            component_total = sum(float(core.variable(label)) for label in
                                  ("SAPT ELST ENERGY", "SAPT EXCH ENERGY", "SAPT IND ENERGY", "SAPT DISP ENERGY"))
            np.testing.assert_allclose([global_dispersion, wfn_dispersion], expected_dispersion,
                                       rtol=KERNEL_RTOL, atol=KERNEL_ATOL)
            np.testing.assert_allclose([float(energy), global_total, float(wfn.variable("SAPT TOTAL ENERGY"))],
                                       component_total, rtol=KERNEL_RTOL, atol=KERNEL_ATOL)
            report.setdefault("published_driver_sum_rules", {})[name] = {
                "global_signed_Eh": published, "wavefunction_signed_Eh": published_wfn,
                "expected_dispersion_Eh": expected_dispersion, "global_dispersion_Eh": global_dispersion,
                "wavefunction_dispersion_Eh": wfn_dispersion, "returned_energy_Eh": float(energy),
                "component_total_Eh": component_total, "global_total_Eh": global_total,
            }
            report.setdefault("whole_case_including_oracles_seconds", {})[name] = time.perf_counter() - t
            report["execution"] = execution
            checkpoint()

        for related in ("ne-water-reversed", "water-ne-repeat"):
            compare(related, "invariance-to-first-geometry", production_values[related], production_values["water-ne"])
        assert any(m["planned_gpu_contractions"] > 0 for m in execution), "No naturally GPU-eligible physical block"
        assert any(m["edge"] == 1 and m["planned_cpu_contractions"] > 0 for m in execution), \
            "CPU edge fallback not covered"

        # Native fragment-resolved oracle/sum rule, not GPU fragment coverage.
        core.clean()
        core.clean_options()
        objects = []
        native_fdisp = core.FISAPT.fdisp

        def fragment_capture(obj):
            native_fdisp(obj)
            objects.append(obj)

        psi4.set_options({"basis": "cc-pvdz", "puream": True, "scf_type": "df", "freeze_core": False,
                          "DF_BASIS_SAPT": "cc-pvdz-ri", "FISAPT_FSAPT_FILEPATH": "none",
                          "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL"})
        os.environ["DISP0_TRACE_PHASE"] = "cpu-fragment-oracle"
        with patch.object(core.FISAPT, "fdisp", fragment_capture):
            psi4.energy("fisapt0", molecule=psi4.geometry(WATER_NE))
        assert len(objects) == 1
        obj = objects[0]
        fragment = {"Disp20,u": obj.scalars()["Disp20"], "Exch-Disp20,u": obj.scalars()["Exch-Disp20"]}
        partition_sum = float(np.asarray(obj.matrices()["Disp_AB"]).sum())
        np.testing.assert_allclose(partition_sum, sum(fragment.values()), rtol=KERNEL_RTOL, atol=KERNEL_ATOL)
        compare("water-ne", "cpu-fragment-signed-sum-oracle", production_values["water-ne"], fragment)
        report["cpu_fragment_sum_rule"] = {"matrix_sum_Eh": partition_sum, "signed_scalar_sum_Eh": sum(fragment.values())}

        # Same unchanged non-HF regression as the CPU gate, now with a real
        # CUDA-capable runtime/device present. Printed references are mEh.
        core.clean()
        core.clean_options()
        psi4.set_options({
            "basis": "sto-3g", "scf_type": "df", "SAPT_DFT_FUNCTIONAL": "PBE0",
            "SAPT_DFT_MP2_DISP_ALG": "SAPT", "SAPT_DFT_EINSUMS_GPU": False,
            "sapt_dft_grac_shift_a": 0.203293, "sapt_dft_grac_shift_b": 0.203293,
            "SAPT_DFT_DO_DHF": False, "SAPT_DFT_DO_HYBRID": False,
            "SAPT_DFT_USE_EINSUMS": True, "SAPT_DFT_EXCH_DISP_SCALE_SCHEME": "NONE",
            "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL",
        })
        os.environ["DISP0_TRACE_PHASE"] = "cpu-nonhf-regression"
        psi4.energy("sapt(dft)", molecule=psi4.geometry("Ne\n--\nNe 1 4.5\nunits bohr"))
        nonhf_refs = {"SAPT ELST ENERGY": -0.22987897, "SAPT EXCH ENERGY": 0.59560159,
                      "SAPT IND ENERGY": -0.00010341, "SAPT DISP ENERGY": 0.00000574,
                      "CURRENT ENERGY": 0.36562495}
        report["nonhf_comparisons_mEh"] = []
        for label, expected in nonhf_refs.items():
            actual = float(core.variable(label)) * 1000
            report["nonhf_comparisons_mEh"].append(
                {"label": label, "actual": actual, "reference": expected, "delta": actual - expected,
                 "atol": 1e-7, "rtol": 0.0})
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)

        report["unsupported_gpu_modes"] = []
        for option, value, message in (("SAPT_DFT_DO_FSAPT", "SAPTDFT", "does not yet support F-SAPT"),
                                        ("DO_DISP_EXCH_SINF", True, "does not support DO_DISP_EXCH_SINF")):
            core.clean()
            core.clean_options()
            psi4.set_options({"basis": "cc-pvdz", "SAPT_DFT_FUNCTIONAL": "HF",
                              "SAPT_DFT_MP2_DISP_ALG": "EINSUMS", "SAPT_DFT_EINSUMS_GPU": True,
                              option: value})
            try:
                psi4.energy("sapt(dft)", molecule=psi4.geometry(WATER_NE))
            except psi4.ValidationError as exc:
                assert message in str(exc), exc
                report["unsupported_gpu_modes"].append({"option": option, "rejected": True})
            else:
                raise AssertionError(f"Unsupported GPU mode silently accepted: {option}")

        prof.flush()
        profile_path = out / "einsums-gate-profile.json"
        assert prof.export_json(str(profile_path)), "Profiler export failed"
        report["profile_dispatch_evidence"] = profile_evidence(profile_path, require_cpu_edge=True)
        trace = Path(os.environ["DISP0_CUBLAS_TRACE"])
        report["native_cublas_trace_evidence"] = trace_evidence(trace, os.getpid())
        ein.gpu.device_synchronize()
        assert cuda_identity()["uuid"] == identity["uuid"]
        report["nvml_after"] = nvml_identity(identity, require_pid=True)
        report["device_free_bytes_after"] = ein.gpu.available_device_memory()
        report["host_maxrss_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["note"] = "Diagnostic timing includes CPU oracles; no speedup claim. CUDA free-memory endpoints are not VRAM peak."
        assert not report["failures"], "Signed numerical discrepancies retained"
        report["status"] = "PASS"
    except Exception:
        report["status"] = "FAIL"
        report["exception"] = traceback.format_exc()
        raise
    finally:
        report["whole_gate_seconds"] = time.perf_counter() - started
        checkpoint()
        print(json.dumps({"status": report["status"], "result": str(result_path)}))


if __name__ == "__main__":
    main()
