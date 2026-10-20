import importlib.util
import json
import os
from pathlib import Path
import tempfile
import subprocess
import copy
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("compare", ROOT / "compare.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.c = json.loads((ROOT / "config.json").read_text())
        self.c["repeats"] = 1

    def record(self, engine, **kwargs):
        r = dict(engine=engine, system="peptide", functional="pbe", profile=False,
                 ok=True, process_returncode=0, scf_wall_s=2 if engine == "psi4" else 1,
                 calculation_wall_s=3 if engine == "psi4" else 2,
                 process_wall_s=4, scf_iterations=15, energy_hartree=-100,
                 nbf=210, naux=600, functional_metadata=dict(
                     omega=0, short_range_exchange=0, long_range_exchange=0,
                     components=self.c["functionals"]["pbe"]["components"]))
        r.update(kwargs)
        return r

    def row(self, records):
        return next(r for r in m.summarize(records, self.c, ["peptide"]) if r["functional"] == "pbe")

    def test_complete_ratio(self):
        row = self.row([self.record(e) for e in m.ENGINES])
        self.assertTrue(row["complete"])
        self.assertTrue(row["comparison_eligible"])
        self.assertEqual(row["psi4_over_pyscf_scf_time"], 2)
        self.assertEqual(row["psi4_over_pyscf_calculation_time"], 1.5)

    def test_full_process_headline_includes_imports_and_rejects_missing_clock(self):
        self.c["headline_timing"] = "full_process_wall_s"
        records = [self.record("psi4", full_process_wall_s=10),
                   self.record("pyscf", full_process_wall_s=20)]
        row = self.row(records)
        self.assertEqual(row["headline_timing"], "full_process_wall_s")
        self.assertEqual(row["psi4_over_pyscf_headline_time"], .5)
        self.assertEqual(row["psi4"]["full_process_wall_s"]["median"], 10)
        del records[0]["full_process_wall_s"]
        self.assertFalse(self.row(records)["complete"])

    def test_matching_nbf_does_not_mask_retained_dimension_mismatch(self):
        self.c["overlap_eigenvalue_cutoff"] = 1e-7
        policy = dict(method="canonical", absolute_eigenvalue_cutoff=1e-7)
        row = self.row([self.record("psi4", nmo=209, overlap_policy=policy),
                        self.record("pyscf", nmo=193, overlap_policy=policy)])
        self.assertFalse(row["comparison_eligible"])
        self.assertNotIn("psi4_over_pyscf_headline_time", row)

    def test_memory_required_without_losing_chemistry_status(self):
        self.c["require_memory_observations"] = True
        records = [self.record(e, memory=dict(peak_rss_kib=123,
                   sampled_gpu_device_peak_mib=100,
                   sampled_gpu_process_peak_mib=None)) for e in m.ENGINES]
        self.assertTrue(self.row(records)["complete"])
        self.assertFalse(self.row(records)["psi4"]["gpu_process_attribution_available"])
        records[0]["memory"]["sampled_gpu_device_peak_mib"] = None
        row = self.row(records)
        self.assertFalse(row["complete"])
        self.assertTrue(row["chemistry_and_timing_complete"])
        self.assertNotIn("psi4_over_pyscf_calculation_time", row)
        records[0]["memory"]["sampled_gpu_device_peak_mib"] = 100
        records[0]["memory"]["peak_rss_kib"] = None
        self.assertFalse(self.row(records)["complete"])

    def test_equal_timed_ranks_must_also_match_pinned_audit(self):
        audit = dict(nmo=209, references={"proof": "hash"})
        self.c["audited_overlap"] = audit
        records = [self.record(e, nmo=208, config=dict(audited_overlap=audit))
                   for e in m.ENGINES]
        self.assertFalse(self.row(records)["comparison_eligible"])
        for record in records:
            record["nmo"] = 209
        self.assertTrue(self.row(records)["comparison_eligible"])
        records[0]["config"] = {}
        self.assertFalse(self.row(records)["comparison_eligible"])

    def test_stored_df_claim_requires_route_evidence(self):
        self.c["df_mode"] = "stored-df"
        row = self.row([self.record("psi4"), self.record("pyscf", stored_df_verified=False)])
        self.assertFalse(row["comparison_eligible"])

    def test_conventional_direct_needs_proof_not_auxiliary_dimensions(self):
        self.c["jk_mode"] = "conventional-direct"
        records = [self.record(e, jk_mode="conventional-direct", naux=None,
                               direct_verified=True) for e in m.ENGINES]
        row = self.row(records)
        self.assertTrue(row["comparison_eligible"])
        self.assertIsNone(row["naux_equal"])
        records[1]["direct_verified"] = False
        self.assertFalse(self.row(records)["comparison_eligible"])
        records[1]["direct_verified"] = True
        records[0]["naux"] = 600
        self.assertFalse(self.row(records)["comparison_eligible"])

    def test_df_and_four_center_direct_repeats_cannot_mix(self):
        self.c["jk_mode"] = "conventional-direct"
        records = [self.record(e, jk_mode="conventional-direct", naux=None,
                               direct_verified=True) for e in m.ENGINES]
        records[1]["jk_mode"] = "density-fitted"
        self.assertFalse(self.row(records)["complete"])

    def test_common_input_file_does_not_mask_live_coordinate_mismatch(self):
        self.c["geometry_angstrom_per_bohr"] = .52917721067
        policy = dict(input_unit="Bohr", angstrom_per_bohr=.52917721067,
                      live_coordinates_verified=True)
        records = [self.record(e, geometry_bohr_sha256="same", geometry_policy=policy)
                   for e in m.ENGINES]
        self.assertTrue(self.row(records)["comparison_eligible"])
        records[1]["geometry_bohr_sha256"] = "different"
        self.assertFalse(self.row(records)["comparison_eligible"])
        records[1]["geometry_bohr_sha256"] = "same"
        records[1]["geometry_policy"] = {}
        self.assertFalse(self.row(records)["comparison_eligible"])

    def test_teardown_failure_is_not_success(self):
        row = self.row([self.record("psi4"), self.record("pyscf", process_returncode=-11)])
        self.assertFalse(row["complete"])
        self.assertNotIn("psi4_over_pyscf_scf_time", row)

    def test_instrumentation_excluded(self):
        row = self.row([self.record(e) for e in m.ENGINES] + [
            self.record("pyscf", profile=True, scf_wall_s=100)])
        self.assertEqual(row["psi4_over_pyscf_scf_time"], 2)

    def test_aux_mismatch_suppresses_ratio(self):
        row = self.row([self.record("psi4"), self.record("pyscf", naux=601)])
        self.assertFalse(row["comparison_eligible"])
        self.assertNotIn("psi4_over_pyscf_scf_time", row)

    def test_functional_mismatch_suppresses_ratio(self):
        p = self.record("pyscf")
        p["functional_metadata"]["omega"] = .3
        row = self.row([self.record("psi4"), p])
        self.assertFalse(row["comparison_eligible"])

    def test_missing_repeat_remains_incomplete(self):
        self.c["repeats"] = 3
        self.assertFalse(self.row([self.record(e) for e in m.ENGINES])["complete"])

    def test_components_mismatch_suppresses_ratio(self):
        p = self.record("pyscf")
        p["functional_metadata"]["components"] = {"LDA_X": 1}
        row = self.row([self.record("psi4"), p])
        self.assertFalse(row["comparison_eligible"])
        self.assertNotIn("psi4_over_pyscf_calculation_time", row)

    def test_vwn_variants_cannot_pass_component_validation(self):
        with self.assertRaises(AssertionError):
            m.verify_components({"LDA_X": 1, "LDA_C_VWN": 1},
                                self.c["functionals"]["svwn"]["components"])

    def test_malformed_result_is_a_failure(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "result.json"
            p.write_text("{not json")
            r = m.load_result(p, dict(engine="pyscf"))
            self.assertFalse(r["ok"])
            self.assertEqual(r["engine"], "pyscf")

    def test_wrong_worker_identity_is_a_failure(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "result.json"
            p.write_text(json.dumps(dict(engine="psi4", ok=False)))
            self.assertFalse(m.load_result(p, dict(engine="pyscf"))["ok"])

    def test_cuda_major_and_mixed_runtime_rejected(self):
        for engine, major in (("psi4", 13), ("pyscf", 12)):
            record = dict(engine=engine, gpu={"cuda_runtime": major * 1000 + 60},
                          runtime_libraries=[f"/cuda/libcudart.so.{major}.0"])
            m.validate_cuda_route(record)
            with self.assertRaises(RuntimeError):
                m.validate_cuda_route(dict(record, gpu={"cuda_runtime": (25-major) * 1000}))
            with self.assertRaises(RuntimeError):
                m.validate_cuda_route(dict(record, runtime_libraries=["/cuda/libcudart.so.11"]))

    def test_cupy_uuid_embedded_nul_uses_visible_device_pci_not_nvml_index(self):
        # CuPy's device-properties char-array conversion truncates at NUL.
        # This is the actual H100 UUID from failed job14197755.
        uuid = "GPU-43b49b76-6aa3-0000-d560-748ac1c97c5d"
        raw = bytes.fromhex(uuid[4:].replace("-", ""))
        self.assertEqual(len(raw.split(b"\0", 1)[0]), 6)
        for bus in ("0000:17:00.0", b"0000:17:00.0"):
            runtime = SimpleNamespace(
                getDeviceCount=lambda: 1, getDevice=lambda: 0,
                getDeviceProperties=lambda device: {"uuid": raw.split(b"\0", 1)[0]},
                deviceGetPCIBusId=lambda device: bus, runtimeGetVersion=lambda: 12060)
            with patch.object(m.subprocess, "check_output", return_value=uuid + "\n") as query:
                identity = m.cuda_identity(runtime)
            self.assertEqual(identity["uuid"], uuid)
            self.assertEqual(identity["pci_bus_id"], "0000:17:00.0")
            self.assertEqual(identity["logical_device"], 0)
            self.assertEqual(identity["cuda_runtime"], 12060)
            query.assert_called_once_with(
                ["nvidia-smi", "-i", "0000:17:00.0", "--query-gpu=uuid",
                 "--format=csv,noheader"], text=True)

    def test_cupy_complete_uuid_needs_no_nvml_query(self):
        raw = bytes.fromhex("11111111222233334444555555555555")
        runtime = SimpleNamespace(getDeviceCount=lambda: 1, getDevice=lambda: 0,
            getDeviceProperties=lambda device: {"uuid": raw}, runtimeGetVersion=lambda: 12060)
        with patch.object(m.subprocess, "check_output") as query:
            self.assertEqual(m.cuda_identity(runtime)["uuid"],
                             "GPU-11111111-2222-3333-4444-555555555555")
        query.assert_not_called()

    def test_cupy_uuid_fallback_rejects_malformed_or_multiple_matches(self):
        runtime = SimpleNamespace(getDeviceCount=lambda: 1, getDevice=lambda: 0,
            getDeviceProperties=lambda device: {"uuid": b""},
            deviceGetPCIBusId=lambda device: "0000:17:00.0", runtimeGetVersion=lambda: 12060)
        for output in ("", "GPU-wrong", "GPU-11111111-2222-3333-4444-555555555555\n"
                       "GPU-11111111-2222-3333-4444-555555555555\n"):
            with patch.object(m.subprocess, "check_output", return_value=output):
                with self.assertRaises(AssertionError):
                    m.cuda_identity(runtime)

    def test_psi4_direct_guess_disabled_accepts_native_integer_boolean(self):
        # core.cc py_psi_get_option exports boolean via py::cast(to_integer()).
        for value in (False, 0):
            evidence = m.psi4_direct_evidence(
                "cuESTDirectJK", "cuESTDirectJK: GPU-Accelerated Four-Center Direct",
                ["SCF/HF/JK/cuEST Direct J/K"], value)
            self.assertTrue(all(evidence.values()), (value, evidence))

    def test_psi4_direct_proof_rejects_enabled_unknown_and_missing_routes(self):
        backend = "cuESTDirectJK"
        header = "cuESTDirectJK: GPU-Accelerated Four-Center Direct"
        timers = ["SCF/HF/JK/cuEST Direct J/K"]
        for value in (True, 1, None, "", "FALSE", "0", 0.0, [], {}):
            self.assertFalse(all(m.psi4_direct_evidence(backend, header, timers, value).values()))
        for args in (("DirectJK", header, timers), (backend, "", timers),
                     (backend, header, []),
                     (backend, header + " cuESTJK: GPU-Accelerated", timers)):
            self.assertFalse(all(m.psi4_direct_evidence(*args, 0).values()))

    def test_observer_entry_interrupt_reaps_spawned_supervisor(self):
        procs = []
        popen = subprocess.Popen
        def spawn(*args, **kwargs):
            proc = popen(*args, **kwargs)
            procs.append(proc)
            return proc
        observer = SimpleNamespace(
            __enter__=None, summary=lambda: {"peak_rss_kib": None})
        class BrokenObserver:
            exited = False
            def __enter__(self):
                raise KeyboardInterrupt("TERM during observer entry")
            def __exit__(self, *args):
                self.exited = True
            def summary(self):
                return {"peak_rss_kib": None}
        observer = BrokenObserver()
        with tempfile.TemporaryDirectory() as directory, patch.object(
                m.subprocess, "Popen", side_effect=spawn), patch.object(
                m, "MemoryMonitor", return_value=observer):
            record, interrupted = m.observed_worker(
                [sys.executable, "-c", "import time; time.sleep(60)"], os.environ.copy(),
                Path(directory), "signal-window", {"engine": "psi4"}, "GPU-unused", 120,
                Path(sys.executable))
        self.assertIsInstance(interrupted, KeyboardInterrupt)
        self.assertTrue(observer.exited)
        self.assertTrue(all(proc.poll() is not None for proc in procs))
        self.assertFalse(record["ok"])

    def test_energy_review_is_separate(self):
        row = self.row([self.record("psi4"), self.record("pyscf", energy_hartree=-99)])
        self.assertTrue(row["complete"])
        self.assertTrue(row["energy_review"])

    def test_memory_summary_retains_failed_worker_and_missing_values(self):
        records = [self.record("psi4", process_returncode=1,
                              memory={"peak_rss_kib": 123, "sampled_gpu_process_peak_mib": None}),
                   self.record("pyscf", memory={"peak_rss_kib": 456,
                                                "sampled_gpu_process_peak_mib": 10})]
        row = self.row(records)
        self.assertFalse(row["complete"])
        self.assertEqual(row["psi4"]["memory"]["peak_rss_kib"]["max"], 123)
        self.assertEqual(row["psi4"]["memory"]["sampled_gpu_process_peak_mib"]["missing"], 1)

    def test_controller_interruption_preserves_attempt_and_peak_rss(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            ready = base / "ready"
            command = [sys.executable, "-c",
                       f"from pathlib import Path;import time;x=bytearray(32*1024**2);"
                       f"Path({str(ready)!r}).touch();time.sleep(100)"]
            real_popen = subprocess.Popen
            def popen(*args, **kwargs):
                proc = real_popen(*args, **kwargs)
                wait = proc.wait
                first = True
                def interrupted_wait(*args, **kwargs):
                    nonlocal first
                    if first:
                        first = False
                        deadline = time.monotonic() + 5
                        while not ready.exists() and time.monotonic() < deadline:
                            time.sleep(.01)
                        if not ready.exists():
                            raise RuntimeError("Fixture did not start")
                        raise KeyboardInterrupt("scheduler TERM fixture")
                    return wait(*args, **kwargs)
                proc.wait = interrupted_wait
                return proc
            identity = dict(engine="psi4", system="peptide", functional="pbe",
                            profile=False, repeat=0)
            with patch.object(m.subprocess, "Popen", side_effect=popen), \
                 patch("memory.gpu_sample", return_value=dict(
                     gpu_device_mib=100, gpu_process_mib=90, error="")):
                record, interrupted = m.observed_worker(
                    command, os.environ.copy(), base, "attempt", identity, "gpu", 20, sys.executable)
            self.assertFalse(record["ok"])
            self.assertGreater(record["memory"]["peak_rss_kib"], 32*1024)
            records = []
            with self.assertRaises(KeyboardInterrupt):
                m.persist_attempt(base, records, record, self.c, ["peptide"], interrupted)
            saved = json.loads((base / "records.json").read_text())
            self.assertEqual(len(saved), 1)
            self.assertIn("controller_interrupted", saved[0])
            self.assertEqual(saved[0]["memory"]["peak_rss_kib"], record["memory"]["peak_rss_kib"])

    def test_no_pruning_and_native_fp64(self):
        opts = m.psi_options(self.c)
        self.assertEqual(opts["dft_pruning_scheme"], "none")
        self.assertFalse(opts["cuest_mixed_precision"])

    def test_native_psi4_grid_options_are_not_overridden(self):
        self.c["grid_mode"] = "native-default"
        opts = m.psi_options(self.c)
        self.assertFalse(any(key.startswith("dft_") for key in opts))
        self.assertTrue(opts["use_cuest"])
        self.assertTrue(opts["cuest_xc"])
        self.assertEqual(opts["basis"], "cc-pvdz")

    def test_unknown_psi4_grid_mode_rejected(self):
        self.c["grid_mode"] = "equal-count"
        with self.assertRaises(ValueError):
            m.psi_options(self.c)

    def test_historical_stage_cannot_silently_export_empty_host_grid(self):
        psi4 = SimpleNamespace(core=SimpleNamespace(VBase=type("VBase", (), {})))
        with self.assertRaisesRegex(RuntimeError, "instrumented"):
            with m.audit_psi4_grid(psi4, {}, True):
                self.fail("Missing exporter should be fatal")

    def test_successful_energy_without_export_hook_cannot_pass_audit(self):
        with self.assertRaisesRegex(RuntimeError, "Requested grid export"):
            m.require_grid_export(dict(engine="psi4", ok=True, energy_hartree=-100))

    def test_zero_point_export_cannot_pass_audit(self):
        record = dict(engine="psi4", grid_points=0,
                      grid_exports={"live-plan-before-scf": dict(stored_points=0, nonzero_weight_points=0)})
        with self.assertRaisesRegex(RuntimeError, "Requested grid export"):
            m.require_grid_export(record)

    def test_psi4_iterator_restored_after_audit_failure(self):
        class HF:
            def iterations(self):
                raise RuntimeError("iteration failure")
        original = HF.iterations
        psi4 = SimpleNamespace(core=SimpleNamespace(
            HF=HF, VBase=SimpleNamespace(export_cuest_grid=lambda: None),
            get_option=lambda *args: "test"))
        # Already exported: isolate the wrapper's error/restore behavior.
        with self.assertRaisesRegex(RuntimeError, "iteration failure"):
            with m.audit_psi4_grid(psi4, {"grid_exports": {}}, True):
                HF().iterations()
        self.assertIs(HF.iterations, original)

    def test_augmented_basis_count_and_ratio(self):
        self.c["basis"] = "aug-cc-pvtz"
        row = self.row([self.record(e, nbf=782) for e in m.ENGINES])
        self.assertTrue(row["comparison_eligible"])
        self.assertEqual(m.expected_nbf(self.c, "protein157"), 5382)

    def test_cc_pvtz_basis_count_and_ratio(self):
        self.c["basis"] = "cc-pvtz"
        row = self.row([self.record(e, nbf=496) for e in m.ENGINES])
        self.assertTrue(row["comparison_eligible"])
        self.assertEqual(m.expected_nbf(self.c, "protein83"), 1786)
        self.assertEqual(m.expected_nbf(self.c, "protein157"), 3430)

    def test_shared_cuest_preserves_native_psi_grid_construction(self):
        self.c["grid_mode"] = "shared-cuest"
        opts = m.psi_options(self.c)
        self.assertFalse(any(key.startswith("dft_") for key in opts))

    def test_old_basis_cannot_pass_augmented_basis_validation(self):
        self.c["basis"] = "aug-cc-pvtz"
        row = self.row([self.record(e) for e in m.ENGINES])
        self.assertFalse(row["comparison_eligible"])
        self.assertNotIn("psi4_over_pyscf_calculation_time", row)

    def test_unsupported_basis_is_rejected(self):
        self.c["basis"] = "not-a-supported-basis"
        with self.assertRaises(KeyError):
            m.expected_nbf(self.c, "protein83")

    def test_vwn_rpa_explicit(self):
        self.assertEqual(self.c["functionals"]["b3lyp"]["pyscf"], "HYB_GGA_XC_B3LYP")
        self.assertEqual(self.c["functionals"]["svwn"]["pyscf"], "LDA_X,LDA_C_VWN_RPA")
        self.assertEqual(set(x["rung"] for x in self.c["functionals"].values()),
                         {"lda", "gga", "mgga", "hybrid", "range-separated"})

    def test_xyz_size_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.xyz"
            p.write_text("1\ncharge=0 mult=1\nH 0 0 0\n")
            with self.assertRaises(AssertionError):
                m.xyz(p, "peptide")

    def test_real_suite_geometry_counts(self):
        suite = Path("/home/awallace43/data/timing_test_suite/geometries")
        if not suite.is_dir():
            self.skipTest("Local timing suite unavailable")
        for system, n in m.ATOMS.items():
            atom = m.xyz(suite / f"{system}.xyz", system)
            atoms = [line.split()[0] for line in atom.splitlines()]
            self.assertEqual(len(atoms), n)
            self.assertEqual(sum(5 if e == "H" else 14 for e in atoms), m.NBF[system])
            self.assertEqual(sum(23 if e == "H" else 46 for e in atoms),
                             m.NBF_BY_BASIS["aug-cc-pvtz"][system])
            self.assertEqual(sum(14 if e == "H" else 30 for e in atoms),
                             m.NBF_BY_BASIS["cc-pvtz"][system])

    def test_nested_timer_counts_are_not_fabricated(self):
        class Obj:
            def f(self):
                return 7
        obj = Obj()
        t = m.Timings(lambda: None)
        t.wrap(obj, "f", "f")
        self.assertEqual(obj.f(), 7)
        self.assertEqual(t.data["f"]["calls"], 1)

    def test_psiapi_setup_accepts_unset_pythonpath_under_nounset(self):
        payload = (ROOT / "payload.sh").read_text()
        path_setup = payload.split("export PYTHONNOUSERSITE=1\n", 1)[1].split(
            "export LD_LIBRARY_PATH=", 1)[0]
        psiapi_setup = payload.split('psi_exe=', 1)[1].split(
            'mkdir -p "$TMPDIR/inputs/geometries"', 1)[0]
        with tempfile.TemporaryDirectory() as d:
            stage = Path(d) / "stage"
            (stage / "bin").mkdir(parents=True)
            exe = stage / "bin/psi4"
            # The actual historical staged binary emits this same expansion.
            exe.write_text("#!/bin/bash\nprintf '%s\\n' 'export PATH=/stage/bin:$PATH PYTHONPATH=/stage/lib:$PYTHONPATH'\n")
            exe.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            env.update(COMPARE_PSI_PACKAGE=str(stage / "lib/psi4"),
                       COMPARE_PSI_ENV=d, COMPARE_PSI_CUDA_LIB=d)
            script = path_setup + "psi_exe=" + psiapi_setup + '\n[[ "$PYTHONPATH" == /stage/lib:* ]]'
            p = subprocess.run(["bash", "-euc", script], env=env,
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)

    def test_statically_linked_cupy_runtime_identity(self):
        class Runtime:
            getDeviceCount = staticmethod(lambda: 1)
            getDevice = staticmethod(lambda: 0)
            getDeviceProperties = staticmethod(lambda _: {
                "uuid": bytes.fromhex("951a11c240ee3fcbf71f49849e744cb1")})
            runtimeGetVersion = staticmethod(lambda: 12080)
        self.assertEqual(m.cuda_identity(Runtime())["uuid"],
                         "GPU-951a11c2-40ee-3fcb-f71f-49849e744cb1")

    def test_multiple_visible_gpus_are_rejected(self):
        class Runtime:
            getDeviceCount = staticmethod(lambda: 2)
            getDevice = staticmethod(lambda: 0)
        with self.assertRaises(AssertionError):
            m.cuda_identity(Runtime())

    def test_df_profile_build_targets_copied_range_object(self):
        class DF:
            def build(self):
                self.tensor = self.omega
                return self
        seed = DF()
        seed.omega = 0
        timers = m.Timings(lambda: None)
        timers.wrap_class(DF, "build", "df_setup")
        ranged = copy.copy(seed)
        ranged.omega = .3
        self.assertIs(ranged.build(), ranged)
        self.assertEqual(ranged.tensor, .3)
        self.assertFalse(hasattr(seed, "tensor"))
        self.assertEqual(timers.data["df_setup"]["calls"], 1)


if __name__ == "__main__":
    unittest.main()
