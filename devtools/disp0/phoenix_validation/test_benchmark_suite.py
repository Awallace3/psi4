"""Offline scheduling/qualification contracts, not native timing evidence."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import benchmark_suite as suite


class BenchmarkContracts(unittest.TestCase):
    def test_preflight_failure_still_cleans_payload_owned_scratch(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, scratch = base/"job", base/"scratch"
            root.mkdir()
            scratch.mkdir(mode=0o700)
            # No manifest: this fails before scheduler/configuration work.
            with patch.dict(os.environ, {"TMPDIR": str(scratch),
                                         "DISP0_SPLIT_SCRATCH_CREATED": str(scratch)}):
                with self.assertRaises(FileNotFoundError):
                    suite.execute(root)
            self.assertFalse(scratch.exists())
            result = json.loads((root/"benchmark-result.json").read_text())
            self.assertEqual(result["status"], "FAILED")
            self.assertTrue(result["scratch_cleanup"]["complete"])

    def test_cleanup_failure_invalidates_previously_qualified_results(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, scratch = base/"job", base/"scratch"
            root.mkdir()
            scratch.mkdir(mode=0o700)
            def run(root, groups):
                suite.save(root/"benchmark-result.json", {
                    "status": "COMPLETED_REQUIRES_RECORDED_CASE_QUALIFICATION",
                    "accepted_speedup_count": 1, "cases": {"peptide": {
                        "status": "PASS_MATCHED_INSTRUMENTED_BENCHMARK",
                        "accepted_speedup": True, "cpu_over_gpu_ratio": 2.0}}})
            with patch.dict(os.environ, {"TMPDIR": str(scratch),
                                         "DISP0_SPLIT_SCRATCH_CREATED": str(scratch)}), \
                    patch.object(suite, "_execute", side_effect=run), \
                    patch.object(suite.OwnedScratch, "cleanup", side_effect=RuntimeError("cleanup")):
                with self.assertRaisesRegex(RuntimeError, "cleanup"):
                    suite.execute(root)
            result = json.loads((root/"benchmark-result.json").read_text())
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["accepted_speedup_count"], 0)
            self.assertFalse(result["cases"]["peptide"]["accepted_speedup"])
            self.assertEqual(result["cases"]["peptide"]["status"], "UNQUALIFIED_CLEANUP_FAILED")
            self.assertFalse(result["scratch_cleanup"]["complete"])

    def test_one_gpu_and_three_cpu_jobs_no_cpu_gpu_reservations(self):
        for label in ("gpu", *suite.SYSTEMS):
            gpu = label == "gpu"
            argv = suite.command(Path("/source"), Path("/run"), label,
                                 "gts-cs207-chemx", None if gpu else "123")
            self.assertIn("--qos=inferno", argv)
            self.assertEqual(any(x.startswith("--gres=") for x in argv), gpu)
            self.assertEqual(any(x.startswith("--dependency=") for x in argv), not gpu)
            if not gpu:
                self.assertIn("--dependency=afterany:123", argv)
                self.assertIn("--kill-on-invalid-dep=yes", argv)
        with self.assertRaises(ValueError):
            suite.command(Path("/s"), Path("/r"), "gpu", "gts-cs207", "123")

    def test_gpu_worker_cannot_execute_cpu_oracle_and_cache_is_explicit(self):
        config = {"stage": "/stage", "dependency_environment": "/env"}
        manifest = {"kind": "gpu", "config": "/config", "config_sha256": "sha",
                    "inputs": "/inputs", "inputs_sha256": "input"}
        argv = suite.worker_command(Path("/helper"), manifest, config, Path("/out"),
                                    system="peptide")
        self.assertIn("--benchmark", argv)
        self.assertEqual(argv[argv.index("--arm")+1], "gpu")
        self.assertEqual(argv[argv.index("--factor-cache-bytes")+1], str(suite.CACHE_BYTES))
        with self.assertRaises(ValueError):
            suite.worker_command(Path("/h"), manifest, config, Path("/o"), oracle=Path("/gpu"))
        cpu = {**manifest, "kind": "cpu", "gpu_config": "/peer", "gpu_config_sha256": "peer"}
        argv = suite.worker_command(Path("/h"), cpu, config, Path("/o"), oracle=Path("/gpu"))
        self.assertEqual(argv[argv.index("--arm")+1], "cpu")
        self.assertNotIn("--factor-cache-bytes", argv)

    def test_medians_require_all_three_positive_finite_samples(self):
        self.assertEqual(suite.summary([3., 1., 2.])["median_seconds"], 2.)
        for samples in ([1, 2], [1, 2, 0], [1, 2, float("nan")], [1, 2, float("inf")]):
            with self.assertRaises(ValueError):
                suite.summary(samples)

    def exercise_gpu(self, *, failure=None, interrupt=False):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base/"job"
            root.mkdir()
            scratch = base/"scratch"
            scratch.mkdir(mode=0o700)
            control = base/"cpu-water.json"
            control.write_text(json.dumps({"source_pin": "science"}))
            manifest = {"kind": "gpu", "config": "/config", "config_sha256": "cfg",
                        "science_pin": "science", "helper_pin": "helper",
                        "systems": list(suite.SYSTEMS), "inputs": "/input",
                        "inputs_sha256": "input", "allocation_wall_seconds": 57600,
                        "cpu_water_result": str(control), "cpu_water_sha256": suite.digest(control)}
            (base/"manifest.json").write_text(json.dumps(manifest))
            calls = []

            def execute(argv, env, out, err, timeout, started, *, reaped):
                label = Path(argv[argv.index("--output")+1]).name
                path = Path(argv[argv.index("--output")+1])
                path.mkdir()
                calls.append(label)
                started(1234)
                reaped(1234)
                if label.startswith(failure or "NO_MATCH"):
                    if interrupt:
                        raise InterruptedError("TERM")
                    return 1
                system = argv[argv.index("--system")+1] if "--system" in argv else None
                doc = {"status": "EXECUTED_REQUIRES_INDEPENDENT_COMPARISON",
                       "source_pin": "science", "runtime_config_sha256": "cfg",
                       "arm": "gpu", "benchmark": True, "system": system,
                       "artifact_sha256": {}, "benchmark_whole_energy_seconds": 2.0}
                (path/"result.json").write_text(json.dumps(doc))
                return 0

            with patch.dict(os.environ, {"SLURM_JOB_ID": "456", "TMPDIR": str(scratch),
                                         "DISP0_SPLIT_SCRATCH_CREATED": str(scratch)}), \
                    patch.object(suite, "read_config", return_value=(
                        {"source_pin": "science", "threads": 8,
                         "stage": "/stage", "dependency_environment": "/env"}, {})), \
                    patch.object(suite, "validate_allocation"), \
                    patch.object(suite.subprocess, "check_output", return_value="job"), \
                    patch.object(suite.os, "sched_getaffinity", return_value=set(range(8))), \
                    patch.object(suite, "environment", return_value={}), \
                    patch.object(suite, "execute_owned", side_effect=execute), \
                    patch.object(suite, "load_point", return_value=({}, {})), \
                    patch.object(suite, "compare_repeat", return_value={}), \
                    patch.object(suite, "compare_energy", return_value={}), \
                    patch.object(suite, "compare", return_value={}):
                if failure:
                    with self.assertRaises(InterruptedError if interrupt else RuntimeError):
                        suite.execute(root)
                else:
                    suite.execute(root)
            result = json.loads((root/"benchmark-result.json").read_text())
            self.assertFalse(scratch.exists())
            self.assertTrue(result["scratch_cleanup"]["complete"])
            return calls, result

    def test_gpu_plan_only_gpu_fresh_calls_zero_accepted_speedups(self):
        calls, result = self.exercise_gpu()
        self.assertEqual(len(calls), 10)  # water + 3x3 molecular calls
        self.assertEqual(result["accepted_speedup_count"], 0)
        self.assertEqual(result["cases"]["peptide"]["timing"]["n"], 3)

    def test_failed_case_is_preserved_not_retried_other_cases_are_distinct(self):
        calls, result = self.exercise_gpu(failure="nanotube")
        self.assertEqual(calls.count("nanotube-r0"), 1)
        self.assertNotIn("nanotube-r1", calls)
        self.assertIn("protein83-r2", calls)
        self.assertEqual(result["cases"]["nanotube"]["status"], "FAILED")

    def test_TERM_stops_all_further_launches_and_cleans_owned_scratch(self):
        calls, _ = self.exercise_gpu(failure="peptide", interrupt=True)
        self.assertEqual(calls, ["water-control", "peptide-r0"])

    def test_CPU_references_still_run_when_GPU_case_failed_no_false_speedup(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, scratch, producer = (base/name for name in ("job", "scratch", "gpu"))
            for path in (root, scratch, producer):
                path.mkdir(mode=0o700)
            manifest = {"kind": "cpu", "config": "/config", "config_sha256": "cfg",
                        "science_pin": "science", "helper_pin": "helper",
                        "systems": ["nanotube"], "inputs": "/input", "inputs_sha256": "input",
                        "allocation_wall_seconds": 86400, "gpu_job_root": str(producer),
                        "gpu_config": "/peer", "gpu_config_sha256": "peer"}
            (base/"manifest.json").write_text(json.dumps(manifest))
            (producer/"benchmark-result.json").write_text(json.dumps({
                "source_pin": "science", "helper_pin": "helper",
                "cases": {"nanotube": {"status": "FAILED"}}}))
            calls = []

            def execute(argv, env, out, err, timeout, started, *, reaped):
                self.assertNotIn("--oracle-from", argv)
                self.assertEqual(argv[argv.index("--arm")+1], "cpu")
                path = Path(argv[argv.index("--output")+1])
                calls.append(path.name)
                path.mkdir()
                started(1234)
                reaped(1234)
                (path/"result.json").write_text(json.dumps({
                    "status": "EXECUTED_REQUIRES_INDEPENDENT_COMPARISON",
                    "source_pin": "science", "runtime_config_sha256": "cfg",
                    "arm": "cpu", "benchmark": True, "system": "nanotube",
                    "artifact_sha256": {}, "benchmark_whole_energy_seconds": 4.0}))
                return 0

            config = {"source_pin": "science", "threads": 24, "stage": "/stage",
                      "dependency_environment": "/env"}
            def read_config(path, checksum):
                return ({**config, "threads": 24 if checksum == "cfg" else 8}, {})
            with patch.dict(os.environ, {"SLURM_JOB_ID": "456", "TMPDIR": str(scratch),
                                         "DISP0_SPLIT_SCRATCH_CREATED": str(scratch)}), \
                    patch.object(suite, "read_config", side_effect=read_config), \
                    patch.object(suite, "validate_allocation"), \
                    patch.object(suite.subprocess, "check_output", return_value="FSTYPE=xfs"), \
                    patch.object(suite.os, "statvfs", return_value=type(
                        "Disk", (), {"f_bavail": 512*1024**3, "f_frsize": 1})()), \
                    patch.object(suite.os, "sched_getaffinity", return_value=set(range(24))), \
                    patch.object(suite, "environment", return_value={}), \
                    patch.object(suite, "execute_owned", side_effect=execute), \
                    patch.object(suite, "load_point", return_value=({}, {})), \
                    patch.object(suite, "compare_repeat", return_value={}):
                suite.execute(root)
            result = json.loads((root/"benchmark-result.json").read_text())
            self.assertEqual(calls, [f"nanotube-r{i}" for i in range(3)])
            case = result["cases"]["nanotube"]
            self.assertEqual(case["status"], "PASS_CPU_REFERENCES_GPU_CASE_UNQUALIFIED")
            self.assertEqual(case["timing"]["n"], 3)
            self.assertFalse(case["accepted_speedup"])
            self.assertEqual(result["accepted_speedup_count"], 0)
            self.assertFalse(scratch.exists())


if __name__ == "__main__":
    unittest.main()
