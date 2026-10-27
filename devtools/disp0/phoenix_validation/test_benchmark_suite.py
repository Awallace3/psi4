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

    def exercise_unqualified_CPU(self, *, gpu_status="FAILED", cleanup=True, invalidated=False):
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
                "scratch_cleanup": {"complete": cleanup},
                "cases": {"nanotube": {"status": gpu_status,
                                      "qualification_invalidated_by_cleanup": invalidated}}}))
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

    def test_CPU_references_still_run_when_GPU_case_failed_no_false_speedup(self):
        self.exercise_unqualified_CPU()

    def test_failed_GPU_cleanup_or_revoked_case_cannot_qualify_CPU_speedup(self):
        status = "PASS_REPEATS_AWAITING_OTHER_RESOURCE_CLASS"
        self.exercise_unqualified_CPU(gpu_status=status, cleanup=False)
        self.exercise_unqualified_CPU(gpu_status=status, invalidated=True)

    def test_CPU_consumer_plan_seals_existing_GPU_without_preparing_a_second_GPU(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source, old_source, producer = (base/name for name in ("new", "old", "producer"))
            for root in (source, old_source):
                shared = root/"devtools/disp0/phoenix_validation"
                shared.mkdir(parents=True)
                for name in ("fragment_gate.py", "gpu_runtime.py", "full_inputs.py",
                             "repeat_gauge.py", "owned_process.py", "run_full.py", "run_gate.py"):
                    (shared/name).write_text("unchanged engine")
            producer.mkdir()
            (producer/"job-id.txt").write_text("123\n")
            (producer/"manifest.json").write_text(json.dumps({
                "kind": "gpu", "source": str(old_source), "helper_pin": "gpu-helper",
                "science_pin": "science", "inputs_sha256": "input", "systems": list(suite.SYSTEMS)}))
            config = base/"build/gpu-runtime/runtime-config.json"
            config.parent.mkdir(parents=True)
            config.write_text("{}")
            water = config.parents[1]/"cpu-fragment-scaled/result.json"
            water.parent.mkdir()
            water.write_text(json.dumps({"status": "EXECUTED_REQUIRES_INDEPENDENT_COMPARISON",
                                        "arm": "cpu", "source_pin": "science"}))
            inputs = base/"inputs"
            inputs.write_text("{}")
            args = SimpleNamespace(source=source, config=config, config_sha256="config",
                                   inputs=inputs, inputs_sha256="input", root=base/"plan",
                                   gpu_producer_root=producer)
            with patch.object(suite, "source_pin", return_value="cpu-helper"), \
                    patch.object(suite, "read_config", return_value=({"source_pin": "science"}, {})), \
                    patch.object(suite, "load_point"):
                suite.prepare(args)
                with self.assertRaisesRegex(RuntimeError, "GPU duplication"):
                    suite.submit(SimpleNamespace(root=args.root, submit_one="gpu", account="gts-cs207"))
                plan = json.loads((args.root/"plan.json").read_text())
                self.assertEqual(plan["labels"], list(suite.SYSTEMS))
                self.assertFalse((args.root/"gpu").exists())
                draft = json.loads((args.root/"peptide/draft-manifest.json").read_text())
                self.assertEqual(draft["gpu_helper_pin"], "gpu-helper")
                self.assertEqual(draft["gpu_manifest_sha256"], suite.digest(producer/"manifest.json"))
                (old_source/"devtools/disp0/phoenix_validation/fragment_gate.py").write_text("different")
                args.root = base/"bad-plan"
                with self.assertRaisesRegex(RuntimeError, "engine"):
                    suite.prepare(args)
                self.assertFalse(args.root.exists())


if __name__ == "__main__":
    unittest.main()
