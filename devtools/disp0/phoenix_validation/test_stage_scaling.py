import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from stage_scaling import (configuration, traced_command, require_producer, submit,
                           SCIENCE_PIN, digest, cpu_scratch_parent)


class StageScalingTests(unittest.TestCase):
    def base(self):
        return {"source_pin": SCIENCE_PIN, "device_family": "H200", "threads": 8,
                "identity_contract": "process-owned-kernel-sealed-snapshots-v1",
                "stage": "/unchanged/stage", "immutable": {"native": "original"}}

    def test_selected_sizes_and_case_memory_policy_without_science_change(self):
        for case, native in (("peptide", 12), ("nanotube", 12), ("protein83", 12)):
            base = self.base()
            gpu = configuration(base, Path("/new/aliases"), case, "gpu")
            self.assertEqual(gpu["native_persistent_bytes"], native*1024**3)
            self.assertEqual(gpu["stage"], base["stage"])
            self.assertEqual(gpu["immutable"], base["immutable"])
            self.assertNotIn("host_memory", base)
            cpu = configuration(base, Path("/new/aliases"), case, "cpu")
            self.assertEqual({k:v for k,v in cpu.items() if k != "threads"},
                             {k:v for k,v in gpu.items() if k != "threads"})
            self.assertEqual((gpu["threads"], cpu["threads"]), (8, 24))
        with self.assertRaises(ValueError):
            configuration(self.base(), Path("/aliases"), "protein157", "gpu")

    def test_trace_wraps_only_authenticated_entry_and_retains_original_args(self):
        argv = ["/bin/bash", "-c", "script", "label", "/stage/psi4", "/env/python",
                "/science/sealed_bootstrap.py", "--config", "/original/config", "--scaled"]
        wrapped = traced_command(argv, Path("/new/tracer.py"), Path("/new/trace.json"))
        self.assertEqual(wrapped[:6], argv[:6])
        self.assertEqual(wrapped[6:11], ["/new/tracer.py", "--bootstrap",
                         "/science/sealed_bootstrap.py", "--trace-output", "/new/trace.json"])
        self.assertEqual(wrapped[11], "--include-localization")
        self.assertEqual(wrapped[12:], argv[7:])
        self.assertEqual(argv[6], "/science/sealed_bootstrap.py")
        with self.assertRaises(ValueError):
            traced_command([*argv[:6], "/other/worker.py", *argv[7:]],
                           Path("/tracer"), Path("/trace"))

    def test_cpu_cannot_consume_partial_preempted_or_mismatched_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proof = {"status": "PASS_GPU_REPEATS_AND_TRACE_AWAITING_CPU",
                     "source_pin": SCIENCE_PIN, "kind": "gpu",
                     "scratch_cleanup": {"complete": True}, "system": "nanotube",
                     "gpu_config_sha256": "abc", "samples": [],
                     "timing": {"n": 3, "median_seconds": 2.,
                                "minimum_seconds": 1., "maximum_seconds": 3.}}
            for i in range(3):
                path = root/f"nanotube-r{i}"
                path.mkdir()
                doc = {"source_pin": SCIENCE_PIN, "arm": "gpu", "system": "nanotube",
                       "runtime_config_sha256": "abc", "benchmark": True,
                       "status": "EXECUTED_REQUIRES_INDEPENDENT_COMPARISON",
                       "benchmark_whole_energy_seconds": float(i+1), "pid": 100+i}
                (path/"result.json").write_text(json.dumps(doc))
                proof["samples"].append({"path": str(path),
                    "result_sha256": digest(path/"result.json"), "energy_seconds": float(i+1)})
            (root/"job-status.env").write_text("exit_code=0\nterm_received=0\n")
            (root/"stage-result.json").write_text(json.dumps(proof))
            self.assertEqual(require_producer(root, "nanotube", "abc"), proof)
            with self.assertRaises(RuntimeError):
                require_producer(root, "peptide", "abc")
            proof["samples"][0]["energy_seconds"] = 999
            (root/"stage-result.json").write_text(json.dumps(proof))
            with self.assertRaisesRegex(RuntimeError, "timing changed"):
                require_producer(root, "nanotube", "abc")
            proof["samples"][0]["energy_seconds"] = 1.
            proof["scratch_cleanup"]["complete"] = False
            (root/"stage-result.json").write_text(json.dumps(proof))
            with self.assertRaises(RuntimeError):
                require_producer(root, "nanotube", "abc")
            (root/"job-status.env").write_text("exit_code=143\nterm_received=1\n")
            with self.assertRaises(RuntimeError):
                require_producer(root, "nanotube", "abc")

    def test_known_over_eight_hour_cpu_and_excluded_protein_are_not_submitted(self):
        with self.assertRaisesRegex(ValueError, "exceeded 8h"):
            submit(SimpleNamespace(kind="cpu", system="protein83"))
        with self.assertRaisesRegex(ValueError, "excluded"):
            submit(SimpleNamespace(kind="gpu", system="protein157"))

    def test_cpu_uses_scheduler_writable_parent_not_site_owned_job_directory(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"TMPDIR": tmp}, clear=True):
            self.assertEqual(cpu_scratch_parent(), Path(tmp).resolve())
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"SLURM_TMPDIR": tmp}, clear=True):
            self.assertEqual(cpu_scratch_parent(), Path(tmp).resolve())
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Scheduler-provided"):
                cpu_scratch_parent()


if __name__ == "__main__":
    unittest.main()
