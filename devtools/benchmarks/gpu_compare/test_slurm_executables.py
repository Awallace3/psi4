"""Orchestration must find SLURM outside a deliberately restricted worker PATH."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


class SlurmPaths(unittest.TestCase):
    def test_absolute_override_reaches_qos_gate_with_restricted_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = root / "tools"
            tools.mkdir()
            slurm = root / "slurm"
            slurm.mkdir()
            trace = root / "queried"
            git = tools / "git"
            git.write_text('#!/bin/bash\n'
                           'case "${3:-}" in rev-parse) echo pin;; status) :;; *) exit 2;; esac\n')
            git.chmod(0o755)
            squeue = slurm / "squeue"
            squeue.write_text('#!/bin/bash\nprintf queried > "$TRACE"\necho embers\n')
            squeue.chmod(0o755)
            build = root / "core"
            build.mkdir()
            (build / "build-manifest.json").write_text("{}")
            env = dict(os.environ, PATH=f"{tools}:/usr/bin:/bin",
                       GPU_COMPARE_SLURM_BIN=str(slurm), SLURM_JOB_ID="1",
                       SLURM_CPUS_PER_TASK="8", TRACE=str(trace),
                       GPU_COMPARE_CUDA12_RECEIPT=str(root / "unused-receipt"),
                       GPU_COMPARE_CUDA12_RECEIPT_SHA256="0" * 64)
            cases = [
                ("gpu_compare/build_grid_export.sh", [str(root), "pin", str(root / "newbuild"),
                                                      "embers", str(root / "absent-sdk"), str(root)]),
                ("gpu_compare/strict_setup_job.sh", [str(root), "pin", str(root / "setup")]),
                ("cuest_direct/compare_job.sh", [str(root), "pin", str(build), str(root / "compare"),
                                                 str(root), "1", "H200", "preflight", "embers"]),
                ("cuest_direct/run_validation.sh", [str(root), "pin", str(build),
                                                     str(root / "validation"), "embers"]),
                ("cuest_direct/recovery_job.sh", [str(root), "pin", str(build), str(root / "recovery"),
                                                  str(root), "1", "H200", "embers", "gates"]),
            ]
            for script, args in cases:
                with self.subTest(script=script):
                    trace.unlink(missing_ok=True)
                    result = subprocess.run(
                        ["/bin/bash", str(ROOT / "devtools/benchmarks" / script), *args],
                        env=env, capture_output=True, text=True, timeout=10)
                    # Fake roots cannot launch chemistry. Only the actual QoS
                    # call is exercised; later setup must stop nonzero.
                    self.assertNotEqual(result.returncode, 0)
                    self.assertTrue(trace.exists(), result.stderr)
                    self.assertEqual(trace.read_text(), "queried")

    def test_setup_uses_absolute_srun_for_sanitized_environment(self):
        source = (ROOT / "devtools/benchmarks/gpu_compare/strict_setup_job.sh").read_text()
        self.assertIn('"$slurm_bin/srun"', source)


if __name__ == "__main__":
    unittest.main()
