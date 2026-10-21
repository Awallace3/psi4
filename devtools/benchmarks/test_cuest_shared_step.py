"""Controller-only tests; no SLURM allocation or GPU correctness evidence."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class SharedStepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "allocation.json"
        self.data = dict(status="allocated", job_id=123, gpus=1, cpus=8,
                         memory_gib=128, run_directory=str(self.root))
        self.write_manifest()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name, text in {
            "squeue": 'printf "%s\\n" "${TEST_STATE:-RUNNING}"',
            "srun": (
                'printf "%s\\n" "$@"\n'
                'for arg in "$@"; do\n'
                ' if [[ "$arg" == */steps/unit-step && "${TEST_NO_CLEANUP:-0}" == 0 ]]; then\n'
                '  touch "$arg/GPU_CLEANUP_VERIFIED"\n'
                ' fi\n'
                'done\n'
                'exit "${TEST_STEP_RC:-0}"'
            ),
        }.items():
            p = self.bin / name
            p.write_text("#!/bin/bash\n" + text + "\n")
            p.chmod(0o755)
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}")
        self.script = Path(__file__).with_name("cuest_shared_step.sh")

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.data))

    def run_step(self, *, seconds="5", **env):
        return subprocess.run(
            ["bash", str(self.script), str(self.manifest), "unit-step",
             str(self.root), seconds, "--", "true"],
            env=dict(self.env, **env), capture_output=True, text=True,
            timeout=5,
        )

    def test_success_records_exclusive_whole_allocation(self):
        p = self.run_step()
        self.assertEqual(p.returncode, 0, p.stderr)
        step = self.root / "steps/unit-step"
        args = (step / "stdout.log").read_text().splitlines()
        for flag in ("--exclusive", "--exact", "--gres=gpu:1",
                     "--cpus-per-task=8", "--mem=128G", "--jobid=123"):
            self.assertIn(flag, args)
        self.assertTrue((step / "COMMAND_EXITED_ZERO").exists())
        self.assertEqual((step / "exit-code.txt").read_text().strip(), "0")

    def test_failure_propagated_and_not_certified(self):
        p = self.run_step(TEST_STEP_RC="17")
        self.assertEqual(p.returncode, 17)
        step = self.root / "steps/unit-step"
        self.assertTrue((step / "TERMINAL").exists())
        self.assertFalse((step / "COMMAND_EXITED_ZERO").exists())

    def test_pending_manifest_rejected(self):
        self.data["status"] = "pending"
        self.write_manifest()
        self.assertNotEqual(self.run_step().returncode, 0)
        self.assertFalse((self.root / "steps").exists())

    def test_nonrunning_job_rejected(self):
        self.assertNotEqual(self.run_step(TEST_STATE="PENDING").returncode, 0)
        self.assertFalse((self.root / "steps").exists())

    def test_prior_evidence_not_overwritten(self):
        self.assertEqual(self.run_step().returncode, 0)
        self.assertNotEqual(self.run_step().returncode, 0)

    def test_excessive_time_rejected(self):
        self.assertNotEqual(self.run_step(seconds="10801").returncode, 0)
        self.assertFalse((self.root / "steps").exists())

    def test_missing_cleanup_quarantines_allocation(self):
        self.assertEqual(self.run_step(TEST_NO_CLEANUP="1").returncode, 96)
        self.assertTrue((self.root / "QUARANTINED").exists())
        self.assertFalse((self.root / "steps/unit-step/COMMAND_EXITED_ZERO").exists())
        self.assertNotEqual(self.run_step().returncode, 0)

    def test_allocation_root_guard_accepts_only_campaign_children(self):
        controller = self.script.with_name("cuest_interactive_allocate.sh").read_text()
        guard = next(line for line in controller.splitlines()
                     if '"$run"' in line and "/psi4-cuest-shared/*" in line)
        for path, expected in [
            ("/storage/project/r-cs207-0/awallace43/runs/psi4-cuest-shared/new-run", 0),
            ("/storage/project/r-cs207-0/awallace43/runs/another-campaign/new-run", 1),
        ]:
            p = subprocess.run(["bash", "-c", 'run=$1; ' + guard, "test", path],
                               capture_output=True, timeout=5)
            self.assertEqual(p.returncode, expected)

    def test_acquisition_requires_current_explicit_paid_time_ceiling(self):
        controller = self.script.with_name("cuest_interactive_allocate.sh").read_text()
        self.assertIn('${CUEST_SHARED_WALLTIME:?', controller)
        guard = next(line for line in controller.splitlines()
                     if line.startswith('[[ "$CUEST_SHARED_WALLTIME"'))
        for value, expected in [("04:00:00", 0), ("08:00:00", 0),
                                ("04:00:01", 1), ("", 1), ("24:00:00", 1)]:
            p = subprocess.run(
                ["bash", "-c", 'CUEST_SHARED_WALLTIME=$1; ' + guard, "test", value],
                capture_output=True, timeout=5)
            self.assertEqual(p.returncode, expected)
        self.assertIn('--time="$CUEST_SHARED_WALLTIME"', controller)
        self.assertNotIn('--time=08:00:00', controller)
        self.assertLess(controller.index('REQUEST_INTENT.json'),
                        controller.index('salloc --no-shell --account='))
        self.assertLess(controller.index('os.fsync(fd)'),
                        controller.index('salloc --no-shell --account='))


if __name__ == "__main__":
    unittest.main()
