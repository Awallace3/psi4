"""Execute actual setup supervisor functions against CPU-only SLURM stubs."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import venv

ROOT = Path(__file__).resolve().parents[3]
SETUP = ROOT / "devtools/benchmarks/gpu_compare/strict_setup_job.sh"


class SetupLifecycle(unittest.TestCase):
    def run_case(self, body):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            srun = root / "srun"
            srun.write_text('#!/bin/bash\nshift\nexec "$@"\n')
            srun.chmod(0o755)
            source = SETUP.read_text()
            supervisor = source.split("terminated=0\n", 1)[1].split(
                "source /etc/profile.d/z00_lmod_pace.sh", 1)[0]
            script = root / "test.sh"
            script.write_text('set -euo pipefail\nrun=$1\nslurm_bin=$1\nterminated=0\n'
                              + supervisor + "\n" + body)
            result = subprocess.run(["/bin/bash", str(script), str(root)],
                                    capture_output=True, text=True, timeout=10)
            return result, {p.name for p in root.iterdir()}

    def test_cancel_before_first_step_launches_nothing(self):
        r, files = self.run_case('on_term\nrun_step /usr/bin/touch "$run/should-not-exist"\n')
        self.assertEqual(r.returncode, 143, r.stderr)
        self.assertNotIn("should-not-exist", files)

    def test_cancel_between_steps_launches_nothing_more(self):
        r, files = self.run_case(
            'run_step /usr/bin/touch "$run/first"\n'
            'on_term\nrun_step /usr/bin/touch "$run/second"\n')
        self.assertEqual(r.returncode, 143, r.stderr)
        self.assertIn("first", files)
        self.assertNotIn("second", files)

    def test_cancel_during_launch_assignment_race_is_forwarded(self):
        r, _ = self.run_case(
            "set -T\n"
            "trap 'if [[ \"$BASH_COMMAND\" == '\\''child=$!'\\'' ]]; then "
            "trap - DEBUG; on_term; fi' DEBUG\n"
            "run_step /bin/sleep 60\n")
        self.assertEqual(r.returncode, 143, r.stderr)

    def test_late_cancel_cannot_leave_success_receipt(self):
        r, files = self.run_case(
            'run_step /usr/bin/touch "$run/SETUP_COMPLETE.json"\non_term\nexit 0\n')
        self.assertEqual(r.returncode, 143, r.stderr)
        self.assertNotIn("SETUP_COMPLETE.json", files)
        self.assertIn("SETUP_INTERRUPTED.json", files)

    def test_unsignalled_success_retains_receipt(self):
        r, files = self.run_case('run_step /usr/bin/touch "$run/SETUP_COMPLETE.json"\n')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("SETUP_COMPLETE.json", files)
        self.assertNotIn("SETUP_INTERRUPTED.json", files)

    def test_payload_failure_is_not_success(self):
        r, files = self.run_case('run_step /bin/bash -c "exit 7"\n')
        self.assertEqual(r.returncode, 7, r.stderr)
        self.assertNotIn("SETUP_COMPLETE.json", files)

    def test_capture_startup_does_not_execute_environment_pth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = root / "venv"
            venv.EnvBuilder(with_pip=False, symlinks=True).create(env)
            site = env / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
            sentinel = root / "hook-ran"
            (site / "hook.pth").write_text(
                f"import pathlib; pathlib.Path({str(sentinel)!r}).write_text('ran')\n")
            capture = ROOT / "devtools/benchmarks/gpu_compare/capture_cuda12.py"
            # First establish that this environment's hook really executes.
            r = subprocess.run([str(env / "bin/python"), "-c", "pass"],
                               capture_output=True, text=True, timeout=10)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(sentinel.exists())
            sentinel.unlink()
            r = subprocess.run([str(env / "bin/python"), "-S", "-B", str(capture), "--help"],
                               capture_output=True, text=True, timeout=10)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertFalse(sentinel.exists())
            self.assertIn('"$gpu_env/bin/python" -S -B', SETUP.read_text())
            self.assertIn("PYTHONDONTWRITEBYTECODE=1", SETUP.read_text())


if __name__ == "__main__":
    unittest.main()
