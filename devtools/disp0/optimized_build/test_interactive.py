"""Mock allocation protocol only. Fixture IDs are NEVER accepted live jobs."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import interactive_acquire as module

JOB = "987654"
FIELDS = (f"JobId={JOB} Account=gts-cs207-chemx QOS=embers JobState=RUNNING Requeue=0 "
          "Restarts=0 NumCPUs=8 NumTasks=1 TimeLimit=02:00:00 Partition=cpu-small "
          "MinMemoryNode=48G BatchFlag=0 NodeList=fixture-node ReqTRES=cpu=8,mem=48G,node=1")


class InteractiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.attempt = self.root / "interactive-attempt"
        self.lines = f"salloc: Pending job allocation {JOB}\nsalloc: Granted job allocation {JOB}\n"
        self.status = 0
        self.controller = FIELDS
        self.rpcs = 0
        self.queries = 0

    def preflight(self, *args, **kwargs):
        if self.attempt.exists():
            raise RuntimeError("NEVER replay")
        self.attempt.mkdir()
        module.admission.record(self.attempt, "intent.json", {"fixture": True})
        self.assertTrue(kwargs["interactive"])
        return None, {"fixture": True}, {"FIXTURE_ONLY": "1"}

    def popen(self, argv, **kwargs):
        self.rpcs += 1
        self.assertTrue((self.attempt / "dispatch-intent.json").exists())
        for flag in ("--no-shell", "--immediate=600", "--qos=embers", "--cpus-per-task=8", "--mem=48G"):
            self.assertIn(flag, argv)
        self.assertNotIn("--no-requeue", argv)  # not a valid salloc option
        self.assertEqual(kwargs["env"], {"FIXTURE_ONLY": "1"})
        class Process:
            stdout = io.StringIO(self.lines)
            def wait(inner):
                return self.status
        return Process()

    def capture(self, argv, **kwargs):
        self.queries += 1
        self.assertEqual((self.attempt / "accepted-job-id.txt").read_text(), JOB + "\n")
        self.assertIn("Granted", (self.attempt / "salloc.raw").read_text())
        return {"argv": argv, "exit": 0, "stdout": self.controller, "stderr": ""}

    def invoke(self):
        with patch.object(module.admission, "preflight", self.preflight), patch.object(
                module.subprocess, "Popen", self.popen), patch.object(
                module.admission, "capture", self.capture), contextlib.redirect_stdout(io.StringIO()):
            module.acquire(self.root / "source", "a" * 40, self.attempt, self.root / "quota", "b" * 64)

    def test_one_acquisition_and_durable_id_before_controller(self):
        self.invoke()
        self.assertEqual(self.rpcs, 1)
        self.assertEqual(self.queries, 1)
        self.assertTrue((self.attempt / "allocation.json").exists())

    def test_no_second_acquisition_from_existing_attempt(self):
        self.invoke()
        with self.assertRaisesRegex(RuntimeError, "NEVER replay"):
            self.invoke()
        self.assertEqual(self.rpcs, 1)

    def test_pending_then_failure_preserves_id_no_go_or_query(self):
        self.lines = f"salloc: Pending job allocation {JOB}\n"
        self.status = 1
        with self.assertRaisesRegex(RuntimeError, "No verified grant"):
            self.invoke()
        self.assertEqual(self.queries, 0)
        self.assertFalse((self.attempt / "allocation.json").exists())
        self.assertEqual((self.attempt / "accepted-job-id.txt").read_text(), JOB + "\n")

    def test_wrong_effective_resources_no_go(self):
        self.controller = FIELDS.replace("QOS=embers", "QOS=inferno")
        with self.assertRaisesRegex(RuntimeError, "Wrong effective"):
            self.invoke()
        self.assertFalse((self.attempt / "allocation.json").exists())

    def test_gpu_or_requeue_not_admitted(self):
        for text in (FIELDS + ",gres/gpu=1", FIELDS.replace("Requeue=0", "Requeue=1")):
            with self.assertRaises(RuntimeError):
                module.validate_controller(text, JOB)

    def test_unparseable_reply_is_unknown_not_absence(self):
        self.lines = "unexpected response\n"
        with self.assertRaisesRegex(RuntimeError, "NEVER replay"):
            self.invoke()
        self.assertEqual(self.rpcs, 1)
        self.assertEqual(self.queries, 0)
        self.assertTrue((self.attempt / "salloc.raw").exists())


if __name__ == "__main__":
    unittest.main()
