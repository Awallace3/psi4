import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from containment import run_contained


@unittest.skipUnless(sys.platform == "linux", "Linux subreaper required")
class ContainmentTests(unittest.TestCase):
    def test_normal_exit_and_status_retained(self):
        proof = {}
        self.assertEqual(run_contained([sys.executable, "-c", "raise SystemExit(42)"],
                                       2, proof, grace=.5), 42)
        self.assertTrue(proof["cleanup_verified"])

    def test_explicit_environment_reaches_contained_child(self):
        proof = {}
        code = "import os; assert os.environ == {'SAFE': 'pinned', 'LC_CTYPE': 'C.UTF-8'} or os.environ == {'SAFE': 'pinned'}"
        self.assertEqual(run_contained([sys.executable, "-c", code],
                                       2, proof, grace=.5, env={"SAFE": "pinned"}), 0)
        self.assertTrue(proof["cleanup_verified"])

    def test_timeout_kills_signal_resistant_process(self):
        proof = {}
        code = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"
        self.assertEqual(run_contained([sys.executable, "-c", code],
                                       .15, proof, grace=.5), 124)
        self.assertTrue(proof["cleanup_verified"])
        self.assertTrue(proof["leader_reaped"])

    def test_exited_shell_leader_detached_resistant_descendant_is_reaped(self):
        proof = {}
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / "pid"
            child = ("import os,signal,time;from pathlib import Path;"
                     "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                     f"Path({str(pidfile)!r}).write_text(str(os.getpid()));time.sleep(60)")
            parent = ("import subprocess,sys,time;"
                      f"subprocess.Popen([sys.executable,'-c',{child!r}],start_new_session=True);"
                      "time.sleep(.15)")
            self.assertEqual(run_contained([sys.executable, "-c", parent],
                                           2, proof, grace=.5), 0)
            self.assertTrue(proof["cleanup_verified"])
            self.assertGreaterEqual(len(proof["observed_pids"]), 2)
            self.assertFalse(Path(f"/proc/{pidfile.read_text()}").exists())
