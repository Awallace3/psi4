import json
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from diagnose_pyscf import PhaseTrace, bounded_process, require_tracked_diagnostic


class DiagnosticTests(unittest.TestCase):
    def test_nested_phases_are_flushed_and_pair_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phases.jsonl"
            with PhaseTrace(path) as trace:
                with self.assertRaises(ValueError):
                    with trace.phase("outer"):
                        self.assertEqual(len(path.read_text().splitlines()), 1)
                        with trace.phase("inner"):
                            raise ValueError("fixture")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([r["event"] for r in rows], ["enter", "enter", "error", "error"])
        self.assertEqual([r["depth"] for r in rows], [0, 1, 1, 0])
        self.assertEqual(rows[-1]["exception"], "ValueError")

    def test_method_hooks_restore_even_after_failure(self):
        class MF:
            def get_init_guess(self):
                raise ValueError("guess fixture")
        for name in ("get_hcore", "get_ovlp", "eig", "get_veff", "get_j", "get_k", "get_jk"):
            setattr(MF, name, lambda self: "unchanged")
        mf = MF()
        mf.grids = SimpleNamespace(build=lambda: None)
        mf._numint = SimpleNamespace(nr_rks=lambda: None)
        original = mf.get_init_guess.__func__
        with tempfile.TemporaryDirectory() as directory:
            with PhaseTrace(Path(directory) / "phases.jsonl") as trace:
                with self.assertRaises(ValueError):
                    with trace.methods(mf):
                        mf.get_init_guess()
            rows = [json.loads(line) for line in
                    (Path(directory) / "phases.jsonl").read_text().splitlines()]
        self.assertIs(mf.get_init_guess.__func__, original)
        self.assertNotIn("get_init_guess", mf.__dict__)
        self.assertEqual(rows[0]["phase"], "mf.get_init_guess")
        self.assertEqual(rows[-1]["event"], "error")

    def test_bounded_child_completes_or_is_reaped(self):
        self.assertEqual(bounded_process([sys.executable, "-c", "pass"], 5), 0)
        import subprocess
        children = []
        popen = subprocess.Popen
        def spawn(*args, **kwargs):
            child = popen(*args, **kwargs)
            children.append(child)
            return child
        with patch("diagnose_pyscf.subprocess.Popen", side_effect=spawn):
            self.assertEqual(bounded_process(
                [sys.executable, "-c", "import time; time.sleep(60)"], .1), 124)
        self.assertTrue(children)
        self.assertTrue(all(child.poll() is not None for child in children))

    def test_watchdog_emits_stack_without_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stacks.log"
            command = [
                sys.executable, "-c",
                "import faulthandler,time;"
                f"f=open({str(path)!r},'w');"
                "faulthandler.dump_traceback_later(.03,file=f);"
                "time.sleep(.15);faulthandler.cancel_dump_traceback_later();f.close()"]
            self.assertEqual(bounded_process(command, 5), 0)
            text = path.read_text()
            self.assertIn("Timeout", text)
            self.assertIn("most recent call first", text)

    def test_term_ignoring_live_child_requires_kill(self):
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / "ready"
            code = ("import signal,time;from pathlib import Path;"
                    "signal.signal(signal.SIGUSR1,signal.SIG_IGN);"
                    "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                    f"Path({str(ready)!r}).touch();time.sleep(60)")
            report = {}
            self.assertEqual(bounded_process(
                [sys.executable, "-c", code], .5, grace=.3,
                cleanup_report=report), 124)
            self.assertTrue(ready.exists(), "Fixture installed ignoring handlers")
            self.assertEqual(report["leader_returncode"], -signal.SIGKILL)
            self.assertTrue(report["cleanup_verified"])

    def test_exited_leader_still_cleans_term_ignoring_descendant_group(self):
        child = SimpleNamespace(pid=987654321, returncode=0,
                                wait=Mock(return_value=0), poll=lambda: 0)
        live = [True]
        def send(pgid, sig):
            if sig == signal.SIGKILL:
                live[0] = False
        report = {}
        with patch("diagnose_pyscf.subprocess.Popen", return_value=child), \
                patch("diagnose_pyscf.group_exists", side_effect=lambda _: live[0]), \
                patch("diagnose_pyscf.os.killpg", side_effect=send) as kill:
            self.assertEqual(bounded_process(["fixture"], 1, grace=.05,
                                             cleanup_report=report), 0)
        kill.assert_any_call(child.pid, signal.SIGTERM)
        kill.assert_any_call(child.pid, signal.SIGKILL)
        self.assertTrue(report["cleanup_verified"])

    def test_post_kill_unreaped_group_is_bounded_and_unverified(self):
        child = SimpleNamespace(pid=987654321, returncode=None, poll=lambda: None,
            wait=Mock(side_effect=subprocess.TimeoutExpired("fixture", .01)))
        report = {}
        with patch("diagnose_pyscf.subprocess.Popen", return_value=child), \
                patch("diagnose_pyscf.group_exists", return_value=True), \
                patch("diagnose_pyscf.os.kill"), patch("diagnose_pyscf.os.killpg"):
            self.assertEqual(bounded_process(["fixture"], .01, grace=.05,
                                             cleanup_report=report), 125)
        child.wait.assert_called_once_with(timeout=.01)
        self.assertFalse(report["cleanup_verified"])
        self.assertTrue(report["owner_containment_required"])

    def test_interruption_cleans_group_and_restores_term_handler(self):
        child = SimpleNamespace(pid=987654321, returncode=None,
                                wait=Mock(side_effect=KeyboardInterrupt))
        live = [True]
        child.poll = lambda: None if live[0] else child.returncode
        def send(pgid, sig):
            if sig == signal.SIGKILL:
                live[0] = False
                child.returncode = -signal.SIGKILL
        previous = signal.getsignal(signal.SIGTERM)
        report = {}
        with patch("diagnose_pyscf.subprocess.Popen", return_value=child), \
                patch("diagnose_pyscf.group_exists", side_effect=lambda _: live[0]), \
                patch("diagnose_pyscf.os.kill"), \
                patch("diagnose_pyscf.os.killpg", side_effect=send):
            with self.assertRaises(KeyboardInterrupt):
                bounded_process(["fixture"], 1, grace=.05, cleanup_report=report)
        self.assertTrue(report["cleanup_verified"])
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_launch_boundary_signal_is_deferred_then_cleans_real_child(self):
        popen = subprocess.Popen
        for signum in (signal.SIGTERM, signal.SIGINT):
            children, report = [], {}
            previous = signal.getsignal(signum)
            def spawn(*args, **kwargs):
                child = popen(*args, **kwargs)
                children.append(child)
                # Deliver an actual signal before Popen returns to the caller.
                os.kill(os.getpid(), signum)
                return child
            with patch("diagnose_pyscf.subprocess.Popen", side_effect=spawn):
                with self.assertRaises(KeyboardInterrupt):
                    bounded_process([sys.executable, "-c",
                                     "import time; time.sleep(60)"],
                                    5, cleanup_report=report)
            self.assertTrue(children)
            self.assertTrue(all(child.poll() is not None for child in children))
            self.assertTrue(report["cleanup_verified"])
            self.assertEqual(signal.getsignal(signum), previous)

    def test_untracked_or_modified_diagnostic_cannot_claim_head_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            def git(*args):
                return subprocess.check_output(["git", "-C", directory, *args],
                                               stderr=subprocess.PIPE, text=True).strip()
            git("init", "-q")
            path = repo / "diagnostic.py"
            path.write_text("# tracked\n")
            git("add", path.name)
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                "commit", "-qm", "fixture")
            commit = git("rev-parse", "HEAD")
            require_tracked_diagnostic(repo, path, commit)
            untracked = repo / "untracked.py"
            untracked.write_text(path.read_text())
            with self.assertRaisesRegex(RuntimeError, "tracked"):
                require_tracked_diagnostic(repo, untracked, commit)
            path.write_text("# modified\n")
            with self.assertRaisesRegex(RuntimeError, "differ"):
                require_tracked_diagnostic(repo, path, commit)


if __name__ == "__main__":
    unittest.main()
