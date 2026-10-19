"""Real interpreter-exit regression; fake endpoint only, never imports Psi4."""
import hashlib
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

import config as s
import controller


CHILD = r'''
import atexit
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace
from unittest.mock import patch
import config as s
import execute

root, label, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
uuid = "GPU-synthetic"
output = root/"raw/sessions"/label
manifest = root/"manifest.json"
mh = s.sha(manifest)
row = next(v for v in s.SEQUENCE if v["label"] == label)
early = dict(pid=os.getpid(), manifest_sha256=mh, environment={})
def begin(*_):
    s.durable(root/"raw"/(label+".effective-env.json"), early)
    return early
def runner(*_, **__):
    output.mkdir()
    (output/"timer.dat").write_text("early pre-energy timer\n")
    (output/"psi4.out").write_text("synthetic endpoint\n")
    # Use the production PRINT3 schema (global is a Python keyword).
    result = output/"result.json"
    result.write_text('{"ok": true, "trace_print_options": {"global": 3, "scf": 3, "sapt": 3}}\n')
    def at_exit():
        with (output/"timer.dat").open("a") as stream:
            stream.write("final energy timer appended at exit\n")
        if mode != "missing":
            (output/"profile.txt").write_text("profile created only at exit\n")
    atexit.register(at_exit)
    raise SystemExit(0)
runtime = SimpleNamespace(Cuda=lambda _:SimpleNamespace(synchronize=lambda:None, maps={}),
                          validate_maps=lambda _: {})
probe = SimpleNamespace(main=lambda:s.durable(root/"raw"/(label+".before.json"),
    dict(core_sha256="synthetic-core", pid=os.getpid())))
repeatability = SimpleNamespace(ALLOWLIST={row["arm"]:dict(core_sha256="synthetic-core")})
with patch.object(s, "RUN", str(root)), patch.object(s, "manifest",
        return_value=dict(native_libraries={}, suite="synthetic",
            arms={row["arm"]:dict(pin="synthetic")})), \
     patch.object(s, "inherited", return_value=runtime), \
     patch.object(execute, "early", side_effect=begin), \
     patch.object(execute, "exclusive_pid", return_value=dict(pid=os.getpid(),uuid=uuid)), \
     patch.object(execute.runpy, "run_path", side_effect=runner), \
     patch.dict(sys.modules, probe=probe, repeatability=repeatability), \
     patch.dict(os.environ, SLURM_JOB_ID="synthetic-job"):
    execute.execute(str(manifest), mh, label, uuid)
assert not (root/"raw"/(label+".COMPLETE.json")).exists(), "child must not claim completion"
assert (root/"raw"/(label+".ENDPOINT-PRE-EXIT.json")).is_file()
assert not (output/"profile.txt").exists(), "profile must be exit-only"
assert (output/"timer.dat").read_text() == "early pre-energy timer\n"
if mode == "nonzero":
    raise SystemExit(7)
if mode == "signal":
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGTERM)
'''


class ExitLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"])
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/"raw/sessions").mkdir(parents=True)
        s.durable(self.root/"manifest.json", {})
        self.row = s.SEQUENCE[0]
        self.label = self.row["label"]
        self.out = self.root/"raw/sessions"/self.label
        self.engine = controller.Engine.__new__(controller.Engine)
        self.engine.raw = self.root/"raw"
        self.engine.scratch = s.HERE
        self.engine.env = os.environ.copy()
        self.engine.signalled = False
        self.engine.active = None
        self.engine.end = __import__("time").time()+4000
        self.engine.job = "synthetic-job"
        self.engine.gpu_selector = "GPU-synthetic"
        self.engine.manifest_hash = s.sha(self.root/"manifest.json")

    def launch(self, mode="success"):
        try:
            return self.engine.run_command(
                [sys.executable, "-B", "-c", CHILD, str(self.root), self.label, mode],
                self.label, timeout=5)
        except BaseException:
            print((self.engine.raw/(self.label+".log")).read_text(), flush=True)
            raise

    def complete(self, pid):
        return self.engine.complete_case(self.row, pid)

    def test_real_atexit_parent_hashes_final_bytes(self):
        pid = self.launch()
        self.assertFalse((self.engine.raw/(self.label+".COMPLETE.json")).exists())
        self.complete(pid)
        receipt = s.load(self.engine.raw/(self.label+".COMPLETE.json"))
        self.assertEqual(receipt["child_exit"]["pid"], pid)
        self.assertEqual(receipt["child_exit"]["exit"], 0)
        self.assertEqual(receipt["endpoint_evidence"]["pid"], pid)
        self.assertEqual(receipt["endpoint_sha256"],
            s.sha(self.engine.raw/(self.label+".ENDPOINT-PRE-EXIT.json")))
        self.assertNotIn("output_hashes", receipt["endpoint_evidence"])
        self.assertEqual((self.out/"timer.dat").read_text(),
            "early pre-energy timer\nfinal energy timer appended at exit\n")
        self.assertEqual((self.out/"profile.txt").read_text(), "profile created only at exit\n")
        self.assertNotEqual(receipt["output_hashes"]["timer.dat"],
            hashlib.sha256(b"early pre-energy timer\n").hexdigest())
        for name, digest in receipt["output_hashes"].items():
            self.assertEqual(digest, s.sha(self.out/name))
            self.assertEqual((self.out/name).stat().st_mode & 0o222, 0)
        self.assertEqual(set(receipt["output_hashes"]),
            {"result.json", "psi4.out", "timer.dat", "profile.txt"})
        with self.assertRaises(FileExistsError):
            self.complete(pid)

    def test_nonzero_after_endpoint_never_completes(self):
        with self.assertRaises(RuntimeError):
            self.launch("nonzero")
        exit_proof = s.load(self.engine.raw/(self.label+".exit.json"))
        self.assertEqual(exit_proof["exit"], 7)
        self.assertTrue((self.out/"profile.txt").exists())  # atexit ran despite nonzero
        with self.assertRaises(RuntimeError):
            self.complete(exit_proof["pid"])
        self.assertFalse((self.engine.raw/(self.label+".COMPLETE.json")).exists())

    def test_signal_after_endpoint_never_completes(self):
        with self.assertRaises(RuntimeError):
            self.launch("signal")
        exit_proof = s.load(self.engine.raw/(self.label+".exit.json"))
        self.assertEqual(exit_proof["exit"], -signal.SIGTERM)
        self.assertFalse((self.out/"profile.txt").exists())
        with self.assertRaises(RuntimeError):
            self.complete(exit_proof["pid"])
        self.assertFalse((self.engine.raw/(self.label+".COMPLETE.json")).exists())

    def test_missing_exit_artifact_never_completes(self):
        pid = self.launch("missing")
        with self.assertRaises(RuntimeError):
            self.complete(pid)
        self.assertFalse((self.engine.raw/(self.label+".COMPLETE.json")).exists())

    def test_wrong_actual_pid_never_completes(self):
        pid = self.launch()
        with self.assertRaises(RuntimeError):
            self.complete(pid+1)
        self.assertFalse((self.engine.raw/(self.label+".COMPLETE.json")).exists())

    def test_symlink_output_never_completes(self):
        pid = self.launch()
        (self.out/"profile.txt").rename(self.root/"escaped")
        (self.out/"profile.txt").symlink_to(self.root/"escaped")
        with self.assertRaises(RuntimeError):
            self.complete(pid)
        self.assertFalse((self.engine.raw/(self.label+".COMPLETE.json")).exists())

    def test_output_and_record_caps_before_completion(self):
        pid = self.launch()
        for key in ("output_cap_bytes", "record_cap_bytes"):
            with self.subTest(cap=key), patch.dict(s.POLICY, {key:1}):
                with self.assertRaises(RuntimeError):
                    self.complete(pid)
                self.assertFalse((self.engine.raw/(self.label+".COMPLETE.json")).exists())

    def test_symlink_session_directory_never_completes(self):
        pid = self.launch()
        self.out.rename(self.root/"escaped-session")
        self.out.symlink_to(self.root/"escaped-session", target_is_directory=True)
        with self.assertRaises(RuntimeError):
            self.complete(pid)
        self.assertFalse((self.engine.raw/(self.label+".COMPLETE.json")).exists())

    def test_parent_fsyncs_final_files_before_receipt(self):
        pid = self.launch()
        real_fsync = os.fsync
        synced = []
        def record_fsync(fd):
            synced.append(Path(os.readlink("/proc/self/fd/"+str(fd))))
            return real_fsync(fd)
        real_durable = s.durable
        def record_durable(path, value):
            if str(path).endswith(".COMPLETE.json"):
                self.assertTrue({self.out/name for name in
                    ("result.json","psi4.out","timer.dat","profile.txt")} <= set(synced))
                self.assertIn(self.out, synced)
            return real_durable(path, value)
        with patch.object(os, "fsync", side_effect=record_fsync), \
             patch.object(s, "durable", side_effect=record_durable):
            self.complete(pid)


if __name__ == "__main__":
    unittest.main()
