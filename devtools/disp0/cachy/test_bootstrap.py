"""Local transport tests only: no conda, SSH, installation or device calls."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).parent))
from finish_env import freeze

spec = importlib.util.spec_from_file_location("bootstrap", Path(__file__).with_name("bootstrap_env.py"))
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class TransportTests(unittest.TestCase):
    def exercise(self, code, timeout):
        folder = Path(self.directory.name)
        bootstrap.run([sys.executable, "-B", "-c", code], folder, "probe", dict(os.environ), timeout)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def result(self):
        return json.loads((Path(self.directory.name) / "probe.result.json").read_text())

    def test_success(self):
        self.exercise("print('ok')", 5)
        self.assertEqual(self.result()["exit"], 0)

    def test_failure_preserved(self):
        with self.assertRaises(RuntimeError):
            self.exercise("raise SystemExit(7)", 5)
        self.assertEqual(self.result()["exit"], 7)

    def test_timeout_preserved(self):
        with self.assertRaises(RuntimeError):
            self.exercise("import time; time.sleep(30)", 0.05)
        self.assertEqual(self.result()["exit"], 124)


class FrozenProposalTests(unittest.TestCase):
    def record(self, name, version="1"):
        return dict(name=name, version=version, build="0",
                    url=f"https://conda.anaconda.org/conda-forge/linux-64/{name}-{version}-0.conda",
                    sha256="a" * 64, md5="b" * 32)

    def test_exact_replacement_and_removal(self):
        old, removed, new = self.record("x"), self.record("y"), self.record("x", "2")
        lock, expected = freeze([old, removed], {"UNLINK": [old, removed], "LINK": [new]}, [])
        self.assertEqual(set(expected), {"x"})
        self.assertIn("x-2-0.conda#" + "b" * 32, lock)
        self.assertNotIn("x-1-0", lock)

    def test_missing_hash_rejected(self):
        row = self.record("x")
        del row["md5"]
        with self.assertRaises(RuntimeError):
            freeze([], {"LINK": [row]}, [])

    def test_foreign_origin_rejected(self):
        row = self.record("x")
        row["url"] = "https://example.invalid/x.conda"
        with self.assertRaises(RuntimeError):
            freeze([row], {}, [])


if __name__ == "__main__":
    unittest.main()
