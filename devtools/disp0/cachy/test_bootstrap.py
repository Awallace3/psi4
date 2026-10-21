"""Local transport tests only: no conda, SSH, installation or device calls."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
