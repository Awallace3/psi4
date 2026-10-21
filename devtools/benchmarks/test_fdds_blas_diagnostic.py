import ast
from pathlib import Path
import unittest


class DiagnosticContract(unittest.TestCase):
    def setUp(self):
        self.source = Path(__file__).with_name("fdds_blas_diagnostic.py").read_text()
        self.tree = ast.parse(self.source)

    def test_numpy_first(self):
        self.assertLess(self.source.index("import numpy"), self.source.index("import psi4"))

    def test_no_threading_layer_or_science_change(self):
        self.assertNotIn("os.environ[", self.source.replace(
            'os.environ["SLURM_JOB_ID"]', "JOB"))
        self.assertNotIn("set_options(", self.source)
        self.assertNotIn("set_num_threads(", self.source.replace("set_num_threads(8)", "THREADS"))
        self.assertIn("rcond=1.e-13", self.source)

    def test_bound_attribution_and_identity(self):
        self.assertIn("dump_traceback_later(20, repeat=True", self.source)
        self.assertIn("cancel_dump_traceback_later()", self.source)
        self.assertIn('hashlib.sha256(core.read_bytes()).hexdigest() == expected_hash', self.source)
        self.assertIn('not_fdds_or_speedup_evidence=True', self.source)


if __name__ == "__main__":
    unittest.main()
