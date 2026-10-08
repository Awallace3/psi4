from pathlib import Path
import tempfile
import unittest

from mgga_pw6b95 import CASES, PROTOCOLS, history


class PW6B95Tests(unittest.TestCase):
    def test_endpoint_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "psi4.out"
            path.write_text("header\n"
                            " @DF-RKS iter 199: -76.5 -1.30029e-10 6.94962e-07 DIIS\n"
                            " @DF-RKS iter 200: -76.5 -4.45795e-11 2.14127e-07 DIIS\n"
                            "Failed to converge.\n")
            h = history(path)
            self.assertEqual(h["count"], 2)
            self.assertEqual(h["last"]["iteration"], 200)
            self.assertEqual(h["last"]["output_line"], 3)
            self.assertEqual(h["min_residual_last20"], 2.14127e-7)

    def test_missing_history(self):
        self.assertEqual(history(Path("/nonexistent/psi4.out"))["count"], 0)

    def test_no_relaxed_thresholds(self):
        for config in PROTOCOLS.values():
            self.assertFalse({"e_convergence", "d_convergence", "fail_on_maxiter"} & config.keys())
        self.assertEqual(len(CASES), 6)
        self.assertEqual(PROTOCOLS["timeout-only"], PROTOCOLS["baseline"])


if __name__ == "__main__":
    unittest.main()
