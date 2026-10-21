"""Offline wrapper/algebra guards, not CUDA numerical validation."""
import ast
from pathlib import Path
import types
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "psi4/driver/procrouting/sapt/sapt_mp2_terms.py"


class ContractTests(unittest.TestCase):
    def helper(self, *, enabled=True, opted_in=True, endpoint=None):
        core = types.SimpleNamespace(get_global_option=lambda key: {
            "USE_CUEST": enabled, "CUEST_FDDS_PINV": opted_in, "CUEST_GEMM_MIN_DIM": 0}[key])
        if endpoint is not None:
            core._cuest_fdds_pinv = endpoint
        module = ast.parse(DRIVER.read_text())
        function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "_fdds_pinv")
        namespace = {"np": np, "core": core}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(DRIVER), "exec"), namespace)
        return namespace["_fdds_pinv"]

    def test_disabled_optout_and_absent_endpoint_retain_numpy(self):
        def forbidden(*args):
            self.fail("GPU endpoint must not run")
        a = np.array([[1., 2.], [3., 7.]])
        for helper in (self.helper(enabled=False, endpoint=forbidden),
                       self.helper(opted_in=False, endpoint=forbidden), self.helper()):
            np.testing.assert_array_equal(helper(a), np.linalg.pinv(a, rcond=1.e-13))

    def test_preflight_none_retains_numpy(self):
        a = np.arange(12.).reshape(3, 4)
        np.testing.assert_array_equal(self.helper(endpoint=lambda *args: None)(a),
                                      np.linalg.pinv(a, rcond=1.e-13))

    def test_endpoint_errors_propagate_without_retry(self):
        def failed(*args):
            raise RuntimeError("solver failure")
        with self.assertRaisesRegex(RuntimeError, "solver failure"):
            self.helper(endpoint=failed)(np.eye(3))

    def test_general_row_major_svd_reconstruction(self):
        rng = np.random.default_rng(119)
        for a in (rng.normal(size=(7, 7)), np.outer(rng.normal(size=7), rng.normal(size=7)),
                  np.diag([1., 2.e-13, 1.e-13, 5.e-14, 0.])):
            u, s, vt = np.linalg.svd(a.T, full_matrices=True)
            inverse = np.zeros_like(s)
            keep = s > 1.e-13 * s[0]
            inverse[keep] = 1. / s[keep]
            # Ddgmm(LEFT, VT, inverse) then Dgemm(T,T), raw D2H row-major.
            device_output = (inverse[:, None] * vt).T @ u.T
            np.testing.assert_allclose(device_output.T, np.linalg.pinv(a, rcond=1.e-13),
                                       rtol=2.e-11, atol=5.e-12)

    def test_four_driver_sites_keep_original_cutoff(self):
        calls = [n for n in ast.walk(ast.parse(DRIVER.read_text())) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "_fdds_pinv"]
        self.assertEqual(len(calls), 4)
        for call in calls:
            self.assertEqual(len(call.args), 1)
            self.assertEqual([(k.arg, ast.literal_eval(k.value)) for k in call.keywords],
                             [("rcond", 1.e-13)])


if __name__ == "__main__":
    unittest.main()
