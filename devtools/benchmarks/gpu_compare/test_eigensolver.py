import unittest
from types import SimpleNamespace

import numpy as np

from eigensolver import canonical_eigh, eigensolver_policy


class CanonicalTests(unittest.TestCase):
    def test_full_rank_generalized_residual_and_orthogonality(self):
        rng = np.random.default_rng(31)
        a = rng.normal(size=(12, 12))
        s = a @ a.T + np.eye(12)
        a = rng.normal(size=(12, 12))
        h = a + a.T
        e, c = canonical_eigh(h, s, 1e-7, np)
        np.testing.assert_allclose(h @ c, (s @ c) * e, atol=1e-12)
        np.testing.assert_allclose(c.T @ s @ c, np.eye(12), atol=1e-12)

    def test_absolute_cutoff_retains_only_prescribed_subspace(self):
        s = np.diag([0., 1e-8, 1e-7, 1., 2.])
        h = np.diag([1., 2., 3., 4., 5.])
        e, c = canonical_eigh(h, s, 1e-7, np)
        np.testing.assert_allclose(e, [2.5, 4.])
        self.assertEqual(c.shape, (5, 2))
        np.testing.assert_allclose(c.T @ s @ c, np.eye(2), atol=1e-14)

    def test_no_input_mutation_and_rotated_truncated_problem(self):
        rng = np.random.default_rng(5)
        q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        s = q @ np.diag([1e-9, 1e-8, 1., 2., 4.]) @ q.T
        a = rng.normal(size=(5, 5))
        h = a + a.T
        before = (h.copy(), s.copy())
        e, c = canonical_eigh(h, s, 1e-7, np)
        self.assertEqual(e.size, 3)
        np.testing.assert_allclose(c.T @ h @ c, np.diag(e), atol=1e-12)
        np.testing.assert_allclose(c.T @ s @ c, np.eye(3), atol=1e-12)
        np.testing.assert_array_equal(h, before[0])
        np.testing.assert_array_equal(s, before[1])

    def test_invalid_cutoff_shapes_and_empty_rank_rejected(self):
        for value in (None, 0., -1., float("nan")):
            with self.assertRaises((ValueError, TypeError)):
                canonical_eigh(np.eye(2), np.eye(2), value, np)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            canonical_eigh(np.eye(2), np.eye(3), 1e-7, np)
        with self.assertRaisesRegex(ValueError, "No retained"):
            canonical_eigh(np.eye(2), np.zeros((2, 2)), 1e-7, np)

    def test_native_is_unmodified_and_unknown_mode_fails(self):
        original = object()
        mf, record = SimpleNamespace(eig=original), {}
        with eigensolver_policy(mf, {}, record):
            self.assertIs(mf.eig, original)
        self.assertEqual(record["eigensolver"], dict(mode="native", modified=False))
        with self.assertRaisesRegex(ValueError, "Unknown"):
            with eigensolver_policy(mf, dict(pyscf_eigensolver="silent-fallback"), {}):
                pass
        with self.assertRaisesRegex(ValueError, "cutoff"):
            with eigensolver_policy(mf, dict(pyscf_eigensolver="cupy-canonical"), {}):
                pass


if __name__ == "__main__":
    unittest.main()
