"""Exercise the installed hf.SCF.eig API, including its x= keyword."""
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

from eigensolver import eigensolver_policy


class GPU(np.ndarray):
    @property
    def device(self):
        return SimpleNamespace(id=0)


def gpu(value):
    return np.asarray(value, dtype=np.float64).view(GPU)


class ContractTests(unittest.TestCase):
    def setup_policy(self):
        cp = SimpleNamespace(
            ndarray=GPU, float64=np.float64, count_nonzero=np.count_nonzero,
            sqrt=np.sqrt, isfinite=np.isfinite, abs=np.abs, max=np.max, eye=np.eye,
            linalg=SimpleNamespace(eigh=lambda a: tuple(gpu(x) for x in np.linalg.eigh(a))),
            cuda=SimpleNamespace(runtime=SimpleNamespace(deviceSynchronize=lambda: None)))
        original = lambda *a, **k: None
        return cp, SimpleNamespace(eig=original), original

    def test_native_x_keyword_call_is_supported_and_restored(self):
        cp, mf, original = self.setup_policy()
        record = {}
        c = dict(pyscf_eigensolver="cupy-canonical", overlap_eigenvalue_cutoff=1e-7)
        with patch.dict(sys.modules, cupy=cp):
            with eigensolver_policy(mf, c, record, profile=True):
                e, coeff = mf.eig(gpu(np.diag([2., 3.])), gpu(np.eye(2)),
                                  overwrite=True, x=gpu(np.eye(2)))
                np.testing.assert_allclose(e, [2., 3.])
                np.testing.assert_allclose(coeff.T @ coeff, np.eye(2))
                self.assertEqual(record["eigensolver"]["calls"], 1)
        self.assertIs(mf.eig, original)

    def test_host_input_refused_and_hook_restored(self):
        cp, mf, original = self.setup_policy()
        c = dict(pyscf_eigensolver="cupy-canonical", overlap_eigenvalue_cutoff=1e-7)
        with patch.dict(sys.modules, cupy=cp):
            with self.assertRaisesRegex(RuntimeError, "visible-GPU"):
                with eigensolver_policy(mf, c, {}):
                    mf.eig(np.eye(2), gpu(np.eye(2)))
        self.assertIs(mf.eig, original)

    def test_unconsumed_policy_refused(self):
        cp, mf, original = self.setup_policy()
        c = dict(pyscf_eigensolver="cupy-canonical", overlap_eigenvalue_cutoff=1e-7)
        with patch.dict(sys.modules, cupy=cp):
            with self.assertRaisesRegex(RuntimeError, "not consumed"):
                with eigensolver_policy(mf, c, {}):
                    pass
        self.assertIs(mf.eig, original)
