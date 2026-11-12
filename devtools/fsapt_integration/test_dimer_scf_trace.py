from types import SimpleNamespace
import unittest
import numpy as np

from dimer_scf_trace import energy_observer, contraction


class EnergyTraceContracts(unittest.TestCase):
    def test_observer_preserves_single_native_call_and_return(self):
        calls, observations = [], []
        class HF:
            def compute_E(self):
                calls.append(self)
                return -12.5
        original = HF.compute_E
        wfn = HF()
        with energy_observer(SimpleNamespace(HF=HF), lambda w, e: observations.append((w, e))):
            self.assertEqual(wfn.compute_E(), -12.5)
        self.assertEqual(calls, [wfn])
        self.assertEqual(observations, [(wfn, -12.5)])
        self.assertIs(HF.compute_E, original)

    def test_observer_restores_method_on_diagnostic_failure(self):
        class HF:
            def compute_E(self):
                return 0.
        original = HF.compute_E
        def fail(*args):
            raise RuntimeError("diagnostic failure")
        with self.assertRaisesRegex(RuntimeError, "diagnostic failure"):
            with energy_observer(SimpleNamespace(HF=HF), fail):
                HF().compute_E()
        self.assertIs(HF.compute_E, original)

    def test_direct_bilinear_difference(self):
        d0 = np.array([[1., 2.], [3., 4.]])
        h0 = np.array([[5., 6.], [7., 8.]])
        d1, h1 = d0 + 0.25, h0 - 0.5
        direct = contraction(d1 - d0, h0) + contraction(d1, h1 - h0)
        self.assertEqual(direct, contraction(d1, h1) - contraction(d0, h0))


if __name__ == "__main__":
    unittest.main()
