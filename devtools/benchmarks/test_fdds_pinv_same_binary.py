import copy
import unittest

from fdds_pinv_same_binary import validate


class SameBinaryControls(unittest.TestCase):
    def setUp(self):
        self.reference = dict(
            ok=True, arm="gpu", system="protein83", natom=83, nbf=922, naux=3820,
            threads=8, memory="96 GiB", options={}, cache_sha256="cache",
            metadata_sha256="metadata", runner_sha256="runner", numpy_version="numpy",
            numpy_configuration="config", job_id="14250361", hostname="node",
            occupied=[1, 2], virtual=[3, 4], binary_pin="source", core_sha256="core",
            libraries_sha256={"solver": "sha"}, gpu_pinv_requested=True,
            gpu_timer_calls=20, gpu_exchange_timer_calls=4, gpu_metric_timer_calls=12,
            gpu_response_timer_calls=60, gpu_pinv_timer_calls=22,
            pinv_calls=[{}] * 22, fdds_seconds=1., energies_Eh={"coupled": -1.})

    def test_both_routes(self):
        self.assertEqual(validate(self.reference, self.reference, True), {"coupled": 0.})
        sample = copy.deepcopy(self.reference)
        sample.update(gpu_pinv_requested=False, gpu_pinv_timer_calls=0)
        self.assertEqual(validate(sample, self.reference, False), {"coupled": 0.})

    def test_nonfinite_energies_rejected(self):
        for value in (float("nan"), float("inf")):
            sample = copy.deepcopy(self.reference)
            sample["energies_Eh"]["coupled"] = value
            with self.assertRaises(AssertionError):
                validate(sample, self.reference, True)

    def test_closure_and_science_rejected(self):
        for key in ("core_sha256", "libraries_sha256", "options", "job_id"):
            sample = copy.deepcopy(self.reference)
            sample[key] = "wrong"
            with self.assertRaises(AssertionError):
                validate(sample, self.reference, True)

    def test_energy_and_route_rejected(self):
        sample = copy.deepcopy(self.reference)
        sample["energies_Eh"]["coupled"] += 1.e-4
        with self.assertRaises(AssertionError):
            validate(sample, self.reference, True)
        with self.assertRaises(AssertionError):
            validate(self.reference, self.reference, False)


if __name__ == "__main__":
    unittest.main()
