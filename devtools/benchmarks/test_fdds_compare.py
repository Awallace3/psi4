"""Report-auditor unit fixtures only; not scientific timing or GPU evidence."""
from copy import deepcopy
import unittest

from fdds_compare import compare


class CompareTests(unittest.TestCase):
    def rows(self, repeats=1):
        common = dict(ok=True, system="peptide", cache_sha256="cache",
                      metadata_sha256="metadata", options={"quadrature": 10},
                      binary_pin="pin", core_sha256="core", runner_sha256="runner",
                      threads=8, memory="96 GiB", hostname="host", job_id="job",
                      energies_Eh={"Disp20": -.01, "Disp20,FDDS (unc)": -.02})
        return [dict(deepcopy(common), arm=arm, fdds_seconds=2. if arm == "cpu" else 1.,
                     gpu_timer_calls=0 if arm == "cpu" else 20,
                     gpu_exchange_timer_calls=0 if arm == "cpu" else 4)
                for arm in ("cpu", "gpu") for _ in range(repeats)]

    def test_first_pair(self):
        result = compare(self.rows(), 1)
        self.assertEqual(result["repeats_per_arm"], 1)
        self.assertEqual(result["cpu_over_gpu"], 2.)

    def test_full_repeats(self):
        self.assertEqual(compare(self.rows(2), 2)["repeats_per_arm"], 2)

    def test_no_speedup_is_reported_honestly(self):
        rows = self.rows()
        rows[-1]["fdds_seconds"] = 4.
        self.assertEqual(compare(rows, 1)["cpu_over_gpu"], .5)

    def test_identity_mismatch(self):
        for key in ("core_sha256", "cache_sha256", "job_id", "threads"):
            with self.subTest(key=key):
                rows = self.rows()
                rows[-1][key] = "wrong"
                with self.assertRaises(AssertionError):
                    compare(rows, 1)

    def test_missing_gpu_route(self):
        rows = self.rows()
        rows[-1]["gpu_timer_calls"] = 0
        with self.assertRaises(AssertionError):
            compare(rows, 1)

    def test_missing_exchange_route(self):
        rows = self.rows()
        rows[-1]["gpu_exchange_timer_calls"] = 0
        with self.assertRaises(AssertionError):
            compare(rows, 1)

    def test_bad_energy(self):
        for value in (float("nan"), -.001):
            with self.subTest(value=value):
                rows = self.rows()
                rows[-1]["energies_Eh"]["Disp20"] = value
                with self.assertRaises(AssertionError):
                    compare(rows, 1)

    def test_invalid_time(self):
        for value in (0., float("inf"), float("nan")):
            with self.subTest(value=value):
                rows = self.rows()
                rows[-1]["fdds_seconds"] = value
                with self.assertRaises(AssertionError):
                    compare(rows, 1)

    def test_missing_repeat(self):
        with self.assertRaises(AssertionError):
            compare(self.rows(), 2)


if __name__ == "__main__":
    unittest.main()
