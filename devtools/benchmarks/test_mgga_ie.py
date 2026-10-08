import unittest
from mgga_ie import delta, options, summarize, FRAGMENTS


class Tests(unittest.TestCase):
    def test_counterpoise_sign(self):
        self.assertAlmostEqual(delta(dict(AB=-20.1, A=-10., B=-10.)), -.1)

    def test_route_isolation(self):
        cpu, jk, xc = [options("aug-cc-pvdz", r) for r in ("cpu", "gpu-jk", "gpu-xc")]
        self.assertEqual({k for k in cpu if cpu[k] != jk[k]}, {"use_cuest"})
        self.assertEqual({k for k in jk if jk[k] != xc[k]}, {"cuest_xc"})
        self.assertFalse(xc["cuest_mixed_precision"])

    def test_errors_not_hidden_by_cancellation(self):
        results = {}
        for route in ("cpu", "gpu-jk"):
            for f, e in zip(FRAGMENTS, [-20.1, -10., -10.]):
                shift = (2e-4 if f == "AB" else 1e-4) if route != "cpu" else 0
                results[("host", "water", "m06", route, f)] = dict(ok=True, nbf=10, energy_hartree=e+shift)
        rows = summarize(results)
        row = next(r for r in rows if r["complete"])
        self.assertLess(abs(row["delta_ie_hartree"]), 1e-12)
        self.assertGreater(row["max_total_error_hartree"], 1e-4)
        self.assertTrue(any(not r["complete"] for r in rows))


if __name__ == "__main__":
    unittest.main()
