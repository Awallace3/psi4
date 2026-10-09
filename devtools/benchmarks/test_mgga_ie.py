import unittest
from mgga_ie import (delta, options, summarize, process_result, FRAGMENTS, FUNCTIONALS,
                     validate_functional, validate_energy_components)


class Tests(unittest.TestCase):
    def test_wb97mv_is_in_default_matrix(self):
        self.assertIn("wb97m-v", FUNCTIONALS)
        self.assertNotIn("dft_vv10_b", options("aug-cc-pvdz", "gpu-xc"))
        self.assertFalse(options("aug-cc-pvdz", "cpu")["dft_vv10_postscf"])

    def test_wb97mv_requires_vv10_and_range_separation(self):
        class Functional:
            def name(self): return "wB97M-V"
            def is_meta(self): return True
            def needs_vv10(self): return self.vv10
            def is_x_lrc(self): return self.lrc
            def x_omega(self): return .3
            def x_alpha(self): return .15
            def x_beta(self): return .85
            def vv10_b(self): return 6.
            def vv10_c(self): return .01
            def needs_grac(self): return False
            vv10, lrc = True, True
        class Wavefunction:
            def functional(self): return f
            def variable(self, key):
                self_key = "DFT VV10 ENERGY"
                assert key == self_key
                return energy
        f, energy = Functional(), .02
        self.assertEqual(validate_functional(Wavefunction(), "wb97m-v")["vv10_energy_hartree"], .02)
        for vv10, lrc, energy in ((False, True, .02), (True, False, .02), (True, True, 0)):
            f.vv10, f.lrc = vv10, lrc
            with self.assertRaises(AssertionError):
                validate_functional(Wavefunction(), "wb97m-v")

    def test_total_energy_requires_vv10_component(self):
        parts = {"NUCLEAR REPULSION ENERGY": 2., "ONE-ELECTRON ENERGY": -30.,
                 "TWO-ELECTRON ENERGY": 10., "DFT XC ENERGY": -2.02,
                 "DFT VV10 ENERGY": .02, "DFT FUNCTIONAL TOTAL ENERGY": -20.}
        class Wavefunction:
            def variable(self, key): return parts[key]
        self.assertEqual(validate_energy_components(Wavefunction(), -20.)["DFT VV10 ENERGY"], .02)
        with self.assertRaisesRegex(AssertionError, "including VV10"):
            validate_energy_components(Wavefunction(), -20.02)

    def test_cp_speedup_uses_all_three_calculations(self):
        results = {}
        for route, walls in (("cpu", [10, 20, 30]), ("gpu-xc", [2, 4, 6])):
            for fragment, energy, wall in zip(FRAGMENTS, [-20.1, -10., -10.], walls):
                shift = 6e-6 if route == "gpu-xc" and fragment == "AB" else 0
                results[("host", "water", "wb97m-v", route, fragment)] = dict(
                    ok=True, nbf=10, energy_hartree=energy+shift, wall_s=wall)
        row = next(r for r in summarize(results) if r["complete"])
        self.assertEqual(row["cpu_cp_wall_s"], 60)
        self.assertEqual(row["gpu_cp_wall_s"], 12)
        self.assertEqual(row["cp_speedup"], 5)
        self.assertFalse(row["within_ie_1e_6_Eh"])
        self.assertTrue(row["within_ie_1e_5_Eh"])
        results[("host", "water", "wb97m-v", "gpu-xc", "B")]["ok"] = False
        self.assertFalse(any(r["complete"] for r in summarize(results)))

    def test_failed_teardown_invalidates_result(self):
        record = process_result(dict(ok=True, energy_hartree=-1), -11)
        self.assertFalse(record["ok"])
        self.assertTrue(record["calculation_reported_ok"])
        self.assertEqual(record["process_returncode"], -11)

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
