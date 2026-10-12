import unittest
from mgga_ie import (delta, options, summarize, process_result, FRAGMENTS, FUNCTIONALS,
                     validate_functional, validate_energy_components,
                     validate_vv10_runtime_options, CONTROL_FUNCTIONALS,
                     ALL_FUNCTIONALS, select_matrix)


class Tests(unittest.TestCase):
    def test_lower_rungs_are_explicit_controls_not_new_defaults(self):
        self.assertEqual(CONTROL_FUNCTIONALS, ["svwn", "blyp", "b3lyp", "pbe0", "wb97x"])
        self.assertEqual(ALL_FUNCTIONALS, FUNCTIONALS + CONTROL_FUNCTIONALS)
        self.assertEqual(select_matrix()[1], FUNCTIONALS)
        cases, functionals = select_matrix(systems=["water"], functionals=CONTROL_FUNCTIONALS)
        self.assertEqual(cases, [("water", "aug-cc-pvdz")])
        self.assertEqual(functionals, CONTROL_FUNCTIONALS)
        self.assertEqual(select_matrix(preflight=True)[1], ["m06"])
        for args in (dict(systems=[]), dict(functionals=[]),
                     dict(systems=["unknown"]), dict(functionals=["unknown"])):
            with self.assertRaises(AssertionError):
                select_matrix(**args)

    def test_lower_rung_metadata_and_exchange_are_validated(self):
        class Functional:
            def name(self): return name.upper()
            def is_meta(self): return self.meta
            def is_gga(self): return self.gga
            def ansatz(self): return 2 if self.meta else 1 if self.gga else 0
            def is_x_hybrid(self): return self.alpha > 0
            def needs_vv10(self): return self.vv10
            def is_x_lrc(self): return self.lrc
            def x_omega(self): return .3 if self.lrc else 0.
            def x_alpha(self): return self.alpha
            def x_beta(self): return self.beta
            def needs_grac(self): return self.grac
            def vv10_b(self): return 6.
            def vv10_c(self): return .01
            meta, vv10, grac = False, False, False
        class Wavefunction:
            def functional(self): return f
            def variable(self, key): return -.02
        f = Functional()
        for name, gga, alpha, beta, lrc, rung in (
                ("svwn", False, 0., 0., False, 1),
                ("blyp", True, 0., 0., False, 2),
                ("b3lyp", True, .2, 0., False, 4),
                ("pbe0", True, .25, 0., False, 4),
                ("wb97x", True, .157706, .842294, True, 4)):
            f.gga, f.alpha, f.beta, f.lrc = gga, alpha, beta, lrc
            self.assertEqual(validate_functional(Wavefunction(), name)["rung"], rung)
            if name in ("svwn", "blyp"):
                f.beta = 1.  # Inactive blank-SuperFunctional default at omega=0.
                self.assertEqual(validate_functional(Wavefunction(), name)["rung"], rung)
                f.beta = beta
            for attr, wrong in (("meta", True), ("gga", not gga),
                                ("lrc", not lrc), ("vv10", True), ("grac", True),
                                ("alpha", alpha + .1)):
                old = getattr(f, attr)
                setattr(f, attr, wrong)
                with self.assertRaises(AssertionError, msg=(name, attr)):
                    validate_functional(Wavefunction(), name)
                setattr(f, attr, old)

    def test_lower_rung_counterpoise_and_speedup_are_summarized(self):
        results = {}
        for functional in CONTROL_FUNCTIONALS:
            for route, scale in (("cpu", 1.), ("gpu-jk", .9), ("gpu-xc", .2)):
                for fragment, energy, wall in zip(FRAGMENTS, [-20.1, -10., -10.], [10, 20, 30]):
                    shift = 6e-6 if route == "gpu-xc" and fragment == "AB" else 0.
                    results[("host", "water", functional, route, fragment)] = dict(
                        ok=True, nbf=10, energy_hartree=energy+shift, wall_s=wall*scale)
        complete = [r for r in summarize(results) if r["complete"]]
        self.assertEqual(len(complete), 10)
        for row in complete:
            self.assertTrue(row["within_ie_1e_5_Eh"])
            if row["route"] == "gpu-xc":
                self.assertEqual(row["cp_speedup"], 5.)

    def test_wb97mv_is_in_default_matrix(self):
        self.assertIn("wb97m-v", FUNCTIONALS)
        self.assertNotIn("dft_vv10_b", options("aug-cc-pvdz", "gpu-xc"))
        self.assertFalse(options("aug-cc-pvdz", "cpu")["dft_vv10_postscf"])

    def test_runtime_boolean_accepts_psi4_integer_zero(self):
        # Captured from all 54 failed workers: get_option returns integer 0.
        captured = {"DFT_VV10_POSTSCF": 0, "DFT_VV10_RADIAL_POINTS": 50,
                    "DFT_VV10_SPHERICAL_POINTS": 146, "DFT_VV10_RHO_CUTOFF": 1e-8}
        validate_vv10_runtime_options(captured)
        validate_vv10_runtime_options(dict(captured, DFT_VV10_POSTSCF=False))
        for value in (True, 1, None, "false"):
            with self.assertRaisesRegex(AssertionError, "Self-consistent VV10"):
                validate_vv10_runtime_options(dict(captured, DFT_VV10_POSTSCF=value))

    def test_wb97mv_requires_vv10_and_range_separation(self):
        class Functional:
            def name(self): return "wB97M-V"
            def is_meta(self): return True
            def is_gga(self): return True
            def is_x_hybrid(self): return True
            def ansatz(self): return 2
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
