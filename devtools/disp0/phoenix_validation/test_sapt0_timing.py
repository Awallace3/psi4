"""Host accounting/acceptance contracts; no numerical GPU claim."""
import copy
from contextlib import ExitStack
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch
import sapt0_timing as timing


class TimingContracts(unittest.TestCase):
    def test_gpu_guard_imports_real_module_name_and_restores_fallbacks(self):
        name = "psi4.driver.procrouting.sapt.sapt_mp2_terms_ein"
        # Check the package contract without loading a native Psi4 core.
        source = Path(__file__).resolve().parents[3] / Path(*name.split(".")).with_suffix(".py")
        self.assertTrue(source.is_file())
        modules = {}
        parts = name.split(".")
        for count in range(1, len(parts) + 1):
            key = ".".join(parts[:count])
            modules[key] = ModuleType(key)
            modules[key].__path__ = []
            if count > 1:
                setattr(modules[".".join(parts[:count - 1])], parts[count - 1], modules[key])
        real_module = modules[name]
        fallback = lambda: "original"
        methods = ("df_mp2_sapt_dispersion", "df_mp2_einsums_dispersion",
                   "df_mp2_fisapt_dispersion")
        for method in methods:
            setattr(real_module, method, fallback)
        with patch.dict(sys.modules, modules):
            with ExitStack() as stack:
                timing.forbid_cpu_dispersion(stack)
                for method in methods:
                    with self.assertRaisesRegex(RuntimeError, "fallback forbidden"):
                        getattr(real_module, method)()
            for method in methods:
                self.assertIs(getattr(real_module, method), fallback)

    def timers(self):
        return {"root;" + name: {"wall_time": seconds, "n_calls": 1}
                for name, seconds in (
                    (timing.TIMER["elst"], 1.), (timing.TIMER["exch"], 2.),
                    (timing.TIMER["ind"], 5.), (timing.TIMER["disp"], 4.),
                    (timing.ASSEMBLY, .5), (timing.DIMER, 6.))}

    def records(self):
        rows = []
        for arm in ("cpu", "gpu"):
            for n in range(3):
                rows.append({
                    "status": "PASS", "arm": arm, "input_sha256": "geometry",
                    "source_pin": "native", "threads": 8, "host": "same-node",
                    "nbf": 300, "dimer_nmo": 300, "host_memory": "96 GiB", "runtime_config_sha256": "runtime",
                    "options": timing.options(arm == "gpu"),
                    "raw_Eh": {key: (i + 1) * 1e-3 for i, key in enumerate(timing.RAW)},
                    "total_Eh": .028,
                    "seconds": {key: (n + 1.) * (2 if arm == "cpu" else 1)
                                for key in (*timing.TERMS, "shared_setup_other", "total")}})
        return rows

    def test_nested_delta_is_not_double_counted(self):
        result = timing.term_times({}, self.timers(), 20.)
        self.assertEqual(result["ind"], 4.5)
        self.assertEqual(result["delta_hf"], 6.5)
        self.assertEqual(result["shared_setup_other"], 2.)
        self.assertEqual(sum(v for k, v in result.items() if k != "total"), result["total"])

    def test_missing_repeated_or_negative_timer_rejects(self):
        for mutation in ("missing", "calls", "negative"):
            records = self.timers()
            key = "root;" + timing.TIMER["disp"]
            if mutation == "missing":
                del records[key]
            elif mutation == "calls":
                records[key]["n_calls"] = 2
            else:
                records[key]["wall_time"] = -1
            with self.assertRaises(ValueError):
                timing.term_times({}, records, 20.)

    def test_prior_intervals_are_not_included(self):
        after = self.timers()
        before = copy.deepcopy(after)
        for row in after.values():
            row["wall_time"] += 4.
            row["n_calls"] += 1
        self.assertEqual(timing.timer_delta(before, after, timing.DIMER), 4.)

    def test_medians_include_all_samples(self):
        result = timing.compare(self.records())
        for term in result["timing"].values():
            self.assertEqual(term["cpu_median_seconds"], 4.)
            self.assertEqual(term["gpu_median_seconds"], 2.)
            self.assertEqual(term["cpu_over_gpu"], 2.)

    def test_cancellation_cannot_hide_signed_failure(self):
        rows = self.records()
        rows[3]["raw_Eh"]["Disp20,u"] += .001
        rows[3]["raw_Eh"]["Exch-Disp20,u"] -= .001
        with self.assertRaisesRegex(ValueError, "signed component"):
            timing.compare(rows)

    def test_repetition_drift_rejects_even_within_cross_arm_tolerance(self):
        rows = self.records()
        rows[-1]["raw_Eh"]["Elst10,r"] += 1e-8
        with self.assertRaisesRegex(ValueError, "repetition"):
            timing.compare(rows)

    def test_partial_failed_and_unmatched_samples_reject(self):
        with self.assertRaises(ValueError):
            timing.compare(self.records()[:-1])
        for field, value in (("status", "FAILED"), ("source_pin", "foreign"),
                             ("host", "other-node"), ("threads", 24), ("nbf", 301)):
            rows = self.records()
            rows[-1][field] = value
            with self.assertRaises(ValueError):
                timing.compare(rows)

    def test_scientific_options_must_match_and_zero_resolution_has_no_ratio(self):
        rows = self.records()
        rows[-1]["options"]["freeze_core"] = False
        with self.assertRaises(ValueError):
            timing.compare(rows)
        rows = self.records()
        for row in rows:
            row["seconds"]["elst"] = 0.
        self.assertIsNone(timing.compare(rows)["timing"]["elst"]["cpu_over_gpu"])


if __name__ == "__main__":
    unittest.main()
