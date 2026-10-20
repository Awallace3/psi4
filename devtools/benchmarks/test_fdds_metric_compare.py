import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("metric_compare", Path(__file__).with_name("fdds_metric_compare.py"))
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class MetricComparisonTests(unittest.TestCase):
    def records(self):
        result = {}
        for label in ("base-gpu-0", "candidate-gpu-0", "candidate-cpu-0", "candidate-gpu-1", "base-gpu-1"):
            base, cpu = label.startswith("base"), "-cpu-" in label
            package = "/base/psi4" if base else "/candidate/psi4"
            result[label] = dict(ok=True, fdds_seconds=20 if cpu else 8 if base else 6,
                cache_sha256="cache", metadata_sha256="metadata", runner_sha256="runner",
                options={"basis": "unchanged"}, hostname="node", job_id="14214205", threads=8, memory="96 GiB",
                libraries_sha256={package + "/core.cpython.so": "base" if base else "candidate",
                                  "/env/einsums/core.cpython.so": "same-einsums", "/env/libcublas.so": "same"},
                binary_pin="base" if base else "candidate", core_sha256="base" if base else "candidate",
                psi4_package=package + "/__init__.py", energies_Eh={"Disp20": -.007, "unc": -.011},
                gpu_timer_calls=0 if cpu else 20, gpu_exchange_timer_calls=0 if cpu else 4,
                gpu_metric_timer_calls=0 if base or cpu else 12)
        return result

    def run_audit(self, records):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, record in records.items():
                p = root / ("protein83-" + label)
                p.mkdir()
                (p / "result.json").write_text(json.dumps(record))
            return compare.audit(root, "protein83")

    def test_all_controls_and_intentional_core_difference(self):
        got = self.run_audit(self.records())
        self.assertEqual(got["controls"], 5)
        self.assertAlmostEqual(got["base_gpu_over_candidate_gpu"], 8 / 6)

    def test_missing_or_failed_sample_never_dropped(self):
        records = self.records()
        del records["base-gpu-1"]
        with self.assertRaises(FileNotFoundError):
            self.run_audit(records)
        records = self.records()
        records["candidate-gpu-1"]["ok"] = False
        with self.assertRaises(AssertionError):
            self.run_audit(records)

    def test_dependency_core_not_confused_with_intentional_psi4_core_change(self):
        records = self.records()
        records["candidate-gpu-1"]["libraries_sha256"]["/env/einsums/core.cpython.so"] = "wrong"
        with self.assertRaises(AssertionError):
            self.run_audit(records)

    def test_settings_and_route_and_energy_fail_closed(self):
        for key, value in (("threads", 4), ("gpu_metric_timer_calls", 0),
                           ("energies_Eh", {"Disp20": 1., "unc": -.011})):
            records = self.records()
            records["candidate-gpu-1"][key] = value
            with self.assertRaises(AssertionError):
                self.run_audit(records)

    def test_nonfinite_time_or_energy_rejected(self):
        for key, value in (("fdds_seconds", float("inf")),
                           ("energies_Eh", {"Disp20": float("nan"), "unc": -.011})):
            records = self.records()
            records["candidate-gpu-1"][key] = value
            with self.assertRaises(AssertionError):
                self.run_audit(records)


if __name__ == "__main__":
    unittest.main()
