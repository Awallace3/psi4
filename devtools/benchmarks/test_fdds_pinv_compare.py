import importlib.util
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "pinv_compare", Path(__file__).with_name("fdds_pinv_compare.py"))
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class ComparisonGuards(unittest.TestCase):
    solver = {"/env/libcusolver.so": "solver"}
    baseline = {"/env/libmkl.so": "mkl", "/env/libcuest.so": "cuest"}

    def samples(self):
        samples = {}
        for label, seconds in zip(compare.ORDER, (1000., 600., 550., 560., 620.)):
            candidate = label.startswith("candidate")
            d = dict(ok=True, system="protein83", natom=83, nbf=922, naux=3820,
                     threads=8, memory="96 GiB", options={"science": "unchanged"},
                     cache_sha256="cache", metadata_sha256="metadata", runner_sha256="runner",
                     numpy_version="same", numpy_configuration="same", job_id="42", hostname="same",
                     occupied=[78, 68], virtual=[844, 854], pinv_calls=[{} for _ in range(22)],
                     gpu_pinv_requested=candidate, fdds_seconds=seconds,
                     arm="cpu" if label == "cpu0" else "gpu",
                     libraries_sha256={"/env/libmkl.so": "mkl", "/env/libcuest.so": "cuest",
                                       "/stage/core.cpython.so": "different-core-is-intentional"},
                     energies_Eh={"Disp20": -.1, "Disp20,FDDS (unc)": -.2})
            if candidate:
                d["libraries_sha256"]["/env/libcusolver.so"] = "solver"
            for k, v in zip(("gpu_timer_calls", "gpu_exchange_timer_calls", "gpu_metric_timer_calls",
                              "gpu_response_timer_calls", "gpu_pinv_timer_calls"),
                             (0, 0, 0, 0, 0) if label == "cpu0"
                             else (20, 4, 12, 60, 22 if candidate else 0)):
                d[k] = v
            samples[label] = d
        return samples

    def test_all_five_controls_and_two_medians(self):
        r = compare.summarize(self.samples(), self.solver, self.baseline)
        self.assertEqual(r["accepted_controls"], list(compare.ORDER))
        self.assertEqual(r["base_gpu_over_candidate_gpu"], 610. / 555.)
        self.assertEqual(r["native_cpu_over_candidate_gpu"], 1000. / 555.)

    def test_missing_control_rejected(self):
        s = self.samples()
        del s["base-gpu1"]
        with self.assertRaises(AssertionError):
            compare.summarize(s, self.solver, self.baseline)

    def test_missing_gpu_pinv_route_rejected(self):
        s = self.samples()
        s["candidate-gpu0"]["gpu_pinv_timer_calls"] = 0
        with self.assertRaises(AssertionError):
            compare.summarize(s, self.solver, self.baseline)

    def test_job_science_runner_or_scope_drift_rejected(self):
        for key, value in (("job_id", "43"), ("options", {}), ("runner_sha256", "other"),
                           ("threads", 4), ("system", "peptide")):
            s = self.samples()
            s["candidate-gpu0"][key] = value
            with self.assertRaises(AssertionError):
                compare.summarize(s, self.solver, self.baseline)

    def test_energy_failure_not_dropped(self):
        s = self.samples()
        s["candidate-gpu1"]["energies_Eh"]["Disp20"] += 2.e-5
        with self.assertRaises(AssertionError):
            compare.summarize(s, self.solver, self.baseline)

    def test_invalid_time_or_failed_worker_rejected(self):
        for key, value in (("fdds_seconds", float("inf")), ("fdds_seconds", -1.), ("ok", False)):
            s = self.samples()
            s["candidate-gpu0"][key] = value
            with self.assertRaises(AssertionError):
                compare.summarize(s, self.solver, self.baseline)

    def test_common_dependency_drift_rejected(self):
        s = self.samples()
        s["candidate-gpu0"]["libraries_sha256"]["/env/libmkl.so"] = "changed"
        with self.assertRaises(AssertionError):
            compare.summarize(s, self.solver, self.baseline)

    def test_unexpected_new_dependency_rejected(self):
        s = self.samples()
        s["candidate-gpu0"]["libraries_sha256"]["/elsewhere/libmkl-new.so"] = "new"
        with self.assertRaises(AssertionError):
            compare.summarize(s, self.solver, self.baseline)

    def test_nonfinite_reference_or_sample_energy_rejected(self):
        for label in ("cpu0", "candidate-gpu0"):
            for bad in (float("nan"), float("inf"), -float("inf")):
                s = self.samples()
                s[label]["energies_Eh"]["Disp20,FDDS (unc)"] = bad
                with self.assertRaises(AssertionError):
                    compare.summarize(s, self.solver, self.baseline)

    def test_additional_wrong_solver_or_baseline_solver_rejected(self):
        for label, path, digest in (("base-gpu1", "/env/libcusolver.so", "solver"),
                                    ("candidate-gpu0", "/other/libcusolver.so", "solver"),
                                    ("candidate-gpu0", "/env/libcusolver.so", "wrong"),
                                    ("candidate-gpu0", "/env/libcusolver-other.so", "solver")):
            s = self.samples()
            s[label]["libraries_sha256"][path] = digest
            with self.assertRaises(AssertionError):
                compare.summarize(s, self.solver, self.baseline)

    def test_solver_inserted_in_all_non_candidate_arms_is_not_trusted(self):
        s = self.samples()
        for label in ("cpu0", "base-gpu0", "base-gpu1"):
            s[label]["libraries_sha256"].update(self.solver)
        with self.assertRaises(AssertionError):
            compare.summarize(s, self.solver, self.baseline)

    def test_preexisting_solver_is_allowed_only_when_independently_pinned(self):
        s = self.samples()
        for label in ("cpu0", "base-gpu0", "base-gpu1"):
            s[label]["libraries_sha256"].update(self.solver)
        r = compare.summarize(s, self.solver, dict(self.baseline, **self.solver))
        self.assertEqual(len(r["accepted_controls"]), 5)

    def test_baseline_worker_provenance_is_not_candidate(self):
        env = compare.worker_environment({"CUEST_FDDS_BINARY_PIN": "new"},
                                         Path("/base"), "old", "oldhash")
        self.assertEqual(env["CUEST_FDDS_BINARY_PIN"], "old")
        self.assertEqual(env["CUEST_FDDS_READONLY_BUILD_ROOT"], "/base")
        self.assertEqual(env["CUEST_FDDS_EXPECTED_CORE_SHA256"], "oldhash")
        self.assertEqual(env["PYTHONPATH"], "/base/build_fdds_validation/stage/lib")

    def test_stdout_failure_cannot_leave_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            pending, verified = (Path(directory) / name for name in ("pending", "VERIFIED.json"))
            pending.write_text("{}")
            with patch("builtins.print", side_effect=BrokenPipeError):
                with self.assertRaises(BrokenPipeError):
                    compare.publish(pending, verified, {})
            self.assertTrue(pending.exists())
            self.assertFalse(verified.exists())


if __name__ == "__main__":
    unittest.main()
