"""Short collected local tests. Synthetic trees are tests, never measurements."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
import publish_scaling_suite as publisher

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / ".pi/validation/scal631-20261016-snapshot-01"
RAW = SNAPSHOT / "raw"
CONTINUATION = ROOT / ".pi/validation/scal631-20261017-snapshot-02/raw"
FINAL = ROOT / ".pi/validation/scal631-20261017-snapshot-03/raw"
DOCS = Path(os.environ.get("CUEST_REPORT_DOCS", "/home/awallace43/docs/saptdft/cuest"))
sys.path.insert(0, str(DOCS))
import build_breakdown_2026_10_04 as report
import docs_html_report


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.campaign = publisher.verify(RAW, DOCS)
        cls.row = json.loads((RAW / "sessions/gpu-peptide-r1/result.json").read_text())
        cls.meta = (RAW / "sessions/gpu-peptide-r1.meta.txt").read_text()
        cls.output = (RAW / "sessions/gpu-peptide-r1/psi4.out").read_text()

    def check(self, row, meta=None, output=None):
        publisher.validate(row, self.meta if meta is None else meta,
                           self.output if output is None else output, "gpu", "peptide")

    def test_archive_hash_and_count(self):
        self.assertEqual(len(publisher.verify_archive(RAW)), 119)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AssertionError, "Archive identity"):
                publisher.verify_archive(Path(directory))
        with mock.patch.object(publisher, "inventory",
                return_value=dict(publisher.inventory(RAW), **{"run.sh": "bad"})):
            with self.assertRaisesRegex(AssertionError, "Archive identity"):
                publisher.verify_archive(RAW)

    def test_wrong_identity_source_geometry_and_backend(self):
        changes = dict(source_commit="wrong", runner_sha256="wrong", science_options_sha256="wrong",
            suite_systems_sha256="wrong", geometry_sha256="wrong", nbf=2491, naux_jkfit=9182,
            basis="aug-cc-pvdz", method="sapt0", threads=24, memory="1 GiB", slurm_job_id="old",
            hostname="other", arm="cpu", suite_system="protein157", ok=False,
            gpu_builder_seen=False, sad_gpu_builder_seen=False, cpu_builder_seen=True,
            direct_seen=True, disk_df_builder_seen=True)
        for key, value in changes.items():
            with self.subTest(key=key):
                row = copy.deepcopy(self.row)
                row[key] = value
                with self.assertRaises(AssertionError):
                    self.check(row)
        row = copy.deepcopy(self.row)
        row["iterator"]["sha256"] = "wrong"
        with self.assertRaises(AssertionError):
            self.check(row)
        for key in row["libraries_sha256"]:
            if "einsums" in key:
                continue
            broken = copy.deepcopy(self.row)
            broken["libraries_sha256"][key] = "wrong"
            with self.subTest(library=key), self.assertRaises(AssertionError):
                self.check(broken)

    def test_wrong_scientific_options_grid_and_grac(self):
        for key, value in [("basis", "aug-cc-pvdz"), ("SAPT_DFT_INDUCTION_TYPE", "NONE"),
                ("DFT_RADIAL_POINTS", 99), ("DFT_SPHERICAL_POINTS_INITIAL", 110),
                ("DFT_GRID_SWITCH_FACTOR", 10), ("PUREAM", False), ("SCF_SUBTYPE", "DIRECT"),
                ("USE_CUEST", False), ("CUEST_XC", False), ("CUEST_SAD", False),
                ("CUEST_MIXED_PRECISION", True), ("SAPT_DFT_DO_DHF", False),
                ("SAPT_DFT_GRAC_SHIFT_A", .2), ("SAPT_DFT_GRAC_COMPUTE", "ITERATIVE")]:
            row = copy.deepcopy(self.row)
            row["options"][key] = value
            with self.subTest(option=key), self.assertRaises(AssertionError):
                self.check(row)
        row = copy.deepcopy(self.row)
        row["grid_policy"]["minimum_final_iterations"] = 1
        with self.assertRaises(AssertionError):
            self.check(row)

    def test_completion_hardware_and_output_proofs(self):
        for before, after in [("QOS=embers", "QOS=inferno"), ("exit=0", "exit=1"),
                ("exit=0", "not-recorded"), (publisher.UUID, "GPU-other"),
                ("NVIDIA H200", "NVIDIA A100"), ("NumCPUs=8", "NumCPUs=24"),
                (publisher.DRIVER, "wrong"), ("build_pin=b9032fdf25", "build_pin=other")]:
            with self.subTest(metadata=before), self.assertRaises(AssertionError):
                self.check(self.row, meta=self.meta.replace(before, after))
        for before in ["SAD J/K backend: cuESTJK", "cuESTJK: GPU-Accelerated",
                "SAPT Coupled Induction Solver", "Energy and wave function converged.",
                "GRAC shift supplied by the user (not computed)."]:
            with self.subTest(output=before), self.assertRaises(AssertionError):
                self.check(self.row, output=self.output.replace(before, "missing"))
        with self.assertRaises(AssertionError):
            self.check(self.row, output=self.output + "\nDirectJK\n")

    def test_failed_missing_convergence_and_residual_proofs(self):
        for change in ("missing", "unconverged", "final1", "not-final", "policy",
                       "solver", "residual", "empty", "calls", "sum"):
            row = copy.deepcopy(self.row)
            if change == "missing":
                row["proofs"].pop()
            elif change == "unconverged":
                row["proofs"][0]["converged"] = False
            elif change == "final1":
                row["proofs"][3]["final"] = 1
            elif change == "not-final":
                row["proofs"][3]["final_grid_after"] = False
            elif change == "policy":
                row["policy_checks"]["problems"] = ["bad"]
            elif change == "solver":
                row["cpks_solvers"][0]["converged"] = False
            elif change == "residual":
                row["cpks_solvers"][1]["residuals"][-1][0] = 1e-6
            elif change == "empty":
                row["cpks_solvers"] = []
            elif change == "calls":
                row["cpks_solve_calls_s"] = []
            else:
                row["cpks_solve_s"] += 1
            with self.subTest(change=change), self.assertRaises(AssertionError):
                self.check(row)

    def test_fixed_grac_absence_and_required_five_scf_stages(self):
        self.check(self.row)  # Correctly has NO iterative GRAC option/stages.
        publisher.coupled_partition(self.row, report)
        for stage in publisher.SCF:
            row = copy.deepcopy(self.row)
            row["timer_records"] = {k: v for k, v in row["timer_records"].items()
                                    if report.normalize(k.split(";")[1] if ";" in k else "") != stage}
            with self.subTest(stage=stage), self.assertRaisesRegex(AssertionError, "Missing required"):
                publisher.coupled_partition(row, report)
        row = copy.deepcopy(self.row)
        row["timer_records"]["SAPT(DFT) Energy;SAPT(DFT):GRAC Shift Monomer A"] = dict(
            wall_time=0., n_calls=1)
        with self.assertRaisesRegex(AssertionError, "Unexpected iterative"):
            publisher.coupled_partition(row, report)

    def test_real_tree_additivity_and_phase_ownership(self):
        indices = {key: i for i, (key, _, _) in enumerate(publisher.OPERATIONS)}
        for arm in ("cpu", "gpu"):
            row = json.loads((RAW / f"sessions/{arm}-peptide-r1/result.json").read_text())
            values = publisher.coupled_partition(row, report)
            self.assertAlmostEqual(sum(values), row["wall_s"], places=12)
            vx = sum(v["wall_time"] for k, v in row["timer_records"].items()
                     if "Form Vx" in k.split(";")[-1])
            self.assertAlmostEqual(values[indices["response"]], vx, places=9)
            # Parent induction includes expensive J/K AND Vx; neither belongs in SAPT remainder.
            self.assertGreater(values[indices["jk"]], 0)
            self.assertGreater(values[indices["response"]], 0)
            self.assertLess(values[indices["sapt"]], sum(v["wall_time"] for k, v
                in row["timer_records"].items() if k.endswith(":ind")))
        broken = copy.deepcopy(self.row)
        broken["wall_s"] = 1
        with self.assertRaises(AssertionError):
            publisher.coupled_partition(broken, report)

    def test_synthetic_test_tree_no_double_counting(self):
        # Deliberately oversized derivative child tests inclusive terminal cuts.
        records = {"SAPT(DFT) Energy": dict(wall_time=100., n_calls=1)}
        for name in publisher.SCF:
            records["SAPT(DFT) Energy;" + name] = dict(wall_time=1., n_calls=1)
        prefix = "SAPT(DFT) Energy;SAPT(DFT):ind"
        records[prefix] = dict(wall_time=80., n_calls=1)
        records[prefix + ";JK: JK"] = dict(wall_time=30., n_calls=1)
        records[prefix + ";RV: Form Vx"] = dict(wall_time=40., n_calls=1)
        records[prefix + ";RV: Form Vx;Derivative Properties"] = dict(wall_time=120., n_calls=1)
        records["SAPT(DFT) Energy;SAPT(DFT):Build JK"] = dict(wall_time=2., n_calls=1)
        values = publisher.coupled_partition(dict(wall_s=101., timer_records=records), report)
        bins = dict(zip((k for k, _, _ in publisher.OPERATIONS), values))
        self.assertEqual((bins["jk"], bins["response"], bins["sapt"]), (32., 40., 10.))
        self.assertEqual(sum(values), 101.)
        records[prefix]["wall_time"] = 60.
        with self.assertRaisesRegex(AssertionError, "Non-additive"):
            publisher.coupled_partition(dict(wall_s=101., timer_records=records), report)

    def test_partial_counts_null_ratios_and_median_vectors(self):
        c = self.campaign
        self.assertEqual((c["measured_records"], c["planned_records"]), (14, 24))
        self.assertEqual([s["cpu"]["n"] if s["cpu"] else 0 for s in c["cases"]], [1, 1, 0, 0])
        self.assertEqual([s["gpu"]["n"] for s in c["cases"]], [3] * 4)
        self.assertEqual([s["ratio"] for s in c["cases"][2:]], [None, None])
        self.assertEqual([s["gpu"]["median_s"] for s in c["cases"]],
                         [8.387831020983867, 17.027822704985738, 41.793996150023304, 199.08791020791978])
        for case, expected in zip(c["cases"], [4.546676332849253, 10.713946310743385]):
            self.assertAlmostEqual(case["ratio"], expected, places=12)
        for s in c["cases"]:
            gpu = s["gpu"]
            self.assertEqual(gpu["median_s"], sorted(r["wall_s"] for r in gpu["runs"])[1])
            self.assertIn(gpu["representative"], gpu["runs"])
            self.assertAlmostEqual(sum(gpu["representative"]["values"]), gpu["median_s"], places=10)
        runs = [dict(repeat=i, wall_s=w, values=v) for i, w, v in [
            (1, 4., [1., 3.]), (2, 8., [7., 1.]), (3, 6., [2., 4.])]]
        self.assertEqual(publisher.summarize(runs)["representative"]["values"], [2., 4.])
        for bad in ([], runs * 2, [runs[0], runs[0]]):
            with self.assertRaises(AssertionError):
                publisher.summarize(bad)

    def test_even_summary_is_averaged_two_run_vector_not_a_measured_run(self):
        runs = [dict(repeat=2, wall_s=8., values=[7., 1.], source="r2"),
                dict(repeat=1, wall_s=4., values=[1., 3.], source="r1")]
        summary = publisher.summarize(runs)
        self.assertEqual(summary["median_s"], 6.)
        self.assertIsNone(summary["representative"])
        self.assertEqual(summary["display_vector"], dict(
            wall_s=6., values=[4., 2.], repeats=[1, 2],
            kind="averaged-two-run", label="averaged two-run vector"))
        self.assertEqual(summary["runs"], runs)
        self.assertNotIn("source", summary["display_vector"])
        self.assertNotIn("repeat", summary["display_vector"])
        self.assertNotIn(summary["display_vector"], runs)
        self.assertEqual((summary["min_s"], summary["max_s"]), (4., 8.))
        self.assertEqual(sum(summary["display_vector"]["values"]), summary["median_s"])
        for n in (1, 3):
            measured = publisher.summarize((runs + [dict(repeat=3, wall_s=6., values=[2., 4.])])[:n])
            self.assertIn(measured["representative"], measured["runs"])
            self.assertEqual(measured["display_vector"]["values"], measured["representative"]["values"])

    def test_idempotent_publication_preserves_old_keys_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            docs = Path(directory)
            original = json.loads((DOCS / "grid-evidence.json").read_text())
            original.pop(publisher.KEY, None)
            target = docs / "grid-evidence.json"
            target.write_text(json.dumps(original))
            publisher.publish(RAW, docs)
            first = target.read_bytes()
            publisher.publish(RAW, docs)
            self.assertEqual(first, target.read_bytes())
            new = json.loads(first)
            new.pop(publisher.KEY)
            self.assertEqual(new, original)
            for name, digest in publisher.inventory(RAW).items():
                self.assertEqual(publisher.sha(docs / publisher.SOURCE / name), digest)
            conflict = docs / publisher.SOURCE / "run.sh"
            conflict.write_text("test conflict")
            with self.assertRaisesRegex(AssertionError, "Refusing mismatched"):
                publisher.publish(RAW, docs)
            self.assertEqual(conflict.read_text(), "test conflict")
            self.assertEqual(first, target.read_bytes())


class ContinuationTests(unittest.TestCase):
    def test_verified_increment_is_17_records_four_ratios_eight_stacks(self):
        old_hashes = publisher.inventory(SNAPSHOT)
        new_hashes = publisher.inventory(CONTINUATION)
        c = publisher.verify(RAW, DOCS, continuation=CONTINUATION)
        self.assertEqual((c["measured_records"], c["planned_records"], c["partial"]), (17, 24, True))
        self.assertEqual([s["cpu"]["n"] for s in c["cases"]], [1, 1, 2, 1])
        self.assertEqual([s["gpu"]["n"] for s in c["cases"]], [3] * 4)
        self.assertEqual(sum(bool(s[a]) for s in c["cases"] for a in ("cpu", "gpu")), 8)
        self.assertEqual(len([s for s in c["cases"] if s["ratio"]]), 4)
        p83, p157 = c["cases"][2:]
        self.assertEqual(p83["cpu"]["median_s"], 632.8501015024958)
        self.assertEqual(p83["ratio"], 15.142129487470486)
        self.assertEqual(p157["ratio"], 23.71055278728423)
        self.assertIsNone(p83["cpu"]["representative"])
        display = p83["cpu"]["display_vector"]
        self.assertEqual(display["label"], "averaged two-run vector")
        for i, value in enumerate(display["values"]):
            self.assertEqual(value, (p83["cpu"]["runs"][0]["values"][i] +
                                     p83["cpu"]["runs"][1]["values"][i]) / 2)
        for s in c["cases"]:
            for a in ("cpu", "gpu"):
                summary = s[a]
                self.assertAlmostEqual(sum(summary["display_vector"]["values"]), summary["median_s"], places=9)
                for run in summary["runs"]:
                    self.assertAlmostEqual(sum(run["values"]), run["wall_s"], places=9)
                    self.assertTrue(run["source"].startswith(
                        publisher.CONTINUATION_SOURCE if a == "cpu" and s["system"].startswith("protein")
                        else publisher.SOURCE))
        self.assertEqual(p83["max_component_delta_eh"], 4.3407555928420133e-7)
        self.assertEqual(p157["max_component_delta_eh"], 6.641532528946226e-6)
        old = publisher.verify(RAW, DOCS)
        for i, s in enumerate(c["cases"]):
            self.assertEqual(s["gpu"], old["cases"][i]["gpu"])
            if i < 2:
                self.assertEqual(s["cpu"], old["cases"][i]["cpu"])
        self.assertEqual(publisher.inventory(SNAPSHOT), old_hashes)
        self.assertEqual(publisher.inventory(CONTINUATION), new_hashes)

    def test_continuation_validation_rejects_wrong_job_partial_source_and_backend(self):
        stem = CONTINUATION / "sessions/cpu-protein83-r1"
        row = json.loads((stem / "result.json").read_text())
        meta = stem.with_suffix(".meta.txt").read_text()
        output = (stem / "psi4.out").read_text()
        publisher.validate(row, meta, output, "cpu", "protein83", job="14120335")
        for key, value in [("slurm_job_id", "14108848"), ("slurm_job_id", "unknown"),
                ("hostname", publisher.HARDWARE["cpu"]["node"] + ".pace.gatech.edu"),
                ("ok", False), ("source_commit", "wrong"), ("cpu_builder_seen", False),
                ("disk_df_builder_seen", True), ("gpu_builder_seen", True)]:
            q = copy.deepcopy(row)
            q[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(AssertionError):
                publisher.validate(q, meta, output, "cpu", "protein83", job="14120335")
        for broken in (meta.replace("exit=0", "exit=1"), meta.replace("exit=0", "partial"),
                       meta + "\nexit=1 end=partial\n"):
            with self.assertRaises(AssertionError):
                publisher.validate(row, broken, output, "cpu", "protein83", job="14120335")
        for arm, system in [("gpu", "protein83"), ("cpu", "peptide")]:
            with self.assertRaises(AssertionError):
                publisher.validate(row, meta, output, arm, system, job="14120335")
        with self.assertRaises(AssertionError):
            publisher.validate(row, meta, output, "cpu", "protein83", job="unknown")

    def test_pinned_incremental_inventory_and_idempotent_separate_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            docs = Path(directory)
            old = json.loads((DOCS / "grid-evidence.json").read_text())
            old.pop(publisher.KEY, None)  # Exercise historical17 publication in isolation.
            (docs / "grid-evidence.json").write_text(json.dumps(old))
            publisher.publish(RAW, docs, continuation=CONTINUATION)
            first = (docs / "grid-evidence.json").read_bytes()
            publisher.publish(RAW, docs, continuation=CONTINUATION)
            self.assertEqual(first, (docs / "grid-evidence.json").read_bytes())
            with self.assertRaisesRegex(AssertionError, "stale snapshot01 downgrade"):
                publisher.publish(RAW, docs)
            self.assertEqual(first, (docs / "grid-evidence.json").read_bytes())
            new = json.loads(first)
            self.assertEqual({k:v for k,v in new.items() if k != publisher.KEY},
                             {k:v for k,v in old.items() if k != publisher.KEY})
            self.assertEqual(len(publisher.inventory(docs / publisher.SOURCE)) - 1, 119)
            for name, digest in publisher.inventory(CONTINUATION).items():
                self.assertEqual(publisher.sha(docs / publisher.CONTINUATION_SOURCE / name), digest)
            for name in ("verification-receipt.md", "inventory.json", "raw-sha256.txt", "recovery-report.md"):
                target = docs / publisher.CONTINUATION_SOURCE / name
                self.assertEqual(publisher.sha(target), publisher.sha(CONTINUATION.parent / name))
            conflict = docs / publisher.CONTINUATION_SOURCE / "sessions/cpu-protein83-r1/result.json"
            conflict.write_text('{"ok": false}')
            with self.assertRaisesRegex(AssertionError, "Refusing mismatched"):
                publisher.publish(RAW, docs, continuation=CONTINUATION)
            self.assertEqual(first, (docs / "grid-evidence.json").read_bytes())
        with mock.patch.object(publisher, "inventory", return_value={}):
            with self.assertRaisesRegex(AssertionError, "continuation.*identity"):
                publisher.verify_continuation_archive(CONTINUATION)


class ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.campaign = publisher.verify(RAW, DOCS, continuation=CONTINUATION)

    def test_all_17_additive_cuts_and_full_measurements_unchanged(self):
        baseline = ROOT / ".pi/validation/scal631-response-toggle-20261017/before/grid-evidence.json"
        old = json.loads(baseline.read_text())[publisher.KEY]
        for case, before in zip(self.campaign["cases"], old["cases"]):
            for arm in ("cpu", "gpu"):
                for run, measured in zip(case[arm]["runs"], before[arm]["runs"]):
                    with self.subTest(system=case["system"], arm=arm, repeat=run["repeat"]):
                        for key, value in measured.items():
                            self.assertEqual(run[key], value)
                        p = run["induction_projection"]
                        self.assertEqual([x["path"] for x in p["phases"]], list(publisher.INDUCTION_CUTS))
                        removed_wall = sum(x["wall_s"] for x in p["phases"])
                        self.assertAlmostEqual(sum(p["removed_values"]), removed_wall, places=8)
                        self.assertEqual(p["wall_s"], run["wall_s"] - removed_wall)
                        self.assertGreater(p["wall_s"], 0)
                        self.assertLess(p["wall_s"], run["wall_s"])
                        self.assertAlmostEqual(sum(p["values"]), p["wall_s"], places=8)
                        for i, v in enumerate(run["values"]):
                            self.assertAlmostEqual(v, p["values"][i] + p["removed_values"][i], places=10)
                        for phase, solve in zip(p["phases"], run["solver_calls_s"]):
                            self.assertLessEqual(solve, phase["wall_s"])
                            self.assertAlmostEqual(sum(phase["values"]), phase["wall_s"], places=8)
                for key in ("median_s", "min_s", "max_s", "display_vector"):
                    self.assertEqual(case[arm][key], before[arm][key])

    def synthetic(self):
        records = {"SAPT(DFT) Energy": dict(wall_time=100., n_calls=1)}
        for stage in publisher.SCF:
            records["SAPT(DFT) Energy;" + stage] = dict(wall_time=1., n_calls=1)
        for path, wall in zip(publisher.INDUCTION_CUTS, (20., 40.)):
            records[path] = dict(wall_time=wall, n_calls=1)
            records[path + ";JK: JK"] = dict(wall_time=5., n_calls=1)
            records[path + ";RV: Form Vx"] = dict(wall_time=10., n_calls=1)
            records[path + ";RV: Form Vx;Derivative Properties"] = dict(wall_time=200., n_calls=1)
        return dict(timer_records=records, wall_s=101., cpks_solve_calls_s=[18., 38.], cpks_solve_s=56.)

    def test_nested_oversized_children_are_terminal_not_redistributed(self):
        row = self.synthetic()
        p = publisher.induction_projection(row, report)
        self.assertEqual(p["wall_s"], 41.)
        self.assertEqual(sum(p["removed_values"]), 60.)  # NOT the nested solver sum, 56.
        bins = dict(zip((k for k, _, _ in publisher.OPERATIONS), p["removed_values"]))
        self.assertEqual((bins["jk"], bins["response"], bins["sapt"]), (10., 20., 30.))

    def test_fail_closed_missing_duplicate_overlap_wrong_cut_and_calls(self):
        for failure in ("missing", "duplicate", "wrong", "calls", "solver", "nan", "negative"):
            row = self.synthetic()
            path = publisher.INDUCTION_CUTS[0]
            if failure == "missing":
                del row["timer_records"][path]
            elif failure == "duplicate":
                row["timer_records"][path + " "] = row["timer_records"][path].copy()
            elif failure == "wrong":
                row["timer_records"][path.replace("SAPT(HF):ind", "SAPT(HF):wrong")] = row["timer_records"].pop(path)
            elif failure == "calls":
                row["timer_records"][path]["n_calls"] = 2
            elif failure == "solver":
                row["cpks_solve_calls_s"][0] = 21.
                row["cpks_solve_s"] = 59.
            else:
                row["timer_records"][path]["wall_time"] = float("nan") if failure == "nan" else -1.
            with self.subTest(failure=failure), self.assertRaises(AssertionError):
                publisher.induction_projection(row, report)
        for cuts in [(publisher.INDUCTION_CUTS[0],) * 2,
                     ("SAPT(DFT) Energy", publisher.INDUCTION_CUTS[1]),
                     (publisher.INDUCTION_CUTS[0], publisher.INDUCTION_CUTS[1] + ";JK: JK")]:
            with self.subTest(cuts=cuts), self.assertRaises(AssertionError):
                publisher.induction_projection(self.synthetic(), report, cut_paths=cuts)

    def test_negative_projected_component_and_wall_fail(self):
        row = self.synthetic()
        full = publisher.coupled_partition(row, report)
        sapt = [k for k, _, _ in publisher.OPERATIONS].index("sapt")
        outer = [k for k, _, _ in publisher.OPERATIONS].index("outer")
        full[outer] += full[sapt] - 29.
        full[sapt] = 29.  # Valid positive full bins/closure, but removed SAPT = 30.
        self.assertEqual(sum(full), row["wall_s"])
        self.assertGreaterEqual(min(full), 0)
        with self.assertRaisesRegex(AssertionError, "Negative projected"):
            publisher.induction_projection(row, report, full_values=full)
        row["wall_s"] = 50.
        with self.assertRaises(AssertionError):
            publisher.induction_projection(row, report)

    def test_each_view_resummarizes_runs_n2_and_changed_n3_ordering(self):
        for case in self.campaign["cases"]:
            view = case["timing_views"]["induction_none_like"]
            for arm in ("cpu", "gpu"):
                a = view[arm]
                runs = case[arm]["runs"]
                self.assertEqual(a["median_s"], __import__("statistics").median(
                    r["induction_projection"]["wall_s"] for r in runs))
                if a["n"] == 2:
                    self.assertIsNone(a["representative"])
                    self.assertEqual(a["display_vector"]["values"], [
                        (runs[0]["induction_projection"]["values"][j] +
                         runs[1]["induction_projection"]["values"][j])/2
                        for j in range(len(publisher.OPERATIONS))])
                else:
                    chosen = sorted(runs, key=lambda r: (r["induction_projection"]["wall_s"], r["repeat"]))[a["n"]//2]
                    self.assertEqual(a["representative"]["repeat"], chosen["repeat"])
                    self.assertEqual(a["display_vector"]["values"], chosen["induction_projection"]["values"])
            self.assertEqual(view["ratio"], view["cpu"]["median_s"]/view["gpu"]["median_s"])
            self.assertEqual(view["ratio_range"], [
                view["cpu"]["min_s"]/view["gpu"]["max_s"], view["cpu"]["max_s"]/view["gpu"]["min_s"]])
        runs = [dict(repeat=i, wall_s=f, values=[f],
            induction_projection=dict(wall_s=p, values=[p]))
            for i, f, p in [(1, 10., 8.), (2, 20., 3.), (3, 30., 5.)]]
        self.assertEqual(publisher.summarize(runs)["representative"]["repeat"], 2)
        derived = publisher.summarize_projection(runs)
        self.assertEqual(derived["representative"]["repeat"], 3)
        self.assertEqual(derived["median_s"], 5.)
        self.assertEqual(derived["display_vector"]["kind"], "projected-run")


class ReportTests(unittest.TestCase):
    def test_one_global_timing_control_and_early_combined_trend(self):
        import scaling_suite_section
        c = publisher.verify(RAW, DOCS, continuation=CONTINUATION)
        soup = BeautifulSoup(scaling_suite_section.render({publisher.KEY: c}), "html.parser")
        self.assertEqual(len(soup.select("#scal-timing")), 1)
        self.assertEqual([n["value"] for n in soup.select("#scal-timing option")],
                         ["full", "induction_none_like"])
        self.assertLess(str(soup).index('id="scal-ratios"'), str(soup).index('id="scal-cases"'))
        self.assertEqual(len(soup.select(".scal-status")), 2)
        text = soup.get_text(" ", strip=True)
        for phrase in ("Induction NONE-like", "projected", "Not independently measured NONE",
                       "surrounding induction", "Full measured wall", "Removed induction",
                       "HF / DFT owners", "nested diagnostics"):
            self.assertIn(phrase, text)

    def test_frozen_evidence_and_before_snapshot_hashes(self):
        baseline = json.loads((ROOT / ".pi/validation/report-refresh-20261016/baseline.json").read_text())
        for name, digest in baseline["evidence"].items():
            self.assertEqual(publisher.sha(Path(name)), digest)
        for name, digest in baseline["files"].items():
            if Path(name).name in ("breakdown.html", "cuest-bottlenecks.html"):
                self.assertEqual(publisher.sha(Path(name)), digest)
        for name, digest in json.loads((SNAPSHOT / "report-before/hashes.json").read_text()).items():
            self.assertEqual(publisher.sha(SNAPSHOT / "report-before" / name), digest)
        self.assertEqual(publisher.sha(DOCS / "operation_pairs.py"),
                         publisher.sha(SNAPSHOT / "report-before/operation_pairs.py"))
        self.assertEqual(len(publisher.verify_archive(RAW)), 119)

    def test_registry_overview_and_preserved_payload(self):
        soup = BeautifulSoup((DOCS / "breakdown-2026-10-04.html").read_text(), "html.parser")
        before = BeautifulSoup((SNAPSHOT / "report-before/breakdown-2026-10-04.html").read_text(), "html.parser")
        old, new = (json.loads(s.select_one("#data").string) for s in (before, soup))
        grid = new["gridEvidence"].copy()
        self.assertIn(publisher.KEY, grid)
        grid.pop(publisher.KEY)
        self.assertEqual(grid, old["gridEvidence"])
        self.assertEqual({k: v for k, v in new.items() if k != "gridEvidence"},
                         {k: v for k, v in old.items() if k != "gridEvidence"})
        self.assertEqual(json.loads(soup.select_one("#report-sections").string), list(docs_html_report.MAIN_SECTIONS))
        self.assertEqual(len(soup.select("main h2")), 11)
        self.assertEqual(soup.select("main > section")[0]["id"], "coupled-scaling")
        self.assertEqual(len(soup.select(".scal-ratio-point")), 4)
        self.assertIn("four observed CPU/GPU ratios",
                      soup.select_one("#coupled-scaling-title")["data-summary"])
        register = soup.select_one(".cohort-register").get_text(" ", strip=True)
        self.assertIn("verified 24/24", register)
        self.assertNotIn("No large-system ratio", register)
        self.assertFalse(soup.select("#coupled-scaling .op-card, #coupled-scaling .op-bar"))
        self.assertTrue({n["id"] for n in before.select("[id]")} <= {n["id"] for n in soup.select("[id]")})
        self.assertIn("11468.475161", soup.select_one("#cpks").get_text())
        self.assertNotIn("PENDING", soup.select_one("#cpks").get_text())
        self.assertLess(len(str(soup).encode()), 3 * 1024 * 1024)
        comparisons = sum(len(c["cases"]) for c in new["campaigns"]) + sum(
            len(c["cases"]) for key in ("cpu_gpu_grid_campaigns", "cpu_gpu_estimate_campaigns",
                "cpu_gpu_basis_matched_campaigns") for c in new["gridEvidence"][key])
        self.assertEqual(comparisons, 49)

    def test_continuation_report_honestly_labels_vectors_counts_and_ranges(self):
        import scaling_suite_section
        c = publisher.verify(RAW, DOCS, continuation=CONTINUATION)
        soup = BeautifulSoup(scaling_suite_section.render({publisher.KEY: c}), "html.parser")
        text = soup.get_text(" ", strip=True)
        self.assertIn("partial 17/24", text)
        self.assertIn("averaged two-run vector", text)
        self.assertIn("not a measured run", text)
        self.assertIn("observation ranges, not confidence intervals", text)
        self.assertIn("6.641532528946226e-06 Eh", text)
        self.assertIn("4.3407555928420133e-07 Eh", text)
        self.assertIn("seven CPU repeats", text)
        self.assertNotIn("CPU awaiting", text)
        self.assertIn("619.4842688259669", text)
        self.assertIn("646.2159341790248", text)
        self.assertEqual(len(soup.select('a[href$="/result.json"]')), 17)
        self.assertEqual(len(soup.select(".scal-ratio-point")), 4)
        for point, case in zip(soup.select(".scal-ratio-point"), c["cases"]):
            self.assertIn(f'CPU n{case["cpu"]["n"]} / GPU n3', point["aria-label"])
            self.assertIn("observation range", point["aria-label"])
            self.assertIn("not a confidence interval", point["aria-label"])
        self.assertEqual(soup.select_one("#scal-ratios").parent["class"], ["svgbox"])

    def test_safe_render_idempotence(self):
        from publish_grid_evidence import render_preserving_history
        page = DOCS / "breakdown-2026-10-04.html"
        before = page.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            render_preserving_history(DOCS)
        self.assertEqual(before, page.read_bytes())


class FinalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.campaign = publisher.verify(RAW, DOCS, continuation=CONTINUATION, final_cpu=FINAL)

    def test_final24_counts_actual_representatives_and_frozen_analysis(self):
        c = self.campaign
        a = json.loads((FINAL.parent / "analysis.json").read_text())
        self.assertEqual((c["measured_records"], c["partial"]), (24, False))
        self.assertEqual(c["final_cpu"]["verification"]["reconciled_timer_paths"], 6354)
        self.assertEqual(len(c["final_cpu"]["verification"]["negative_controls_rejected"]), 20)
        identities = []
        for case in c["cases"]:
            for name, view in (("full", case), ("projected", case["timing_views"]["induction_none_like"])):
                reference = a["comparisons"][case["system"]][name]
                self.assertEqual(view["ratio"], reference["speedup"])
                self.assertEqual(view["ratio_range"], reference["speedup_extrema"])
                for arm in ("cpu", "gpu"):
                    s = view[arm]
                    self.assertEqual(s["n"], 3)
                    for k in ("median_s", "min_s", "max_s", "display_vector"):
                        self.assertEqual(s[k], reference[arm][k])
                    self.assertIsNotNone(s["representative"])
                    self.assertEqual(s["display_vector"]["values"], s["representative"]["values"])
                    # Ordering input cannot select a different median/vector.
                    runs = case[arm]["runs"]
                    summary = publisher.summarize(runs[::-1]) if name == "full" else publisher.summarize_projection(runs[::-1])
                    self.assertEqual(summary["display_vector"], s["display_vector"])
            self.assertEqual(case["max_component_delta_eh"],
                             a["comparisons"][case["system"]]["max_component_delta"]["abs_delta_hartree"])
            for arm in ("cpu", "gpu"):
                identities.extend(r["source"] for r in case[arm]["runs"])
        self.assertEqual(len(set(identities)), 24)
        before = publisher.verify(RAW, DOCS, continuation=CONTINUATION)
        for old, new in zip(before["cases"], c["cases"]):
            for arm in ("cpu", "gpu"):
                for run in old[arm]["runs"]:
                    self.assertEqual(run, next(r for r in new[arm]["runs"] if r["repeat"] == run["repeat"]))

    def test_all_seven_identity_partial_and_actual_output_rejection(self):
        rp = publisher.final_validator(FINAL).rp
        for label, (job, node) in publisher.FINAL_CASES.items():
            arm, system, repeat = label.split("-")
            stem = FINAL / "sessions" / label
            row = json.loads((stem / "result.json").read_text())
            meta, output = stem.with_suffix(".meta.txt").read_text(), (stem / "psi4.out").read_text()
            for key, value in (("slurm_job_id", "14132959" if job != "14132959" else "14132958"),
                               ("hostname", "wrong.pace.gatech.edu"), ("source_commit", "wrong"),
                               ("ok", False), ("threads", 8)):
                q = copy.deepcopy(row)
                q[key] = value
                with self.subTest(label=label, key=key), self.assertRaises(AssertionError):
                    publisher.validate(q, meta, output, arm, system, job=job, label=label)
            for bad_label in (None, "cpu-orphan-r1", label.replace(system, "wrong")):
                with self.subTest(label=label, bad_label=bad_label), self.assertRaises(AssertionError):
                    publisher.validate(row, meta, output, arm, system, job=job, label=bad_label)
            for text, m in ((output.replace("XC-FINAL", "XC-INITIAL", 1), meta),
                            (output, meta.replace("exit=0", "exit=1"))):
                with self.subTest(label=label), self.assertRaises(AssertionError):
                    rp.validate_data(row, text, m, arm, system, job, node + ".pace.gatech.edu")
            q = copy.deepcopy(row)
            q["timer_records"][publisher.INDUCTION_CUTS[0]]["wall_time"] += 1
            with self.subTest(timer=label), self.assertRaises(AssertionError):
                rp.reconcile_timers(q, (stem / "timer.dat").read_text())

    def test_pinned_raw_orphan_changed_and_receipt_rejection(self):
        hashes = publisher.inventory(FINAL)
        for altered in ({}, dict(hashes, orphan="0"*64),
                        dict(hashes, **{next(iter(hashes)): "0"*64})):
            with mock.patch.object(publisher, "inventory", return_value=altered):
                with self.assertRaisesRegex(AssertionError, "final archive identity"):
                    publisher.verify_final_archive(FINAL)
        with mock.patch.object(publisher, "inventory", return_value=hashes), \
                mock.patch.object(publisher, "sha", return_value="wrong"):
            with self.assertRaisesRegex(AssertionError, "final evidence identity"):
                publisher.verify_final_archive(FINAL)

    def test_final_publication_idempotence_conflict_preflight_and_no_downgrade(self):
        with tempfile.TemporaryDirectory() as directory:
            docs = Path(directory)
            original = json.loads((DOCS / "grid-evidence.json").read_text())
            original.pop(publisher.KEY, None)
            target = docs / "grid-evidence.json"
            target.write_text(json.dumps(original))
            publisher.publish(RAW, docs, continuation=CONTINUATION, final_cpu=FINAL)
            first, files = target.read_bytes(), publisher.inventory(docs)
            publisher.publish(RAW, docs, continuation=CONTINUATION, final_cpu=FINAL)
            self.assertEqual(files, publisher.inventory(docs))
            self.assertEqual({k:v for k,v in json.loads(first).items() if k != publisher.KEY}, original)
            for continuation in (None, CONTINUATION):
                with self.assertRaisesRegex(AssertionError, "final24 downgrade"):
                    publisher.publish(RAW, docs, continuation=continuation)
            self.assertEqual(first, target.read_bytes())
            conflict = docs / publisher.FINAL_SOURCE / "raw/sessions/cpu-peptide-r2/result.json"
            conflict.write_text("conflict")
            before = publisher.inventory(docs)
            with self.assertRaisesRegex(AssertionError, "Refusing mismatched"):
                publisher.publish(RAW, docs, continuation=CONTINUATION, final_cpu=FINAL)
            self.assertEqual(before, publisher.inventory(docs))
        with self.assertRaisesRegex(AssertionError, "requires independent"):
            publisher.verify(RAW, DOCS, final_cpu=FINAL)

    def test_final_report_has_no_partial_or_averaged_current_labels(self):
        import scaling_suite_section
        soup = BeautifulSoup(scaling_suite_section.render({publisher.KEY: self.campaign}), "html.parser")
        text = soup.get_text(" ", strip=True)
        self.assertIn("24/24", text)
        self.assertIn("CPU n3", text)
        self.assertIn("GPU n3", text)
        for stale in ("partial 17/24", "seven CPU repeats still missing", "averaged two-run vector",
                      "single CPU sample", "CPU n1/1/2/1"):
            self.assertNotIn(stale, text)
        self.assertEqual(len(soup.select('a[href$="/result.json"]')), 24)
        self.assertEqual(len(soup.select(".scal-ratio-point")), 4)
        for p in soup.select(".scal-ratio-point"):
            self.assertIn("CPU n3 / GPU n3", p["aria-label"])
        self.assertIn("snapshot-03/verification-receipt.md", str(soup))


if __name__ == "__main__":
    unittest.main()
