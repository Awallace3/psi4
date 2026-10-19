"""Host/fake-only diagnostic tests. NOT GPU correctness or numerical repeatability evidence."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import campaign
import handoff
import observe as o
import policy as p
import repeatability as r
import repeatability_campaign as c
import repeatability_handoff as h
import science

ROOT = Path(__file__).resolve().parents[2]


def fixture():
    payload = p.read_json(os.environ["MAJOR2_REPEATABILITY_COLLECTION"])
    files = p.file_manifest(ROOT)
    return dict(mode=r.MODE, approval=r.REVIEW_ID, diagnostic_pair=payload["pair"],
        run_root=r.RUN, route=p.ROUTE, gates=p.GATES, suite=p.H0+"/geometries", suite_sha256=p.SUITE_SHA,
        scratch_parent="/storage/scratch1/3/awallace43/cpks-major2-p157",
        sequence=r.sequence(), matrix_cap_bytes=r.MATRIX_CAP, per_process_timeout_seconds=1800,
        build_budget_seconds=[0, 0], accepted_speedup_data=False, identity_helper=c.IDENTITY,
        harness=dict(path=r.HARNESS, pin=r.PARENT_PIN, files=files, files_digest=p.digest(files)),
        arms={arm: dict(source=item["source"], build=item["build"], pin=item["pin"],
            files=p.file_manifest(ROOT.parent/("saptdft-cuest-major2-jkexception-baseline-20261018"
                if arm == "baseline" else "saptdft-cuest-major2-nonc1-candidate-20261019")))
            for arm, item in r.ALLOWLIST.items()}, preservation=payload["preservation"])


def records():
    rows = []
    for i, row in enumerate(r.sequence()):
        d = dict(proofs=[dict(iterations=20-i, functional="HF" if k < 3 else "PBE0", nbf=1786) for k in range(5)],
                 cpks_solvers=[dict(iterations=18-i, residuals=[[1e-9, 1e-9]])]*2,
                 components_hartree={key: 0.0 for key in science.COMPONENTS},
                 wall_s=987654321+i, timer_records={"SECRET_TIMER": 876543210}, cpks_solve_s=765432109)
        rows.append(dict(row, data=d))
    return rows


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = fixture()

    def test_exact_pair_and_manifest(self):
        r.validate(self.m)
        self.assertEqual(r.PAIR_DIGEST, p.digest(self.m["diagnostic_pair"]))
        self.assertEqual([x["arm"] for x in r.sequence()], ["baseline", "candidate", "candidate", "baseline", "baseline", "candidate"])
        self.assertEqual(len(h.FAILED_JOBS), 10)

    def test_wrong_pin_old_candidate_wrong_core_and_arbitrary_adoption(self):
        for arm in r.ALLOWLIST:
            for key, val in [("pin", campaign.j.CANDIDATE), ("core_sha256", "0"*64),
                             ("build", "/arbitrary"), ("origin_job", "123")]:
                with self.subTest(arm=arm, key=key), self.assertRaises(RuntimeError):
                    value = copy.deepcopy(self.m["diagnostic_pair"])
                    value["arms"][arm][key] = val
                    r.validate_pair(value)

    def test_mode_route_order_budget_science_negative_controls(self):
        for change in [lambda m: m.update(mode=campaign.n.MODE),
                       lambda m: m.update(sequence=r.sequence()[::-1]),
                       lambda m: m["route"].update(qos="inferno"),
                       lambda m: m["arms"]["candidate"].update(pin=campaign.j.CANDIDATE),
                       lambda m: m["gates"].update(component_abs_eh=1e-7),
                       lambda m: m.update(matrix_cap_bytes=2**60),
                       lambda m: m.update(build_budget_seconds=[0, 3600]),
                       lambda m: m.update(accepted_speedup_data=True)]:
            with self.assertRaises(RuntimeError):
                value = copy.deepcopy(self.m)
                change(value)
                r.validate(value)

    def test_all_existing_81_checks_and_primary_apis_unchanged(self):
        names = subprocess.check_output(["git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", r.PARENT_PIN,
                                         "devtools/major2"], text=True).splitlines()
        for name in names:
            original = subprocess.check_output(["git", "-C", str(ROOT), "show", r.PARENT_PIN+":"+name])
            self.assertEqual(original, (ROOT/name).read_bytes(), name)

    def test_observation_source_seam_internal_not_ooo_surrogate(self):
        for arm in ("baseline", "candidate"):
            item = self.m["arms"][arm]
            files = item["files"]
            for name in ("psi4/src/export_wavefunction.cc", "psi4/src/export_fock.cc",
                         "psi4/driver/procrouting/scf_proc/scf_iterator.py", "psi4/driver/procrouting/proc.py"):
                original = subprocess.check_output(["git", "-C", str(ROOT), "show", item["pin"]+":"+name])
                import hashlib
                self.assertEqual(hashlib.sha256(original).hexdigest(), files[name]["sha256"])
            iterator = subprocess.check_output(["git", "-C", str(ROOT), "show",
                item["pin"]+":psi4/driver/procrouting/scf_proc/scf_iterator.py"], text=True)
            self.assertIn("self.iteration_ += 1", iterator)
            self.assertIn('with p4util.timer("HF: Form G"):\n            self.form_G()', iterator)
            self.assertIn("if (self.iteration_ == 0) and self.sad_:", iterator)
            binding = subprocess.check_output(["git", "-C", str(ROOT), "show",
                item["pin"]+":psi4/src/export_wavefunction.cc"], text=True)
            self.assertIn('.def("form_G", &scf::HF::form_G', binding)
            rhf = binding.split('m, "RHF",')[1].split('m, "ROHF",')[0]
            self.assertNotIn('.def("form_G"', rhf)

    def test_blinded_numerical_summary_retains_all_drift(self):
        rows = records()
        report = r.numerical_report(rows)
        self.assertEqual(len(report["records"]), 6)
        self.assertEqual(len(report["comparisons"]), 15)
        self.assertTrue(all(v["scf_iteration_drift"] for v in report["comparisons"]))
        text = json.dumps(report)
        for secret in ("987654321", "876543210", "765432109", "wall_s", "timer_records", "median", "speedup_percent"):
            self.assertNotIn(secret, text)
        self.assertFalse(report["accepted_speedup_data"])
        self.assertTrue(report["timings_blinded"])

    def test_missing_shuffled_or_cherry_picked_records_rejected(self):
        for value in (records()[:-1], records()[::-1], records()[:3]*2):
            with self.assertRaises(RuntimeError):
                r.numerical_report(value)

    def test_component_and_residual_gates_remain_fatal(self):
        value = records()
        value[5]["data"]["components_hartree"][science.COMPONENTS[0]] = 1.1e-8
        with self.assertRaises(RuntimeError):
            r.numerical_report(value)
        value = records()
        value[0]["data"]["cpks_solvers"][0]["residuals"][-1] = [1e-8, 0]
        with self.assertRaises(RuntimeError):
            r.numerical_report(value)

    def test_batch_no_build_new_entry_exact_route(self):
        text = c.batch_text("a"*64)
        for token in ("--qos=embers", "--time=04:00:00", "--gres=gpu:h200:1", "--cpus-per-task=8",
                      "--mem=128G", "--no-requeue", "--signal=TERM@300", r.RUN, r.HARNESS,
                      "repeatability_campaign.py run"):
            self.assertIn(token, text)
        self.assertNotIn("campaign.py run", text.replace("repeatability_campaign.py run", ""))
        self.assertNotIn("build.sh", text)

    def test_real_submit_api_durable_numeric_id_and_permanent_lock(self):
        with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as td:
            root = Path(td)
            (root/"logs").mkdir()
            p.write_json(root/"logs/pre-submit-gates.json", dict(harness_pin="pin", manifest_sha256="m",
                batch_sha256="b", preflight_sha256="h", pristine_staging=True, old_run_unchanged=True,
                old_build_sentinels_unchanged=True, test_only_stdout="gpu-h200", test_only_stderr=""))
            def accept(*_):
                lock = root/"logs/submission.lock"
                self.assertEqual((lock/"job-id.txt").read_text(), "12345\n")
                attempt = p.read_json(lock/"attempt.json")
                self.assertFalse(attempt["compiled_in_new_job"])
                self.assertTrue(all(not v["compiled_in_this_job"] for v in attempt["build_origins"].values()))
                raise RuntimeError("fake unavailable acceptance")
            with patch.object(r, "RUN", str(root)), patch.object(handoff, "queue", return_value=[]), \
                 patch.object(p, "sha", return_value="h"), patch.object(handoff, "output", side_effect=accept), \
                 patch.object(handoff.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="12345\n", stderr="")):
                for exc in (RuntimeError, FileExistsError):
                    with self.assertRaises(exc):
                        handoff.submit("pin", "m", "b", cohort=r, verifier=lambda *a: ({}, {}),
                                       cache=root, failed_jobs=h.FAILED_JOBS)

    def test_no_default_submission_or_unreviewed_parent_entry(self):
        with patch.object(h.os, "environ", {"USER": "awallace43"}), \
             patch.object(p, "git", return_value="pin"), patch.object(h.sys, "argv", ["x", "parent-submit", "pin", "m", "b"]), \
             patch.object(h, "audit") as audit, self.assertRaisesRegex(RuntimeError, "parent review"):
            h.main()
        audit.assert_not_called()

    def test_wrapper_does_not_write_installed_scientific_state(self):
        tree = ast.parse((ROOT/"devtools/major2/observe.py").read_text())
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        for forbidden in ("set_Da", "set_Db", "set_options", "set_local_option", "form_D", "compute", "warmup",
                          "set_geometry", "set_jk", "set_variable"):
            self.assertNotIn(forbidden, calls)
        text = (ROOT/"devtools/major2/repeatability_campaign.py").read_text()
        self.assertNotIn("science.compare(", text)
        self.assertNotIn("science.summary(", text)
        self.assertNotIn('self.build(', text)

    def test_six_process_fake_engine_retains_drift_without_primary_acceptance(self):
        calls = []
        values = {row["label"]: row for row in records()}
        with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as td:
            engine = object.__new__(c.Engine)
            engine.root, engine.raw = Path(td), Path(td)/"raw"
            engine.m = self.m
            engine.manifest_path = Path(td)/"manifest.json"
            engine.manifest_path.write_text("{}")
            engine.job, engine.signalled, engine.records, engine.sampler = "123", False, [], None
            engine.inputs = {}
            def prolog():
                calls.append("prolog")
                engine.raw.mkdir()
            engine.prolog = prolog
            engine.pins = lambda: None
            engine.inventory = lambda: calls.append("inventory")
            sampler = SimpleNamespace(terminate=lambda: calls.append("terminate"), wait=lambda timeout: 0)
            engine.start_sampler = lambda: setattr(engine, "sampler", sampler)
            def case(row):
                calls.append(row["label"])
                out = engine.raw/"sessions"/row["label"]
                out.mkdir()
                p.write_json(out/"result.json", values[row["label"]]["data"])
                p.write_json(engine.raw/(row["label"]+".NUMERICAL.json"), dict(trajectory=[
                    dict(stage=stage, rows=[dict(row="SAD", energy_eh=0., delta_eh=0., density_residual=0.)])
                    for stage in science.STAGES]))
                return values[row["label"]]["data"]
            engine.case = case
            with patch.object(c.signal, "signal"), patch.object(r, "verify_artifact"), \
                 patch.object(c, "matrix_report", return_value={}), \
                 patch.object(science, "compare", side_effect=AssertionError("primary compare called")), \
                 patch.object(science, "summary", side_effect=AssertionError("primary summary called")):
                engine.execute()
            report = p.read_json(Path(td)/"numerical-repeatability.json")
            self.assertEqual(len(report["records"]), 6)
            self.assertTrue(all(x["scf_iteration_drift"] for x in report["comparisons"]))
            self.assertEqual(calls, ["prolog", *values, "terminate", "inventory"])
            self.assertEqual(len(list(engine.raw.glob("*.RETAINED.json"))), 6)

    def test_trajectory_first_difference_preserves_stages_and_row_counts(self):
        a = [dict(stage=stage, rows=[dict(energy_eh=0.)]) for stage in science.STAGES]
        b = copy.deepcopy(a)
        b[1]["rows"][0]["energy_eh"] = 1e-12
        self.assertEqual(r.first_trajectory_difference(a, b),
                         dict(stage=science.STAGES[1], ordinal=0, fields=["energy_eh"]))

    def test_analysis_not_permitted_on_login(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "allocation"):
            c.matrix_report(Path("/unused"), [])


class HookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = np.zeros((1786, 1786), dtype=np.float64)
        cls.j = np.full((1786, 1786), .25, dtype=np.float64)
        cls.k = np.full((1786, 1786), -.125, dtype=np.float64)

    def make(self, root, cap=r.PROCESS_CAP):
        test = self
        class HF:
            def form_G(self):
                self.native += 1
                if self.fail:
                    raise RuntimeError("native failure")
                return 17
            def finalize(self):
                self.native_finalize += 1
                return 23
        class RHF(HF):
            iteration_ = 0
            sad_ = True
            native = 0
            native_finalize = 0
            fail = False
            def Da(self): return SimpleNamespace(np=test.d)
            def Db(self): return SimpleNamespace(np=test.d)
            def jk(self): return SimpleNamespace(J=lambda: [SimpleNamespace(np=test.j)], K=lambda: [SimpleNamespace(np=test.k)])
            def nalpha(self): return 20
            def nbeta(self): return 20
            def basisset(self): return SimpleNamespace(nbf=lambda: 1786)
            def molecule(self): return SimpleNamespace(natom=lambda: 157, nfragments=lambda: 2, Z=lambda _: 1)
            def functional(self): return SimpleNamespace(name=lambda: self.func)
        def stamp(wfn, _):
            return dict(banner="SAPT(DFT): "+science.STAGES[wfn.stage],
                        caller=dict(path="/exact/staged/iterator.py", function="scf_iterate", line=714))
        capture = o.Capture(RHF, "/exact/stage", root/"matrices", {"pin": "fake"}, stamp=stamp, cap=cap)
        return RHF, capture

    def test_hook_fake_pass_exact_once_restore_capture_counts_no_state_write(self):
        with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as td:
            cls, capture = self.make(Path(td))
            original = cls.form_G
            old = self.d.tobytes(), self.j.tobytes(), self.k.tobytes()
            with capture.installed():
                for index in range(5):
                    wfn = cls()
                    wfn.stage, wfn.func = index, "HF" if index < 3 else "PBE0"
                    self.assertEqual(wfn.form_G(), 17)
                    wfn.iteration_ = 1
                    self.assertEqual(wfn.form_G(), 17)
                    self.assertEqual(wfn.finalize(), 23)
                    self.assertEqual(wfn.native, 2)
                    self.assertEqual(wfn.native_finalize, 1)
            capture.finish()
            self.assertIs(cls.form_G, original)
            self.assertNotIn("form_G", cls.__dict__)
            self.assertNotIn("finalize", cls.__dict__)
            self.assertEqual(old, (self.d.tobytes(), self.j.tobytes(), self.k.tobytes()))
            value = p.read_json(Path(td)/"matrices/COMPLETE.json")
            self.assertEqual(value["finalized"], list(range(5)))
            self.assertEqual([x["calls"] for x in value["stages"]], [2]*5)
            for index in range(5):
                row = value["stages"][index]
                with np.load(Path(td)/"matrices"/row["file"], allow_pickle=False) as arrays:
                    self.assertEqual(set(arrays.files), {"Da", "J", "K"})
                    for name in arrays.files:
                        self.assertEqual(o.array_info(arrays[name]), row["arrays"][name])

    def test_native_exception_no_second_call_and_restore_partial(self):
        with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as td:
            cls, capture = self.make(Path(td))
            wfn = cls()
            wfn.stage, wfn.func, wfn.fail = 0, "HF", True
            with self.assertRaisesRegex(RuntimeError, "native failure"), capture.installed():
                wfn.form_G()
            self.assertEqual(wfn.native, 1)
            self.assertNotIn("form_G", cls.__dict__)
            self.assertTrue((Path(td)/"matrices/stage-0-before.json").exists())
            self.assertTrue((Path(td)/"matrices/partial-restored.json").exists())
            self.assertFalse((Path(td)/"matrices/COMPLETE.json").exists())

    def test_storage_budget_failure_preserves_partial_and_restores(self):
        with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as td:
            cls, capture = self.make(Path(td), cap=1)
            wfn = cls()
            wfn.stage, wfn.func = 0, "HF"
            with self.assertRaisesRegex(RuntimeError, "budget"), capture.installed():
                wfn.form_G()
            self.assertEqual(wfn.native, 1)
            self.assertNotIn("form_G", cls.__dict__)
            self.assertFalse(list((Path(td)/"matrices").glob("*.npz")))

    def test_missed_first_sad_wrong_stage_or_class_stop_before_native(self):
        for attribute, value in [("iteration_", 1), ("sad_", False), ("stage", 1)]:
            with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as td:
                cls, capture = self.make(Path(td))
                wfn = cls()
                wfn.stage, wfn.func = 0, "HF"
                setattr(wfn, attribute, value)
                with self.assertRaises(RuntimeError), capture.installed():
                    wfn.form_G()
                self.assertEqual(wfn.native, 0)

    def test_wrong_caller_no_surrogate(self):
        with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as td:
            cls, capture = self.make(Path(td))
            capture.stamp = o.stage_stamp
            wfn = cls()
            wfn.stage, wfn.func = 0, "HF"
            with self.assertRaisesRegex(RuntimeError, "actual staged"), capture.installed():
                wfn.form_G()
            self.assertEqual(wfn.native, 0)

    def test_actual_stamp_reads_popped_local_banner_and_iterator_identity(self):
        stage = Path("/exact/stage")
        namespace = dict(o=o, stage=stage)
        exec(compile("def scf_iterate(self):\n return o.stage_stamp(self, stage)\n",
             str(stage/"lib/psi4/driver/procrouting/scf_proc/scf_iterator.py"), "exec"), namespace)
        exec(compile("def scf_helper(wfn, **kwargs):\n banner=kwargs.pop('banner')\n return scf_iterate(wfn)\n",
             str(stage/"lib/psi4/driver/procrouting/proc.py"), "exec"), namespace)
        stamp = namespace["scf_helper"](object(), banner="SAPT(DFT): delta HF Dimer")
        self.assertEqual(stamp["banner"], "SAPT(DFT): delta HF Dimer")
        self.assertEqual(stamp["caller"]["function"], "scf_iterate")

    def test_fp32_wrong_shape_and_nonfinite_rejected(self):
        for val in (np.zeros((1786,1786), dtype=np.float32), np.zeros((2,2)), np.full((1786,1786), np.nan)):
            with self.assertRaises(RuntimeError):
                o.array(SimpleNamespace(np=val))

    def test_signed_zero_hash_and_numeric_equality_not_conflated(self):
        a, b = np.zeros((2, 2)), np.full((2, 2), -0.0)
        diff = o.matrix_difference(a, b)
        self.assertTrue(diff["exact_equal"])
        self.assertFalse(diff["bitwise_equal"])
        self.assertEqual(diff["maxabs"], 0)

    def test_matrix_delta_and_norm(self):
        a, b = np.ones((2, 2)), np.full((2, 2), 2.0)
        diff = o.matrix_difference(a, b)
        self.assertEqual(diff["maxabs"], 1)
        self.assertEqual(diff["normrel"], .5)

    def test_atomic_receipts_exclusive_never_rewrite(self):
        with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as td:
            file = Path(td)/"receipt.json"
            o.atomic(file, dict(first=True))
            with self.assertRaises(FileExistsError):
                o.atomic(file, dict(first=False))
            self.assertEqual(p.read_json(file), dict(first=True))


if __name__ == "__main__":
    unittest.main()
