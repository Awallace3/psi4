"""Policy negatives + fake-command hybrid order, not GPU numerical evidence."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import campaign
import commonfix as c
import fitcase as f
import handoff
import jkexception as j
import nonc1 as n
import nonc1_handoff as h
import policy as p
from test_jkexception import fresh_manifest

ROOT = Path(__file__).resolve().parents[2]


def fixture():
    m = fresh_manifest()
    m.pop("shared_fix")
    m.update(build_mode=n.MODE, run_root=n.RUN, build_budget_seconds=[0, 3600],
             constructor_proof=n.PROOF, test_counts=n.COUNTS)
    m["harness"]["path"] = n.HARNESS
    m["approval"].update(review_id=n.REVIEW_ID, candidate_pin=n.CANDIDATE)
    pair = p.read_json(os.environ["MAJOR2_PAIR_JSON"])
    origin = n.baseline_origin(pair)
    m["baseline_origin"] = origin
    # Use complete REAL source manifests, not synthetic source digests.
    for arm in ("baseline", "candidate"):
        local = ROOT.parent/("saptdft-cuest-major2-jkexception-baseline-20261018" if arm == "baseline"
                            else "saptdft-cuest-major2-nonc1-candidate-20261019")
        files = p.file_manifest(local)
        m["arms"][arm].update(source=n.SOURCES[arm], build=n.BUILDS[arm], pin=n.PINS[arm],
                             files=files, files_digest=p.digest(files))
    test = ROOT.parent/"saptdft-cuest-major2-nonc1-tests-20261019"
    files = p.file_manifest(test)
    m["test_source"] = dict(source=n.TEST_SOURCE, pin=n.TEST_PIN, tree=n.TEST_TREE, proof=n.TEST_PROOF,
                           files=files, files_digest=p.digest(files))
    m["selectors_ast"] = p.selectors_ast(test)
    return m


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = fixture()

    def test_strict_mode_paths_pins_origins_budgets_source_test_negatives(self):
        p.validate_manifest(copy.deepcopy(self.frozen))
        for change in [
            lambda m: m.update(build_mode=j.MODE),
            lambda m: m.update(artifact_pair={}),
            lambda m: m.update(run_root=f.RUN),
            lambda m: m["harness"].update(path=f.HARNESS),
            lambda m: m.update(build_budget_seconds=[0, 0]),
            lambda m: m.update(build_budget_seconds=[3600, 3600]),
            lambda m: m.update(test_smoke_budget_seconds=600),
            lambda m: m.update(production_budget_seconds=100),
            lambda m: m.update(per_process_timeout_seconds=601),
            lambda m: m["approval"].update(review_id=f.REVIEW_ID),
            lambda m: m["arms"]["candidate"].update(pin=j.CANDIDATE),
            lambda m: m["arms"]["candidate"].update(build=j.SOURCES["candidate"]+"/"+j.BUILD),
            lambda m: m["arms"]["baseline"].update(source=p.SOURCES["baseline"]),
            lambda m: m["baseline_origin"]["arms"].update(candidate=c.ALLOWLIST["candidate"]),
            lambda m: m["baseline_origin"]["arms"]["baseline"].update(origin_job="14169328"),
            lambda m: m["baseline_origin"]["arms"]["baseline"].update(core_sha256="0"*64),
            lambda m: m["baseline_origin"].update(normalized_configs_digest="0"*64),
            lambda m: m["test_source"].update(pin=f.TEST_PIN),
            lambda m: m["test_source"].update(source=f.TEST_SOURCE),
            lambda m: m["test_source"]["proof"].update(parent=j.CANDIDATE),
            lambda m: m["constructor_proof"].update(constructor_patch_sha256="0"*64),
            lambda m: m["test_counts"].update(candidate_plan=17),
            lambda m: m["test_source"]["files"][j.JK].update(sha256="0"*64),
            lambda m: m["arms"]["candidate"]["files"][n.CPP].update(sha256="0"*64),
        ]:
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                value = copy.deepcopy(self.frozen)
                change(value)
                p.validate_manifest(value)

    def test_actual_source_and_one_test_child_proofs(self):
        local = {arm: ROOT.parent/("saptdft-cuest-major2-jkexception-baseline-20261018" if arm == "baseline"
                                  else "saptdft-cuest-major2-nonc1-candidate-20261019") for arm in n.PINS}
        self.assertEqual(n.proof(local), n.PROOF)
        self.assertEqual(n.test_proof(ROOT.parent/"saptdft-cuest-major2-nonc1-tests-20261019"), n.TEST_PROOF)
        g = lambda pin, file: subprocess.check_output(["git", "-C", str(ROOT), "show", pin+":"+file], text=True)
        old, new = g(f.TEST_PIN, n.TEST_FILE), g(n.TEST_PIN, n.TEST_FILE)
        funcs = lambda s: {x.name: ast.get_source_segment(s, x) for x in ast.parse(s).body if isinstance(x, ast.FunctionDef)}
        of, nf = funcs(old), funcs(new)
        self.assertEqual(set(nf)-set(of), {"test_cuest_material_preconstruction_displacement_fails_closed"})
        for name, text in of.items():
            self.assertEqual(text, nf[name], name)  # includes original non-C1 BYTE-identical
        material = nf["test_cuest_material_preconstruction_displacement_fails_closed"]
        self.assertIn("0.125", material)
        self.assertIn("psi4.core.JK.build_JK(primary, aux)", material)
        self.assertNotIn("initialize", material)
        # 10 single functions + 9 param values - 1 param function = 18.
        tree = ast.parse(new)
        count = 0
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                values = [ast.literal_eval(d.args[1]) for d in node.decorator_list if isinstance(d, ast.Call)
                          and isinstance(d.func, ast.Attribute) and d.func.attr == "parametrize"]
                count += len(values[0]) if values else 1
        self.assertEqual(count, 18)

    def test_existing_72_checks_and_legacy_apis_byte_identical(self):
        for name in ("test_harness.py", "test_jkexception.py", "test_fitcase.py", "artifacts.py", "commonfix.py",
                     "fitcase.py", "jkexception.py", "science.py", "build.sh", "arm.sh", "run_pytest.py"):
            old = subprocess.check_output(["git", "-C", str(ROOT), "show",
                   "11e852dc4a41835f5c27bc198b01b8fb644d03ac:devtools/major2/"+name])
            self.assertEqual(old, (ROOT/"devtools/major2"/name).read_bytes(), name)

    def test_new_batch_and_no_candidate_adoption(self):
        batch = campaign.batch_text("a"*64, hybrid=True)
        self.assertIn(n.HARNESS+"/devtools/major2/supervise.sh "+n.RUN, batch)
        self.assertIn("--qos=embers", batch)
        self.assertIn("--no-requeue", batch)
        self.assertNotIn(f.RUN, batch)
        engine = object.__new__(campaign.Engine)
        engine.m = self.frozen
        engine.end = 10**20
        engine.pins = lambda: None
        with self.assertRaisesRegex(RuntimeError, "cannot import old CAND"):
            engine.adopt("candidate")
        with self.assertRaisesRegex(RuntimeError, "fresh candidate ONLY"):
            engine.build("baseline", 1)

    def test_fake_command_fresh_candidate_then_baseline_before_all_gates(self):
        # Real execute() ordering, fake external commands. Stop at first test gate.
        class ReachedGate(Exception):
            pass
        calls = []
        with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as td:
            engine = object.__new__(campaign.Engine)
            engine.m = copy.deepcopy(self.frozen)
            engine.root = Path(td)
            engine.raw = engine.root/"raw"  # deliberately absent; no inventory walk
            engine.manifest_path = engine.root/"manifest.json"
            engine.manifest_path.write_text("{}")
            engine.job, engine.signalled, engine.records, engine.sampler = "123", False, [], None
            engine.prolog = lambda: calls.append("prolog")
            engine.drivers = dict(baseline={}, candidate={})
            engine.configurations = dict(baseline={}, candidate={})
            engine.build = lambda arm, remaining: calls.append(("fresh", arm, remaining))
            engine.adopt = lambda arm: calls.append(("adopt", arm))
            def gate(*args):
                calls.append(("tests", args))
                raise ReachedGate()
            engine.tests = gate
            with patch.object(campaign.signal, "signal"), \
                 patch.object(campaign.artifacts, "normalized_configs", return_value={}), \
                 patch.object(campaign.artifacts, "normalized_compile_commands", return_value=[]), \
                 self.assertRaises(ReachedGate):
                engine.execute()
        self.assertEqual(calls[:3], ["prolog", ("fresh", "candidate", 1), ("adopt", "baseline")])
        self.assertEqual(calls[3][1][0:2], ("baseline", "baseline-common"))

    def test_baseline_origin_receipt_copied_and_candidate_origin_truth(self):
        # Exercise real adopt copy/receipt control flow, fake validation only.
        with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as td:
            root = Path(td)
            meta = root/"old"
            meta.mkdir()
            (meta/"BUILD_OK").write_text(n.BASE+" cuda\n")
            m = copy.deepcopy(self.frozen)
            origin = m["baseline_origin"]["arms"]["baseline"]
            origin["metadata"] = str(meta)
            origin["metadata_files"] = {"BUILD_OK": p.sha(meta/"BUILD_OK")}
            engine = object.__new__(campaign.Engine)
            engine.m, engine.root, engine.end = m, root, 10**20
            engine.pins = lambda: None
            engine.verify_build = lambda _: None
            with patch.object(n, "verify_baseline"):
                engine.adopt("baseline")
            receipt = p.read_json(root/"build-baseline/BUILD_ORIGIN.json")
            self.assertEqual(receipt["origin_job"], "14165998")
            self.assertFalse(receipt["compiled_in_this_job"])
            self.assertTrue(receipt["copied_receipts"])
            self.assertEqual(receipt["compiled_source_pin"], n.BASE)
            self.assertEqual(receipt["correctness_test_source_pin"], n.TEST_PIN)
            self.assertEqual((meta/"BUILD_OK").read_text(), n.BASE+" cuda\n")

    def test_hybrid_parent_id_durable_and_lock_not_reusable_on_ambiguity(self):
        with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as td:
            root = Path(td)
            (root/"logs").mkdir()
            gates = dict(harness_pin="pin", manifest_sha256="m", batch_sha256="b", preflight_sha256="h",
                         pristine_staging=True, old_run_unchanged=True, old_build_sentinels_unchanged=True,
                         test_only_stdout="gpu-h200", test_only_stderr="")
            p.write_json(root/"logs/pre-submit-gates.json", gates)
            def acceptance(*args):
                self.assertEqual((root/"logs/submission.lock/job-id.txt").read_text(), "12345\n")
                attempt = p.read_json(root/"logs/submission.lock/attempt.json")
                self.assertEqual(attempt["replacement_for"], "14169328")
                self.assertFalse(attempt["build_origins"]["baseline"]["compiled_in_this_job"])
                self.assertTrue(attempt["build_origins"]["candidate"]["compiled_in_this_job"])
                raise RuntimeError("acceptance unavailable")
            with patch.object(n, "RUN", str(root)), patch.object(handoff, "queue", return_value=[]), \
                 patch.object(p, "sha", return_value="h"), \
                 patch.object(handoff.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="12345\n", stderr="")), \
                 patch.object(handoff, "output", side_effect=acceptance):
                for exc in (RuntimeError, FileExistsError):
                    with self.assertRaises(exc):
                        handoff.submit("pin", "m", "b", cohort=n, verifier=lambda *a: ({}, {}),
                                       cache=root, failed_jobs=h.FAILED_JOBS)
