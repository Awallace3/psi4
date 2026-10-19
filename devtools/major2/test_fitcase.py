"""Host-only source and mocked policy regressions; GPU J/K remains pending."""
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
from unittest.mock import Mock, patch

import artifacts as a
import campaign
import commonfix as c
import fitcase as f
import fitcase_handoff as h
import handoff
import jkexception as j
import policy as p
from test_jkexception import fresh_manifest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def source(pin, path=f.TEST_FILE):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", pin+":"+path]).decode()


def fixture():
    m = fresh_manifest()
    m.update(build_mode=f.MODE, run_root=f.RUN, build_budget_seconds=[0, 0], test_counts=f.COUNTS)
    m["harness"]["path"] = f.HARNESS
    m["approval"]["review_id"] = f.REVIEW_ID
    original = hashlib.sha256(source(j.CANDIDATE).encode()).hexdigest()
    m["arms"]["candidate"]["files"][f.TEST_FILE] = dict(mode="100644", sha256=original)
    m["arms"]["candidate"]["files_digest"] = p.digest(m["arms"]["candidate"]["files"])
    files = copy.deepcopy(m["arms"]["candidate"]["files"])
    files[f.TEST_FILE]["sha256"] = f.TEST_SHA
    m["test_source"] = dict(source=f.TEST_SOURCE, pin=f.TEST_PIN, tree=f.TEST_TREE,
                           files=files, files_digest=p.digest(files),
                           proof=dict(parent=j.CANDIDATE, diff_sha256=f.PATCH_SHA, test_sha256=f.TEST_SHA))
    m["selectors_ast"] = {p.PLAN: dict(sha256=f.TEST_SHA)}
    pair = dict(mode=c.MODE, review_id=c.REVIEW_ID, arms=copy.deepcopy(c.ALLOWLIST))
    for arm in pair["arms"]:
        pair["arms"][arm]["source_files_digest"] = m["arms"][arm]["files_digest"]
    m["artifact_pair"] = pair
    return m


class FitcaseTests(unittest.TestCase):
    def test_original_default_noop_and_new_genuine_effective_mutation(self):
        old = ast.parse(source(j.CANDIDATE))
        new = ast.parse(source(f.TEST_PIN))
        get = lambda tree: next(n for n in tree.body if isinstance(n, ast.FunctionDef) and
                                n.name == "test_cuest_changed_dependencies_rebuild")
        actions = next(n.value for n in get(old).body[2].body if isinstance(n, ast.Assign) and
                       any(isinstance(t, ast.Name) and t.id == "actions" for t in n.targets))
        old_action = dict(zip([ast.literal_eval(k) for k in actions.keys], actions.values))["fitting_condition"]
        default = 1.e-10
        self.assertIn('options.add_double("DF_FITTING_CONDITION", 1.0E-10);',
                      source(j.CANDIDATE, "psi4/src/read_options.cc"))
        value = [default]
        psi = SimpleNamespace(core=SimpleNamespace(get_option=lambda *_: value[0]),
                              set_options=lambda opts: value.__setitem__(0, opts["df_fitting_condition"]))
        import numpy as np
        scope = dict(psi4=psi, np=np)
        eval(compile(ast.Expression(old_action), "<old source lambda>", "eval"), scope)()
        self.assertEqual(value[0], default)  # ORIGINAL bug, not a mutation.
        helper = next(n for n in ast.walk(get(new)) if isinstance(n, ast.FunctionDef) and
                      n.name == "change_fitting_condition")
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "<exact new helper>", "exec"), scope)
        scope["change_fitting_condition"]()
        self.assertEqual(value[0], default*10)
        # Effective-option readback assertion catches an ignored setter.
        psi.set_options = lambda _: None
        with self.assertRaises(AssertionError):
            scope["change_fitting_condition"]()

    def test_other_mutations_source_audit(self):
        options = source(j.CANDIDATE, "psi4/src/read_options.cc")
        jk = source(j.CANDIDATE, "psi4/src/psi4/libfock/jk.cc")
        for literal in ['CUEST_DFK_SLICES", 5', 'CUEST_DFK_MODULI", 8',
                        'CUEST_MIXED_PRECISION", true', 'INTS_TOLERANCE", 1E-12']:
            self.assertIn(literal, options)
        for literal in ['omega_ = 0.0;', 'omega_alpha_ = 1.0;', 'omega_beta_ = 0.0;', 'cutoff_ = 1.0E-12;']:
            self.assertIn(literal, jk)
        text = source(f.TEST_PIN)
        self.assertIn('"cuest_mixed_precision": False', text)  # setup overrides default true
        for literal in ['set_omega_alpha(0.25)', 'set_omega_beta(0.2)', 'set_omega(0.3)',
                        '"cuest_mixed_precision": True', '"cuest_dfk_slices": 6',
                        '"cuest_dfk_moduli": 7', 'set_cutoff(1.e-11)', '"ints_tolerance": 1.e-11']:
            self.assertIn(literal, text)

    def test_exact_test_only_direct_child_and_patch(self):
        root = ROOT.parent/"saptdft-cuest-major2-fitcase-tests-20261019"
        self.assertEqual(f.test_proof(root),
                         dict(parent=j.CANDIDATE, diff_sha256=f.PATCH_SHA, test_sha256=f.TEST_SHA))
        original = ast.parse(source(j.CANDIDATE))
        corrected = ast.parse(source(f.TEST_PIN))
        funcs = lambda tree: {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
        old, new = funcs(original), funcs(corrected)
        self.assertEqual(set(new)-set(old), {"test_cuest_same_fitting_condition_reuses_plan"})
        for name in old.keys()-{"test_cuest_changed_dependencies_rebuild"}:
            self.assertEqual(old[name], new[name])
        old_calls = [ast.dump(n) for n in ast.walk(original) if isinstance(n, ast.Call) and
                     isinstance(n.func, ast.Name) and n.func.id == "_equal"]
        new_calls = [ast.dump(n) for n in ast.walk(corrected) if isinstance(n, ast.Call) and
                     isinstance(n.func, ast.Name) and n.func.id == "_equal"]
        for call in old_calls:
            self.assertIn(call, new_calls)  # fresh-object numerical comparisons preserved
        self.assertEqual(source(j.CANDIDATE, j.RANGE_TEST), source(f.TEST_PIN, j.RANGE_TEST))

    def test_samevalue_source_contract_and_reviewed_seventeen_count(self):
        tree = ast.parse(source(f.TEST_PIN))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and
                    n.name == "test_cuest_same_fitting_condition_reuses_plan")
        text = ast.unparse(node)
        self.assertIn("{'df_fitting_condition': before}", text)
        self.assertEqual(text.count("_creations(trace) == 1"), 3)
        self.assertIn("_equal(reference, _compute(jk, primary.nbf()))", text)
        # AST expansion of the only parameter matrix, not a fake GPU collection.
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
        self.assertEqual(len(functions)+8, 17)

    def test_new_manifest_failclosed_separate_test_and_scientific_pins(self):
        m = fixture()
        with patch.object(c, "PAIR_DIGEST", p.digest(m["artifact_pair"])):
            p.validate_manifest(m)
            for mutate in [
                lambda x: x.update(build_budget_seconds=[3600, 3600]),
                lambda x: x.update(test_smoke_budget_seconds=100),
                lambda x: x.update(test_counts=dict(f.COUNTS, candidate_plan=16)),
                lambda x: x["test_source"].update(pin=j.CANDIDATE),
                lambda x: x["test_source"].update(tree="0"*40),
                lambda x: x["test_source"].update(source=j.SOURCES["candidate"]),
                lambda x: x["test_source"]["proof"].update(parent=p.OBSERVED_CANDIDATE),
                lambda x: x["test_source"]["proof"].update(diff_sha256="0"*64),
                lambda x: x["test_source"]["files"].update({"injected-driver": dict(sha256="0"*64)}),
                lambda x: x["selectors_ast"][p.PLAN].update(sha256="0"*64),
                lambda x: x["arms"]["candidate"].update(pin=f.TEST_PIN),
                lambda x: x["artifact_pair"]["arms"]["baseline"].update(origin_job="14164895"),
                lambda x: x["artifact_pair"]["arms"]["candidate"].update(core_sha256=a.ALLOWLIST["candidate"]["core_sha256"]),
                lambda x: x["artifact_pair"]["arms"]["baseline"].update(build="/any/existing/dir"),
            ]:
                bad = copy.deepcopy(m)
                mutate(bad)
                with self.subTest(mutate=mutate), self.assertRaises(RuntimeError):
                    p.validate_manifest(bad)

    def test_actual_captured_pair_and_legacy_api_unchanged(self):
        pair = p.read_json(os.environ["MAJOR2_PAIR_JSON"])
        c.validate_pair(pair)
        for arm in pair["arms"]:
            item = pair["arms"][arm]
            self.assertEqual(item["stage_files"]["lib/psi4/core.cpython-313-x86_64-linux-gnu.so"], c.CORE[arm])
            self.assertIn("abi-after-build/result.json", item["metadata_files"])
        for name in ["artifacts.py", "test_harness.py", "test_jkexception.py", "run_pytest.py"]:
            self.assertEqual(source("ede11975fd80a7f42bd2cff5a3029a2ea029ef71", "devtools/major2/"+name),
                             (HERE/name).read_text())
        with self.assertRaises(RuntimeError):
            a.validate_pair(pair)
        with self.assertRaises(RuntimeError):
            c.validate_pair(dict(mode="verified-artifact-pair", review_id=a.REVIEW_ID, arms=a.ALLOWLIST))

    def test_all_suites_normal_test_paths_and_staged_arm_wrappers(self):
        # Observe argv/count without running pytest/GPU; no chemistry monkeypatch.
        engine = object.__new__(campaign.Engine)
        with tempfile.TemporaryDirectory() as name:
            engine.m = fixture()
            engine.root = Path(name)
            engine.raw, engine.scratch = Path(name)/"raw", Path(name)/"scratch"
            engine.raw.mkdir()
            engine.scratch.mkdir()
            engine.end, engine.env = 1.e20, {}
            engine.pins = Mock()
            class Captured(Exception):
                pass
            captured = []
            def run(argv, label, **kwargs):
                captured.append((argv, kwargs["env"]["MAJOR2_EXPECTED_COUNT"]))
                raise Captured()
            engine.run_command = run
            for arm, selectors, expression, red, count in [
                ("baseline", p.COMMON, None, False, "48"),
                ("candidate", [p.FIXED], p.FIXED_K, False, "4"),
                ("baseline", [p.RED], None, True, "1"),
                ("candidate", [p.PLAN], None, False, "17")]:
                with self.assertRaises(Captured):
                    engine.tests(arm, "fake-"+count, selectors, expression, red)
                argv, got = captured[-1]
                self.assertEqual(got, count)
                self.assertEqual(argv[2], j.SOURCES[arm]+"/"+j.BUILD+"/stage")
                self.assertIn(f.TEST_SOURCE+"/pytest.ini", list(map(str, argv)))
                for selector in selectors:
                    self.assertIn(f.TEST_SOURCE+"/tests/pytests/"+selector, list(map(str, argv)))

    def test_new_adoption_never_builds_and_retains_gate_order(self):
        from test_harness import HarnessTests
        helper = HarnessTests()
        helper.setUp()
        try:
            engine = helper.fake_engine()
            engine.raw.rmdir()
            engine.m = fixture()
            engine.manifest_path = helper.root/"manifest.json"
            engine.manifest_path.write_text("{}")
            engine.prolog, engine.inventory, engine.build = Mock(), Mock(), Mock()
            engine.drivers = dict(baseline={}, candidate={})
            engine.configurations = dict(baseline={}, candidate={})
            engine.adopt = Mock()
            class Stop(Exception):
                pass
            engine.tests = Mock(side_effect=Stop)
            engine.records, engine.sampler = [], None
            engine.job = "FAKE"
            with patch.object(campaign.signal, "signal"), self.assertRaises(Stop):
                engine.execute()
            engine.build.assert_not_called()
            self.assertEqual(engine.adopt.call_args_list,
                             [unittest.mock.call("candidate"), unittest.mock.call("baseline")])
            self.assertEqual(engine.tests.call_args.args, ("baseline", "baseline-common", p.COMMON))
        finally:
            helper.doCleanups()

    def test_new_adopt_copied_origin_receipts_not_compilation(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            meta = root/"origin"
            meta.mkdir()
            (meta/"BUILD_OK").write_text(j.CANDIDATE+" cuda\n")
            hashes = a.hashed_tree(meta)
            item = dict(metadata=str(meta), build="/exact/approved/build", origin_job="14165998",
                        pin=j.CANDIDATE, core_sha256=c.CORE["candidate"],
                        metadata_files=hashes)
            engine = object.__new__(campaign.Engine)
            engine.root, engine.end = root, 1.e20
            engine.m = dict(build_mode=f.MODE, artifact_pair={"arms": {"candidate": item}},
                            arms={"candidate": {"files": {}}})
            engine.pins, engine.verify_build = Mock(), Mock()
            with patch.object(c, "verify_artifact") as verify:
                engine.adopt("candidate")
                verify.assert_called_once()
            proof = p.read_json(root/"build-candidate/BUILD_ORIGIN.json")
            self.assertEqual(proof["origin_job"], "14165998")
            self.assertTrue(proof["copied_receipts"])
            self.assertFalse(proof["compiled_in_this_job"])
            self.assertEqual(proof["artifact_pair_digest"], c.PAIR_DIGEST)
            self.assertEqual(proof["compiled_source_pin"], j.CANDIDATE)
            self.assertEqual(proof["correctness_test_source_pin"], f.TEST_PIN)
            self.assertEqual(a.hashed_tree(meta), hashes)
            engine.verify_build.assert_called_once_with("candidate")

    def test_commonfix_artifact_byte_and_normalized_config_rejection(self):
        from test_harness import HarnessTests
        helper = HarnessTests()
        helper.setUp()
        try:
            source_root, files, stage = helper.driver_fixture()
            build = stage.parent
            inner = build/"psi4-core-prefix/src/psi4-core-build"
            inner.mkdir(parents=True)
            cache = build/"CMakeCache.txt"
            cache.write_text("FLAGS:STRING=-O3\n")
            (inner/"CMakeCache.txt").write_bytes(cache.read_bytes())
            (inner/"compile_commands.json").write_text('[{"command":"fake -O3"}]')
            core = stage/"lib/psi4/core.cpython-313-x86_64-linux-gnu.so"
            core.write_bytes(b"fake-core")
            meta = helper.root/"origin"
            meta.mkdir()
            (meta/"BUILD_OK").write_text(j.CANDIDATE+" cuda\n")
            item = dict(source=str(source_root), pin=j.CANDIDATE, build=str(build), metadata=str(meta),
                        source_files_digest=p.digest(files), core_sha256=p.sha(core),
                        stage_files=a.hashed_tree(stage), build_files={"CMakeCache.txt": p.sha(cache)},
                        metadata_files=a.hashed_tree(meta))
            pair = dict(arms=dict(candidate=item), normalized_configs_digest=p.digest(a.normalized_configs(item)),
                        normalized_commands_digest=p.digest(a.normalized_compile_commands(item)))
            with patch.object(c, "validate_pair"), patch.object(p, "check_source"), patch.object(p, "cmake_policy"):
                c.verify_artifact(pair, "candidate", files)
                for file in [meta/"BUILD_OK", core, cache, inner/"compile_commands.json"]:
                    old = file.read_bytes()
                    file.write_bytes(b"changed")
                    with self.subTest(file=file), self.assertRaises((RuntimeError, ValueError)):
                        c.verify_artifact(pair, "candidate", files)
                    file.write_bytes(old)
                pair["normalized_configs_digest"] = "0"*64
                with self.assertRaisesRegex(RuntimeError, "normalized configs"):
                    c.verify_artifact(pair, "candidate", files)
        finally:
            helper.doCleanups()

    def test_new_parent_one_shot_id_order_and_origin_claim(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/"logs").mkdir()
            gates = dict(harness_pin="pin", manifest_sha256="m", batch_sha256="b", preflight_sha256="h",
                         pristine_staging=True, old_run_unchanged=True, old_build_sentinels_unchanged=True,
                         test_only_stdout="gpu-h200", test_only_stderr="")
            p.write_json(root/"logs/pre-submit-gates.json", gates)
            def fail(*_):
                lock = root/"logs/submission.lock"
                self.assertEqual((lock/"job-id.txt").read_text(), "12345\n")
                attempt = p.read_json(lock/"attempt.json")
                self.assertFalse(attempt["compiled_in_new_job"])
                self.assertEqual(attempt["replacement_for"], "14165998")
                raise RuntimeError("acceptance unavailable")
            with patch.object(f, "RUN", str(root)), patch.object(handoff, "queue", return_value=[]), \
                    patch.object(p, "sha", return_value="h"), patch.object(handoff, "output", side_effect=fail), \
                    patch.object(handoff.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="12345\n", stderr="")):
                kwargs = dict(cohort=f, verifier=Mock(return_value=({}, {})), cache=root, failed_jobs=h.FAILED_JOBS)
                with self.assertRaisesRegex(RuntimeError, "acceptance unavailable"):
                    handoff.submit("pin", "m", "b", **kwargs)
                with self.assertRaises(FileExistsError):
                    handoff.submit("pin", "m", "b", **kwargs)

    def test_batch_namespace_and_unchanged_faulthandler(self):
        text = campaign.batch_text("a"*64, commonfix=True)
        self.assertIn(f.HARNESS+"/devtools/major2/supervise.sh "+f.RUN, text)
        self.assertNotIn(j.HARNESS+"/", text)
        self.assertIn("--qos=embers", text)
        self.assertIn("--no-requeue", text)
        with tempfile.NamedTemporaryFile(mode="w") as stream:
            stream.write(text)
            stream.flush()
            subprocess.run(["bash", "-n", stream.name], check=True)
        self.assertEqual((HERE/"run_pytest.py").read_text(),
                         source("ede11975fd80a7f42bd2cff5a3029a2ea029ef71", "devtools/major2/run_pytest.py"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
