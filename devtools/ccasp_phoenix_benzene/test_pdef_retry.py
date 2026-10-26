# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Engine-free retry adapters: real driver kwargs, closed plan, honest receipts."""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

import test_pdef_full as F
from pdef_boundary import drive

HERE = Path(__file__).resolve().parent
load = F.A.load
J = load("retry_test_job", "pdef_retry_job.py")
P = load("retry_test_profile", "pdef_retry_profile.py")
D = load("retry_test_driver", "pdef_retry_driver.py")


class ProfileTests(unittest.TestCase):
    def test_exact_science_parent_and_actual_new_plan_identity_no_global_mutation(self):
        plans = [copy.deepcopy(r.PLAN) for r in (J.bootstrap, J.R, F.J.R, F.A.R)]
        cfg, identity = P.configuration(J.R)
        old_cfg, old_identity = F.P.configuration(F.J.R)
        self.assertEqual(cfg, old_cfg)
        self.assertEqual(F.P.canonical_sha(cfg), P.CONFIG_SHA)
        self.assertEqual(old_identity["identity_sha256"], P.PDEF_IDENTITY)
        self.assertEqual(identity["parent_pdef_profile"], old_identity)
        self.assertNotEqual(identity["identity_sha256"], P.PDEF_IDENTITY)
        self.assertEqual(identity["plan_sha256"], J.R.sha(HERE/"pdef_retry_plan.json"))
        self.assertNotEqual(identity["plan_sha256"], old_identity["plan_sha256"])
        self.assertEqual(identity["identity_sha256"], F.P.canonical_sha(
            {k: v for k, v in identity.items() if k != "identity_sha256"}))
        self.assertEqual(identity, J.R.effective_profile())
        for name in ("pdef_retry_profile.py", "pdef_retry_driver.py", "pdef_retry_job.py",
                     "pdef_retry_full.sbatch", "pdef_retry_plan.json", "pdef_profile.py",
                     "integrated_runtime.py", "integrated_driver.py", "pdef_driver.py"):
            self.assertEqual(identity["files_sha256"][name], J.R.sha(HERE/name))
        for key in ("native", "fixture", "prior_probe", "prior_probe_sha256", "prior_driver_source",
                    "prior_driver_commit", "prior_driver_tree", "prior_driver_files",
                    "fingerprints", "allocation", "routing", "overrides", "parent_profile_sha256"):
            self.assertEqual(J.R.PLAN[key], F.J.R.PLAN[key])
        self.assertEqual(plans, [r.PLAN for r in (J.bootstrap, J.R, F.J.R, F.A.R)])
        self.assertIsNone(J.bootstrap.effective_profile())
        self.assertEqual(F.J.R.effective_profile(), old_identity)

    def test_all_path_schema_pin_science_and_scheduling_changes_fail_closed(self):
        mutations = [
            lambda p, key=key: p.update({key: "/arbitrary"})
            for key in ("source", "root", "runtime", "run", "scratch", "fixture", "prior_probe")]
        mutations += [
            lambda p: p.update(schema="ccasp-benzene-full-pdef-plan/1"),
            lambda p: p.update(effective_profile_module="pdef_profile.py"),
            lambda p: p.update(extra_override=True),
            lambda p: p.update(activation_authority="old grant"),
            lambda p: p.update(parent_pdef_identity_sha256="0"*64),
            lambda p: p.update(effective_config_sha256="0"*64),
            lambda p: p["retry_grant"].update(fresh_allocations=2),
            lambda p: p["retry_grant"].update(automatic_further_retry=True),
            lambda p: p["native"].update(core_sha256="0"*64),
            lambda p: p["fingerprints"].update(main="0"*64),
            lambda p: p.update(prior_probe_sha256="0"*64),
            lambda p: p["overrides"].update({"isa.q_grid": [200, 590]}),
            lambda p: p["overrides"]["refinement.declared_variables"].pop()]
        mutations += [lambda p, key=k, value=v: p["allocation"].update({key: value})
            for k, v in (("cpus", 16), ("memory", "64G"), ("time", "08:00:00"),
                ("node", "any"), ("qos", "inferno"), ("nodes", True),
                ("numerical_threads", 8), ("psi4_memory_bytes", 24*1024**3))]
        for number, mutate in enumerate(mutations):
            plan = copy.deepcopy(J.R.PLAN)
            mutate(plan)
            with self.subTest(mutation=number), mock.patch.object(J.R, "PLAN", plan):
                with self.assertRaises(ValueError):
                    P.configuration(J.R)
                with self.assertRaises(ValueError):
                    J.argv("a"*40, "b"*40, "c"*64, "d"*64)

    def test_actual_plan_bytes_and_plan_path_must_agree_and_cfg_is_frozen(self):
        read_text = Path.read_text
        for name, mutate in (
            ("pdef_retry_plan.json", lambda p: p.update(unknown=1)),
            ("pilot.json", lambda p: p["response"].update(production_tolerance=1e-5))):
            def changed(path, *a, **kw):
                data = read_text(path, *a, **kw)
                if path == HERE/name:
                    value = json.loads(data)
                    mutate(value)
                    return json.dumps(value)
                return data
            with mock.patch.object(Path, "read_text", changed), self.assertRaises(ValueError):
                P.configuration(J.R)
        with mock.patch.object(J.R, "PLAN_PATH", HERE/"pdef_plan.json"), self.assertRaises(ValueError):
            P.configuration(J.R)

    def test_stdlib_only_isolated_imports(self):
        script = ("import importlib.util,sys;"
            f"s=importlib.util.spec_from_file_location('j',{str(HERE/'pdef_retry_job.py')!r});"
            "j=importlib.util.module_from_spec(s);sys.modules[s.name]=j;s.loader.exec_module(j);"
            "j.R.effective_profile();"
            f"s=importlib.util.spec_from_file_location('d',{str(HERE/'pdef_retry_driver.py')!r});"
            "d=importlib.util.module_from_spec(s);s.loader.exec_module(d);"
            "assert not {'numpy','scipy','psi4','adcc','einsums'} & set(sys.modules)")
        subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script], check=True)

    def test_source_and_derivative_bind_new_adapters_and_keep_original_probe(self):
        real_sha = J.R.sha
        local_root = J.R.ROOT
        def mapped_sha(path):
            path = Path(path)
            return real_sha(local_root/path.relative_to(P.SOURCE)) if path.is_relative_to(Path(P.SOURCE)) else real_sha(path)
        with mock.patch.object(J.R, "ROOT", Path(P.SOURCE)), \
             mock.patch.object(J.R.OLD, "identity"), \
             mock.patch.object(J.R.OLD, "git", return_value=""), \
             mock.patch.object(J.R, "sha", side_effect=mapped_sha):
            binding = J.R.source_binding("a"*40, "b"*40)
            original = {"binding": {"original_profile": F.P.PARENT_SHA, "private_probe": "unchanged"}}
            with mock.patch.object(J.R, "prior_probe", return_value=original):
                doc = J.R.derived_document(binding, "c"*64)
        self.assertEqual(doc["original_probe"]["binding"], original["binding"])
        self.assertEqual(doc["original_probe"]["sha256"], F.J.R.PLAN["prior_probe_sha256"])
        self.assertEqual(doc["effective_profile"], binding["effective_profile"])
        for name in ("pdef_retry_job.py", "pdef_retry_driver.py", "pdef_retry_plan.json"):
            self.assertEqual(binding["tools_sha256"][name], real_sha(HERE/name))
        self.assertEqual(binding["native"], {
            k: F.J.R.PLAN["native"][k] for k in ("source", "commit", "tree", "stage", "core_sha256")})


class DriverTests(unittest.TestCase):
    def test_real_wrapper_exact_configuration_kwargs_and_reused_validator(self):
        captured = []
        for job, profile, driver in ((F.J, F.P, F.D), (J, P, D)):
            module = job.R.module
            shared = NS(capture_science=mock.Mock(return_value={"science_pass": False}))
            def route(name, file):
                return shared if file == "integrated_driver.py" else module(name, file)
            _, identity = profile.configuration(job.R)
            fake = {"psi4.driver.procrouting": NS(isapol_lw=NS(PRODUCTION_TOLERANCE=1e-6))}
            with mock.patch.dict(sys.modules, fake), mock.patch.object(job.R, "compute_guard"), \
                 mock.patch.object(job.R, "module", side_effect=route), mock.patch.object(job.R, "write"):
                driver.capture_science(job.R, object(), {"effective_profile": identity}, "r", "d")
            captured.append(shared.capture_science.call_args.kwargs)
        self.assertEqual(captured[0]["configuration"], captured[1]["configuration"])
        self.assertEqual(len(captured[1]["configuration"]["refinement"]["declared_variables"]), 72)
        self.assertEqual(captured[1]["profile_identity"], P.configuration(J.R)[1])
        self.assertIs(captured[1]["validate_result"], D.parent_driver.validate_complete)
        self.assertEqual(captured[0]["validate_result"].__code__.co_code,
                         captured[1]["validate_result"].__code__.co_code)

    def test_actual_shared_pipeline_kwargs_models_all72_and_static_penalties_unchanged(self):
        # Reuse these already source-loaded wrappers rather than registering a
        # second for_plan module with the same deterministic sys.modules name.
        # No profile/scientific boundary is stubbed by this loader selection.
        wrappers = {"pdef_job.py": F.J, "pdef_profile.py": F.P, "pdef_driver.py": F.D,
            "pdef_retry_job.py": J, "pdef_retry_profile.py": P, "pdef_retry_driver.py": D}
        with mock.patch("pdef_boundary.load", side_effect=lambda name, file: wrappers[file]):
            frozen, retry = drive("pdef"), drive("pdef_retry")
        for key in ("memory", "grid", "exported", "models", "sites", "nodes"):
            if key in ("sites", "nodes"):  # contain NumPy arrays
                continue
            self.assertEqual(frozen[key], retry[key])
        for key in ("declared_variables", "cutoff", "weight_type", "weight_coefficient",
                    "npoints", "max_refinement_parameters", "localization_rank_limit",
                    "shell_cutoff", "charge_penalty", "anchor_metric_damping"):
            self.assertEqual(frozen["kwargs"][key], retry["kwargs"][key])
        self.assertEqual(vars(frozen["kwargs"]["resources"]), vars(retry["kwargs"]["resources"]))
        self.assertEqual(len(retry["models"]), 11)
        self.assertEqual(retry["models"][0].unpenalized_variables, ("H1_11s_21c_A",))
        self.assertEqual(len(retry["models"][0].parameter_labels), 72)
        # Constructor-only synthetic results do not establish native Solved.

    def test_original_identity_or_lw_change_refused_before_shared_science(self):
        cfg, identity = P.configuration(J.R)
        for binding, tolerance in ((F.P.configuration(F.J.R)[1], 1e-6), (identity, 1e-5)):
            fake = {"psi4.driver.procrouting": NS(isapol_lw=NS(PRODUCTION_TOLERANCE=tolerance))}
            module = J.R.module
            shared = mock.Mock()
            def route(name, file):
                return shared if file == "integrated_driver.py" else module(name, file)
            with mock.patch.dict(sys.modules, fake), mock.patch.object(J.R, "compute_guard"), \
                 mock.patch.object(J.R, "module", side_effect=route), mock.patch.object(J.R, "write"), \
                 self.assertRaises(ValueError):
                D.capture_science(J.R, object(), {"effective_profile": binding}, "r", "d")
            shared.capture_science.assert_not_called()


class OperationalTests(unittest.TestCase):
    def test_argv_exact_new_batch_concrete_pins_and_compute_guard(self):
        args = J.argv("a"*40, "b"*40, "c"*64, "d"*64)
        self.assertEqual(args, ["/opt/slurm/current/bin/sbatch", "--test-only", "--export=NONE",
            "--chdir="+J.R.PLAN["run"], P.SOURCE+"/devtools/ccasp_phoenix_benzene/pdef_retry_full.sbatch",
            "a"*40, "b"*40, "c"*64, "d"*64])
        self.assertEqual(J.argv(*args[-4:], test_only=False)[1], "--parsable")
        for value in (None, "PLACEHOLDER", "A"*64, "a"*63):
            with self.assertRaises(ValueError):
                J.argv("a"*40, "b"*40, value, "d"*64)
        with mock.patch.object(J.R, "compute_guard", side_effect=ValueError("local forbidden")), \
             mock.patch.object(J.R, "source_binding") as binding, \
             self.assertRaisesRegex(ValueError, "local forbidden"):
            J.run("a"*40, "b"*40, "c"*64, "d"*64)
        binding.assert_not_called()

    def test_prepare_exclusive_real_hashes_receipt_and_no_scheduler(self):
        cfg, identity = P.configuration(J.R)
        binding = dict(effective_profile=identity, synthetic=True)
        for occupied in (None, "root", "runtime", "run", "scratch", "dangling_scratch"):
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                root = base/"root"
                paths = dict(root=str(root), runtime=str(root/"runtime-001"),
                    run=str(root/"full-001"), scratch=str(base/"scratch"))
                if occupied == "dangling_scratch":
                    Path(paths["scratch"]).symlink_to(base/"missing")
                elif occupied:
                    Path(paths[occupied]).mkdir(parents=True)
                with mock.patch.dict(J.R.PLAN, paths), \
                     mock.patch.object(J.R, "effective_profile", return_value=identity), \
                     mock.patch.object(J.R, "source_binding", return_value=binding), \
                     mock.patch.object(J.R, "sha", side_effect=lambda p: cfg["fixture_sha256"]
                         if str(p) == J.R.PLAN["fixture"] else F.J.R.sha(p)), \
                     mock.patch.object(J.R, "prior_probe", return_value={}), \
                     mock.patch.object(J.R, "compose", return_value={"synthetic_runtime": True}) as compose, \
                     mock.patch.object(J.R, "verify_runtime") as verify, \
                     mock.patch.object(J.R, "derived_document", return_value={"synthetic_basis": True}), \
                     mock.patch.object(subprocess, "run", side_effect=AssertionError("no scheduler")), \
                     mock.patch.object(subprocess, "check_output", side_effect=AssertionError("no scheduler")), \
                     redirect_stdout(io.StringIO()):
                    if occupied:
                        with self.assertRaisesRegex(ValueError, "destinations must all be new"):
                            J.prepare("a"*40, "b"*40)
                        compose.assert_not_called()
                        self.assertFalse((Path(paths["run"])/"metadata/preparation.json").exists())
                    else:
                        J.prepare("a"*40, "b"*40)
                        compose.assert_called_once()
                        verify.assert_called_once()
                        receipt = json.loads((Path(paths["run"])/"metadata/preparation.json").read_text())
                        self.assertEqual(receipt["runtime_manifest_sha256"],
                            F.J.R.sha(Path(paths["runtime"])/"runtime-manifest.json"))
                        self.assertEqual(receipt["derivative_sha256"],
                            F.J.R.sha(Path(paths["runtime"])/"basis-derivative-admission.json"))
                        self.assertEqual(receipt["test_only_argv"][-2:],
                            [receipt["runtime_manifest_sha256"], receipt["derivative_sha256"]])
                        self.assertEqual(receipt["retry_grant"], P.GRANT)
                        self.assertTrue(receipt["no_native_import_build_or_submission"])

    def test_prepare_fixture_and_probe_refuse_before_destinations(self):
        identity = P.configuration(J.R)[1]
        for bad_fixture in (True, False):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)/"root"
                with mock.patch.dict(J.R.PLAN, root=str(root)), \
                     mock.patch.object(J.R, "effective_profile", return_value=identity), \
                     mock.patch.object(J.R, "source_binding", return_value=dict(effective_profile=identity)), \
                     mock.patch.object(J.R, "sha", return_value="bad" if bad_fixture else
                         identity["effective_config"]["fixture_sha256"]), \
                     mock.patch.object(J.R, "prior_probe", side_effect=ValueError("original probe differs")), \
                     mock.patch.object(J.R, "compose") as compose, self.assertRaises(ValueError):
                    J.prepare("a"*40, "b"*40)
                self.assertFalse(root.exists())
                compose.assert_not_called()

    def test_batch_operational_only_diff_and_term_safe_receipts(self):
        old = (HERE/"pdef_full.sbatch").read_text()
        script = (HERE/"pdef_retry_full.sbatch").read_text()
        expected = old.replace("ccasp-bz-full-pdef", "ccasp-bz-pdef-retry").replace(
            "ccasp-benzene-full-pdef-20261025", "ccasp-benzene-pdef-retry-20261026").replace(
            "pdef_job.py", "pdef_retry_job.py").replace(
            "# FUTURE ONLY. NO NEW ACTIVATION AUTHORITY; prior job14397036 grant consumed.",
            "# ONE fresh free retry, parent activation only. No automatic further retry.")
        self.assertEqual(script, expected)
        traps = script.split("term_received=0\n", 1)[1].split('test "$#" = 4', 1)[0]
        for action, code, complete in (("child_ok=1; exit 0", 0, True),
                ("exit 0", 0, False), ("exit 2", 2, False),
                ("child_ok=1; kill -TERM $$; exit 0", 143, False)):
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp)/"metadata").mkdir()
                result = subprocess.run(["bash"], text=True, capture_output=True,
                    input="set -euo pipefail\nRUN="+tmp+"\nterm_received=0\n"+traps+action)
                self.assertEqual(result.returncode, code)
                receipt = json.loads((Path(tmp)/"metadata/batch-terminal.json").read_text())
                self.assertEqual(receipt["complete"], complete)

    def test_job_selection_primary_failure_closure_failure_and_success(self):
        for outcome in ("science_failure", "closure_failure", "success", "initialize_failure"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root/"metadata").mkdir()
                binding = dict(effective_profile={"synthetic": True})
                J.R.write(root/"metadata/preparation.json", dict(binding=binding,
                    runtime_manifest_sha256="c"*64, derivative_sha256="d"*64,
                    script_sha256=J.R.sha(HERE/"pdef_retry_full.sbatch"),
                    retry_grant=P.GRANT, activation_authority=P.AUTHORITY))
                handles_seen = []
                def initialize(*args):
                    args[-1].append(object())
                    handles_seen.append(args[-1])
                    if outcome == "initialize_failure":
                        raise RuntimeError("synthetic initialization refusal after handle")
                    return object(), object()
                def science(*args):
                    marker = root/"results/diagnostic/node-01/pre-lw-complete.json"
                    marker.parent.mkdir(parents=True)
                    J.R.write(marker, {"properties_accepted": False})
                    if outcome == "science_failure":
                        raise RuntimeError("synthetic scientific refusal")
                    return dict(status="complete", science_pass=True)
                def snapshot(*args, **kwargs):
                    self.assertEqual(len(handles_seen[0]), 1)
                    if outcome in ("science_failure", "closure_failure", "initialize_failure"):
                        raise ValueError("synthetic closure refusal")
                selected = []
                def module(name, file):
                    selected.append(file)
                    return NS(capture_science=science)
                with mock.patch.dict(J.R.PLAN, run=str(root)), \
                     mock.patch.dict(os.environ, SLURM_JOB_ID="SYNTHETIC"), \
                     mock.patch.object(J.R, "compute_guard"), \
                     mock.patch.object(J.R, "effective_profile"), \
                     mock.patch.object(J.R, "source_binding", return_value=binding), \
                     mock.patch.object(J.R, "pinned_json", return_value={}), \
                     mock.patch.object(J.R, "derived_document", return_value={}), \
                     mock.patch.object(J.R, "initialize", side_effect=initialize), \
                     mock.patch.object(J.R, "module", side_effect=module), \
                     mock.patch.object(J.R.OLD, "snapshot", side_effect=snapshot), \
                     mock.patch.object(J.R, "verify_runtime") as verify:
                    self.assertEqual(J.run("a"*40, "b"*40, "c"*64, "d"*64),
                                     0 if outcome == "success" else 1)
                record = json.loads((root/"results/result.json").read_text())
                terminal = json.loads((root/"metadata/payload-terminal.json").read_text())
                self.assertEqual(record["science_pass"], outcome == "success")
                self.assertEqual(terminal["complete"], outcome == "success")
                if outcome != "initialize_failure":
                    self.assertEqual(selected, ["pdef_retry_driver.py"])
                    self.assertEqual(len(record["complete_pre_lw_capture_receipts"]), 1)
                if outcome in ("science_failure", "initialize_failure"):
                    self.assertIn("refusal", record["error"])
                    self.assertIn("refusal", record["traceback"])
                if outcome != "success":
                    self.assertIn("closure refusal", record["closure_error"])
                else:
                    verify.assert_called_once()
                    self.assertTrue(verify.call_args.kwargs["hash_core"])


if __name__ == "__main__":
    unittest.main()
