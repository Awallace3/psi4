# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Engine-free explicit paid adapter contracts, not native/science acceptance."""
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

import test_pdef_retry as T
from pdef_boundary import drive

F = T.F
HERE = Path(__file__).resolve().parent
load = F.A.load
J = load("inferno_test_job", "pdef_inferno_job.py")
P = load("inferno_test_profile", "pdef_inferno_profile.py")
D = load("inferno_test_driver", "pdef_inferno_driver.py")


class ProfileTests(unittest.TestCase):
    def test_exact_frozen_science_distinct_real_plan_identity_no_plan_mutation(self):
        runtimes = (J.bootstrap, J.R, T.J.R, F.J.R, F.A.R)
        plans = [copy.deepcopy(r.PLAN) for r in runtimes]
        cfg, identity = P.configuration(J.R)
        old_cfg, old_id = F.P.configuration(F.J.R)
        retry_cfg, retry_id = T.P.configuration(T.J.R)
        self.assertEqual(cfg, old_cfg)
        self.assertEqual(cfg, retry_cfg)
        self.assertEqual(F.P.canonical_sha(cfg), P.CONFIG_SHA)
        self.assertEqual(old_id["identity_sha256"], P.PDEF_IDENTITY)
        self.assertEqual(retry_id["identity_sha256"], P.RETRY_IDENTITY)
        self.assertEqual(identity["parent_pdef_profile"], old_id)
        self.assertNotIn(identity["identity_sha256"], (P.PDEF_IDENTITY, P.RETRY_IDENTITY))
        self.assertEqual(identity["plan_sha256"], J.R.sha(HERE/"pdef_inferno_plan.json"))
        self.assertNotEqual(identity["plan_sha256"], old_id["plan_sha256"])
        self.assertNotEqual(identity["plan_sha256"], retry_id["plan_sha256"])
        self.assertEqual(identity["identity_sha256"], F.P.canonical_sha(
            {k: v for k, v in identity.items() if k != "identity_sha256"}))
        self.assertEqual(identity, J.R.effective_profile())
        for name, digest in identity["files_sha256"].items():
            self.assertEqual(digest, J.R.sha(HERE/name))
        self.assertEqual(identity["allocation"]["qos"], "inferno")
        for key in ("native", "fixture", "prior_probe", "prior_probe_sha256", "prior_driver_source",
                    "prior_driver_commit", "prior_driver_tree", "prior_driver_files", "fingerprints",
                    "routing", "overrides", "parent_profile_sha256"):
            self.assertEqual(J.R.PLAN[key], F.J.R.PLAN[key])
        expected = dict(F.J.R.PLAN["allocation"], qos="inferno")
        self.assertEqual(J.R.PLAN["allocation"], expected)
        self.assertEqual(plans, [r.PLAN for r in runtimes])
        self.assertIsNone(J.bootstrap.effective_profile())
        self.assertEqual(F.J.R.effective_profile(), old_id)
        self.assertEqual(T.J.R.effective_profile(), retry_id)
        # Shared runtime implementation, no replacement of any admission function.
        for name in ("compose", "initialize", "source_binding", "prior_probe", "admit_basis",
                     "verify_runtime", "compute_guard"):
            self.assertEqual(getattr(J.R, name).__code__.co_code, getattr(F.J.R, name).__code__.co_code)

    def test_all_operational_and_scientific_mutations_reject(self):
        mutations = [lambda p, k=k: p.update({k: "/arbitrary"})
            for k in ("source", "root", "runtime", "run", "scratch", "fixture", "prior_probe")]
        mutations += [lambda p, k=k, v=v: p["allocation"].update({k: v}) for k, v in (
            ("account", "gts-cs207"), ("partition", "cpu-medium"), ("qos", "embers"),
            ("cpus", 16), ("memory", "64G"), ("time", "08:00:00"), ("nodes", True),
            ("tasks", 2), ("node", "any"), ("numerical_threads", 8),
            ("psi4_memory_bytes", 24*1024**3))]
        mutations += [
            lambda p: p.update(extra_override=True),
            lambda p: p.update(schema="ccasp-benzene-pdef-retry-plan/1"),
            lambda p: p.update(effective_profile_module="pdef_profile.py"),
            lambda p: p.update(activation_authority="old grant"),
            lambda p: p.update(parent_pdef_identity_sha256="0"*64),
            lambda p: p.update(historical_retry_identity_sha256="0"*64),
            lambda p: p.update(effective_config_sha256="0"*64),
            lambda p: p["paid_grant"].update(fresh_allocations=2),
            lambda p: p["paid_grant"].update(maximum_usd="0.40"),
            lambda p: p["paid_grant"].update(automatic_further_retry=True),
            lambda p: p["paid_grant"].update(parent_price_route_approval_required=False),
            lambda p: p["paid_grant"].update(identity="another paid grant"),
            lambda p: p["native"].update(core_sha256="0"*64),
            lambda p: p["fingerprints"].update(main="0"*64),
            lambda p: p.update(prior_probe_sha256="0"*64),
            lambda p: p["overrides"].update({"isa.q_grid": [200, 590]}),
            lambda p: p["overrides"]["refinement.declared_variables"].pop()]
        for n, mutate in enumerate(mutations):
            plan = copy.deepcopy(J.R.PLAN)
            mutate(plan)
            with self.subTest(mutation=n), mock.patch.object(J.R, "PLAN", plan):
                with self.assertRaises(ValueError):
                    P.configuration(J.R)
                with self.assertRaises(ValueError):
                    J.argv("a"*40, "b"*40, "c"*64, "d"*64)

    def test_disk_plan_bytes_path_and_adapter_bytes_are_bound(self):
        read_text = Path.read_text
        for name, mutate in (
            ("pdef_inferno_plan.json", lambda p: p.update(unknown=1)),
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
        real_sha = J.R.sha
        identity = P.configuration(J.R)[1]
        # Formatting-only plan edits preserve semantics but change execution identity.
        for name in ("pdef_inferno_plan.json", "pdef_inferno_driver.py"):
            with mock.patch.object(J.R, "sha", side_effect=lambda path:
                    "e"*64 if Path(path) == HERE/name else real_sha(path)):
                changed = P.configuration(J.R)[1]
            self.assertNotEqual(identity["identity_sha256"], changed["identity_sha256"])
            self.assertEqual(changed["files_sha256"][name], "e"*64)
        with mock.patch.object(J.R, "sha", side_effect=lambda path:
                "e"*64 if Path(path) == HERE/"pdef_plan.json" else real_sha(path)), self.assertRaises(ValueError):
            P.configuration(J.R)

    def test_stdlib_only_isolated_imports(self):
        script = ("import importlib.util,sys;"
            f"s=importlib.util.spec_from_file_location('j',{str(HERE/'pdef_inferno_job.py')!r});"
            "j=importlib.util.module_from_spec(s);sys.modules[s.name]=j;s.loader.exec_module(j);"
            "j.R.effective_profile();"
            f"s=importlib.util.spec_from_file_location('d',{str(HERE/'pdef_inferno_driver.py')!r});"
            "d=importlib.util.module_from_spec(s);s.loader.exec_module(d);"
            "assert not {'numpy','scipy','psi4','adcc','einsums'} & set(sys.modules)")
        subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script], check=True)

    def test_source_derivative_original_probe_and_new_tools_bindings(self):
        real_sha, local_root = J.R.sha, J.R.ROOT
        def mapped_sha(path):
            path = Path(path)
            return real_sha(local_root/path.relative_to(P.SOURCE)) if path.is_relative_to(Path(P.SOURCE)) else real_sha(path)
        with mock.patch.object(J.R, "ROOT", Path(P.SOURCE)), \
             mock.patch.object(J.R.OLD, "identity"), mock.patch.object(J.R.OLD, "git", return_value=""), \
             mock.patch.object(J.R, "sha", side_effect=mapped_sha):
            binding = J.R.source_binding("a"*40, "b"*40)
            original = {"binding": {"original_profile": F.P.PARENT_SHA, "private_probe": "unchanged"}}
            with mock.patch.object(J.R, "prior_probe", return_value=original):
                doc = J.R.derived_document(binding, "c"*64)
        self.assertEqual(doc["original_probe"]["binding"], original["binding"])
        self.assertEqual(doc["original_probe"]["sha256"], F.J.R.PLAN["prior_probe_sha256"])
        self.assertEqual(doc["effective_profile"], binding["effective_profile"])
        for name in ("pdef_inferno_job.py", "pdef_inferno_driver.py", "pdef_inferno_plan.json"):
            self.assertEqual(binding["tools_sha256"][name], real_sha(HERE/name))
        self.assertEqual(binding["native"], {
            k: F.J.R.PLAN["native"][k] for k in ("source", "commit", "tree", "stage", "core_sha256")})


class DriverTests(unittest.TestCase):
    def test_actual_wrapper_kwargs_exact_scientific_config_equal_to_both_parents(self):
        captured = []
        for job, profile, driver in ((F.J, F.P, F.D), (T.J, T.P, T.D), (J, P, D)):
            module = job.R.module
            shared = NS(capture_science=mock.Mock(return_value={"science_pass": False}))
            def route(name, file):
                return shared if file == "integrated_driver.py" else module(name, file)
            _, identity = profile.configuration(job.R)
            fake = {"psi4.driver.procrouting": NS(isapol_lw=NS(PRODUCTION_TOLERANCE=1e-6))}
            with mock.patch.dict(sys.modules, fake), mock.patch.object(job.R, "compute_guard"), \
                 mock.patch.object(job.R, "module", side_effect=route), mock.patch.object(job.R, "write"):
                driver.capture_science(job.R, object(), {"effective_profile": identity}, "r", "d")
            kw = shared.capture_science.call_args.kwargs
            self.assertEqual(set(kw), {"configuration", "profile_identity", "validate_result"})
            self.assertEqual(kw["profile_identity"], identity)
            self.assertEqual(F.P.canonical_sha(kw["configuration"]), P.CONFIG_SHA)
            captured.append(kw)
        self.assertEqual(captured[0]["configuration"], captured[2]["configuration"])
        self.assertEqual(captured[1]["configuration"], captured[2]["configuration"])
        self.assertIs(captured[2]["validate_result"], D.parent_driver.validate_complete)
        self.assertEqual(captured[0]["validate_result"].__code__.co_code,
                         captured[2]["validate_result"].__code__.co_code)

    def test_real_shared_pipeline_all72_static_penalties_and_resources_unchanged(self):
        wrappers = {"pdef_job.py": F.J, "pdef_profile.py": F.P, "pdef_driver.py": F.D,
            "pdef_retry_job.py": T.J, "pdef_retry_profile.py": T.P, "pdef_retry_driver.py": T.D,
            "pdef_inferno_job.py": J, "pdef_inferno_profile.py": P, "pdef_inferno_driver.py": D}
        # Reuse real source-loaded wrappers; do not register a duplicate same-plan namespace.
        with mock.patch("pdef_boundary.load", side_effect=lambda name, file: wrappers[file]):
            results = [drive(which) for which in ("pdef", "pdef_retry", "pdef_inferno")]
        import numpy as np
        frozen, retry, inferno = results
        for other in (retry, inferno):
            for key in ("memory", "grid", "exported", "models"):
                self.assertEqual(frozen[key], other[key])
            self.assertEqual(vars(frozen["kwargs"]["resources"]), vars(other["kwargs"]["resources"]))
            for key in frozen["kwargs"]:
                if key not in ("resources", "diagnostic_sink", "quadrature", "partition_grid",
                               "partition_recipe", "scratch_directory"):
                    self.assertEqual(frozen["kwargs"][key], other["kwargs"][key])
            np.testing.assert_array_equal(frozen["kwargs"]["partition_grid"], other["kwargs"]["partition_grid"])
            self.assertTrue(other["kwargs"]["scratch_directory"].endswith("/scratch/payload-SYNTHETIC"))
        self.assertEqual(len(inferno["models"]), 11)
        self.assertEqual(inferno["models"][0].unpenalized_variables, ("H1_11s_21c_A",))
        self.assertEqual(len(inferno["models"][0].parameter_labels), 72)

    def test_wrong_identity_or_lw_tolerance_refused_before_science(self):
        identity = P.configuration(J.R)[1]
        for binding, tolerance in ((T.P.configuration(T.J.R)[1], 1e-6), (identity, 1e-5)):
            fake = {"psi4.driver.procrouting": NS(isapol_lw=NS(PRODUCTION_TOLERANCE=tolerance))}
            module, shared = J.R.module, mock.Mock()
            def route(name, file):
                return shared if file == "integrated_driver.py" else module(name, file)
            with mock.patch.dict(sys.modules, fake), mock.patch.object(J.R, "compute_guard"), \
                 mock.patch.object(J.R, "module", side_effect=route), mock.patch.object(J.R, "write"), \
                 self.assertRaises(ValueError):
                D.capture_science(J.R, object(), {"effective_profile": binding}, "r", "d")
            shared.capture_science.assert_not_called()


class OperationalTests(unittest.TestCase):
    def test_argv_new_batch_pins_and_compute_guard(self):
        args = J.argv("a"*40, "b"*40, "c"*64, "d"*64)
        self.assertEqual(args, ["/opt/slurm/current/bin/sbatch", "--test-only", "--export=NONE",
            "--chdir="+J.R.PLAN["run"], P.SOURCE+"/devtools/ccasp_phoenix_benzene/pdef_inferno_full.sbatch",
            "a"*40, "b"*40, "c"*64, "d"*64])
        self.assertEqual(J.argv(*args[-4:], test_only=False)[1], "--parsable")
        for value in (None, "PLACEHOLDER", "A"*64, "a"*63):
            with self.assertRaises(ValueError):
                J.argv("a"*40, "b"*40, value, "d"*64)
        with mock.patch.object(J.R, "compute_guard", side_effect=ValueError("local forbidden")), \
             mock.patch.object(J.R, "source_binding") as binding, self.assertRaisesRegex(ValueError, "local forbidden"):
            J.run("a"*40, "b"*40, "c"*64, "d"*64)
        binding.assert_not_called()

    def test_prepare_exclusive_real_metadata_hashes_no_scheduler_or_approval(self):
        cfg, identity = P.configuration(J.R)
        binding = dict(effective_profile=identity, synthetic=True)
        for occupied in (None, "root", "runtime", "run", "scratch", "dangling_scratch"):
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                paths = dict(root=str(base/"root"), runtime=str(base/"root/runtime-001"),
                    run=str(base/"root/full-001"), scratch=str(base/"scratch"))
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
                    else:
                        J.prepare("a"*40, "b"*40)
                        compose.assert_called_once_with(J.R.ROOT, J.R.OLD.STAGE,
                            Path(paths["runtime"])/"stage", binding)
                        verify.assert_called_once()
                        receipt = json.loads((Path(paths["run"])/"metadata/preparation.json").read_text())
                        for field, name in (("runtime_manifest_sha256", "runtime-manifest.json"),
                                            ("derivative_sha256", "basis-derivative-admission.json")):
                            self.assertEqual(receipt[field], F.J.R.sha(Path(paths["runtime"])/name))
                        self.assertEqual(receipt["test_only_argv"][-2:],
                            [receipt["runtime_manifest_sha256"], receipt["derivative_sha256"]])
                        self.assertEqual(receipt["future_submit_argv_not_executed"][1], "--parsable")
                        self.assertEqual(receipt["paid_grant"], P.GRANT)
                        self.assertTrue(receipt["no_native_import_build_or_submission"])
                        self.assertFalse(Path(receipt["required_external_price_approval"]).exists())

    def test_prepare_fixture_probe_and_partial_compose_refusal_preserves_paths(self):
        identity = P.configuration(J.R)[1]
        for failure in ("fixture", "probe", "compose"):
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                paths = dict(root=str(base/"root"), runtime=str(base/"root/runtime-001"),
                    run=str(base/"root/full-001"), scratch=str(base/"scratch"))
                with mock.patch.dict(J.R.PLAN, paths), \
                     mock.patch.object(J.R, "effective_profile", return_value=identity), \
                     mock.patch.object(J.R, "source_binding", return_value=dict(effective_profile=identity)), \
                     mock.patch.object(J.R, "sha", return_value="bad" if failure == "fixture" else
                         identity["effective_config"]["fixture_sha256"]), \
                     mock.patch.object(J.R, "prior_probe", side_effect=ValueError("probe") if failure == "probe" else None), \
                     mock.patch.object(J.R, "compose", side_effect=ValueError("compose")) as compose, \
                     self.assertRaises(ValueError):
                    J.prepare("a"*40, "b"*40)
                self.assertEqual(Path(paths["root"]).exists(), failure == "compose")
                if failure != "compose":
                    compose.assert_not_called()
                else:
                    self.assertTrue(Path(paths["scratch"]).exists())
                    self.assertFalse((Path(paths["run"])/"metadata/preparation.json").exists())

    def test_batch_only_approved_operational_diff_and_honest_term_wait(self):
        old = (HERE/"pdef_retry_full.sbatch").read_text()
        script = (HERE/"pdef_inferno_full.sbatch").read_text()
        expected = old.replace("ccasp-bz-pdef-retry", "ccasp-bz-pdef-inferno").replace(
            "ccasp-benzene-pdef-retry-20261026", "ccasp-benzene-pdef-inferno-20261026").replace(
            "pdef_retry_job.py", "pdef_inferno_job.py").replace("--qos=embers", "--qos=inferno").replace(
            "# ONE fresh free retry, parent activation only. No automatic further retry.",
            "# ONE paid allocation up to USD0.39, parent price/route approval required. No automatic further retry.")
        self.assertEqual(script, expected)
        self.assertIn('wait "$child"\n    rc=143', script)
        self.assertIn("--no-requeue", script)
        self.assertIn("OMP_NUM_THREADS=1", script)
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

    def test_job_selection_honest_failure_closure_success_and_paid_admission(self):
        for outcome in ("science_failure", "closure_failure", "success", "initialize_failure", "price_failure"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root/"metadata").mkdir()
                binding = dict(effective_profile={"synthetic": True})
                J.R.write(root/"metadata/preparation.json", dict(binding=binding,
                    runtime_manifest_sha256="c"*64, derivative_sha256="d"*64,
                    script_sha256=J.R.sha(HERE/"pdef_inferno_full.sbatch"),
                    paid_grant=P.GRANT, activation_authority=P.AUTHORITY,
                    required_external_price_approval=str(root/"metadata/paid-price-approval.json"),
                    parent_must_approve_price_route_and_consume_one_grant_before_submission=True))
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
                def snapshot(*args, **kw):
                    self.assertEqual(len(handles_seen[0]), 1)
                    if outcome in ("science_failure", "closure_failure", "initialize_failure"):
                        raise ValueError("synthetic closure refusal")
                selected = []
                def module(name, file):
                    selected.append(file)
                    return NS(capture_science=science)
                with mock.patch.dict(J.R.PLAN, run=str(root)), \
                     mock.patch.dict(os.environ, SLURM_JOB_ID="SYNTHETIC"), \
                     mock.patch.object(J.R, "compute_guard"), mock.patch.object(J.R, "effective_profile"), \
                     mock.patch.object(J.R, "source_binding", return_value=binding), \
                     mock.patch.object(J, "price_approval", return_value=dict(path="synthetic", sha256="e"*64, approval={}),
                         side_effect=ValueError("external price refusal") if outcome == "price_failure" else None), \
                     mock.patch.object(J.R, "pinned_json", return_value={}), \
                     mock.patch.object(J.R, "derived_document", return_value={}), \
                     mock.patch.object(J.R, "initialize", side_effect=initialize) as init, \
                     mock.patch.object(J.R, "module", side_effect=module), \
                     mock.patch.object(J.R.OLD, "snapshot", side_effect=snapshot), \
                     mock.patch.object(J.R, "verify_runtime") as verify:
                    self.assertEqual(J.run("a"*40, "b"*40, "c"*64, "d"*64), 0 if outcome == "success" else 1)
                record = json.loads((root/"results/result.json").read_text())
                terminal = json.loads((root/"metadata/payload-terminal.json").read_text())
                self.assertEqual(record["science_pass"], outcome == "success")
                self.assertEqual(terminal["complete"], outcome == "success")
                if outcome == "price_failure":
                    init.assert_not_called()
                    self.assertEqual(selected, [])
                    self.assertIn("external price refusal", record["error"])
                elif outcome != "initialize_failure":
                    self.assertEqual(selected, ["pdef_inferno_driver.py"])
                    self.assertEqual(len(record["complete_pre_lw_capture_receipts"]), 1)
                if outcome in ("science_failure", "initialize_failure"):
                    self.assertIn("refusal", record["error"])
                    self.assertIn("refusal", record["traceback"])
                if outcome in ("science_failure", "closure_failure", "initialize_failure"):
                    self.assertIn("closure refusal", record["closure_error"])
                elif outcome == "success":
                    verify.assert_called_once()
                    self.assertTrue(verify.call_args.kwargs["hash_core"])


class PriceApprovalTests(unittest.TestCase):
    def approval(self, binding):
        return dict(schema="ccasp-benzene-pdef-inferno-price-approval/1",
            grant_identity=P.GRANT["identity"], approved_by="parent",
            approved_for_single_submission=True, automatic_further_retry=False,
            commit=binding["commit"], tree=binding["tree"],
            execution_identity_sha256=binding["effective_profile"]["identity_sha256"],
            runtime_manifest_sha256="c"*64, derivative_sha256="d"*64,
            allocation=copy.deepcopy(J.R.PLAN["allocation"]), quoted_maximum_usd="0.39",
            checked_at_utc="2026-10-26T00:00:00Z", price_route_evidence="SYNTHETIC TEST ONLY; not live evidence")

    def test_external_record_exact_pins_hash_no_query_or_generation(self):
        binding = dict(commit="a"*40, tree="b"*40, effective_profile=P.configuration(J.R)[1])
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(J.R.PLAN, run=tmp):
            path = Path(tmp)/"metadata/paid-price-approval.json"
            path.parent.mkdir()
            with self.assertRaises(FileNotFoundError):
                J.price_approval(binding, "c"*64, "d"*64)
            approval = self.approval(binding)
            pin = J.R.write(path, approval)
            with mock.patch.object(subprocess, "run", side_effect=AssertionError("no query")), \
                 mock.patch.object(subprocess, "check_output", side_effect=AssertionError("no query")):
                record = J.price_approval(binding, "c"*64, "d"*64)
            self.assertEqual(record, dict(path=str(path), sha256=pin, approval=approval))

    def test_missing_unknown_wrong_route_pins_grant_and_excess_price_refuse(self):
        binding = dict(commit="a"*40, tree="b"*40, effective_profile=P.configuration(J.R)[1])
        mutations = [
            lambda a: a.update(unknown=1),
            lambda a: a.update(quoted_maximum_usd="0.390001"),
            lambda a: a.update(quoted_maximum_usd="NaN"),
            lambda a: a.update(quoted_maximum_usd=0.39),
            lambda a: a.update(checked_at_utc=""),
            lambda a: a.update(price_route_evidence=""),
            lambda a: a.update(approved_for_single_submission=1),
            lambda a: a.update(approved_by="worker"),
            lambda a: a.update(automatic_further_retry=True),
            lambda a: a.update(grant_identity="old grant"),
            lambda a: a["allocation"].update(qos="embers"),
            lambda a: a["allocation"].update(account="gts-cs207"),
            lambda a: a.update(commit="0"*40),
            lambda a: a.update(tree="0"*40),
            lambda a: a.update(runtime_manifest_sha256="0"*64),
            lambda a: a.update(derivative_sha256="0"*64),
            lambda a: a.update(execution_identity_sha256=P.PDEF_IDENTITY),
            lambda a: a.pop("allocation")]
        for n, mutate in enumerate(mutations):
            with self.subTest(mutation=n), tempfile.TemporaryDirectory() as tmp, mock.patch.dict(J.R.PLAN, run=tmp):
                path = Path(tmp)/"metadata/paid-price-approval.json"
                path.parent.mkdir()
                approval = self.approval(binding)
                mutate(approval)
                J.R.write(path, approval)
                with self.assertRaises(ValueError):
                    J.price_approval(binding, "c"*64, "d"*64)

    def test_price_record_symlink_fifo_and_size_reject_before_read(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(J.R.PLAN, run=tmp):
            path = Path(tmp)/"metadata/paid-price-approval.json"
            path.parent.mkdir()
            target = Path(tmp)/"target"
            target.write_text("{}")
            path.symlink_to(target)
            with self.assertRaises(OSError):
                J.price_approval({}, "c"*64, "d"*64)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(J.R.PLAN, run=tmp):
            path = Path(tmp)/"metadata/paid-price-approval.json"
            path.parent.mkdir()
            os.mkfifo(path)
            with self.assertRaises(ValueError):
                J.price_approval({}, "c"*64, "d"*64)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(J.R.PLAN, run=tmp):
            path = Path(tmp)/"metadata/paid-price-approval.json"
            path.parent.mkdir()
            path.write_bytes(b" "*(64*1024+1))
            with self.assertRaises(ValueError):
                J.price_approval({}, "c"*64, "d"*64)


if __name__ == "__main__":
    unittest.main()
