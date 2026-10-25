# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Engine-free only: actual source wrappers, independent fake native seam."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

import numpy as np
from test_integrated_runtime import load

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/"tests/pytests"))
from isapol_source_test_support import load_source

B = load_source("isapol_bounded")
F = load_source("isapol_refine")
S = load_source("isapol_pfit_stream")
D = load("smoke_test_driver", "pfit_smoke_driver.py")
J = load("smoke_test_job", "pfit_smoke_job.py")
P = load("smoke_test_profile", "pfit_smoke_profile.py")


class Failed:
    status = "RankDeficient"

    @property
    def parameters(self):
        raise RuntimeError("PFIT fake: parameters unavailable for failed fit")


class FakeCore:
    """NOT a native implementation or fallback. Test-only boundary model.

    Normal equations here differ from the driver's augmented-lstsq oracle.
    Dense input design is independently reconstructed from actual channel
    tensors, not supplied directly by the driver.
    """
    IsaPfitTargetOrigin = NS(SyntheticAnalyticTest="SyntheticAnalyticTest")
    IsaPfitTargetConvention = NS(NegativeInducedPotentialPerUnitSourceChargeAtomicUnits="negative-dphi/dq")
    IsaPfitStatus = NS(Solved="Solved", RankDeficient="RankDeficient")
    IsaPfitSolver = NS(NormalEquationsDSYSV="DSYSV", StreamingQR="QR")
    IsaPfitTargetProvenance = NS
    IsaPfitRowModel = NS
    IsaPfitCloudRows = NS
    IsaPfitRowProblem = NS
    IsaPfitMatrix = NS
    IsaPfitMatrixPenalty = NS
    IsaPfitProblem = NS
    IsaPfitModel = NS
    IsaPfitBatch = NS

    def __init__(self):
        self.calls = []
        self.stream_passes = []

    def IsaPfitOptions(self):
        # Native struct default is QR; actual Python wrapper must change this.
        return NS(solver="QR", maximum_work_bytes=256*1024**2)

    def result(self, problem, a, y, options):
        assert options.solver == "DSYSV"
        assert problem.model.fixed == [False]*72
        assert problem.model.parameter_labels == list(P.declaration(HERE))
        assert len(problem.model.channel_labels) == 108
        assert problem.target_provenance.origin == "SyntheticAnalyticTest"
        assert problem.target_provenance.auxiliary_basis_id == ""
        weights = np.array(problem.penalty.matrix.values).reshape(72, 72)
        anchors = np.array(problem.penalty.anchor)
        h = a.T@a+weights
        if np.linalg.matrix_rank(h) < 72:
            return Failed()
        z = np.linalg.solve(h, a.T@y+weights@anchors)
        residual = a@z-y
        delta = z-anchors
        diag = NS(free_indices=list(range(72)), numerical_rank=72, data_rows=len(a),
            objective_available=True, work_budget_bytes=2*1024**2,
            lapack_info=0, condition_estimate_available=True, normal_h_rcond=1/np.linalg.cond(h, 1),
            total_objective=float(residual@residual+delta@weights@delta), native_verified=False)
        return NS(parameters=z.tolist(), status="Solved", diagnostics=diag, settings=options,
                  matrix_penalty_rhs=(weights@anchors).tolist(),
                  effective_penalty_matrix=NS(values=weights.ravel().tolist()),
                  parameter_labels=problem.model.parameter_labels,
                  channel_labels=problem.model.channel_labels, frequency_au=problem.frequency_au)

    def isa_pfit_solve_rows_multi(self, problems, factory, options):
        self.calls.append(("stream", len(problems)))
        a, y = [], []
        hashes = []
        for pass_index in range(2):
            h = hashlib.sha256()
            count = 0
            for start, design, targets in factory():
                assert start == count
                assert design.dtype == targets.dtype == np.float64
                assert design.flags.c_contiguous and targets.flags.c_contiguous
                assert design.shape[1] == 72
                assert len(design) <= problems[0].cloud.maximum_block_rows
                count += len(design)
                h.update(design.tobytes()); h.update(targets.tobytes())
                if pass_index == 0:
                    a.append(design); y.append(targets)
            assert count == problems[0].cloud.full_row_count == 528
            hashes.append(h.hexdigest())
            self.stream_passes.append(count)
            if pass_index == 0:
                a, y = np.concatenate(a), np.concatenate(y)
                results = [self.result(p, a, y[:, t], options) for t, p in enumerate(problems)]
                if all(r.status != "Solved" for r in results):
                    return results
        assert hashes[0] == hashes[1]
        return results

    def isa_pfit_solve(self, problem, options):
        self.calls.append(("dense", 1))
        batch = problem.batches[0]
        fields = np.array(batch.fields.values).reshape(batch.fields.rows, batch.fields.cols)
        i, j = np.tril_indices(len(fields))
        a = np.zeros((len(i), 72))
        for k, tensor in enumerate(problem.model.parameter_tensors):
            m = np.array(tensor.values).reshape(tensor.rows, tensor.cols)
            assert np.array_equal(m, m.T)
            for r, c in np.argwhere(m):
                a[:, k] += m[r, c]*fields[i, r]*fields[j, c]
        return self.result(problem, a, np.array(batch.targets), options)


class NumericalTests(unittest.TestCase):
    def exercise(self, core=None, emit=None, p=None):
        core = core or FakeCore()
        p = p or J.R.PLAN["procedure"]
        with mock.patch.object(F, "core", core), mock.patch.object(S, "core", core):
            result = D.exercise(B, F, S, core, P.declaration(HERE), p, emit or (lambda *args: None))
        return result, core

    def test_actual_complete_design_objective_recovery_and_solver_wiring(self):
        snapshots = {}
        result, core = self.exercise(emit=lambda name, value: snapshots.update({name: value}))
        self.assertEqual(result["solved"], 11)
        self.assertEqual(core.calls, [("stream", 11)]+[("dense", 1)]*12+[("stream", 1)])
        self.assertEqual(core.stream_passes, [528, 528, 528])  # singular: assembly only
        self.assertEqual(len(snapshots), 24)
        self.assertLess(result["design_condition"], 100.)
        self.assertLess(result["max_reference_error"], 1e-10)
        self.assertEqual(result["resources"]["dense_source_work_bound"], 40794513408)
        self.assertEqual(result["resources"]["total_work_bound"], 41453630720)
        self.assertTrue(all(len(json.dumps(v).encode()) < 256*1024 for v in snapshots.values()))
        for k in range(11):
            model = snapshots[f"model-{k:02d}.json"]
            self.assertEqual(sum(model["static_mask"]), 71)
            self.assertEqual(model["static_reference_sha256"], result["static_reference_sha256"])
            self.assertEqual(model["anchors"][0], .02 if k == 0 else 3e-5)
            self.assertEqual(model["raw_upward_anchor"], 4e-5 if k < 3 else 2e-4)
            self.assertGreater(snapshots[f"solution-{k:02d}.json"]["free_extra"], .1)
        print("ENGINE_FREE_ONLY_SMOKE " + json.dumps(result, sort_keys=True))

    def test_rank_guard_precedes_any_solver_call(self):
        inputs = D.synthetic_inputs(F, J.R.PLAN["procedure"])
        bad = (*inputs[:3], np.zeros_like(inputs[3]))
        core = FakeCore()
        with mock.patch.object(D, "synthetic_inputs", return_value=bad), \
             self.assertRaisesRegex(ValueError, "full column rank"):
            self.exercise(core)
        self.assertEqual(core.calls, [])

    def test_work_and_bytes_fail_before_solver(self):
        for key in ("max_work", "max_bytes", "max_io_bytes"):
            p = dict(J.R.PLAN["procedure"], **{key: 1})
            core = FakeCore()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "admission"):
                self.exercise(core, p=p)
            self.assertEqual(core.calls, [])

    def test_oracle_current_penalties_not_static_or_row_normalized(self):
        p = J.R.PLAN["procedure"]
        sites, nodes, _, fields = D.synthetic_inputs(F, p)
        models = B.refinement_models(sites, nodes, cutoff=1e-4, weight_type=3,
            weight_coefficient=.001, declared_variables=P.declaration(HERE))
        m = models[4]
        a = D.independent_design(m, fields)
        truth = .1+np.sin(np.arange(72))
        y = a@truth
        expected = D.oracle(m, a, y)
        w, anchor = np.diag(m.strengths), np.array(m.anchors)
        np.testing.assert_allclose(expected, np.linalg.solve(a.T@a+w, a.T@y+w@anchor), atol=1e-12)
        wrong_scale = np.linalg.solve(a.T@a+len(a)*w, a.T@y+len(a)*w@anchor)
        self.assertGreater(np.max(np.abs(expected-wrong_scale)), 1e-4)
        static = models[0]
        wrong_static = np.linalg.solve(a.T@a+np.diag(static.strengths),
                                       a.T@y+np.asarray(static.strengths)*static.anchors)
        self.assertGreater(np.max(np.abs(expected-wrong_static)), 1e-8)


class IdentityTests(unittest.TestCase):
    def test_profile_separate_plan_original_declaration_no_full_allocation(self):
        p, ident = P.configuration(J.R)
        self.assertEqual(p, J.R.PLAN["procedure"])
        self.assertEqual(ident["declaration_provenance"]["declaration_sha256"], P.DECLARATION_SHA)
        old = json.loads((HERE/"pdef_plan.json").read_text())
        self.assertEqual(old["allocation"]["cpus"], 8)
        self.assertEqual(old["allocation"]["psi4_memory_bytes"], 26843545600)
        self.assertEqual(J.bootstrap.PLAN["schema"], "ccasp-integrated-capture-plan/1")
        self.assertNotIn("fixture", J.R.PLAN)
        self.assertNotIn("prior_probe", J.R.PLAN)
        doc = J.procedure_document({"effective_profile": ident}, "a"*64)
        self.assertEqual(doc["original_declaration"], ident["declaration_provenance"])
        self.assertIn("UNTESTED", doc["basis_admission"])
        for key in ("max_parameters", "max_design_condition", "parameter_atol", "max_bytes"):
            changed = copy.deepcopy(J.R.PLAN)
            changed["procedure"][key] += 1
            with mock.patch.object(J.R, "PLAN", changed), self.assertRaisesRegex(ValueError, "exact synthetic"):
                P.configuration(J.R)

    def test_isolated_bootstrap_and_driver_import_no_scientific_packages(self):
        command = ("import importlib.util,sys;"
            f"s=importlib.util.spec_from_file_location('j',{str(HERE/'pfit_smoke_job.py')!r});"
            "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);"
            "m.R.effective_profile();"
            f"s=importlib.util.spec_from_file_location('d',{str(HERE/'pfit_smoke_driver.py')!r});"
            "d=importlib.util.module_from_spec(s);s.loader.exec_module(d);"
            "assert not {'numpy','scipy','psi4','adcc','einsums'} & set(sys.modules)")
        subprocess.run([sys.executable, "-I", "-S", "-B", "-c", command], check=True)

    def test_testonly_no_submit_argv_and_before_guard(self):
        args = J.argv("a"*40, "b"*40, "c"*64, "d"*64)
        self.assertIn("--test-only", args)
        self.assertTrue(args[4].endswith("/pfit_smoke.sbatch"))
        with self.assertRaises(ValueError):
            J.argv("a"*40, "b"*40, None, None)
        with mock.patch.object(J.R, "compute_guard", side_effect=ValueError("local guard")), \
             mock.patch.object(J.R, "initialize") as init, mock.patch.object(J, "source_binding") as binding:
            with self.assertRaisesRegex(ValueError, "local guard"):
                J.run("a"*40, "b"*40, "c"*64, "d"*64)
            init.assert_not_called()
            binding.assert_not_called()
            with self.assertRaisesRegex(ValueError, "local guard"):
                D.run_native(J.R, mock.Mock())

    def test_stronger_production_and_additions_only_guards(self):
        def git(path, *args):
            if args[:2] == ("rev-parse", BASE+":psi4"):
                return "unchanged"
            if args[0] == "rev-parse":
                return "unchanged"
            return "M\tdevtools/ccasp_phoenix_benzene/pdef_profile.py"
        BASE = J.BASE
        with mock.patch.object(J.R, "source_binding", return_value={}), \
             mock.patch.object(J.R.OLD, "git", side_effect=git), self.assertRaisesRegex(ValueError, "additions"):
            J.source_binding("a"*40, "b"*40)
        with mock.patch.object(J.R, "source_binding", return_value={}), \
             mock.patch.object(J.R.OLD, "git", side_effect=["new", "old"]), \
             self.assertRaisesRegex(ValueError, "complete production"):
            J.source_binding("a"*40, "b"*40)

    def test_prepare_is_metadata_only_exclusive_and_no_geometry_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"fresh"
            paths = dict(runtime=str(root/"runtime-001"), run=str(root/"smoke-001"),
                         scratch=str(Path(tmp)/"scratch"))
            profile = J.R.effective_profile()
            binding = {"effective_profile": profile}
            with mock.patch.dict(J.R.PLAN, paths), mock.patch.object(J, "source_binding", return_value=binding), \
                 mock.patch.object(J.R, "compose", return_value={"toy": True}), \
                 mock.patch.object(J.R, "verify_runtime"), mock.patch.object(J.R, "prior_probe") as probe, \
                 mock.patch.object(J.R, "initialize") as init, \
                 mock.patch.object(J.R, "sha", return_value="e"*64):
                J.prepare("a"*40, "b"*40)
                receipt = json.loads((root/"smoke-001/metadata/preparation.json").read_text())
                self.assertIn("--test-only", receipt["test_only_argv"])
                self.assertNotIn("future_submit_argv_not_executed", receipt)
                self.assertFalse(receipt["basis_or_private_geometry_read"])
                with self.assertRaisesRegex(ValueError, "fresh"):
                    J.prepare("a"*40, "b"*40)
                probe.assert_not_called(); init.assert_not_called()

    def test_small_output_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "snapshot"):
                J.small_write(root/"large.json", "x"*262145)
            (root/"file").write_bytes(b"x")
            self.assertEqual(J.output_inventory(root)["files"], 1)
            with mock.patch.dict(J.R.PLAN["procedure"], max_output_files=1):
                with self.assertRaisesRegex(ValueError, "aggregate"):
                    J.output_inventory(root, reserve=1)

    def test_output_traversal_errors_are_not_silent(self):
        def failed_walk(root, *, followlinks=False, onerror=None):
            yield str(root), ["nested"], []
            if onerror is not None:
                onerror(PermissionError("nested output traversal denied"))
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(J.os, "walk", side_effect=failed_walk):
            with self.assertRaisesRegex(PermissionError, "nested output traversal denied"):
                J.output_inventory(Path(tmp))

    def test_partial_preparation_is_retained_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"partial"
            paths = dict(runtime=str(root/"runtime-001"), run=str(root/"smoke-001"),
                         scratch=str(Path(tmp)/"scratch"))
            binding = dict(effective_profile=J.R.effective_profile())
            with mock.patch.dict(J.R.PLAN, paths), mock.patch.object(J, "source_binding", return_value=binding), \
                 mock.patch.object(J.R, "compose", side_effect=ValueError("partial composition")):
                with self.assertRaisesRegex(ValueError, "partial composition"):
                    J.prepare("a"*40, "b"*40)
                self.assertTrue(root.exists())
                self.assertTrue(Path(paths["scratch"]).exists())
                with self.assertRaisesRegex(ValueError, "fresh"):
                    J.prepare("a"*40, "b"*40)


class TerminalTests(unittest.TestCase):
    def test_closure_failure_no_false_success_primary_preserved_handles_live(self):
        for fail_science, fail_closure, fail_output in (
                (False, False, False), (True, False, False), (False, True, False),
                (True, True, False), (False, False, True)):
            with self.subTest(science=fail_science, closure=fail_closure, output=fail_output), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root/"metadata").mkdir()
                profile = J.R.effective_profile()
                binding = dict(effective_profile=profile)
                preparation = dict(binding=binding, runtime_manifest_sha256="c"*64, procedure_sha256="d"*64,
                    script_sha256=J.R.sha(HERE/"pfit_smoke.sbatch"),
                    test_only_argv=J.argv("a"*40, "b"*40, "c"*64, "d"*64))
                token = object()
                retained = []
                def initialize(*args):
                    args[-1].append(token)
                    retained.append(args[-1])
                    return NS(set_memory=mock.Mock()), token
                def science(*args):
                    if fail_science:
                        raise RuntimeError("primary synthetic refusal")
                    return dict(status="SYNTHETIC_CHECKS_PASSED_PENDING_FINAL_CLOSURE", solved=11)
                def snapshot(*args, **kwargs):
                    self.assertIs(retained[0][0], token)
                    if fail_closure:
                        raise ValueError("loader closure refusal")
                def pinned(path, pin):
                    return preparation if Path(path).name == "preparation.json" else J.procedure_document(binding, "c"*64)
                original_walk = J.os.walk
                def audit_walk(path, *, followlinks=False, onerror=None):
                    if fail_output:
                        yield str(path), ["nested"], []
                        if onerror is not None:
                            onerror(PermissionError("nested output traversal denied"))
                        return
                    yield from original_walk(path, followlinks=followlinks, onerror=onerror)
                # test_only argv contains the original run path; substitute
                # after patching PLAN, just as actual preparation would.
                with mock.patch.dict(J.R.PLAN, run=str(root)), \
                     mock.patch.dict(os.environ, SLURM_JOB_ID="FAKE"), \
                     mock.patch.object(J, "compute_guard"), \
                     mock.patch.object(J, "source_binding", return_value=binding) as source, \
                     mock.patch.object(J.R, "pinned_json", side_effect=pinned), \
                     mock.patch.object(J.R, "initialize", side_effect=initialize), \
                     mock.patch.object(J.R, "module", return_value=NS(run_native=science)), \
                     mock.patch.object(J.R.OLD, "snapshot", side_effect=snapshot), \
                     mock.patch.object(J.R, "verify_runtime") as verify, \
                     mock.patch.object(J.os, "walk", side_effect=audit_walk), \
                     mock.patch.object(J.resource, "setrlimit"):
                    preparation["test_only_argv"] = J.argv("a"*40, "b"*40, "c"*64, "d"*64)
                    # Real pinned read is mocked, so no preparation file needed
                    # but its sha must have an actual regular input.
                    J.R.write(root/"metadata/preparation.json", preparation)
                    code = J.run("a"*40, "b"*40, "c"*64, "d"*64)
                self.assertEqual(code, int(fail_science or fail_closure or fail_output))
                result = json.loads((root/"results/result.json").read_text())
                self.assertEqual(result["smoke_pass"], code == 0)
                self.assertFalse(result["scientific_acceptance"])
                if fail_science:
                    self.assertIn("primary synthetic refusal", result["error"])
                if fail_closure:
                    self.assertIn("loader closure refusal", result["closure_errors"][0]["error"])
                if fail_output:
                    self.assertIn("nested output traversal denied", result["output_bound_error"])
                    self.assertFalse(json.loads((root/"metadata/payload-terminal.json").read_text())["complete"])
                verify.assert_called_once()  # loader failure does not skip runtime
                self.assertEqual(source.call_count, 2)

    def test_shell_resources_exact_slurm_before_sanitization_and_term_receipts(self):
        script = (HERE/"pfit_smoke.sbatch").read_text()
        for value in ("--account=gts-cs207-chemx", "--partition=cpu-small", "--qos=embers",
                      "--cpus-per-task=2", "--mem=8G", "--time=00:30:00",
                      "--nodelist=atl1-1-02-007-23-1", "--no-requeue", "-I -S -B"):
            self.assertIn(value, script)
        self.assertLess(script.index("SRUN=$(command -v srun)"), script.index("export PATH="))
        self.assertIn('test "$SRUN" = /opt/slurm/current/bin/srun', script)
        self.assertNotIn("pdef_job.py", script)
        subprocess.run(["bash", "-n", str(HERE/"pfit_smoke.sbatch")], check=True)
        traps = script.split("term_received=0\n", 1)[1].split('test "$#" = 4', 1)[0]
        for action, code in (("child_ok=1; exit 0", 0), ("exit 2", 2),
                             ("child_ok=1; kill -TERM $$; exit 0", 143)):
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp)/"metadata").mkdir()
                process = subprocess.run(["bash"], input="set -euo pipefail\nRUN="+tmp+
                    "\nterm_received=0\n"+traps+action, text=True, capture_output=True)
                self.assertEqual(process.returncode, code)
                result = json.loads((Path(tmp)/"metadata/batch-terminal.json").read_text())
                self.assertEqual(result["complete"], code == 0)
                self.assertFalse(result["molecular_acceptance"])


if __name__ == "__main__":
    unittest.main()
