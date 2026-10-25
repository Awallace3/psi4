# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Source-only contracts, not native Solved/all-node/property acceptance."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

import test_angular_full as A
from pdef_boundary import B, F, BR, anchors, drive, load_source
import pdef_resources as RES
import numpy as np

HERE = Path(__file__).resolve().parent
J = A.load("pdef_test_job", "pdef_job.py")
P = A.load("pdef_test_profile", "pdef_profile.py")
D = A.load("pdef_test_driver", "pdef_driver.py")
STREAM = load_source("isapol_pfit_stream")


class ProfileTests(unittest.TestCase):
    def test_exact_one_new_model_field_vs_frozen_angular_and_full_binding(self):
        cfg, identity = P.configuration(J.R)
        angular, _ = A.P.configuration(A.R)
        restored = copy.deepcopy(cfg)
        labels = restored["refinement"].pop("declared_variables")
        self.assertEqual(restored, angular)
        self.assertEqual(len(labels), 72)
        self.assertEqual(len(set(labels)), 72)
        self.assertEqual(P.COMPONENT_NAMES, F.COMPONENT_NAMES[:9])
        self.assertEqual(labels, list(P.declared_variables()))
        self.assertFalse(any("00" in label.split("_")[1:3] for label in labels))
        self.assertEqual(identity["effective_config_sha256"], P.canonical_sha(identity["effective_config"]))
        self.assertEqual(identity["declaration"]["sha256"], P.canonical_sha(labels))
        self.assertEqual(identity["identity_sha256"], P.canonical_sha(
            {k: v for k, v in identity.items() if k != "identity_sha256"}))
        self.assertIn("FREE", identity["semantics"])
        self.assertIsNone(J.bootstrap.effective_profile())
        self.assertEqual(A.R.PLAN["effective_profile_module"], "angular_profile.py")
        for key in ("native", "fixture", "prior_probe", "prior_probe_sha256", "prior_driver_source",
                    "prior_driver_commit", "prior_driver_tree", "prior_driver_files", "fingerprints", "allocation", "routing"):
            self.assertEqual(J.R.PLAN[key], A.R.PLAN[key])

    def test_tampered_declaration_parent_resources_authority_refuse(self):
        for mutate in (
            lambda p: p["overrides"]["refinement.declared_variables"].pop(),
            lambda p: p["overrides"]["refinement.declared_variables"].append(P.declared_variables()[0]),
            lambda p: p["overrides"].update({"isa.q_grid": [200, 590]}),
            lambda p: p["allocation"].update(numerical_threads=8),
            lambda p: p["allocation"].update(psi4_memory_bytes=24*1024**3),
            lambda p: p["native"].update(core_sha256="0"*64),
            lambda p: p.update(parent_profile_sha256="0"*64),
            lambda p: p.update(activation_authority="old grant")):
            plan = copy.deepcopy(J.R.PLAN)
            mutate(plan)
            with mock.patch.object(J.R, "PLAN", plan), self.assertRaises(ValueError):
                P.configuration(J.R)

    def test_stdlib_isolated_import_and_hash_no_real_engines(self):
        command = ("import importlib.util,sys;"
            f"s=importlib.util.spec_from_file_location('j',{str(HERE/'pdef_job.py')!r});"
            "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);"
            "m.R.effective_profile();"
            f"s=importlib.util.spec_from_file_location('d',{str(HERE/'pdef_driver.py')!r});"
            "d=importlib.util.module_from_spec(s);s.loader.exec_module(d);"
            "assert not {'numpy','scipy','psi4','adcc','einsums'} & set(sys.modules)")
        subprocess.run([sys.executable, "-I", "-S", "-B", "-c", command], check=True)

    def test_derivative_keeps_original_evidence_and_new_effective_identity(self):
        _, identity = P.configuration(J.R)
        original = {"binding": {"original_profile": P.PARENT_SHA}}
        with mock.patch.object(J.R, "prior_probe", return_value=original):
            doc = J.R.derived_document(dict(effective_profile=identity, commit="a"*40), "b"*64)
        self.assertEqual(doc["original_probe"]["binding"], original["binding"])
        self.assertEqual(doc["effective_profile"], identity)


class BoundaryTests(unittest.TestCase):
    def test_frozen_actual_driver_red_below_and_exact_cutoff(self):
        for value in (4e-5, 1e-4):
            seen = {}
            with self.assertRaisesRegex(ValueError, "declared_variables omits.*H1_11s_21c_A"):
                drive("angular", crossing=value, observed=seen)
            self.assertNotIn("declared_variables", seen["kwargs"])
            self.assertEqual(seen["kwargs"]["max_refinement_parameters"], 72)

    def test_new_actual_driver_green_inventory_mask_current_anchors_copy_and_guards(self):
        for value in (4e-5, 1e-4):
            out = drive("pdef", crossing=value)
            self.assertEqual((out["memory"], out["grid"], out["exported"]), (26843545600, (200, 974), 1))
            self.assertEqual(out["kwargs"]["resources"].max_bytes, 26843545600)
            self.assertEqual(out["kwargs"]["npoints"], 2000)
            models, sites, nodes = (out[k] for k in ("models", "sites", "nodes"))
            static_hash = F._anchor_hash(F._validated_anchor_tensors(sites, nodes[0][1]))
            baseline = B.refinement_models(sites, nodes[:1], cutoff=1e-4, weight_type=3, weight_coefficient=.001)[0]
            self.assertEqual(baseline.parameter_count, 71)
            for m, (omega, aa) in zip(models, nodes):
                self.assertEqual(m.parameter_count, 72)
                self.assertEqual(m.parameter_labels, P.declared_variables())
                self.assertEqual(m.declared_variables, P.declared_variables())
                self.assertEqual(m.unpenalized_variables, ("H1_11s_21c_A",))
                self.assertEqual(m.static_reference_sha256, static_hash)
                self.assertEqual(m.static_reference_sha256, baseline.static_reference_sha256)
                up = m.parameter_labels.index("H1_11s_21c_A")
                self.assertEqual((m.anchors[up], m.strengths[up]), (0., 0.))
                self.assertEqual(m.parameter_entries[up], tuple((s, 3, 5) for s in range(6, 12)))
                down = m.parameter_labels.index("C1_10_10_A")
                self.assertEqual(m.anchors[down], aa[0][1, 1])
                self.assertEqual(m.strengths[down], F.penalty_weight(weight_type=3,
                    weight_coefficient=.001, alpha=aa[0][1, 1], frequency=omega, rank1=1, rank2=1))
                current = F.refinement_model(sites, aa, frequency_au=omega, cutoff=1e-4)
                self.assertLessEqual(set(current.parameter_labels), set(m.parameter_labels))
                self.assertEqual(m.reference_sites, (0, 6))
                self.assertEqual(m.parameter_entries[down], tuple((s, 1, 1) for s in range(6)))
                self.assertGreater(m.copy_anchor_discrepancy, 1.)
            self.assertNotEqual(models[0].anchor_sha256, models[3].anchor_sha256)

    def test_actual_driver_bad_declarations_still_refuse(self):
        mutations = (
            lambda v: v.remove("H1_11s_21c_A"),
            lambda v: v.__setitem__(1, v[0]),
            lambda v: v.__setitem__(0, "malformed"),
            lambda v: v.__setitem__(0, "C1_30_30_A"),
            lambda v: v.__setitem__(0, "C2_10_10_A"),
            lambda v: v.__setitem__(0, "C1_11c_10_A"))
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                drive("pdef", config_mutator=lambda c: mutate(c["refinement"]["declared_variables"]))

    def test_free_extra_is_nonzero_design_column_and_native_declaration_not_fixed(self):
        models = drive("pdef")["models"]
        model = models[0]
        fields = np.arange(1., 1.+3*108).reshape(3, 108)/100.
        rows = STREAM.PackedDesignRows(model, fields, block_rows=6)
        extra = model.parameter_labels.index("H1_11s_21c_A")
        self.assertTrue(np.any(next(rows.blocks())[1][:, extra] != 0.))
        seen = []
        def no_native(problems, blocks, options):
            seen.extend(problems)
            raise RuntimeError("synthetic boundary; no native PFIT executed")
        replacements = dict(IsaPfitTargetProvenance=NS,
            IsaPfitTargetConvention=NS(NegativeInducedPotentialPerUnitSourceChargeAtomicUnits="synthetic"),
            IsaPfitRowModel=NS, IsaPfitCloudRows=NS, IsaPfitRowProblem=NS,
            IsaPfitMatrix=NS, IsaPfitMatrixPenalty=NS, IsaPfitOptions=NS,
            IsaPfitSolver=NS(NormalEquationsDSYSV="synthetic"),
            isa_pfit_solve_rows_multi=no_native)
        with mock.patch.dict(F.core.__dict__, replacements):
            with self.assertRaisesRegex(RuntimeError, "no native PFIT executed"):
                STREAM.refine_streamed(models, np.ones((3, 3)), np.zeros((6, 11)),
                    fields=fields, target_origin="NativeFittedPointResponse",
                    response_representation="fitted_density_coefficients", auxiliary_basis_id="synthetic",
                    source_id="synthetic", generation_record="not a molecular result")
        self.assertEqual(len(seen), 11)
        for problem in seen:
            self.assertEqual(problem.model.fixed, [False]*72)
            self.assertEqual(problem.penalty.anchor[extra], 0.)
            self.assertEqual(problem.penalty.matrix.values[extra*72+extra], 0.)

    def test_validator_checks_all_actual_entries_mask_and_labels_not_just_count(self):
        out = drive("pdef")
        cfg, _ = P.configuration(J.R)
        quad = NS(frequencies=out["result"].frequencies)
        for field, value in (
            ("parameter_labels", ("C1_00_00_A",)+P.declared_variables()[1:]),
            ("parameter_entries", (((0, 0, 0),),)+out["models"][1].parameter_entries[1:]),
            ("declared_variables", P.declared_variables()[:-1]),
            ("unpenalized_variables", ()),
            ("static_reference_sha256", "other"),
            ("anchors", out["models"][1].anchors[:-1])):
            bad = copy.deepcopy(out["result"])
            bad.refinements[1].model = replace(bad.refinements[1].model, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                D.validate_complete(bad, quad, cfg)

    def test_default64_refusal_explicit72_admission_73_refusal_before_runner(self):
        sites, _ = anchors()
        class Options:
            npoints = 2000
        args = dict(caller_converged=True, sites=sites, bonds=[], quadrature=NS(frequencies=tuple(range(11))),
            response_grid=np.zeros((1, 4)), smoothing=object(), shell_cutoff=1e-8,
            charge_penalty=1000., anchor_metric_damping=.0005, lattice_options=Options(),
            localization_rank_limit=2, weight_type=3, weight_coefficient=.001, cutoff=1e-4,
            resources=BR.BoundedResources(26843545600, 64000000000000, 68719476736, "large"),
            scf_correction="NONE", declared_variables=P.declared_variables())
        with mock.patch.dict(B.core.__dict__, FitPointsOptions=Options, IsaMultipoleSite=NS, print_out=lambda s: None):
            for limit in (None, 72, 73):
                kw = {} if limit is None else dict(max_parameters=limit)
                with mock.patch.object(B, "BoundedResponse", side_effect=RuntimeError("metadata admission reached runner")) as runner:
                    if limit == 72:
                        with self.assertRaisesRegex(RuntimeError, "metadata admission reached runner"):
                            B.bounded_properties(object(), object(), **args, **kw)
                        runner.assert_called_once()
                    else:
                        with self.assertRaises(ValueError):
                            B.bounded_properties(object(), object(), **args, **kw)
                        runner.assert_not_called()


class ResourceTests(unittest.TestCase):
    def test_actual72_metadata_design_and_exact_byte_work_boundaries(self):
        m = drive("pdef")["models"][0]
        admission = RES.design_admission(STREAM, m)
        self.assertEqual(admission["used_channels"], 96)
        self.assertEqual(admission["planned_work"], 60047184000)
        self.assertLess(admission["planned_work"], STREAM.PackedDesignRows.MAX_WORK)
        RES.design_admission(STREAM, m, max_bytes=admission["planned_bytes"])
        with self.assertRaisesRegex(ValueError, "byte resource limit"):
            RES.design_admission(STREAM, m, max_bytes=admission["planned_bytes"]-1)
        with mock.patch.object(STREAM.PackedDesignRows, "MAX_WORK", admission["planned_work"]):
            RES.design_admission(STREAM, m)
        with mock.patch.object(STREAM.PackedDesignRows, "MAX_WORK", admission["planned_work"]-1):
            with self.assertRaisesRegex(ValueError, "replay work"):
                RES.design_admission(STREAM, m)
        print("ACTUAL72_DESIGN_METADATA " + json.dumps(admission, sort_keys=True))

    def test_actual_planners_fullchain_io_and_exact_ledger(self):
        isa = A.planner_evidence()["candidate"]
        self.assertEqual(isa["standalone_ISA_bytes"]+isa["shared_bytes"], 26570915696)
        self.assertEqual(26843545600-isa["total_bytes"], 272629904)
        evidence = RES.evidence(B, isa)
        self.assertEqual(B._pfit_plan(2000, 108, 72, 11), (326608384, 94490544000))
        self.assertEqual(evidence["checkpoint_io_bytes"], 6700030544)
        self.assertLess(evidence["fullchain_source_work_bound"], 64000000000000)
        self.assertLess(evidence["total_io_bytes"], 68719476736)
        # Full-count work and I/O tested at exact ledger boundaries, never refunded.
        with mock.patch.dict(B.core.__dict__, print_out=lambda s: None):
            for short in (False, True):
                budget = BR.BoundedResources(isa["total_bytes"]-int(short), 64000000000000,
                                             68719476736, "large")
                ledger = BR._Ledger(budget, reserved=isa["shared_bytes"])
                if short:
                    with self.assertRaises(ValueError):
                        ledger.admit("ISA numeric boundary", isa["standalone_ISA_bytes"])
                else:
                    ledger.admit("ISA numeric boundary", isa["standalone_ISA_bytes"])
            for short in (False, True):
                work = evidence["fullchain_source_work_bound"]
                io = evidence["total_io_bytes"]
                budget = BR.BoundedResources(26843545600, work-int(short), io-int(short), "large")
                ledger = BR._Ledger(budget, reserved=isa["shared_bytes"])
                if short:
                    with self.assertRaises(ValueError):
                        ledger.admit("complete source bound", isa["standalone_ISA_bytes"], work)
                    with self.assertRaises(ValueError):
                        ledger.charge_io(io)
                else:
                    ledger.admit("complete source bound", isa["standalone_ISA_bytes"], work)
                    ledger.charge_io(io)
                    self.assertEqual((ledger.work, ledger.io, ledger.peak), (work, io, 26570915696))
        print("ACTUAL72_FULLCHAIN_SOURCE_EVIDENCE " + json.dumps(evidence, sort_keys=True))


class OperationalTests(unittest.TestCase):
    def test_argv_guards_and_new_paths(self):
        args = J.argv("a"*40, "b"*40, "c"*64, "d"*64)
        self.assertIn("--test-only", args)
        self.assertTrue(args[4].endswith("/pdef_full.sbatch"))
        with self.assertRaises(ValueError):
            J.argv("a"*40, "b"*40, None, None)
        with mock.patch.object(J.R, "compute_guard", side_effect=ValueError("local")), \
             mock.patch.object(J.R, "source_binding") as binding:
            with self.assertRaisesRegex(ValueError, "local"):
                J.run("a"*40, "b"*40, "c"*64, "d"*64)
            binding.assert_not_called()

    def test_actual_shell_term_failure_success_and_no_authority(self):
        script = (HERE/"pdef_full.sbatch").read_text()
        for value in ("--account=gts-cs207-chemx", "--partition=cpu-small", "--qos=embers",
                      "--cpus-per-task=8", "--mem=48G", "--time=06:00:00",
                      "--nodelist=atl1-1-02-007-23-1", "--no-requeue", "-I -S -B",
                      "NO NEW ACTIVATION AUTHORITY"):
            self.assertIn(value, script)
        for key in ("source", "run", "scratch"):
            self.assertIn(J.R.PLAN[key], script)
        traps = script.split("term_received=0\n", 1)[1].split('test "$#" = 4', 1)[0]
        for action, code in (("child_ok=1; exit 0", 0), ("exit 2", 2),
                             ("child_ok=1; kill -TERM $$; exit 0", 143)):
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp)/"metadata").mkdir()
                process = subprocess.run(["bash"], input="set -euo pipefail\nRUN="+tmp+"\nterm_received=0\n"+traps+action,
                                         text=True, capture_output=True)
                self.assertEqual(process.returncode, code)
                record = json.loads((Path(tmp)/"metadata/batch-terminal.json").read_text())
                self.assertEqual(record["complete"], code == 0)

    def test_job_failure_keeps_capture_and_primary_error_even_if_closure_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"metadata").mkdir()
            binding = dict(effective_profile={"synthetic": True})
            J.R.write(root/"metadata/preparation.json", dict(binding=binding,
                runtime_manifest_sha256="c"*64, derivative_sha256="d"*64,
                script_sha256=J.R.sha(HERE/"pdef_full.sbatch")))
            def initialize(*args):
                args[-1].append(object())
                return object(), object()
            def science(*args):
                marker = root/"results/diagnostic/node-01/pre-lw-complete.json"
                marker.parent.mkdir(parents=True)
                J.R.write(marker, {"properties_accepted": False})
                raise RuntimeError("synthetic scientific refusal")
            with mock.patch.dict(J.R.PLAN, run=str(root)), mock.patch.dict(os.environ, SLURM_JOB_ID="SYNTHETIC"), \
                 mock.patch.object(J.R, "compute_guard"), mock.patch.object(J.R, "source_binding", return_value=binding), \
                 mock.patch.object(J.R, "pinned_json", return_value={}), mock.patch.object(J.R, "derived_document", return_value={}), \
                 mock.patch.object(J.R, "initialize", side_effect=initialize), \
                 mock.patch.object(J.R, "module", return_value=NS(capture_science=science)), \
                 mock.patch.object(J.R.OLD, "snapshot", side_effect=ValueError("synthetic closure refusal")):
                self.assertEqual(J.run("a"*40, "b"*40, "c"*64, "d"*64), 1)
            record = json.loads((root/"results/result.json").read_text())
            self.assertFalse(record["science_pass"])
            self.assertIn("scientific refusal", record["error"])
            self.assertIn("closure refusal", record["closure_error"])
            self.assertEqual(len(record["complete_pre_lw_capture_receipts"]), 1)
            self.assertFalse(json.loads((root/"metadata/payload-terminal.json").read_text())["complete"])


if __name__ == "__main__":
    unittest.main()
