# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Synthetic engine-free convergence contracts, not native admission."""
import copy
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
import weakref

HERE = Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE/file)
    item = importlib.util.module_from_spec(spec)
    sys.modules[name] = item
    spec.loader.exec_module(item)
    return item


C = load("toy_convergence", "qgrid_convergence.py")
J = load("toy_convergence_job", "convergence_job.py")


class CaseTests(unittest.TestCase):
    def test_four_explicit_counts_and_cumulative_work(self):
        self.assertEqual([(c["name"], c["points"]) for c in C.CASES],
            [("baseline", 1408920), ("radial", 2116920),
             ("angular", 2325912), ("joint", 3494712)])
        plan = C.accounting(C.CASES)
        self.assertEqual(plan["values"], sum(c["points"] for c in C.CASES)*1146)
        self.assertEqual(plan["max_grid_bytes"], 3494712*32)
        self.assertEqual(plan["vector_bytes"], 4*5*1146*8)
        self.assertLessEqual(plan["estimated_output_bytes"], J.PLAN["bounds"]["max_output_bytes"])
        self.assertEqual(J.PLAN["bounds"]["max_values"], plan["values"])
        self.assertEqual(J.PLAN["bounds"]["deadline_seconds"], 1500)

    def test_unknown_reordered_or_extended_cases_refused(self):
        for bad in (list(reversed(C.CASES)), C.CASES[:3], C.CASES+C.CASES[:1]):
            with self.assertRaises(ValueError):
                C.accounting(bad)
        bad = copy.deepcopy(C.CASES)
        bad[1]["recipe"]["radial_points"] = 400
        with self.assertRaises(ValueError):
            C.admit_case(bad[1])

    def test_options_and_actual_atom_major_layout(self):
        for case in C.CASES:
            opt = SimpleNamespace()
            C.configure(opt, case)
            self.assertEqual(vars(opt), case["recipe"])
            stride = (opt.radial_points-1)*opt.spherical_points
            grid = SimpleNamespace(npoints=lambda: stride*12,
                radial_points=lambda: opt.radial_points,
                spherical_points=lambda: opt.spherical_points,
                atom_start=lambda a: stride*a, atom_npoints=lambda a: stride)
            layout = C.layout(grid, case)
            points = C.selection(layout)
            self.assertEqual(len(points), 24)
            self.assertEqual({p["atom"] for p in points}, set(range(12)))
            self.assertEqual({p["index"]//stride for p in points}, set(range(12)))
            self.assertTrue(all(p["index"] == p["atom"]*stride+
                p["radial"]*opt.spherical_points+p["angular"] for p in points))
            if case["name"] == "baseline":
                self.assertEqual(tuple(p["index"] for p in points), J.Q.P.selected_points())
            grid.atom_npoints = lambda a: stride-1
            with self.assertRaises(ValueError):
                C.layout(grid, case)

    def test_new_selection_each_layout(self):
        indices = []
        for case in C.CASES:
            radial = case["recipe"]["radial_points"]-1
            angular = case["recipe"]["spherical_points"]
            indices.append([p["index"] for p in C.selection(dict(
                starts=[a*radial*angular for a in range(13)],
                shells=radial, angular=angular))])
        self.assertEqual(len({tuple(v) for v in indices}), 4)

    def test_hash_baseline_fail_closed_variations_measured_not_pretended(self):
        import numpy as np
        grid = np.ones((2, 4), dtype="<f8")
        digest = hashlib.sha256(grid.tobytes()).hexdigest()
        # Small synthetic canonical hashing seam, no relaxation of case admission.
        self.assertEqual(C.canonical_hash(grid, 2), digest)
        for bad in (grid.astype("f4"), grid[:, :3], grid[::-1], grid*float("nan")):
            with self.assertRaises(ValueError):
                C.canonical_hash(bad, 2)
        with self.assertRaisesRegex(ValueError, "baseline"):
            C.hash_gate(C.CASES[0], digest, [])
        C.hash_gate(C.CASES[0], C.BASELINE_HASH, [])
        with self.assertRaises(ValueError):
            C.hash_gate(C.CASES[1], C.BASELINE_HASH, [C.BASELINE_HASH])
        C.hash_gate(C.CASES[1], digest, [C.BASELINE_HASH])
        with self.assertRaises(ValueError):
            C.hash_gate(C.CASES[2], digest, [C.BASELINE_HASH, digest])
        with self.assertRaises(ValueError):
            C.hash_gate(C.CASES[1], digest, [])


class ReductionTests(unittest.TestCase):
    def test_signed_effects_and_interaction_no_monotonicity_assumption(self):
        vectors = dict(baseline=[1., -2.], radial=[3., -5.],
                       angular=[-1., 4.], joint=[7., 1.])
        effects = C.effects(vectors)
        self.assertEqual(effects["radial_at_590"], [2., -3.])
        self.assertEqual(effects["radial_at_974"], [8., -3.])
        self.assertEqual(effects["angular_at_200"], [-2., 6.])
        self.assertEqual(effects["angular_at_300"], [4., 6.])
        self.assertEqual(effects["interaction"], [6., 0.])
        with self.assertRaises(ValueError):
            C.effects({k: v for k, v in vectors.items() if k != "joint"})

    def test_signed_site_shell_and_old_S_label(self):
        rows = [dict(column=i, site="C1", shell=i//2, centre=0, l=0) for i in range(3)]
        report = C.diagnostics([2., -3., 1.], [0.]*3, [4.]*3, rows, baseline=False)
        self.assertEqual(report["U_minus_q"]["minimum"]["signed"], -3.)
        self.assertEqual(report["U_minus_q"]["maximum"]["signed"], 2.)
        self.assertEqual(len(report["by_shell"]), 2)
        self.assertNotIn("S_minus_U", report)
        self.assertIn("cross_grid", report["S_baseline_minus_U_label"])
        self.assertIn("S_minus_U", C.diagnostics([2., -3., 1.], [0.]*3, [4.]*3,
                                               rows, baseline=True))

    def test_baseline_exact_vector_differences_retained_before_refusal(self):
        import numpy as np
        old = {key: np.array([1., 2.]) for key in C.VECTOR_KEYS}
        new = {key: value.copy() for key, value in old.items()}
        new["U"][0] += 1e-12
        result = C.remeasurement(new, old)
        self.assertFalse(result["all_exact"])
        self.assertGreater(result["vectors"]["U"]["maxabs"], 0)
        self.assertIn("difference_hex", result["vectors"]["U"])

    def test_single_deadline_not_reset(self):
        clock = mock.Mock(side_effect=[0., 1499., 1501.])
        deadline = C.deadline(1500, clock=clock)
        deadline()
        with self.assertRaises(TimeoutError):
            deadline()


class EvidenceTests(unittest.TestCase):
    def fixture(self):
        import struct
        pins = J.PLAN["baseline_probe"]["files"]
        capture = dict(marker_sha256="marker", anchor_sha256="anchor",
                       basis=dict(auxiliary={"synthetic": True}))
        binding = {"separate_original_source": True}
        docs = {
            "results/result.json": dict(job="14308581", probe_complete=True,
                properties_accepted=False, after_runtime_closure="passed",
                commit=J.PLAN["base"], tree="1fae89606b94f8d54dbf9f621f96ab03e114657f",
                native_commit=J.R.OLD.BUILD_COMMIT, native_core_sha256=J.R.OLD.CORE_SHA,
                diagnostic_sha256=pins["results/grid-diagnostic.json"],
                runtime_manifest_sha256="runtime", derivative_admission_sha256="derivative"),
            "results/grid-diagnostic.json": dict(grid=J.Q.PLAN["grid"],
                original_marker_sha256="marker", original_anchor_sha256="anchor",
                vectors={k: dict(file=k+".f64", dtype="<f8", shape=[1146], nbytes=9168,
                    sha256=pins["results/"+k+".f64"]) for k in C.VECTOR_KEYS}),
            "results/grid-identity.json": dict(measured_sha256=C.BASELINE_HASH,
                                               hash_verified_before_integration=True),
            "results/basis-identities-grid-only.json": dict(auxiliary={"synthetic": True}),
            "metadata/batch-terminal.json": dict(exit_code=0, term_received=0,
                child_unsignalled_zero=1, complete=True),
            "metadata/payload-terminal.json": dict(complete=True, after_runtime_closure="passed"),
            "metadata/runtime.json": dict(binding=binding, runtime_manifest_sha256="runtime"),
            "metadata/preparation.json": dict(binding=binding, runtime_manifest_sha256="runtime",
                                              derivative_sha256="derivative"),
            "metadata/before-capture.json": {"closure": "exact"},
            "metadata/after-capture.json": {"closure": "exact"}}
        vectors = {"results/"+k+".f64": struct.pack("<1146d", *([0.]*1146)) for k in C.VECTOR_KEYS}
        return capture, docs, vectors

    def verify(self, capture, docs, vectors):
        raw = {k: json.dumps(v).encode() for k, v in docs.items()} | vectors
        def read(path, pin):
            relative = str(path.relative_to(Path("baseline")))
            self.assertEqual(pin, J.PLAN["baseline_probe"]["files"][relative])
            return raw[relative]
        with mock.patch.object(J.Q, "verify_capture", return_value=capture), \
             mock.patch.object(J.Q, "read_pinned", side_effect=read):
            return J.verify_inputs("capture", "baseline")

    def test_original_source_runtime_grid_capture_and_vector_links(self):
        capture, docs, vectors = self.fixture()
        self.assertEqual(len(self.verify(capture, docs, vectors)["baseline_vectors"]["U"]), 1146)
        mutations = [
            ("results/result.json", "job", "other"),
            ("results/result.json", "commit", J.Q.PLAN["base"]),
            ("results/result.json", "after_runtime_closure", "failed"),
            ("results/grid-identity.json", "measured_sha256", "0"*64),
            ("metadata/batch-terminal.json", "term_received", 1),
            ("metadata/payload-terminal.json", "complete", False),
            ("metadata/after-capture.json", "closure", "changed"),
            ("metadata/runtime.json", "binding", {"wrong": True}),
            ("results/grid-diagnostic.json", "original_anchor_sha256", "wrong"),
            ("results/basis-identities-grid-only.json", "auxiliary", {"wrong": True})]
        for file, field, value in mutations:
            with self.subTest(file=file, field=field):
                bad = copy.deepcopy(docs)
                bad[file][field] = value
                with self.assertRaises(ValueError):
                    self.verify(capture, bad, vectors)

    def test_repinning_wrong_vector_descriptor_or_nonfinite_is_not_enough(self):
        capture, docs, vectors = self.fixture()
        docs["results/grid-diagnostic.json"]["vectors"]["U"]["dtype"] = ">f8"
        with self.assertRaisesRegex(ValueError, "descriptor"):
            self.verify(capture, docs, vectors)
        capture, docs, vectors = self.fixture()
        import struct
        vectors["results/U.f64"] = struct.pack("<1146d", *([float("nan")]*1146))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.verify(capture, docs, vectors)


class StorageAdmissionTests(unittest.TestCase):
    def finalize_batch(self, run, code):
        """Execute the actual shell terminal trap, without scheduler commands."""
        import subprocess
        text = (HERE/"convergence.sbatch").read_text()
        traps = text.split("term_received=0\n", 1)[1].split('test "$#" = 4', 1)[0]
        script = ("set -euo pipefail\nRUN="+str(run)+"\nterm_received=0\n"+traps+
                  f"child_ok={int(code == 0)}; exit {code}\n")
        result = subprocess.run(["bash"], input=script, text=True, capture_output=True)
        self.assertEqual(result.returncode, code, result.stderr)
        return json.loads((run/"metadata/batch-terminal.json").read_text())

    def run_synthetic(self, run, scratch, *, snapshot=None, science=None):
        """Real orchestration/storage/receipts, synthetic native/runtime seams."""
        def initialize(*args):
            args[-1].append(object())
            return object(), args[-1][0]
        with mock.patch.dict(J.PLAN, run=str(run), scratch=str(scratch)), \
             mock.patch.dict(J.os.environ, SLURM_JOB_ID="TOY"), \
             mock.patch.object(J.R, "compute_guard"), \
             mock.patch.object(J, "binding_for", return_value={}), \
             mock.patch.object(J, "verify_preparation"), \
             mock.patch.object(J.R, "initialize", side_effect=initialize), \
             mock.patch.object(J.R, "verify_runtime"), \
             mock.patch.object(J.R.OLD, "snapshot", side_effect=snapshot), \
             mock.patch.object(J, "science", side_effect=science, return_value=dict(
                 study_complete=True, completed_cases=[c["name"] for c in C.CASES])), \
             mock.patch.object(J.signal, "signal"), \
             mock.patch.object(J.resource, "setrlimit"), \
             mock.patch.object(J.time, "monotonic", return_value=0.):
            return J.run("a"*40, "b"*40, "c"*64, "d"*64)

    def test_storage_traversal_permission_errors_are_not_omitted(self):
        for role in ("run", "scratch"):
            for subtree in (False, True):
                with self.subTest(role=role, subtree=subtree), tempfile.TemporaryDirectory() as tmp:
                    roots = {name: Path(tmp)/name for name in ("run", "scratch")}
                    for root in roots.values():
                        root.mkdir()
                    blocked = roots[role]/"private" if subtree else roots[role]
                    blocked.mkdir(exist_ok=True)
                    real_scandir = J.os.scandir
                    error = PermissionError(13, "synthetic inaccessible storage", str(blocked))
                    def scandir(path):
                        if Path(path) == blocked:
                            raise error
                        return real_scandir(path)
                    with mock.patch.dict(J.PLAN, {k: str(v) for k, v in roots.items()}), \
                         mock.patch.object(J.os, "scandir", side_effect=scandir), \
                         self.assertRaises(PermissionError) as caught:
                        J.storage_bounds()
                    self.assertIs(caught.exception, error)

    def test_both_storage_roots_must_be_real_directories(self):
        for role in ("run", "scratch"):
            for kind in ("missing", "file", "symlink"):
                with self.subTest(role=role, kind=kind), tempfile.TemporaryDirectory() as tmp:
                    roots = {name: Path(tmp)/name for name in ("run", "scratch")}
                    for name, root in roots.items():
                        if name != role:
                            root.mkdir()
                    if kind == "file":
                        roots[role].write_bytes(b"x")
                    elif kind == "symlink":
                        roots[role].symlink_to(roots["scratch" if role == "run" else "run"])
                    with mock.patch.dict(J.PLAN, {k: str(v) for k, v in roots.items()}), \
                         self.assertRaises((OSError, ValueError)):
                        J.storage_bounds()

    def test_storage_root_disappearance_during_scan_is_not_omitted(self):
        for role in ("run", "scratch"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as tmp:
                roots = {name: Path(tmp)/name for name in ("run", "scratch")}
                for root in roots.values():
                    root.mkdir()
                real_walk = J.os.walk
                def walk(root, **kw):
                    yield from real_walk(root, **kw)
                    if Path(root) == roots[role]:
                        roots[role].rmdir()
                with mock.patch.dict(J.PLAN, {k: str(v) for k, v in roots.items()}), \
                     mock.patch.object(J.os, "walk", side_effect=walk), \
                     self.assertRaises(OSError):
                    J.storage_bounds()

    def test_inaccessible_closure_subtree_withholds_actual_run_completion(self):
        for role in ("run", "scratch"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as tmp:
                run, scratch = Path(tmp)/"run", Path(tmp)/"scratch"
                (run/"metadata").mkdir(parents=True)
                scratch.mkdir()
                blocked = (run if role == "run" else scratch)/"private"
                def snapshot(*args, **kw):
                    blocked.mkdir()
                real_scandir = J.os.scandir
                def scandir(path):
                    if Path(path) == blocked:
                        raise PermissionError(13, "synthetic closure storage", str(blocked))
                    return real_scandir(path)
                with mock.patch.object(J.os, "scandir", side_effect=scandir):
                    self.assertEqual(self.run_synthetic(run, scratch, snapshot=snapshot), 1)
                result = json.loads((run/"results/result.json").read_text())
                terminal = json.loads((run/"metadata/payload-terminal.json").read_text())
                self.assertEqual(result["after_runtime_closure"], "passed")
                self.assertEqual(result["final_budgets"], "failed")
                self.assertIn("PermissionError", result["budget_error"])
                self.assertFalse(result["study_complete"])
                self.assertFalse(terminal["complete"])

    def test_all_three_final_receipts_fit_artifact_boundary_including_batch(self):
        for existing, expected in ((93, 0), (94, 1)):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as tmp:
                run, scratch = Path(tmp)/"run", Path(tmp)/"scratch"
                (run/"metadata").mkdir(parents=True)
                scratch.mkdir()
                def snapshot(*args, **kw):
                    for index in range(existing):
                        (run/"results"/f"partial-{index}").write_bytes(b"x")
                code = self.run_synthetic(run, scratch, snapshot=snapshot)
                batch = self.finalize_batch(run, code)
                files = [p for p in run.rglob("*") if p.is_file()]
                self.assertEqual(code, expected, f"{len(files)} artifacts after actual batch finalization")
                result = json.loads((run/"results/result.json").read_text())
                payload = json.loads((run/"metadata/payload-terminal.json").read_text())
                self.assertEqual(batch["complete"], expected == 0)
                self.assertEqual(payload["complete"], expected == 0)
                self.assertEqual(result["study_complete"], expected == 0)
                if expected == 0:
                    self.assertEqual(len(files), 96)
                    self.assertEqual(result["final_storage"]["files"], 93)
                else:
                    self.assertEqual(result["final_budgets"], "failed")

    def test_final_receipt_bytes_reserved_at_output_and_total_boundaries(self):
        # Two <=4MiB JSON writes and <=64KiB for the fixed shell receipt.
        reservation = 8454144
        for bound in ("max_output_bytes", "max_total_bytes"):
            for delta, expected in ((0, 0), (1, 1)):
                with self.subTest(bound=bound, delta=delta), tempfile.TemporaryDirectory() as tmp:
                    run, scratch = Path(tmp)/"run", Path(tmp)/"scratch"
                    (run/"metadata").mkdir(parents=True)
                    scratch.mkdir()
                    retained_bytes = 2*1024**2+delta
                    def snapshot(*args, **kw):
                        (run/"results/retained").write_bytes(b"x"*retained_bytes)
                    def science(*args):
                        return dict(study_complete=True, completed_cases=[c["name"] for c in C.CASES],
                                    retained_diagnostic="x"*65536)
                    limit = 2*1024**2+reservation
                    with mock.patch.dict(J.PLAN["bounds"], {bound: limit}):
                        code = self.run_synthetic(run, scratch, snapshot=snapshot, science=science)
                    batch = self.finalize_batch(run, code)
                    self.assertEqual(code, expected)
                    self.assertEqual(batch["complete"], expected == 0)
                    result = json.loads((run/"results/result.json").read_text())
                    self.assertEqual(result["final_budgets"], "passed" if expected == 0 else "failed")
                    self.assertEqual(result["study_complete"], expected == 0)
                    if expected == 0:
                        receipts = [run/"results/result.json", run/"metadata/payload-terminal.json",
                                    run/"metadata/batch-terminal.json"]
                        receipt_bytes = sum(p.stat().st_size for p in receipts)
                        self.assertGreater(receipt_bytes, 65536)
                        self.assertLessEqual(receipt_bytes, reservation)
                        self.assertLessEqual(retained_bytes+receipt_bytes, limit)


class JobTests(unittest.TestCase):
    def test_sequential_stops_first_failure_without_retry_or_extension(self):
        for stop in range(4):
            called = []
            def execute(case, previous):
                called.append(case["name"])
                self.assertEqual(len(previous), len(called)-1)
                if len(called)-1 == stop:
                    raise ValueError("stop")
                return dict(name=case["name"], grid_sha256=case["name"])
            with self.assertRaisesRegex(ValueError, "stop"):
                J.sequential(execute, lambda: None)
            self.assertEqual(called, [c["name"] for c in C.CASES[:stop+1]])

    def test_cumulative_storage_scratch_logs_and_nonregular_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, scratch = root/"run", root/"scratch"
            (run/"logs").mkdir(parents=True)
            scratch.mkdir()
            (run/"a").write_bytes(b"x"*5)
            (run/"logs/stdout").write_bytes(b"x"*7)
            (scratch/"tile").write_bytes(b"x"*3)
            with mock.patch.dict(J.PLAN, run=str(run), scratch=str(scratch)):
                actual = J.storage_bounds()
                self.assertEqual((actual["output_bytes"], actual["log_bytes"],
                                  actual["scratch_bytes"]), (5, 7, 3))
                for key, limit in (("max_output_bytes", 4), ("max_scratch_bytes", 2),
                        ("max_total_bytes", 14), ("max_log_bytes", 6), ("max_files", 2)):
                    with mock.patch.dict(J.PLAN["bounds"], {key: limit}), self.assertRaises(ValueError):
                        J.storage_bounds()
                with self.assertRaises(ValueError):
                    J.storage_bounds(reserve=J.PLAN["bounds"]["max_output_bytes"])
                with self.assertRaises(ValueError):
                    J.storage_bounds(reserve_files=J.PLAN["bounds"]["max_files"])
                (run/"link").symlink_to(scratch)
                with self.assertRaises(ValueError):
                    J.storage_bounds()

    def test_actual_one_case_hash_gate_before_stream_and_release(self):
        """Small fake grid through the real case path; never a molecular grid."""
        import numpy as np
        real_empty = np.empty
        references = []
        def allocate(shape, **kw):
            gc.collect()
            self.assertTrue(all(ref() is None for ref in references))
            grid = real_empty((12, 4), **kw)
            references.append(weakref.ref(grid))
            return grid
        native_refs = []
        class Native:
            def __init__(self, *args):
                native_refs.append(weakref.ref(self))
            def x(self): return np.arange(12)
            def y(self): return 0.
            def z(self): return 0.
            def w(self): return 1.
        rows = [dict(column=i, site="C1", shell=i//15, centre=0, l=0) for i in range(1146)]
        zero = np.zeros(1146)
        measured = dict(U=zero, U_compensated=zero, absolute_sum=zero, roundoff_bound=zero,
            tiles=1, collocation_checks=[], accumulation="TOY", error_bound_scope="TOY")
        data = dict(capture=dict(rows=rows, vectors=dict(q=zero, QT_c=zero)),
                    runtime_identity={"synthetic": True},
                    baseline_vectors={k: zero for k in C.VECTOR_KEYS})
        core = SimpleNamespace(IsaGridOptions=SimpleNamespace, IsaGrid=Native)
        psi4 = SimpleNamespace(core=core)
        molecule = SimpleNamespace(clone=lambda: object())
        fake_layout = dict(starts=[a*199*590 for a in range(13)], shells=199, angular=590)
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.dict(J.PLAN, run=tmp), \
             mock.patch.object(J.C, "layout", return_value=fake_layout), \
             mock.patch.object(np, "empty", side_effect=allocate), \
             mock.patch.object(J.C, "canonical_hash", return_value="0"*64), \
             mock.patch.object(J.C, "hash_gate", side_effect=ValueError("baseline")) as gate, \
             mock.patch.object(J.Q.P, "stream", return_value=measured) as stream, \
             mock.patch.object(J.Q.P, "selected_columns", return_value=(0,)), \
             mock.patch.object(J, "storage_bounds"):
            (Path(tmp)/"results").mkdir()
            with self.assertRaisesRegex(ValueError, "baseline"):
                J.one_case(psi4, molecule, object(), C.CASES[0], data, [], {}, lambda: None, [])
            stream.assert_not_called()
            self.assertTrue((Path(tmp)/"results/baseline/grid-identity.json").is_file())
            J.C.canonical_hash.reset_mock()
            J.C.layout.reset_mock()
            # Every next case allocation asserts earlier grid references released.
            gate.side_effect = None
            for case in C.CASES[1:]:
                result = J.one_case(psi4, molecule, object(), case, data, [], {}, lambda: None, [])
                self.assertEqual(result["name"], case["name"])
                self.assertNotIn("grid", result)
                self.assertEqual(stream.call_args.kwargs["check_deadline"](), None)
                stream.reset_mock()
                J.C.canonical_hash.reset_mock()
                J.C.layout.reset_mock()
            gc.collect()
            self.assertTrue(all(ref() is None for ref in references+native_refs))

    def test_one_case_retains_baseline_difference_then_refuses(self):
        import numpy as np
        zero, changed = np.zeros(1146), np.zeros(1146)
        changed[0] = .125
        measured = dict(U=changed, U_compensated=zero, absolute_sum=zero, roundoff_bound=zero,
            tiles=1, collocation_checks=[], accumulation="TOY", error_bound_scope="TOY")
        rows = [dict(column=i, site="C1", shell=0, centre=0, l=0) for i in range(1146)]
        data = dict(capture=dict(rows=rows, vectors=dict(q=zero, QT_c=zero)),
            runtime_identity={"synthetic": True},
            baseline_vectors={k: zero for k in C.VECTOR_KEYS})
        native = SimpleNamespace(x=lambda: 0., y=lambda: 0., z=lambda: 0., w=lambda: 1.)
        psi4 = SimpleNamespace(core=SimpleNamespace(IsaGridOptions=SimpleNamespace,
                                                    IsaGrid=lambda *args: native))
        real_empty = np.empty
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(J.PLAN, run=tmp), \
             mock.patch.object(np, "empty", side_effect=lambda *a, **kw: real_empty((12, 4), **kw)), \
             mock.patch.object(J.C, "layout", return_value=dict(starts=[a*199*590 for a in range(13)],
                 shells=199, angular=590)), \
             mock.patch.object(J.C, "canonical_hash", return_value=C.BASELINE_HASH), \
             mock.patch.object(J.Q.P, "selected_columns", return_value=(0,)), \
             mock.patch.object(J.Q.P, "stream", return_value=measured), \
             mock.patch.object(J, "storage_bounds"):
            (Path(tmp)/"results").mkdir()
            with self.assertRaisesRegex(ValueError, "byte reproduction"):
                J.one_case(psi4, SimpleNamespace(clone=lambda: object()), object(),
                           C.CASES[0], data, [], {}, lambda: None, [])
            saved = json.loads((Path(tmp)/"results/baseline/remeasurement.json").read_text())
            self.assertFalse(saved["all_exact"])
            self.assertEqual(saved["vectors"]["U"]["maxabs"], .125)
            self.assertTrue((Path(tmp)/"results/baseline/diagnostic.json").is_file())
            self.assertEqual((Path(tmp)/"results/baseline/U.f64").stat().st_size, 9168)

    def test_no_forbidden_calls_and_exact_TERM_safe_script(self):
        import ast
        for file in ("convergence_job.py", "qgrid_convergence.py"):
            tree = ast.parse((HERE/file).read_text())
            calls = [ast.unparse(n.func).split(".")[-1] for n in ast.walk(tree)
                     if isinstance(n, ast.Call)]
            self.assertFalse(set(calls)&{"energy", "oeprop", "capture_science", "native_partition",
                                        "solve", "replay", "invoke_pilot"})
        text = (HERE/"convergence.sbatch").read_text()
        for required in ("--qos=embers", "--time=00:30:00", "--cpus-per-task=2",
                "--mem=8G", "--nodelist=atl1-1-02-007-23-1", "--no-requeue",
                "convergence_job.py", "-I -S -B", "wait \"$child\"",
                "trap '' TERM INT HUP", "$child_ok = 1", "$term_received = 0"):
            self.assertIn(required, text)
        self.assertNotIn("sbatch ", text)
        self.assertNotIn("--dependency", text)
        self.assertNotIn("qgrid_job.py", text)

    def test_shell_terminal_traps_actual_synthetic_success_failure_TERM(self):
        import subprocess
        text = (HERE/"convergence.sbatch").read_text()
        traps = text.split("term_received=0\n", 1)[1].split('test "$#" = 4', 1)[0]
        for action, expected in (("child_ok=1; exit 0", 0),
                ("exit 2", 2), ("child_ok=1; kill -TERM $$; exit 0", 143)):
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp)/"metadata").mkdir()
                script = "set -euo pipefail\nRUN="+tmp+"\nterm_received=0\n"+traps+action
                result = subprocess.run(["bash"], input=script, text=True, capture_output=True)
                self.assertEqual(result.returncode, expected, result.stderr)
                terminal = json.loads((Path(tmp)/"metadata/batch-terminal.json").read_text())
                self.assertEqual(terminal["complete"], expected == 0)
                self.assertEqual(terminal["exit_code"], expected)

    def test_prior_and_new_plans_separate_original_capture_identity(self):
        self.assertNotEqual(J.R.PLAN["source"], J.Q.R.PLAN["source"])
        self.assertEqual(J.PLAN["base"], "e24f97e9b7fbafcc25231be50feb1c39a38b5862")
        self.assertEqual(J.Q.PLAN["base"], "c31ad7fbe6ca2f91e6654e25807536009c58bb29")
        self.assertEqual(J.PLAN["cases"], C.CASES)

    def test_actual_science_orchestrates_four_cases_same_deadline_and_effects(self):
        rows = [dict(column=i, site="C1", shell=0, centre=0, l=0) for i in range(1146)]
        runtime = dict(python="TOY", numpy="TOY", adcc="TOY", node="TOY")
        data = dict(capture=dict(rows=rows), baseline_runtime=runtime)
        clock_check = mock.Mock()
        called = []
        def one_case(psi4, molecule, explicit, case, actual_data, xyz, binding, check, previous):
            self.assertIs(check, clock_check)
            self.assertEqual(len(previous), len(called))
            self.assertEqual(actual_data["runtime_identity"]["runtime_manifest_sha256"], "c"*64)
            called.append(case["name"])
            index = len(called)
            return dict(name=case["name"], grid_sha256=str(index)*64,
                diagnostic_sha256="f"*64, U=[float(index)]*1146,
                U_compensated=[float(index)]*1146)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"metadata").mkdir()
            (root/"results").mkdir()
            (root/"scratch").mkdir()
            J.R.write(root/"metadata/runtime.json", runtime)
            J.R.write(root/"results/basis-identities-grid-only.json", {"synthetic": True})
            with mock.patch.dict(J.PLAN, run=str(root), scratch=str(root/"scratch")), \
                 mock.patch.dict(J.os.environ, SLURM_JOB_ID="TOY"), \
                 mock.patch.object(J.R, "compute_guard"), \
                 mock.patch.object(J, "verify_inputs", return_value=data), \
                 mock.patch.object(J, "basis_only", return_value=(object(), object(), [])), \
                 mock.patch.object(J, "one_case", side_effect=one_case), \
                 mock.patch.object(J.os, "statvfs", return_value=SimpleNamespace(f_bavail=2**30, f_frsize=1)):
                result = J.science(object(), {"toy": True}, "c"*64, "d"*64, clock_check)
            self.assertEqual(called, [c["name"] for c in C.CASES])
            self.assertTrue(result["study_complete"])
            saved = json.loads((root/"results/comparisons.json").read_text())
            self.assertEqual(saved["effects"]["U"]["radial_at_590"]["summary"]["signed_sum"], 1146.)
            self.assertEqual(saved["effects"]["U"]["interaction"]["summary"]["maxabs"], 0.)

    def test_final_storage_scan_is_unthrottled_and_failure_withholds_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"metadata").mkdir()
            def initialize(*args):
                args[-1].append(object())
                return object(), args[-1][0]
            with mock.patch.dict(J.PLAN, run=str(root)), \
                 mock.patch.dict(J.os.environ, SLURM_JOB_ID="TOY"), \
                 mock.patch.object(J.R, "compute_guard"), \
                 mock.patch.object(J, "binding_for", return_value={}), \
                 mock.patch.object(J, "verify_preparation"), \
                 mock.patch.object(J.R, "initialize", side_effect=initialize), \
                 mock.patch.object(J.R, "verify_runtime"), \
                 mock.patch.object(J.R.OLD, "snapshot"), \
                 mock.patch.object(J, "science", return_value=dict(study_complete=True,
                     completed_cases=[c["name"] for c in C.CASES])), \
                 mock.patch.object(J.signal, "signal"), \
                 mock.patch.object(J.resource, "setrlimit"), \
                 mock.patch.object(J.time, "monotonic", return_value=0.), \
                 mock.patch.object(J, "storage_bounds", side_effect=[{}, ValueError("final mixed storage")]) as scan:
                self.assertEqual(J.run("a"*40, "b"*40, "c"*64, "d"*64), 1)
                self.assertEqual(scan.call_count, 2)
            result = json.loads((root/"results/result.json").read_text())
            self.assertEqual(result["after_runtime_closure"], "passed")
            self.assertEqual(result["final_budgets"], "failed")
            self.assertFalse(result["study_complete"])

    def test_guard_before_any_native_or_output(self):
        with mock.patch.object(J.R, "compute_guard", side_effect=ValueError("local")), \
             mock.patch.object(J, "science") as science:
            with self.assertRaisesRegex(ValueError, "local"):
                J.run("a"*40, "b"*40, "c"*64, "d"*64)
            science.assert_not_called()

    def test_actual_argv_requires_real_manifest_pins(self):
        with self.assertRaises(ValueError):
            J.argv("a"*40, "b"*40, None, None)
        args = J.argv("a"*40, "b"*40, "c"*64, "d"*64)
        self.assertIn("--test-only", args)
        self.assertTrue(args[4].endswith("/convergence.sbatch"))
        self.assertIn("--parsable", J.argv("a"*40, "b"*40, "c"*64, "d"*64, test_only=False))

    def test_lifecycle_partial_loader_case_failure_closure_and_TERM(self):
        for failure in ("none", "loader", "science", "closure", "term", "incomplete"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root/"metadata").mkdir()
                (root/"scratch").mkdir()
                binding = {"toy": True}
                retained = []
                token = object()
                def initialize(*args):
                    args[-1].append(token)
                    retained.append(args[-1])
                    if failure == "loader":
                        raise ValueError("loader")
                    return object(), token
                def snapshot(*args, **kw):
                    self.assertIs(retained[0][0], token)
                    if failure == "closure":
                        raise ValueError("closure")
                def science(*args):
                    if failure == "science":
                        raise ValueError("case")
                    if failure == "term":
                        raise J.Terminated("TERM")
                    return dict(study_complete=failure != "incomplete",
                        completed_cases=[c["name"] for c in C.CASES] if failure != "incomplete" else ["baseline"])
                with mock.patch.dict(J.PLAN, dict(run=str(root), runtime=str(root/"runtime"),
                                                scratch=str(root/"scratch"))), \
                     mock.patch.object(J.R, "compute_guard"), \
                     mock.patch.object(J, "binding_for", return_value=binding), \
                     mock.patch.object(J, "verify_preparation"), \
                     mock.patch.object(J.R, "initialize", side_effect=initialize), \
                     mock.patch.object(J.R, "verify_runtime"), \
                     mock.patch.object(J.R.OLD, "snapshot", side_effect=snapshot), \
                     mock.patch.object(J, "science", side_effect=science), \
                     mock.patch.object(J.signal, "signal"), \
                     mock.patch.object(J.resource, "setrlimit"), \
                     mock.patch.dict(J.os.environ, SLURM_JOB_ID="TOY"):
                    self.assertEqual(J.run("a"*40, "b"*40, "c"*64, "d"*64),
                                     0 if failure == "none" else 1)
                result = json.loads((root/"results/result.json").read_text())
                self.assertEqual(result["study_complete"], failure == "none")
                self.assertFalse(result["science_pass"])
                self.assertFalse(result["properties_accepted"])

    def test_import_stdlib_only(self):
        import subprocess
        command = ("import importlib.util,sys; "
            f"s=importlib.util.spec_from_file_location('c', {str(HERE/'convergence_job.py')!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "assert not {'numpy','psi4','adcc','einsums'} & set(sys.modules)")
        subprocess.run([sys.executable, "-I", "-S", "-B", "-c", command], check=True)


if __name__ == "__main__":
    unittest.main()
