# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Engine-free synthetic probes; not molecular/native admission."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE/file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P = load("toy_qgrid_probe", "qgrid_probe.py")
J = load("toy_qgrid_job", "qgrid_job.py")


def synthetic_snapshot():
    return dict(metadata=dict(nbf=6), centres_bohr_hex=[
        ["0x0p+0"]*3, ["0x1p+0", "0x0p+0", "0x0p+0"]],
        ordered_shells=[dict(centre=a, l=1, is_pure=False, nprimitive=1,
            exponents_hex=[(.7).hex()],
            native_normalized_coefficients_hex=[(1.25).hex()]) for a in (0, 1)])


class OracleTests(unittest.TestCase):
    def test_literal_gamint_order_and_factors_through_g(self):
        self.assertEqual([len(p) for p in P.POWERS], [1, 3, 6, 10, 15])
        self.assertEqual(P.POWERS[3][7], (1, 0, 2))
        self.assertEqual(P.POWERS[4][9:12], ((2, 2, 0), (2, 0, 2), (0, 2, 2)))
        self.assertEqual(P.angular_factor((2, 0, 0)), 1.)
        self.assertAlmostEqual(P.angular_factor((1, 1, 0)), math.sqrt(3))
        self.assertAlmostEqual(P.angular_factor((2, 2, 0)), math.sqrt(35/3))

    def test_independent_signed_contraction_and_translation(self):
        row = dict(powers=(1, 0, 2), centre_bohr=(1., -2., 3.),
                   exponents=(.5, 2.), coefficients=(2., -.3))
        point = (1.25, -2.5, 3.75)
        r2 = .25**2 + .5**2 + .75**2
        expected = math.sqrt(5)*.25*.75**2*(2*math.exp(-.5*r2)-.3*math.exp(-2*r2))
        self.assertAlmostEqual(P.oracle(point, row), expected)
        self.assertEqual(P.oracle(row["centre_bohr"], row), 0.)

    def test_every_component_through_g_even_odd(self):
        for powers in sum(P.POWERS, ()):
            row = dict(powers=powers, centre_bohr=(0., 0., 0.),
                       exponents=(.7,), coefficients=(1.3,))
            point = (.2, -.3, .4)
            reflected = (-.2, -.3, .4)
            self.assertEqual(P.oracle(reflected, row),
                             (-1)**powers[0]*P.oracle(point, row))
            self.assertTrue(math.isfinite(P.oracle(point, row)))

    def test_nonfinite_or_bad_effective_snapshot_rejected(self):
        snapshot = synthetic_snapshot()
        for field, value in (("exponents_hex", ["nan"]),
                             ("native_normalized_coefficients_hex", ["inf"]),
                             ("exponents_hex", ["-0x1p+0"])):
            bad = copy.deepcopy(snapshot)
            bad["ordered_shells"][0][field] = value
            with self.assertRaises(ValueError):
                P.columns(bad)

    def test_columns_are_shell_major_and_complete_all_centres(self):
        rows = P.columns(synthetic_snapshot())
        self.assertEqual(len(rows), 6)
        self.assertEqual([r["centre"] for r in rows], [0, 0, 0, 1, 1, 1])
        self.assertEqual([r["powers"] for r in rows[:3]], list(P.POWERS[1]))
        self.assertEqual([r["column"] for r in rows], list(range(6)))
        self.assertEqual(rows[0]["coefficients"], (1.25,))


class GridTests(unittest.TestCase):
    def test_recipe_is_exact_not_default_or_similar(self):
        options = SimpleNamespace()
        P.configure_grid(options, P.GRID_RECIPE)
        self.assertEqual(vars(options), P.GRID_RECIPE)
        for key in P.GRID_RECIPE:
            recipe = dict(P.GRID_RECIPE)
            recipe[key] += 1
            with self.assertRaisesRegex(ValueError, "recipe"):
                P.configure_grid(SimpleNamespace(), recipe)

    def test_canonical_hash_preserves_coordinates_weights_and_order(self):
        import numpy as np
        grid = np.array([[1., 2., 3., .25], [4., 5., 6., -.5]], dtype="<f8")
        pin = hashlib.sha256(grid.tobytes(order="C")).hexdigest()
        self.assertEqual(P.validate_grid(grid, 2, pin), pin)
        for bad in (grid[::-1].copy(), grid+1, grid[:, ::-1].copy()):
            with self.assertRaisesRegex(ValueError, "grid hash"):
                P.validate_grid(bad, 2, pin)
        for bad in (grid.astype("f4"), grid[:, :3], grid[:1],
                    np.array([[math.nan]*4]*2)):
            with self.assertRaises(ValueError):
                P.validate_grid(bad, 2, pin)

    def test_complete_selection_not_only_probe_origin(self):
        basis = SimpleNamespace(nfunction=1146)
        self.assertEqual(P.all_centres(basis, 12, 1146), list(range(12)))
        with self.assertRaises(ValueError):
            P.all_centres(SimpleNamespace(nfunction=912), 12, 1146)


class StreamTests(unittest.TestCase):
    def toy(self, *, tile=2, limit=100, output=1024**2, deadline=None, mutate=None):
        import numpy as np
        grid = np.array([[.1, .2, .3, .5], [.2, .1, -.4, 1.],
                         [-.5, .4, .1, -.2], [.8, .2, .1, .1]], dtype="<f8")
        values = np.arange(24, dtype=float).reshape(4, 6)/10-1
        calls = []
        def evaluate(points, sites):
            self.assertLessEqual(len(points), tile)
            self.assertEqual(sites, [0, 1])
            start = len(calls)*tile
            calls.append(points)
            data = values[start:start+len(points)].copy()
            if mutate:
                data = mutate(data)
            return SimpleNamespace(np=data)
        basis = SimpleNamespace(nfunction=6, evaluate_screened=evaluate)
        limits = dict(tile_points=tile, max_values=limit, max_output_bytes=output,
                      deadline_seconds=100)
        actual = P.stream(basis, grid, P.columns(synthetic_snapshot()), limits,
                          selected_points=(0, 3), selected_columns=(0, 4),
                          check_deadline=deadline or (lambda: None))
        return actual, grid, values, calls

    def test_one_bounded_pass_signed_integral_abs_and_compensated(self):
        import numpy as np
        actual, grid, values, calls = self.toy()
        expected = np.sum(values*grid[:, 3, None], axis=0)
        np.testing.assert_allclose(actual["U"], expected, atol=1e-15)
        np.testing.assert_allclose(actual["U_compensated"], expected, atol=1e-15)
        np.testing.assert_allclose(actual["absolute_sum"],
                                   np.sum(np.abs(values*grid[:, 3, None]), axis=0))
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(actual["collocation_checks"]), 4)
        self.assertGreaterEqual(actual["roundoff_bound"].min(), 0)

    def test_work_output_and_tile_bounds_fail_before_native_call(self):
        for kwargs in (dict(limit=23), dict(output=1), dict(tile=0), dict(tile=4097)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.toy(**kwargs)

    def test_deadline_rejection(self):
        with self.assertRaisesRegex(TimeoutError, "TOY"):
            self.toy(deadline=mock.Mock(side_effect=TimeoutError("TOY budget")))

    def test_nonfinite_or_wrong_width_refuses(self):
        def infinite(x):
            x[:] = float("inf")
            return x
        for mutate in (lambda x: x[:, :1], infinite):
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.toy(mutate=mutate)

    def test_measured_oracle_differences_not_forced_tolerance(self):
        actual, _, _, _ = self.toy()
        checks = actual["collocation_checks"]
        self.assertTrue(any(c["native_minus_oracle"] != 0 for c in checks))
        self.assertTrue(all("tolerance" not in c for c in checks))

    def test_signed_comparison_columns_and_decomposition(self):
        rows = P.columns(synthetic_snapshot())
        q, s, u = [1., 0., 2., 0., 3., 0.], [2., -3., 2., 1., 3., 0.], [1.5, -2., 2., .5, 3., 0.]
        result = P.compare(u, q, s, rows)
        self.assertEqual(result["U_minus_q"]["peak"]["column"], 1)
        self.assertEqual(result["U_minus_q"]["peak"]["signed"], -2.)
        self.assertEqual(result["S_minus_U"]["peak"]["signed"], -1.)
        self.assertEqual(result["deltaq"]["peak"]["signed"], -3.)
        self.assertEqual(result["decomposition_maxabs"], 0.)

    def test_exclusive_vector_lossless_and_finite_output(self):
        import struct
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"U.f64"
            meta = P.write_vector(path, [0., -0., 1.25], expected=3)
            self.assertEqual(path.read_bytes(), struct.pack("<3d", 0., -0., 1.25))
            self.assertEqual(meta["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                P.write_vector(path, [0.]*3, expected=3)
            with self.assertRaises(ValueError):
                P.write_vector(Path(tmp)/"bad.f64", [math.nan], expected=1)


class EvidenceAndLaunchTests(unittest.TestCase):
    def test_exact_marker_anchor_and_q_s_input_pins(self):
        pins = J.PLAN["capture"]["files"]
        self.assertEqual(pins["results/diagnostic/node-01/pre-lw-complete.json"],
                         "8e14c234d725008be6431cfdfa2d02cd6c87c33ad5fd4ff1f58c6809b7a6fe9c")
        self.assertEqual(pins["results/diagnostic/anchor/q.f64"],
                         "1ed38a3488788baba91b6c30ce044442a15d485b58bafc65f0fad99b955cea82")
        self.assertEqual(pins["results/diagnostic/anchor/QT_c.f64"],
                         "f999e897727090869400590e2032d5cf563ffd3fd4c993e0ed4cf875696288e7")
        self.assertEqual(J.PLAN["grid"]["sha256"],
                         "a01a17a4ab55dab608883cf5a184bfc13a42a15fae15efecf0256a9df383ab29")

    def test_import_is_metadata_only_fresh_isolated_interpreter(self):
        script = ("import importlib.util,sys; "
                  f"s=importlib.util.spec_from_file_location('q', {str(HERE/'qgrid_job.py')!r}); "
                  "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                  "assert not {'numpy','psi4','adcc','einsums'} & set(sys.modules)")
        subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script], check=True)

    def test_guard_local_before_numerical_or_output(self):
        with mock.patch.object(J.R, "compute_guard", side_effect=ValueError("TOY local")), \
             mock.patch.object(J, "science", side_effect=AssertionError("science forbidden")):
            with self.assertRaisesRegex(ValueError, "TOY local"):
                J.run("a"*40, "b"*40, "c"*64, "d"*64)

    def test_only_new_grid_launcher_and_no_forbidden_dispatch(self):
        args = J.argv("a"*40, "b"*40, "c"*64, "d"*64)
        self.assertIn("--test-only", args)
        self.assertTrue(any(x.endswith("/qgrid_probe.sbatch") for x in args))
        for file in ("qgrid_job.py", "qgrid_probe.py"):
            tree = ast.parse((HERE/file).read_text())
            calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
            for forbidden in ("energy", "oeprop", "native_partition", "capture_science",
                              "integrated_driver", "invoke_pilot", "solve", "replay"):
                self.assertFalse(any(c.split(".")[-1] == forbidden for c in calls), (file, forbidden))
        text = (HERE/"qgrid_probe.sbatch").read_text()
        for required in ("--account=gts-cs207-chemx", "--qos=embers", "--partition=cpu-small",
                         "--cpus-per-task=2", "--mem=8G", "--time=00:30:00",
                         "--nodelist=atl1-1-02-007-23-1", "-I -S -B", "qgrid_job.py"):
            self.assertIn(required, text)
        self.assertNotIn("integrated_job.py", text)
        self.assertNotIn("pilot.py", text)
        self.assertNotIn("sbatch ", text)

    def test_arguments_hex_guards_and_no_fallback(self):
        for pins in (("bad", "b"*40, "c"*64, "d"*64),
                     ("a"*40, "b"*40, "bad", "d"*64)):
            with self.assertRaises(ValueError):
                J.argv(*pins)

    def test_retained_handle_final_closure_and_probe_not_science_acceptance(self):
        text = (HERE/"qgrid_job.py").read_text()
        self.assertIn("handles", text)
        self.assertIn('snapshot("after-capture"', text)
        self.assertIn("hash_core=True", text)
        self.assertIn("science_pass=False", text)
        self.assertIn("finally:", text)

    def test_hash_match_precedes_any_grid_integral_or_comparison(self):
        tree = ast.parse((HERE/"qgrid_job.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "science")
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
        where = lambda name: [n.lineno for n in calls if ast.unparse(n.func) == name]
        self.assertEqual(len(where("P.validate_grid")), 1)
        self.assertEqual(len(where("P.stream")), 1)
        self.assertLess(where("P.validate_grid")[0], where("P.stream")[0])
        self.assertTrue(all(where("P.validate_grid")[0] < line for line in where("P.compare")))
        self.assertEqual(len(where("psi4.core.IsaGrid")), 1)

    def test_plan_namespace_is_separate_and_default_capture_unchanged(self):
        self.assertNotEqual(J.R.PLAN["run"], J.GENERIC.PLAN["run"])
        self.assertEqual(J.GENERIC.PLAN["allocation"]["cpus"], 8)
        self.assertEqual(J.R.PLAN["allocation"]["cpus"], 2)
        self.assertEqual(J.R.PLAN["allocation"]["psi4_memory_bytes"], 4*1024**3)
        with self.assertRaises(ValueError):
            J.GENERIC.for_plan(HERE.parent/"outside.json")

    def test_sbatch_term_safe_guards_and_frozen_final_status(self):
        text = (HERE/"qgrid_probe.sbatch").read_text()
        final = text.split("on_exit() {", 1)[1].split("\n}", 1)[0]
        self.assertIn("trap - EXIT", final)
        self.assertIn("trap '' TERM INT HUP", final)
        self.assertIn('exit "$rc"', final)
        self.assertIn("$child_ok = 1", final)
        before, launch = text.split("/opt/slurm/current/bin/srun", 1)
        self.assertIn("if (( term_received )); then exit 143; fi", before)
        self.assertIn('if (( term_received )); then kill -TERM "$child" || true; fi', launch)
        self.assertIn("--no-requeue", text)
        self.assertNotIn("--dependency", text)


class CaptureChainTests(unittest.TestCase):
    """Entirely synthetic saved-byte fixtures, including1146-column dimensions."""
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        snapshot = dict(metadata=dict(nbf=1146), centres_bohr_hex=[["0x0p+0"]*3]*12,
            ordered_shells=[dict(centre=a, l=0, is_pure=False, nprimitive=1,
                exponents_hex=["0x1p+0"], native_normalized_coefficients_hex=["0x1p+0"])
                for a in range(12) for _ in range(101 if a == 0 else 95)])
        main = dict(synthetic_main="metadata only")
        for item in (snapshot, main):
            item["fingerprint_sha256"] = J.R.BASIS.digest(item)
        basis = dict(auxiliary=snapshot, main=main)
        vectors = dict(q=[0.]*1146, QT_c=[0.]*1146, deltaq=[0.]*1146)
        vectors["QT_c"][254] = vectors["deltaq"][254] = -.125
        self.pins = {}
        import struct
        for key, values in vectors.items():
            self.put("results/diagnostic/anchor/"+key+".f64", struct.pack("<1146d", *values))
        arrays = {key: dict(dtype="<f8", file=key+".f64", nbytes=9168, shape=[1146],
                            sha256=self.pins["results/diagnostic/anchor/"+key+".f64"])
                  for key in vectors}
        self.anchor = dict(arrays=arrays, metadata=dict(Q_sha256="a"*64, Q_shape=[300, 1146]))
        self.put_json("results/diagnostic/anchor/manifest.json", self.anchor)
        self.put_json("results/basis-identities-before-scf.json", basis)
        cfg = json.loads((HERE/"pilot.json").read_text())
        self.context = dict(context=dict(
            basis_receipt_sha256=self.pins["results/basis-identities-before-scf.json"],
            original_probe_sha256=J.PLAN["prior_probe_sha256"]),
            identities=dict(driver=dict(source_commit=J.PLAN["base"]),
                fixture_sha256=cfg["fixture_sha256"], config_sha256=J.R.sha(HERE/"pilot.json")))
        self.put_json("results/diagnostic/context.json", self.context)
        source = dict(partition=dict(q_sha256="a"*64, diagnostics=dict(
            integration_grid_sha256=J.PLAN["grid"]["sha256"],
            integration_grid_points=J.PLAN["grid"]["points"])))
        self.marker = dict(anchor_manifest_sha256=self.pins[
            "results/diagnostic/anchor/manifest.json"], properties_accepted=False, source_repr=repr(source))
        self.put_json("results/diagnostic/node-01/pre-lw-complete.json", self.marker)
        patch = mock.patch.dict(J.PLAN, dict(capture=dict(root=str(self.root), files=self.pins),
            fingerprints={role: basis[role]["fingerprint_sha256"] for role in basis}))
        patch.start()
        self.addCleanup(patch.stop)

    def put(self, name, data):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.pins[name] = hashlib.sha256(data).hexdigest()

    def put_json(self, name, data):
        self.put(name, json.dumps(data, sort_keys=True).encode())

    def test_exact_saved_marker_chain_and_all_columns(self):
        capture = J.verify_capture(self.root)
        self.assertEqual(len(capture["rows"]), 1146)
        self.assertEqual(capture["vectors"]["QT_c"][254], -.125)

    def test_mutated_vector_bytes_refuse_even_same_length(self):
        (self.root/"results/diagnostic/anchor/q.f64").write_bytes(b"0"*9168)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            J.verify_capture(self.root)

    def test_repinning_input_does_not_hide_wrong_original_marker_link(self):
        self.marker["anchor_manifest_sha256"] = "0"*64
        self.put_json("results/diagnostic/node-01/pre-lw-complete.json", self.marker)
        with self.assertRaisesRegex(ValueError, "chain mismatch"):
            J.verify_capture(self.root)

    def test_repinning_wrong_descriptor_or_snapshot_refuses(self):
        self.anchor["arrays"]["q"]["dtype"] = ">f8"
        self.put_json("results/diagnostic/anchor/manifest.json", self.anchor)
        self.marker["anchor_manifest_sha256"] = self.pins["results/diagnostic/anchor/manifest.json"]
        self.put_json("results/diagnostic/node-01/pre-lw-complete.json", self.marker)
        with self.assertRaisesRegex(ValueError, "descriptor"):
            J.verify_capture(self.root)

    def test_repinning_changed_effective_coefficient_still_refuses_fingerprint(self):
        path = "results/basis-identities-before-scf.json"
        basis = json.loads((self.root/path).read_text())
        basis["auxiliary"]["ordered_shells"][0]["native_normalized_coefficients_hex"] = ["0x1p+1"]
        self.put_json(path, basis)
        self.context["context"]["basis_receipt_sha256"] = self.pins[path]
        self.put_json("results/diagnostic/context.json", self.context)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            J.verify_capture(self.root)

    def test_capture_nonregular_symlink_and_oversize_are_rejected(self):
        path = self.root/"metadata-input"
        os = __import__("os")
        os.mkfifo(path)
        with self.assertRaisesRegex(ValueError, "type/size"):
            J.read_pinned(path, "0"*64)
        path.unlink()
        path.symlink_to(self.root/"results/diagnostic/context.json")
        with self.assertRaises(OSError):
            J.read_pinned(path, "0"*64)
        path.unlink()
        with path.open("wb") as stream:
            stream.truncate(4*1024**2+1)
        with self.assertRaisesRegex(ValueError, "type/size"):
            J.read_pinned(path, "0"*64)


class LifecycleTests(unittest.TestCase):
    def test_actual_run_success_failure_and_partial_loader_all_close_retained_handle(self):
        for failure in ("none", "science", "loader", "closure"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root/"metadata").mkdir()
                binding = {"new": "TOY binding"}
                runtime_sha, derivative_sha = "c"*64, "d"*64
                preparation = dict(binding=binding, runtime_manifest_sha256=runtime_sha,
                    derivative_sha256=derivative_sha, plan_sha256=J.R.sha(HERE/"qgrid_plan.json"),
                    script_sha256=J.R.sha(HERE/"qgrid_probe.sbatch"),
                    capture=J.PLAN["capture"], grid=J.PLAN["grid"], bounds=J.PLAN["bounds"])
                J.R.write(root/"metadata/preparation.json", preparation)
                retained, token = [], object()
                def initialize(*args):
                    args[-1].append(token)
                    retained.append(args[-1])
                    if failure == "loader":
                        raise ValueError("TOY partial loader error")
                    return object(), token
                def snapshot(*args, **kwargs):
                    self.assertIs(retained[0][0], token)
                    if failure == "closure":
                        raise ValueError("TOY closure error")
                scientific = mock.Mock(return_value=dict(probe_complete=True))
                if failure == "science":
                    scientific.side_effect = RuntimeError("TOY primary grid error")
                with mock.patch.dict(J.PLAN, dict(run=str(root), runtime=str(root/"runtime"))), \
                     mock.patch.object(J.R, "compute_guard"), \
                     mock.patch.object(J, "binding_for", return_value=binding), \
                     mock.patch.object(J, "verify_capture"), \
                     mock.patch.object(J.R, "pinned_json", return_value={}), \
                     mock.patch.object(J.R, "derived_document", return_value={}), \
                     mock.patch.object(J.R, "initialize", side_effect=initialize), \
                     mock.patch.object(J.R, "verify_runtime"), \
                     mock.patch.object(J.R.OLD, "snapshot", side_effect=snapshot) as closure, \
                     mock.patch.object(J, "science", scientific), \
                     mock.patch.object(J.signal, "signal"), \
                     mock.patch.object(J.resource, "setrlimit"), \
                     mock.patch.dict(J.os.environ, {"SLURM_JOB_ID": "TOY"}):
                    self.assertEqual(J.run("a"*40, "b"*40, runtime_sha, derivative_sha),
                                     0 if failure == "none" else 1)
                closure.assert_called_once()
                result = json.loads((root/"results/result.json").read_text())
                terminal = json.loads((root/"metadata/payload-terminal.json").read_text())
                self.assertEqual(result["probe_complete"], failure == "none")
                self.assertEqual(terminal["complete"], failure == "none")
                self.assertFalse(result["science_pass"])
                self.assertFalse(result["properties_accepted"])
                if failure == "science":
                    self.assertEqual(result["error"], "RuntimeError: TOY primary grid error")
                if failure == "loader":
                    scientific.assert_not_called()
                if failure == "closure":
                    self.assertEqual(result["after_runtime_closure"], "failed")


if __name__ == "__main__":
    unittest.main()
