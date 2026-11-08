"""Pure accounting only: no Psi4, arrays, grids, SCF or native imports."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "psi4/driver/procrouting/isapol_distribution.py"
HELPER = ROOT / "psi4/driver/procrouting/isapol_isa_grid_plan.py"
PLANNER = ROOT / "devtools/isapol_s66_campaign/resource_plan_isa_grid_v1.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def runtime_plan(*, width=1340, grid_changes=None, source=None):
    """Execute only the real scalar planner, using dimension-only descriptors."""
    tree = ast.parse(PYTHON.read_text() if source is None else source)
    names = {"isa_resource_plan", "_q_accumulation_workspace_bytes"}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    native = ModuleType("_pure_isa.isapol_native_partition")
    native.DRHO_REFINEMENT_ITERATIONS = 32
    package = ModuleType("_pure_isa")
    modules = {"_pure_isa": package, native.__name__: native}
    if HELPER.exists():
        helper = load(HELPER, "_pure_isa.isapol_isa_grid_plan")
        modules[helper.__name__] = helper
    namespace = {"__package__": "_pure_isa", "_width": lambda value: value}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(PYTHON), "exec"), namespace)
    atomic = (109, 49, 49, 49, 109, 49, 49, 109, 49, 49, 109, 49, 49, 109, 49, 49, 49)
    shapes = (12, 8, 8, 8, 12, 8, 8, 12, 8, 8, 12, 8, 8, 12, 8, 8, 8)
    recipe = NS(sites=[NS(atomic=a, shape=s) for a, s in zip(atomic, shapes)],
                grid=NS(radial_points=100, spherical_points=200, becke_smoothing=3,
                        radius_scaling=1., radius_policy="native_tabulated_bragg_slater",
                        neighbour_policy="all_sites_unscreened_full_molecular_grid"),
                auxiliary=1340, controller=NS(max_iterations=120))
    for name, value in (grid_changes or {}).items():
        setattr(recipe.grid, name, value)
    wfn = NS(basisset=lambda: NS(nbf=lambda: 506))
    with patch.dict(sys.modules, modules):
        value = namespace["isa_resource_plan"](wfn, recipe, width, range(13769490), 4)
    return value, namespace["_q_accumulation_workspace_bytes"]


class GridAccountingTests(unittest.TestCase):
    def test_runtime_closed_grid_reduction_not_thread_credit(self):
        numeric, work = runtime_plan()[0]
        # Original scalar formula, with actual 99*230 rows rather than 99*406.
        old_work = 93794716607520
        row_credit = 25839479761920
        self.assertLess(work, old_work)
        self.assertGreater(work, old_work-row_credit)  # extra Q control charged
        self.assertLess(numeric, 95179407072-624444224)

    def test_closed_recipe_and_conservative_fallback(self):
        helper = load(HELPER, "_isa_grid_test_helper")
        recipe = dict(radial_points=100, spherical_points=200, becke_smoothing=3,
                      radius_scaling=1., radius_policy="native_tabulated_bragg_slater",
                      neighbour_policy="all_sites_unscreened_full_molecular_grid")
        for sites in (1, 3, 17, 100):
            self.assertEqual(helper.isa_grid_rows(sites, **recipe), sites*99*230)
        for field, value in (("radial_points", 101), ("spherical_points", 194),
                             ("becke_smoothing", 2), ("radius_scaling", 1.1),
                             ("radius_policy", "other"), ("neighbour_policy", "other")):
            changed = dict(recipe, **{field: value})
            expected = 17*(changed["radial_points"]-1)*min(5294, 2*changed["spherical_points"]+6)
            self.assertEqual(helper.isa_grid_rows(17, **changed), expected)
        self.assertEqual(helper.isa_grid_rows(17, **dict(recipe, spherical_points=9999)),
                         17*99*5294)
        for field, value in (("radial_points", True), ("radial_points", 1),
                             ("spherical_points", 0), ("becke_smoothing", -1),
                             ("radius_scaling", float("nan"))):
            with self.assertRaises(ValueError):
                helper.isa_grid_rows(17, **dict(recipe, **{field: value}))
        with self.assertRaises(ValueError):
            helper.isa_grid_rows(True, **recipe)

    def test_work_and_workspace_boundaries(self):
        helper = load(HELPER, "_isa_work_test_helper")
        workspace = runtime_plan()[1]
        for width in (1, 64, 65, 1340, 1048576):
            for rank in range(5):
                block = min(4096, 1048576//width)
                for rows in (0, 1, block, block+1, 13769490):
                    extra = helper.q_preparation_work(17, width, rank, rows)
                    self.assertGreaterEqual(extra, 0)
                    self.assertLessEqual(workspace(width, rank, rows), 827392)
                    if rows:
                        self.assertGreater(extra, 17*rows*(rank+1)**2)
        for args in ((0, 1, 4, 1), (17, 1048577, 4, 1), (17, 1, 5, 1),
                     (17, 1, 4, -1), (17, True, 4, 1)):
            with self.assertRaises(ValueError):
                helper.q_preparation_work(*args)

    def test_general_runtime_retains_legacy_grid_and_adds_control_work(self):
        source = subprocess.check_output(
            ["git", "-C", str(ROOT), "show",
             "2fe7a711dd915e102fe6976f44d22571064aa31e:"+str(PYTHON.relative_to(ROOT))],
            text=True)
        helper = load(HELPER, "_isa_fallback_test_helper")
        for changes in ({"radial_points": 101}, {"spherical_points": 194},
                        {"spherical_points": 9999}, {"becke_smoothing": 2},
                        {"radius_scaling": 1.1}):
            old = runtime_plan(grid_changes=changes, source=source)[0]
            new = runtime_plan(grid_changes=changes)[0]
            self.assertEqual(new[0], old[0])
            self.assertEqual(new[1]-old[1], helper.q_preparation_work(17, 1340, 4, 13769490))

    def test_native_source_selector_and_envelope_for_all_requests(self):
        native = (ROOT/"psi4/src/psi4/libisapol/isa_grid.cc").read_text()
        cubature = (ROOT/"psi4/src/psi4/libfock/cubature.cc").read_text()
        angular = sorted({int(n) for n in re.findall(r"\{\d+,\s*(\d+),\s*mk\w+Grid", cubature)
                          if 6 <= int(n) <= 5294})
        self.assertIn("lebedev_npoints_at_least(requested)", native)
        self.assertIn("const int n_shell = options_.radial_points - 1;", native)
        self.assertIn("if (n >= npoints && (best < 0 || n < best)) best = n;", cubature)
        self.assertEqual(min(n for n in angular if n >= 200), 230)
        for request in range(1, 6001):
            actual = min(n for n in angular if n >= min(request, 5294))
            self.assertLessEqual(actual, min(5294, 2*request+6))

    def test_new_ledger_all_stage_consistency_caps_and_runtime(self):
        planner = load(PLANNER, "_isa_grid_test_planner")
        dimensions = planner.Dimensions(
            17, 16, 506, 178, 21, 1340, 1340,
            (109, 49, 49, 49, 109, 49, 49, 109, 49, 49, 109, 49, 49, 109, 49, 49, 49),
            (12, 8, 8, 8, 12, 8, 8, 12, 8, 8, 12, 8, 8, 12, 8, 8, 8))
        resources = dict(max_bytes=96*1024**3, max_work=128*10**12,
                         max_io_bytes=16*1024**3, capacity="extended")
        kwargs = dict(max_replay_work=10**12, pfit_maximum_work_bytes=2*1024**3)
        old = planner._legacy.plan(dimensions, resources, **kwargs)
        self.assertEqual(old["charged_work"], 128996205999228)
        self.assertEqual(old["refusals"], ["cumulative_work"])
        result = planner.plan(dimensions, resources, **kwargs)
        self.assertEqual(result["refusals"], [])
        self.assertEqual(result["dimensions"]["isa_grid_accounting_rows"], 387090)
        self.assertEqual(result["charged_work"], sum(s["charged_work"] for s in result["stages"]))
        self.assertEqual(result["peak_numeric_bytes"], max(s["numeric_bytes"] for s in result["stages"]))
        self.assertEqual(result["admission_numeric_bytes"], result["peak_numeric_bytes"])
        stage = result["stages"][1]
        numeric, work = runtime_plan()[0]
        self.assertEqual((stage["transient_bytes"], stage["work_per_call"]), (numeric, work))
        for before, after in zip(old["stages"], result["stages"]):
            if before["name"] != stage["name"]:
                self.assertEqual(before, after)
        self.assertEqual(old["checkpoint_io_bytes"], result["checkpoint_io_bytes"])
        self.assertIsNone(result["policy_scientific_commit"])
        for key, limit, refusal in (
                ("max_work", result["charged_work"], "cumulative_work"),
                ("max_bytes", result["peak_numeric_bytes"], "numeric_bytes"),
                ("max_io_bytes", result["checkpoint_io_bytes"], "checkpoint_io")):
            exact = planner.plan(dimensions, dict(resources, **{key: limit}), **kwargs)
            short = planner.plan(dimensions, dict(resources, **{key: limit-1}), **kwargs)
            self.assertNotIn(refusal, exact["refusals"])
            self.assertIn(refusal, short["refusals"])
        with self.assertRaises(ValueError):
            planner.plan(dimensions, dict(resources, max_work=128*10**12+1), **kwargs)

    def test_source_binding_and_historical_policy_untouched(self):
        planner = load(PLANNER, "_isa_grid_binding_test")
        binding = planner.source_binding(ROOT)
        self.assertFalse(binding["qualified"])
        for name, expected in planner.FILES.items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), expected)
        # Simulate one changed byte without copying a scientific source tree.
        original = Path.read_bytes
        target = ROOT/"psi4/driver/procrouting/isapol_isa_grid_plan.py"
        with patch.object(Path, "read_bytes",
                          lambda path: original(path)+(b"\n" if path == target else b"")):
            with self.assertRaisesRegex(ValueError, "sources differ"):
                planner.source_binding(ROOT)
        original_runtime = subprocess.check_output(
            ["git", "-C", str(ROOT), "show",
             "2fe7a711dd915e102fe6976f44d22571064aa31e:"+str(PYTHON.relative_to(ROOT))])
        with patch.object(Path, "read_bytes",
                          lambda path: original_runtime if path == PYTHON else original(path)):
            with self.assertRaisesRegex(ValueError, "sources differ"):
                planner.source_binding(ROOT)
        with self.assertRaisesRegex(ValueError, "sources differ"):
            planner._legacy.source_binding(ROOT)  # old policy is NOT repinned
        for name in ("resource_plan_cb63305.py", "resource_plan_ed561666.py",
                     "resource_plan_grid400.py"):
            path = ROOT/"devtools/isapol_s66_campaign"/name
            old = subprocess.check_output(["git", "-C", str(ROOT), "show",
                "2fe7a711dd915e102fe6976f44d22571064aa31e:"+str(path.relative_to(ROOT))])
            self.assertEqual(path.read_bytes(), old)


if __name__ == "__main__":
    unittest.main(verbosity=2)
