# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Static-only adapter contract checks. No adapter/Psi4/NumPy import or execution."""
import ast
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TEXT = (HERE / "pilot.py").read_text()
TREE = ast.parse(TEXT)
CONFIG = json.loads((HERE / "pilot.json").read_text())


def source_tree(name):
    return ast.parse((ROOT / "psi4/driver/procrouting" / name).read_text())


class SourceContract(unittest.TestCase):
    def test_syntax_and_import_boundary(self):
        compile(TREE, str(HERE / "pilot.py"), "exec")  # compile ONLY; never exec
        for node in TREE.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.assertNotIn("numpy", [a.name for a in node.names])
                self.assertNotIn("psi4", [a.name for a in node.names])
        self.assertIn('host != allocated_node', TEXT)
        self.assertIn('host.startswith("login")', TEXT)

    def test_public_oeprop_keywords(self):
        api = source_tree("isapol_bounded_oeprop.py")
        settings = next(n for n in api.body if isinstance(n, ast.FunctionDef) and n.name == "settings")
        allowed = {k.arg for n in ast.walk(settings) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name) and n.func.id == "dict" for k in n.keywords}
        call = next(n for n in ast.walk(TREE) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == "oeprop")
        keys = {k.arg for k in call.keywords if k.arg} - {"atomic_backend", "preset"}
        keys.update(set(CONFIG["refinement"]) - {"penalty_selection"})
        keys.update(CONFIG["dispersion"])
        self.assertFalse(keys - allowed)

    def test_controller_keywords(self):
        api = source_tree("isapol_native_partition.py")
        controller = next(n for n in api.body if isinstance(n, ast.ClassDef) and n.name == "ControllerRecipe")
        fields = {n.target.id for n in controller.body if isinstance(n, ast.AnnAssign)}
        call = next(n for n in ast.walk(TREE) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == "ControllerRecipe")
        self.assertEqual({k.arg for k in call.keywords}, fields)

    def test_public_export_contract(self):
        api = source_tree("isapol_lw.py")
        coefficient = next(n for n in api.body if isinstance(n, ast.ClassDef) and n.name == "Coefficient")
        fields = {n.target.id for n in coefficient.body if isinstance(n, ast.AnnAssign)}
        self.assertIn("unrestricted_complete", fields)
        export = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "export")
        attrs = {n.attr for n in ast.walk(export) if isinstance(n, ast.Attribute)}
        self.assertNotIn("complete", attrs)
        self.assertIn("unrestricted_complete", attrs)
        self.assertIn('local.array[..., :8, :8]', TEXT)
        self.assertIn("COMPONENT_NAMES[1:9]", TEXT)

    def test_scientific_pins(self):
        self.assertEqual(CONFIG["candidate_base"], "6f147f4829ffc83c868b9e78618ebe7c4f17fbdb")
        self.assertEqual(CONFIG["resources"]["psi4_memory_bytes"], 24 * 1024**3)
        self.assertEqual(CONFIG["resources"]["numerical_threads"], 1)
        self.assertEqual(CONFIG["refinement"]["max_refinement_parameters"], 72)
        self.assertEqual(CONFIG["response"]["production_tolerance"], 1e-6)
        self.assertEqual(CONFIG["isa"]["tail_fit_policy"], "always")
        self.assertEqual(CONFIG["isa"]["q_grid"], [200, 590])
        self.assertEqual(CONFIG["response"]["grid"], [100, 200])


if __name__ == "__main__":
    unittest.main()
