"""Host-only tests for the explicit private-root qualification allowlist."""
import ast
from pathlib import Path
import unittest


class QualificationPaths(unittest.TestCase):
    def test_only_dedicated_roots_are_admitted(self):
        path = Path(__file__).resolve().parents[1] / "disp0/cachy/qualify_construction_root.py"
        tree = ast.Module(body=[
            node for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "require_private_root"
        ], type_ignores=[])
        namespace = {}
        exec(compile(tree, str(path), "exec"), namespace)
        require = namespace["require_private_root"]
        home = Path("/home/example")
        for name in ("psi4-disp0-perf-cachy", "psi4-fsapt-cpks-integration-cachy",
                     "psi4-fsapt-opt-cachy"):
            require(home / "gits" / name, home)
        for root in (
            home / "gits/psi4-disp0-cachy",
            home / "gits/psi4-saptdft-cuest-cachy",
            home / "gits/psi4-fsapt-cpks-integration-cachy/source",
            Path("/home/someone-else/gits/psi4-fsapt-cpks-integration-cachy"),
        ):
            with self.assertRaises(RuntimeError):
                require(root, home)


if __name__ == "__main__":
    unittest.main()
