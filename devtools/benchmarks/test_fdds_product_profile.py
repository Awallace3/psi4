import importlib.util
import ast
from pathlib import Path
import unittest
import subprocess
import tempfile

spec = importlib.util.spec_from_file_location("profile_products", Path(__file__).with_name("fdds_product_profile.py"))
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class Product:
    shape = (3, 3)

    def __init__(self, name):
        self.name = name

    def dot(self, other):
        return Product(f"({self.name}.{other.name})")


def fixture():
    K_A = K_B = Product("K")
    Rtinv_A = Rtinv_B = Product("R")
    metric = Product("M")
    metric_inv = Product("I")
    W_A = W_B = Product("W")
    X_A = X_B = Product("X")
    XSW_A = XSW_B = Product("S")
    amplitude = Product("A")
    X_A_uc = X_A_coupled = Product("U")
    return [
        K_A.dot(Rtinv_A).dot(metric), K_B.dot(Rtinv_B).dot(metric),
        X_A.dot(metric_inv).dot(W_A), X_B.dot(metric_inv).dot(W_B),
        XSW_A.dot(amplitude).dot(X_A), XSW_B.dot(amplitude).dot(X_B),
        metric_inv.dot(X_A_uc).dot(metric_inv),
        metric_inv.dot(X_A_coupled).dot(metric_inv),
    ]


def incompatible():
    return Product("a").dot(Product("b"))


class TestProfile(unittest.TestCase):
    def test_numpy_precedes_psi4_like_frozen_runner(self):
        imports = []
        sentinel = object()

        def import_module(name):
            imports.append(name)
            return sentinel

        self.assertIs(profile.load_driver(import_module), sentinel)
        self.assertEqual(imports, ["numpy", "psi4.driver.procrouting.sapt.sapt_mp2_terms"])

    def test_exact_association_and_repeated_calls(self):
        records = []
        instrumented, digest = profile.instrument(fixture, records)
        expected = [value.name for value in fixture()]
        for _ in range(2):
            self.assertEqual([v.name for v in instrumented()], expected)
        self.assertEqual(len(records), 16)
        self.assertEqual(set(r["expression"] for r in records), profile.EXPECTED)
        self.assertEqual(len(digest), 64)
        self.assertTrue(all(r["seconds"] >= 0 for r in records))

    def test_fail_closed_on_other_driver(self):
        with self.assertRaises(ValueError):
            profile.instrument(incompatible, [])

    def test_original_globals_unchanged(self):
        profile.instrument(fixture, [])
        self.assertNotIn("_fdds_profile_product", globals())

    def test_actual_frozen_driver_has_exact_eight_sites(self):
        root = Path(__file__).resolve().parents[2]
        frozen = subprocess.check_output([
            "git", "show", "ada1a1d2128d6d6f3f5b89952b08f6ffd4ca5917:"
            "psi4/driver/procrouting/sapt/sapt_mp2_terms.py"], cwd=root, text=True)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "frozen_driver.py"
            source.write_text(frozen)
            tree = ast.parse(frozen)
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                            and n.name == "df_fdds_dispersion")
            namespace = {}
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
            transformed, _ = profile.instrument(namespace["df_fdds_dispersion"], [])
            self.assertEqual(len(transformed.profile_ast_sha256), 64)


if __name__ == "__main__":
    unittest.main()
