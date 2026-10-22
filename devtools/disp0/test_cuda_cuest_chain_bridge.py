"""CPU-only regression for the native cuEST/Einsums-v2 merge boundary."""
import ast
from pathlib import Path
import types
import unittest
from unittest.mock import Mock


class Matrix:
    def __init__(self, label):
        self.label = label


class NativeChainBridge(unittest.TestCase):
    def fixture(self):
        path = Path(__file__).resolve().parents[2] / "psi4/driver/procrouting/sapt/sapt_jk_terms_ein.py"
        names = {"_use_cuest_gemm", "chain_gemm_einsums"}
        tree = ast.Module(body=[node for node in ast.parse(path.read_text()).body
                                if isinstance(node, ast.FunctionDef) and node.name in names],
                          type_ignores=[])
        outputs = [Matrix("product")]
        native = Mock(return_value=outputs)
        core = types.SimpleNamespace(Matrix=Matrix, cuest_chain_gemm=native,
            get_global_option=lambda name: {"USE_CUEST":True, "CUEST_GEMM_MIN_DIM":1}[name])
        convert = Mock(side_effect=lambda value: ("einsums", value.label))
        ns = {"core":core, "_HAVE_CUEST_GEMM":True, "_chain_gemm_work":lambda *args:8,
              "_ein":convert}
        exec(compile(tree, str(path), "exec"), ns)
        return ns, native, convert, outputs

    def test_native_matrix_output_does_not_convert(self):
        ns, native, convert, outputs = self.fixture()
        actual = ns["chain_gemm_einsums"]([Matrix("a"), Matrix("b")])
        self.assertIs(actual, outputs[0])
        convert.assert_not_called()
        self.assertEqual(native.call_args.args[3], [True])

    def test_native_tensor_output_converts_once(self):
        ns, native, convert, _ = self.fixture()
        actual = ns["chain_gemm_einsums"]([Matrix("a"), Matrix("b")], out="tensor")
        self.assertEqual(actual, ("einsums", "product"))
        self.assertEqual(convert.call_count, 1)

    def test_requested_intermediates_keep_output_contract(self):
        ns, native, convert, outputs = self.fixture()
        outputs.append(Matrix("intermediate"))
        actual = ns["chain_gemm_einsums"]([Matrix("a"), Matrix("b"), Matrix("c")],
                                        return_tensors=[True, True], out="tensor")
        self.assertEqual(actual, [("einsums", "product"), ("einsums", "intermediate")])
        self.assertEqual(convert.call_count, 2)
        self.assertEqual(native.call_args.args[3], [True, True])

    def test_einsums_values_are_not_sent_to_sharedmatrix_binding(self):
        ns, native, _, _ = self.fixture()
        self.assertFalse(ns["_use_cuest_gemm"]([Matrix("a"), object()], ["N", "N"], [0.]))
        native.assert_not_called()

    def test_native_failure_propagates_without_einsums_retry(self):
        ns, native, convert, _ = self.fixture()
        native.side_effect = RuntimeError("native failed")
        with self.assertRaisesRegex(RuntimeError, "native failed"):
            ns["chain_gemm_einsums"]([Matrix("a"), Matrix("b")])
        convert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
