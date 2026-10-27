"""Snapshot deferral contracts without importing/executing a native Psi4 core."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from fragment_gate import FIELDS


def nested_function(name, namespace):
    tree = ast.parse(Path(__file__).with_name("fragment_gate.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    node = next(n for n in ast.walk(main) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 "isolated-snapshot-contract", "exec"), namespace)
    return namespace[name]


class SnapshotContracts(unittest.TestCase):
    def test_retains_only_output_owners_no_DFHelper_JK_or_pre_timer_array_access(self):
        class Owner:
            @property
            def np(self):
                raise AssertionError("Snapshot array accessed before timer stop")
        owner = Owner()
        deferred = []
        namespace = {"args": SimpleNamespace(benchmark=True),
                     "deferred_cache": deferred, "FIELDS": FIELDS}
        capture = nested_function("capture", namespace)
        capture({"Elst_AB": owner, "Uocc_A": owner, "wfn_A": owner,
                 "Cvir_A": owner, "dfh": object(), "jk": object(), "large_unrelated": object()})
        self.assertEqual(set(deferred[0]), {"Elst_AB", "Uocc_A", "wfn_A", "Cvir_A"})
        self.assertIs(deferred[0]["Elst_AB"], owner)

    def test_raw_HF_snapshot_retains_exact_requested_owners_without_copy_or_IO(self):
        class Owner:
            @property
            def np(self):
                raise AssertionError("Raw HF copied before timer stop")
        owner = Owner()
        deferred = []
        np = Mock()
        namespace = {"args": SimpleNamespace(benchmark=True),
                     "deferred_raw": deferred, "np": np}
        capture = nested_function("raw_snapshot", namespace)
        capture({"S": owner, "J_A": owner, "unrelated": object()}, ("S", "J_A"))
        self.assertEqual(set(deferred[0]), {"S", "J_A"})
        self.assertIs(deferred[0]["S"], owner)
        np.savez.assert_not_called()
        np.array.assert_not_called()

    def test_CPU_pair_arrays_are_retained_not_copied_inside_benchmark_timer(self):
        disp, exch = object(), object()
        np = Mock()
        namespace = {"args": SimpleNamespace(benchmark=True), "np": np,
                     "store_components": lambda cache, *args, **kwargs: None}
        store = nested_function("components", namespace)
        cache = {}
        store(cache, disp, exch)
        self.assertIs(cache["Disp20 occupied pairs"], disp)
        self.assertIs(cache["Exch-Disp20 occupied pairs"], exch)
        np.array.assert_not_called()


if __name__ == "__main__":
    unittest.main()
