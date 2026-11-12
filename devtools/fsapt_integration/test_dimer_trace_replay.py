from unittest.mock import Mock
import ast
from pathlib import Path
from typing import Optional
from types import SimpleNamespace
import unittest

from dimer_trace_replay import build_plan


class TraceReplayContracts(unittest.TestCase):
    def test_real_python_factory_forwards_memory_overload(self):
        # Exercise the real wrapper, not a mock accepting arbitrary kwargs.
        source = Path(__file__).resolve().parents[2] / "psi4/driver/p4util/python_helpers.py"
        function = next(n for n in ast.parse(source.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == "_core_jk_build")
        function.decorator_list = []
        core = SimpleNamespace(BasisSet=object, JK=Mock(), get_memory=lambda: 96 * 1024**3,
                               set_global_option=Mock())
        core.JK.build_JK.return_value.name.return_value = "MemDFJK"
        ns = dict(core=core, Optional=Optional, optproc=SimpleNamespace(OptionsState=Mock()))
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), ns)
        core.JK.build = ns["_core_jk_build"]
        _, memory = build_plan(core, "primary", "auxiliary", "cpu")
        core.JK.build_JK.assert_called_once_with("primary", "auxiliary", False, memory)

    def test_cpu_memory_is_supplied_to_factory_not_only_afterwards(self):
        core = Mock()
        core.get_memory.return_value = 96 * 1024**3
        core.JK.build.return_value.name.return_value = "MemDFJK"
        jk, memory = build_plan(core, "primary", "auxiliary", "cpu")
        self.assertEqual(memory * 8, 72 * 1024**3)
        core.JK.build.assert_called_once_with("primary", "auxiliary", jk_type="DF", do_wK=False, memory=memory)
        jk.set_memory.assert_called_once_with(memory)

    def test_diskdf_substitution_is_rejected(self):
        core = Mock()
        core.get_memory.return_value = 96 * 1024**3
        core.JK.build.return_value.name.return_value = "DiskDFJK"
        with self.assertRaisesRegex(RuntimeError, "Expected MemDFJK"):
            build_plan(core, "primary", "auxiliary", "cpu")
        core.JK.build.return_value.initialize.assert_not_called()

    def test_gpu_must_select_native_cuest(self):
        core = Mock()
        core.get_memory.return_value = 96 * 1024**3
        core.JK.build.return_value.name.return_value = "MemDFJK"
        with self.assertRaisesRegex(RuntimeError, "Expected cuESTJK"):
            build_plan(core, "primary", "auxiliary", "gpu")

    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError):
            build_plan(Mock(), None, None, "auto")


if __name__ == "__main__":
    unittest.main()
