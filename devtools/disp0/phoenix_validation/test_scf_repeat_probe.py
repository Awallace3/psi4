import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scf_repeat_probe import (CALLS, SCOPE, digest, matrix_delta, require_reference,
                              sample_plan, native_device_identity)


class FixedDensityTests(unittest.TestCase):
    def fake_core(self, *, fail=False, name="cuESTJK"):
        state = SimpleNamespace(finalized=False, inputs=[], calls=0)
        class Matrix:
            def __init__(self, array):
                self.np = np.array(array, copy=True)
            @staticmethod
            def from_array(array):
                return Matrix(array)
        class JK:
            def name(self):
                return name
            def set_omega_alpha(self, value):
                self.alpha = value
            def set_omega_beta(self, value):
                self.beta = value
            def set_do_J(self, value):
                pass
            def set_do_K(self, value):
                pass
            def initialize(self):
                pass
            def print_header(self):
                pass
            def C_clear(self):
                self.coeff = None
            def C_left_add(self, value):
                self.coeff = value
                state.inputs.append(value.np.copy())
            def compute(self):
                state.calls += 1
                if fail and state.calls == 2:
                    raise RuntimeError("injected compute failure")
                self.d = self.coeff.np@self.coeff.np.T
                self.j = self.d*2+state.calls*1e-13
                self.k = self.d*.5
            def J(self):
                return [Matrix(self.j)]
            def K(self):
                return [Matrix(self.k)]
            def D(self):
                return [Matrix(self.d)]
            def finalize(self):
                state.finalized = True
        native = JK()
        return SimpleNamespace(Matrix=Matrix, JK=SimpleNamespace(build=lambda *a, **k:native)), state

    def test_all_calls_receive_identical_coefficients_and_own_output_copies(self):
        core, state = self.fake_core()
        coefficients = np.array([[1.], [0.]])
        records = sample_plan(core, None, None, coefficients)
        self.assertEqual(state.calls, CALLS)
        self.assertTrue(state.finalized)
        for value in state.inputs:
            np.testing.assert_array_equal(value, coefficients)
        self.assertEqual(len(records), CALLS)
        self.assertNotEqual(records[0][0][0,0], records[-1][0][0,0])
        self.assertGreater(matrix_delta(records[0], records[-1], records[0][3])["J_max_abs"], 0)
        self.assertIn("NOT_SAPT_TIMING_OR_QUALIFICATION", SCOPE)

    def test_native_cleanup_runs_on_compute_failure_and_cpu_fallback_rejects(self):
        core, state = self.fake_core(fail=True)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            sample_plan(core, None, None, np.array([[1.], [0.]]))
        self.assertTrue(state.finalized)
        core, state = self.fake_core(name="MemDFJK")
        with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
            sample_plan(core, None, None, np.array([[1.], [0.]]))
        self.assertEqual(state.calls, 0)

    def test_closed_shell_delta_uses_full_trace_not_symmetry_assumption(self):
        d = np.array([[1.,2.],[3.,4.]])
        dj, dk = np.array([[1.,5.],[7.,3.]]), np.array([[2.,1.],[0.,4.]])
        zeros = np.zeros((2,2))
        result = matrix_delta((zeros, zeros), (dj, dk), d)
        self.assertEqual(result["fixed_D_closed_shell_JK_energy_delta_Eh"],
                         2*np.trace(d@dj)-np.trace(d@dk))

    def test_reference_requires_bound_source_basis_input_and_native_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"result.json"
            document = {"status":"EXECUTED_REQUIRES_INDEPENDENT_COMPARISON",
                "arm":"gpu", "system":"protein83", "source_pin":"a"*40,
                "basis_profile":"def2-svp", "runtime_config_sha256":"config",
                "input_manifest_sha256":"inputs", "options":{"use_cuest":True,
                "cuest_mixed_precision":False, "sapt_dft_functional":"HF"}}
            manifest = {"reference_result":str(p), "science_pin":"a"*40,
                        "reference_config_sha256":"config", "inputs_sha256":"inputs"}
            def seal():
                p.write_text(json.dumps(document))
                manifest["reference_result_sha256"] = digest(p)
            seal()
            self.assertEqual(require_reference(manifest), document)
            for key, bad in (("source_pin", "b"*40), ("arm","cpu"),
                             ("basis_profile","aug-cc-pvdz"), ("input_manifest_sha256","other")):
                original = document[key]
                document[key] = bad
                seal()
                with self.assertRaises(RuntimeError):
                    require_reference(manifest)
                document[key] = original
            document["options"]["cuest_mixed_precision"] = True
            seal()
            with self.assertRaises(RuntimeError):
                require_reference(manifest)

    def test_actual_native_h200_context_required_without_foreign_compute(self):
        with patch("scf_repeat_probe.os.getpid", return_value=123), patch(
                "scf_repeat_probe.subprocess.check_output",
                side_effect=["123, GPU-a, 2048\n", "GPU-a, NVIDIA H200, 9.0, 590.0, 143771\n"]):
            self.assertEqual(native_device_identity()["uuid"], "GPU-a")
        for processes, hardware in (
            ("", ""),
            ("123, GPU-a, 2048\n456, GPU-a, 100\n", ""),
            ("123, GPU-a, 2048\n", "GPU-a, NVIDIA H100, 9.0, 590.0, 80000\n")):
            with patch("scf_repeat_probe.os.getpid", return_value=123), patch(
                    "scf_repeat_probe.subprocess.check_output",
                    side_effect=[processes, hardware]):
                with self.assertRaises(RuntimeError):
                    native_device_identity()


if __name__ == "__main__":
    unittest.main()
