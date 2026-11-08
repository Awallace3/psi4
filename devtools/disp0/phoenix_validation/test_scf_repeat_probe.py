import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scf_repeat_probe import (CALLS, SCOPE, digest, matrix_delta, require_reference,
                              sample_plan, native_device_identity, cpu_comparison,
                              require_gpu_control, REPEATS, PLANS)


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
            def set_memory(self, value):
                state.memory_words = value
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
        return SimpleNamespace(Matrix=Matrix, JK=SimpleNamespace(build=lambda *a, **k:native),
                               get_memory=lambda:1024), state

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

    def test_independent_cpu_mode_cannot_use_gpu_and_gets_explicit_df_memory(self):
        core, state = self.fake_core(name="MemDFJK")
        rows = sample_plan(core, None, None, np.array([[1.], [0.]]), gpu=False)
        self.assertEqual(len(rows), CALLS)
        self.assertEqual(rows[0][4], "MemDFJK")
        self.assertEqual(state.memory_words, 64)
        self.assertTrue(state.finalized)
        core, _ = self.fake_core()
        with self.assertRaisesRegex(RuntimeError, "no GPU allowed"):
            sample_plan(core, None, None, np.array([[1.], [0.]]), gpu=False)

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

    def test_cpu_comparison_checks_all_gpu_plans_and_rejects_changed_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gpu, cpu = root/"gpu", root/"cpu"
            gpu.mkdir()
            (cpu/"r0").mkdir(parents=True)
            density = np.eye(2)
            j, k = np.eye(2)*2., np.eye(2)*.5
            np.savez(cpu/"r0/fixed-jk.npz", density=density, J_0_0=j, K_0_0=k)
            (gpu/"job-status.env").write_text("exit_code=0\nterm_received=0\n")
            report = {"status":"EXECUTED_FIXED_DENSITY_DIAGNOSTIC_ONLY", "scope":SCOPE,
                      "accepted_speedup_count":0, "scratch_cleanup":{"complete":True},
                      "processes":[]}
            documents = []
            for i in range(REPEATS):
                out = gpu/f"r{i}"
                out.mkdir()
                values = {"density":density}
                for p in range(PLANS):
                    values[f"J_{p}_0"], values[f"K_{p}_0"] = j+1e-10, k-1e-11
                np.savez(out/"fixed-jk.npz", **values)
                doc = {"status":"EXECUTED_FIXED_DENSITY_DIAGNOSTIC_ONLY", "science_pin":"a"*40,
                    "reference_result_sha256":"reference", "scf_executed":False,
                    "fisapt_executed":False, "device":{"name":"NVIDIA H200"},
                    "fixed_coefficients_sha256":"coefficients",
                    "matrices_sha256":digest(out/"fixed-jk.npz")}
                documents.append(doc)
                (out/"probe-result.json").write_text(json.dumps(doc))
                report["processes"].append({"returncode":0,"retirement_reason":None,
                    "owned_groups_reaped":True,"result_sha256":digest(out/"probe-result.json")})
            path = gpu/"diagnostic-result.json"
            path.write_text(json.dumps(report))
            manifest = {"gpu_control":str(gpu), "gpu_control_sha256":digest(path),
                        "reference_result_sha256":"reference", "science_pin":"a"*40}
            cpu_doc = {"fixed_coefficients_sha256":"coefficients"}
            result = cpu_comparison(cpu,manifest,cpu_doc)
            self.assertTrue(result["all_original_AO_JK_checks_passed"])
            self.assertEqual(len(result["comparisons"]),REPEATS*PLANS)
            self.assertEqual((result["rtol"],result["atol"]),(2e-7,2e-8))
            out = gpu/"r0"
            with np.load(out/"fixed-jk.npz") as arrays:
                values = dict(arrays)
            values["J_0_0"] = j+1e-3
            np.savez(out/"fixed-jk.npz",**values)
            with self.assertRaisesRegex(RuntimeError,"seal changed"):
                cpu_comparison(cpu,manifest,cpu_doc)
            # Even a fully sealed, completed calculation must fail the
            # original oracle check if its actual values disagree.
            documents[0]["matrices_sha256"] = digest(out/"fixed-jk.npz")
            (out/"probe-result.json").write_text(json.dumps(documents[0]))
            report["processes"][0]["result_sha256"] = digest(out/"probe-result.json")
            path.write_text(json.dumps(report))
            manifest["gpu_control_sha256"] = require_gpu_control(gpu,"reference")
            self.assertFalse(cpu_comparison(cpu,manifest,cpu_doc)["all_original_AO_JK_checks_passed"])


if __name__ == "__main__":
    unittest.main()
