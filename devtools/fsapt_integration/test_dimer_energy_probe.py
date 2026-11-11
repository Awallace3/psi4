import unittest
from pathlib import Path
from unittest.mock import Mock

from dimer_energy_probe import energy_only, scf_options
from phoenix_dimer import resources, native_headers
from submit_dimer import command


class DimerEnergyContracts(unittest.TestCase):
    def test_only_backend_differs(self):
        cpu, gpu = scf_options("cpu"), scf_options("gpu")
        self.assertEqual({k for k in cpu if cpu[k] != gpu[k]}, {"use_cuest"})
        self.assertFalse(gpu["cuest_mixed_precision"])
        self.assertEqual((gpu["e_convergence"], gpu["d_convergence"], gpu["maxiter"]),
                         (10, 12, 100))
        self.assertFalse(any(k.startswith(("sapt", "fisapt")) for k in gpu))

    def test_one_public_hf_energy_on_full_geometry(self):
        psi4 = Mock()
        psi4.energy.return_value = (-123.0, object())
        result = energy_only(psi4, "full dimer geometry", "gpu")
        psi4.geometry.assert_called_once_with("full dimer geometry")
        psi4.energy.assert_called_once_with(
            "hf", molecule=psi4.geometry.return_value, return_wfn=True)
        self.assertEqual(result, psi4.energy.return_value)

    def test_failure_not_retried_or_threshold_relaxed(self):
        psi4 = Mock()
        psi4.energy.side_effect = RuntimeError("not converged")
        with self.assertRaisesRegex(RuntimeError, "not converged"):
            energy_only(psi4, "full dimer geometry", "gpu")
        self.assertEqual(psi4.energy.call_count, 1)
        psi4.set_options.assert_called_once_with(scf_options("gpu"))

    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError):
            scf_options("auto")

    def test_cpu_control_never_requests_gpu(self):
        argv = command(Path("/source"), Path("/run"), "cpu")
        self.assertIn("--qos=embers", argv)
        self.assertIn("--partition=cpu-gnr", argv)
        self.assertIn("--cpus-per-task=48", argv)
        self.assertFalse(any(arg.startswith("--gres") for arg in argv))
        self.assertEqual(resources("cpu")["wall_seconds"], 3600)

    def test_gpu_control_requests_one_h200_and_bounded_embers(self):
        argv = command(Path("/source"), Path("/run"), "gpu")
        self.assertIn("--qos=embers", argv)
        self.assertIn("--gres=gpu:h200:1", argv)
        self.assertIn("--constraint=H200", argv)
        self.assertIn("--signal=TERM@120", argv)
        self.assertEqual(resources("gpu")["wall_seconds"], 1800)

    def test_actual_native_header_padding(self):
        observed = native_headers(
            "cuESTJK: GPU-Accelerated Density-Fitted J/K Matrices\n"
            "    Math mode:             Native FP64\n")
        self.assertTrue(all(observed.values()))
        self.assertFalse(native_headers(
            "Math mode: Default (mixed precision permitted)")["native_fp64_header_observed"])


if __name__ == "__main__":
    unittest.main()
