"""Host tests only; device qualification is the allocated pilot's job."""
import unittest
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
from gpu_runtime import AuditedRuntime
from sapt0_hardware import PROFILES, validate_device
from submit_sapt0 import route_matches


class HardwareContracts(unittest.TestCase):
    def test_supported_device_pairs_and_budgets(self):
        for family, memory in (("H200", 143771), ("RTX_PRO_BLACKWELL", 97887)):
            p = PROFILES[family]
            validate_device(family, "NVIDIA " + p["name"], p["architecture"],
                            memory, p["budget_gib"] * 1024**3)

    def test_wrong_name_architecture_and_insufficient_memory_fail_closed(self):
        for name, arch, mem in (("NVIDIA H200", "compute_120", 97887),
                                ("NVIDIA RTX PRO 6000 Blackwell", "compute_90", 97887),
                                ("NVIDIA RTX PRO 6000 Blackwell", "compute_120", 40000)):
            with self.assertRaises(RuntimeError):
                validate_device("RTX_PRO_BLACKWELL", name, arch, mem, 80 * 1024**3)

    def test_unknown_family_is_not_a_fallback(self):
        with self.assertRaises(KeyError):
            validate_device("ANY_GPU", "H200", "compute_90", 143771, 1024)

    def test_site_default_partition_does_not_relax_typed_gpu_requirement(self):
        fields = dict(Account="gts-cs207", QOS="inferno", TimeLimit="02:00:00",
                      Partition="gpu-rtxpro-blackwell,gpu-v100",
                      TresPerNode="gres/gpu:rtx_pro_6000_blackwell:1")
        profile = PROFILES["RTX_PRO_BLACKWELL"]
        self.assertTrue(route_matches(fields, profile, "gts-cs207", "inferno", 2))
        for key, value in (("Partition", "gpu-v100"),
                           ("Partition", "gpu-rtxpro-blackwell,gpu-h100"),
                           ("TresPerNode", "gres/gpu:1"), ("TimeLimit", "08:00:00"),
                           ("Account", "gts-cs207-chemx"), ("QOS", "embers")):
            self.assertFalse(route_matches(dict(fields, **{key: value}), profile,
                                           "gts-cs207", "inferno", 2))

    def test_actual_context_uuid_is_checked_for_blackwell(self):
        runtime = object.__new__(AuditedRuntime)
        runtime.config = {"device_family": "RTX_PRO_BLACKWELL",
                          "device_memory_limit_bytes": 80 * 1024**3}
        session = SimpleNamespace(arch="compute_120", ptx_sha256="test-ptx")
        own = f"{os.getpid()}, GPU-test, 12\n"
        hardware = "GPU-test, NVIDIA RTX PRO 6000 Blackwell Server Edition, 580, 97887\n"
        with patch("gpu_runtime.subprocess.check_output", side_effect=[own, hardware]):
            receipt = runtime.device_identity(session)
        self.assertEqual(receipt["uuid"], "GPU-test")
        self.assertEqual(receipt["nvrtc_architecture"], "compute_120")
        with patch("gpu_runtime.subprocess.check_output", return_value=own + "99999999, GPU-test, 4\n"):
            with self.assertRaisesRegex(RuntimeError, "Foreign compute"):
                runtime.device_identity(session)

    def test_psiapi_activation_handles_unset_pythonpath_and_preserves_errors(self):
        # Same shell sequence as the worker, with a fixture launcher.
        activation = 'set -eo pipefail; activation=$("$1" --psiapi); eval "$activation"; set -u; test -n "$PYTHONPATH"'
        ok = subprocess.run(["bash", "-c", activation, "fixture", "/bin/echo"],
                            env={"PATH": "/usr/bin:/bin"}, capture_output=True)
        self.assertNotEqual(ok.returncode, 0)  # invalid activation does not continue
        fixture = ('set -eo pipefail; unset PYTHONPATH; '
                   'activation=$(printf \'export PYTHONPATH="/stage:$PYTHONPATH"\\n\'); '
                   'eval "$activation"; set -u; test "$PYTHONPATH" = /stage:')
        self.assertEqual(subprocess.run(["bash", "-c", fixture]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
