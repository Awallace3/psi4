"""Host-only regressions of production ENV guards; no SDK or scientific imports."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import common as n
import handoff

OBSERVED_ROOT = ("/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/"
                 "gcc-12.3.0/cuda-12.6.1-fgbhrk26npivybjj65zswagcf4zv6p2n")
LOCATORS = dict(CUDA_ROOT=OBSERVED_ROOT, CUDA_HOME=n.CUDA, CUDA_PATH=n.CUDA)
CONTROLS = ("CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32",
            "CUDA_UNKNOWN_CONTROL", "CUDA_MATH", "CUDA_MATH_MODE",
            "CUDA_LAUNCH_BLOCKING", "CUDA_DEVICE_MAX_CONNECTIONS",
            "CUDNN_FOO", "CUSOLVER_FOO", "CUSPARSE_FOO", "CUEST_FOO",
            "OMP_PROC_BIND", "OMP_PLACES", "MKL_CBWR", "MKL_DEBUG_CPU_TYPE")


class EnvironmentRepairTests(unittest.TestCase):
    def test_observed_module_locator_accepted(self):
        self.assertEqual(n.check_env(LOCATORS), {})

    def test_locator_plus_each_control_still_fatal(self):
        for key in CONTROLS:
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, key):
                n.check_env(dict(LOCATORS, **{key: ":4096:8"}))

    def test_no_environment_mutation(self):
        for controls in ({}, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
                         {"CUDA_UNKNOWN_CONTROL": "1"}):
            with patch.dict(os.environ, dict(LOCATORS, **controls), clear=True):
                before = dict(os.environ)
                if controls:
                    with self.assertRaises(RuntimeError):
                        n.check_env(os.environ)
                else:
                    n.check_env(os.environ)
                self.assertEqual(dict(os.environ), before)

    def test_production_allocation_guard_accepts_locator_before_imports(self):
        with patch.dict(os.environ, dict(LOCATORS, SLURM_JOB_ID="999"), clear=True), \
             patch.object(n.os, "uname") as uname:
            uname.return_value.nodename = "allocated-host"
            n.allocation_only()

    def test_production_native_guard_agrees(self):
        # Compile ONLY the exact production guard extracted from native.cc.
        # No CUDA headers, SDK, replay code, link or device operation.
        source = (n.HERE/"native.cc").read_text()
        guard = source[source.index("static void env_check() {"):
                       source.index("\nstruct Workspace")]
        host = ('#include <string>\n#include <stdexcept>\n'
                'static void need(bool ok, const char* m) { if (!ok) throw std::runtime_error(m); }\n'
                + guard +
                '\nint main(){try{env_check();return 0;}catch(...){return 1;}}\n')
        with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as tmp:
            exe = Path(tmp)/"env-host-check"
            subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-x", "c++", "-", "-o", str(exe)], input=host, text=True,
                           capture_output=True, check=True, timeout=30)
            for controls in ({}, *({k: "1"} for k in CONTROLS)):
                env = dict(LOCATORS, **controls)
                before = dict(env)
                result = subprocess.run([exe], env=env, timeout=5)
                self.assertEqual(result.returncode, 1 if controls else 0, env)
                self.assertEqual(env, before)

    def test_export_records_locators_before_scientific_imports(self):
        source = (n.HERE/"export.py").read_text()
        self.assertLess(source.index('out/"export-toolchain-locators.json"'),
                        source.index("import probe"))
        self.assertIn('"CUDA_ROOT","CUDA_HOME","CUDA_PATH","PATH"', source)

    def test_fifteen_roots_accounting_requires_failed_v2_batch(self):
        rows = [f"{j}{suffix}|{state}|{code}|"
                for jobs, state, code in ((handoff.FAILED, "FAILED", "1:0"),
                                          (handoff.COMPLETED, "COMPLETED", "0:0"))
                for j in jobs for suffix in ("", ".batch")]
        self.assertEqual(len(handoff.FAILED), 13)
        self.assertEqual(len(handoff.COMPLETED), 2)
        with patch.object(n, "output", return_value="\n".join(rows)):
            self.assertEqual(len(handoff.accounting()), 30)
        for row in ("14178058.batch|FAILED|1:0|", "14178058|FAILED|1:0|"):
            with patch.object(n, "output", return_value="\n".join(x for x in rows if x != row)), \
                 self.assertRaisesRegex(RuntimeError, "historical accounting"):
                handoff.accounting()
        with patch.object(n, "output", return_value="\n".join(rows).replace(
                "14178058.batch|FAILED|1:0|", "14178058.batch|FAILED|0:0|")), \
             self.assertRaisesRegex(RuntimeError, "14178058.batch"):
            handoff.accounting()

    def test_module_proof_is_only_original_bootstrap(self):
        original = (n.HERE.parent/"arm.sh").read_text()
        proof = (n.HERE/"module_env_check.sh").read_text()
        for line in ("source /etc/profile.d/z00_lmod_pace.sh",
                     "module load gcc/12.3.0 cuda/12.6.1",
                     "unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH",
                     "unset PYTHONPATH PYTHONHOME",
                     "export CUDA_HOME=$P/software/toolchains/cuda13 CUDA_PATH=$P/software/toolchains/cuda13",
                     'export PATH="$E/bin:$CUDA_HOME/bin:$PATH"'):
            self.assertIn(line, original)
            self.assertIn(line, proof)
        for line in proof.splitlines():
            if not line.startswith("#"):
                for banned in ("--psiapi", "arm.sh", "nvcc", "import numpy", "import psi4"):
                    self.assertNotIn(banned, line)


if __name__ == "__main__":
    unittest.main()
