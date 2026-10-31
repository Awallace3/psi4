"""Offline H200 scheduling/bootstrap contracts, not device evidence."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import benchmark_suite
import sealed_bootstrap
import sealed_h200
from gpu_runtime import sealed_startup_path
from run_gate import environment


class SealedH200Contracts(unittest.TestCase):
    def test_failed_preflight_cleans_only_owned_scratch_and_records_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            root, scratch = base/"output", base/"scratch"
            root.mkdir()
            scratch.mkdir(mode=0o700)
            with patch.dict(os.environ, {"TMPDIR": str(scratch),
                    "DISP0_SPLIT_SCRATCH_CREATED": str(scratch), "DISP0_JOB_RUN": str(root)}):
                with self.assertRaises(FileNotFoundError):
                    sealed_h200.execute(base/"missing-manifest.json")
            self.assertFalse(scratch.exists())
            proof = json.loads((root/"pilot-result.json").read_text())
            self.assertEqual(proof["status"], "FAILED")
            self.assertEqual(proof["accepted_speedup_count"], 0)
            self.assertTrue(proof["scratch_cleanup"]["complete"])

    def test_only_explicit_embers_h200_afterany_with_fixed_caps(self):
        argv = sealed_h200.command(Path("/source"), Path("/fresh"), "123")
        for option in ("--qos=embers", "--account=gts-cs207-chemx",
                       "--constraint=H200", "--gres=gpu:h200:1",
                       "--cpus-per-task=8", "--mem=160G", "--time=08:00:00",
                       "--dependency=afterany:123", "--kill-on-invalid-dep=yes",
                       "--no-requeue"):
            self.assertIn(option, argv)
        self.assertFalse(any("inferno" in value or "afterok" in value for value in argv))
        with self.assertRaises(ValueError):
            sealed_h200.command(Path("/s"), Path("/r"), "not-a-job")

    def test_phoenix_bootstrap_rejects_other_contracts_before_import(self):
        for schema, device, contract in ((2, "RTX3090", sealed_h200.CONTRACT),
                                         (1, "H200", "fresh-bytes-v1")):
            with self.assertRaisesRegex(RuntimeError, "Explicit Phoenix"):
                sealed_bootstrap.bootstrap(
                    {"schema": schema, "device_family": device,
                     "identity_contract": contract}, {})

    def test_sealed_h200_has_private_home_no_toolkit_path_and_no_local_lease(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp).resolve()
            aliases = parent/"worker"
            aliases.mkdir(mode=0o700)
            config = {"schema": 1, "device_family": "H200",
                      "identity_contract": sealed_h200.CONTRACT,
                      "sealed_alias_parent": str(parent),
                      "startup_library_path": "/env/lib",
                      "dependency_environment": "/env",
                      "optional_python_path": "/optional", "einsums_prefix": "/e2",
                      "threads": 8}
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2",
                                         "LD_PRELOAD": "/foreign.so"}):
                env = environment(config, gpu=True, sealed_alias_directory=aliases)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(env["LD_LIBRARY_PATH"], "/env/lib")
            self.assertNotIn("LD_PRELOAD", env)
            self.assertEqual(env["CUDA_CACHE_DISABLE"], "1")
            self.assertEqual(Path(env["HOME"]).stat().st_mode & 0o777, 0o700)
            self.assertEqual(sealed_startup_path(config, aliases), "/env/lib")
            with self.assertRaisesRegex(RuntimeError, "fresh startup"):
                environment(config, gpu=True)
            config["startup_library_path"] += ":/sdk/lib"
            with self.assertRaisesRegex(RuntimeError, "qualified environment"):
                sealed_startup_path(config, aliases)

    def test_benchmark_uses_early_bootstrap_and_keeps_cache_and_oracle_guard(self):
        config = {"identity_contract": sealed_h200.CONTRACT,
                  "stage": "/stage", "dependency_environment": "/env"}
        manifest = {"kind": "gpu", "config": "/config", "config_sha256": "sha",
                    "inputs": "/input", "inputs_sha256": "input-sha"}
        argv = benchmark_suite.worker_command(Path("/helper"), manifest,
                                              config, Path("/out"), system="peptide")
        self.assertIn("/helper/sealed_bootstrap.py", argv)
        self.assertIn("--benchmark", argv)
        self.assertEqual(argv[argv.index("--factor-cache-bytes")+1], str(64*1024**2))
        with self.assertRaises(ValueError):
            benchmark_suite.worker_command(Path("/h"), manifest, config,
                                            Path("/o"), oracle=Path("/gpu"))

    def test_configuration_pins_real_versioned_preload_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            sdk = root/"sdk/lib/13"
            sdk.mkdir(parents=True)
            (sdk/"libcuest.so.0.2.50").write_bytes(b"fixture")
            rows = []
            for name in sealed_h200.PRELOAD[:-1]:
                path = root/(name+".123")
                path.write_bytes(b"fixture")
                rows.append({"path": str(path), "sha256": "fixture"})
            base = {"schema": 1, "device_family": "H200", "source_pin": "pin",
                    "cuest_root": str(root/"sdk"), "dependency_environment": "/env"}
            build = {"status": "PASS", "pin": "pin"}
            with patch.object(sealed_h200, "require_cuda_manifest",
                              return_value={"components": rows}):
                config = sealed_h200.configuration(base, build, root)
            self.assertEqual(len(config["native_preload_paths"]), 9)
            self.assertEqual(config["startup_library_path"], "/env/lib")
            self.assertEqual(config["identity_contract"], sealed_h200.CONTRACT)
            self.assertEqual(config["device_memory_limit_bytes"], 120*1024**3)


if __name__ == "__main__":
    unittest.main()
