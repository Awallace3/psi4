"""Offline H200 scheduling/bootstrap contracts, not device evidence."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_suite
import sealed_bootstrap
import sealed_h200
from gpu_runtime import digest, retirement_channel, sealed_startup_path
from run_gate import environment


class SealedH200Contracts(unittest.TestCase):
    def test_completed_cpu_artifacts_need_no_dependency_on_a_purged_job_id(self):
        argv = sealed_h200.command(Path("/frozen-source"),Path("/fresh"),None)
        self.assertFalse(any(value.startswith("--dependency=") for value in argv))
        self.assertNotIn("--kill-on-invalid-dep=yes",argv)
        for value in ("--qos=embers","--constraint=H200","--gres=gpu:h200:1",
                      "--cpus-per-task=8","--mem=160G","--time=08:00:00"):
            self.assertIn(value,argv)

    def test_completed_prerequisite_rejects_failed_changed_or_wrong_pin_proofs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertIsNone(sealed_h200.completed_cpu_prerequisite(root,"pin"))
            status = root/"job-status.env"
            status.write_text("exit_code=1\nterm_received=0\n")
            with self.assertRaisesRegex(RuntimeError,"failed/interrupted"):
                sealed_h200.completed_cpu_prerequisite(root,"pin")
            status.write_text("exit_code=0\nterm_received=0\n")
            proof = root/"result.json"
            proof.write_text(json.dumps({"status":"PASS","scope":"CPU_BUILD_VALIDATION_ONLY","pin":"pin"}))
            runtime = root/"gpu-runtime"
            runtime.mkdir()
            config = {"schema":1,"device_family":"H200","source_pin":"pin",
                      "build_proof":str(proof),"build_proof_sha256":digest(proof)}
            path = runtime/"runtime-config.json"
            path.write_text(json.dumps(config))
            with patch.object(sealed_h200,"require_cpu_admission",return_value="admission") as admit:
                receipt = sealed_h200.completed_cpu_prerequisite(root,"pin")
                self.assertEqual(receipt["cpu_admission_sha256"],"admission")
                admit.assert_called_once_with(root,"pin")
                for changed in ({**config,"source_pin":"old"},
                                {**config,"build_proof_sha256":"changed"}):
                    path.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(RuntimeError,"exact whole CPU"):
                        sealed_h200.completed_cpu_prerequisite(root,"pin")

    def test_submission_recipe_cannot_advance_the_frozen_execution_source(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source,build = base/"frozen-source",base/"build"/"job-123"
            source.mkdir()
            build.mkdir(parents=True)
            inputs = base/"inputs.json"
            inputs.write_text("{}")
            root = base/"fresh-h200"
            args = SimpleNamespace(source=source,pin="execution-pin",build_job="123",
                                   build_root=build,inputs=inputs,inputs_sha256="input",root=root)
            responses = [SimpleNamespace(returncode=0,stdout="test-only\n",stderr=""),
                         SimpleNamespace(returncode=0,stdout="456\n",stderr="")]
            scheduler = Mock(side_effect=responses)
            with patch.object(sealed_h200,"source_pin",side_effect=["execution-pin","recipe-pin"]), \
                    patch.object(sealed_h200,"completed_cpu_prerequisite",return_value={"proof":"sealed"}), \
                    patch("full_inputs.load_point"), \
                    patch.object(sealed_h200.subprocess,"run",scheduler):
                sealed_h200.submit(args)
            manifest = json.loads((root/"manifest.json").read_text())
            self.assertEqual(manifest["helper_pin"],"execution-pin")
            self.assertEqual(manifest["science_pin"],"execution-pin")
            self.assertEqual(manifest["submission_recipe_pin"],"recipe-pin")
            self.assertIsNone(manifest["scheduler_parent"])
            argv = scheduler.call_args.args[0]
            env = scheduler.call_args.kwargs["env"]
            self.assertEqual(env["DISP0_SOURCE"],str(source))
            self.assertEqual(env["DISP0_PIN"],"execution-pin")
            self.assertEqual(argv[-1],str(source/"devtools/disp0/phoenix_validation/sealed_h200.sbatch"))
            self.assertFalse(any(value.startswith("--dependency=") for value in argv))

    def test_cpu_admission_never_creates_a_cuda_session_or_claims_gpu_evidence(self):
        from unittest.mock import Mock
        runtime = SimpleNamespace(
            identity_owner=SimpleNamespace(contract=sealed_h200.CONTRACT,
                                           verify_symbols=Mock(), report=Mock(return_value={})),
            report={"identity_contract": sealed_h200.CONTRACT},
            manifest={"components": [
                {"path": "/manifest/libcublas.so.13", "sha256": "blas"},
                {"path": "/manifest/libcusolver.so.12", "sha256": "solver"}]},
            cuda=SimpleNamespace(
                CUDASession=Mock(side_effect=AssertionError("GPU session")),
                library=Mock(side_effect=lambda path,sha256,owner:
                             (object(),"/canonical/"+Path(path).name))))
        with tempfile.TemporaryDirectory() as temp, \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "SLURM_JOB_GPUS": ""}), \
                patch.object(sealed_bootstrap, "AuditedRuntime", return_value=runtime):
            output = Path(temp)/"admission.json"
            sealed_bootstrap.cpu_admission({"source_pin": "pin"}, Path("/config"), "sha", output)
            proof = json.loads(output.read_text())
            self.assertEqual(proof["status"], "PASS")
            self.assertFalse(proof["cuda_session_created"])
            self.assertFalse(proof["gpu_correctness_verified"])
            self.assertEqual(len(proof["bindings"]), 2)
            self.assertEqual(runtime.identity_owner.verify_symbols.call_count, 2)
            runtime.cuda.CUDASession.assert_not_called()

    def test_cpu_admission_rejects_gpu_visibility_or_allocation_before_adapter(self):
        for visible, allocated in (("0", ""), ("", "0")):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visible,
                                        "SLURM_JOB_GPUS": allocated}), \
                    patch.object(sealed_bootstrap, "AuditedRuntime") as adapter:
                with self.assertRaisesRegex(RuntimeError, "CPU-only"):
                    sealed_bootstrap.cpu_admission({}, Path("/config"), "sha", Path("/out"))
                adapter.assert_not_called()

    def test_retirement_channel_is_private_nonblocking_and_removed_after_use(self):
        import socket
        env = {}
        with retirement_channel(env) as channel:
            self.assertIsNone(channel.reason())
            fd = int(env["DISP0_RETIREMENT_FD"])
            with socket.socket(fileno=os.dup(fd)) as sender:
                sender.send(b"bounded fixture reason", socket.MSG_DONTWAIT | socket.MSG_NOSIGNAL)
            self.assertEqual(channel.reason(), "bounded fixture reason")
            self.assertEqual(channel.pass_fds, (fd,))
        self.assertNotIn("DISP0_RETIREMENT_FD", env)

    def test_retirement_descriptor_survives_real_psiapi_bash_exec_chain(self):
        from run_full import execute_owned
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            launcher = root/"psiapi"
            launcher.write_text("#!/bin/sh\nprintf 'export FIXTURE_PSIAPI=1\\n'\n")
            launcher.chmod(0o700)
            env = dict(os.environ)
            groups = set()
            code = ("import os,socket; assert os.environ['FIXTURE_PSIAPI']=='1'; "
                    "s=socket.socket(fileno=int(os.environ['DISP0_RETIREMENT_FD'])); "
                    "s.send(b'bridge reason',socket.MSG_DONTWAIT|socket.MSG_NOSIGNAL); "
                    "os._exit(86)")
            argv = ["/bin/bash", "-c",
                    'set -euo pipefail; eval "$("$1" --psiapi)"; exec "$2" -B -c "$3"',
                    "fixture-bootstrap", str(launcher), sys.executable, code]
            with (root/"stdout").open("w") as out, (root/"stderr").open("w") as err, \
                    retirement_channel(env) as channel:
                result = execute_owned(argv,env,out,err,10,groups.add,
                                       reaped=groups.discard,pass_fds=channel.pass_fds)
                self.assertEqual(result,86)
                self.assertEqual(channel.reason(),"bridge reason")
            self.assertFalse(groups)
            self.assertEqual((root/"stderr").read_text(),"")

    def test_h200_requires_exact_successful_cpu_admission_and_reaped_children(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root/"cpu-sealed-admission"
            folder.mkdir()
            proof_path = folder/"runtime-admission.json"
            proof = {"status": "PASS", "source_pin": "pin", "cuda_session_created": False,
                     "gpu_correctness_verified": False}
            proof_path.write_text(json.dumps(proof))
            receipt = {"status": "PASS", "source_pin": "pin", "owned_groups_reaped": True,
                       "runtime_admission_sha256": digest(proof_path)}
            path = folder/"preflight-result.json"
            path.write_text(json.dumps(receipt))
            self.assertEqual(sealed_h200.require_cpu_admission(root, "pin"), digest(path))
            for changed in ({**receipt,"owned_groups_reaped": False},
                            {**receipt,"status": "FAILED"}, {**receipt,"source_pin": "old"}):
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(RuntimeError, "CPU runtime admission"):
                    sealed_h200.require_cpu_admission(root, "pin")
            path.write_text(json.dumps(receipt))
            proof_path.write_text(json.dumps({**proof,"cuda_session_created": True}))
            with self.assertRaisesRegex(RuntimeError, "CPU runtime admission"):
                sealed_h200.require_cpu_admission(root, "pin")

    def test_failed_preflight_cleans_only_owned_scratch_and_records_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            root, scratch = base/"output", base/"disp0-sealed-h200-123"
            root.mkdir()
            with patch.object(sealed_h200, "STEP_SCRATCH_PARENT", base), \
                    patch.dict(os.environ, {"TMPDIR": "/scheduler/step/tmp",
                    "DISP0_SPLIT_SCRATCH_CREATED": "/stale/batch/tmp",
                    "SLURM_JOB_ID": "123", "DISP0_JOB_RUN": str(root)}):
                with self.assertRaises(FileNotFoundError):
                    sealed_h200.execute(base/"missing-manifest.json")
            self.assertFalse(scratch.exists())
            proof = json.loads((root/"pilot-result.json").read_text())
            self.assertEqual(proof["status"], "FAILED")
            self.assertEqual(proof["accepted_speedup_count"], 0)
            self.assertTrue(proof["scratch_cleanup"]["complete"])

    def test_step_scratch_rebinds_srun_overrides_without_adopting_them(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            root = base/"output"
            root.mkdir()
            scheduler = base/"scheduler-tmp"
            scheduler.mkdir()
            with patch.object(sealed_h200, "STEP_SCRATCH_PARENT", base), \
                    patch.dict(os.environ, {"TMPDIR": str(scheduler),
                    "SCRATCH": "/stale/batch/tmp", "PSI_SCRATCH": "/stale/batch/tmp",
                    "DISP0_SPLIT_SCRATCH_CREATED": "/stale/batch/tmp", "SLURM_JOB_ID": "456"}):
                owner = sealed_h200.step_scratch(root)
                expected = base/"disp0-sealed-h200-456"
                for key in ("TMPDIR","SCRATCH","PSI_SCRATCH","DISP0_SPLIT_SCRATCH_CREATED"):
                    self.assertEqual(os.environ[key], str(expected))
                self.assertEqual(expected.stat().st_mode & 0o777, 0o700)
                with self.assertRaises(FileExistsError):
                    sealed_h200.step_scratch(root)
                self.assertTrue(owner.cleanup(set())["complete"])
            self.assertTrue(scheduler.is_dir())  # Never delete scheduler-owned scratch.

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
