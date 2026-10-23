"""Offline checks; these are not native build, import or GPU evidence."""
import importlib.util
import hashlib
from pathlib import Path
import tempfile
import unittest
import json
from unittest.mock import patch
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("phoenix_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Contract(unittest.TestCase):
    def test_matched_public_fragment_science_differs_only_in_native_GPU_selection(self):
        gate=load("fragment_gate")
        for scaled in (False,True):
            cpu,gpu=gate.options(False,scaled),gate.options(True,scaled)
            self.assertEqual({**cpu,"use_cuest":True},gpu)
            self.assertEqual(cpu["sapt_dft_do_fsapt"],"SAPTDFT")
            self.assertTrue(cpu["fisapt_fsapt_ind_response"])
            self.assertEqual(cpu["sapt_dft_do_dhf"],scaled)
            self.assertEqual(cpu["df_basis_scf"],"aug-cc-pvdz-jkfit")
            self.assertEqual(cpu["df_basis_mp2"],"aug-cc-pvdz-ri")

    def test_header_stream_uses_POSIX_text_order_not_path_component_order(self):
        validate = load("validate")
        with tempfile.TemporaryDirectory() as directory:
            include = Path(directory)
            (include / "cuest").mkdir()
            (include / "cuest.h").write_bytes(b"root header")
            (include / "cuest/context.h").write_bytes(b"context header")
            files = list(include.rglob("*.h"))
            self.assertNotEqual([p.relative_to(include).as_posix() for p in sorted(files)],
                                sorted(p.relative_to(include).as_posix() for p in files))
            stream = b"".join(name.encode() + b"\0" + hashlib.sha256(data).hexdigest().encode() + b"\n"
                              for name, data in (("cuest.h", b"root header"),
                                                 ("cuest/context.h", b"context header")))
            self.assertEqual(validate.header_identity(include),
                             (2, hashlib.sha256(stream).hexdigest()))
            (include / "cuest/context.h").write_bytes(b"changed")
            self.assertNotEqual(validate.header_identity(include)[1],
                                hashlib.sha256(stream).hexdigest())

    def test_job_payload_is_present(self):
        self.assertTrue((HERE / "payload.sh").is_file())
        self.assertIn("phoenix_validation/payload.sh", (HERE / "job.sbatch").read_text())

    def test_explicit_free_cpu_shape_and_absolute_logs(self):
        submit = load("submit")
        argv = submit.command(Path("/source"), Path("/fresh-run"))
        for value in ("--qos=embers", "--account=gts-cs207-chemx", "--mem=48G",
                      "--cpus-per-task=8", "--no-requeue", "--time=03:00:00"):
            self.assertIn(value, argv)
        self.assertFalse(any(x.startswith(("--gres", "--gpus")) for x in argv))
        self.assertIn("--output=/fresh-run/logs/%x-%j.out", argv)

    def test_inherited_scheduler_overrides_are_removed(self):
        self.assertEqual(load("submit").clean_environment(
            {"SBATCH_QOS": "inferno", "SBATCH_GRES": "gpu:1", "PATH": "/bin"}),
            {"PATH": "/bin"})

    def test_inferno_is_an_explicit_same_resource_shape(self):
        submit = load("submit")
        argv = submit.command(Path("/source"), Path("/fresh-run"),
                              qos="inferno", account="gts-cs207")
        self.assertIn("--qos=inferno", argv)
        self.assertNotIn("--qos=embers", argv)
        self.assertIn("--account=gts-cs207", argv)
        self.assertIn("--time=03:00:00", argv)
        self.assertIn("--cpus-per-task=8", argv)
        self.assertFalse(any(x.startswith(("--gres", "--gpus")) for x in argv))

    def test_missing_sdk_headers_reject_before_submission(self):
        submit = load("submit")
        source = HERE.parents[2]
        with tempfile.TemporaryDirectory() as directory:
            sdk = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "cusparse.h"):
                submit.validate_sdk_shape(source, sdk)

    def test_dependency_world_must_match_exact_pin_and_prefix(self):
        validate = load("validate")
        world = {"git_commit": validate.E2_PIN, "version_major": 2,
                 "compiler_major": 14, "library_path": "/e2/lib/libEinsums.so"}
        self.assertEqual(validate.validate_world(world, Path("/e2")),
                         Path("/e2/lib/libEinsums.so"))
        for key, value in (("git_commit", "old"), ("version_major", 1),
                           ("compiler_major", 12), ("library_path", "/foreign/libEinsums.so")):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate.validate_world({**world, key: value}, Path("/e2"))

    def test_proof_files_cannot_overwrite_previous_attempts(self):
        validate = load("validate")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proof.json"
            validate.save(path, {"status": "PASS"})
            with self.assertRaises(FileExistsError):
                validate.save(path, {"status": "different"})

    def test_gpu_loader_locator_preserves_aliases_and_hashes_real_DSO_once(self):
        prepare = load("prepare_gpu_runtime")
        identity = prepare.identity_module(HERE.parents[2])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdk = root / "source-sdk"
            for name in ("bin", "include", "nvvm", "lib64"):
                (sdk / name).mkdir(parents=True)
            for stem, version in (("cudart", "13.3"), ("cublas", "13.6"),
                    ("cublasLt", "13.6"), ("cusolver", "12.2"), ("cusparse", "12.8"),
                    ("nvJitLink", "13.3"), ("nvrtc", "13.3"), ("nvrtc-builtins", "13.3")):
                real = sdk / "lib64" / ("lib" + stem + ".so." + version)
                real.write_bytes(("explicit fake ELF fixture " + stem).encode())
                (sdk / "lib64" / ("lib" + stem + ".so")).symlink_to(real)
            target = root / "runtime-sdk"
            manifest = prepare.loader_manifest(sdk, target, identity)
            self.assertEqual(len(manifest["components"]), 8)
            for record in manifest["components"]:
                self.assertTrue(Path(record["path"]).is_relative_to(target))
                self.assertEqual(prepare.digest(record["path"]), record["sha256"])
            self.assertEqual((target / "lib64/libcublas.so").resolve(),
                             (sdk / "lib64/libcublas.so.13.6").resolve())

    def test_gpu_runtime_refuses_changed_science_or_optional_dependency_files(self):
        runtime = load("gpu_runtime")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root / "stage"
            package = stage / "lib/psi4"
            package.mkdir(parents=True)
            (stage / "bin").mkdir()
            (stage / "bin/psi4").write_text("frozen launcher")
            (package / "science.py").write_text("frozen science")
            optional = root / "optional"
            optional.mkdir()
            (optional / "binding.py").write_text("frozen binding")
            lock = root / "wheel-lock.json"
            lock.write_text("[]")
            proof = root / "proof.json"
            proof.write_text(json.dumps({"status": "PASS", "pin": "pin", "stage": str(stage),
                "python_package": str(package),
                "dependency_python_sha256": {},
                "launcher_sha256": runtime.digest(stage / "bin/psi4"),
                "scientific_python_sha256": {"science.py": runtime.digest(package / "science.py")}}))
            config = root / "config.json"
            config.write_text(json.dumps({"schema": 1, "device_family": "H200",
                "source_pin": "pin", "stage": str(stage), "build_proof": str(proof),
                "build_proof_sha256": runtime.digest(proof), "wheel_lock": str(lock),
                "wheel_lock_sha256": runtime.digest(lock), "optional_python_path": str(optional),
                "optional_files": {"binding.py": runtime.digest(optional / "binding.py")}}))
            checksum = runtime.digest(config)
            runtime.read_config(config, checksum)
            (optional / "binding.py").write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "dependency files"):
                runtime.read_config(config, checksum)
            (optional / "binding.py").write_text("frozen binding")
            (package / "unexpected.py").write_text("overlay")
            with self.assertRaisesRegex(RuntimeError, "scientific Python"):
                runtime.read_config(config, checksum)
            with self.assertRaisesRegex(RuntimeError, "configuration"):
                runtime.read_config(config, "0"*64)

    def test_loaded_optional_modules_not_just_metadata_are_bound(self):
        runtime = load("gpu_runtime")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module = root / "binding.so"
            module.write_bytes(b"fake extension identity")
            inventory = {"binding.so": runtime.digest(module)}
            with patch.dict("sys.modules", {"cuda.bindings.runtime":
                            SimpleNamespace(__file__=str(module))}):
                runtime.verify_modules(("cuda",), root, inventory)
            with patch.dict("sys.modules", {"cuda.bindings.runtime":
                            SimpleNamespace(__file__="/foreign/binding.so")}):
                with self.assertRaisesRegex(RuntimeError, "loaded runtime module"):
                    runtime.verify_modules(("cuda",), root, inventory)
            with patch.dict("sys.modules", {"cuda":
                            SimpleNamespace(__path__=["/foreign/cuda"])}):
                with self.assertRaisesRegex(RuntimeError, "namespace path"):
                    runtime.verify_modules(("cuda",), root, inventory)

    def test_provider_records_cleanup_on_workload_failure_without_masking_primary(self):
        runtime = load("gpu_runtime")
        class Session:
            def __init__(self, **kwargs):
                self.cleanup_complete=False
                self.live_bytes=0;self.native_owners=[]
                self.peak_live_bytes=123;self.blas_calls={};self.kernel_launches={}
            def __enter__(self):return self
            def __exit__(self,*error):self.cleanup_complete=True
        class Owner:
            def __init__(self,*args,**kwargs):
                self.handle=self.plan=self.pid=None;self.closed=False
            def __enter__(self):return self
            def __exit__(self,*error):self.closed=True
            def report(self):return {"closed":self.closed}
        class Provider:
            def __init__(self,**kwargs):self.closed=False
            def report(self):return {"closed":self.closed}
        for failure in (False,True):
            obj=object.__new__(runtime.AuditedRuntime)
            obj.config={"device_memory_limit_bytes":1024,"native_persistent_bytes":0,
                "cuest_root":"/sdk","cuest_archive":"/archive","cuda_manifest_sha256":"frozen"}
            obj.blas={"path":"/blas","sha256":"frozen"};obj.manifest={}
            obj.cuda=SimpleNamespace(CUDASession=Session)
            obj.abi=SimpleNamespace(admit_factory_api=lambda **kwargs:None)
            obj.plans=SimpleNamespace(DiagnosticPlanOwner=Owner)
            obj.dfmo=SimpleNamespace(CuESTDFMOProvider=Provider,POLICY={})
            obj.report={"plans":[]};obj.device_identity=lambda session:{"mock":True}
            def work():
                with obj.provider(object(),object()) as provider:
                    if failure:raise ValueError("original workload error")
                    provider.closed=True
            if failure:
                with self.assertRaisesRegex(ValueError,"original workload error"):work()
            else:work()
            record=obj.report["plans"][0]
            self.assertTrue(record["owner_cleanup"]["closed"])
            self.assertTrue(record["session_cleanup"]["complete"])
            self.assertEqual(record["session_cleanup"]["live_bytes"],0)


if __name__ == "__main__":
    unittest.main()
