"""Offline checks; these are not native build, import or GPU evidence."""
import importlib.util
import hashlib
from pathlib import Path
import tempfile
import unittest
import json
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("phoenix_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Contract(unittest.TestCase):
    def test_geometry_extraction_never_executes_module_or_selects_expensive_systems(self):
        prepare=load("prepare_inputs")
        values={name:"literal "+name for name in prepare.SYSTEMS}
        values["pla15_3pe2"]="off limits"
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"systems.py"
            source.write_text('raise RuntimeError("must never execute")\nSYSTEMS = '+repr(values))
            actual,checksum=prepare.literals(source)
            self.assertEqual(set(actual),set(prepare.SYSTEMS))
            self.assertNotIn("pla15_3pe2",actual)
            self.assertEqual(checksum,hashlib.sha256(source.read_bytes()).hexdigest())
            source.write_text("SYSTEMS = dict()")
            with self.assertRaises(ValueError):prepare.literals(source)

    def test_input_basis_counts_are_actual_spherical_SPDF_files(self):
        prepare=load("prepare_inputs")
        basis=HERE.parents[2]/"psi4/share/psi4/basis"
        for role,expected in (
            ("primary",{"H":9,"C":23,"N":23,"O":23}),
            ("jkfit",{"H":32,"C":86,"N":86,"O":86}),
            ("rifit",{"H":23,"C":72,"N":72,"O":72})):
            actual=prepare.spherical_counts(basis/(prepare.BASIS[role]+".gbs"))
            self.assertEqual(actual["spherical_functions"],expected)
        with self.assertRaisesRegex(ValueError,"SPDF"):
            prepare.spherical_counts(basis/"def2-universal-jkfit.gbs")

    def test_geometry_admission_rejects_ghosts_charges_nonfinite_and_changed_axes(self):
        prepare=load("prepare_inputs")
        text="0 1\nH 0 0 0\nH 0 0 1\n--\n0 1\nH 0 0 3\nH 0 0 4\nunits angstrom\nsymmetry c1\nno_reorient\nno_com\n"
        actual=prepare.geometry_metadata(text)
        self.assertEqual([v["occupied"] for v in actual],[1,1])
        self.assertEqual([v["frozen_occupied"] for v in actual],[0,0])
        for bad in (text.replace("no_com",""),text.replace("0 1","1 1",1),
                    text.replace("H 0 0 0","@H 0 0 0"),text.replace("H 0 0 1","H 0 0 nan")):
            with self.assertRaises(ValueError):prepare.geometry_metadata(bad)

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
            self.assertEqual(cpu["sapt_dft_induction_type"],"CPKS")

    def test_h200_gate_explicit_bounded_route_and_no_hidden_CPU_resource_change(self):
        submit=load("submit_gate")
        for qos in ("embers","inferno"):
            argv=submit.command(Path("/source"),Path("/run"),qos=qos,account="gts-cs207-chemx")
            for flag in ("--partition=gpu-h200","--gres=gpu:h200:1","--cpus-per-task=4",
                         "--mem=32G","--time=01:00:00","--qos="+qos,"--no-requeue"):
                self.assertIn(flag,argv)
            self.assertIn("--output=/run/logs/%x-%j.out",argv)

    def test_signed_gate_comparison_rejects_sign_changes_not_just_absolute_energies(self):
        gate=load("run_gate")
        sums={name:-.02 for name in (
            "Elst_AB","Exch_AB","IndAB_AB","IndBA_AB","Disp_AB",
            "Ind20u_AB","Ind20u_BA","ExchInd20u_AB","ExchInd20u_BA",
            "Ind20r_AB","Ind20r_BA","ExchInd20r_AB","ExchInd20r_BA")}
        document={"geometry_sha256":"fixed","scaled":True,"signed_matrix_sums":sums,
                  "output_file_sha256":{name:"fixed" for name in
                     ("geom.xyz","geom.psi4mol","ZA.dat","ZB.dat")},
                  "Disp20,u":-.003,"Exch-Disp20,u":.0004}
        self.assertEqual(len(gate.compare(document,document)),15)
        changed={**document,"signed_matrix_sums":{**sums,"Ind20r_BA":.02}}
        with self.assertRaises(AssertionError):gate.compare(document,changed)
        with self.assertRaises(RuntimeError):gate.compare(document,{**document,"scaled":False})

    def test_native_IAO_oracle_reconstructs_GPU_gauge_without_copying_populations(self):
        gate=load("fragment_gate")
        class Matrix:
            def __init__(self,array):self.np=np.array(array,copy=True);self.shape=self.np.shape
            @staticmethod
            def from_array(array):return Matrix(array)
        core=SimpleNamespace(Matrix=Matrix)
        minimal=SimpleNamespace(nbf=lambda:3,function_to_center=lambda i:i)
        wfn=SimpleNamespace(get_basisset=lambda name:minimal)
        cache={"S":Matrix(np.eye(3))}
        gpu={}
        for side in ("A","B"):
            co=np.eye(3)[:,:2];locc=co@np.diag([1.,-1.])
            cache["Cocc_"+side]=Matrix(co)
            cache["Lfocc0"+side]=Matrix(co[:,:1])
            gpu["Locc_"+side]=locc;gpu["Qocc0"+side]=locc**2
        gate.replay_localization(core,cache,wfn,gpu,[np.eye(3)]*3)
        np.testing.assert_allclose(cache["Qocc0A"].np,gpu["Qocc0A"])
        np.testing.assert_allclose(cache["Uocc_A"].np,np.diag([1.,-1.]))
        np.testing.assert_allclose(cache["Uaocc0A"].np,[[-1.]])
        bad={**gpu,"Qocc0B":gpu["Qocc0B"]+.01}
        with self.assertRaises(AssertionError):
            gate.replay_localization(core,cache,wfn,bad,[np.eye(3)]*3)
        with self.assertRaises(RuntimeError):
            gate.replay_localization(core,cache,wfn,gpu,[np.eye(3)])
        theta=.27
        mixed=np.eye(3)
        mixed[:2,:2]=[[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]]
        for side in ("A","B"):
            cache["Cocc_"+side]=Matrix(np.eye(3))
            cache["Lfocc0"+side]=Matrix(np.zeros((3,0)))
            gpu["Locc_"+side]=mixed
            gpu["Qocc0"+side]=mixed**2
        with self.assertRaisesRegex(RuntimeError,"stationarity"):
            gate.replay_localization(core,cache,wfn,gpu,[np.eye(3)]*3)

    def test_oracle_rejects_inconsistent_raw_projectors_before_native_JK(self):
        gate=load("fragment_gate")
        class Raw(dict):
            @property
            def files(self):return list(self)
        raw=Raw(S=np.eye(3),V_A=np.zeros((3,3)),V_B=np.zeros((3,3)))
        for side in ("A","B"):
            co=np.eye(3)[:,:1];cv=np.eye(3)[:,1:]
            raw.update({"Cocc_"+side:co,"Cvir_"+side:cv,
                "eps_occ_"+side:np.array([-1.]),"eps_vir_"+side:np.array([1.,2.]),
                "D_"+side:co@co.T,"P_"+side:cv@cv.T})
        cache={name:SimpleNamespace(np=raw[name]) for name in ("S","V_A","V_B")}
        cache.update({name:SimpleNamespace(np=raw[name],shape=raw[name].shape)
                      for name in ("Cocc_A","Cocc_B","Cvir_A","Cvir_B")})
        core=SimpleNamespace(get_global_option=lambda name:False)
        for field in ("D_A","P_B"):
            changed=Raw(raw);changed[field]=changed[field]+.01
            with self.assertRaises(AssertionError):
                gate.replay_hf(core,cache,object(),changed)

    def test_exact_gate_tightens_integral_inputs_without_relaxing_one_electron_oracle(self):
        gate=load("fragment_gate")
        for gpu in (False,True):
            for scaled in (False,True):
                self.assertEqual(gate.options(gpu,scaled)["ints_tolerance"],1e-16)
        class Raw(dict):
            @property
            def files(self):return list(self)
        raw=Raw(S=np.eye(2),V_A=np.zeros((2,2)),V_B=np.zeros((2,2)))
        cache={name:SimpleNamespace(np=value.copy()) for name,value in raw.items()}
        raw["V_A"][0,1]=2.360213447686937e-10
        core=SimpleNamespace(get_global_option=lambda name:False)
        with self.assertRaisesRegex(AssertionError,"one-electron V_A"):
            gate.replay_hf(core,cache,object(),raw)

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
                "binary_sha256": {},
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
            obj.config={"device_family":"H200","device_memory_limit_bytes":1024,"native_persistent_bytes":0,
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
