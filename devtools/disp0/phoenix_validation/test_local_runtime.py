"""CPU-only lease/target checks; no GPU, NVML, cuEST or other-agent inspection."""
import fcntl
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch

import gpu_runtime as runtime
import owned_process


class LocalRuntime(unittest.TestCase):
    def lease(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name);lock=root/"gpu0.lock";lock.touch()
        fd=os.open(lock,os.O_RDWR);self.addCleanup(os.close,fd)
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return fd,{"lock_path":str(lock),"hold_path":str(root/"gpu0.HOLD"),
                   "expected_uuid":"GPU-d36fd194-bdd6-3d53-96c9-6a67e3225eac"}

    def test_existing_descriptor_and_absent_hold_are_required_without_acquisition(self):
        fd,policy=self.lease()
        with patch.dict(os.environ,DISP0_LOCAL_LEASE_FD=str(fd)):
            self.assertEqual(runtime.require_local_lease(policy),fd)
            Path(policy["hold_path"]).touch()
            with self.assertRaisesRegex(RuntimeError,"HOLD"):
                runtime.require_local_lease(policy)
            Path(policy["hold_path"]).unlink()
            fcntl.flock(fd,fcntl.LOCK_UN)
            with self.assertRaisesRegex(RuntimeError,"held common"):
                runtime.require_local_lease(policy)
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(RuntimeError,"inherited"):
                runtime.require_local_lease(policy)

    def test_replaced_lock_and_unrelated_descriptor_do_not_admit_a_lease(self):
        fd,policy=self.lease()
        Path(policy["lock_path"]).unlink();Path(policy["lock_path"]).touch()
        with patch.dict(os.environ,DISP0_LOCAL_LEASE_FD=str(fd)):
            with self.assertRaisesRegex(RuntimeError,"held common"):
                runtime.require_local_lease(policy)

    def test_local_actual_context_uuid_and_hardware_without_process_queries(self):
        fd,policy=self.lease()
        target=uuid.UUID(policy["expected_uuid"].removeprefix("GPU-")).bytes
        obj=runtime.AuditedRuntime.__new__(runtime.AuditedRuntime)
        obj.config={"device_family":"RTX3090","local_lease":policy}
        session=SimpleNamespace(arch="compute_86",ptx_sha256="ptx",
            rt=lambda result:result[1],
            driver=SimpleNamespace(cuCtxGetDevice=lambda:(0,73),
                cuDeviceGetUuid=lambda dev:(0,SimpleNamespace(bytes=target))))
        def hardware(argv,**kw):
            self.assertNotIn("--query-compute-apps", " ".join(argv))
            return policy["expected_uuid"]+", NVIDIA GeForce RTX 3090, 615.78.08, 24576\n"
        with patch.dict(os.environ,DISP0_LOCAL_LEASE_FD=str(fd)),patch.object(
                runtime.subprocess,"check_output",side_effect=hardware) as query:
            actual=obj.device_identity(session)
            self.assertEqual(actual["uuid"],policy["expected_uuid"])
            self.assertIn("cooperative",actual["isolation_policy"])
            self.assertEqual(query.call_count,1)
            session.arch="compute_90"
            with self.assertRaisesRegex(RuntimeError,"RTX3090"):obj.device_identity(session)
            session.arch="compute_86"
            session.driver.cuDeviceGetUuid=lambda dev:(0,SimpleNamespace(bytes=b"\0"*16))
            with self.assertRaisesRegex(RuntimeError,"leased local UUID"):obj.device_identity(session)

    def test_h200_foreign_compute_rejection_remains_independent_of_local_lease(self):
        obj=runtime.AuditedRuntime.__new__(runtime.AuditedRuntime)
        obj.config={"device_family":"H200"}
        with patch.object(runtime.subprocess,"check_output",
                          return_value=f"{os.getpid()}, GPU-test, 1\n999999, GPU-test, 1\n"):
            with self.assertRaisesRegex(RuntimeError,"Foreign compute"):
                obj.device_identity(SimpleNamespace())

    def test_optional_namespace_inventory_is_explicit_and_unchanged_h200_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"cuda").mkdir();(root/"cuda/x.py").write_text("pass")
            (root/"cuda_bindings-13.3.0.dist-info").mkdir()
            (root/"cuda_bindings-13.3.0.dist-info/METADATA").write_text("Version: 13.3.0")
            (root/"unrelated.py").write_text("not a CUDA optional dependency")
            config={"optional_python_path":str(root)}
            self.assertIn("unrelated.py",runtime.optional_inventory(config))
            config.update(schema=2,device_family="RTX3090",
                          optional_inventory_roots=["cuda","cuda_bindings-13.3.0.dist-info"])
            self.assertNotIn("unrelated.py",runtime.optional_inventory(config))
            config["optional_inventory_roots"]=["cuda/bindings"]
            with self.assertRaisesRegex(RuntimeError,"complete local"):
                runtime.optional_inventory(config)

    def test_inherited_parent_owned_flock_reaches_fresh_child_and_timeout_reaps_it(self):
        fd,policy=self.lease()
        code=("import os;from pathlib import Path;"
              "fd=int(os.environ['DISP0_LOCAL_LEASE_FD']);"
              "assert 'FLOCK  ADVISORY  WRITE' in Path(f'/proc/self/fdinfo/{fd}').read_text();"
              "print(os.getpid(),flush=True)")
        with patch.dict(os.environ,DISP0_LOCAL_LEASE_FD=str(fd)),tempfile.TemporaryFile(
                mode="w+") as out,tempfile.TemporaryFile(mode="w+") as err:
            self.assertEqual(owned_process.execute([sys.executable,"-c",code],
                dict(os.environ),out,err,10,fd,policy),0)
            self.assertEqual(runtime.require_local_lease(policy),fd)
            with self.assertRaises(TimeoutError):
                owned_process.execute([sys.executable,"-c",
                    "import time;print('owned',flush=True);time.sleep(30)"],
                    dict(os.environ),out,err,.1,fd,policy)
            self.assertEqual(runtime.require_local_lease(policy),fd)

    def test_hold_during_owned_child_unwinds_without_releasing_parent_lease(self):
        fd,policy=self.lease()
        code="from pathlib import Path;import time;Path(%r).touch();time.sleep(30)"%policy["hold_path"]
        with patch.dict(os.environ,DISP0_LOCAL_LEASE_FD=str(fd)),tempfile.TemporaryFile(
                mode="w+") as out,tempfile.TemporaryFile(mode="w+") as err:
            with self.assertRaisesRegex(RuntimeError,"HOLD"):
                owned_process.execute([sys.executable,"-c",code],dict(os.environ),
                                      out,err,10,fd,policy)
            Path(policy["hold_path"]).unlink()
            self.assertEqual(runtime.require_local_lease(policy),fd)

    def test_exited_leader_cannot_release_lease_with_surviving_grandchild(self):
        fd,policy=self.lease()
        code=("import os,time\np=os.fork()\n"
              "if p:\n print(p,flush=True)\n os._exit(0)\n"
              "time.sleep(30)\n")
        with patch.dict(os.environ,DISP0_LOCAL_LEASE_FD=str(fd)),tempfile.TemporaryFile(
                mode="w+") as out,tempfile.TemporaryFile(mode="w+") as err:
            with self.assertRaisesRegex(RuntimeError,"surviving descendants"):
                owned_process.execute([sys.executable,"-c",code],dict(os.environ),
                                      out,err,10,fd,policy)
            out.seek(0);pid=int(out.read().strip())
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)
            self.assertEqual(runtime.require_local_lease(policy),fd)

    def test_termination_signal_unwinds_our_child_instead_of_escaping_supervision(self):
        fd,policy=self.lease()
        code=("import os,signal,time\nprint(os.getpid(),flush=True)\n"
              "os.kill(os.getppid(),signal.SIGTERM)\ntime.sleep(30)\n")
        with patch.dict(os.environ,DISP0_LOCAL_LEASE_FD=str(fd)),tempfile.TemporaryFile(
                mode="w+") as out,tempfile.TemporaryFile(mode="w+") as err:
            with self.assertRaisesRegex(InterruptedError,"termination signal"):
                owned_process.execute([sys.executable,"-c",code],dict(os.environ),
                                      out,err,10,fd,policy)
            out.seek(0);pid=int(out.read().strip())
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)
            self.assertEqual(runtime.require_local_lease(policy),fd)

    def test_changed_cuda_manifest_is_rejected_before_any_native_load(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"cachy"))
        self.addCleanup(sys.path.pop,0)
        import bootstrap_gate
        fd,policy=self.lease()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"cuda.json";path.write_text(json.dumps({"components":[]}))
            config={"local_lease":policy,"startup_library_path":"exact",
                    "native_alias_directory":directory,"native_alias_links":{str(path):"not-used"},
                    "cuda_manifest":str(path),"cuda_manifest_sha256":"0"*64}
            # Test the early manifest admission directly; it is called by
            # read_config AND again by bootstrap before its first CDLL.
            with patch.object(bootstrap_gate.ctypes,"CDLL") as load:
                with self.assertRaisesRegex(RuntimeError,"before native loading"):
                    runtime.require_cuda_manifest(config)
                load.assert_not_called()

    def test_real_loader_rc_policy_precedes_callers_first_lazy_attribute(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"cachy"))
        self.addCleanup(sys.path.pop,0)
        from bootstrap_gate import ConfigureEinsums
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);package=root/"einsums";package.mkdir()
            (package/"__init__.py").write_text(
                "from types import SimpleNamespace\n"
                "rc=SimpleNamespace(gpu_disable=False)\n"
                "def __getattr__(name):\n"
                " if name=='RuntimeTensor':\n"
                "  assert rc.gpu_disable and rc.profile_disable\n"
                "  return 'actual-lazy-lookup'\n"
                " raise AttributeError(name)\n")
            observer=ConfigureEinsums({"threads":4})
            sys.path.insert(0,directory);sys.meta_path.insert(0,observer)
            try:
                with patch.dict(os.environ,PSI_SCRATCH=directory):
                    module=importlib.import_module("einsums")
                    self.assertEqual(module.RuntimeTensor,"actual-lazy-lookup")
                    self.assertTrue(observer.configured)
                    self.assertEqual(Path(module.__file__),package/"__init__.py")
            finally:
                sys.meta_path.remove(observer);sys.path.pop(0);sys.modules.pop("einsums",None)

    def test_local_environment_does_not_inherit_cpu_visibility_or_Phoenix_paths(self):
        import run_gate
        config={"schema":2,"startup_library_path":"/exact/aliases:/exact/cuda/lib",
                "optional_python_path":"/exact/site","einsums_prefix":"/exact/env",
                "local_lease":{"expected_uuid":"GPU-explicit"},"threads":4}
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES="",LD_PRELOAD="/foreign",
                        CUDA_HOME="/foreign",PYTHONHOME="/foreign"):
            gpu=run_gate.environment(config,gpu=True)
            self.assertEqual(gpu["CUDA_VISIBLE_DEVICES"],"GPU-explicit")
            self.assertEqual(gpu["LD_LIBRARY_PATH"],config["startup_library_path"])
            self.assertNotIn("CUDA_HOME",gpu);self.assertNotIn("LD_PRELOAD",gpu)
            self.assertNotIn("PYTHONHOME",gpu)
            self.assertEqual(run_gate.environment(config,gpu=False)["CUDA_VISIBLE_DEVICES"],"")

    def test_foreign_namespace_inside_site_and_escaping_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory,tempfile.TemporaryDirectory() as foreign:
            root=Path(directory);(root/"cuda").mkdir();(root/"other").mkdir()
            config={"schema":2,"device_family":"RTX3090","optional_python_path":directory,
                    "optional_inventory_roots":["cuda"]}
            module=SimpleNamespace(__path__=[str(root/"other")])
            with patch.dict(sys.modules,{"cuda.fake":module}):
                with self.assertRaisesRegex(RuntimeError,"namespace"):
                    runtime.verify_modules(("cuda",),root,{},("cuda",))
            path=Path(foreign)/"foreign.py";path.write_text("pass")
            (root/"cuda/escaped.py").symlink_to(path)
            with self.assertRaisesRegex(RuntimeError,"escaped"):
                runtime.optional_inventory(config)

    def test_qualified_dependencies_can_belong_to_whole_stage_or_private_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);env=root/"environment";stage=root/"whole-stage"
            files={};inventory={};modules={}
            for name in ("einsums","waggle","qcelemental","qcengine","optking"):
                package=(env if name in ("einsums","waggle") else stage)/name
                package.mkdir(parents=True);path=package/"__init__.py";path.write_text("pass")
                files[name]=str(path);inventory[str(path)]=runtime.digest(path)
                modules[name]=SimpleNamespace(__file__=str(path),__path__=[str(package)])
            datum=stage/"qcelemental/datum.py";datum.write_text("datum=True")
            inventory[str(datum)]=runtime.digest(datum)
            modules["qcelemental.datum"]=SimpleNamespace(__file__=str(datum))
            config={"dependency_environment":str(env),"einsums_prefix":str(env),"stage":str(stage)}
            build={"module_files":files,"dependency_python_sha256":inventory,"binary_sha256":{}}
            with patch.dict(sys.modules,modules),patch.object(
                    runtime.importlib,"import_module",side_effect=modules.__getitem__):
                runtime.verify_qualified_imports(config,build)
                datum.write_text("changed=True")
                with self.assertRaisesRegex(RuntimeError,"Foreign/changed"):
                    runtime.verify_qualified_imports(config,build)


if __name__=="__main__":unittest.main()
