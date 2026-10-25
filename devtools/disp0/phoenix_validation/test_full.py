"""Offline full-job contracts, NOT GPU/large-system qualification."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import os
import subprocess
import sys
import signal
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from full_inputs import load_point
from run_full import require_release,validate_allocation,compare_energy,execute_owned
from submit_full import command,LIMITS
from prepare_inputs import prepare
from repeat_gauge import compare_repeat
from test_repeat_gauge import write_water_fixture,WATER_DIMENSIONS

HERE=Path(__file__).resolve().parent


class FullContracts(unittest.TestCase):
    def test_TERM_becomes_checkpointable_failure_after_own_child_cleanup(self):
        with tempfile.TemporaryFile("w+") as out,tempfile.TemporaryFile("w+") as err:
            pids=[]
            timer=threading.Timer(.15,lambda:os.kill(os.getpid(),signal.SIGTERM))
            def started(pid):
                pids.append(pid);timer.start()
            try:
                with self.assertRaises(InterruptedError):
                    execute_owned([sys.executable,"-c","import time; time.sleep(30)"],
                                  dict(os.environ),out,err,30,started)
            finally:timer.cancel();timer.join()
            with self.assertRaises(ProcessLookupError):os.kill(pids[0],0)

    def test_full_signed_energy_offsets_nonfinite_and_sign_errors_fail(self):
        self.assertEqual(compare_energy({"energy_Eh":-.1},{"energy_Eh":-.1})["difference_Eh"],0.)
        for value in (.1,-.10001,float("nan"),float("inf")):
            with self.subTest(value=value),self.assertRaises((AssertionError,ValueError)):
                compare_energy({"energy_Eh":-.1},{"energy_Eh":value})

    def test_bounded_full_child_is_reaped_after_timeout(self):
        with tempfile.TemporaryFile("w+") as out,tempfile.TemporaryFile("w+") as err:
            pids=[]
            with self.assertRaises(subprocess.TimeoutExpired):
                execute_owned([sys.executable,"-c","import time; time.sleep(30)"],
                              dict(os.environ),out,err,.1,pids.append)
            with self.assertRaises(ProcessLookupError):os.kill(pids[0],0)

    def test_oracle_retained_rank_can_be_less_than_AO_but_not_incomplete(self):
        from fragment_gate import replay_hf
        import numpy as np
        class Raw(dict):
            @property
            def files(self):return list(self)
        raw=Raw(S=np.eye(4),V_A=np.zeros((4,4)),V_B=np.zeros((4,4)))
        for side in "AB":
            co=np.eye(4)[:,:1];cv=np.eye(4)[:,1:3]
            raw.update({"Cocc_"+side:co,"Cvir_"+side:cv,
                "eps_occ_"+side:np.array([-1.]),"eps_vir_"+side:np.array([1.,2.]),
                "D_"+side:co@co.T,"P_"+side:cv@cv.T})
        cache={key:SimpleNamespace(np=value,shape=value.shape) for key,value in raw.items()}
        def stop(value):raise LookupError("validated orbital inputs before native conversion")
        core=SimpleNamespace(get_global_option=lambda _:False,
                             Matrix=SimpleNamespace(from_array=stop))
        with self.assertRaisesRegex(LookupError,"validated orbital"):
            replay_hf(core,cache,object(),raw)
        changed=Raw(raw)
        cv=np.eye(4)[:,2:4]  # same rank, different retained subspace
        changed["Cvir_B"]=cv;changed["P_B"]=cv@cv.T
        with self.assertRaisesRegex(AssertionError,"retained orbital subspace"):
            replay_hf(core,cache,object(),changed)
        changed=Raw(raw);changed["Cvir_B"]=raw["Cvir_B"][:,:1]
        changed["eps_vir_B"]=np.array([1.])
        with self.assertRaises(ValueError):replay_hf(core,cache,object(),changed)

    def test_native_lifetime_observation_does_not_reopen_a_closed_provider(self):
        from full_observe import observed_factory
        session=SimpleNamespace(synchronize_device=lambda:None,
            rt=lambda value:value,runtime=SimpleNamespace(cudaMemGetInfo=lambda:(100,200)),
            live_bytes=10,native_bytes=5)
        class Provider:
            def __init__(self):
                self.session=session;self.closed=False
                self.owner=SimpleNamespace(report=lambda:{"known_host_workspace_bytes":3})
            def owned_plan(self):
                if self.closed:raise RuntimeError("closed")
                return self.owner
            @contextmanager
            def sibling_provider(self,aux):
                sibling=Provider()
                try:yield sibling
                finally:sibling.closed=True
        provider=Provider()
        @contextmanager
        def factory(primary,aux):
            try:yield provider
            finally:provider.closed=True
        runtime=SimpleNamespace(provider=factory,
            report={"primitive_probe":{"device":{"uuid":"GPU-fixed"}}})
        with patch("full_observe.subprocess.check_output",return_value="GPU-fixed, 200, 100, 100\n"):
            with observed_factory(runtime)("primary","aux") as p:
                with p.sibling_provider("ri") as ri:self.assertIs(ri.session,session)
                p.closed=True  # public entry owns its inner provider context
        rows=runtime.report["full_device_observations"]
        self.assertEqual(len(rows),6)
        self.assertTrue(all(row["sampled_not_peak"] for row in rows))
        self.assertEqual(rows[4]["native_plan"]["known_host_workspace_bytes"],3)

    def test_effective_allocation_cannot_change_hardware_charge_or_limits(self):
        job={"Account":"gts-cs207-chemx","QOS":"inferno","Partition":"gpu-h200",
             "NumNodes":"1","NumCPUs":"8","CPUs/Task":"8","NumTasks":"1",
             "TimeLimit":"08:00:00","MinMemoryNode":"160G","AllocTRES":"cpu=8,gres/gpu:h200=1"}
        manifest={"account":"gts-cs207-chemx"}
        validate_allocation(job,manifest)
        for key,value in (("QOS","embers"),("Account","gts-cs207"),("Partition","gpu-v100"),
                          ("NumCPUs","16"),("MinMemoryNode","320G"),("TimeLimit","1-00:00:00"),
                          ("AllocTRES","cpu=8,gres/gpu:v100=1")):
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                validate_allocation({**job,key:value},manifest)

    def test_route_is_one_bounded_explicit_inferno_H200(self):
        args=command(Path("/source"),Path("/run"),"gts-cs207-chemx")
        for flag in ("--qos=inferno","--partition=gpu-h200","--gres=gpu:h200:1",
                     "--cpus-per-task=8","--mem=160G","--time=08:00:00","--no-requeue"):
            self.assertIn(flag,args)
        self.assertEqual(LIMITS["host_memory"],"96 GiB")
        self.assertLess(LIMITS["native_persistent_bytes"],LIMITS["device_memory_limit_bytes"])

    def test_frozen_full_inputs_bind_actual_independent_dimensions(self):
        module=Path("/home/awallace43/data/timing_test_suite/geometries/systems.py")
        if not module.exists():self.skipTest("Literal input suite not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"frozen"
            prepare(module,HERE.parents[2]/"psi4/share/psi4/basis",root)
            path=root/"inputs.json"
            checksum=hashlib.sha256(path.read_bytes()).hexdigest()
            for name,counts in (("peptide",(24,356,[20,20],[5,5])),
                                ("nanotube",(42,742,[8,78],[2,24])),
                                ("protein83",(83,1293,[78,68],[20,19]))):
                point,d=load_point(path,checksum,name)
                self.assertEqual((d["atoms"],d["ao"],d["occupied"],d["frozen"]),counts)
                self.assertEqual(point["geometry_sha256"],
                                 hashlib.sha256(point["geometry"].encode()).hexdigest())
            value=json.loads(path.read_text());value["systems"]["nanotube"]["fragments"][1]["occupied"]=79
            path.write_text(json.dumps(value))
            with self.assertRaises(RuntimeError):
                load_point(path,hashlib.sha256(path.read_bytes()).hexdigest(),"nanotube")
            with self.assertRaises(RuntimeError):load_point(path,checksum,"peptide")

    def test_small_release_must_be_complete_source_and_config_matched(self):
        config={"source_pin":"science","qualified_small_config_sha256":"config"}
        value={"status":"PASS_SMALL_H200_ONLY","gpu_small_case_correctness_verified":True,
               "source_pin":"science","runtime_config_sha256":"config",
               "cases":[{"returncode":0} for _ in range(10)]}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"gate-result.json"
            def check(v):
                p.write_text(json.dumps(v))
                return require_release(p,hashlib.sha256(p.read_bytes()).hexdigest(),config)
            check(value)
            for key,bad in (("status","FAILED"),("source_pin","other"),
                            ("runtime_config_sha256","changed"),("cases",value["cases"][:9])):
                with self.subTest(key=key),self.assertRaises(RuntimeError):
                    check({**value,key:bad})
            bad=copy.deepcopy(value);bad["cases"][3]["returncode"]=1
            with self.assertRaises(RuntimeError):check(bad)

    def test_full_repeat_dimensions_are_external_and_cpu_arm_is_not_GPU(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=write_water_fixture(tmp)
            for path in paths:
                p=path/"result.json";value=json.loads(p.read_text())
                value["arm"]="cpu";p.write_text(json.dumps(value))
            with self.assertRaises(RuntimeError):compare_repeat(*paths)
            compare_repeat(*paths,dimensions=WATER_DIMENSIONS,arm="cpu")
            wrong={**WATER_DIMENSIONS,"ao":83}
            with self.assertRaises(RuntimeError):compare_repeat(*paths,dimensions=wrong,arm="cpu")


if __name__=="__main__":unittest.main()
