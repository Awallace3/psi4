"""Focused host/fake tests only. Compiles input.h, never native.cc or GPU libraries."""
import copy
import json
import io
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import common as n
import wire
import compare
import handoff
import run


def tiny():
    s=dict(index=0,center=0,offset=0,pure=1,am=0,xyz=[-0.,1.,2.],
           exp=[1.,2.],coef=[0.25,-0.5])
    return [[-0.,1.,2.]], [dict(nbf=1,shells=[copy.deepcopy(s)]) for _ in range(2)]


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"])
        cls.base=Path(cls.tmp.name)
        cls.host=cls.base/"host-check"
        subprocess.run(["g++","-std=c++17","-O0","-Wall","-Wextra","-Werror",
                        str(n.HERE/"host_check.cc"),"-o",str(cls.host)],
                       check=True,capture_output=True,timeout=30)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def setUp(self):
        self.tmpcase=tempfile.TemporaryDirectory(dir=self.base)
        self.root=Path(self.tmpcase.name)

    def tearDown(self): self.tmpcase.cleanup()

    def parse(self,data,density=None,ok=True):
        p=self.root/("input-"+str(len(list(self.root.iterdir())))+".bin")
        p.write_bytes(data)
        args=[str(self.host),str(p)]
        if density is not None:
            q=p.with_suffix(".D"); q.write_bytes(density); args.append(str(q))
        v=subprocess.run(args,capture_output=True,text=True,timeout=5)
        self.assertEqual(v.returncode==0,ok,v.stderr)
        return v

    def test_native_roundtrip(self):
        data=wire.encode(*tiny())
        self.assertEqual(self.parse(data,struct.pack("<d",-0.)).stdout,"1 1 1\n")
        self.assertEqual(data[:8],b"NDFPLAN1")
        self.assertEqual(struct.unpack_from("<d",data,32)[0],-0.)
        self.assertEqual(data[32:40],struct.pack("<d",-0.))

    def test_native_truncation_boundaries(self):
        data=wire.encode(*tiny())
        for end in [0,7,8,16,24,32,55,len(data)-1]:
            self.parse(data[:end],ok=False)

    def test_native_decoded_float_bits_exact(self):
        xyz,b=tiny()
        b[0]["shells"][0]["coef"]=[np.nextafter(1.,2.),-0.]
        p=self.root/"basis.bin";p.write_bytes(wire.encode(xyz,b))
        values=list(xyz[0])
        for basis in b:
            for s in basis["shells"]: values+=s["xyz"]+s["exp"]+s["coef"]
        expected=[str(struct.unpack("<Q",struct.pack("<d",x))[0]) for x in values]
        self.assertEqual(subprocess.check_output([self.host,"--bits",p],text=True).splitlines(),expected)

    def test_native_file_cap_before_read(self):
        p=self.root/"oversize.bin"
        with p.open("wb") as f: f.truncate(128*1024**2+1)
        r=subprocess.run([self.host,p],capture_output=True,text=True,timeout=5)
        self.assertNotEqual(r.returncode,0)
        self.assertIn("input byte cap",r.stderr)

    def test_native_trailing(self): self.parse(wire.encode(*tiny())+b"x",ok=False)
    def test_native_magic(self): self.parse(b"BADPLAN1"+wire.encode(*tiny())[8:],ok=False)

    def test_native_natom_bounds(self):
        for x in (0,513,2**64-1):
            data=bytearray(wire.encode(*tiny()));struct.pack_into("<Q",data,8,x)
            self.parse(data,ok=False)

    def test_native_nbf_bounds(self):
        for x in (0,16385,2**64-1):
            data=bytearray(wire.encode(*tiny()));struct.pack_into("<Q",data,16,x)
            self.parse(data,ok=False)

    def test_native_nonfinite(self):
        for x in (float("nan"),float("inf"),-float("inf")):
            data=bytearray(wire.encode(*tiny()));struct.pack_into("<d",data,32,x)
            self.parse(data,ok=False)

    def test_native_shell_count_limits(self):
        for pos,val in ((56,32769),(64,0),(64,513),(64,2)):
            data=bytearray(wire.encode(*tiny()));struct.pack_into("<Q",data,pos,val)
            self.parse(data,ok=False)

    def test_native_shell_fields(self):
        for field,val in ((0,1),(1,1),(2,1),(3,0),(3,2),(4,11),(5,0),(5,257)):
            data=bytearray(wire.encode(*tiny()));struct.pack_into("<Q",data,72+8*field,val)
            self.parse(data,ok=False)

    def test_native_exponent_invalid(self):
        for val in (0.,-1.,float("nan")):
            data=bytearray(wire.encode(*tiny()));struct.pack_into("<d",data,144,val)
            self.parse(data,ok=False)

    def test_native_density_limits(self):
        data=wire.encode(*tiny())
        for d in (b"",struct.pack("<d",float("nan")),struct.pack("<dd",1.,1.)):
            self.parse(data,d,ok=False)

    def test_native_schedule_shared_template(self):
        result=subprocess.check_output([self.host,"--schedule"],text=True)
        self.assertEqual(result.splitlines(),["up+","A+","A1","A2","plan-","B+","B1",
            "B2","plan-","up-","up+","C+","C1","C2","plan-","up-"])

    def test_encoder_rejects_nonfinite(self):
        xyz,b=tiny();b[0]["shells"][0]["coef"][0]=float("nan")
        with self.assertRaises(ValueError): wire.encode(xyz,b)

    def test_encoder_rejects_order(self):
        xyz,b=tiny();b[0]["shells"][0]["offset"]=1
        with self.assertRaises(ValueError): wire.encode(xyz,b)

    def test_public_api_missing_no_coefficient_fallback(self):
        class Missing:
            def nshell_on_center(self,a): return 1
            def shell_on_center(self,a,q): return 0
            def shell(self,i): return object()
            def shell_to_center(self,i): return 0
        with self.assertRaises(AttributeError): wire.snapshot(Missing(),1)

    def test_numeric_environment_unset(self): self.assertEqual(n.check_env({"CUDA_VISIBLE_DEVICES":"0"}),{})

    def test_numeric_environment_rejection(self):
        for key in ["CUBLAS_WORKSPACE_CONFIG","CUBLASLT_FOO","CUEST_FOO","CUSPARSE_FOO",
                    "CUSOLVER_FOO","CUDA_UNKNOWN_CONTROL","NVIDIA_TF32_OVERRIDE",
                    "CUDA_LAUNCH_BLOCKING","MKL_CBWR","OMP_PLACES"]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                n.check_env({key:":4096:8"})

    def cohort(self):
        raw=self.root/"raw";raw.mkdir()
        d=raw/"D.bin";n.durable(d,np.eye(2,dtype="<f8").tobytes(),binary=True)
        for proc in range(1,4):
            p=raw/f"p{proc}";p.mkdir()
            for state in n.STATES:
                n.durable(p/(state+".bin"),np.eye(2,dtype="<f8").tobytes(),binary=True)
        return raw,d

    def mutate(self,path,value):
        path.chmod(0o644);path.write_bytes(np.asarray(value,dtype="<f8").tobytes());path.chmod(0o444)

    def test_comparison_counts_and_no_variation(self):
        raw,d=self.cohort();v=compare.analyze(raw,d,2)
        self.assertEqual(len(v["pairs"]),153)
        self.assertEqual(v["verdict"],"MINIMIZATION_FAILED_NO_VARIATION")
        self.assertFalse(v["clears_wrapper"])
        self.assertEqual(v["outputs"][0]["scalar_TrDJ"],2.)
        self.assertEqual(v["group_counts"],{"same-plan":9,"DF-only":12,"upstream-rebuild":24,"fresh-process":108})

    def test_same_plan_variation(self):
        raw,d=self.cohort();self.mutate(raw/"p1/A2.bin",[[1.01,0],[0,1.]])
        self.assertEqual(compare.analyze(raw,d,2)["verdict"],"SAME_PLAN_VARIATION")

    def test_df_only_variation(self):
        raw,d=self.cohort()
        for s in ("B1","B2"): self.mutate(raw/f"p1/{s}.bin",[[1.01,0],[0,1.]])
        self.assertEqual(compare.analyze(raw,d,2)["verdict"],"DF_BOUNDARY_VARIATION")

    def test_upstream_variation(self):
        raw,d=self.cohort()
        for s in ("C1","C2"): self.mutate(raw/f"p1/{s}.bin",[[1.01,0],[0,1.]])
        self.assertEqual(compare.analyze(raw,d,2)["verdict"],"UPSTREAM_BOUNDARY_VARIATION")

    def test_fresh_process_variation(self):
        raw,d=self.cohort()
        for s in n.STATES: self.mutate(raw/f"p2/{s}.bin",[[1.01,0],[0,1.]])
        self.assertEqual(compare.analyze(raw,d,2)["verdict"],"FRESH_PROCESS_ONLY_VARIATION")

    def test_nan_fatal(self):
        raw,d=self.cohort();self.mutate(raw/"p1/A1.bin",[[float("nan"),0],[0,1]])
        with self.assertRaises(RuntimeError): compare.analyze(raw,d,2)

    def test_wrong_count_fatal(self):
        raw,d=self.cohort();(raw/"p1/A1.bin").unlink()
        with self.assertRaises(RuntimeError): compare.analyze(raw,d,2)

    def test_writable_output_fatal(self):
        raw,d=self.cohort();(raw/"p1/A1.bin").chmod(0o644)
        with self.assertRaises(RuntimeError): compare.analyze(raw,d,2)

    def test_cap_fatal(self):
        raw,d=self.cohort()
        with patch.object(n,"CAP",1),self.assertRaises(RuntimeError): compare.analyze(raw,d,2)

    def test_native_only_link_command(self):
        args=run.link_command(self.root/"exe")
        self.assertEqual(args[2],n.CXX)
        self.assertNotIn(n.STAGE," ".join(map(str,args)))
        self.assertFalse(any("libpsi4" in str(a) or "libpython" in str(a) for a in args))
        self.assertIn(n.SDK+"/lib/13/libcuest.so",args)

    def test_forbidden_native_mapping(self):
        for path in ("/x/libpsi4.so","/x/libpython3.so","/x/psi4/core.so","/x/libEinsums.so"):
            with self.assertRaises(RuntimeError): run.native_maps("a b c "+path,{})

    def test_missing_native_mapping(self):
        with self.assertRaises(RuntimeError): run.native_maps("a b c /x/libcublas.so",{})

    def test_accept_id_strict(self):
        self.assertEqual(handoff.accept_id("12345;phoenix-slurm\n"),"12345")
        for x in ("","0","123\n124\n","Submitted batch job 123\n","123;cluster;extra"):
            with self.assertRaises(RuntimeError): handoff.accept_id(x)

    def test_permanent_lock(self):
        p=self.root/"attempt";handoff.permanent_lock(p)
        with self.assertRaises(FileExistsError): handoff.permanent_lock(p)

    def test_id_durable_before_query_and_no_retry(self):
        (self.root/"logs").mkdir()
        result=subprocess.CompletedProcess([],0,"12345\n","")
        def query(*args):
            self.assertEqual((self.root/"logs/job-id.txt").read_text(),"12345\n")
            self.assertFalse((self.root/"logs/job-id.txt").stat().st_mode&0o222)
            raise RuntimeError("controller query unavailable")
        with patch.object(n,"RUN",str(self.root)),patch.object(handoff,"audit",return_value={}),\
             patch.dict(os.environ,{"MAJOR2_NATIVE_PARENT_REVIEW":"h"}),\
             patch.object(subprocess,"run",return_value=result) as submit,\
             patch.object(n,"output",side_effect=query):
            with self.assertRaises(RuntimeError): handoff.parent_submit("p","h","b")
            self.assertEqual(submit.call_count,1)
            with self.assertRaises(FileExistsError): handoff.parent_submit("p","h","b")
            self.assertEqual(submit.call_count,1)

    def test_ambiguous_submission_keeps_attempt(self):
        (self.root/"logs").mkdir()
        with patch.object(n,"RUN",str(self.root)),patch.object(handoff,"audit",return_value={}),\
             patch.dict(os.environ,{"MAJOR2_NATIVE_PARENT_REVIEW":"h"}),\
             patch.object(subprocess,"run",return_value=subprocess.CompletedProcess([],0,"?","")):
            with self.assertRaises(RuntimeError): handoff.parent_submit("p","h","b")
            self.assertTrue((self.root/"logs/parent-attempt").is_dir())
            self.assertFalse((self.root/"logs/job-id.txt").exists())

    def test_parent_approval_required(self):
        with patch.dict(os.environ,{},clear=True),self.assertRaises(RuntimeError):
            handoff.parent_submit("p","h","b")

    def test_local_attempt_is_permanent_on_transport_ambiguity(self):
        bundle=self.root/"bundle";bundle.mkdir()
        mh="a"*64;bh="b"*64;pin="c"*40
        receipt=dict(pin=pin,manifest_sha256=mh,batch_sha256=bh)
        n.durable(bundle/"bundle-receipt.json",receipt)
        n.durable(bundle/"campaign.sbatch",handoff.batch_text(mh))
        m=dict(pin=pin)
        with patch.dict(os.environ,{"MAJOR2_NATIVE_PARENT_REVIEW":mh}),\
             patch.object(n,"manifest",return_value=m),patch.object(n,"sha",return_value=bh),\
             patch.object(n,"source_check"),patch.object(subprocess,"run",side_effect=subprocess.TimeoutExpired("phx",1)) as cmd:
            with self.assertRaises(subprocess.TimeoutExpired): handoff.parent_local(bundle,"/fake/phx")
            self.assertEqual(cmd.call_count,1)
            self.assertTrue((self.root/"parent-finalizer.attempt").is_dir())
            with self.assertRaises(FileExistsError): handoff.parent_local(bundle,"/fake/phx")
            self.assertEqual(cmd.call_count,1)

    def test_stage_generated_data_only(self):
        target=self.root/"new-run"
        payload={"manifest.json":"{}\n","campaign.sbatch":"#!/bin/bash\n",
                 "bundle-receipt.json":json.dumps(dict(pin="p",manifest_sha256="m",
                                                       batch_sha256="b",no_submission=True))}
        with patch.object(n,"RUN",str(target)),patch.object(sys,"stdin",io.StringIO(json.dumps(payload))),\
             patch.object(sys,"stdout",io.StringIO()) as out,patch.object(handoff,"verify") as verify:
            handoff.stage("p")
            verify.assert_called_once_with("p","m","b")
            self.assertTrue(json.loads(out.getvalue())["no_submission"])
            self.assertEqual(set(p.name for p in target.iterdir()),set(payload)|{"logs"})
            self.assertFalse((target/"campaign.sbatch").stat().st_mode&0o222)

    def test_audit_is_test_only_no_real_submission(self):
        result=subprocess.CompletedProcess([],0,"Partition=gpu-h200,gpu-v100","")
        with patch.object(handoff,"verify"),patch.object(handoff,"queue",return_value=[]),\
             patch.object(handoff,"accounting",return_value=[]),\
             patch.object(subprocess,"run",return_value=result) as cmd:
            v=handoff.audit("p","m","b")
            self.assertTrue(v["no_submission"])
            self.assertEqual(cmd.call_args.args[0],["sbatch","--test-only",n.RUN+"/campaign.sbatch"])

    def supervise(self,code):
        return subprocess.run(["bash",str(n.HERE/"supervise.sh"),str(self.root),sys.executable,"-c",code],
                              capture_output=True,text=True,timeout=10)

    def test_supervisor_failure(self):
        self.assertNotEqual(self.supervise("raise SystemExit(4)").returncode,0)
        self.assertFalse((self.root/"COMPLETED").exists())
        self.assertEqual(n.load(self.root/"batch-exit.json")["exit"],4)

    def test_success_requires_complete_artifacts(self):
        self.assertNotEqual(self.supervise("pass").returncode,0)
        self.assertFalse((self.root/"COMPLETED").exists())

    def test_supervisor_true_success(self):
        n.durable(self.root/"ENDPOINT-COMPLETE.json",dict(processes=3,outputs=18,pairs=153,
                 accepted_speedup_data=False,compiled_diagnostic_only=True))
        n.durable(self.root/"NATIVE-DIAGNOSTIC.json",{})
        self.assertEqual(self.supervise("pass").returncode,0)
        self.assertTrue((self.root/"COMPLETED").exists())
        self.assertFalse((self.root/"COMPLETED").stat().st_mode&0o222)
        self.assertNotEqual(self.supervise("pass").returncode,0)

    def test_supervisor_term(self):
        p=subprocess.Popen(["bash",str(n.HERE/"supervise.sh"),str(self.root),sys.executable,"-c",
                            "from pathlib import Path;import time;Path('ready').touch();time.sleep(20)"],
                           cwd=self.root,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        until=time.monotonic()+5
        while not (self.root/"ready").exists() and time.monotonic()<until: time.sleep(.01)
        self.assertTrue((self.root/"ready").exists())
        p.send_signal(signal.SIGTERM);p.communicate(timeout=10)
        self.assertNotEqual(p.returncode,0)
        self.assertFalse((self.root/"COMPLETED").exists())
        self.assertEqual(n.load(self.root/"batch-exit.json")["signalled"],1)

    def test_command_failure_no_fallback(self):
        r=object.__new__(run.Runner)
        r.signalled=False;r.end=time.time()+4000;r.scratch=self.root;r.raw=self.root
        r.env=dict(os.environ);r.active=None
        with self.assertRaises(RuntimeError): r.command([sys.executable,"-c","raise SystemExit(5)"],"fake",5)
        self.assertEqual(n.load(self.root/"fake.exit.json")["exit"],5)
        self.assertFalse((self.root/"COMPLETED").exists())

    def test_allocation_gate_before_import(self):
        with patch.dict(os.environ,{},clear=True),self.assertRaises(RuntimeError):
            n.allocation_only()

    def test_actual_runner_override_gate_before_native_import(self):
        env=dict(os.environ,SLURM_JOB_ID="999",CUBLAS_WORKSPACE_CONFIG=":4096:8")
        r=subprocess.run([sys.executable,"-B",str(n.HERE/"run.py"),"missing","wrong"],
                         env=env,capture_output=True,text=True,timeout=5)
        self.assertNotEqual(r.returncode,0)
        self.assertIn("inherited numerical override",r.stderr)
        self.assertNotIn("No such file",r.stderr)

    def test_manifest_bytes_gate(self):
        p=self.root/"m.json";n.durable(p,{})
        with self.assertRaises(RuntimeError): n.manifest(p,"0"*64)

    def test_manifest_policy_gate(self):
        p=self.root/"m.json"
        n.durable(p,dict(policy=dict(n.POLICY,outputs=17),source=n.SOURCE,run=n.RUN,
                        base_pin=n.BASE_PIN,base_stage=n.STAGE,pin="1"*40))
        with self.assertRaises(RuntimeError): n.manifest(p,n.sha(p))

    def test_wrong_native_lifecycle_receipt_fatal(self):
        r=object.__new__(run.Runner);r.raw=self.root;r.m={}
        p=self.root/"p1";p.mkdir()
        n.durable(p/"native-complete.json",dict(outputs=5))
        with patch.dict(os.environ,{"SLURM_JOB_ID":"999"}),self.assertRaises(RuntimeError):
            r.verify_native(p,123,"GPU-fake")


if __name__=="__main__": unittest.main()
