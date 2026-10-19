"""Host/fake workspace probe checks. No native JK, device, SCF or submission."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import fixed_jk as f
import fixed_jk_campaign as c
import fixed_jk_handoff as h
import handoff
import policy as p
import workspace_control as w

HERE = Path(__file__).resolve().parent


class WorkspaceChecks(unittest.TestCase):
    def test_exact_single_control(self):
        self.assertEqual(w.check_env(w.CONTROL, True), w.CONTROL)
        for bad in ({}, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"},
                    dict(w.CONTROL, CUDA_LAUNCH_BLOCKING="1"),
                    dict(w.CONTROL, CUBLAS_ATOMICS_MODE="0"),
                    dict(w.CONTROL, CUSOLVER_DETERMINISTIC="1"),
                    dict(w.CONTROL, NVIDIA_TF32_OVERRIDE="0")):
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                w.check_env(bad, True)

    def test_unset_parent_and_no_silent_removal(self):
        for env in (w.CONTROL, {"CUBLAS_WORKSPACE_CONFIG": ""},
                    {"CUDNN_DETERMINISTIC": "1"}, {"CUDA_LAUNCH_BLOCKING": "0"}):
            before = dict(env)
            with self.assertRaises(RuntimeError):
                w.child_env(env)
            self.assertEqual(env, before)

    def test_controller_copy_unchanged(self):
        parent = {"CUDA_VISIBLE_DEVICES":"0", "OMP_NUM_THREADS":"8"}
        before = dict(parent)
        child = w.child_env(parent)
        self.assertEqual(parent, before)
        self.assertEqual(child, dict(parent, **w.CONTROL))
        self.assertEqual(w.check_env(parent), {})

    def engine(self):
        engine = object.__new__(c.Engine)
        engine.env = {"CUDA_VISIBLE_DEVICES": "0"}
        engine.manifest_path, engine.manifest_hash = "manifest", "a"*64
        engine.arm_command = lambda arm, args:["bash", str(HERE/"arm.sh"), arm, "runtime", *args]
        return engine

    def test_six_actual_children_both_arms(self):
        engine = self.engine()
        calls = []
        engine.run_command = lambda argv,name,**kw:calls.append((argv,name,kw))
        for row in f.sequence():
            engine.fixed_child(row["arm"], "jk", row["label"], (row["label"], "GPU"))
        self.assertEqual(len(calls), 6)
        for argv,name,kw in calls:
            self.assertEqual(Path(argv[1]).name, "child-arm.sh")
            self.assertEqual(w.overrides(kw["env"]), w.CONTROL)
        self.assertEqual(w.overrides(engine.env), {})

    def test_reuse_and_probe_paths_uncontrolled(self):
        engine = self.engine()
        with patch.object(engine, "run_command") as run:
            engine.fixed_child("baseline", "reuse-factor", "factor-reuse")
        self.assertIsNone(run.call_args.kwargs["env"])
        self.assertEqual(Path(run.call_args.args[0][1]).name, "arm.sh")
        old = (HERE/"repeatability_campaign.py").read_text()
        self.assertIn('k.startswith(("CUBLAS_",', old)
        # No changes to MAIN prolog, identity, probe-only or arm.sh.
        for name in ("repeatability_campaign.py", "arm.sh", "probe.py", "campaign.py", "handoff.py"):
            prior = subprocess.check_output(["git","show",w.PARENT+":devtools/major2/"+name],cwd=HERE)
            self.assertEqual((HERE/name).read_bytes(), prior)

    def test_parent_override_stops_before_popen(self):
        engine = self.engine()
        engine.env.update(w.CONTROL)
        with patch.object(engine, "run_command") as run, self.assertRaises(RuntimeError):
            engine.fixed_child("baseline", "jk", f.sequence()[0]["label"])
        run.assert_not_called()

    def test_only_jk_label_can_be_controlled(self):
        engine = self.engine()
        with self.assertRaises(RuntimeError):
            engine.fixed_child("baseline", "jk", "identity")
        with self.assertRaises(RuntimeError):
            engine.fixed_child("baseline", "factor", "factor")

    def test_preimport_gate_executes_before_numpy_and_any_native(self):
        # Fresh interpreter: fail immediately at first numpy import, after inspecting
        # durable receipt. No numpy/Psi4/native library can load in this subprocess.
        with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as tmp:
            root = Path(tmp)
            (root/"raw").mkdir()
            m = dict(run_root=tmp, numerical_control=w.POLICY, sequence=f.sequence())
            manifest = root/"manifest.json"
            manifest.write_text(json.dumps(m))
            sha = p.sha(manifest)
            script = """
import builtins,json,os,runpy,sys
from pathlib import Path
import workspace_control as w
w.RUN=sys.argv[1]
manifest,sha,label=sys.argv[2:]
sys.argv=['fixed_jk_runner.py','jk',manifest,sha,label,'GPU-test']
original=builtins.__import__
def gate(name,*args,**kw):
    if name == 'numpy':
        v=json.loads((Path(w.RUN)/'raw'/(label+'.effective-env.json')).read_bytes())
        assert v['observed']==w.CONTROL and v['before_numpy_psi4_import']
        print('RECEIPT_BEFORE_FIRST_NUMPY_NATIVE_IMPORT')
        raise SystemExit(0)
    return original(name,*args,**kw)
builtins.__import__=gate
runpy.run_path('fixed_jk_runner.py',run_name='__main__')
raise AssertionError('runner failed to reach gate')
"""
            env = w.child_env({k:v for k,v in os.environ.items()})
            env["SLURM_JOB_ID"] = "999"
            result = subprocess.run([sys.executable,"-B","-c",script,tmp,str(manifest),sha,
                f.sequence()[0]["label"]],cwd=HERE,env=env,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn("RECEIPT_BEFORE_FIRST_NUMPY_NATIVE_IMPORT",result.stdout)
            file = root/"raw"/(f.sequence()[0]["label"]+".effective-env.json")
            self.assertFalse(file.stat().st_mode & 0o222)

    def test_wrong_env_rejected_by_actual_entry_before_numpy(self):
        script = """
import builtins,runpy,sys
sys.argv=['fixed_jk_runner.py','jk','missing','bad','fixed-baseline-r1','GPU']
orig=builtins.__import__
def reject(name,*a,**kw):
    if name=='numpy': raise AssertionError('native/numpy reached before rejection')
    return orig(name,*a,**kw)
builtins.__import__=reject
runpy.run_path('fixed_jk_runner.py',run_name='__main__')
"""
        env = dict(os.environ)
        self.assertEqual(w.check_env(env), {})
        result = subprocess.run([sys.executable,"-B","-c",script],cwd=HERE,env=env,
                                capture_output=True,text=True,timeout=10)
        self.assertNotEqual(result.returncode,0)
        self.assertIn("unexpected/missing numerical environment",result.stderr)
        self.assertNotIn("native/numpy reached",result.stderr)

    def test_shell_gate_precedes_inherited_psiapi(self):
        text = (HERE/"child-arm.sh").read_text()
        self.assertLess(text.index("--child-shell-check"),text.index('eval "$("$stage/bin/psi4" --psiapi)"'))
        self.assertEqual(text.count("\ncheck_control\n"),3)
        self.assertNotIn("unset CUBLAS",text)
        self.assertNotIn("export CUBLAS",text)
        tree = ast.parse((HERE/"workspace_control.py").read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertEqual(imports, ["pathlib"])
        self.assertNotIn("numpy", [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names])

    def test_receipt_identity_and_each_corruption(self):
        row = f.sequence()[0]
        value = dict(label=row["label"],arm=row["arm"],pid=12,job="99",gpu_uuid="GPU",
                     observed=w.CONTROL,before_numpy_psi4_import=True,manifest_sha256="sha")
        w.verify_receipt(value,row,12,"99","GPU","sha")
        for key in value:
            bad = copy.deepcopy(value)
            bad[key] = None
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                w.verify_receipt(bad,row,12,"99","GPU","sha")

    def test_receipt_and_per_state_integrity_enforced_in_verifier(self):
        text = (HERE/"fixed_jk_campaign.py").read_text()
        for token in ('w.verify_receipt(', 'value["effective_env_sha256"] == p.sha(env_file)',
                      'v["numerical_environment"] == w.CONTROL',
                      'not env_file.stat().st_mode & 0o222'):
            self.assertIn(token,text)
        runner = (HERE/"fixed_jk_runner.py").read_text()
        self.assertIn('p.read_json(env_path) == EARLY_ENV',runner)
        self.assertIn('numerical_environment=effective_env',runner)

    def test_all_prior161_tests_byte_identical(self):
        for file in HERE.glob("test_*.py"):
            prior = subprocess.check_output(["git","show",w.PARENT+":devtools/major2/"+file.name],cwd=HERE)
            self.assertEqual(file.read_bytes(),prior,file.name)

    def test_completed_not_in_failed_accounting(self):
        self.assertEqual(len(h.FAILED_JOBS),12)
        self.assertNotIn("14175243",h.FAILED_JOBS)
        rows = [f"{j}{suffix}|FAILED|1:0" for j in h.FAILED_JOBS for suffix in ("",".batch")]
        completed = ["14175243|COMPLETED|0:0","14175243.batch|COMPLETED|0:0"]
        with patch.object(h.h,"output",return_value="\n".join(rows+completed)):
            h.accounting()
        for bad in ([], ["14175243|FAILED|1:0","14175243.batch|FAILED|1:0"]):
            with patch.object(h.h,"output",return_value="\n".join(rows+bad)),self.assertRaises(RuntimeError):
                h.accounting()

    def test_permanent_one_shot_numeric_id_before_queries(self):
        text = (HERE/"handoff.py").read_text()
        self.assertLess(text.index('durable(lock/"job-id.txt"'),text.index('info = output("scontrol"'))
        self.assertIn("lock.mkdir()",text)
        self.assertIn("os.fsync(directory)",text)
        self.assertIn('gates = audit(pin,*hashes)', (HERE/"fixed_jk_handoff.py").read_text())

    def manifest(self):
        return p.read_json(os.environ["MAJOR2_FIXED_MANIFEST"])

    def test_new_policy_negative_manifest(self):
        m = self.manifest()
        f.validate(m)
        for value in ({}, dict(w.POLICY, jk_children={}), dict(w.POLICY, controller=w.CONTROL),
                      dict(w.POLICY, jk_children={"CUBLAS_WORKSPACE_CONFIG":":16:8"})):
            bad = copy.deepcopy(m)
            bad["numerical_control"] = value
            with self.assertRaises(RuntimeError):
                f.validate(bad)

    def test_explicit_old_and_new_grouping_bounds_not_timings(self):
        old = self.manifest()["completed_control"]["report"]
        result = w.summary(old)
        self.assertEqual(len(result["groups"]),8)
        self.assertTrue(all(v["pairs"] == 3 for v in result["groups"].values()))
        self.assertEqual(result["groups"]["baseline/same-plan"]["matrices"]["J"]["maxabs"],0)
        self.assertGreater(result["groups"]["baseline/unchanged-initialize"]["scalar_maxabs_eh"],0)
        self.assertEqual(result["groups"]["candidate/unchanged-initialize"]["scalar_maxabs_eh"],0)
        f.no_timing(result)
        changed = copy.deepcopy(old)
        changed["differences"].pop()
        with self.assertRaises(RuntimeError):
            w.summary(changed)

    def test_only_narrow_owned_diff_scientific_unchanged(self):
        names = subprocess.check_output(["git","diff","--name-only",w.PARENT],cwd=HERE,text=True).splitlines()
        self.assertTrue(all(n.startswith("devtools/major2/") for n in names))
        for name in ("science.py","observe.py","execute_runner.py","policy.py","summary.py"):
            prior = subprocess.check_output(["git","show",w.PARENT+":devtools/major2/"+name],cwd=HERE)
            self.assertEqual((HERE/name).read_bytes(),prior)


if __name__ == "__main__":
    unittest.main()
