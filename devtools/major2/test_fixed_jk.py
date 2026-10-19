"""Host/fake protocol controls only. Not actual cuEST endpoint correctness."""
import ast
import copy
import json
import os
from pathlib import Path
import signal
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import fixed_jk as f
import fixed_jk_campaign as c
import fixed_jk_handoff as h
import fixed_jk_runner as runner
import observe as o
import policy as p
import repeatability as r

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT.parent/"saptdft-cuest-major2-p157-repeatability-20261019"
BASE = ROOT.parent/"saptdft-cuest-major2-jkexception-baseline-20261018"
CAND = ROOT.parent/"saptdft-cuest-major2-nonc1-candidate-20261019"


class FakeJK:
    def __init__(self, arm):
        self.arm, self.creates, self.reuses, self.alive = arm, 0, 0, False
        self.text, self.events, self.coeff = "", [], None
    def name(self):
        return "cuESTJK"
    def C_clear(self):
        self.events.append("clear")
    def C_left_add(self, value):
        self.events.append("left")
        self.coeff = value
    def initialize(self):
        self.events.append("initialize")
        if self.arm == "candidate" and self.alive:
            self.reuses += 1
            self.text += f"cuESTJK plan reused #{self.creates}\n"
        else:
            self.creates += 1
            self.text += "Device memory, DF int plan:\n"
            if self.arm == "candidate":
                self.text += f"cuESTJK plan created #{self.creates}\n"
        self.alive = True
    def compute(self):
        assert self.alive
        self.events.append("compute")
    def finalize(self):
        self.events.append("finalize")
        self.alive = False


class TestFixedJK(unittest.TestCase):
    def test_six_balanced_fresh_labels(self):
        rows = f.sequence()
        self.assertEqual([(v["arm"],v["repeat"]) for v in rows],
            [("baseline",1),("candidate",1),("candidate",2),("baseline",2),("baseline",3),("candidate",3)])
        self.assertEqual(len({v["label"] for v in rows}),6)

    def test_baseline_dispatch_exact(self):
        self.check_dispatch("baseline",3,0)
    def test_candidate_dispatch_exact(self):
        self.check_dispatch("candidate",2,1)
    def check_dispatch(self,arm,creations,reuses):
        jk = FakeJK(arm)
        coeff = np.ones((3,2))
        frozen = coeff.tobytes()
        states = []
        def snap(state,before,after,calls):
            self.assertIs(jk.coeff,coeff)
            self.assertEqual(coeff.tobytes(),frozen)
            self.assertEqual(after,f.expected_trace(arm,state))
            states.append(state)
            return state
        records,calls = f.dispatch(jk,coeff,arm,lambda:jk.text,snap)
        self.assertEqual(records,list(f.STATES))
        self.assertEqual(states,list(f.STATES))
        self.assertEqual(calls,dict(initialize=3,compute=4,finalize=2,C_left_add=1,C_right_add=0))
        self.assertEqual(jk.events,["clear","left","initialize","compute","compute",
                                  "initialize","compute","finalize","initialize","compute","finalize"])
        self.assertEqual((jk.creates,jk.reuses),(creations,reuses))

    def test_all_six_total_24_computes(self):
        total = 0
        for row in f.sequence():
            jk = FakeJK(row["arm"])
            _,calls = f.dispatch(jk,np.ones((2,1)),row["arm"],lambda:jk.text,lambda *args:None)
            total += calls["compute"]
        self.assertEqual(total,24)

    def test_wrong_backend_fatal(self):
        jk = FakeJK("baseline")
        jk.name = lambda:"DirectJK"
        with self.assertRaisesRegex(RuntimeError,"backend"):
            f.dispatch(jk,None,"baseline",lambda:jk.text,lambda *args:None)
        self.assertNotIn("compute",jk.events)

    def test_trace_wrong_creation_fatal(self):
        jk = FakeJK("candidate")
        jk.alive = True
        with self.assertRaises(RuntimeError):
            f.dispatch(jk,None,"candidate",lambda:jk.text,lambda *args:None)
        self.assertEqual(jk.events[-1],"finalize")

    def test_error_native_cleanup(self):
        jk = FakeJK("baseline")
        def failed():
            raise RuntimeError("native SDK exception")
        jk.compute = failed
        with self.assertRaisesRegex(RuntimeError,"native SDK"):
            f.dispatch(jk,None,"baseline",lambda:jk.text,lambda *args:None)
        self.assertEqual(jk.events[-1],"finalize")

    def test_error_snapshot_cleanup(self):
        jk = FakeJK("candidate")
        def failed(*_):
            raise RuntimeError("corrupt input")
        with self.assertRaisesRegex(RuntimeError,"corrupt input"):
            f.dispatch(jk,None,"candidate",lambda:jk.text,failed)
        self.assertEqual(jk.events[-1],"finalize")

    def test_input_mutation_fatal_before_compute(self):
        jk = FakeJK("candidate")
        calls = []
        def guard():
            calls.append(1)
            if len(calls) > 1:
                raise RuntimeError("coefficient bytes mutated")
        with self.assertRaisesRegex(RuntimeError,"mutated"):
            f.dispatch(jk,None,"candidate",lambda:jk.text,lambda *args:None,guard)
        self.assertNotIn("compute",jk.events)
        self.assertEqual(jk.events[-1],"finalize")

    def test_term_cleanup(self):
        jk = FakeJK("candidate")
        jk.compute = lambda:runner.stop(signal.SIGTERM,None)
        with self.assertRaisesRegex(RuntimeError,"signal"):
            f.dispatch(jk,None,"candidate",lambda:jk.text,lambda *args:None)
        self.assertEqual(jk.events[-1],"finalize")

    def test_finalize_reinitialize_counts(self):
        for arm in r.ALLOWLIST:
            self.assertEqual(f.expected_trace(arm,"D")["created"]-f.expected_trace(arm,"C")["created"],1)

    def test_trace_fallback_fatal(self):
        for text in ("DirectJK", "DiskDFJK", "MemDFJK", "status code 2","CUDA error"):
            with self.subTest(text=text),self.assertRaises(RuntimeError):
                f.traces(text,"baseline")
    def test_trace_banner_disagreement(self):
        with self.assertRaises(RuntimeError):
            f.traces("cuESTJK plan created #1","candidate")
    def test_baseline_cannot_adopt_candidate(self):
        with self.assertRaises(RuntimeError):
            f.traces("cuESTJK plan reused #1","baseline")

    def test_factor_api_exact_once_and_inexact_reconstruction(self):
        calls = []
        da = np.diag([1.0,0.5,1e-8])
        da.flags.writeable = False
        cc = np.array([[1.0,0],[0,np.sqrt(0.5)],[0,0]])
        class Matrix:
            @staticmethod
            def from_array(value):
                calls.append(("from_array",value.tobytes()))
                class M:
                    def partial_cholesky_factorize(self,tol):
                        calls.append(("cholesky",tol))
                        return types.SimpleNamespace(np=cc)
                return M()
        with patch.object(f,"RANK",2):
            coeff,density,proof = f.factor_once(types.SimpleNamespace(Matrix=Matrix),da)
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[1],("cholesky",1e-7))
        self.assertFalse(coeff.flags.writeable)
        self.assertFalse(density.flags.writeable)
        self.assertGreater(proof["maxabs"],0)
        self.assertAlmostEqual(proof["trace_difference"],-1e-8)

    def test_wrong_factor_rank_stops_review(self):
        class Matrix:
            @staticmethod
            def from_array(_):
                return types.SimpleNamespace(partial_cholesky_factorize=lambda tol:types.SimpleNamespace(np=np.eye(2)))
        with self.assertRaisesRegex(RuntimeError,"actual SAD factor rank"):
            f.factor_once(types.SimpleNamespace(Matrix=Matrix),np.eye(2))

    def test_factor_nonfinite_fatal(self):
        class Matrix:
            @staticmethod
            def from_array(_):
                return types.SimpleNamespace(partial_cholesky_factorize=lambda tol:types.SimpleNamespace(np=np.full((2,1),np.nan)))
        with patch.object(f,"RANK",1),self.assertRaisesRegex(RuntimeError,"finite"):
            f.factor_once(types.SimpleNamespace(Matrix=Matrix),np.eye(2))

    def test_original_archive_pin_and_da_exact(self):
        path = ROOT.parent/"saptdft-cuest/.pi/major2-gpu-optimization-20261017/p157-repeatability/matrix-download/baseline/stage-0.npz"
        self.assertEqual(p.sha(path),f.INPUT_SHA)
        self.assertEqual(o.array_info(f.load_Da(path)),f.DA)

    def test_wrong_archive_fatal(self):
        with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as tmp:
            file = Path(tmp)/"wrong.npz"
            file.write_bytes(b"bad")
            with self.assertRaisesRegex(RuntimeError,"archive"):
                f.load_Da(file)

    def test_immutable_save_and_hash(self):
        with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as tmp:
            file = Path(tmp)/"C.npy"
            value = np.ones((3,2))
            proof = f.save_array(file,value)
            self.assertEqual(proof["file_sha256"],p.sha(file))
            self.assertEqual(o.array_info(np.load(file)),proof["array"])
            with self.assertRaises(FileExistsError):
                f.save_array(file,value)

    def test_wrong_factor_file_bytes_fatal(self):
        with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as tmp:
            file = Path(tmp)/"C.npy"
            file.write_bytes(b"bad")
            with self.assertRaisesRegex(RuntimeError,"factor file"):
                f.load_factor(tmp,{"C":{"file":"C.npy","file_sha256":"0"*64}})

    def test_storage_cap_actual_readback(self):
        with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as tmp:
            store = f.Store(Path(tmp)/"out",cap=2*1024**2)
            arrays = dict(J=np.eye(3),K=np.eye(3))
            with patch.object(f,"finite_array",lambda value,shape:value):
                value = store.save("A",arrays)
                self.assertEqual(store.used,value["file_bytes"])
                self.assertEqual(p.sha(store.root/"A.npz"),value["file_sha256"])
                with self.assertRaises(FileExistsError):
                    store.save("A",arrays)

    def test_storage_pre_reservation_fails_before_write(self):
        with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as tmp:
            store = f.Store(Path(tmp)/"out",cap=1)
            with patch.object(f,"finite_array",lambda value,shape:value),self.assertRaisesRegex(RuntimeError,"cap"):
                store.save("A",dict(J=np.eye(2),K=np.eye(2)))
            self.assertFalse(list(store.root.iterdir()))

    def test_total_cap_arithmetic(self):
        payload = 48*1786**2*8
        self.assertEqual(payload,1224881664)
        self.assertLess(payload+32*1024**2+24*1024**2,f.CAP)

    def test_wrong_array_shape_dtype_nonfinite(self):
        for value in (np.ones((2,3)),np.ones((2,2),dtype=np.float32),np.full((2,2),np.nan)):
            with self.assertRaises(RuntimeError):
                f.finite_array(value,(2,2))

    def test_scalar_rhf_bare_formula(self):
        d,j,k = np.diag([2.,3.]),np.diag([4.,5.]),np.diag([6.,7.])
        self.assertEqual(f.scalar(d,j,k),2*(8+15)-(12+21))
    def test_scalar_nonfinite_fatal(self):
        with self.assertRaises(RuntimeError):
            f.scalar(np.eye(2),np.full((2,2),np.nan),np.eye(2))

    def test_finite_differences_do_not_stop(self):
        value = f.difference(np.eye(2),100*np.eye(2))
        self.assertTrue(value["exceeds_maxabs"])
        self.assertTrue(value["exceeds_relative_frobenius"])
    def test_bitwise_not_numeric(self):
        aa = np.zeros((2,2))
        bb = aa.copy()
        bb[0,0] = -0.
        delta = f.difference(aa,bb)
        self.assertFalse(delta["bitwise_equal"])
        self.assertEqual(delta["maxabs"],0)
    def test_difference_nonfinite_fatal(self):
        with self.assertRaises(RuntimeError):
            f.difference(np.eye(2),np.full((2,2),np.inf))

    def test_no_measured_fields_report(self):
        for key in ("wall_s","elapsed","time","ratio","median","speedup"):
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                f.no_timing({"outputs":[{key:3}]})
        f.no_timing(dict(outputs=[dict(scalar_eh=1.0,relative_frobenius=0.1)],accepted_speedup_data=False))

    def test_no_login_execution(self):
        with patch.dict(os.environ,{},clear=True),self.assertRaises(RuntimeError):
            f.allocated()
    def test_wrong_gpu_pid(self):
        with patch.object(runner.subprocess,"check_output",return_value="GPU-wrong, "+str(os.getpid())):
            with self.assertRaisesRegex(RuntimeError,"UUID"):
                runner.gpu_pid("GPU-expected")

    def test_source_semantics_both_pins(self):
        proof = c.source_review(dict(baseline=BASE,candidate=CAND))
        self.assertEqual(set(proof),{"baseline","candidate"})
        self.assertEqual(proof["baseline"]["psi4/src/psi4/libscf_solver/sad.cc"],
                         proof["candidate"]["psi4/src/psi4/libscf_solver/sad.cc"])

    def test_prior107_helpers_byte_identical(self):
        files = list((OLD/"devtools/major2").iterdir())
        for file in files:
            if file.is_file():
                self.assertEqual(file.read_bytes(),(HERE/file.name).read_bytes(),file.name)

    def test_scientific_api_calls_absent(self):
        tree = ast.parse((HERE/"fixed_jk_runner.py").read_text())
        attrs = [n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for forbidden in ("energy","gradient","hessian","Wavefunction","RHF","guess","scf","compute_energy"):
            self.assertNotIn(forbidden,attrs)
        self.assertNotIn("partial_cholesky_factorize",attrs)
        self.assertEqual((HERE/"fixed_jk.py").read_text().count(".partial_cholesky_factorize("),1)

    def test_no_jk_array_state_writes(self):
        tree = ast.parse((HERE/"fixed_jk_runner.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node,(ast.Assign,ast.AugAssign)):
                targets = node.targets if isinstance(node,ast.Assign) else [node.target]
                for target in targets:
                    self.assertNotIn("jk.",ast.unparse(target))

    def test_batch_one_free_h200_no_rebuild(self):
        text = c.batch_text("a"*64)
        for token in ("--qos=embers","--time=04:00:00","--cpus-per-task=8","--mem=128G",
                      "--gres=gpu:h200:1","--no-requeue",f.HARNESS,"fixed_jk_campaign.py run"):
            self.assertIn(token,text)
        self.assertNotIn("inferno",text)
        self.assertNotIn("repeatability_campaign.py run",text)
        self.assertNotIn("build.sh",text)

    def test_parent_only_permanent_submit_uses_old_api(self):
        text = (HERE/"fixed_jk_handoff.py").read_text()
        self.assertIn('os.environ.get("MAJOR2_PARENT_REVIEW") == f.REVIEW_ID',text)
        self.assertIn('mode == "parent-submit"',text)
        self.assertIn("h.submit(",text)
        prior = (HERE/"handoff.py").read_text()
        self.assertLess(prior.index('durable(lock/"job-id.txt"'),prior.index('info = output("scontrol"'))
        self.assertIn("lock.mkdir()",prior)
        self.assertIn("os.fsync(directory)",prior)

    def test_wrong_manifest_policy_negatives(self):
        # A real generated manifest is optional at initial development, mandatory in run-tests after freeze.
        path = os.environ.get("MAJOR2_FIXED_MANIFEST")
        if not path:
            self.skipTest("freeze not yet generated; rerun with manifest")
        original = p.read_json(path)
        f.validate(original)
        for key,wrong in [("settings",{}),("input_sha256","0"*64),("sequence",[]),("findings_bounds",{}),
                          ("matrix_cap_bytes",1),("route",{}),("factor_tolerance",1e-6),("identity_helper",{})]:
            m = copy.deepcopy(original)
            m[key] = wrong
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                f.validate(m)
        m = copy.deepcopy(original)
        m["arms"]["candidate"]["pin"] = "c9c266e5a3"
        with self.assertRaises(RuntimeError):
            f.validate(m)

    def test_all24_outputs_276_pairs_retain_finite_drift(self):
        with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as tmp:
            raw = Path(tmp)
            (raw/"factor").mkdir()
            factor = dict(C={"array":{"sha256":"frozen"}},D={"array":{"sha256":"common"}},
                          reconstruction=dict(maxabs=1e-8))
            o.atomic(raw/"factor/FACTOR.json",factor)
            values = []
            density = np.eye(2)
            for i,row in enumerate(f.sequence()):
                root = raw/row["label"]
                root.mkdir()
                records = []
                for state in f.STATES:
                    arrays = dict(J=np.eye(2)*(i+1),K=np.eye(2))
                    with (root/(state+".npz")).open("xb") as stream:
                        np.savez_compressed(stream,**arrays)
                    records.append(dict(state=state,file=state+".npz",file_sha256=p.sha(root/(state+".npz")),
                        arrays={k:o.array_info(v) for k,v in arrays.items()},
                        scalar_eh=f.scalar(density,**dict(j=arrays["J"],k=arrays["K"])),
                        symmetry_maxabs=dict(J=0.,K=0.),symmetry_findings=dict(J=False,K=False)))
                values.append(dict(label=row["label"],arm=row["arm"],pid=i+1,records=records,
                                   molecule_geometry={"sha256":"same"}))
            with patch.object(f,"allocated"),patch.object(f,"load_factor",return_value=(None,density)):
                report = c.analyze(raw,values)
                self.assertEqual(len(report["outputs"]),24)
                self.assertEqual(len(report["differences"]),276)
                self.assertTrue(any(v["scalar_exceeds_bound"] for v in report["differences"]))
                self.assertFalse(report["accepted_speedup_data"])
                f.no_timing(report)
                values[-1]["pid"] = values[0]["pid"]
                with self.assertRaisesRegex(RuntimeError,"six distinct"):
                    c.analyze(raw,values)


if __name__ == "__main__":
    unittest.main()
