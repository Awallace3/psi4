"""HOST/FAKE ONLY. No Psi4 import, GPU operation, scheduler call or real submission."""
import ast
import copy
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import config as s
import custody
import fixed_jk as f
import observe as o
from numerics import Analysis, Store, total_storage
from protocol import Timer, dispatch, expected_trace
import runner
import runtime
from statistics_report import summarize
from controller import accept_inputs, accept_candidate_stability, records_check


class FakeJK:
    def __init__(self, events, arm="candidate", fail=None):
        self.events, self.arm, self.fail = events, arm, fail
        self.calls = dict.fromkeys(s.CALLS, 0)
        self.initializations = 0

    def operation(self, name):
        self.events.append(name)
        self.calls[name] = self.calls.get(name, 0)+1
        if name == self.fail:
            raise RuntimeError("injected "+name)

    def initialize(self):
        self.operation("initialize")
        self.initializations += 1

    def compute(self):
        self.operation("compute")

    def finalize(self):
        self.operation("finalize")

    def C_clear(self):
        self.events.append("C_clear")

    def C_left_add(self, _):
        self.operation("C_left_add")

    def text(self):
        if self.arm == "baseline":
            return f.SIMPLE_DF_WARNING+"Device memory, DF int plan:\n"*self.initializations
        return (f.SIMPLE_DF_WARNING+"cuESTJK plan created #1\nDevice memory, DF int plan:\n"+
                "cuESTJK plan reused #1\n"*(self.initializations-1))


class DispatchTests(unittest.TestCase):
    def run_dispatch(self, arm="candidate", sync_fail=None, clock=None,
                     op_fail=None, snapshot_fail=None, guard_fail=False):
        self.events = events = []
        self.jk = jk = FakeJK(events, arm, op_fail)
        tick = iter(range(100, 10000, 7))
        def default_clock():
            events.append("clock")
            return next(tick)
        sync_count = 0
        def sync():
            nonlocal sync_count
            events.append("sync")
            sync_count += 1
            if sync_count == sync_fail:
                raise RuntimeError("injected sync failure")
        def guard():
            events.append("guard")
            if guard_fail:
                raise RuntimeError("injected guard")
        def snapshot(phase, round_id, timing, calls):
            events.append("snapshot")
            if phase == snapshot_fail:
                raise RuntimeError("injected snapshot")
            trace = f.traces(jk.text(), arm)
            self.assertEqual(trace, expected_trace(arm, calls["initialize"]))
            return dict(phase=phase, round=round_id, timing=timing, calls=calls,
                        trace=trace, fake_host_only=True)
        return dispatch(jk, object(), Timer(sync, clock or default_clock), snapshot, guard)

    def test_real_dispatch_exact_order_and_counts_both_arms(self):
        for arm in ("baseline", "candidate"):
            records, calls = self.run_dispatch(arm)
            self.assertEqual(calls, s.CALLS)
            self.assertEqual(self.jk.calls, s.CALLS)
            self.assertEqual(len(records), 11)
            want = ["guard", "sync", "clock", "initialize", "sync", "clock",
                    "C_clear", "C_left_add", "sync", "clock", "compute", "sync", "clock",
                    "snapshot", "guard"]
            for _ in range(5):
                want += ["sync", "clock", "initialize", "sync", "clock",
                         "compute", "sync", "clock", "snapshot", "guard",
                         "sync", "clock", "compute", "sync", "clock", "snapshot", "guard"]
            want += ["finalize"]
            self.assertEqual(self.events, want)
            self.assertEqual(records[-1]["trace"], expected_trace(arm, 6))
            self.assertTrue(all(r["fake_host_only"] for r in records))

    def test_combined_clock_boundaries(self):
        records, _ = self.run_dispatch()
        for row in records[1:]:
            if row["phase"] == "initialize-compute":
                self.assertEqual(row["timing"], dict(initialize_ns=7, compute_ns=7, combined_ns=14))
            else:
                self.assertEqual(row["timing"], dict(control_ns=7))

    def test_every_sync_failure_is_fatal_and_finalizes(self):
        # Warmup 4 sync + five rounds * (3+2) =29.
        for number in range(1, 30):
            with self.subTest(sync=number), self.assertRaisesRegex(RuntimeError, "sync"):
                self.run_dispatch(sync_fail=number)
            self.assertEqual(self.jk.calls["finalize"], 1)

    def test_native_failure_finalizes(self):
        for op in ("initialize", "compute", "C_left_add"):
            with self.subTest(op=op), self.assertRaises(RuntimeError):
                self.run_dispatch(op_fail=op)
            self.assertEqual(self.jk.calls["finalize"], 1)

    def test_snapshot_failure_finalizes(self):
        for phase in ("warmup", "initialize-compute", "control"):
            with self.assertRaisesRegex(RuntimeError, "snapshot"):
                self.run_dispatch(snapshot_fail=phase)
            self.assertEqual(self.jk.calls["finalize"], 1)

    def test_early_guard_failure_finalizes(self):
        with self.assertRaisesRegex(RuntimeError, "guard"):
            self.run_dispatch(guard_fail=True)
        self.assertEqual(self.jk.calls["finalize"], 1)

    def test_signal_propagation(self):
        def clock():
            runner.stop(signal.SIGTERM, None)
        with self.assertRaisesRegex(RuntimeError, "signal"):
            self.run_dispatch(clock=clock)
        self.assertEqual(self.jk.calls["finalize"], 1)

    def test_invalid_clock_samples_fail(self):
        for ticks in ((2, 2), (2, 1), (-1, 1), (1.0, 2), (False, 1)):
            with self.subTest(ticks=ticks), self.assertRaisesRegex(RuntimeError, "timing"):
                self.run_dispatch(clock=iter(ticks).__next__)
            self.assertEqual(self.jk.calls["finalize"], 1)

    def test_trace_fallback_errors_and_wrong_counts(self):
        for arm in ("candidate", "baseline"):
            jk = FakeJK([], arm)
            jk.initializations = 6
            self.assertEqual(f.traces(jk.text(), arm), expected_trace(arm, 6))
            for suffix in ("MemDFJK", "DiskDFJK", "DirectJK", "CUDA error", "status code"):
                with self.assertRaises(RuntimeError):
                    f.traces(jk.text()+suffix, arm)
        with self.assertRaises(RuntimeError):
            f.traces("cuESTJK plan created #1", "candidate")

    def test_no_analysis_in_production_timing_functions(self):
        source = (s.HERE/"protocol.py").read_text()
        tree = ast.parse(source)
        timer = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        for node in ast.walk(timer):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                self.assertFalse(any(x in name for x in ("snapshot", "trace", "hash", "save",
                                                         "nvidia", "array", "json", "env")))
        for phase in ("initialize-compute", "control", "warmup"):
            self.assertIn(phase, source)


class TemporaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"])
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()


class NumericTests(TemporaryTests):
    def analysis(self):
        return Analysis(Store(self.root), (2, 2))

    @staticmethod
    def arrays(delta=0.):
        return dict(J=np.ones((2, 2), dtype="<f8")+delta,
                    K=np.full((2, 2), .5, dtype="<f8"))

    def test_findings_not_fatal_and_no_full_intermediate_outputs(self):
        analysis = self.analysis()
        d = np.eye(2, dtype="<f8")
        first = analysis.snapshot("warmup", 0, self.arrays(), d)
        self.assertTrue(first["stable_within_original_roundoff"])
        row = analysis.snapshot("initialize-compute", 1, self.arrays(1e-5), d)
        self.assertTrue(row["findings"]["matrices"]["J"]["exceeds_maxabs"])
        self.assertTrue(row["findings"]["matrices"]["J"]["exceeds_relative_frobenius"])
        self.assertTrue(row["findings"]["scalar_exceeds_bound"])
        self.assertFalse(row["stable_within_original_roundoff"])
        self.assertFalse(row["saved"])
        self.assertEqual(len(list(self.root.glob("*.npy"))), 2)

    def test_stores_only_four_readonly_arrays(self):
        analysis = self.analysis()
        d = np.eye(2, dtype="<f8")
        for phase, i in [("warmup", 0)]+[(p, i) for i in range(1, 6)
                                       for p in ("initialize-compute", "control")]:
            row = analysis.snapshot(phase, i, self.arrays(), d)
            self.assertEqual(bool(row["saved"]), phase == "warmup" or (phase == "control" and i == 5))
        files = list(self.root.glob("*.npy"))
        self.assertEqual(len(files), 4)
        self.assertTrue(all(not p.stat().st_mode & 0o222 for p in files))
        self.assertEqual(analysis.store.used, sum(p.stat().st_size for p in files))

    def test_actual_density_change_is_detected_not_frozen_D_assumption(self):
        a = self.analysis()
        a.snapshot("warmup", 0, self.arrays(), np.eye(2, dtype="<f8"))
        row = a.snapshot("control", 1, self.arrays(), np.ones((2, 2), dtype="<f8"))
        self.assertFalse(row["matched_D"])
        self.assertFalse(row["actual_D_difference"]["bitwise_equal"])

    def test_fatal_nonfinite_and_bad_shape(self):
        for key in ("J", "K", "D"):
            a = self.analysis()
            arrays, d = self.arrays(), np.eye(2, dtype="<f8")
            if key == "D":
                d[0, 0] = np.nan
            else:
                arrays[key][0, 0] = np.inf
            with self.assertRaisesRegex(RuntimeError, "finite"):
                a.snapshot("warmup", 0, arrays, d)
        for a in (np.ones((3, 3)), np.ones((2, 2), dtype=np.float32), np.ones((2, 3))[:, :2]):
            with self.assertRaises(RuntimeError):
                f.finite_array(a, (2, 2))

    def test_storage_reservation_precedes_any_write(self):
        a = Analysis(Store(self.root, cap=1), (2, 2))
        with self.assertRaisesRegex(RuntimeError, "cap"):
            a.snapshot("warmup", 0, self.arrays(), np.eye(2, dtype="<f8"))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_disallowed_or_duplicate_storage_fails(self):
        store = Store(self.root)
        for label in ("round1-control", "round5-initialize-compute"):
            with self.assertRaises(RuntimeError):
                store.save(label, self.arrays())
        store.save("warmup", self.arrays())
        with self.assertRaises(RuntimeError):
            store.save("warmup", self.arrays())

    def test_global_array_count_cap(self):
        child = self.root/"speed-baseline-1"
        child.mkdir()
        for i in range(25):
            (child/(str(i)+".npy")).touch()
        with self.assertRaisesRegex(RuntimeError, "cap"):
            total_storage(self.root)

    def test_frozen_factor_receipt_0600_does_not_get_chmodded(self):
        receipt = self.root/"FACTOR.json"
        receipt.write_bytes(b"historical receipt")
        receipt.chmod(0o600)
        c = self.root/"C.npy"
        c.write_bytes(b"frozen array")
        c.chmod(0o444)
        pins = {p.name:s.sha(p) for p in (receipt, c)}
        before = [(p.stat().st_mode, p.stat().st_mtime_ns) for p in (receipt, c)]
        s.n.check_factor(self.root, pins)
        self.assertEqual(before, [(p.stat().st_mode, p.stat().st_mtime_ns) for p in (receipt, c)])
        c.chmod(0o644)
        with self.assertRaises(RuntimeError):
            s.n.check_factor(self.root, pins)


def fake_values():
    values = []
    for index, row in enumerate(s.SEQUENCE):
        records = [dict(phase="warmup", round=0,
                        timing=dict(initialize_ns=999999, compute_ns=999999))]
        for i in range(1, 6):
            base = (index+1)*10+i
            records.extend([dict(phase="initialize-compute", round=i,
                timing=dict(initialize_ns=base, compute_ns=base*2, combined_ns=base*3)),
                dict(phase="control", round=i, timing=dict(control_ns=base*2))])
        for r in records:
            r.update(actual_D=dict(sha256="actual-wrapper-not-frozen-D", shape=[1786, 1786]),
                     stable_within_original_roundoff=True)
        values.append(dict(row, records=records, actual_geometry={"sha256":"frame"},
                           factor_C={"sha256":"C"}))
    return values


class StatisticsTests(unittest.TestCase):
    def test_all_raw_samples_and_balanced_pair_ratios(self):
        values = fake_values()
        result = summarize(values)
        self.assertEqual(result["arms"]["baseline"]["combined_ns"]["process_medians"], [39, 129, 159])
        self.assertEqual(result["arms"]["candidate"]["combined_ns"]["process_medians"], [69, 99, 189])
        self.assertEqual(result["primary"], 129/99)
        self.assertEqual([v["ratios"]["combined_ns"] for v in result["balanced_adjacent_pairs"]],
                         [39/69, 129/99, 159/189])
        self.assertEqual(result["processes"][0]["samples"]["combined_ns"]["raw"], [33, 36, 39, 42, 45])
        self.assertEqual(result["processes"][0]["samples"]["combined_ns"]["minimum"], 33)
        self.assertEqual(result["processes"][0]["samples"]["combined_ns"]["maximum"], 45)
        self.assertNotIn("actual_gpu_execution", result)  # fake output cannot be labeled actual GPU

    def test_no_outlier_filtering(self):
        values = fake_values()
        values[0]["records"][1]["timing"] = dict(initialize_ns=1, compute_ns=999999, combined_ns=1000000)
        result = summarize(values)
        self.assertIn(1000000, result["processes"][0]["samples"]["combined_ns"]["raw"])
        self.assertEqual(result["processes"][0]["samples"]["combined_ns"]["maximum"], 1000000)

    def test_missing_reordered_or_duplicate_cohort_fails(self):
        for indices in ([0, 1, 2, 3, 4], [1, 0, 2, 3, 4, 5], [0, 1, 2, 3, 4, 4]):
            values = fake_values()
            with self.assertRaises(RuntimeError):
                summarize([values[i] for i in indices])

    def test_missing_or_nonpositive_samples_fail(self):
        for change in ("drop", "negative", "nan", "float", "bad-total"):
            values = fake_values()
            if change == "drop":
                values[0]["records"].pop()
            else:
                values[0]["records"][1]["timing"]["combined_ns"] = {
                    "negative":-1, "nan":float("nan"), "float":33.0, "bad-total":44}[change]
            with self.assertRaises(RuntimeError):
                summarize(values)

    def test_density_consistency_all_sixtysix_calls(self):
        values = fake_values()
        accept_inputs(values)
        for process, ordinal in ((0, 1), (5, 10), (2, 0)):
            changed = copy.deepcopy(values)
            changed[process]["records"][ordinal]["actual_D"]["sha256"] = "different"
            with self.assertRaisesRegex(RuntimeError, "density"):
                accept_inputs(changed)

    def test_geometry_and_coefficients_must_match(self):
        for key in ("actual_geometry", "factor_C"):
            values = fake_values()
            values[-1][key] = {}
            with self.assertRaises(RuntimeError):
                accept_inputs(values)

    def test_finite_baseline_findings_do_not_discard_samples(self):
        values = fake_values()
        values[0]["records"][1]["stable_within_original_roundoff"] = False
        accept_candidate_stability(values)
        self.assertEqual(len(summarize(values)["processes"]), 6)

    def test_candidate_roundoff_breach_withholds_acceptance(self):
        values = fake_values()
        values[1]["records"][1]["stable_within_original_roundoff"] = False
        with self.assertRaisesRegex(RuntimeError, "candidate"):
            accept_candidate_stability(values)

    def test_real_parent_per_call_counter_validation(self):
        import repeatability as r
        import policy as p
        value = fake_values()[0]
        row = s.SEQUENCE[0]
        value.update(pid=123, job="456", gpu_uuid="GPU-fake",
            actual_gpu_execution=True, backend="cuESTJK", calls=s.CALLS,
            trace=expected_trace(row["arm"], 6),
            core_sha256=r.ALLOWLIST[row["arm"]]["core_sha256"], numerical_overrides={},
            cudart=p.RUNTIME_LIBS["libcudart.so"], maps_after_CDLL={}, final_runtime_maps={},
            factor_C={"array":{"sha256":"C"}})
        for i, rec in enumerate(value["records"]):
            initials = 1+(i+1)//2
            rec.update(calls=dict(initialize=initials, compute=i+1, finalize=0,
                C_left_add=1, C_right_add=0), trace=expected_trace(row["arm"], initials),
                gpu=dict(pid=123, uuid="GPU-fake"), C={"sha256":"C"}, matched_D=True)
        # Internal fake metadata only, never written/labeled as actual GPU results.
        with patch.object(p, "check_runtime_maps"), patch.object(p, "check_required_runtime_maps"):
            records_check(value, row, 123, "GPU-fake", "456")
            for field, replacement in (("calls", {}), ("backend", "MemDFJK"),
                                       ("core_sha256", "wrong"), ("pid", 124)):
                wrong = copy.deepcopy(value)
                wrong[field] = replacement
                with self.assertRaises(RuntimeError):
                    records_check(wrong, row, 123, "GPU-fake", "456")
            for index in range(11):
                wrong = copy.deepcopy(value)
                wrong["records"][index]["calls"]["compute"] += 1
                with self.assertRaises(RuntimeError):
                    records_check(wrong, row, 123, "GPU-fake", "456")


class Function:
    def __init__(self, value=0, rc=0):
        self.value, self.rc = value, rc
    def __call__(self, *args):
        if args:
            args[0]._obj.value = self.value
        return self.rc


class RuntimeTests(TemporaryTests):
    def fake_cuda(self, version=13030, count=1, device=0, sync_rc=0):
        lib = type("HostFakeLibrary", (), {})()
        lib.cudaRuntimeGetVersion = Function(version)
        lib.cudaGetDeviceCount = Function(count)
        lib.cudaGetDevice = Function(device)
        lib.cudaDeviceSynchronize = Function(rc=sync_rc)
        file = self.root/"libcudart.fake"
        file.touch(exist_ok=True)
        pins = dict(alias=str(file), realpath=str(file), sha256=s.sha(file))
        order = []
        def cdll(path):
            order.append(("CDLL", path))
            return lib
        def maps():
            order.append(("maps", None))
            return {}
        with patch.dict(runtime.p.RUNTIME_LIBS, {"libcudart.so":pins}), \
             patch.object(runtime.ctypes, "CDLL", side_effect=cdll), \
             patch.object(runtime, "validate_maps", side_effect=maps):
            cuda = runtime.Cuda()
        return cuda, lib, order

    def test_exact_cudart_signatures_and_mapping_after_CDLL(self):
        cuda, lib, order = self.fake_cuda()
        self.assertEqual([v[0] for v in order], ["CDLL", "maps"])
        self.assertIs(lib.cudaDeviceSynchronize.restype, ctypes.c_int)
        self.assertEqual(lib.cudaDeviceSynchronize.argtypes, [])
        for fn in (lib.cudaRuntimeGetVersion, lib.cudaGetDeviceCount, lib.cudaGetDevice):
            self.assertIs(fn.restype, ctypes.c_int)
            self.assertEqual(fn.argtypes, [ctypes.POINTER(ctypes.c_int)])
        cuda.synchronize()

    def test_sync_error_propagates(self):
        cuda, _, _ = self.fake_cuda(sync_rc=99)
        with self.assertRaisesRegex(RuntimeError, "Synchronize"):
            cuda.synchronize()

    def test_wrong_runtime_or_logical_mapping_is_fatal(self):
        for args in (dict(version=12060), dict(count=2), dict(device=1)):
            with self.assertRaisesRegex(RuntimeError, "logical"):
                self.fake_cuda(**args)

    def test_no_numerical_overrides_even_with_cuda_locators(self):
        self.assertEqual(s.n.check_env(dict(CUDA_ROOT="/site/12", CUDA_HOME="/pin/13",
                                            CUDA_PATH="/pin/13")), {})
        for key in ("CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "CUDA_LAUNCH_BLOCKING",
                    "CUDA_UNKNOWN_CONTROL", "CUSOLVER_FOO", "OMP_PROC_BIND", "OMP_PLACES"):
            env = dict(CUDA_ROOT="/site/12", **{key:"1"})
            with self.assertRaisesRegex(RuntimeError, key):
                s.n.check_env(env)
            self.assertIn(key, env)  # never unset to force passing

    def test_exclusive_own_pid_and_uuid(self):
        uuid, pid = "GPU-fake", os.getpid()
        with patch.object(runner.subprocess, "check_output", return_value=f"{uuid}, {pid}\n"):
            self.assertEqual(runner.exclusive_pid(uuid), dict(pid=pid, uuid=uuid))
        for text in ("", f"GPU-wrong, {pid}", f"{uuid}, 99999", f"{uuid}, {pid}\n{uuid}, 99999"):
            with patch.object(runner.subprocess, "check_output", return_value=text), \
                 self.assertRaises(RuntimeError):
                runner.exclusive_pid(uuid)

    def test_early_receipt_precedes_scientific_imports(self):
        source = (s.HERE/"runner.py").read_text()
        tree = ast.parse(source)
        imports = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any(any(k in v for k in ("numpy", "psi4", "runtime", "probe")) for v in imports))
        self.assertIn("receipt = early(sys.argv)", source)
        self.assertIn("execute(m, row, uuid, receipt, started)", source)
        self.assertIn('getattr(jk, k)()[0].np', source)
        self.assertIn('jk.D()[0].np', source)
        self.assertNotIn("partial_cholesky_factorize", source)


class CustodyTests(TemporaryTests):
    def test_sixteen_roots_accounting_requires_job_and_batch(self):
        self.assertEqual((len(custody.FAILED), len(custody.COMPLETED)), (13, 3))
        rows = [f"{j}{suffix}|{state}|{code}|" for jobs, state, code in (
            (custody.FAILED, "FAILED", "1:0"), (custody.COMPLETED, "COMPLETED", "0:0"))
            for j in jobs for suffix in ("", ".batch")]
        with patch.object(s.n, "output", return_value="\n".join(rows)):
            self.assertEqual(len(custody.accounting()), 32)
        for r in ("14178506.batch|COMPLETED|0:0|", "14178058.batch|FAILED|1:0|"):
            with patch.object(s.n, "output", return_value="\n".join(x for x in rows if x != r)), \
                 self.assertRaises(RuntimeError):
                custody.accounting()

    def test_exact_resource_test_only_and_no_submission(self):
        text = custody.batch_text("a"*64)
        for token in ("--qos=embers", "--partition=gpu-h200", "--constraint=H200",
                      "--gres=gpu:h200:1", "--cpus-per-task=8", "--mem=128G", "--time=04:00:00",
                      "--signal=TERM@300", s.RUN+"/logs/", s.SOURCE, "supervise.sh"):
            self.assertIn(token, text)
        result = subprocess.CompletedProcess([], 0, "", "partition gpu-h200")
        with patch.object(custody, "verify"), patch.object(custody, "queue", return_value=[]), \
             patch.object(custody, "accounting", return_value=[]), \
             patch.object(custody.subprocess, "run", return_value=result) as run:
            report = custody.audit("p", "m", "b")
        self.assertEqual(run.call_args.args[0], ["sbatch", "--test-only", s.RUN+"/campaign.sbatch"])
        self.assertTrue(report["no_submission"])

    def test_unreviewed_parent_submission_never_calls_scheduler(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(custody.subprocess, "run") as run, self.assertRaises(RuntimeError):
            custody.parent_submit("p", "m", "b")
        run.assert_not_called()

    def test_id_parser_rejects_ambiguity(self):
        self.assertEqual(custody.accept_id("12345;phoenix-slurm\n"), "12345")
        for text in ("", "123\n456\n", "Submitted batch job 123", "0", "123 junk", "123\nwarning"):
            with self.assertRaises(RuntimeError):
                custody.accept_id(text)

    def test_numeric_id_durable_before_query_and_one_shot(self):
        (self.root/"logs").mkdir()
        response = subprocess.CompletedProcess([], 0, "987654;phoenix-slurm\n", "")
        def query(*_):
            self.assertEqual((self.root/"logs/job-id.txt").read_text(), "987654\n")
            self.assertFalse((self.root/"logs/job-id.txt").stat().st_mode & 0o222)
            return (f"JobId=987654 QOS=embers Account=gts-cs207-chemx Partition=gpu-h200 "
                    f"Command={self.root}/campaign.sbatch")
        with patch.object(s, "RUN", str(self.root)), \
             patch.dict(os.environ, MAJOR2_SPEED_PARENT_REVIEW="m"), \
             patch.object(custody, "audit", return_value={}), \
             patch.object(custody.subprocess, "run", return_value=response) as submit, \
             patch.object(s.n, "output", side_effect=query):
            custody.parent_submit("p", "m", "b")  # ONLY MOCK sbatch; local temp root
            with self.assertRaises(FileExistsError):
                custody.parent_submit("p", "m", "b")
        self.assertEqual(submit.call_count, 1)

    def test_ambiguous_timeout_leaves_permanent_attempt(self):
        (self.root/"logs").mkdir()
        with patch.object(s, "RUN", str(self.root)), \
             patch.dict(os.environ, MAJOR2_SPEED_PARENT_REVIEW="m"), \
             patch.object(custody, "audit", return_value={}), \
             patch.object(custody.subprocess, "run", side_effect=subprocess.TimeoutExpired("fake", 1)) as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                custody.parent_submit("p", "m", "b")
            with self.assertRaises(FileExistsError):
                custody.parent_submit("p", "m", "b")
        self.assertEqual(run.call_count, 1)
        self.assertTrue((self.root/"logs/parent-attempt").is_dir())

    def test_exclusive_durable_permissions_and_no_overwrite(self):
        p = self.root/"receipt.json"
        s.durable(p, {"host_fake":True})
        self.assertEqual(p.stat().st_mode & 0o777, 0o444)
        with self.assertRaises(FileExistsError):
            s.durable(p, {})

    def test_local_finalizer_is_reviewed_and_separate_from_job(self):
        source = (s.HERE/"parent-finalize.sh").read_text()
        self.assertIn("MAJOR2_SPEED_PARENT_REVIEW", source)
        self.assertNotIn("parent-submit", custody.batch_text("a"*64))
        self.assertNotIn("parent-finalize", custody.batch_text("a"*64))
        self.assertIn('permanent_lock(bundle.parent/"parent-finalizer.attempt")',
                      (s.HERE/"custody.py").read_text())

    def test_local_timeout_cannot_be_retried(self):
        bundle = self.root/"bundle"
        bundle.mkdir()
        receipt = dict(pin="pin", manifest_sha256="m", batch_sha256="b")
        s.durable(bundle/"bundle-receipt.json", receipt)
        (bundle/"campaign.sbatch").write_text("batch")
        with patch.dict(os.environ, MAJOR2_SPEED_PARENT_REVIEW="m"), \
             patch.object(s, "manifest", return_value={"pin":"pin"}), \
             patch.object(s, "source_check"), patch.object(s, "sha", return_value="b"), \
             patch.object(custody, "batch_text", return_value="batch"), \
             patch.object(custody.subprocess, "run", side_effect=subprocess.TimeoutExpired("fake", 1)) as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                custody.parent_local(bundle, "/never-executed-fake-phx")
            with self.assertRaises(FileExistsError):
                custody.parent_local(bundle, "/never-executed-fake-phx")
        self.assertEqual(run.call_count, 1)
        self.assertTrue((self.root/"parent-finalizer.attempt").is_dir())
        self.assertTrue(s.load(self.root/"local-submit-ambiguous.json")["ambiguous"])


class SupervisorTests(TemporaryTests):
    def run_fake(self, code):
        return subprocess.run(["bash", str(s.HERE/"supervise.sh"), str(self.root),
            sys.executable, "-B", "-c", code, str(self.root)],
            capture_output=True, text=True, timeout=10)

    def test_zero_exit_without_verified_complete_is_failed(self):
        result = self.run_fake("pass")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.root/"FAILED").exists())
        self.assertFalse((self.root/"COMPLETED").exists())

    def test_nonzero_exit_is_failed(self):
        result = self.run_fake("raise SystemExit(7)")
        self.assertEqual(result.returncode, 7)
        self.assertTrue((self.root/"FAILED").exists())

    def test_exact_completion_is_success(self):
        code = ('import json,sys;from pathlib import Path;p=Path(sys.argv[1]);'
                '(p/"SPEED-COMPLETE.json").write_text(json.dumps(dict(processes=6,computes=66,'
                'stored_arrays=24,matched_actual_D=True,plan_reuse_benchmark_only=True)));'
                '(p/"SPEED-REPORT.json").write_text(\'{"host_fake_only":true}\')')
        result = self.run_fake(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root/"COMPLETED").exists())
        self.assertEqual(json.loads((self.root/"batch-exit.json").read_text()), dict(exit=0, signalled=0))
        again = self.run_fake("pass")
        self.assertNotEqual(again.returncode, 0)

    def test_term_never_false_success(self):
        code = ('import signal,sys,time;from pathlib import Path;'
                'signal.signal(signal.SIGTERM,lambda *_:sys.exit(0));'
                '(Path(sys.argv[1])/"ready").touch();time.sleep(8)')
        proc = subprocess.Popen(["bash", str(s.HERE/"supervise.sh"), str(self.root),
            sys.executable, "-B", "-c", code, str(self.root)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic()+5
            while not (self.root/"ready").exists() and time.monotonic() < deadline:
                time.sleep(.01)  # bounded host fake fixture rendezvous, not GPU monitoring
            self.assertTrue((self.root/"ready").exists())
            proc.send_signal(signal.SIGTERM)
            proc.communicate(timeout=5)
            self.assertNotEqual(proc.returncode, 0)
            self.assertFalse((self.root/"COMPLETED").exists())
            self.assertTrue(json.loads((self.root/"batch-exit.json").read_text())["signalled"])
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()


if __name__ == "__main__":
    unittest.main()
