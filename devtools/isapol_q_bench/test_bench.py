"""Stdlib/fake-child tests only: never import Psi4/NumPy or compute native Q."""
import copy
import builtins
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bench", HERE/"bench.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
ops = b.ops


class Sampler:
    def __init__(self, *unused):
        self.errors, self.peaks = [], {}
    def start(self):
        pass
    def stop(self):
        return dict(errors=[])


class BenchTests(unittest.TestCase):
    def claim(self):
        return dict(model=dict(method="PBE0", basis="aug-cc-pVTZ"), protocols={"stdout": False},
                    molecule=dict(geometry=[0., 0., 0.]),
                    keywords=dict(function_kwargs=dict(native_monomer=dict(
                        implementation_sha256=dict(ops.OLD_ID), pfit_maximum_work_bytes=268435456,
                        declaration=dict(profile="unchanged"), resources=dict(max_bytes=12884901888)))))

    def test_derived_claim_changes_exactly_two_approved_identity_leaves(self):
        original = self.claim()
        frozen = copy.deepcopy(original)
        candidate, diff = b.derive_claim(original, "candidate")
        self.assertEqual(original, frozen)
        self.assertEqual(len(diff), 2)
        self.assertEqual({d["path"][-1] for d in diff}, {"core", "isapol_python"})
        for d in diff:
            self.assertEqual(d["before"], ops.OLD_ID[d["path"][-1]])
            self.assertEqual(d["after"], b.CAND_ID[d["path"][-1]])
        self.assertEqual(len(b.changes(original, candidate)), 2)
        self.assertEqual(b.derive_claim(original, "old"), (original, []))
        wrong = self.claim()
        wrong["keywords"]["function_kwargs"]["native_monomer"]["implementation_sha256"] = b.CAND_ID
        with self.assertRaisesRegex(ValueError, "accepted claim identity"):
            b.derive_claim(wrong, "candidate")

    def test_full_order_and_proxy_bounds_and_frozen_files(self):
        self.assertEqual(b.FULL_ORDER, ("old", "candidate", "candidate", "old", "old", "candidate"))
        text = (HERE/"bench.py").read_text()
        self.assertIn("range(6)", text)
        self.assertIn("size=(65543, 3)", text)
        self.assertIn("columns=246", text)
        self.assertIn("SYNTHETIC_ALL_S_WIDTH_MATCHED_NOT_MOLECULAR_Q", text)
        self.assertNotIn("original_serial(", text)  # Python oracle is not timed.
        for name, digest in ops.FROZEN.items():
            self.assertEqual(ops.sha(HERE.parents[1]/name), digest)

    def test_actual_validation_identity_is_direct_pair_not_nested(self):
        self.assertEqual(b.CAND_ID["core"],
                         "676ea1e1b616a4c4df4f221688b31b8ae7a367539e9187676141b3109c85c013")
        self.assertEqual(b.CAND_ID["isapol_python"],
                         "2b7758f3eec48bb8cfc3565d50505ba1ce1fd911cbf1d762aab7773549d3713d")
        self.assertEqual(b.VALIDATION_SHA,
                         "1582867ceb6f1ab0b8bd9f7d3ddeba443c35b745436713cf3e828e8abac0b3f8")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"result.json"
            path.write_text(json.dumps(dict(identity=dict(implementation=b.CAND_ID))))
            with self.assertRaisesRegex(ValueError, "validation receipt differs"):
                b.validation_gate(path)

    def test_pool_and_affinity_mismatches_fail(self):
        state = dict(process_threads=4, native_mkl_max_threads=4, affinity=list(range(8)),
                     threadpools=None, readonly_pool_queries=[
                         dict(queries={"scipy_openblas_get_num_threads64_": 1})])
        b.ensure_threads(state, 4, list(range(8)))
        for key, value in (("native_mkl_max_threads", 8), ("affinity", [0])):
            broken = copy.deepcopy(state)
            broken[key] = value
            with self.assertRaises(ValueError):
                b.ensure_threads(broken, 4, list(range(8)))
        state["readonly_pool_queries"][0]["queries"]["scipy_openblas_get_num_threads64_"] = 8
        with self.assertRaisesRegex(ValueError, "OpenBLAS"):
            b.ensure_threads(state, 4, list(range(8)))

    def test_real_child_preimport_path_owns_cwd_even_when_import_raises(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            fixtures = base/"fixtures/monomer-0001"
            fixtures.mkdir(parents=True)
            accepted = self.claim()
            ops.write(fixtures/"claimed-input.json", dict(function_kwargs=dict(input_data=accepted)))
            normalized = dict(molecule=accepted["molecule"],
                              specification={k: v for k, v in accepted.items() if k != "molecule"})
            args = types.SimpleNamespace(source=base/"source", profiler_source=base/"profile",
                    compiled_source=base/"compiled", fixtures=fixtures, arm="old", threads=8, task="warm",
                    root=base/"output", scratch_root=base/"scratch", evidence=base/"evidence/proof",
                    validation_result=base/"validation/result", affinity=json.dumps(list(range(8))))
            p = types.SimpleNamespace(read_fixtures=lambda *a: (
                normalized, {}, dict(monomer_id="monomer-0001", fixture_sha256={"input": "frozen"})))
            original_import = builtins.__import__
            reached = []
            def guarded_import(name, *a, **kw):
                if name == "psi4":
                    self.assertEqual(Path.cwd(), args.scratch_root/"cwd")
                    reached.append(name)
                    raise RuntimeError("fake import failure; no Psi4 loaded")
                return original_import(name, *a, **kw)
            sentinel = base/"protected-cwd"
            sentinel.mkdir()
            (sentinel/"timer.dat").write_text("protected")
            old_cwd = Path.cwd()
            os.chdir(sentinel)
            try:
                with mock.patch.dict(os.environ, SLURM_JOB_ID="fake", SLURM_CPUS_PER_TASK="8",
                         OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1"), \
                     mock.patch.object(b, "source_gate"), mock.patch.object(b, "validation_gate"), \
                     mock.patch.object(b, "profile", return_value=p), \
                     mock.patch.object(builtins, "__import__", guarded_import):
                    with self.assertRaisesRegex(RuntimeError, "fake import failure"):
                        b.child(args)
                self.assertEqual(reached, ["psi4"])
                self.assertEqual((sentinel/"timer.dat").read_text(), "protected")
                self.assertEqual(Path.cwd(), args.scratch_root/"cwd")
                self.assertTrue((args.root/"used-qcschema-input.json").is_file())
            finally:
                os.chdir(old_cwd)

    def controller(self, mismatch=False, child_failure=False):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            source, profiler, inputs = base/"source", base/"profiler", base/"inputs"
            for p in (source, profiler, inputs):
                p.mkdir()
            telemetry = profiler/"devtools/isapol_native_profile/telemetry.py"
            telemetry.parent.mkdir(parents=True)
            telemetry.write_text("fake telemetry bytes; never imported")
            args = types.SimpleNamespace(
                source=source, profiler_source=profiler, compiled_source=base/"compiled",
                fixtures=inputs, evidence=base/"evidence/proof.json",
                validation_result=base/"validation/result.json", commit="commit", tree="tree",
                root=base/"out", scratch_root=base/"scratch", launch_token="ours", exit_code=0)
            calls = []
            p = types.SimpleNamespace(telemetry=types.SimpleNamespace(Sampler=Sampler))
            def fake_run(runner, name, command):
                calls.append(name)
                command = list(map(str, command))
                value = lambda key: command[command.index("--"+key)+1]
                out = Path(value("root"))
                out.mkdir()
                self.assertEqual(Path.cwd(), args.root/"cwd")
                self.assertIn("-I", command)
                self.assertIn("-B", command)
                self.assertEqual(runner.env["OPENBLAS_NUM_THREADS"], "1")
                self.assertEqual(runner.env["OMP_NUM_THREADS"], value("threads"))
                if child_failure:
                    raise ValueError("fake child exited nonzero")
                sig = dict(bits_sha256="fixed", offsets=[0, 25, 50, 75])
                if mismatch and name == "proxy-2-candidate":
                    sig["bits_sha256"] = "different"
                ops.write(out/"summary.json", dict(status="RETURNED_NOT_ACCEPTED",
                          fixture={"seed": 4103}, signature=sig))
                return ""
            old_cwd = Path.cwd()
            try:
                with mock.patch.dict(os.environ, SLURM_JOB_ID="fake", SLURM_CPUS_PER_TASK="8"), \
                     mock.patch.object(b, "source_gate"), mock.patch.object(b, "profile", return_value=p), \
                     mock.patch.object(b, "seal", return_value={"immutable": True}) as seal, \
                     mock.patch.object(ops, "allocation", return_value={"fake": True}), \
                     mock.patch.object(os, "sched_getaffinity", return_value=set(range(8))), \
                     mock.patch.object(ops.Runner, "run", fake_run):
                    if mismatch or child_failure:
                        with self.assertRaises(ValueError):
                            b.run(args)
                    else:
                        b.run(args)
                    self.assertEqual(seal.call_count, 2)
                    result = ops.read(args.root/"result.json")
                    self.assertEqual(result["status"], "FAILED" if mismatch or child_failure else b.SUCCESS)
                    self.assertFalse(result["acceptance"])
                    self.assertFalse(result["adoption"])
                    self.assertEqual(result["historical_build_status"], "FAILED_UNCHANGED")
                    status = b.finalize(args)
                    self.assertEqual(status, 1 if mismatch or child_failure else 0)
                    self.assertEqual((args.root/"COMPLETED").exists(), not (mismatch or child_failure))
                if not (mismatch or child_failure):
                    self.assertEqual(len(calls), 16)
                    self.assertEqual([n.rsplit("-", 1)[1] for n in calls if n.startswith("water")],
                                     list(b.FULL_ORDER))
            finally:
                os.chdir(old_cwd)

    def test_controller_exact_matrix_fresh_children_and_frozen_proxy_comparison(self):
        self.controller()

    def test_fixed_operand_mismatch_cannot_complete_or_start_full_runs(self):
        self.controller(mismatch=True)

    def test_child_failure_still_runs_post_integrity_seal_and_cannot_complete(self):
        self.controller(child_failure=True)

    def test_wrapper_finalization_term_and_int_before_after_publication(self):
        for phase in ("before", "after"):
            for sig in (signal.SIGTERM, signal.SIGINT):
                with self.subTest(phase=phase, signal=sig), tempfile.TemporaryDirectory() as d:
                    root = Path(d)
                    helper = root/"source/devtools/isapol_q_bench/bench.py"
                    helper.parent.mkdir(parents=True)
                    helper.write_text("""
import argparse,importlib.util,os,pathlib,signal,sys,time
p=argparse.ArgumentParser(); p.add_argument("op")
for n in ("source","commit","tree","root","scratch-root","launch-token"): p.add_argument("--"+n)
p.add_argument("--exit-code",type=int); a=p.parse_args(); a.root=pathlib.Path(a.root)
s=importlib.util.spec_from_file_location("real",os.environ["REAL_BENCH"])
m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.source_gate=lambda *a:None
state=pathlib.Path(os.environ["STATE"])
if a.op=="run":
 a.root.mkdir();m.ops.write(a.root/"owner.json",dict(token=a.launch_token,mode="benchmark"))
 m.ops.write(a.root/"result.json",dict(status=m.SUCCESS,launch_token=a.launch_token,
 old_and_candidate_unchanged=True,fixed_operand_bits_equal=True,acceptance=False,adoption=False))
 sys.exit(0)
if a.exit_code==0:
 def stop(*a):
  (state/"TERM").write_text("forwarded");sys.exit(143)
 signal.signal(signal.SIGTERM,stop)
 if os.environ["PHASE"]=="after": m.finalize(a)
 (state/"FINALIZING").write_text("ready")
 while not (state/"RELEASE").exists():time.sleep(.01)
sys.exit(m.finalize(a))
""")
                    fake = root/"srun"
                    fake.write_text("#!/usr/bin/python3\nimport os,sys\na=sys.argv[1:]\n"
                                    "while a[0].startswith('--'):a.pop(0)\nos.execv(a[0],a)\n")
                    fake.chmod(0o700)
                    wrapper = root/"wrapper"
                    wrapper.write_text((HERE/"benchmark.sbatch").read_text().replace(
                        "/opt/slurm/current/bin/srun", str(fake)))
                    process = subprocess.Popen(["bash", str(wrapper), "--source", str(root/"source"),
                        "--commit", "c", "--tree", "t", "--root", str(root/"out"),
                        "--scratch-root", str(root/"scratch")], cwd=root,
                        env=dict(os.environ, REAL_BENCH=str(HERE/"bench.py"), STATE=str(root), PHASE=phase),
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
                    try:
                        deadline = time.monotonic()+8
                        while not (root/"FINALIZING").exists() and time.monotonic() < deadline:
                            if process.poll() is not None:
                                break
                            time.sleep(.01)
                        self.assertTrue((root/"FINALIZING").exists())
                        process.send_signal(sig)
                        stdout, stderr = process.communicate(timeout=8)
                        self.assertEqual(process.returncode, 143, stderr)
                        self.assertTrue((root/"TERM").exists())
                        self.assertFalse((root/"out/COMPLETED").exists())
                        self.assertEqual(ops.read(root/"out/FAILED")["exit_code"], 143)
                    finally:
                        (root/"RELEASE").touch()
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=5)

    def test_wrapper_finite_free_embers_and_no_new_affinity_policy(self):
        text = (HERE/"benchmark.sbatch").read_text()
        subprocess.run(["bash", "-n", str(HERE/"benchmark.sbatch")], check=True)
        for s in ("--qos=embers", "--cpus-per-task=8", "--mem=32G", "--time=02:00:00", "--no-requeue"):
            self.assertIn("#SBATCH "+s, text)
        self.assertNotIn("OMP_PROC_BIND", text)
        self.assertNotIn("OMP_PLACES", text)
        self.assertIn("GIT_OPTIONAL_LOCKS=0", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
