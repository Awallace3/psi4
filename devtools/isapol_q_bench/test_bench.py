"""Stdlib/fake-child tests only: never import Psi4/NumPy or compute native Q."""
import copy
import builtins
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bench", HERE/"bench.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
ops = b.ops
FAKE_PAIR = {"core": "a"*64, "isapol_python": "b"*64}  # Synthetic tests only.


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
        candidate, diff = b.derive_claim(original, "candidate", FAKE_PAIR)
        self.assertEqual(original, frozen)
        self.assertEqual(len(diff), 2)
        self.assertEqual({d["path"][-1] for d in diff}, {"core", "isapol_python"})
        for d in diff:
            self.assertEqual(d["before"], ops.OLD_ID[d["path"][-1]])
            self.assertEqual(d["after"], FAKE_PAIR[d["path"][-1]])
        self.assertEqual(len(b.changes(original, candidate)), 2)
        self.assertEqual(b.derive_claim(original, "old", FAKE_PAIR), (original, []))
        wrong = self.claim()
        wrong["keywords"]["function_kwargs"]["native_monomer"]["implementation_sha256"] = FAKE_PAIR
        with self.assertRaisesRegex(ValueError, "accepted claim identity"):
            b.derive_claim(wrong, "candidate", FAKE_PAIR)

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

    @contextmanager
    def normal_build(self):
        """Real hashes/mappings on fake files; never load the fake .so/Python."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, cand = root/"compiled8573", root/"build"
            package = cand/"env"/b.PACKAGE
            stage = cand/"build/stage"/b.PACKAGE
            inner = cand/"build/psi4-core-prefix/src/psi4-core-build"
            def put(path, text):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            for name in ("__init__.py", "extras.py", "metadata.py"):
                put(source/"psi4"/name, "raw template/stub")
                for parent in (inner, package, stage):
                    put(parent/name, "configured "+name)
            for name in ("procrouting/isapol_schema.py", "procrouting/isapol_distribution.py",
                         "schema_wrapper.py", "driver.py", "task_planner.py", "p4util/util.py",
                         "procrouting/proc.py"):
                put(source/"psi4/driver"/name, "never executed "+name)
                put(package/"driver"/name, "never executed "+name)
            for installed, original in (
                ("header.py", source/"psi4/header.py"),
                ("tests/isapol_factorial_oracle.py", source/"tests/pytests/isapol_factorial_oracle.py"),
                ("tests/aediis-1/test_input.py", source/"tests/aediis-1/test_input.py"),
            ):
                put(original, "source "+installed)
                put(package/installed, "source "+installed)
            core_name = "core.cpython-fake.so"
            for path in (package/core_name, stage/core_name, inner/"src"/core_name):
                put(path, "FAKE BINARY - NEVER LOADED")
            dependency = cand/"env/lib/dependency.txt"
            put(dependency, "original dependency bytes")
            manifest = dict(installed_files={}, runtime_files={
                "lib/dependency.txt": ops.sha(dependency)}, external_runtime_files={})
            suite = ET.Element("testsuites")
            tests = ET.SubElement(suite, "testsuite")
            for name, count in ops.NATIVE_CASES.items():
                for i in range(count):
                    ET.SubElement(tests, "testcase", classname=name, name=str(i))
            ET.ElementTree(suite).write(cand/"junit.xml")
            ops.write(cand/"COMPLETED", dict(exit_code=0, acceptance=False, mode="build"))
            with mock.patch.object(b, "CAND", cand), \
                 mock.patch.object(b, "BUILD_RESULT", cand/"result.json"), \
                 mock.patch.object(b, "COMPILED_SOURCE", source), \
                 mock.patch.object(ops, "old_seal", return_value=manifest), \
                 mock.patch.object(ops, "approved_record", return_value={
                     "configure": list(ops.APPROVED_CONFIGURE)}):
                record = dict(configure=list(ops.APPROVED_CONFIGURE))
                command = ops.configure(record, source, cand)
                ops.write(cand/"configure-command.json", command)
                ops.write(cand/"build-install-command.json", [
                    str(ops.BASE/"bin/cmake"), "--build", str(cand/"build"), "--target", "install", "-j", "2"])
                pair = dict(core=ops.sha(package/core_name), isapol_python=b.python_digest(package/"driver"))
                value = dict(schema="q-omp-ops/1", mode="build", status="READY_BUILD_NOT_QUALIFIED",
                    acceptance=False, source=b.compiled_source(), root=str(cand),
                    allocation={"JobId": b.BUILD_JOB}, old_manifest_sha256=ops.OLD_MANIFEST,
                    old_runtime_unchanged=True, original_configure_sha256=ops.APPROVED_RECORD_SHA,
                    oracle_receipt_sha256=b.ORACLE_SHA, configure=command,
                    identity=dict(package=str(package/"__init__.py"), core=str(package/core_name),
                                  version="FAKE-NOT-NATIVE", implementation=pair),
                    tests=ops.junit_gate(cand/"junit.xml", False),
                    installed_python_checks=ops.installed_python_gate(source, cand/"build", package, []),
                    installed_science_sha256={str(p.relative_to(cand/"env")): ops.sha(p)
                        for p in package.rglob("*") if p.is_file() and p.suffix in (".py", ".so")})
                def publish():
                    for side in ("before", "after"):
                        put(cand/f"identity-{side}.stdout",
                            "non-JSON diagnostic prefix\nQOMP_IDENTITY="+json.dumps(value["identity"])+"\n")
                    put(cand/"result.json", json.dumps(value))
                    return ops.sha(cand/"result.json")
                yield types.SimpleNamespace(source=source, cand=cand, package=package, stage=stage,
                    inner=inner, value=value, publish=publish, sha=publish(), put=put, dependency=dependency)

    def test_normal84_build_uses_real_direct_identity_hashes_and_marked_stdout(self):
        with self.normal_build() as f:
            got = b.build_gate(f.sha)
            self.assertEqual(got, f.value)
            self.assertEqual(got["identity"]["implementation"]["core"],
                             ops.sha(Path(got["identity"]["core"])))
            self.assertEqual(got["tests"]["cases"], 84)
            self.assertFalse(hasattr(b, "CAND_ID"))
            self.assertFalse(hasattr(b, "validation"))
            for wrong in (None, "", "auto", "A"*64, "bad", "0"*64):
                with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                    b.build_gate(wrong)

    def test_stale32_82_failure_source_root_job_and_identity_mismatch_rejected(self):
        with self.normal_build() as f:
            original = copy.deepcopy(f.value)
            for path, value in (
                (("status",), "FAILED"), (("status",), "READY_DIAGNOSTIC_NOT_ACCEPTANCE"),
                (("mode",), "validate"), (("acceptance",), True),
                (("source", "commit"), "old-source"), (("source", "frozen_patch"), "old-SCI"),
                (("source", "path"), "/foreign/source"), (("source", "tree"), "other-tree"),
                (("root",), str(f.cand.parent/"other-root")), (("allocation", "JobId"), "old-job"),
                (("identity", "core"), str(f.cand.parent/"foreign/core.fake.so")),
                (("identity", "package"), str(f.cand.parent/"foreign/__init__.py")),
                (("identity", "implementation"), {"sha256": FAKE_PAIR}),
                (("identity", "implementation"), dict(ops.OLD_ID)),
            ):
                f.value.clear(); f.value.update(copy.deepcopy(original))
                cursor = f.value
                for key in path[:-1]:
                    cursor = cursor[key]
                cursor[path[-1]] = value
                with self.subTest(path=path, value=value), self.assertRaises((ValueError, KeyError)):
                    b.build_gate(f.publish())
            f.value.clear(); f.value.update(original)
            # Even a correctly digest-bound forged old82/32-Q receipt cannot
            # replace actual new84 JUnit. No stale hash/data is blessed.
            root = ET.parse(f.cand/"junit.xml").getroot()
            suite = root.find("testsuite")
            qcases = [c for c in suite if c.get("classname") == "test_isapol_q_parallel"]
            for case in qcases[-2:]:
                suite.remove(case)
            ET.ElementTree(root).write(f.cand/"junit.xml")
            f.value["tests"].update(cases=82, frozen_q_cases=32, sha256=ops.sha(f.cand/"junit.xml"))
            with self.assertRaises(ValueError):
                b.build_gate(f.publish())

    def test_marker_failure_actual_file_tamper_unknown_mapping_and_dependency_fail_closed(self):
        for kind in ("failed", "core", "python", "unknown", "dependency", "marker", "generated"):
            with self.subTest(kind=kind), self.normal_build() as f:
                if kind == "failed":
                    f.put(f.cand/"FAILED", "existing failure")
                elif kind == "core":
                    f.put(Path(f.value["identity"]["core"]), "tampered binary bytes")
                elif kind == "python":
                    f.put(f.package/"driver/procrouting/isapol_distribution.py", "tampered Python")
                elif kind == "unknown":
                    f.put(f.package/"unexpected.py", "unknown mapping, never imported")
                elif kind == "dependency":
                    f.put(f.dependency, "tampered dependency")
                elif kind == "marker":
                    f.put(f.cand/"identity-after.stdout", "prefix only, no identity marker")
                else:
                    f.put(f.inner/"metadata.py", "wrong configured output")
                with self.assertRaises(ValueError):
                    b.build_gate(f.sha)

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

    def test_cli_requires_digest_and_has_no_candidate_or_compiled_root_override(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            command = [sys.executable, "-I", "-B", str(HERE/"bench.py"), "run",
                "--source", str(HERE.parents[1]), "--commit", "not-run", "--tree", "not-run",
                "--root", str(root/"out"), "--scratch-root", str(root/"scratch"),
                "--launch-token", "synthetic"]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
            for flag in ("--candidate-root", "--compiled-source", "--build-result",
                         "--validation-result", "--evidence", "--candidate-identity"):
                with self.subTest(flag=flag):
                    result = subprocess.run(command + ["--build-result-sha256", "e"*64,
                        flag, "/forbidden-override"], capture_output=True)
                    self.assertEqual(result.returncode, 2)
            self.assertFalse((root/"out").exists())
            self.assertFalse((root/"scratch").exists())

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
                    root=base/"output", scratch_root=base/"scratch",
                    build_result_sha256="e"*64, affinity=json.dumps(list(range(8))))
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
                     mock.patch.object(b, "source_gate"), \
                     mock.patch.object(b, "build_gate", return_value={
                         "identity": {"implementation": FAKE_PAIR}}), \
                     mock.patch.object(b, "profile", return_value=p), \
                     mock.patch.object(sys, "pycache_prefix", str(args.scratch_root/"cache")), \
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
                fixtures=inputs, build_result_sha256="e"*64, commit="commit", tree="tree",
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
                     mock.patch.object(b, "build_gate", return_value={
                         "identity": {"implementation": FAKE_PAIR}}), \
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
                    self.assertEqual(result["build_status"], "READY_BUILD_NOT_QUALIFIED")
                    self.assertEqual(result["candidate_identity"], FAKE_PAIR)
                    self.assertEqual(result["build_result_sha256"], args.build_result_sha256)
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
for n in ("source","commit","tree","root","scratch-root","launch-token","build-result-sha256"): p.add_argument("--"+n)
p.add_argument("--exit-code",type=int); a=p.parse_args(); a.root=pathlib.Path(a.root)
s=importlib.util.spec_from_file_location("real",os.environ["REAL_BENCH"])
m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.source_gate=lambda *a:None
state=pathlib.Path(os.environ["STATE"])
if a.op=="run":
 a.root.mkdir();m.ops.write(a.root/"owner.json",dict(token=a.launch_token,mode="benchmark"))
 m.ops.write(a.root/"result.json",dict(status=m.SUCCESS,launch_token=a.launch_token,
 old_and_candidate_unchanged=True,fixed_operand_bits_equal=True,acceptance=False,adoption=False,
 build_result_sha256=a.build_result_sha256))
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
                        "--scratch-root", str(root/"scratch"), "--build-result-sha256", "e"*64], cwd=root,
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
