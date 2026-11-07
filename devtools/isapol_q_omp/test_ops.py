"""Pure ops metadata tests, no Psi4, real SLURM, compiler or native test calls."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]
spec = importlib.util.spec_from_file_location("q_ops", HERE / "ops.py")
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


class OpsTests(unittest.TestCase):
    def finalization_window(self, wrapper_name, phase, sig, original=False):
        """Actual Bash finish + real ops.finalize, fake completed operation only."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root/"source"
            helper_dir = source/"devtools/isapol_q_omp"
            helper_dir.mkdir(parents=True)
            fake_srun = root/"fake-srun"
            fake_srun.write_text("""#!/usr/bin/python3
import os,sys
args=sys.argv[1:]
while args and args[0].startswith("--"): args.pop(0)
os.execv(args[0],args)
""")
            fake_srun.chmod(0o700)
            # This shim uses the current real finalizer, but never imports Psi4,
            # compiles, runs pytest, or queries an allocation.
            (helper_dir/"ops.py").write_text("""
import argparse,importlib.util,json,os,pathlib,signal,sys,time
p=argparse.ArgumentParser()
p.add_argument("operation")
for name in ("source","commit","tree","root","scratch-root","launch-token"):
    p.add_argument("--"+name,required=True)
p.add_argument("--exit-code",type=int)
a=p.parse_args();a.root=pathlib.Path(a.root);a.source=pathlib.Path(a.source)
a.mode=a.operation.split("-")[0]
s=importlib.util.spec_from_file_location("real_ops",os.environ["FINALIZE_REAL_OPS"])
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.source_gate=lambda *unused:None
state=pathlib.Path(os.environ["FINALIZE_STATE"])
if not a.operation.endswith("-finalize"):
    a.root.mkdir()
    m.write(a.root/"owner.json",dict(token=a.launch_token,mode=a.mode))
    m.write(a.root/"result.json",dict(
        status="READY_ORACLE_NOT_ACCEPTANCE" if a.mode=="oracle" else "READY_BUILD_NOT_QUALIFIED",
        old_runtime_unchanged=True,launch_token=a.launch_token))
    sys.exit(0)
if a.exit_code==0:
    def term(*unused):
        (state/"FORWARDED_TERM").write_text("finalizer received TERM")
        sys.exit(143)
    signal.signal(signal.SIGTERM,term)
    if os.environ["FINALIZE_PHASE"]=="after_publish": m.finalize(a)
    (state/"FINALIZING").write_text(str(os.getpid()))
    while not (state/"RELEASE").exists(): time.sleep(.01)
sys.exit(m.finalize(a))
""")
            text = (HERE/wrapper_name).read_text()
            if original:
                text = subprocess.check_output(
                    ["git", "-C", str(SOURCE), "show",
                     "b0f9c0c5d6ac54eb159cfb9df7a312f99b79ed6e:devtools/isapol_q_omp/"+wrapper_name],
                    text=True)
            wrapper = root/"wrapper.sbatch"
            wrapper.write_text(text.replace("/opt/slurm/current/bin/srun", str(fake_srun)))
            out = root/"output"
            env = dict(os.environ, FINALIZE_REAL_OPS=str(HERE/"ops.py"),
                       FINALIZE_STATE=str(root), FINALIZE_PHASE=phase)
            command = ["bash", str(wrapper), "--source", str(source), "--commit", "0"*40,
                       "--tree", "1"*40, "--root", str(out), "--scratch-root", str(root/"scratch")]
            process = subprocess.Popen(command, env=env, cwd=root, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       start_new_session=True)
            try:
                deadline = time.monotonic()+8
                while not (root/"FINALIZING").exists() and time.monotonic() < deadline:
                    if process.poll() is not None:
                        break
                    time.sleep(.01)
                self.assertTrue((root/"FINALIZING").exists(), "did not reach finalization window")
                process.send_signal(sig)
                if original:
                    # Deterministically release the ignored signal's finalizer:
                    # b0 then publishes COMPLETED/0, reproducing the blocker.
                    (root/"RELEASE").write_text("release original ignored-signal finalizer")
                else:
                    deadline = time.monotonic()+5
                    while not (root/"FORWARDED_TERM").exists() and time.monotonic() < deadline:
                        if process.poll() is not None:
                            break
                        time.sleep(.01)
                    self.assertTrue((root/"FORWARDED_TERM").exists(),
                                    "TERM/INT was not forwarded during FINALIZATION")
                stdout, stderr = process.communicate(timeout=8)
                if original:
                    self.assertEqual(process.returncode, 0, stderr)
                    self.assertTrue((out/"COMPLETED").exists())
                    self.assertFalse((out/"FAILED").exists())
                else:
                    self.assertEqual(process.returncode, 143, stderr)
                    self.assertFalse((out/"COMPLETED").exists())
                    self.assertEqual(ops.read(out/"FAILED")["exit_code"], 143)
                    if phase == "after_publish":
                        self.assertEqual(ops.read(out/"COMPLETED_INTERRUPTED")["exit_code"], 0)
            finally:
                (root/"RELEASE").touch()
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)

    def test_term_and_int_during_finalization_never_publish_success(self):
        for wrapper in ("oracle.sbatch", "build.sbatch"):
            for phase in ("before_publish", "after_publish"):
                for sig in (signal.SIGTERM, signal.SIGINT):
                    with self.subTest(wrapper=wrapper, phase=phase, signal=sig):
                        self.finalization_window(wrapper, phase, sig)

    def test_regression_reproduces_ignored_finalization_term_on_b0(self):
        for wrapper in ("oracle.sbatch", "build.sbatch"):
            with self.subTest(wrapper=wrapper):
                self.finalization_window(wrapper, "before_publish", signal.SIGTERM, original=True)

    def test_five_reviewed_files_are_still_frozen(self):
        for name, digest in ops.FROZEN.items():
            self.assertEqual(ops.sha(SOURCE / name), digest, name)

    def test_finite_wrappers_and_no_site_stack_mutation(self):
        for name, duration in (("oracle.sbatch", "02:00:00"), ("build.sbatch", "08:00:00")):
            path = HERE / name
            subprocess.run(["bash", "-n", str(path)], check=True)
            text = path.read_text()
            for flag in ("--qos=embers", "--cpus-per-task=8", "--mem=32G", "--no-requeue",
                         "--nodes=1", "--ntasks=1", "--time="+duration):
                self.assertIn("#SBATCH "+flag, text)
            self.assertIn('trap finish EXIT', text)
            self.assertIn('wait "$child"', text)
        env = ops.environment(Path("/candidate/env"), Path("/candidate/scratch"))
        self.assertNotIn("GCC_ROOT", env)
        self.assertNotIn("CPATH", env)
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")
        self.assertEqual(env["PIP_NO_INDEX"], "1")

    def test_fresh_paths_reject_existing_and_protected_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            protected = root / "old"
            protected.mkdir()
            ops.fresh_paths(root/"new", root/"scratch", [protected])
            for a, b in ((protected, root/"s"), (protected/"new", root/"s"),
                         (root/"new", root/"new/sub")):
                with self.assertRaises(ValueError):
                    ops.fresh_paths(a, b, [protected])

    def test_atomic_metadata_never_overwrites_existing_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "record"
            ops.write(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                ops.write(path, {"value": 2})
            self.assertEqual(ops.read(path), {"value": 1})

    def test_offline_clone_retargets_old_links_without_old_writes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            old, base, new = root/"old", root/"base", root/"new"
            old.mkdir()
            base.mkdir()
            (old/"data").write_text("original bytes")
            (old/"absolute").symlink_to(old/"data")
            (old/"relative").symlink_to("data")
            (base/"python").write_text("same dependency bytes")
            (old/"python").symlink_to(base/"python")
            (old/"__pycache__").mkdir()
            (old/"__pycache__/unused.pyc").write_bytes(b"not executable evidence")
            before = ops.sha(old/"data")
            records = ops.clone_runtime(old, new, base)
            self.assertEqual((new/"absolute").resolve(), new/"data")
            self.assertEqual((new/"relative").resolve(), new/"data")
            self.assertFalse((new/"python").is_symlink())
            self.assertEqual(ops.sha(new/"python"), ops.sha(base/"python"))
            self.assertFalse((new/"__pycache__").exists())
            self.assertEqual(ops.sha(old/"data"), before)
            self.assertTrue((old/"absolute").is_symlink())
            self.assertEqual(len(records), 2)

    def test_external_directory_links_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            old, base = root/"old", root/"base"
            old.mkdir()
            base.mkdir()
            (old/"bad").symlink_to(base, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "external directory"):
                ops.clone_runtime(old, root/"new", base)

    def test_configure_replays_original_stack_into_new_prefix(self):
        flags = ["-DCMAKE_BUILD_TYPE=Release", "-DENABLE_OPENMP=ON", "-DENABLE_XHOST=OFF",
                 "-DBLAS_TYPE=MKL", "-DBUILD_Libint2_GENERATOR=OFF"]
        flags += ["-DCMAKE_INSIST_FIND_PACKAGE_"+p+"=ON" for p in
                  ("pybind11", "gau2grid", "Libint2", "Libxc", "qcelemental", "qcengine", "optking")]
        record = dict(configure=[str(ops.BASE/"bin/cmake"), "-S", str(ops.OLD/"source"),
            "-B", str(ops.OLD/"build"), "-DCMAKE_INSTALL_PREFIX="+str(ops.OLD/"env"),
            "-DPython_EXECUTABLE="+str(ops.OLD/"env/bin/python"),
            "-DCMAKE_PREFIX_PATH="+str(ops.BASE), *flags])
        result = ops.configure(record, Path("/candidate/source"), Path("/candidate/root"))
        self.assertIn("-DCMAKE_INSTALL_PREFIX=/candidate/root/env", result)
        self.assertIn("-DPython_EXECUTABLE=/candidate/root/env/bin/python", result)
        self.assertIn("-DCMAKE_PREFIX_PATH="+str(ops.BASE), result)
        self.assertIn("-DFETCHCONTENT_FULLY_DISCONNECTED=ON", result)
        self.assertEqual(record["configure"][2], str(ops.OLD/"source"))
        broken = dict(configure=[v for v in record["configure"] if v != "-DBLAS_TYPE=MKL"])
        with self.assertRaises(ValueError):
            ops.configure(broken, Path("/source"), Path("/new"))

    def junit(self, path, count=32, skipped=False):
        root = ET.Element("testsuites")
        suite = ET.SubElement(root, "testsuite")
        for i in range(count):
            case = ET.SubElement(suite, "testcase", classname="test_isapol_q_parallel", name=str(i))
            if skipped and i == 0:
                ET.SubElement(case, "skipped")
        ET.ElementTree(root).write(path)

    def test_exact_oracle_case_count_and_no_skips(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"junit.xml"
            self.junit(path)
            self.assertEqual(ops.junit_gate(path, True)["frozen_q_cases"], 32)
            for count, skip in ((31, False), (33, False), (32, True)):
                self.junit(path, count, skip)
                with self.assertRaises(ValueError):
                    ops.junit_gate(path, True)
        command = ops.test_command(Path("/old/env/bin/python"), Path("/source"), Path("/out"), Path("/scr"), True)
        self.assertEqual(sum(str(a).endswith(".py") for a in command), 1)
        self.assertTrue(str(command[-1]).endswith(ops.TEST))

    def test_finalize_never_marks_failed_or_foreign_root_completed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args = types.SimpleNamespace(root=root, launch_token="ours", mode="oracle", exit_code=0,
                                         source=Path("/source"), commit="commit", tree="tree")
            ops.write(root/"owner.json", dict(token="other", mode="oracle"))
            self.assertNotEqual(ops.finalize(args), 0)
            self.assertFalse((root/"COMPLETED").exists())
            self.assertFalse((root/"FAILED").exists())
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args.root = root
            ops.write(root/"owner.json", dict(token="ours", mode="oracle"))
            ops.write(root/"result.json", dict(status="READY_ORACLE_NOT_ACCEPTANCE",
                                              old_runtime_unchanged=True, launch_token="ours"))
            with mock.patch.object(ops, "source_gate"):
                self.assertEqual(ops.finalize(args), 0)
            self.assertFalse(ops.read(root/"COMPLETED")["acceptance"])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args.root, args.exit_code = root, 143
            ops.write(root/"owner.json", dict(token="ours", mode="oracle"))
            self.assertEqual(ops.finalize(args), 143)
            self.assertFalse((root/"COMPLETED").exists())
            self.assertEqual(ops.read(root/"FAILED")["exit_code"], 143)

    def test_plain_child_term_cannot_be_reported_as_success(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runner = ops.Runner(root, root, dict(os.environ), 10)
            timer = threading.Timer(.2, runner.stop)
            timer.start()
            try:
                with self.assertRaises(InterruptedError):
                    runner.run("fake-child", [sys.executable, "-I", "-B", "-c",
                                              "import time; time.sleep(30)"])
            finally:
                timer.cancel()
                timer.join()
            self.assertTrue(runner.interrupted)
            self.assertIsNone(runner.child)

    def test_oracle_build_gate_requires_wrapper_completion_and_bound_digest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root/"result.json"
            source = dict(path="/source", commit="commit", tree="tree", frozen_patch=ops.PATCH)
            value = dict(status="READY_ORACLE_NOT_ACCEPTANCE", source=source,
                         old_manifest_sha256=ops.OLD_MANIFEST, old_runtime_unchanged=True,
                         identity={"implementation": ops.OLD_ID},
                         tests=dict(cases=32, frozen_q_cases=32, failures=0, errors=0, skipped=0))
            ops.write(path, value)
            with self.assertRaises(FileNotFoundError):
                ops.oracle_gate(path, ops.sha(path), source)
            ops.write(root/"COMPLETED", dict(exit_code=0, acceptance=False, mode="oracle"))
            self.assertEqual(ops.oracle_gate(path, ops.sha(path), source), value)
            with self.assertRaises(ValueError):
                ops.oracle_gate(path, "0"*64, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
