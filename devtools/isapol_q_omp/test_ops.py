"""Pure ops metadata tests, no Psi4, real SLURM, compiler or native test calls."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
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
