"""Pure fake-file/child tests. No Psi4 import, quantum, build or SLURM calls."""
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("q_validate", HERE/"validate.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
ops = v.ops


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class ValidationTests(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            source, candidate = base/"source003", base/"candidate"
            validator = base/"source004"
            validator.mkdir()
            args = types.SimpleNamespace(
                source=validator, commit="new-commit", tree="new-tree", compiled_source=source,
                candidate_root=candidate, evidence=base/"evidence/proof.json",
                evidence_sha256="", root=base/"validation", scratch_root=base/"scratch",
                launch_token="fresh-token", mode="validate", exit_code=0)
            inner = candidate/"build/psi4-core-prefix/src/psi4-core-build"
            package = candidate/"env"/v.PACKAGE
            staged = candidate/"build/stage"/v.PACKAGE
            files = {}
            for name in v.GENERATED:
                put(source/"psi4"/name, "raw stub @ONLY@")
                for parent in (inner, package, staged):
                    put(parent/name, "generated "+name)
                files[name] = ""
            names = ["procrouting/isapol_schema.py", "procrouting/isapol_distribution.py",
                     "procrouting/proc.py", "schema_wrapper.py", "driver.py",
                     "task_planner.py", "p4util/util.py"]
            for name in names:
                put(source/"psi4/driver"/name, "source "+name)
                put(package/"driver"/name, "source "+name)
                files["driver/"+name] = ""
            core_name = "core.cpython-fake.so"
            for path in (inner/"src"/core_name, package/core_name, staged/core_name):
                put(path, "never executed fake core bytes")
            put(candidate/"env/bin/python", "never executed fake Python")
            for cache, home, prefix in (
                (candidate/"build/CMakeCache.txt", source, candidate/"env"),
                (inner/"CMakeCache.txt", source/"psi4", candidate/"build/stage"),
            ):
                put(cache, "\n".join([
                    "CMAKE_HOME_DIRECTORY:INTERNAL="+str(home),
                    "CMAKE_INSTALL_PREFIX:PATH="+str(prefix),
                    "CMAKE_CXX_COMPILER:FILEPATH="+str(ops.BASE/"bin/x86_64-conda-linux-gnu-c++"),
                    "ENABLE_OPENMP:BOOL=ON", "ENABLE_XHOST:BOOL=OFF",
                    "CMAKE_CXX_FLAGS:STRING=-march=x86-64-v3"]))
            command = ops.configure(dict(configure=list(ops.APPROVED_CONFIGURE)), source, candidate)
            result = dict(status="FAILED", mode="build", acceptance=False,
                          source=v.compiled_source(args), root=str(candidate),
                          old_manifest_sha256=ops.OLD_MANIFEST,
                          original_configure_sha256=ops.APPROVED_RECORD_SHA,
                          oracle_receipt_sha256=v.ORACLE_SHA, configure=command,
                          old_runtime_unchanged=True,
                          exception=dict(type="ValueError",
                                         message="candidate Python install differs: metadata.py"))
            for name, value in {
                "result.json": result, "FAILED": dict(exit_code=1, mode="build", acceptance=False),
                "configure-command.json": command,
                "build-install-command.json": [str(ops.BASE/"bin/cmake"), "--build",
                                               str(candidate/"build"), "--target", "install", "-j", "2"],
            }.items():
                ops.write(candidate/name, value)
            for name in ("configure.stdout", "configure.stderr", "build-install.stdout",
                         "build-install.stderr"):
                put(candidate/name, "")
            trace = base/"historical-logs/build.err"
            put(trace, f'File "{source}/devtools/isapol_q_omp/ops.py", in installed_python_gate\n'
                       "ValueError: candidate Python install differs: metadata.py\n")
            proof = dict(
                schema="q-omp-compiled-evidence/1", candidate_root=str(candidate),
                compiled_source=v.compiled_source(args),
                historical_job=dict(id="15280334", status="FAILED", exit_code="1:0"),
                command_completion_basis=v.BASIS,
                artifact_sha256={n: ops.sha(candidate/n) for n in v.ARTIFACTS},
                traceback=dict(path=str(trace), sha256=ops.sha(trace)),
                core=dict(env_relative=str(v.PACKAGE/core_name),
                          inner_build_relative="src/"+core_name, sha256=ops.sha(package/core_name)))
            args.evidence.parent.mkdir()
            ops.write(args.evidence, proof)
            args.evidence_sha256 = ops.sha(args.evidence)
            manifest = dict(runtime_files={}, external_runtime_files={}, installed_files=files)
            with mock.patch.object(v, "CANDIDATE", candidate), \
                 mock.patch.object(v, "KNOWN_ARTIFACTS", proof["artifact_sha256"]), \
                 mock.patch.object(v, "METADATA_SHA", ops.sha(inner/"metadata.py")), \
                 mock.patch.object(ops, "approved_record",
                                   return_value=dict(configure=list(ops.APPROVED_CONFIGURE))), \
                 mock.patch.object(ops, "source_gate") as gate:
                yield args, proof, manifest, gate

    def test_historical_argv_lists_and_failed_status_remain_honest(self):
        with self.fixture() as (args, proof, manifest, gate):
            self.assertEqual(v.evidence_gate(args), proof)
            self.assertIsInstance(ops.read(args.candidate_root/"configure-command.json"), list)
            self.assertIn("not_stored_exit_codes", proof["command_completion_basis"])
            put(args.candidate_root/"build-install.stdout", "changed old evidence")
            with self.assertRaisesRegex(ValueError, "historical artifact changed"):
                v.evidence_gate(args)

    def test_parent_evidence_digest_and_original_failure_are_mandatory(self):
        with self.fixture() as (args, proof, manifest, gate):
            args.evidence_sha256 = "0"*64
            with self.assertRaisesRegex(ValueError, "evidence digest"):
                v.evidence_gate(args)
            args.evidence_sha256 = ops.sha(args.evidence)
            put(args.candidate_root/"COMPLETED", "forbidden relabel")
            with self.assertRaisesRegex(ValueError, "relabeled"):
                v.evidence_gate(args)

    def test_strict_generated_core_source_and_cmake_checks(self):
        with self.fixture() as (args, proof, manifest, gate):
            checks, core = v.candidate_gate(args, proof, manifest)
            self.assertEqual(checks["metadata.py"]["kind"], "configured")
            for path, message in (
                (args.candidate_root/"env"/v.PACKAGE/"metadata.py", "install differs"),
                (args.candidate_root/"env"/v.PACKAGE/"driver/procrouting/isapol_distribution.py",
                 "install differs"),
                (core, "core differs"),
                (args.candidate_root/"build/CMakeCache.txt", "cache source"),
            ):
                original = path.read_text()
                try:
                    # For CMake retain well-formed keys, but point to another source.
                    put(path, original.replace(str(args.compiled_source), "/wrong-source")
                        if path.name == "CMakeCache.txt" else "corrupt bytes")
                    with self.assertRaisesRegex(ValueError, message):
                        v.candidate_gate(args, proof, manifest)
                finally:
                    put(path, original)

    def test_snapshot_covers_whole_historical_tree_not_only_prefix(self):
        with self.fixture() as (args, proof, manifest, gate):
            before = v.tree_snapshot(args.candidate_root, lambda: None)
            put(args.candidate_root/"build/object.o", "new unexpected artifact")
            self.assertNotEqual(before, v.tree_snapshot(args.candidate_root, lambda: None))
            with self.assertRaisesRegex(ValueError, "unsafe artifact"):
                v.owned_file(args.candidate_root, "../outside")

    def fake_children(self, args, proof, fail):
        """Only stdlib fake output. Actual pre-import/CWD/controller path runs."""
        test = self
        calls = []

        def run(runner, label, command):
            calls.append(label)
            test.assertEqual(Path.cwd(), args.root/"cwd")
            test.assertEqual(runner.cwd, args.root/"cwd")
            test.assertIn("-B", command)
            test.assertIn("-I", command)
            test.assertEqual(str(command[0]), str(args.candidate_root/"env/bin/python"))
            # Simulate CWD-relative core finalization, never import it.
            Path("timer.dat").write_text("new owned CWD only")
            Path("psi.fake.clean").write_text("new owned CWD only")
            if fail == label:
                raise RuntimeError("fake import exception")
            if label.startswith("identity-"):
                return "QOMP_IDENTITY="+json.dumps(dict(
                    package=str(args.candidate_root/"env"/v.PACKAGE/"__init__.py"),
                    core=str(args.candidate_root/"env"/proof["core"]["env_relative"]),
                    version="fake only",
                    implementation=dict(core=proof["core"]["sha256"],
                                        isapol_python=v.source_python_sha(args.compiled_source))))
            test.assertEqual(label, "native-tests")
            test.assertTrue(str(command[-3]).endswith(ops.TEST))
            root = ET.Element("testsuites")
            suite = ET.SubElement(root, "testsuite")
            for name, count in (("test_isapol_q_parallel", 32),
                                ("test_isapol_native_performance", 1),
                                ("test_isapol_partitioned_response", 1)):
                for i in range(count):
                    ET.SubElement(suite, "testcase", classname=name, name=str(i))
            ET.ElementTree(root).write(args.root/"junit.xml")
            return ""
        return run, calls

    def exercise(self, fail=None):
        with self.fixture() as (args, proof, manifest, gate):
            sentinel = args.root.parent/"protected-cwd"
            sentinel.mkdir()
            put(sentinel/"timer.dat", "sentinel bytes")
            original_cwd = Path.cwd()
            os.chdir(sentinel)
            before = v.tree_snapshot(args.candidate_root, lambda: None)
            run, calls = self.fake_children(args, proof, fail)
            try:
                with mock.patch.dict(os.environ, SLURM_JOB_ID="fake", SLURM_CPUS_PER_TASK="8"), \
                     mock.patch.object(ops, "allocation", return_value={"fake": True}), \
                     mock.patch.object(ops, "old_seal", return_value=manifest) as seal, \
                     mock.patch.object(ops.Runner, "run", run):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "fake import"):
                            v.perform(args)
                    else:
                        v.perform(args)
                    self.assertEqual(seal.call_count, 2)
                self.assertEqual(before, v.tree_snapshot(args.candidate_root, lambda: None))
                self.assertEqual((sentinel/"timer.dat").read_text(), "sentinel bytes")
                self.assertFalse((sentinel/"psi.fake.clean").exists())
                result = ops.read(args.root/"result.json")
                self.assertFalse(result["acceptance"])
                self.assertFalse(result["adoption"])
                self.assertEqual(result["historical_build"]["status"], "FAILED")
                self.assertTrue(result["candidate_prefix_unchanged"])
                self.assertTrue(result["compiled_tree_unchanged"])
                self.assertTrue(result["old_runtime_prechecked"])
                self.assertTrue(result["old_runtime_unchanged"])
                self.assertEqual(result["status"], "FAILED" if fail else "READY_DIAGNOSTIC_NOT_ACCEPTANCE")
                if not fail:
                    self.assertEqual(result["tests"]["frozen_q_cases"], 32)
                    self.assertEqual(calls, ["identity-before", "native-tests", "identity-after"])
                args.exit_code = 1 if fail else 0
                self.assertEqual(ops.finalize(args), args.exit_code)
                self.assertEqual((args.root/"COMPLETED").exists(), not fail)
                self.assertTrue((args.candidate_root/"FAILED").exists())
                self.assertFalse((args.candidate_root/"COMPLETED").exists())
            finally:
                os.chdir(original_cwd)

    def test_fresh_validation_preimport_cwd_and_readonly_native_command_path(self):
        self.exercise()

    def test_import_exception_retains_failed_diagnostic_and_original_evidence(self):
        self.exercise("identity-before")

    def test_native_child_failure_cannot_publish_diagnostic_success(self):
        self.exercise("native-tests")

    def test_candidate_mutation_is_failure_and_still_postchecks_qualified_old(self):
        with self.fixture() as (args, proof, manifest, gate):
            original_cwd = Path.cwd()
            run, calls = self.fake_children(args, proof, None)
            def tamper(runner, label, command):
                output = run(runner, label, command)
                if label == "identity-after":
                    put(args.candidate_root/"env/unexpected-file", "fake write must be detected")
                return output
            try:
                with mock.patch.dict(os.environ, SLURM_JOB_ID="fake", SLURM_CPUS_PER_TASK="8"), \
                     mock.patch.object(ops, "allocation", return_value={"fake": True}), \
                     mock.patch.object(ops, "old_seal", return_value=manifest) as seal, \
                     mock.patch.object(ops.Runner, "run", tamper):
                    with self.assertRaisesRegex(ValueError, "compiled tree changed"):
                        v.perform(args)
                    self.assertEqual(seal.call_count, 2)
                result = ops.read(args.root/"result.json")
                self.assertEqual(result["status"], "FAILED")
                self.assertTrue(result["old_runtime_unchanged"])
                self.assertNotIn("compiled_tree_unchanged", result)
                self.assertEqual(ops.finalize(args), 1)
                self.assertFalse((args.root/"COMPLETED").exists())
                self.assertTrue((args.candidate_root/"FAILED").exists())
            finally:
                os.chdir(original_cwd)

    def test_no_build_install_or_clone_in_readonly_action(self):
        text = (HERE/"validate.py").read_text()
        self.assertNotIn("clone_runtime(", text)
        self.assertNotIn('runner.run("configure"', text)
        self.assertNotIn('runner.run("build-install"', text)
        self.assertIn("inferred_command_exit_codes", text)
        self.assertNotIn("stored_command_exit_codes", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
