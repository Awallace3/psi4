"""Pure ops metadata tests, no Psi4, real SLURM, compiler or native test calls."""
import importlib.util
import ast
import hashlib
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
        status={"oracle":"READY_ORACLE_NOT_ACCEPTANCE","build":"READY_BUILD_NOT_QUALIFIED"}[a.mode],
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

    def test_source_gate_binds_731_and_only_five_reviewed_ops_paths(self):
        self.assertEqual(ops.PATCH, "731fb3d52f3c301b28b22fae469e0deb3f7e805c")
        expected = {"devtools/isapol_q_omp/"+name for name in
                    ("ops.py", "oracle.sbatch", "build.sbatch", "test_ops.py", "OPS.md")}
        self.assertEqual(ops.OPS, expected)
        source, commit, tree = Path("/source"), "ops-commit", "ops-tree"
        answers = {
            ("rev-parse", "HEAD"): commit, ("rev-parse", "HEAD^{tree}"): tree,
            ("rev-parse", "--show-toplevel"): str(source),
            ("merge-base", ops.PATCH, commit): ops.PATCH,
            ("status", "--porcelain", "--untracked-files=all"): "",
            ("diff", "--name-only", ops.PATCH, commit): "\n".join(sorted(expected)),
        }
        with mock.patch.object(ops, "git", side_effect=lambda unused, *a: answers[a]), \
             mock.patch.object(ops, "sha", side_effect=lambda p: ops.FROZEN[str(p.relative_to(source))]):
            ops.source_gate(source, commit, tree)
            for bad in ("psi4/src/psi4/libisapol/partitioned_response.cc",
                        "forecast_resources.py", "devtools/isapol_q_omp/validate.py"):
                answers[("diff", "--name-only", ops.PATCH, commit)] = bad
                with self.assertRaisesRegex(ValueError, "outside the allowed"):
                    ops.source_gate(source, commit, tree)
            answers[("diff", "--name-only", ops.PATCH, commit)] = ""
            answers[("merge-base", ops.PATCH, commit)] = "unrelated-SCI"
            with self.assertRaisesRegex(ValueError, "descend from reviewed SCI"):
                ops.source_gate(source, commit, tree)
        for name in ("validate.py", "validate.sbatch", "test_validate.py"):
            self.assertFalse((HERE/name).exists())

    def test_native_q_count_deliberately_matches_new_source_not_old32(self):
        tree = ast.parse((SOURCE/ops.TEST).read_text())  # Never import Psi4/test module.
        total = 0
        for function in tree.body:
            if not isinstance(function, ast.FunctionDef) or not function.name.startswith("test_"):
                continue
            count = 1
            for decorator in function.decorator_list:
                if not (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                        and decorator.func.attr == "parametrize"):
                    continue
                values = decorator.args[1]
                if isinstance(values, (ast.List, ast.Tuple)):
                    count *= len(values.elts)
                else:
                    self.assertIsInstance(values, ast.Call)
                    self.assertIsInstance(values.func, ast.Name)
                    self.assertEqual(values.func.id, "range")
                    count *= len(range(*(ast.literal_eval(a) for a in values.args)))
            total += count
        self.assertEqual(total, 34)
        self.assertEqual(ops.NATIVE_CASES, dict(test_isapol_q_parallel=34,
                         test_isapol_native_performance=18, test_isapol_partitioned_response=32))
        self.assertEqual(sum(ops.NATIVE_CASES.values()), 84)

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
        record = dict(configure=list(ops.APPROVED_CONFIGURE))
        canonical = json.dumps(record["configure"], separators=(",", ":"), ensure_ascii=True).encode()
        # Independently recorded from the transferred, parent-approved record.
        self.assertEqual(hashlib.sha256(canonical).hexdigest(),
                         "04478ce47a99f0399bbd414b0ede79e018757c0ec62444918c7a7a3d28aaf3fe")
        result = ops.configure(record, Path("/candidate/source"), Path("/candidate/root"))
        self.assertIn("-DCMAKE_INSTALL_PREFIX=/candidate/root/env", result)
        self.assertIn("-DPython_EXECUTABLE=/candidate/root/env/bin/python", result)
        self.assertIn("-DCMAKE_PREFIX_PATH="+str(ops.BASE), result)
        self.assertIn("-DPYMOD_INSTALL_LIBDIR=/python3.13/site-packages", result)
        self.assertEqual(len(result), len(record["configure"]))
        # Every other byte must be preserved; no added flags or stack changes.
        expected = [arg.replace(str(ops.OLD/"env"), "/candidate/root/env")
                    for arg in record["configure"]]
        expected[2], expected[4] = "/candidate/source", "/candidate/root/build"
        self.assertEqual(result, expected)
        self.assertEqual(record["configure"][2], str(ops.OLD/"source"))

    def test_configure_rejects_unapproved_overrides_and_even_duplicate_equal_flags(self):
        original = list(ops.APPROVED_CONFIGURE)
        mutations = [original + extra for extra in (
            ["-DCMAKE_BUILD_TYPE=Release"], ["-DCMAKE_INSTALL_PREFIX=/old/install"],
            ["-C", "/preload.cmake"], ["-C/preload.cmake"],
            ["--toolchain", "/toolchain.cmake"], ["-DCMAKE_TOOLCHAIN_FILE=/toolchain.cmake"],
            ["-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=/injected.cmake"],
            ["-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"])]
        mutations += [[a.replace(str(ops.BASE), "/other-stack") for a in original],
                      original[:-1]]
        for args in mutations:
            with self.subTest(args=args):
                with self.assertRaisesRegex(ValueError, "unapproved"):
                    ops.configure(dict(configure=args), Path("/source"), Path("/new"))

    def test_build_record_requires_parent_pinned_raw_bytes_not_just_safe_argv(self):
        self.assertEqual(ops.APPROVED_RECORD_BYTES, 4356)
        self.assertEqual(ops.APPROVED_RECORD_SHA,
                         "896949860e3eb012e07b2ec45556ab4580d87e36de094ca5567d800149803380")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"build-record.json"
            raw = json.dumps(dict(configure=list(ops.APPROVED_CONFIGURE))).encode()
            # Same argv with different record/provenance/serialization is NOT
            # authorization. Also reject forged records of the expected size.
            for data in (raw, b" "*4356):
                path.write_bytes(data)
                with self.assertRaisesRegex(ValueError, "parent-approved bytes"):
                    ops.approved_record(path)

    def test_inherited_install_toolchain_and_site_flags_are_removed(self):
        poisoned = dict(DESTDIR="/old", CMAKE_INSTALL_MODE="SYMLINK",
                        CMAKE_TOOLCHAIN_FILE="/toolchain", CMAKE_ARGS="-C /preload",
                        CMAKE_INSTALL_PREFIX="/old", CPATH="/site", GCC_ROOT="/site",
                        LD_PRELOAD="/inject", LD_AUDIT="/inject", CC="/site/cc",
                        CXXFLAGS="-include /site/header", MAKEFLAGS="DESTDIR=/old",
                        PYTHONPATH="/site", PYTEST_ADDOPTS="--skip-all")
        with mock.patch.dict(os.environ, poisoned):
            env = ops.environment(Path("/candidate/env"), Path("/candidate/scratch"))
        for key in poisoned:
            self.assertNotIn(key, env)
        self.assertEqual(env["PATH"], f"/candidate/env/bin:{ops.BASE}/bin:/usr/bin:/bin")
        self.assertEqual(env["LD_LIBRARY_PATH"], f"/candidate/env/lib:{ops.BASE}/lib")

    def test_installed_templates_use_configured_outputs_and_isapol_stays_strict(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, build, package = root/"source", root/"build", root/"env/psi4"
            inner = build/"psi4-core-prefix/src/psi4-core-build"
            inner.mkdir(parents=True)
            package.mkdir(parents=True)
            (source/"psi4/driver/procrouting").mkdir(parents=True)
            (package/"driver/procrouting").mkdir(parents=True)
            for name in ("__init__.py", "extras.py", "metadata.py"):
                (source/"psi4"/name).write_text("unconfigured @TEMPLATE@")
                (inner/name).write_text("known configured bytes")
                (package/name).write_text("known configured bytes")
            name = "driver/procrouting/isapol_distribution.py"
            (source/"psi4"/name).write_text("committed isapol bytes")
            (package/name).write_text("committed isapol bytes")
            names = ["__init__.py", "extras.py", "metadata.py", name]
            checks = ops.installed_python_gate(source, build, package, names)
            self.assertEqual(checks["extras.py"]["kind"], "configured")
            self.assertEqual(checks["metadata.py"]["kind"], "configured")
            self.assertEqual(checks[name]["kind"], "source")
            (package/name).write_text("wrong installed isapol bytes")
            with self.assertRaisesRegex(ValueError, "install differs"):
                ops.installed_python_gate(source, build, package, names)
            (package/name).write_text("committed isapol bytes")
            (source/"psi4"/name).unlink()
            with self.assertRaisesRegex(ValueError, "missing isapol source"):
                ops.installed_python_gate(source, build, package, names)
            (source/"psi4"/name).write_text("committed isapol bytes")
            (inner/"metadata.py").unlink()
            with self.assertRaisesRegex(ValueError, "configured build output"):
                ops.installed_python_gate(source, build, package, names)

    def test_every_cmake_python_mapping_class_including_installed_tests(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, build, package = root/"source", root/"build", root/"package"
            inner = build/"psi4-core-prefix/src/psi4-core-build"
            cases = {
                "__init__.py": ("configured", inner/"__init__.py"),
                "extras.py": ("configured", inner/"extras.py"),
                "metadata.py": ("configured", inner/"metadata.py"),
                "header.py": ("source", source/"psi4/header.py"),
                "driver/procrouting/isapol_distribution.py":
                    ("source", source/"psi4/driver/procrouting/isapol_distribution.py"),
            }
            for name in ("isapol_factorial_oracle.py", "test_isapol_q_parallel.py", "__init__.py",
                         "addons.py", "conftest.py", "standard_suite_runner.py",
                         "standard_suite_ref_local.py", "utils.py", "data_isapol/fixture.py",
                         "test_fchk_writer/helper.py", "test_gauxc_writer/helper.py",
                         "test_molden_writer/helper.py", "test_psi4_qcschema/helper.py"):
                cases["tests/"+name] = ("pytest-source", source/"tests/pytests"/name)
            for name in ("scf/test_input.py", "scf/input.py", "psithon2/psiaux1/helper.py",
                         "psithon2/psiaux1/myplugin1/helper.py", "psithon2/psiaux2/helper.py",
                         "dftd3/psithon2/psiaux1/helper.py",
                         "dftd3/psithon2/psiaux1/myplugin1/helper.py",
                         "dftd3/psithon2/psiaux2/helper.py"):
                cases["tests/"+name] = ("sample-test-source", source/"tests"/name)
            for name, (kind, expected) in cases.items():
                expected.parent.mkdir(parents=True, exist_ok=True)
                expected.write_text("expected "+name)
                installed = package/name
                installed.parent.mkdir(parents=True, exist_ok=True)
                installed.write_text("expected "+name)
            # Empty manifest still audits EVERY installed Python entry.
            checks = ops.installed_python_gate(source, build, package, [])
            self.assertEqual(set(checks), set(cases))
            for name, (kind, expected) in cases.items():
                self.assertEqual(checks[name]["kind"], kind)
                self.assertEqual(checks[name]["expected"], str(expected))

    def test_installed_test_mapping_missing_and_tamper_fail_not_skip(self):
        for name, original in (
            ("tests/isapol_factorial_oracle.py", "tests/pytests/isapol_factorial_oracle.py"),
            ("tests/data_isapol/fixture.py", "tests/pytests/data_isapol/fixture.py"),
            ("tests/scf/test_input.py", "tests/scf/test_input.py"),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                source, build, package = root/"source", root/"build", root/"package"
                inner = build/"psi4-core-prefix/src/psi4-core-build"
                for parent in (inner, package):
                    parent.mkdir(parents=True)
                    for generated in ("__init__.py", "extras.py", "metadata.py"):
                        (parent/generated).write_text("generated")
                expected = source/original
                expected.parent.mkdir(parents=True)
                expected.write_text("strict test bytes")
                actual = package/name
                actual.parent.mkdir(parents=True)
                actual.write_text("strict test bytes")
                self.assertIn(name, ops.installed_python_gate(source, build, package, []))
                actual.write_text("tampered test")
                with self.assertRaisesRegex(ValueError, "install differs: tests/"):
                    ops.installed_python_gate(source, build, package, [])
                actual.write_text("strict test bytes")
                expected.unlink()
                with self.assertRaisesRegex(ValueError, "missing installed test source"):
                    ops.installed_python_gate(source, build, package, [])
                expected.write_text("strict test bytes")
                actual.unlink()
                with self.assertRaisesRegex(ValueError, "missing/outside candidate"):
                    ops.installed_python_gate(source, build, package, [name])

    def test_ambiguous_test_sources_fail_instead_of_guessing_by_priority_or_hash(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d)
            pytest_source, sample_source = source/"tests/pytests/test_input.py", source/"tests/test_input.py"
            pytest_source.parent.mkdir(parents=True)
            pytest_source.write_text("first pytest copy")
            sample_source.write_text("later raw-tests copy")
            with self.assertRaisesRegex(ValueError, "ambiguous installed test"):
                ops.installed_python_source(source, source/"build", "tests/test_input.py")
            pytest_source.unlink()
            kind, expected = ops.installed_python_source(source, source/"build", "tests/test_input.py")
            self.assertEqual(kind, "sample-test-source")
            self.assertEqual(expected, sample_source)
            # No arbitrary source/psi4/tests fallback, or excluded raw CFour.
            (source/"tests/cfour").mkdir()
            (source/"tests/cfour/test_input.py").write_text("excluded from raw copy")
            with self.assertRaisesRegex(ValueError, "missing installed test"):
                ops.installed_python_source(source, source/"build", "tests/cfour/test_input.py")

    def test_unknown_or_escaping_python_mapping_is_never_silently_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ("unexpected.py", "../outside.py", "/absolute.py", "tests/unselected.py"):
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        ops.installed_python_source(root, root/"build", name)

    def junit(self, path, count=34, skipped=False, legacy=None):
        root = ET.Element("testsuites")
        suite = ET.SubElement(root, "testsuite")
        for i in range(count):
            case = ET.SubElement(suite, "testcase", classname="test_isapol_q_parallel", name=str(i))
            if skipped and i == 0:
                ET.SubElement(case, "skipped")
        for name, number in (legacy or {}).items():
            for i in range(number):
                ET.SubElement(suite, "testcase", classname=name, name=str(i))
        ET.ElementTree(root).write(path)

    def test_exact_oracle_case_count_and_no_skips(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"junit.xml"
            self.junit(path)
            self.assertEqual(ops.junit_gate(path, True)["frozen_q_cases"], 34)
            for count, skip in ((32, False), (33, False), (35, False), (34, True)):
                self.junit(path, count, skip)
                with self.assertRaises(ValueError):
                    ops.junit_gate(path, True)
        command = ops.test_command(Path("/old/env/bin/python"), Path("/source"), Path("/out"), Path("/scr"), True)
        self.assertEqual(sum(str(a).endswith(".py") for a in command), 1)
        self.assertTrue(str(command[-1]).endswith(ops.TEST))

    def test_exact84_candidate_modules_and_all_native_failure_kinds(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"junit.xml"
            good = dict(test_isapol_native_performance=18, test_isapol_partitioned_response=32)
            self.junit(path, legacy=good)
            self.assertEqual(ops.junit_gate(path, False)["cases"], 84)
            with self.assertRaisesRegex(ValueError, "unexpected native test module"):
                ops.junit_gate(path, True)  # Original oracle must contain ONLY34 Q cases.
            for counts in (
                dict(test_isapol_native_performance=17, test_isapol_partitioned_response=33),
                dict(test_isapol_native_performance=18, test_isapol_partitioned_response=31),
                dict(good, unexpected_module=1),
            ):
                self.junit(path, legacy=counts)
                with self.assertRaises(ValueError):
                    ops.junit_gate(path, False)
            for tag in ("failure", "error", "skipped"):
                self.junit(path, legacy=good)
                root = ET.parse(path).getroot()
                ET.SubElement(root.findall(".//testcase")[-1], tag)
                ET.ElementTree(root).write(path)
                with self.assertRaisesRegex(ValueError, "native tests failed/skipped"):
                    ops.junit_gate(path, False)

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
                         tests=dict(cases=34, frozen_q_cases=34, failures=0, errors=0, skipped=0))
            ops.write(path, value)
            with self.assertRaises(FileNotFoundError):
                ops.oracle_gate(path, ops.sha(path), source)
            ops.write(root/"COMPLETED", dict(exit_code=0, acceptance=False, mode="oracle"))
            self.assertEqual(ops.oracle_gate(path, ops.sha(path), source), value)
            with self.assertRaises(ValueError):
                ops.oracle_gate(path, "0"*64, source)
            for bad_source in (dict(source, commit="other-ops"),
                               dict(source, path="/old-source"),
                               dict(source, frozen_patch="5ae597cd9305588d14681d0da435834ef761c264")):
                with self.assertRaisesRegex(ValueError, "passing original"):
                    ops.oracle_gate(path, ops.sha(path), bad_source)
            value["tests"].update(cases=32, frozen_q_cases=32)
            path.write_text(json.dumps(value))  # Synthetic receipt only; never real evidence.
            with self.assertRaisesRegex(ValueError, "passing original"):
                ops.oracle_gate(path, ops.sha(path), source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
