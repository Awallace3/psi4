"""Auxiliary Ninja rules must not be mistaken for compiler/link commands."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
spec = importlib.util.spec_from_file_location("audit_codegen_fixture", HERE.parent / "build_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class CodegenAudit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        self.source = self.root / "source"
        self.scratch = self.root / "scratch"
        for folder in (self.run, self.source, self.scratch):
            folder.mkdir()
        sdk = self.root / "sdk"
        self.sdk = {"source": str(sdk), "include": str(sdk / "include"),
                    "library_root": str(sdk / "lib64"),
                    "support": {"bin/nvcc": {"path": str(sdk / "bin/nvcc")}}}
        self.compiler = self.run / "environment/bin/x86_64-conda-linux-gnu-c++"

    def check(self, command):
        return audit.audit_link_commands(command, self.run, self.run,
                                         self.source, self.sdk, self.scratch)

    def test_actual_apiary_aggregate_response_is_not_a_linker_response(self):
        command = (f"cd {self.run} && {self.run}/environment/bin/python "
                   f"{self.run}/_deps/apiary-src/scripts/apiary_aggregate_stubs.py "
                   f"@{self.run}/generated/.stubs.fragments --pkg-dir {self.run}/lib/einsums "
                   f'--overlay-class-regex "^class (RuntimeTensor): *\\$" && '
                   f"{self.run}/environment/bin/cmake -E touch {self.run}/.stubs.stamp")
        self.assertEqual(self.check(command), 0)

    def test_real_link_with_cc_source_and_origin_is_still_checked(self):
        command = (f': && {self.compiler} -shared {self.source}/fixture.cc '
                   f'-Wl,-rpath,\\$ORIGIN/../ -o {self.run}/fixture.so && :')
        self.assertEqual(self.check(command), 1)

    def test_unexpanded_actual_compiler_or_linker_response_is_still_rejected(self):
        for response in (f"@{self.run}/missing.rsp", f"-Wl,@{self.run}/missing.rsp"):
            with self.assertRaises(RuntimeError):
                self.check(f"{self.compiler} -shared {response} -o {self.run}/fixture.so")

    def test_foreign_compiler_and_include_are_still_rejected(self):
        for command in (f"/usr/bin/g++ -shared -o {self.run}/bad.so",
                        f"{self.compiler} -shared -I/foreign/include -o {self.run}/bad.so",
                        f"cd /foreign && {self.compiler} -shared -o {self.run}/bad.so",
                        f"ccache {self.compiler} -shared -o {self.run}/bad.so"):
            with self.assertRaises(RuntimeError):
                self.check(command)

    def test_opaque_shell_and_environment_compiler_wrappers_are_rejected(self):
        for command in (
                f"/bin/sh -c '{self.compiler} -shared @{self.run}/hidden.rsp'",
                f"/bin/bash -c '/usr/bin/g++ -shared -o {self.run}/foreign.so'",
                f"/usr/bin/env sh -c '{self.compiler} -shared @hidden.rsp'",
                f"{self.run}/environment/bin/python -c 'import os; os.system(\"/usr/bin/g++ @hidden.rsp\")'",
                f"{self.run}/environment/bin/cmake -E env sh -c '{self.compiler} @hidden.rsp'"):
            with self.assertRaises(RuntimeError):
                self.check(command)

    def test_cmake_configuration_compiler_declarations_are_not_executed_wrappers(self):
        cmake = self.run / "environment/bin/cmake"
        self.assertEqual(self.check(
            f"{cmake} -DCMAKE_CXX_COMPILER={self.compiler} -S {self.source} -B {self.run}/build"), 0)
        for command in (
                f"{cmake} -DCMAKE_CXX_COMPILER=/usr/bin/g++ -S {self.source}",
                f"{cmake} -DOPAQUE_WRAPPER={self.compiler}",
                f"unknown {self.compiler} -shared @hidden.rsp"):
            with self.assertRaises(RuntimeError):
                self.check(command)

    def test_validation_collects_independent_checks_but_requires_every_one(self):
        script = (HERE / "validate_install.sh").read_text()
        self.assertIn("SKIPPED_REQUIRES_INSTALLED_IDENTITY_PROOF", script)
        self.assertIn("audit_code == 0 && proof_code == 0 && api_code == 0", script)
        self.assertIn("snapshot_code == 0 && binary_code == 0 && checks_code == 0", script)
        self.assertIn('"$out/audit.stderr"', script)
        self.assertIn('"$out/installed-proof.stderr"', script)
        self.assertIn('"$out/binaries-after.json"', script)


if __name__ == "__main__":
    unittest.main()
