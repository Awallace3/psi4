"""Real CMake configure-only loader fixtures; no compiler detection or build."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


class HookTests(unittest.TestCase):
    def configure(self, corruption=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / "private13"
            (private / "lib64").mkdir(parents=True)
            (private / "include").mkdir()
            allowed = {}
            for family in ("cudart", "cublas", "cublasLt", "cusolver", "cusparse", "nvJitLink"):
                path = private / "lib64" / f"lib{family}.so"
                path.write_bytes(b"CONFIGURE FIXTURE ONLY NO ELF")
                allowed[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            if corruption == "hash":
                allowed[str(path)] = "0" * 64
            sdk = private / "contract.json"
            sdk.write_text(json.dumps({"private_prefix": str(private), "include": str(private / "include"),
                                       "library_root": str(private / "lib64"), "allowed_files": allowed}))
            src = root / "src"
            src.mkdir()
            fixture = HERE.parent / "contract_config_fixture/CMakeLists.txt"
            text = fixture.read_text()
            # This remains LANGUAGES NONE, and every compile action is cmake -E false.
            text += """
add_library(waggle SHARED fixture.cc)
set_target_properties(waggle PROPERTIES LINKER_LANGUAGE CXX)
add_library(waggle_core MODULE fixture.cc)
set_target_properties(waggle_core PROPERTIES LINKER_LANGUAGE CXX)
"""
            (src / "CMakeLists.txt").write_text(text)
            (src / "fixture.cc").write_text("// NEVER COMPILED\n")
            args = ["cmake", "-S", str(src), "-B", str(root / "build"),
                    f"-DCMAKE_PROJECT_Einsums_INCLUDE={HERE / 'install_hook.cmake'}",
                    f"-DDISP0_SDK_CONTRACT={sdk}", f"-DDISP0_PRIVATE_CUDA={private}",
                    f"-DDISP0_DEPENDENCY_PREFIX={root / 'env'}", "-DCMAKE_INSTALL_LIBDIR=lib"]
            if corruption == "missing":
                args.append("-DFIXTURE_MISSING=nvJitLink")
            if corruption == "interface":
                args.append("-DFIXTURE_INTERFACE=nvJitLink")
            if corruption == "no-python":
                args.append("-DFIXTURE_MISSING_PYTHON=ON")
            if corruption == "no-stubs":
                args.append("-DFIXTURE_MISSING_STUBS=ON")
            if corruption == "foreign":
                bad = root / "foreign.so"
                bad.write_bytes(b"unattested")
                args += ["-DFIXTURE_FOREIGN=nvJitLink", f"-DFIXTURE_FOREIGN_PATH={bad}"]
            result = subprocess.run(args, capture_output=True, text=True, timeout=15)
            if corruption:
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            else:
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = (root / "build/disp0-install-targets.tsv").read_text()
                for name in ("Einsums", "PyEinsums", "waggle", "waggle_core"):
                    self.assertIn(name + "\t", report)
                self.assertIn("--enable-new-dtags", (root / "build/fixture-properties.txt").read_text())

    def test_positive_whole_install_policy(self):
        self.configure()

    def test_reject_missing_component(self):
        self.configure("missing")

    def test_reject_interface_component(self):
        self.configure("interface")

    def test_reject_changed_component(self):
        self.configure("hash")

    def test_reject_foreign_component(self):
        self.configure("foreign")

    def test_reject_no_python(self):
        self.configure("no-python")

    def test_reject_no_stubs(self):
        self.configure("no-stubs")


if __name__ == "__main__":
    unittest.main()
