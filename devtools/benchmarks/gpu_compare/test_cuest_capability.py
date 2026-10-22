"""Direct capability on strictly admitted synthetic SDKs; no CUDA/device calls."""
from pathlib import Path
import importlib.util
import shutil
import unittest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("cmake_fixtures", ROOT / "devtools/test_cuest_cmake_admission.py")
FIXTURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURES)


@unittest.skipUnless(shutil.which("cmake") and shutil.which("c++"), "CMake/C++ required")
class CuestCapabilityTests(unittest.TestCase):
    names = ["cuestDirectIntPlanCreate", "cuestDirectIntPlanCreateWorkspaceQuery",
             "cuestDirectIntPlanDestroy", "cuestDirectCoulombCompute",
             "cuestDirectCoulombComputeWorkspaceQuery", "cuestDirectSymmetricExchangeCompute",
             "cuestDirectSymmetricExchangeComputeWorkspaceQuery"]

    def fixture(self, declarations, symbols):
        f = FIXTURES.CMakeELF()
        f.setUp()
        self.addCleanup(f.doCleanups)
        header = f.include / "cuest.h"
        header.write_text("\n".join(f'extern "C" int {n}();' for n in declarations))
        code = (f.base / (f.library.name + ".c")).read_text()
        code += "\n".join(f"int {n}() {{ return 0; }}" for n in symbols)
        f.compile_library(f.library, "libcuest.so.0", code, [str(f.cuda / "libcublas.so.13")])
        f.pin_fixture_contract()  # Only temporary fixture hashes; production pins unchanged.
        cmake = f.project / "CMakeLists.txt"
        cmake.write_text(cmake.read_text().replace("project(contract NONE)", "project(contract CXX)")
                         + '\nfile(WRITE "${CMAKE_BINARY_DIR}/capability" "${cuEST_HAS_DIRECT_JK}")\n')
        return f

    def test_declarations_and_linkable_symbols_required(self):
        for variant, declarations, symbols, expected in [
            ("new", self.names, self.names, "1"), ("old", [], [], ""),
            ("incomplete", self.names, self.names[:-1], "")]:
            with self.subTest(variant=variant):
                f = self.fixture(declarations, symbols)
                f.passed(f.configure())
                self.assertEqual((f.build / "capability").read_text(), expected)

    def test_unchanged_admitted_sdk_recomputes_stale_capability_cache(self):
        f = self.fixture(self.names, self.names)
        f.passed(f.configure())
        f.passed(f.configure("-DcuEST_HAS_DIRECT_JK:INTERNAL=0"))
        self.assertEqual((f.build / "capability").read_text(), "1")

    def test_changed_admitted_sdk_cannot_reuse_identity(self):
        f = self.fixture(self.names, self.names)
        f.passed(f.configure())
        header = f.include / "cuest.h"
        header.write_text(header.read_text() + "\n/* changed identity */\n")
        f.pin_fixture_contract()
        f.failed(f.configure(), "identity changed")


if __name__ == "__main__":
    unittest.main()
