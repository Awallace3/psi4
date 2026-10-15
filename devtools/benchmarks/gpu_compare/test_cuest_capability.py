"""Probe real FindcuEST logic with tiny fake SDKs; no CUDA or proprietary code."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which("cmake") and shutil.which("c++"), "CMake/C++ required")
class CuestCapabilityTests(unittest.TestCase):
    def test_declarations_and_linkable_symbols_required_even_after_reconfigure(self):
        repo = Path(__file__).resolve().parents[3]
        names = ["cuestDirectIntPlanCreate", "cuestDirectIntPlanCreateWorkspaceQuery",
                 "cuestDirectIntPlanDestroy", "cuestDirectCoulombCompute",
                 "cuestDirectCoulombComputeWorkspaceQuery", "cuestDirectSymmetricExchangeCompute",
                 "cuestDirectSymmetricExchangeComputeWorkspaceQuery"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "modules").mkdir()
            (root / "modules/FindCUDAToolkit.cmake").write_text("""
set(CUDAToolkit_FOUND TRUE)
set(CUDAToolkit_VERSION "13.0")
if(NOT TARGET CUDA::cudart)
  add_library(CUDA::cudart INTERFACE IMPORTED)
endif()
""")
            (root / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.20)
project(cuest_capability CXX)
list(PREPEND CMAKE_MODULE_PATH "${{CMAKE_CURRENT_SOURCE_DIR}}/modules" "{repo}/cmake")
find_package(cuEST MODULE REQUIRED)
file(WRITE "${{CMAKE_BINARY_DIR}}/capability" "${{cuEST_HAS_DIRECT_JK}}")
""")
            env = dict(os.environ)
            env.pop("LD_PRELOAD", None)
            for variant, declarations, symbols, expected in [
                ("new", names, names, "1"), ("old", [], [], ""),
                ("incomplete", names, names[:-1], "")]:
                sdk = root / variant
                (sdk / "include").mkdir(parents=True)
                (sdk / "lib").mkdir()
                (sdk / "include/cuest.h").write_text(
                    "\n".join(f'extern "C" int {n}();' for n in declarations))
                source = sdk / "fake.cc"
                source.write_text("\n".join(f'extern "C" int {n}() {{ return 0; }}' for n in symbols))
                result = subprocess.run(["c++", "-shared", "-fPIC", str(source), "-o",
                                         str(sdk / "lib/libcuest.so")], env=env,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                result = subprocess.run(["cmake", "-S", str(root), "-B", str(root / "build"),
                    f"-DcuEST_ROOT={sdk}", f"-DcuEST_INCLUDE_DIR={sdk}/include",
                    f"-DcuEST_LIBRARY={sdk}/lib/libcuest.so"], env=env,
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-3000:])
                self.assertEqual((root / "build/capability").read_text(), expected, variant)


if __name__ == "__main__":
    unittest.main()
