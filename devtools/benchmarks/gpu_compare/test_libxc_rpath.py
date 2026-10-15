"""Loader regression using the real core RPATH construction, no chemistry stack."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("cmake") and shutil.which("cc"),
                     "Linux CMake/C compiler required for loader regression")
class LibxcRpathTests(unittest.TestCase):
    def test_selected_libxc_precedes_host_library_with_same_soname(self):
        repo = Path(__file__).resolve().parents[3]
        source = (repo / "psi4/src/CMakeLists.txt").read_text()
        block = source.split("# <<<  RPATH  >>>", 1)[1].split("set_target_properties(core", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "host.c").write_text("int xc_func_init(void) { return 0; }\n")
            (root / "cuda.c").write_text("int xc_func_init_flags(void) { return 42; }\n")
            (root / "core.c").write_text(
                "extern int xc_func_init_flags(void);\nint probe(void) { return xc_func_init_flags(); }\n")
            (root / "CMakeLists.txt").write_text("""
cmake_minimum_required(VERSION 3.20)
project(libxc_loader_regression C)
set(CMAKE_INSTALL_LIBDIR lib)
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/stage")
set(PYMOD_INSTALL_LIBDIR "")
set(Libxc_ENABLE_CUDA ON)
add_library(host SHARED host.c)
add_library(cuda SHARED cuda.c)
set_target_properties(host PROPERTIES OUTPUT_NAME xc SOVERSION 15 LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/host")
set_target_properties(cuda PROPERTIES OUTPUT_NAME xc SOVERSION 15 LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/cuda")
add_library(Libxc::xc SHARED IMPORTED)
set_target_properties(Libxc::xc PROPERTIES IMPORTED_CONFIGURATIONS RELEASE
  IMPORTED_LOCATION_RELEASE "${CMAKE_BINARY_DIR}/cuda/libxc.so.15"
  IMPORTED_SONAME_RELEASE "libxc.so.15")
add_library(core SHARED core.c)
add_dependencies(core host cuda)
target_link_libraries(core PRIVATE Libxc::xc)
target_link_options(core PRIVATE "-Wl,--disable-new-dtags")
set(_addons "${CMAKE_BINARY_DIR}/host/libxc.so" "${CMAKE_BINARY_DIR}/cuda/libxc.so")
""" + block + """
set_target_properties(core PROPERTIES INSTALL_RPATH "${psi4_RPATH}" BUILD_WITH_INSTALL_RPATH ON)
""")
            env = dict(os.environ, CC=shutil.which("cc"), LD_LIBRARY_PATH="")
            env.pop("LD_PRELOAD", None)
            for command in (["cmake", "-S", str(root), "-B", str(root / "build"), "-DCMAKE_BUILD_TYPE=Release"],
                            ["cmake", "--build", str(root / "build"), "--parallel", "2"]):
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, (result.stdout+result.stderr)[-2500:])
            result = subprocess.run([sys.executable, "-c",
                "import ctypes,sys; assert ctypes.CDLL(sys.argv[1]).probe()==42",
                str(root / "build/libcore.so")], env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr[-2500:])


if __name__ == "__main__":
    unittest.main()
