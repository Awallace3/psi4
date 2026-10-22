"""Real local CMake/ELF integration with explicitly synthetic vendor fixtures.

Only a temporary COPY of the Python contract has fixture hashes substituted.
The production contract/SDK are never changed. No CUDA code/load/device calls.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("contract", ROOT / "cmake/cuest_sdk_contract.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


class CMakeELF(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.modules = self.base / "modules"
        self.modules.mkdir()
        self.sdk = self.base / "sdk"
        self.include = self.sdk / "include"
        self.include.mkdir(parents=True)
        (self.include / "cuest.h").write_text("/* synthetic SDK fixture */\n#define ADMITTED_MARK 0\n")
        self.lib = self.sdk / "lib/13"
        self.lib.mkdir(parents=True)
        self.cuda = self.base / "cuda"
        self.cuda.mkdir()
        self.compile_library(self.cuda / "libcudart.so.13", "libcudart.so.13", "void runtime_stub(){}")
        self.compile_library(self.cuda / "libcublas.so.13", "libcublas.so.13", "void blas_stub(){}")
        self.library = self.lib / "libcuest.so.0.2.50"
        self.compile_library(self.library, "libcuest.so.0",
                             "extern void blas_stub();\n"
                             "void cuestDFMOIntegralsCompute(){blas_stub();}\n"
                             "void cuestDFMOIntegralsComputeWorkspaceQuery(){}\n"
                             "void cuestGetMajorVersion(){}\nvoid cuestGetMinorVersion(){}\n"
                             "void cuestGetPatchVersion(){}\n"
                             "void cuestDFIntPlanCreate(){}\nvoid cuestDFIntPlanCreateWorkspaceQuery(){}\n"
                             "void cuestDFIntPlanDestroy(){}\n",
                             [str(self.cuda / "libcublas.so.13")])
        (self.lib / "libcuest.so").symlink_to(self.library.name)
        (self.lib / "libcuest.so.0").symlink_to(self.library.name)
        shutil.copy(ROOT / "cmake/FindcuEST.cmake", self.modules)
        self.pin_fixture_contract()
        (self.modules / "FindCUDAToolkit.cmake").write_text(
            "set(CUDAToolkit_FOUND TRUE)\nset(CUDAToolkit_VERSION 13.3.0)\n"
            "set(CUDAToolkit_VERSION_MAJOR 13)\n"
            "if(NOT TARGET CUDA::cudart)\n"
            "add_library(CUDA::cudart SHARED IMPORTED)\n"
            f'set_target_properties(CUDA::cudart PROPERTIES IMPORTED_LOCATION "{self.cuda / "libcudart.so.13"}")\n'
            "endif()\n")
        self.project = self.base / "project"
        self.project.mkdir()
        (self.project / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.21)\nproject(contract NONE)\n"
            f'list(PREPEND CMAKE_MODULE_PATH "{self.modules}")\n'
            "find_package(cuEST 0.2.50 EXACT MODULE REQUIRED)\n"
            'file(WRITE "${CMAKE_BINARY_DIR}/selected.txt" "${cuEST_LIBRARY}")\n')
        self.env = dict(os.environ, LD_LIBRARY_PATH=str(self.cuda))
        self.build = self.base / "build"

    def pin_fixture_contract(self):
        contract = (ROOT / "cmake/cuest_sdk_contract.py").read_text()
        # Test-fixture-only hash substitution; retain all production logic.
        contract = contract.replace('if __name__ == "__main__":',
                                    f'LIBRARY_SHA256[13] = "{C.sha256(self.library)}"\n'
                                    f'HEADERS_SHA256 = "{C.header_digest(self.include)}"\n'
                                    'if __name__ == "__main__":')
        (self.modules / "cuest_sdk_contract.py").write_text(contract)

    def compile_library(self, path, soname, code, link=()):
        source = self.base / (path.name + ".c")
        source.write_text(code)
        result = subprocess.run(["cc", "-shared", "-fPIC", str(source), *link,
                                 f"-Wl,-soname,{soname}", "-o", str(path)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def configure(self, *extra):
        return subprocess.run(
            ["cmake", "-S", str(self.project), "-B", str(self.build),
             f"-DcuEST_ROOT={self.sdk}", f"-DcuEST_CUDA_RUNTIME_ROOT={self.cuda}", *extra],
            env=self.env, capture_output=True, text=True, timeout=45)

    def passed(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def failed(self, result, text):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(text, result.stdout + result.stderr)

    def test_exact_configure_and_unchanged_reconfigure(self):
        self.passed(self.configure())
        identity = json.loads((self.build / "cuest-sdk-identity.json").read_text())
        self.assertEqual(identity["version"], "0.2.50")
        self.assertEqual(identity["library"], str(self.library))
        self.passed(self.configure())

    def test_parent_identity_accepted(self):
        self.passed(self.configure())
        identity = self.build / "cuest-sdk-identity.json"
        self.build = self.base / "inner"
        self.passed(self.configure(f"-DcuEST_PARENT_IDENTITY={identity}"))

    def test_parent_identity_change_rejected(self):
        self.passed(self.configure())
        identity = self.build / "cuest-sdk-identity.json"
        data = json.loads(identity.read_text())
        data["library_sha256"] = "old"
        parent = self.base / "parent.json"
        parent.write_text(json.dumps(data))
        self.build = self.base / "inner"
        self.failed(self.configure(f"-DcuEST_PARENT_IDENTITY={parent}"), "identity changed")

    def test_cached_identity_change_rejected(self):
        self.passed(self.configure())
        self.compile_library(self.cuda / "libcublas.so.13", "libcublas.so.13",
                             "int changed(){return 3;} void blas_stub(){}")
        self.failed(self.configure(), "identity changed")

    def test_changed_header_rejected(self):
        (self.include / "cuest.h").write_text("old header")
        self.failed(self.configure(), "headers are not canonical")

    def test_changed_library_rejected(self):
        with self.library.open("ab") as stream:
            stream.write(b"changed")
        self.failed(self.configure(), "native library is not canonical")

    def test_no_cross_cuda_library_override(self):
        self.failed(self.configure("-DcuEST_LIBRARY=/other/lib/12/libcuest.so"), "Stale/cross-CUDA")

    def test_no_ambient_header_override(self):
        self.failed(self.configure("-DcuEST_INCLUDE_DIR=/other/include"), "Stale/cross-CUDA")

    def test_static_override_rejected(self):
        self.failed(self.configure("-DcuEST_STATIC_LIBRARY=/other/libcuest.a"), "Unverified static")

    def test_missing_matching_cuda_library_does_not_use_12(self):
        (self.lib / "libcuest.so").unlink()
        old = self.sdk / "lib/12"
        old.mkdir()
        shutil.copy(self.library, old / "libcuest.so")
        self.failed(self.configure(), "admission failed")

    def test_superbuild_inner_missing_identity_rejected(self):
        self.failed(self.configure("-DcuEST_SUPERBUILD_INNER=ON"), "requires the outer cuEST identity")

    def test_foreign_imported_target_rejected(self):
        cmake = self.project / "CMakeLists.txt"
        source = cmake.read_text().replace("find_package(cuEST", "add_library(cuEST::cuEST SHARED IMPORTED)\nfind_package(cuEST", 1)
        cmake.write_text(source)
        self.failed(self.configure(), "Foreign pre-existing")

    def test_actual_superbuild_subdirectory_scope_and_inner_arguments(self):
        inner = self.project
        outer = self.base / "outer"
        discovery = outer / "external/upstream/cuest"
        discovery.mkdir(parents=True)
        (discovery / "CMakeLists.txt").write_text("find_package(cuEST 0.2.50 EXACT MODULE REQUIRED)\n")
        (outer / "external/upstream/CMakeLists.txt").write_text("add_subdirectory(cuest)\n")
        (outer / "external/CMakeLists.txt").write_text("add_subdirectory(upstream)\n")
        (outer / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.21)\nproject(outer NONE)\n"
            f'list(PREPEND CMAKE_MODULE_PATH "{self.modules}")\n'
            "add_subdirectory(external)\n"
            "get_property(cuEST_IDENTITY_FILE GLOBAL PROPERTY PSI4_CUEST_IDENTITY)\n"
            'if(NOT EXISTS "${cuEST_IDENTITY_FILE}")\nmessage(FATAL_ERROR "identity lost in subdirectory scope")\nendif()\n'
            'file(WRITE "${CMAKE_BINARY_DIR}/forwarded.txt" "${cuEST_IDENTITY_FILE}")\n'
            "include(ExternalProject)\n"
            f'ExternalProject_Add(inner SOURCE_DIR "{inner}" DOWNLOAD_COMMAND "" UPDATE_COMMAND ""\n'
            'CMAKE_ARGS "-DcuEST_ROOT=${cuEST_ROOT}" "-DcuEST_CUDA_RUNTIME_ROOT=${cuEST_CUDA_RUNTIME_ROOT}"\n'
            '"-DcuEST_PARENT_IDENTITY=${cuEST_IDENTITY_FILE}" "-DcuEST_SUPERBUILD_INNER=ON"\n'
            'BUILD_COMMAND "" INSTALL_COMMAND "" STEP_TARGETS configure)\n')
        self.project = outer
        self.passed(self.configure())
        expected = str(self.build / "cuest-sdk-identity.json")
        self.assertEqual((self.build / "forwarded.txt").read_text(), expected)
        result = subprocess.run(["cmake", "--build", str(self.build), "--target", "inner-configure"],
                                env=self.env, capture_output=True, text=True, timeout=60)
        self.passed(result)
        args = (self.build / "inner-prefix/tmp/inner-cfgcmd.txt").read_text()
        self.assertIn(f"-DcuEST_PARENT_IDENTITY={expected}", args)
        receipt = self.build / "inner-prefix/src/inner-build/cuest-sdk-identity.json"
        self.assertEqual(json.loads(receipt.read_text()),
                         json.loads((self.build / "cuest-sdk-identity.json").read_text()))

    def header_build(self, precedence):
        old = self.base / "old-env/include"
        old.mkdir(parents=True)
        (old / "cuest.h").write_text("/* old environment, NOT admitted */\n#define ADMITTED_MARK 9\n")
        cpp = self.project / "header.c"
        cpp.write_text("#include <cuest.h>\nint main(){return ADMITTED_MARK;}\n")
        cmake = self.project / "CMakeLists.txt"
        source = cmake.read_text().replace("project(contract NONE)", "project(contract C)")
        if precedence:
            source += ('include_directories(BEFORE "${cuEST_INCLUDE_DIR}")\n'
                       "set(CMAKE_NO_SYSTEM_FROM_IMPORTED ON)\n")
        source += ("add_library(required INTERFACE)\n"
                   f'target_include_directories(required INTERFACE "{old}")\n'
                   "add_executable(header_probe header.c)\n"
                   "target_link_libraries(header_probe PRIVATE required cuEST::cuEST)\n")
        cmake.write_text(source)
        self.passed(self.configure("-G", "Ninja"))
        result = subprocess.run(["cmake", "--build", str(self.build), "--target", "header_probe"],
                                env=self.env, capture_output=True, text=True, timeout=45)
        self.passed(result)
        shutil.copy(ROOT / "cmake/cuest_header_dependencies.py", self.modules)
        return subprocess.run([sys.executable, str(self.modules / "cuest_header_dependencies.py"),
                               "--sdk-root", str(self.sdk), "--build-directory", str(self.build),
                               "--generator", "Ninja", "--output", str(self.build / "headers.json")],
                              env=self.env, capture_output=True, text=True, timeout=45)

    def test_compiler_consumes_sdk_before_old_required_dependency(self):
        self.passed(self.header_build(True))
        proved = json.loads((self.build / "headers.json").read_text())
        self.assertIn(str(self.include / "cuest.h"), proved)

    def test_dependency_audit_rejects_actually_compiled_shadow(self):
        self.failed(self.header_build(False), "compiler consumed an unadmitted")

    def test_poisoned_release_location_rejected_on_repeat_find(self):
        cmake = self.project / "CMakeLists.txt"
        with cmake.open("a") as stream:
            stream.write('set_property(TARGET cuEST::cuEST PROPERTY IMPORTED_LOCATION_RELEASE "/old/libcuest.so")\n'
                         "find_package(cuEST 0.2.50 EXACT MODULE REQUIRED)\n")
        self.failed(self.configure(), "Configuration-specific cuEST target binding")

    def test_poisoned_interface_headers_rejected_on_repeat_find(self):
        cmake = self.project / "CMakeLists.txt"
        with cmake.open("a") as stream:
            stream.write('set_property(TARGET cuEST::cuEST PROPERTY INTERFACE_INCLUDE_DIRECTORIES "/old/include")\n'
                         "find_package(cuEST 0.2.50 EXACT MODULE REQUIRED)\n")
        self.failed(self.configure(), "cuEST target property changed")


if __name__ == "__main__":
    unittest.main()
