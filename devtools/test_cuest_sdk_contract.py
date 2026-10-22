"""Offline contract tests: synthetic ELF metadata is explicitly mocked.

No SDK load, CUDA call, GPU, package install, or remote access.
"""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("contract", ROOT / "cmake/cuest_sdk_contract.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


class Contract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.sdk = self.base / "sdk"
        self.cuda = self.base / "cuda"
        self.include = self.sdk / "include"
        self.include.mkdir(parents=True)
        (self.include / "cuest.h").write_text("fixture, NOT vendor headers")
        self.library = self.sdk / "lib/13/libcuest.so.0.2.50"
        self.library.parent.mkdir(parents=True)
        self.library.write_bytes(b"fixture, NOT vendor ELF")
        (self.library.parent / "libcuest.so").symlink_to(self.library.name)
        self.cuda.mkdir()
        self.cudart = self.cuda / "libcudart.so.13.3.29"
        self.cublas = self.cuda / "libcublas.so.13.6.0.2"
        self.cudart.write_bytes(b"runtime")
        self.cublas.write_bytes(b"blas")
        self.core = self.base / "core.so"
        self.core.write_bytes(b"core")
        self.addCleanup(patch.stopall)
        patch.object(C, "HEADERS_SHA256", C.header_digest(self.include)).start()
        patch.dict(C.LIBRARY_SHA256, {13: C.sha256(self.library)}).start()
        self.run = patch.object(C, "command", side_effect=self.command).start()

    def command(self, argv):
        path = Path(argv[-1])
        if argv[0] == "readelf" and argv[1] == "-d":
            if path == self.cudart:
                return "(SONAME) Library soname: [libcudart.so.13]"
            return ("(NEEDED) Shared library: [libcublas.so.13]\n"
                    "(SONAME) Library soname: [libcuest.so.0]")
        if argv[0] == "readelf":
            names = ("cuestDFMOIntegralsCompute", "cuestDFMOIntegralsComputeWorkspaceQuery",
                     "cuestDFIntPlanCreate", "cuestDFIntPlanCreateWorkspaceQuery", "cuestDFIntPlanDestroy",
                     "cuestGetMajorVersion", "cuestGetMinorVersion", "cuestGetPatchVersion")
            return "".join(f" 1: 0 1 FUNC GLOBAL DEFAULT 11 {name}\n" for name in names)
        if argv[0] == "ldd":
            result = f"libcublas.so.13 => {self.cublas} (0x1)\n"
            if path == self.core:
                result += (f"libcudart.so.13 => {self.cudart} (0x2)\n"
                           f"libcuest.so.0 => {self.library} (0x3)\n")
            return result
        raise AssertionError(argv)

    def validate(self):
        return C.validate_sdk(self.sdk, 13, self.cuda, self.cudart)

    def test_valid_synthetic_sdk(self):
        sdk = self.validate()
        self.assertEqual(sdk["version"], "0.2.50")
        self.assertEqual(sdk["cuda_major"], 13)
        self.assertEqual(sdk["library"], str(self.library))
        self.assertIn("libcudart.so.13", sdk["cuda_closure"])

    def test_old_native_bytes_rejected(self):
        self.library.write_bytes(b"old 0.2.2")
        with self.assertRaisesRegex(ValueError, "canonical 0.2.50"):
            self.validate()

    def test_header_change_rejected(self):
        (self.include / "cuest.h").write_text("old headers")
        with self.assertRaisesRegex(ValueError, "headers"):
            self.validate()

    def test_extra_header_rejected(self):
        (self.include / "shadow.h").write_bytes(b"shadow")
        with self.assertRaisesRegex(ValueError, "headers"):
            self.validate()

    def test_missing_headers_rejected(self):
        (self.include / "cuest.h").unlink()
        with self.assertRaisesRegex(ValueError, "headers"):
            self.validate()

    def test_header_escape_rejected(self):
        foreign = self.base / "old.h"
        foreign.write_bytes(b"old")
        (self.include / "escape.h").symlink_to(foreign)
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.validate()

    def test_library_escape_rejected(self):
        link = self.library.parent / "libcuest.so"
        link.unlink()
        foreign = self.base / "foreign.so"
        foreign.write_bytes(self.library.read_bytes())
        link.symlink_to(foreign)
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.validate()

    def test_no_cross_cuda_fallback(self):
        self.library.unlink()
        other = self.sdk / "lib/12/libcuest.so"
        other.parent.mkdir()
        other.write_bytes(b"anything")
        with self.assertRaises(FileNotFoundError):
            self.validate()

    def test_unsupported_cuda_major(self):
        with self.assertRaisesRegex(ValueError, "12/13"):
            C.validate_sdk(self.sdk, 14, self.cuda, self.cudart)

    def test_wrong_cudart_major(self):
        original = self.command
        self.run.side_effect = lambda argv: (
            "(SONAME) Library soname: [libcudart.so.12]"
            if argv[:2] == ["readelf", "-d"] and Path(argv[-1]) == self.cudart
            else original(argv))
        with self.assertRaisesRegex(ValueError, "toolkit major"):
            self.validate()

    def test_cuda_dependency_major_rejected(self):
        original = self.command
        self.run.side_effect = lambda argv: original(argv).replace("libcublas.so.13", "libcublas.so.12")
        with self.assertRaisesRegex(ValueError, "cross-CUDA"):
            self.validate()

    def test_cusolver_major_is_not_toolkit_major(self):
        C.check_cuda_names(["libcusolver.so.12", "libcusparse.so.12"], 13)
        with self.assertRaisesRegex(ValueError, "cross-CUDA"):
            C.check_cuda_names(["libcusolver.so.13"], 13)

    def test_cuda_outside_root_rejected(self):
        original = self.command
        outside = self.base / "libcublas.so.13"
        outside.write_bytes(b"other")
        self.run.side_effect = lambda argv: original(argv).replace(str(self.cublas), str(outside))
        with self.assertRaisesRegex(ValueError, "outside"):
            self.validate()

    def test_cudart_outside_root_rejected(self):
        outside = self.base / "libcudart.so"
        outside.write_bytes(b"other")
        with self.assertRaisesRegex(ValueError, "escapes"):
            C.validate_sdk(self.sdk, 13, self.cuda, outside)

    def test_unresolved_cuda_rejected(self):
        original = self.command
        self.run.side_effect = lambda argv: (
            "libcublas.so.13 => not found" if argv[0] == "ldd" else original(argv))
        with self.assertRaisesRegex(ValueError, "unresolved"):
            self.validate()

    def test_missing_symbol_rejected(self):
        original = self.command
        self.run.side_effect = lambda argv: (
            "" if argv[:2] == ["readelf", "--dyn-syms"] else original(argv))
        with self.assertRaisesRegex(ValueError, "symbol missing"):
            self.validate()

    def test_undefined_symbol_rejected(self):
        original = self.command
        self.run.side_effect = lambda argv: original(argv).replace("DEFAULT 11", "DEFAULT UND")
        with self.assertRaisesRegex(ValueError, "symbol missing"):
            self.validate()

    def test_unexpected_soname_rejected(self):
        original = self.command
        self.run.side_effect = lambda argv: original(argv).replace("[libcuest.so.0]", "[libcuest.so.9]")
        with self.assertRaisesRegex(ValueError, "SONAME"):
            self.validate()

    def test_old_parent_identity_rejected(self):
        actual = self.validate()
        with self.assertRaisesRegex(ValueError, "fresh isolated"):
            C.compare_identity(actual, dict(actual, library_sha256="old"))

    def test_changed_root_even_same_bytes_rejected(self):
        actual = self.validate()
        with self.assertRaisesRegex(ValueError, "fresh isolated"):
            C.compare_identity(actual, dict(actual, sdk_root="/other"))

    def test_snapshot_core(self):
        snap = C.snapshot_core(self.core, self.validate())
        self.assertEqual(snap["core_sha256"], C.sha256(self.core))

    def test_core_linked_to_old_library_rejected(self):
        sdk = self.validate()
        old = self.base / "libcuest.so.0.2.2"
        old.write_bytes(b"old")
        original = self.command
        self.run.side_effect = lambda argv: original(argv).replace(str(self.library), str(old))
        with self.assertRaisesRegex(ValueError, "different cuEST"):
            C.snapshot_core(self.core, sdk)

    def test_cuda_modified_after_sdk_admission_rejected(self):
        sdk = self.validate()
        self.cublas.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "dependency changed"):
            C.snapshot_core(self.core, sdk)

    def maps(self, paths, include_core=True):
        import os
        paths = list(paths)
        if include_core and not any(str(p).removesuffix(" (deleted)") == str(self.core) for p in paths):
            paths.append(self.core)
        lines = []
        for p in paths:
            stat = Path(str(p).removesuffix(" (deleted)")).stat()
            dev = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}"
            lines.append(f"0-1 r-xp 0 {dev} {stat.st_ino} {p}")
        return "\n".join(lines)

    def test_actual_runtime_maps_admitted(self):
        snap = C.snapshot_core(self.core, self.validate())
        with patch.object(Path, "read_text", return_value=self.maps([self.library, self.cublas, self.cudart])):
            self.assertEqual(C.admit_runtime(self.core, snap), snap)

    def test_missing_mapped_library_rejected(self):
        snap = C.snapshot_core(self.core, self.validate())
        with patch.object(Path, "read_text", return_value=self.maps([self.library])):
            with self.assertRaisesRegex(ValueError, "actual mapped"):
                C.admit_runtime(self.core, snap)

    def test_extra_mapped_gpu_library_rejected(self):
        snap = C.snapshot_core(self.core, self.validate())
        old = self.base / "libcublas.so.12"
        old.write_bytes(b"other")
        with patch.object(Path, "read_text", return_value=self.maps([self.library, self.cublas, self.cudart, old])):
            with self.assertRaisesRegex(ValueError, "actual mapped"):
                C.admit_runtime(self.core, snap)

    def test_changed_core_after_build_rejected(self):
        snap = C.snapshot_core(self.core, self.validate())
        self.core.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "fresh isolated"):
            C.admit_runtime(self.core, snap)

    def test_deleted_gpu_mapping_rejected(self):
        snap = C.snapshot_core(self.core, self.validate())
        with patch.object(Path, "read_text", return_value=self.maps([str(self.library) + " (deleted)"])):
            with self.assertRaisesRegex(ValueError, "deleted GPU"):
                C.admit_runtime(self.core, snap)

    def test_duplicate_ldd_entries_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            C.parse_ldd(f"libcudart.so.13 => {self.cudart} (0x1)\n" * 2)

    def test_ldd_cudart_not_overwritten_by_selected_runtime(self):
        foreign = self.base / "libcudart.so.13"
        foreign.write_bytes(b"other")
        original = self.command
        self.run.side_effect = lambda argv: (
            original(argv) + f"libcudart.so.13 => {foreign} (0x4)\n"
            if argv[0] == "ldd" else original(argv))
        with self.assertRaisesRegex(ValueError, "different CUDA runtime"):
            self.validate()

    def test_mapped_inode_cannot_be_replaced_by_new_file(self):
        snap = C.snapshot_core(self.core, self.validate())
        text = self.maps([self.library, self.cublas, self.cudart])
        inode = self.library.stat().st_ino
        text = text.replace(f" {inode} {self.library}", f" {inode + 1} {self.library}")
        with patch.object(Path, "read_text", return_value=text):
            with self.assertRaisesRegex(ValueError, "inode/device"):
                C.admit_runtime(self.core, snap)

    def test_extra_unexpected_curand_mapping_rejected(self):
        snap = C.snapshot_core(self.core, self.validate())
        extra = self.base / "libcurand.so.10"
        extra.write_bytes(b"extra")
        with patch.object(Path, "read_text", return_value=self.maps([self.library, self.cublas, self.cudart, extra])):
            with self.assertRaisesRegex(ValueError, "actual mapped"):
                C.admit_runtime(self.core, snap)

    def test_system_curl_and_site_driver_not_toolkit_components(self):
        for name in ("libcurl.so.4", "libcuda.so.1", "libcudadebugger.so.1"):
            self.assertFalse(C.gpu_library(name))

    def optional(self):
        path = self.cuda / "lib/libnvrtc-builtins.so.13.3"
        path.parent.mkdir()
        path.write_bytes(b"lazy-builtins")
        original = self.command
        self.run.side_effect = lambda argv: (
            "(SONAME) Library soname: [libnvrtc-builtins.so.13.3]"
            if argv[:2] == ["readelf", "-d"] and Path(argv[-1]) == path else original(argv))
        return path

    def test_explicit_lazy_cuda_component_admitted(self):
        optional = self.optional()
        snap = C.snapshot_core(self.core, self.validate())
        self.assertEqual(snap["optional_cuda"][str(optional)], C.sha256(optional))
        with patch.object(Path, "read_text", return_value=self.maps([self.library, self.cublas, self.cudart, optional])):
            self.assertEqual(C.admit_runtime(self.core, snap), snap)

    def test_lazy_component_changed_since_build_rejected(self):
        optional = self.optional()
        snap = C.snapshot_core(self.core, self.validate())
        optional.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "identity changed"):
            C.admit_runtime(self.core, snap)

    def test_regenerating_current_sdk_cannot_bless_old_core_receipt(self):
        original = C.snapshot_core(self.core, self.validate())
        self.cublas.write_bytes(b"changed after original build")
        regenerated = C.snapshot_core(self.core, self.validate())
        self.assertEqual(original["core_sha256"], regenerated["core_sha256"])
        with self.assertRaisesRegex(ValueError, "identity changed"):
            C.compare_identity(regenerated, original)

    def test_core_requires_actual_executable_mapping(self):
        snap = C.snapshot_core(self.core, self.validate())
        with patch.object(Path, "read_text", return_value=self.maps([self.library, self.cublas, self.cudart], include_core=False)):
            with self.assertRaisesRegex(ValueError, "core has no proved executable"):
                C.admit_runtime(self.core, snap)

    def test_core_inode_cannot_be_swapped_after_import(self):
        snap = C.snapshot_core(self.core, self.validate())
        text = self.maps([self.library, self.cublas, self.cudart])
        inode = self.core.stat().st_ino
        text = text.replace(f" {inode} {self.core}", f" {inode+1} {self.core}")
        with patch.object(Path, "read_text", return_value=text):
            with self.assertRaisesRegex(ValueError, "inode/device"):
                C.admit_runtime(self.core, snap)

    def test_deleted_core_mapping_rejected(self):
        snap = C.snapshot_core(self.core, self.validate())
        with patch.object(Path, "read_text", return_value=self.maps([str(self.core) + " (deleted)"])):
            with self.assertRaisesRegex(ValueError, "deleted GPU/core"):
                C.admit_runtime(self.core, snap)

    def test_data_only_core_mapping_not_execution_proof(self):
        snap = C.snapshot_core(self.core, self.validate())
        text = self.maps([self.library, self.cublas, self.cudart])
        text = "\n".join(line.replace("r-xp", "r--p") if line.endswith(str(self.core)) else line
                         for line in text.splitlines())
        with patch.object(Path, "read_text", return_value=text):
            with self.assertRaisesRegex(ValueError, "core has no proved executable"):
                C.admit_runtime(self.core, snap)


class Wiring(unittest.TestCase):
    def test_outer_and_inner_require_exact_module(self):
        for filename in ("external/upstream/cuest/CMakeLists.txt", "psi4/CMakeLists.txt"):
            source = (ROOT / filename).read_text()
            self.assertIn("find_package(cuEST 0.2.50 EXACT MODULE REQUIRED)", source)
            self.assertNotIn("find_package(cuEST 0.2 CONFIG", source)

    def test_superbuild_forwards_selected_identity(self):
        source = (ROOT / "CMakeLists.txt").read_text()
        for arg in ("ROOT", "INCLUDE_DIR", "LIBRARY", "CUDA_RUNTIME_ROOT"):
            self.assertIn(f"-DcuEST_{arg}=${{cuEST_{arg}}}", source)
        self.assertIn("-DcuEST_PARENT_IDENTITY=${cuEST_IDENTITY_FILE}", source)

    def test_no_find_library_fallback(self):
        source = (ROOT / "cmake/FindcuEST.cmake").read_text()
        self.assertNotIn("find_library(", source)
        self.assertIn("Stale/cross-CUDA", source)

    def test_superbuild_identity_escapes_discovery_directory_scope(self):
        module = (ROOT / "cmake/FindcuEST.cmake").read_text()
        outer = (ROOT / "CMakeLists.txt").read_text()
        self.assertIn("set_property(GLOBAL PROPERTY PSI4_CUEST_IDENTITY", module)
        self.assertIn("get_property(cuEST_IDENTITY_FILE GLOBAL PROPERTY PSI4_CUEST_IDENTITY)", outer)
        self.assertIn("-DcuEST_SUPERBUILD_INNER=ON", outer)

    def test_consumed_header_gate_and_explicit_precedence(self):
        inner = (ROOT / "psi4/CMakeLists.txt").read_text()
        self.assertIn('include_directories(BEFORE "${cuEST_INCLUDE_DIR}")', inner)
        self.assertIn("set(CMAKE_NO_SYSTEM_FROM_IMPORTED ON)", inner)
        self.assertIn("cuest_header_dependencies.py", (ROOT / "psi4/src/CMakeLists.txt").read_text())

    def test_comparison_build_requires_resolved_runtime_and_compiled_direct_route(self):
        script = (ROOT / "devtools/benchmarks/gpu_compare/build_grid_export.sh").read_text()
        self.assertIn("cuda_runtime=${6:?", script)
        self.assertIn('-DcuEST_CUDA_RUNTIME_ROOT="$cuda_runtime"', script)
        self.assertIn('assert direct_available,', script)
        self.assertIn('cuest_consumed_headers=compiled_headers', script)
        self.assertIn('cuest_native_build_identity=native_build_identity', script)

    def test_direct_feature_and_rpath_definitions_preserved(self):
        inner = (ROOT / "psi4/src/psi4/CMakeLists.txt").read_text()
        self.assertIn("USING_cuEST_DIRECT_JK", inner)
        module = (ROOT / "cmake/FindcuEST.cmake").read_text()
        self.assertIn('unset(cuEST_HAS_DIRECT_JK CACHE)', module)
        core = (ROOT / "psi4/src/CMakeLists.txt").read_text()
        for target in ("cuEST::cuEST", "Libxc::xc"):
            self.assertIn(f'list(PREPEND psi4_RPATH "$<TARGET_FILE_DIR:{target}>")', core)

    def configure(self, args, stub):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.21)\nproject(admission NONE)\n"
                f'list(PREPEND CMAKE_MODULE_PATH "{path.as_posix()}" "{(ROOT / "cmake").as_posix()}")\n'
                "if(ENABLE_cuEST)\nfind_package(cuEST 0.2.50 EXACT MODULE REQUIRED)\nendif()\n")
            (path / "FindCUDAToolkit.cmake").write_text(stub)
            return subprocess.run(["cmake", "-S", str(path), "-B", str(path / "build"), *args],
                                  capture_output=True, text=True, timeout=45)

    def test_cmake_cpu_only_does_not_seek_sdk(self):
        result = self.configure(["-DENABLE_cuEST=OFF"], 'message(FATAL_ERROR "must not seek CUDA")')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cmake_rejects_missing_explicit_root(self):
        result = self.configure(["-DENABLE_cuEST=ON"], "set(CUDAToolkit_VERSION_MAJOR 13)")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires explicit cuEST_ROOT", result.stderr)

    def test_cmake_rejects_stale_library_before_hash_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.configure(
                ["-DENABLE_cuEST=ON", f"-DcuEST_ROOT={tmp}",
                 f"-DcuEST_CUDA_RUNTIME_ROOT={tmp}", "-DcuEST_LIBRARY=/old/libcuest.so"],
                "set(CUDAToolkit_VERSION_MAJOR 13)")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Stale/cross-CUDA", result.stderr)


if __name__ == "__main__":
    unittest.main()
