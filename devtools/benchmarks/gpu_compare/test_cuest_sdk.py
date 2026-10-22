"""Offline admission tests. No SDK dlopen, native handle, CUDA, or cluster calls."""
import copy
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cuest_sdk as sdk


class SDKTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.library = self.root / "lib/13/libcuest.so.0.2.50"
        self.library.parent.mkdir(parents=True)
        self.library.write_bytes(b"fake ELF, never loaded")
        (self.library.parent / "libcuest.so").symlink_to(self.library.name)
        include = self.root / "include"
        include.mkdir()
        (include / "cuest.h").write_text(
            "#define CUEST_VER_MAJOR 0\n#define CUEST_VER_MINOR 2\n#define CUEST_VER_PATCH 50\n")
        (include / "direct.h").write_text("fake declaration")
        self.contract = copy.deepcopy(sdk.CONTRACT)
        self.contract["headers"] = {name: sdk.sha256(include / name)
                                    for name in ("cuest.h", "direct.h")}
        self.contract["library_sha256"] = sdk.sha256(self.library)
        self.contract["header_tree"] = sdk.header_tree_identity(include)
        self.addCleanup(patch.stopall)
        patch.object(sdk, "CONTRACT", self.contract).start()
        symbols = "\n".join(f"000000 T {name}" for names in self.contract["symbols"].values()
                            for name in names)
        self.nm = patch.object(sdk.subprocess, "check_output", return_value=symbols).start()

    def maps(self, path=None, perms="r-xp"):
        path = path or self.library
        st = path.stat()
        device = f"{os.major(st.st_dev):x}:{os.minor(st.st_dev):x}"
        return f"1000-2000 {perms} 00000000 {device} {st.st_ino} {path}\n"

    def manifest(self):
        return dict(cuest_sdk=str(self.root), cuest_library=str(self.library),
                    cuest_library_sha256=sdk.sha256(self.library),
                    cuest_sdk_identity=sdk.verify_sdk(self.root),
                    cuda_linkage_identity=self.cuda_receipt())

    def cuda_receipt(self, major=13):
        path = self.root / f"libcudart.so.{major}.3.29"
        path.write_bytes(b"mock CUDA library")
        return {f"libcudart.so.{major}": dict(path=str(path), sha256=sdk.sha256(path))}

    def test_static_canonical_sdk_does_not_load_or_call_native(self):
        with patch.object(sdk.ctypes, "CDLL") as load:
            identity = sdk.verify_sdk(self.root)
        load.assert_not_called()
        self.assertTrue(identity["verified"])
        self.assertFalse(identity["native_version_queried"])
        self.assertEqual(set(identity["capabilities"]), {"direct_jk", "dfmo", "version"})
        self.assertEqual(self.nm.call_args.args[0][:3], ["nm", "-D", "--defined-only"])

    def test_wrong_headers_and_library_rejected(self):
        for path in (self.root / "include/direct.h", self.library):
            old = path.read_bytes()
            path.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                sdk.verify_sdk(self.root)
            path.write_bytes(old)

    def test_even_self_consistent_old_header_hash_is_not_version_proof(self):
        p = self.root / "include/cuest.h"
        p.write_text(p.read_text().replace("PATCH 50", "PATCH 2"))
        self.contract["headers"]["cuest.h"] = sdk.sha256(p)
        self.contract["header_tree"] = sdk.header_tree_identity(self.root / "include")
        with self.assertRaisesRegex(RuntimeError, "0.2.50 required"):
            sdk.verify_sdk(self.root)

    def test_exact_unique_version_macros_required(self):
        for text in ("", "#define CUEST_VER_MAJOR 0\n" * 2,
                     "#define CUEST_VER_MAJOR 0\n#define CUEST_VER_MINOR 2\n#define CUEST_VER_PATCH 2\n"):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                sdk.header_version(text)

    def test_lib13_required_no_fallback(self):
        (self.library.parent / "libcuest.so").unlink()
        (self.root / "lib/libcuest.so").write_bytes(self.library.read_bytes())
        with self.assertRaises(FileNotFoundError):
            sdk.verify_sdk(self.root)

    def test_exported_direct_dfmo_and_version_symbols_required(self):
        for group in self.contract["symbols"]:
            missing = self.contract["symbols"][group][0]
            self.nm.return_value = "\n".join(
                f"0000 T {n}" for names in self.contract["symbols"].values()
                for n in names if n != missing)
            with self.subTest(group=group), self.assertRaisesRegex(RuntimeError, group):
                sdk.verify_sdk(self.root)

    def test_old_self_consistent_build_manifest_fails_closed(self):
        manifest = self.manifest()
        self.assertEqual(sdk.verify_build_manifest(manifest)["version"], [0, 2, 50])
        del manifest["cuest_sdk_identity"]
        with self.assertRaisesRegex(RuntimeError, "strict"):
            sdk.verify_build_manifest(manifest)

    def test_tampered_build_receipts_rejected(self):
        manifest = self.manifest()
        for key in ("version", "cuda_major", "header_sha256", "header_tree", "library_sha256", "capabilities"):
            altered = copy.deepcopy(manifest)
            altered["cuest_sdk_identity"][key] = None
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                sdk.verify_build_manifest(altered)
        for key in ("cuest_library", "cuest_library_sha256"):
            altered = copy.deepcopy(manifest)
            altered[key] = "/noncanonical"
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                sdk.verify_build_manifest(altered)

    def test_header_tree_serialization_matches_reconciled_contract(self):
        entries = "".join(f"{name}\0{value}\n" for name, value in
                          sorted(self.contract["headers"].items()))
        tree = sdk.header_tree_identity(self.root / "include")
        self.assertEqual(tree["sha256"], hashlib.sha256(entries.encode("utf-8")).hexdigest())
        self.assertEqual(tree["files"], 2)

    def test_header_tree_sorts_relative_posix_strings_not_path_components(self):
        include = self.root / "include"
        nested = include / "cuest/context.h"
        nested.parent.mkdir()
        nested.write_text("nested canonical fixture")
        records = {p.relative_to(include).as_posix(): sdk.sha256(p)
                   for p in include.rglob("*.h")}
        # "cuest.h" precedes "cuest/context.h" lexically. Path sorting reverses
        # them because it compares directory components rather than POSIX bytes.
        self.assertLess(sorted(records).index("cuest.h"), sorted(records).index("cuest/context.h"))
        serialized = "".join(f"{name}\0{records[name]}\n" for name in sorted(records))
        self.assertEqual(sdk.header_tree_identity(include)["sha256"],
                         hashlib.sha256(serialized.encode()).hexdigest())

    def test_whole_tree_rejects_added_and_missing_headers(self):
        extra = self.root / "include/extra.h"
        extra.write_text("unapproved header")
        with self.assertRaisesRegex(RuntimeError, "whole header tree"):
            sdk.verify_sdk(self.root)
        extra.unlink()
        (self.root / "include/direct.h").unlink()
        with self.assertRaisesRegex(RuntimeError, "whole header tree"):
            sdk.verify_sdk(self.root)

    def test_whole_tree_covers_headers_outside_selected_api_pin_subset(self):
        extra = self.root / "include/unlisted.h"
        extra.write_text("canonical mock dependency")
        self.contract["header_tree"] = sdk.header_tree_identity(self.root / "include")
        sdk.verify_sdk(self.root)
        extra.write_text("changed dependency not in selected subset")
        with self.assertRaisesRegex(RuntimeError, "whole header tree"):
            sdk.verify_sdk(self.root)

    def test_header_tree_rejects_symlink_escape(self):
        outside = self.root / "outside.h"
        outside.write_text("outside include")
        (self.root / "include/escape.h").symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, "escaped include root"):
            sdk.header_tree_identity(self.root / "include")

    def test_actual_mapped_inode_device_and_hash_required(self):
        maps = self.maps() + "/mock/libcudart.so.13.3.29"
        rows, proof = sdk._mapped_identity(maps)
        self.assertEqual(len(rows), 1)
        self.assertEqual(proof["sha256"], self.contract["library_sha256"])
        bad_maps = self.maps().replace(str(self.library.stat().st_ino), "0", 1)
        with self.assertRaisesRegex(RuntimeError, "inode/device"):
            sdk._mapped_identity(bad_maps)
        self.library.write_bytes(b"replacement")
        with self.assertRaisesRegex(RuntimeError, "not canonical"):
            sdk._mapped_identity(self.maps() + "/mock/libcudart.so.13")

    def test_missing_duplicate_deleted_and_unknown_native_maps_rejected(self):
        duplicate = self.root / "libcuest.so.0.2.50"
        duplicate.write_bytes(self.library.read_bytes())
        for maps in ("", self.maps() + self.maps(duplicate),
                     self.maps().rstrip() + " (deleted)\n",
                     self.maps().replace(self.library.name, "libcuest-unknown.so")):
            with self.subTest(maps=maps), self.assertRaises(RuntimeError):
                sdk._mapped_identity(maps + "/mock/libcudart.so.13")

    def test_missing_or_mixed_cuda_runtime_rejected(self):
        for suffix in ("", "/mock/libcudart.so.12", "/mock/libcudart.so.12 /mock/libcudart.so.13"):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(RuntimeError, "CUDA13"):
                sdk._mapped_identity(self.maps() + suffix)

    def test_cuda_closure_actual_maps_hash_and_no_unpinned_library(self):
        receipt = self.cuda_receipt()
        path = Path(receipt["libcudart.so.13"]["path"])
        proof = sdk.mapped_cuda_identity(self.maps(path), receipt)
        self.assertEqual(proof[str(path)]["sha256"], receipt["libcudart.so.13"]["sha256"])
        unpinned = self.root / "libcublas.so.13"
        unpinned.write_bytes(b"not approved")
        for maps in (self.maps(path) + self.maps(unpinned), self.maps(path).rstrip() + " (deleted)"):
            with self.assertRaises(RuntimeError):
                sdk.mapped_cuda_identity(maps, receipt)
        path.write_bytes(b"replacement")
        with self.assertRaisesRegex(RuntimeError, "hash"):
            sdk.mapped_cuda_identity(self.maps(path), receipt)

    def test_cuda_static_linkage_requires_unmixed_pinned_cuda13(self):
        receipt = self.cuda_receipt()
        path = receipt["libcudart.so.13"]["path"]
        self.assertEqual(sdk.cuda_linkage_identity(f"libcudart.so.13 => {path} (0x1234)"), receipt)
        for linkage in ("", "libcudart.so.13 => not found", f"libcudart.so.12 => {path} (0x1234)"):
            with self.subTest(linkage=linkage), self.assertRaises(RuntimeError):
                sdk.cuda_linkage_identity(linkage)

    def test_version_bridge_requires_exact_builtin_int_tuple(self):
        self.assertEqual(sdk.validate_version_receipt((0, 2, 50)), [0, 2, 50])
        for result in ((0, 2, 2), [0, 2, 50], (False, 2, 50), (0., 2, 50),
                       (0, 2), None, {"version": [0, 2, 50]}):
            with self.subTest(result=result), self.assertRaises(RuntimeError):
                sdk.validate_version_receipt(result)

    def test_provider_presence_check_does_not_call_or_initialize(self):
        core = SimpleNamespace(__name__="math", _cuest_runtime_version=math.gcd)
        self.assertIs(sdk.checked_provider(core), math.gcd)
        for value in (None, lambda: (0, 2, 50), math.gcd):
            bad = SimpleNamespace(__name__="psi4.core", _cuest_runtime_version=value)
            with self.assertRaisesRegex(RuntimeError, "provider"):
                sdk.checked_provider(bad)

    def test_native_core_extension_pin_and_provider_are_both_required(self):
        package = self.root / "build/stage/lib/psi4"
        package.mkdir(parents=True)
        extension = package / "core.so"
        extension.write_bytes(b"mock core")
        manifest = dict(package=str(package), core_sha256=sdk.sha256(extension))
        core = SimpleNamespace(__name__="math", __file__=str(extension),
                               _cuest_runtime_version=math.gcd)
        with patch.object(sdk, "manifest_for_package", return_value=manifest):
            self.assertIs(sdk.verify_core(core, package), manifest)
            extension.write_bytes(b"not the pinned core")
            with self.assertRaisesRegex(RuntimeError, "pinned extension"):
                sdk.verify_core(core, package)

    def test_manifest_package_layout_must_be_private_stage(self):
        with self.assertRaisesRegex(RuntimeError, "layout"):
            sdk.manifest_for_package(self.root)

    def test_runtime_rejects_symbol_binding_outside_canonical_executable_map(self):
        with patch.object(sdk, "verify_core", return_value=dict(cuda_linkage_identity={})), \
                patch.object(Path, "read_text", return_value="mock maps"), \
                patch.object(sdk, "_mapped_identity", return_value=(
                    [dict(start=1, stop=100, permissions="r-xp")], dict(path=str(self.library)))), \
                patch.object(sdk, "mapped_cuda_identity", return_value={}), \
                patch.object(sdk.ctypes, "CDLL"), \
                patch.object(sdk.ctypes, "cast", return_value=SimpleNamespace(value=9999)), \
                patch.object(sdk, "checked_provider") as provider:
            with self.assertRaisesRegex(RuntimeError, "outside canonical mapping"):
                sdk.runtime_identity("psi4", object(), self.root)
            provider.assert_not_called()

    def test_no_vendor_call_in_early_check_then_exact_provider_in_late_check(self):
        rows = [dict(start=1, stop=1000, permissions="r-xp")]
        proof = dict(path=str(self.library))
        provider = Mock(return_value=(0, 2, 50))
        with patch.object(sdk, "verify_core", return_value=dict(core_sha256="pin", cuda_linkage_identity={})), \
                patch.object(Path, "read_text", return_value="mock proc maps"), \
                patch.object(sdk, "_mapped_identity", return_value=(rows, proof)), \
                patch.object(sdk, "mapped_cuda_identity", return_value={}), \
                patch.object(sdk, "checked_provider", return_value=provider), \
                patch.object(sdk.ctypes, "CDLL") as load, \
                patch.object(sdk.ctypes, "cast", return_value=SimpleNamespace(value=100)):
            early = sdk.runtime_identity("psi4", object(), self.root, require_version=False)
            provider.assert_not_called()
            self.assertFalse(early["native_version_queried"])
            late = sdk.runtime_identity("psi4", object(), self.root)
            provider.assert_called_once_with()
            self.assertEqual(late["version"], [0, 2, 50])
            self.assertTrue(load.call_args.kwargs["mode"] & os.RTLD_NOLOAD)
            provider.side_effect = RuntimeError("cuEST handle not live")
            with self.assertRaisesRegex(RuntimeError, "not live"):
                sdk.runtime_identity("psi4", object(), self.root)

    def test_pyscf_process_rejects_cuest_and_cuda13_without_loading_any_sdk(self):
        for maps in (self.maps(), "/mock/libcudart.so.13.3.29"):
            with patch.object(Path, "read_text", return_value=maps), \
                    patch.object(sdk.ctypes, "CDLL") as load, self.assertRaises(RuntimeError):
                sdk.runtime_identity("pyscf")
            load.assert_not_called()
        receipt = self.cuda_receipt(12)
        with patch.object(Path, "read_text", return_value=self.maps(Path(receipt["libcudart.so.12"]["path"]))), \
                patch.object(sdk, "read_cuda12_receipt", return_value=dict(libraries=receipt)), \
                patch.dict(os.environ, GPU_COMPARE_CUDA12_RECEIPT_SHA256="0" * 64):
            self.assertTrue(sdk.runtime_identity("pyscf")["cuest_absent"])

    def test_cuda12_cudart_does_not_hide_unpinned_cuda13_blas(self):
        receipt = self.cuda_receipt(12)
        extra = self.root / "libcublas.so.13"
        extra.write_bytes(b"mock CUDA13 BLAS")
        maps = self.maps(Path(receipt["libcudart.so.12"]["path"])) + self.maps(extra)
        with patch.object(Path, "read_text", return_value=maps), \
                patch.object(sdk, "read_cuda12_receipt", return_value=dict(libraries=receipt)):
            with self.assertRaisesRegex(RuntimeError, "outside pinned CUDA12"):
                sdk.runtime_identity("pyscf")
        receipt["libcublas.so.13"] = dict(path=str(extra), sha256=sdk.sha256(extra))
        with self.assertRaisesRegex(RuntimeError, "Non-CUDA12"):
            sdk.verify_cuda_receipt(receipt, major=12)

    def test_cuda12_receipt_cannot_relabel_actual_cuda13_file_or_symlink(self):
        receipt = self.cuda_receipt(12)
        path = self.root / "libcublas.so.13"
        path.write_bytes(b"mock CUDA13 with its accurate hash")
        receipt["libcublas.so.12"] = dict(path=str(path), sha256=sdk.sha256(path))
        maps = self.maps(Path(receipt["libcudart.so.12"]["path"])) + self.maps(path)
        with self.assertRaisesRegex(RuntimeError, "label differs"):
            sdk.verify_cuda_receipt(receipt, major=12)
        with self.assertRaisesRegex(RuntimeError, "label differs"):
            sdk.mapped_cuda_identity(maps, receipt, major=12)
        link = self.root / "libcublas.so.12"
        link.symlink_to(path.name)
        receipt["libcublas.so.12"]["path"] = str(link)
        with self.assertRaisesRegex(RuntimeError, "label differs"):
            sdk.verify_cuda_receipt(receipt, major=12)

    def test_cuda12_receipt_requires_independent_file_pin_and_environment(self):
        path = self.root / "reviewed-cuda12.json"
        value = dict(schema=1, cuda_major=12, python_prefix=sys.prefix,
                     libraries=self.cuda_receipt(12))
        path.write_text(json.dumps(value))
        with patch.dict(os.environ, GPU_COMPARE_CUDA12_RECEIPT=str(path),
                        GPU_COMPARE_CUDA12_RECEIPT_SHA256=sdk.sha256(path)):
            self.assertEqual(sdk.read_cuda12_receipt(), value)
            with self.assertRaisesRegex(RuntimeError, "different Python"):
                sdk.read_cuda12_receipt(self.root)
            path.write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "receipt changed"):
                sdk.read_cuda12_receipt()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Explicit pinned"):
                sdk.read_cuda12_receipt()

    def test_cuda12_environment_is_sanitized_for_controller_probes(self):
        from compare import engine_env
        args = SimpleNamespace(gpu_env=self.root, cuda12_lib=self.root / "cuda12/lib64",
                               threads=8)
        inherited = dict(LD_PRELOAD="/cuda13/libdanger.so",
                         LD_LIBRARY_PATH="/cuda13/lib64", PYTHONPATH="/psi4/source",
                         PYTHONHOME="/wrong", CUPY_CUDA_LIB_PATH="/cuda13/lib64",
                         PATH="/cuda13/bin:/usr/bin")
        with patch.dict(os.environ, inherited), patch("compare.read_cuda12_receipt") as pin:
            env = engine_env("pyscf", args)
        for key in ("LD_PRELOAD", "PYTHONHOME", "CUPY_CUDA_LIB_PATH"):
            self.assertNotIn(key, env)
        self.assertEqual(env["PYTHONPATH"], "")
        self.assertEqual(env["LD_LIBRARY_PATH"], f"{self.root}/lib:{self.root}/cuda12/lib64")
        self.assertEqual(env["CUDA_HOME"], f"{self.root}/cuda12")
        self.assertNotIn("cuda13", env["PATH"])
        pin.assert_called_once_with(self.root)

    def test_device_canary_checks_isolation_before_initialization(self):
        import compare
        fake = SimpleNamespace(cuda=SimpleNamespace(Device=Mock()))
        with patch.dict(sys.modules, cupy=fake), patch.object(sys, "argv", ["compare", "device"]), \
                patch.object(compare, "runtime_identity", side_effect=RuntimeError("bad closure")):
            with self.assertRaisesRegex(RuntimeError, "bad closure"):
                compare.main()
        fake.cuda.Device.assert_not_called()

    def test_device_canary_rechecks_after_initialization(self):
        import compare
        fake = SimpleNamespace(cuda=SimpleNamespace(Device=Mock(), runtime=object()))
        with patch.dict(sys.modules, cupy=fake), patch.object(sys, "argv", ["compare", "device"]), \
                patch.object(compare, "runtime_identity", return_value={"verified": True}) as audit, \
                patch.object(compare, "cuda_identity", return_value={"uuid": "mock"}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(compare.main(), 0)
        self.assertEqual(audit.call_count, 2)
        fake.cuda.Device.return_value.use.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
