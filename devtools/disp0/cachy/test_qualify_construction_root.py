"""Private whole-install and CUDA-copy provenance CPU-only regressions."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import qualify_construction_root as qualify
import bootstrap_gate


class Qualification(unittest.TestCase):
    def test_cuda_relocation_matches_bytes_not_source_filenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"runtime/lib"
            directory.mkdir(parents=True)
            first,second = directory/"libcudart.so.13",directory/"libcublas.so.13"
            first.write_bytes(b"runtime")
            second.write_bytes(b"blas")
            (directory/"libcuest.so.0").write_bytes(b"separate pinned SDK")
            original = {"cuda_root":"/gone/tmpfs","cuda_runtime_version":13030,"nvrtc_version":[13,3],
                "components":[{"path":"/gone/tmpfs/libcudart.so.13.3.29","sha256":qualify.digest(first)},
                              {"path":"/gone/tmpfs/libcublas.so.13.6.0.2","sha256":qualify.digest(second)}]}
            frozen = copy.deepcopy(original)
            result = qualify.relocate_manifest(original,directory)
            self.assertEqual(original,frozen)
            self.assertEqual(result["cuda_root"],str(directory.parent))
            self.assertEqual([r["path"] for r in result["components"]],[str(first),str(second)])

    def test_unknown_duplicate_and_missing_private_cuda_bytes_reject(self):
        for failure in ("unknown","duplicate","missing"):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                path = directory/"libcudart.so.13"
                path.write_bytes(b"runtime")
                original = {"cuda_runtime_version":13030,"nvrtc_version":[13,3],
                    "components":[{"path":"/old/runtime","sha256":qualify.digest(path)}]}
                if failure == "unknown":
                    (directory/"foreign.so").write_bytes(b"foreign")
                elif failure == "duplicate":
                    (directory/"alias.so").write_bytes(path.read_bytes())
                else:
                    path.unlink()
                with self.assertRaises(RuntimeError):
                    qualify.relocate_manifest(original,directory)

    def fixture(self,root):
        installed = root/"install/lib/python3.13/site-packages/psi4"
        staged = root/"build/stage/lib/python3.13/site-packages/psi4"
        source = root/"source/psi4"
        configured = root/"build/psi4-core-prefix/src/psi4-core-build"
        configured.mkdir(parents=True)
        for directory in (installed,staged,source):
            directory.mkdir(parents=True)
            (directory/"science.py").write_text("science = 'frozen'\n")
            (directory/"metadata.py").write_text(
                "__version_long = 'test+frozen'\n__version_is_clean = 'True'\n")
        (source/"metadata.py").write_text("__version_long = 'inplace-placeholder'\n")
        for name in ("metadata.py","__init__.py","extras.py"):
            if name != "metadata.py":
                for directory in (installed,staged):
                    (directory/name).write_text("configured = 'private stage'\n")
                (source/name).write_text("configured = '@CONFIGURE_TEMPLATE@'\n")
            (configured/name).write_bytes((installed/name).read_bytes())
        (installed/"core.fake.so").write_bytes(b"whole native core")
        (staged/"core.fake.so").write_bytes(b"whole native core")
        (root/"install/BUILT_FROM").write_text("frozen\n")
        return installed,staged,source

    def test_whole_install_matches_source_stage_core_and_pin(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(qualify,"clean") as clean:
            root = Path(tmp)
            installed,_,_ = self.fixture(root)
            package,core,inventory = qualify.installed_inventory(root,"frozen")
            self.assertEqual(package,installed)
            self.assertEqual(core,installed/"core.fake.so")
            self.assertEqual(inventory,{name:qualify.digest(installed/name)
                                       for name in ("science.py","metadata.py","__init__.py","extras.py")})
            clean.assert_called_once_with(root/"source","frozen")

    def test_source_stage_core_and_built_from_drift_each_reject(self):
        for failure in ("source","stage","core","pin","extra_core","missing_core",
                        "version","dirty_version","configured"):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp, \
                    patch.object(qualify,"clean"):
                root = Path(tmp)
                installed,staged,source = self.fixture(root)
                if failure == "source":
                    (source/"science.py").write_text("changed\n")
                elif failure == "stage":
                    (staged/"science.py").write_text("changed\n")
                elif failure == "core":
                    (installed/"core.fake.so").write_bytes(b"foreign core")
                elif failure == "pin":
                    (root/"install/BUILT_FROM").write_text("foreign\n")
                elif failure == "extra_core":
                    (installed/"core.extra.so").write_bytes(b"extra core")
                elif failure in ("version","dirty_version"):
                    contents = ("__version_long = 'test+foreign'\n__version_is_clean = 'True'\n"
                                if failure == "version" else
                                "__version_long = 'test+frozen'\n__version_is_clean = 'False'\n")
                    for directory in (installed,staged,root/"build/psi4-core-prefix/src/psi4-core-build"):
                        (directory/"metadata.py").write_text(contents)
                elif failure == "configured":
                    (root/"build/psi4-core-prefix/src/psi4-core-build/extras.py").write_text("foreign\n")
                else:
                    (installed/"core.fake.so").unlink()
                with self.assertRaises(RuntimeError):
                    qualify.installed_inventory(root,"frozen")

    def test_actual_sdk_mapping_outside_stage_is_required_and_bootstrap_admits_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage = root/"install"
            stage.mkdir()
            sdk = root/"runtime/libcuest.so.0.2.50"
            sdk.parent.mkdir()
            sdk.write_bytes(b"pinned SDK")
            core = stage/"core.so"
            core.write_bytes(b"private whole core")
            extra = {str(sdk):qualify.digest(sdk)}
            line = lambda p:f"1000-2000 r-xp 00000000 00:01 1234 {p}\n"
            text = line(core)+line(sdk)
            binary = qualify.mapped_binary_inventory(text,(stage,),extra)
            self.assertEqual(binary,{str(core):qualify.digest(core),**extra})
            with self.assertRaisesRegex(RuntimeError,"SDK preload"):
                qualify.mapped_binary_inventory(line(core),(stage,),extra)
            with self.assertRaisesRegex(RuntimeError,"SDK preload"):
                qualify.mapped_binary_inventory(text,(stage,),{str(sdk):"wrong hash"})
            aliases = root/"aliases"
            aliases.mkdir()
            config = {"local_lease":{},"startup_library_path":"private",
                "native_alias_directory":str(aliases),"native_alias_links":{},
                "native_preload_paths":[str(sdk)],"metadata_path":str(root),
                "stage":str(stage),"threads":4}
            proof = {"binary_sha256":binary,"module_files":{"psi4.core":str(core)}}
            class StopAfterPreload(Exception):
                pass
            loader = types.SimpleNamespace(import_psi4_core_first=Mock(side_effect=StopAfterPreload))
            with patch.object(bootstrap_gate,"require_local_lease"), \
                 patch.object(bootstrap_gate,"require_cuda_manifest",return_value={"components":[]}), \
                 patch.object(bootstrap_gate.ctypes,"CDLL") as cdll, \
                 patch.dict(os.environ,{"LD_LIBRARY_PATH":"private","LD_PRELOAD":""}), \
                 patch.dict(sys.modules,{"import_order":loader}):
                with self.assertRaises(StopAfterPreload):
                    bootstrap_gate.bootstrap(config,proof)
                cdll.assert_called_once()
                proof["binary_sha256"] = {str(core):binary[str(core)]}
                with self.assertRaisesRegex(RuntimeError,"Unauthenticated native preload"):
                    bootstrap_gate.bootstrap(config,proof)


if __name__ == "__main__":
    unittest.main()
