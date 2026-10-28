"""Private whole-install and CUDA-copy provenance CPU-only regressions."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import qualify_construction_root as qualify


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
        for directory in (installed,staged,source):
            directory.mkdir(parents=True)
            (directory/"science.py").write_text("science = 'frozen'\n")
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
            self.assertEqual(inventory,{"science.py":qualify.digest(installed/"science.py")})
            clean.assert_called_once_with(root/"source","frozen")

    def test_source_stage_core_and_built_from_drift_each_reject(self):
        for failure in ("source","stage","core","pin","extra_core","missing_core"):
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
                else:
                    (installed/"core.fake.so").unlink()
                with self.assertRaises(RuntimeError):
                    qualify.installed_inventory(root,"frozen")


if __name__ == "__main__":
    unittest.main()
