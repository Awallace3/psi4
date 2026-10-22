"""Offline static inventory capture tests; no GPU packages/device imports."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import capture_cuda12 as capture


class StaticInventory(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = self.root / "env"
        self.env.mkdir()
        self.site = self.env / "site-packages"
        self.site.mkdir()
        self.addCleanup(patch.stopall)
        patch.object(capture.sys, "prefix", str(self.env)).start()
        patch.object(capture.sysconfig, "get_paths", return_value={"purelib": str(self.site)}).start()
        patch.object(capture.importlib.metadata, "distributions", return_value=[]).start()
        self.cmd = patch.object(capture, "command", return_value="source-pin").start()

    def library(self, root, name):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        return path

    def collect(self, extra=()):
        with patch.object(capture, "dynamic", side_effect=lambda p:
                          dict(soname=p.name, needed=[], rpath=[])):
            return capture.collect(self.env, extra)

    def test_capture_is_never_approved_receipt(self):
        path = self.library(self.env / "lib", "libcudart.so.12")
        r = self.collect()
        self.assertIs(r["approved"], False)
        self.assertNotIsInstance(r["libraries"], dict)
        self.assertEqual(r["libraries"][0]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertIsNone(r["libraries"][0]["cuda12_abi_error"])

    def test_duplicate_providers_retained_for_review(self):
        a = self.library(self.env / "lib", "libcudart.so.12")
        outside = self.root / "module"
        b = self.library(outside, "libcudart.so.12")
        r = self.collect([outside])
        self.assertEqual(set(r["providers_by_soname"]["libcudart.so.12"]), {str(a), str(b)})

    def test_wrong_major_marked_not_blessed(self):
        self.library(self.env / "lib", "libcudart.so.13")
        r = self.collect()
        self.assertIn("Non-CUDA12", r["libraries"][0]["cuda12_abi_error"])
        self.assertFalse(r["approved"])

    def test_symlink_alias_cannot_relabel_wrong_abi(self):
        path = self.library(self.env / "lib", "libcudart.so.13")
        (path.parent / "libcudart.so.12").symlink_to(path.name)
        with patch.object(capture, "dynamic", return_value=
                          dict(soname="libcudart.so.12", needed=[], rpath=[])):
            r = capture.collect(self.env, [])
        self.assertIn("differs", r["libraries"][0]["cuda12_abi_error"])
        self.assertEqual(len(r["libraries"][0]["aliases"]), 2)

    def test_prefix_must_match_selected_python(self):
        with self.assertRaisesRegex(RuntimeError, "own Python"):
            capture.collect(self.root, [])

    def test_record_digest_independent_of_library_discovery(self):
        raw = b"expected installed wheel file"
        digest = hashlib.sha256(raw).hexdigest()
        record = SimpleNamespace(hash=SimpleNamespace(
            mode="sha256", value=base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip("=")))
        self.assertTrue(capture.record_proof(record, digest)["matches"])
        self.assertFalse(capture.record_proof(record, "0" * 64)["matches"])
        self.assertIsNone(capture.record_proof(SimpleNamespace(hash=None), digest)["matches"])

    def test_dynamic_requires_unambiguous_soname(self):
        self.cmd.return_value = "(SONAME) Library soname: [libcudart.so.12]\n"
        self.assertEqual(capture.dynamic("file")["soname"], "libcudart.so.12")
        for text in ("", self.cmd.return_value * 2):
            self.cmd.return_value = text
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                capture.dynamic("file")

    def test_source_has_no_gpu_import_or_dlopen(self):
        source = Path(capture.__file__).read_text()
        import ast
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(alias.name.split(".")[0], ("cupy", "gpu4pyscf", "ctypes"))
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or "").split(".")[0], ("cupy", "gpu4pyscf", "ctypes"))


if __name__ == "__main__":
    unittest.main()
