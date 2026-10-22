# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Exercise the actual map scanner without importing science or hashing libraries."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "benzene_job_snapshot_test", Path(__file__).with_name("benzene_job.py"))
JOB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(JOB)


class MapSnapshotTests(unittest.TestCase):
    def scan(self, extra):
        libraries = (
            JOB.OLD / "lib/libEinsums.so.1.1.3",
            JOB.ENV / "lib/libomp.so",
            JOB.ENV / "lib/libmkl_gnu_thread.so.2",
        )
        rows = extra + "".join(
            f"1000-2000 r-xp 00000000 00:01 1 {path}\n" for path in libraries)
        original_read = Path.read_text

        def read(path, *args, **kwargs):
            if path == Path("/proc/self/maps"):
                return rows
            if path.name in ("dependency-hashes.json", "after-tests.json"):
                return "{}"
            return original_read(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metadata").mkdir()
            with mock.patch.object(JOB, "RUN", root), \
                    mock.patch.object(JOB, "sha", return_value="test-hash") as hasher, \
                    mock.patch.object(Path, "read_text", new=read):
                JOB.snapshot("test")
                self.assertEqual(hasher.call_count, len(libraries))
            recorded = json.loads((root / "metadata/test.json").read_text())
            self.assertEqual(set(recorded), {str(path.resolve()) for path in libraries})

    def test_unlinked_tmpfile_aliases_do_not_block_library_identity(self):
        # Same non-library pathname and W^X alias pattern as failed job14256310.
        self.scan(
            "0000-1000 rw-s 00000000 fd:01 42 /tmp/#42 (deleted)\n"
            "1000-2000 r-xs 00000000 fd:01 42 /tmp/#42 (deleted)\n")

    def test_deleted_numerical_library_is_still_rejected(self):
        path = JOB.OLD / "lib/libEinsums.so.1.1.3"
        with self.assertRaisesRegex(ValueError, "deleted"):
            self.scan(f"0000-1000 r-xp 00000000 00:01 1 {path} (deleted)\n")

    def test_deleted_core_extension_is_still_rejected(self):
        path = JOB.STAGE / "lib/psi4/core.cpython-313-x86_64-linux-gnu.so"
        with self.assertRaisesRegex(ValueError, "deleted"):
            self.scan(f"0000-1000 r-xp 00000000 00:01 1 {path} (deleted)\n")

    def test_other_deleted_shared_object_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "deleted"):
            self.scan(
                "0000-1000 r-xp 00000000 00:01 1 /private/other_extension.so (deleted)\n")

    def test_unrelated_deleted_pathname_with_spaces_is_ignored(self):
        self.scan("0000-1000 rw-s 00000000 00:01 1 /dev/shm/shared memory (deleted)\n")

    def test_deleted_runtime_libraries_are_not_whitelisted_by_namespace(self):
        for name in ("libmkl_core.so.2", "libomp.so",
                     "libEinsums.so.1.1.3", "core.cpython-313-x86_64-linux-gnu.so"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "deleted"):
                self.scan(f"0000-1000 r-xp 00000000 00:01 1 /dev/shm/{name} (deleted)\n")

    def test_anonymous_mappings_and_duplicate_segments_are_harmless(self):
        library = JOB.OLD / "lib/libEinsums.so.1.1.3"
        self.scan(
            "0000-1000 rw-p 00000000 00:00 0\n"
            "1000-2000 rw-p 00000000 00:00 0 [anonymous]\n"
            f"2000-3000 r--p 00000000 00:01 1 {library}\n")


if __name__ == "__main__":
    unittest.main()
