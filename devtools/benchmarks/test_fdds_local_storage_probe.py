import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "storage_probe", Path(__file__).with_name("fdds_local_storage_probe.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class StorageProbeTests(unittest.TestCase):
    def test_network_and_memory_filesystems_rejected(self):
        for fs in ("lustre", "nfs", "nfs4", "tmpfs", "unknown"):
            self.assertFalse(probe.local_filesystem(fs))
        for fs in ("xfs", "ext4", "btrfs", "zfs"):
            self.assertTrue(probe.local_filesystem(fs))

    def test_partial_tail_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            report = probe.write_read_probe(root, 65537)
            self.assertEqual(report["bytes"], 65537)
            self.assertTrue(report["fsync"])
            self.assertEqual(len(report["sha256"]), 64)
            before = (root / "write-read-probe.bin").read_bytes()
            with self.assertRaises(FileExistsError):
                probe.write_read_probe(root, 128)
            self.assertEqual((root / "write-read-probe.bin").read_bytes(), before)

    def test_nonpositive_size_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for size in (0, -1):
                with self.assertRaises(ValueError):
                    probe.write_read_probe(Path(d), size)

    def test_job_binding_rejected_before_storage_mutation(self):
        for env in ({"SLURM_JOB_ID": "1", "CUEST_STORAGE_JOB_ID": "2"},
                    {"SLURM_JOB_ID": "0", "CUEST_STORAGE_JOB_ID": "0"}):
            with patch.dict(os.environ, env, clear=True), patch.object(Path, "mkdir") as mkdir:
                with self.assertRaises(AssertionError):
                    probe.main()
                mkdir.assert_not_called()


if __name__ == "__main__":
    unittest.main()
