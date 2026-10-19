"""Host-only regressions for the pre-submit repair. No native build or CUDA."""
import hashlib
import os
from pathlib import Path
import tempfile
import subprocess
import unittest
import common as n
import handoff


class FactorGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"])
        self.root = Path(self.tmp.name)
        self.files = {}
        for name, data in (("FACTOR.json", b'{"original":true}\n'),
                           ("C.npy", b"original C"), ("D.npy", b"original D")):
            p = self.root/name
            p.write_bytes(data)
            p.chmod(0o600 if name == "FACTOR.json" else 0o444)
            self.files[name] = hashlib.sha256(data).hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def stats(self):
        return {p.name: (p.stat().st_ino, p.stat().st_size,
                         p.stat().st_mtime_ns, p.stat().st_mode)
                for p in self.root.iterdir()}

    def test_original_0600_receipt_0444_arrays(self):
        before = self.stats()
        n.check_factor(self.root, self.files)
        self.assertEqual(self.stats(), before)

    def test_writable_array_rejected(self):
        for name in ("C.npy", "D.npy"):
            with self.subTest(name=name):
                p = self.root/name
                p.chmod(0o644)
                with self.assertRaisesRegex(RuntimeError, name):
                    n.check_factor(self.root, self.files)
                p.chmod(0o444)

    def test_changed_receipt_bytes_rejected(self):
        (self.root/"FACTOR.json").write_bytes(b"changed receipt")
        with self.assertRaisesRegex(RuntimeError, "FACTOR.json"):
            n.check_factor(self.root, self.files)

    def test_changed_array_bytes_rejected(self):
        for name in ("C.npy", "D.npy"):
            with self.subTest(name=name):
                p = self.root/name
                data = p.read_bytes()
                p.chmod(0o644)
                p.write_bytes(b"changed array")
                p.chmod(0o444)
                with self.assertRaisesRegex(RuntimeError, name):
                    n.check_factor(self.root, self.files)
                p.chmod(0o644)
                p.write_bytes(data)
                p.chmod(0o444)

    def test_missing_file_rejected(self):
        for name in self.files:
            with self.subTest(name=name):
                p = self.root/name
                data, mode = p.read_bytes(), p.stat().st_mode & 0o777
                p.unlink()
                with self.assertRaises(FileNotFoundError):
                    n.check_factor(self.root, self.files)
                p.write_bytes(data)
                p.chmod(mode)

    def test_export_uses_actual_guard(self):
        self.assertIn("n.check_factor(n.FACTOR,n.FACTOR_FILES)",
                      (n.HERE/"export.py").read_text())


class UUIDTests(unittest.TestCase):
    def test_production_formatter_exact_bytes(self):
        # Only this tiny CUDA-free formatter is compiled, never native.cc.
        with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as tmp:
            exe = Path(tmp)/"uuid-host-check"
            source = r'''
#include "uuid.h"
#include <iostream>
int main() {
    char b[16];
    const unsigned values[16]={0,1,2,3,0x80,0xff,0xa0,0x0b,0x10,0x20,0x30,0x40,0x50,0x60,0x70,0x90};
    for (int k=0;k<16;++k) b[k]=static_cast<char>(values[k]);
    std::cout << native_plan::format_uuid(b) << "\n";
}
'''
            subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-I"+str(n.HERE), "-x", "c++", "-", "-o", str(exe)],
                           input=source, text=True, capture_output=True, check=True, timeout=30)
            self.assertEqual(subprocess.check_output([exe], text=True, timeout=5),
                             "GPU-00010203-80ff-a00b-1020-304050607090\n")

    def test_supported_properties_and_physical_matching_path(self):
        source = (n.HERE/"native.cc").read_text()
        self.assertNotIn("cudaDeviceGetUuid", source)
        for text in ("CA(cudaGetDeviceProperties(&props,0))",
                     "format_uuid(props.uuid.bytes)", "need(uuid==argv[4]",
                     "Replay b(in,argv[3],h,d,uuid)",
                     '"nvidia-smi --id="+uuid+', 'data.substr(0,comma)==uuid',
                     'id==getpid()', 'need(rows==1,"one native GPU holder")'):
            self.assertIn(text, source)
        controller = (n.HERE/"run.py").read_text()
        for text in ('uuid=proof["uuid"]', '"--id="+uuid', 'gpu["uuid"]==uuid',
                     '[exe,inp/"basis.bin",inp/"D.bin",directory,uuid]'):
            self.assertIn(text, controller)


class StagingTests(unittest.TestCase):
    def test_original_unsubmitted_staging_is_preserved_not_a_jobroot(self):
        with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as tmp:
            root = Path(tmp)
            (root/"logs").mkdir()
            for name in ("manifest.json", "campaign.sbatch", "bundle-receipt.json"):
                n.durable(root/name, {})
            before = handoff.unsubmitted_stage(root)
            value = dict(roots={}, builds={},
                         small_hashes={str(root/k): n.sha(root/k) for k in before},
                         unsubmitted_staging={str(root): before})
            handoff.preserve(value)
            self.assertEqual(len(value["roots"]), 0)
            (root/"manifest.json").chmod(0o600)
            with self.assertRaisesRegex(RuntimeError, "immutable original native staging"):
                handoff.preserve(value)
            (root/"manifest.json").chmod(0o444)
            (root/"logs"/"unexpected-attempt").mkdir()
            with self.assertRaisesRegex(RuntimeError, "unsubmitted staging layout"):
                handoff.preserve(value)


if __name__ == "__main__":
    unittest.main()
