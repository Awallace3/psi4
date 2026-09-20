"""CPU-only unit tests for the memory instrumentation (no GPU, no Psi4 import).

The device sampler is exercised against a stub `nvidia-smi` so the parsing and
the fallback are tested on a machine with no NVIDIA driver at all, which is
where these tests actually run.
"""
import os
from pathlib import Path
import stat
import tempfile
import unittest

import process_memory
from process_memory import DeviceMemorySampler, host_peak_rss_mib, host_rss_mib, reset_host_peak_rss

UUID = "GPU-0000"


def stub_nvidia_smi(directory, compute_apps, gpu="%s, A100-SXM4-80GB, 81920" % UUID):
    """A fake nvidia-smi answering the two queries the sampler issues."""
    path = Path(directory) / "nvidia-smi"
    path.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        f"  --query-gpu=*) printf '%s\\n' '{gpu}' ;;\n"
        f"  --query-compute-apps=*) printf '%b' \"{compute_apps}\" ;;\n"
        "esac\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


class HostMemoryTests(unittest.TestCase):
    def test_peak_is_at_least_current(self):
        # Read the current size first. The process grows between the two reads
        # -- allocating the two strings alone can fault a page in -- so sampling
        # the peak first races against its own invariant.
        current = host_rss_mib()
        self.assertGreaterEqual(host_peak_rss_mib(), current)

    def test_peak_reset_reports_whether_it_took(self):
        # Either outcome is legitimate -- the kernel may not support the reset --
        # but the answer has to be a bool, because the summarizer labels the
        # scope of the number with it rather than assuming.
        self.assertIsInstance(reset_host_peak_rss(), bool)

    def test_reset_peak_drops_a_freed_allocation(self):
        if not reset_host_peak_rss():
            self.skipTest("kernel does not honor /proc/self/clear_refs peak reset")
        big = bytearray(256 * 1024 * 1024)
        big[::4096] = b"\x01" * (len(big) // 4096)  # fault the pages in
        peak_held = host_peak_rss_mib()
        del big
        self.assertGreater(peak_held, 200)
        self.assertTrue(reset_host_peak_rss())
        self.assertLess(host_peak_rss_mib(), peak_held)


class DeviceSamplerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def sampler(self, compute_apps, **kwargs):
        smi = stub_nvidia_smi(self.tmp.name, compute_apps)
        return DeviceMemorySampler(nvidia_smi=smi, interval=0.01, **kwargs)

    def test_counts_only_this_process(self):
        pid = os.getpid()
        sampler = self.sampler(f"{pid}, 4096, {UUID}\\n999999, 70000, {UUID}\\n")
        sampler._describe_devices()
        used, source = sampler._sample()
        self.assertEqual(used, {UUID: 4096.0})
        self.assertEqual(source, "per-process")

    def test_sums_a_process_across_devices(self):
        pid = os.getpid()
        sampler = self.sampler(f"{pid}, 4096, {UUID}\\n{pid}, 1024, GPU-0001\\n")
        used, _ = sampler._sample()
        self.assertEqual(sum(used.values()), 5120.0)

    def test_refused_accounting_reports_none_rather_than_zero(self):
        # A driver that answers "[N/A]" must not be read as "this process used
        # no device memory", which is what silently reporting 0.0 would say.
        sampler = self.sampler("[N/A], [N/A], %s\\n" % UUID)
        used, _ = sampler._sample()
        self.assertIsNone(used)

    def test_falls_back_to_device_wide_and_says_so(self):
        sampler = self.sampler("[N/A], [N/A], %s\\n" % UUID)
        smi = Path(self.tmp.name) / "nvidia-smi"
        smi.write_text(
            "#!/bin/sh\n"
            'case "$1" in\n'
            f"  --query-gpu=uuid,memory.used) printf '%s\\n' '{UUID}, 7777' ;;\n"
            f"  --query-gpu=*) printf '%s\\n' '{UUID}, A100, 81920' ;;\n"
            "  --query-compute-apps=*) printf '%s\\n' '[N/A], [N/A], " + UUID + "' ;;\n"
            "esac\n")
        smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
        sampler.start()
        for _ in range(500):
            if sampler.source == "device-wide" and sampler.peak_mib is not None:
                break
            process_memory.time.sleep(0.01)
        report = sampler.stop()
        self.assertEqual(report["source"], "device-wide")
        self.assertEqual(report["peak_mib"], 7777.0)
        self.assertIn("device-wide", report["measurement"])
        self.assertGreaterEqual(report["samples_without_accounting"], 1)

    def test_peak_survives_a_later_lower_sample(self):
        sampler = self.sampler("")
        sampler.peak_mib = 9000.0
        sampler.peak_by_uuid = {UUID: 9000.0}
        used = {UUID: 10.0}
        for uuid, value in used.items():
            sampler.peak_by_uuid[uuid] = max(sampler.peak_by_uuid.get(uuid, 0.0), value)
        sampler.peak_mib = max(sampler.peak_mib, sum(used.values()))
        self.assertEqual(sampler.peak_mib, 9000.0)

    def test_absent_nvidia_smi_is_reported_not_raised(self):
        # No driver on this host: the benchmark still has to run and produce a
        # record that says why the device figure is missing.
        which = process_memory.shutil.which
        process_memory.shutil.which = lambda name: None
        self.addCleanup(setattr, process_memory.shutil, "which", which)
        report = DeviceMemorySampler().start().stop()
        self.assertIsNone(report["peak_mib"])
        self.assertEqual(report["error"], "nvidia-smi not found")

    def test_unrunnable_nvidia_smi_is_reported_not_raised(self):
        report = DeviceMemorySampler(nvidia_smi="/nonexistent/nvidia-smi").start().stop()
        self.assertIsNone(report["peak_mib"])
        self.assertIn("FileNotFoundError", report["error"])

    def test_report_carries_the_sampling_caveat(self):
        report = self.sampler("").report()
        self.assertIn("sampled peak", report["measurement"])
        self.assertIn("sample_interval_s", report)


if __name__ == "__main__":
    unittest.main()
