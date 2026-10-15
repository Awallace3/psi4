import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import memory


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def supervise(self, code):
        usage = self.root / "usage.json"
        p = subprocess.run([sys.executable, str(Path(memory.__file__)),
                            "--usage", str(usage), "--", sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=10)
        return p, json.loads(usage.read_text())

    def test_failed_worker_still_has_peak_rss(self):
        p, data = self.supervise("x=bytearray(32*1024**2);raise SystemExit(1)")
        self.assertEqual(p.returncode, 1)
        self.assertEqual(data["child_returncode"], 1)
        self.assertGreater(data["maxrss_kib"], 32*1024)

    def test_full_process_clock_includes_imports_and_exit_handlers(self):
        (self.root / "expensive_import.py").write_text("import time\ntime.sleep(.15)\n")
        p, data = self.supervise(
            f"import sys;sys.path.insert(0,{str(self.root)!r});"
            "import expensive_import;import atexit,time;atexit.register(time.sleep,.15)")
        self.assertEqual(p.returncode, 0)
        self.assertGreaterEqual(data["full_process_wall_s"], .30)

    def test_sigkill_worker_still_has_peak_rss(self):
        p, data = self.supervise(
            "import os,signal;x=bytearray(32*1024**2);os.kill(os.getpid(),signal.SIGKILL)")
        self.assertEqual(p.returncode, 137)
        self.assertEqual(data["child_returncode"], -9)
        self.assertGreater(data["maxrss_kib"], 32*1024)

    def test_termination_reaps_worker_and_preserves_peak(self):
        usage, ready = self.root / "usage.json", self.root / "ready"
        code = (f"from pathlib import Path;import time;x=bytearray(32*1024**2);"
                f"Path({str(ready)!r}).touch();time.sleep(100)")
        p = subprocess.Popen([sys.executable, str(Path(memory.__file__)),
                              "--usage", str(usage), "--", sys.executable, "-c", code],
                             start_new_session=True)
        try:
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(ready.exists())
            os.killpg(p.pid, signal.SIGTERM)
            self.assertEqual(p.wait(timeout=5), 143)
            data = json.loads(usage.read_text())
            self.assertGreater(data["maxrss_kib"], 32*1024)
        finally:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()

    def test_gpu_pid_and_uuid_filtering(self):
        device, apps = "gpu, 100\n", "gpu, 1, 30\ngpu, 2, 40\nother, 1, 900\n"
        self.assertEqual(memory.parse_gpu(device, apps, "gpu", {1}), (100, 30))

    def test_gpu_unknown_is_not_zero(self):
        self.assertEqual(memory.parse_gpu("gpu, 100", "gpu, 1, [N/A]", "gpu", {1}),
                         (100, None))
        with self.assertRaises(ValueError):
            memory.parse_gpu("other, 100", "", "gpu", {1})
        with self.assertRaises(ValueError):
            memory.parse_gpu("gpu", "", "gpu", {1})

    def test_process_gpu_query_failure_retains_device_sample(self):
        with patch.object(memory.subprocess, "check_output",
                          side_effect=["gpu, 100", OSError("process query unavailable")]):
            row = memory.gpu_sample("gpu", {1})
        self.assertEqual(row["gpu_device_mib"], 100)
        self.assertIsNone(row["gpu_process_mib"])
        self.assertIn("process query unavailable", row["error"])

    def test_cpu_sampling_is_not_blocked_by_gpu_query(self):
        def slow_gpu(*args):
            time.sleep(.15)
            return dict(gpu_device_mib=100, gpu_process_mib=90, error="")
        usage, samples = self.root / "usage.json", self.root / "samples.csv"
        usage.write_text(json.dumps({"child_pid": os.getpid(), "state": "RUNNING"}))
        with patch.object(memory, "gpu_sample", side_effect=slow_gpu):
            with memory.MemoryMonitor(os.getpid(), usage, samples, "gpu", interval=.01) as mon:
                deadline = time.monotonic() + 3
                while mon.counts["tree_rss_kib"] < 5 and time.monotonic() < deadline:
                    time.sleep(.01)
            summary = mon.summary()
        self.assertGreaterEqual(summary["metric_sample_counts"]["tree_rss_kib"], 5)
        self.assertGreater(summary["metric_sample_counts"]["tree_rss_kib"],
                           summary["metric_sample_counts"]["gpu_device_mib"])

    def test_exited_worker_cannot_leave_term_ignoring_descendant(self):
        pidfile, ready = self.root / "grandchild.pid", self.root / "grandchild.ready"
        childcode = (f"import signal,time;from pathlib import Path;"
                     f"signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                     f"Path({str(ready)!r}).touch();time.sleep(100)")
        code = (f"import subprocess,sys,time;from pathlib import Path;"
                f"p=subprocess.Popen([sys.executable,'-c',{childcode!r}]);"
                f"Path({str(pidfile)!r}).write_text(str(p.pid));"
                f"\nwhile not Path({str(ready)!r}).exists():time.sleep(.01)\n"
                "raise SystemExit(1)")
        try:
            p, data = self.supervise(code)
            self.assertEqual(p.returncode, 1)
            pid = int(pidfile.read_text())
            status = Path(f"/proc/{pid}/status")
            if status.exists():
                self.assertIn("Z", next(l for l in status.read_text().splitlines() if l.startswith("State:")))
        finally:
            if pidfile.exists():
                try:
                    os.kill(int(pidfile.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_thread_children_and_rss_sum(self):
        for pid, children, rss in [(1, "2", 10), (2, "", 20)]:
            directory = self.root / str(pid)
            (directory / "task" / str(pid)).mkdir(parents=True)
            (directory / "task" / str(pid) / "children").write_text(children)
            (directory / "status").write_text(f"Name:\ttest\nVmRSS:\t{rss} kB\n")
        pids = memory.process_tree(1, self.root)
        self.assertEqual(pids, {1, 2})
        self.assertEqual(memory.cpu_sample(pids, self.root), 30)
        self.assertIsNone(memory.cpu_sample({999}, self.root))

    def test_monitor_survives_unavailable_gpu_and_keeps_cpu(self):
        usage, samples = self.root / "usage.json", self.root / "memory.csv"
        usage.write_text(json.dumps({"state": "EXITED", "child_pid": os.getpid(),
                                    "maxrss_kib": 123, "child_returncode": 1, "source": "test"}))
        with patch.object(memory, "gpu_sample", side_effect=OSError("NVML unavailable")):
            with memory.MemoryMonitor(os.getpid(), usage, samples, "gpu", interval=.01) as mon:
                deadline = time.monotonic() + 3
                while not mon.samples and time.monotonic() < deadline:
                    time.sleep(.01)
            data = mon.summary()
        self.assertGreater(data["sampled_tree_rss_peak_kib"], 0)
        self.assertIsNone(data["sampled_gpu_process_peak_mib"])
        self.assertGreater(data["gpu_query_errors"], 0)
        self.assertEqual(data["peak_rss_kib"], 123)
        self.assertIn("NVML unavailable", samples.read_text())

    def test_missing_supervisor_stats_are_explicit(self):
        mon = memory.MemoryMonitor(1, self.root / "missing", self.root / "samples", "gpu")
        data = mon.summary()
        self.assertIsNone(data["peak_rss_kib"])
        self.assertIn("peak_rss_error", data)


if __name__ == "__main__":
    unittest.main()
