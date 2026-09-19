#!/usr/bin/env python3
"""Host and device memory measurement for the paired CPU/GPU benchmark cases.

A speedup table that omits memory is not comparable: the GPU arm can only be run
at all if its host *and* device footprints fit the allocation, and the memory
work this campaign is rerun against changes the host side specifically. Both
numbers are therefore recorded next to every wall time.

Host memory is the kernel's own high-water mark (``VmHWM``), not a sample, so it
cannot miss a peak. ``/proc/self/clear_refs`` resets that mark, which is what
makes it possible to report the peak of the timed region alone rather than the
peak of a process that has already imported Psi4 and built three basis sets.
Whether the reset took is recorded, because on a kernel without
``CONFIG_PROC_PAGE_MONITOR`` the write is silently a no-op and the mark then
still covers the whole process.

Device memory has no equivalent: NVML reports what is resident now, so the peak
has to be sampled and a spike shorter than the interval is missed. Every device
figure therefore travels with its sample count and interval, and is labeled a
sampled peak. Per-process accounting is used in preference to device-wide use so
another tenant on a shared GPU cannot inflate the number; when the driver
refuses it (MIG, some virtualized setups) that is reported rather than quietly
substituted.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

KIB_PER_MIB = 1024.0


def _status_kib(field, pid="self"):
    """One /proc/<pid>/status field in KiB, or None if the kernel does not offer it."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def host_rss_mib():
    """Resident set size right now, MiB."""
    value = _status_kib("VmRSS")
    return None if value is None else value / KIB_PER_MIB


def host_peak_rss_mib():
    """Highest resident set size since the process started or the mark was last reset, MiB."""
    value = _status_kib("VmHWM")
    return None if value is None else value / KIB_PER_MIB


def reset_host_peak_rss():
    """Try to reset VmHWM to the current VmRSS. True only if the mark actually moved.

    The write itself succeeds on kernels that ignore it, so success is confirmed
    against the mark rather than against the return of write().
    """
    before = host_peak_rss_mib()
    current = host_rss_mib()
    if before is None or current is None:
        return False
    try:
        Path("/proc/self/clear_refs").write_text("5\n")
    except OSError:
        return False
    after = host_peak_rss_mib()
    if after is None:
        return False
    # Nothing to prove if the mark was already at the current set size.
    return after <= current + 1.0 and (before <= current + 1.0 or after < before)


class DeviceMemorySampler:
    """Sampled peak device memory of one process, via nvidia-smi.

    Polls per-process compute-app accounting on a background thread. The parent
    is sampled together with its children because each benchmark case runs the
    calculation in a fresh subprocess; when the sampler is started inside that
    subprocess the child set is simply empty.
    """

    def __init__(self, pid=None, interval=0.5, nvidia_smi=None):
        self.pid = os.getpid() if pid is None else pid
        self.interval = interval
        self.nvidia_smi = nvidia_smi or shutil.which("nvidia-smi")
        self._stop = threading.Event()
        self._thread = None
        self.samples = 0
        self.peak_mib = None
        self.peak_by_uuid = {}
        self.unavailable_samples = 0
        self.failed_samples = 0
        self.source = "per-process"
        self.error = None
        self.devices = {}

    def _query(self, args):
        out = subprocess.run([self.nvidia_smi] + args, capture_output=True, text=True, timeout=20)
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip() or f"nvidia-smi exited {out.returncode}")
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]

    def _describe_devices(self):
        for line in self._query(["--query-gpu=uuid,name,memory.total", "--format=csv,noheader,nounits"]):
            uuid, name, total = (field.strip() for field in line.split(",", 2))
            self.devices[uuid] = {"name": name, "memory_total_mib": float(total)}

    def _sample(self):
        """Total MiB this process tree holds, keyed by GPU uuid. None means accounting refused."""
        rows = self._query(["--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
                            "--format=csv,noheader,nounits"])
        mine = {}
        refused = False
        pids = self._pids()
        for line in rows:
            pid_text, used_text, uuid = (field.strip() for field in line.split(",", 2))
            # A consumer driver reports "[N/A]" for either field rather than omitting the row.
            if not pid_text.isdigit() or not used_text.replace(".", "", 1).isdigit():
                refused = True
                continue
            if int(pid_text) not in pids:
                continue
            mine[uuid] = mine.get(uuid, 0.0) + float(used_text)
        if mine:
            return mine, "per-process"
        if refused:
            return None, "per-process"
        return {}, "per-process"

    def _sample_device_wide(self):
        """Fallback for drivers without per-process accounting: everything resident on the GPU."""
        used = {}
        for line in self._query(["--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"]):
            uuid, value = (field.strip() for field in line.split(",", 1))
            if value.replace(".", "", 1).isdigit():
                used[uuid] = float(value)
        return used, "device-wide"

    def _pids(self):
        pids = {self.pid}
        try:
            for task in Path(f"/proc/{self.pid}/task").iterdir():
                children = (task / "children").read_text().split()
                pids.update(int(child) for child in children)
        except OSError:
            pass
        return pids

    def _loop(self):
        while not self._stop.is_set():
            try:
                if self.source == "device-wide":
                    mine, source = self._sample_device_wide()
                else:
                    mine, source = self._sample()
                    if mine is None:
                        # The driver refuses per-process accounting, and will keep refusing.
                        # Device-wide use is a different quantity, so say which one is reported.
                        self.unavailable_samples += 1
                        self.samples += 1
                        self.source = "device-wide"
                        self._stop.wait(self.interval)
                        continue
            except Exception as exc:  # a transient nvidia-smi failure must not end the benchmark
                self.failed_samples += 1
                self.error = self.error or f"{type(exc).__name__}: {exc}"
                self._stop.wait(self.interval)
                continue
            self.samples += 1
            self.source = source
            for uuid, used in mine.items():
                self.peak_by_uuid[uuid] = max(self.peak_by_uuid.get(uuid, 0.0), used)
            total = sum(mine.values())
            self.peak_mib = total if self.peak_mib is None else max(self.peak_mib, total)
            self._stop.wait(self.interval)

    def start(self):
        if self.nvidia_smi is None:
            self.error = "nvidia-smi not found"
            return self
        try:
            self._describe_devices()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return self
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5 * self.interval + 20)
        return self.report()

    def report(self):
        return {
            "measurement": f"sampled peak of {self.source} NVML accounting; a spike shorter "
                           "than the interval is missed",
            "source": self.source,
            "sample_interval_s": self.interval,
            "samples": self.samples,
            "samples_without_accounting": self.unavailable_samples,
            "failed_samples": self.failed_samples,
            "peak_mib": self.peak_mib,
            "peak_by_uuid_mib": self.peak_by_uuid or None,
            "devices": self.devices or None,
            "error": self.error,
        }


def host_report(peak_reset_ok, rss_before_mib, committed_before_doubles=None,
                committed_after_doubles=None):
    """Host side of the record, read at the end of the timed region."""
    return {
        "measurement": "kernel VmHWM, an exact high-water mark rather than a sample",
        "peak_covers_timed_region_only": bool(peak_reset_ok),
        "rss_before_mib": rss_before_mib,
        "peak_rss_mib": host_peak_rss_mib(),
        "rss_after_mib": host_rss_mib(),
        "ledger_committed_before_mib": _doubles_to_mib(committed_before_doubles),
        "ledger_committed_after_mib": _doubles_to_mib(committed_after_doubles),
    }


def _doubles_to_mib(doubles):
    return None if doubles is None else doubles * 8 / (1024.0 * 1024.0)


if __name__ == "__main__":
    sampler = DeviceMemorySampler().start()
    time.sleep(2)
    print(json.dumps({"host": host_report(reset_host_peak_rss(), host_rss_mib()),
                      "device": sampler.stop()}, indent=2))
