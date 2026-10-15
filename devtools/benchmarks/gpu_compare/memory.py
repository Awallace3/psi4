#!/usr/bin/env python3
"""Linux child peak RSS and sampled process-tree/device memory, outside workers."""
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import threading
import time


def save_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def signal_group(pgid, signum):
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass


def stop_supervisor(proc, usage):
    signal_group(proc.pid, signal.SIGTERM)
    try:
        return proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        # Worker owns a separate group; killing only its supervisor would orphan it.
        try:
            signal_group(json.loads(usage.read_text())["child_pid"], signal.SIGKILL)
        except (OSError, ValueError, KeyError):
            pass
        signal_group(proc.pid, signal.SIGKILL)
        return proc.wait()


def supervise(usage, command):
    """Separate fresh supervisor: RUSAGE_CHILDREN does not accumulate other runs."""
    child = None
    received = None

    def terminate(signum, frame):
        nonlocal received
        received = signum
        if child is not None:
            signal_group(child.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    # The controller masks signals during spawn/PID ownership. Masks survive
    # exec, so unblock only after our cleanup handlers are installed.
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    if received is not None:
        save_json(usage, dict(state="EXITED", child_pid=None, child_returncode=None,
                             maxrss_kib=None, signal_received=received,
                             source="Interrupted before worker launch; RSS unavailable"))
        return 128 + received
    worker_start = time.perf_counter()
    child = subprocess.Popen(command, start_new_session=True)
    if received is not None:
        signal_group(child.pid, signal.SIGTERM)
    save_json(usage, dict(child_pid=child.pid, state="RUNNING"))
    while True:
        try:
            rc = child.wait(timeout=.2)
            break
        except subprocess.TimeoutExpired:
            if received is not None:
                try:
                    rc = child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    signal_group(child.pid, signal.SIGKILL)
                    rc = child.wait()
                break
    full_process_wall_s = time.perf_counter() - worker_start
    # A descendant may ignore TERM even though the direct child has exited.
    # Do not let such work survive into the next benchmark attempt.
    signal_group(child.pid, signal.SIGKILL)
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    save_json(usage, dict(child_pid=child.pid, state="EXITED", child_returncode=rc,
                         maxrss_kib=peak, signal_received=received,
                         full_process_wall_s=full_process_wall_s,
                         source="Linux RUSAGE_CHILDREN.ru_maxrss: largest individual child peak, not tree sum"))
    return 128 + received if received else (rc if rc >= 0 else 128 - rc)


def process_tree(pid, proc=Path("/proc")):
    """Descendants may have been spawned from any thread, not just the main one."""
    found, pending = set(), [pid]
    while pending:
        current = pending.pop()
        if current in found:
            continue
        found.add(current)
        for path in (proc / str(current) / "task").glob("*/children"):
            try:
                pending.extend(int(x) for x in path.read_text().split())
            except (OSError, ValueError):
                continue
    return found


def cpu_sample(pids, proc=Path("/proc")):
    rss, count = 0, 0
    for pid in pids:
        try:
            values = dict(line.split(":", 1) for line in
                          (proc / str(pid) / "status").read_text().splitlines() if ":" in line)
            if "VmRSS" in values:
                rss += int(values["VmRSS"].split()[0])
                count += 1
        except (OSError, ValueError):
            continue
    return rss if count else None


def parse_device(device_text, uuid):
    device_rows = list(csv.reader(device_text.splitlines(), skipinitialspace=True))
    if len(device_rows) != 1 or len(device_rows[0]) != 2 or device_rows[0][0] != uuid:
        raise ValueError("GPU memory sample UUID mismatch")
    return float(device_rows[0][1])


def parse_apps(apps_text, uuid, pids):
    total, unknown = 0., False
    for row in csv.reader(apps_text.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 3:
            raise ValueError("Malformed GPU process sample")
        if row[0] == uuid and int(row[1]) in pids:
            try:
                total += float(row[2])
            except ValueError:
                unknown = True
    return None if unknown else total


def parse_gpu(device_text, apps_text, uuid, pids):
    return parse_device(device_text, uuid), parse_apps(apps_text, uuid, pids)


def gpu_sample(uuid, pids):
    def query(arg):
        return subprocess.check_output(
            ["nvidia-smi", "-i", uuid, arg, "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.PIPE, timeout=3)
    row = dict(gpu_device_mib=None, gpu_process_mib=None, error="")
    for key, arg, parser in (
            ("gpu_device_mib", "--query-gpu=uuid,memory.used", lambda text: parse_device(text, uuid)),
            ("gpu_process_mib", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
             lambda text: parse_apps(text, uuid, pids))):
        try:
            row[key] = parser(query(arg))
            if row[key] is None:
                raise ValueError("NVML process memory unavailable")
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            row["error"] += f"{key}: {type(e).__name__}: {e}; "
    return row


class MemoryMonitor:
    """1 s lower-bound observed peaks. Sampling errors are never coerced to zero."""
    fields = ("unix_s", "stream", "tree_rss_kib", "gpu_process_mib", "gpu_device_mib", "error")

    def __init__(self, root_pid, usage_path, samples_path, uuid, interval=1.0):
        self.root_pid, self.usage_path, self.samples_path = root_pid, usage_path, samples_path
        self.uuid, self.interval = uuid, interval
        self.stop_event = threading.Event()
        self.threads = [threading.Thread(target=self.run, args=(stream,), daemon=True)
                        for stream in ("cpu", "gpu")]
        self.lock = threading.Lock()
        self.peaks = dict(tree_rss_kib=None, gpu_process_mib=None, gpu_device_mib=None)
        self.counts = dict.fromkeys(self.peaks, 0)
        self.samples, self.errors, self.fatal_error = 0, 0, None

    def __enter__(self):
        try:
            self.file = self.samples_path.open("w", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=self.fields, lineterminator="\n")
            self.writer.writeheader()
            for thread in self.threads:
                thread.start()
        except OSError as e:
            self.fatal_error = f"{type(e).__name__}: {e}"
        return self

    def __exit__(self, *exc):
        self.stop_event.set()
        for thread in self.threads:
            if thread.ident is not None:
                thread.join(timeout=10)
        if hasattr(self, "file"):
            self.file.close()

    def run(self, stream):
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                pid = self.root_pid
                try:
                    pid = json.loads(self.usage_path.read_text())["child_pid"]
                except (OSError, ValueError, KeyError):
                    pass
                pids = process_tree(pid)
                row = dict(unix_s=time.time(), stream=stream, tree_rss_kib=None,
                           gpu_process_mib=None, gpu_device_mib=None, error="")
                if stream == "cpu":
                    row["tree_rss_kib"] = cpu_sample(pids)
                else:
                    try:
                        row.update(gpu_sample(self.uuid, pids))
                    except (OSError, ValueError, subprocess.SubprocessError) as e:
                        row["error"] = f"{type(e).__name__}: {e}"[:500]
                # GPU queries have their own thread. Record at completion so a slow
                # query is not attributed to a stale CPU sampling timestamp.
                row["unix_s"] = time.time()
                with self.lock:
                    if row["error"]:
                        self.errors += 1
                    for key in self.peaks:
                        value = row[key]
                        if value is not None:
                            self.peaks[key] = max(self.peaks[key] or 0, value)
                            self.counts[key] += 1
                    self.writer.writerow(row)
                    self.file.flush()  # Preserve observations across controller/step failure.
                    self.samples += 1
                self.stop_event.wait(max(0, self.interval - (time.monotonic() - started)))
        except Exception as e:
            self.fatal_error = f"{type(e).__name__}: {e}"

    def summary(self):
        result = dict(sample_interval_s=self.interval, samples=self.samples,
                      metric_sample_counts=dict(self.counts),
                      gpu_query_errors=self.errors, monitor_error=self.fatal_error,
                      sampled_tree_rss_peak_kib=self.peaks["tree_rss_kib"],
                      sampled_gpu_process_peak_mib=self.peaks["gpu_process_mib"],
                      sampled_gpu_device_peak_mib=self.peaks["gpu_device_mib"],
                      gpu_uuid=self.uuid, samples_file=str(self.samples_path),
                      sampling_note="Observed peaks are lower bounds; tree RSS sums shared pages; "
                                    "NVML process memory includes allocator pools; device memory is whole-device, "
                                    "not attributed to worker. PID visibility depends on host namespace. "
                                    "CPU/GPU streams are independent; slow queries stretch GPU cadence.")
        try:
            usage = json.loads(self.usage_path.read_text())
            if usage["state"] != "EXITED":
                raise ValueError("Supervisor did not finish")
            result.update(peak_rss_kib=usage["maxrss_kib"], peak_rss_source=usage["source"],
                          child_returncode=usage["child_returncode"])
        except (OSError, ValueError, KeyError) as e:
            result.update(peak_rss_kib=None, peak_rss_error=str(e))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usage", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Worker command required")
    return supervise(args.usage, command)


if __name__ == "__main__":
    raise SystemExit(main())
