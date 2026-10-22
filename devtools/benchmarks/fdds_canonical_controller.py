"""Compute-node owner controller. Reap descendants before claiming cleanup."""
import ctypes
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def stamp(path):
    path.write_text(datetime.datetime.now(datetime.timezone.utc).isoformat() + "\n")


def gpu_empty(uuid, path):
    result = subprocess.run(
        ["nvidia-smi", "-i", uuid, "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    path.write_text(result.stdout)
    path.with_suffix(".stderr").write_text(result.stderr)
    result.check_returncode()
    return all(not line.strip() or line.strip() == "No running processes found"
               for line in result.stdout.splitlines())


def validate_job(text):
    fields = dict(re.findall(r"(\w+)=([^\s]+)", text))
    for key, expected in {"QOS": "embers", "Account": "gts-cs207-chemx",
                          "Requeue": "0", "TimeLimit": "04:00:00", "NumNodes": "1",
                          "NumTasks": "1", "NumCPUs": "8", "MinMemoryNode": "128G"}.items():
        assert fields.get(key) == expected, (key, fields.get(key), expected)
    assert fields.get("TresPerNode", "").lower() == "gres/gpu:h200:1"
    return fields


def children(pid):
    """Immediate process descendants. Fail closed on an unreadable live owner."""
    path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        return [int(p) for p in path.read_text().split()]
    except FileNotFoundError:
        return []


def descendants():
    todo = children(os.getpid())
    found = []
    while todo:
        pid = todo.pop()
        try:
            # PID+starttime prevents signalling an unrelated reused PID.
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            continue
        found.append((pid, stat[19]))
        todo.extend(children(pid))
    return found


def signal_owned(sig):
    # All children of this subreaper belong to the one admitted payload.
    for pid, started in reversed(descendants()):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if stat[19] == started:
                os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except FileNotFoundError:
            pass


def reap():
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                return
        except ChildProcessError:
            return


def supervised(command, seconds, out, grace=30):
    """Bound a payload, including double-fork/setsid descendants, on Linux."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Unable to establish child ownership")
    interrupted = []
    old = {sig: signal.signal(sig, lambda s, _f: interrupted.append(s))
           for sig in (signal.SIGTERM, signal.SIGINT)}
    proc = None
    clean = False
    rc = 125
    try:
        proc = subprocess.Popen(command, start_new_session=True)
        deadline = time.monotonic() + seconds
        while proc.poll() is None and not interrupted and time.monotonic() < deadline:
            time.sleep(.1)
        if interrupted:
            rc = 128 + interrupted[0]
        elif proc.poll() is None:
            rc = 124
        else:
            rc = proc.returncode
            if rc < 0:
                rc = 128 - rc
        # Even an exit-zero leader can leave orphaned workers. They are not
        # allowed to survive into cleanup/timing, nor counted as a clean pass.
        leftovers = descendants()
        if leftovers:
            if rc == 0:
                rc = 125
            signal_owned(signal.SIGTERM)
            until = time.monotonic() + grace
            while descendants() and time.monotonic() < until:
                if proc.poll() is not None:
                    reap()
                time.sleep(.1)
            signal_owned(signal.SIGKILL)
        proc.wait(timeout=5)
        until = time.monotonic() + 5
        while descendants() and time.monotonic() < until:
            reap()
            time.sleep(.05)
        reap()
        clean = not descendants()
    finally:
        if not clean:
            signal_owned(signal.SIGKILL)
            if proc is not None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            reap()
        for sig, handler in old.items():
            signal.signal(sig, handler)
        (out / "containment.json").write_text(json.dumps(
            {"exit_code": rc, "cleanup_verified": clean, "remaining": descendants()}, indent=2) + "\n")
    return rc if clean else 125


def main():
    assert len(sys.argv) == 7, "SOURCE PIN HARNESS HARNESS_PIN ENV RUN required"
    source, pin, harness, harness_pin, dep, run = sys.argv[1:]
    source, harness, run = map(Path, (source, harness, run))
    assert all(p.is_absolute() and p.is_dir() for p in (source, harness, run, Path(dep)))
    job = os.environ["SLURM_JOB_ID"]
    assert job.isdecimal() and os.environ["SLURM_CPUS_PER_TASK"] == "8"
    for root, revision in ((source, pin), (harness, harness_pin)):
        assert re.fullmatch("[0-9a-f]{40}", revision)
        assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == revision
        assert not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True)
    out = run / ("results-" + job)
    out.mkdir()
    stamp(out / "STARTED")
    state = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True, timeout=15)
    (out / "job.txt").write_text(state)
    validate_job(state)
    # NVML is not filtered by CUDA_VISIBLE_DEVICES. Select only the GPU assigned
    # to this step (Slurm global index or UUID); never inspect a neighbour.
    selected = os.environ.get("SLURM_STEP_GPUS", os.environ.get("SLURM_JOB_GPUS", ""))
    assert re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+)", selected), selected
    identity = subprocess.check_output(
        ["nvidia-smi", "-i", selected, "--query-gpu=uuid,name,driver_version",
         "--format=csv,noheader"], text=True, timeout=15)
    (out / "gpu.csv").write_text(identity)
    assert len(identity.splitlines()) == 1 and "H200" in identity
    uuid = identity.split(",")[0].strip()
    assert uuid.startswith("GPU-")
    with (run / "validation.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert gpu_empty(uuid, out / "gpu-before.csv"), "GPU busy before payload"
        rc = 125
        try:
            rc = supervised(["bash", str(harness / "devtools/benchmarks/fdds_canonical_payload.sh"),
                             str(source), pin, dep, str(out)], 13500, out)
        finally:
            clean = json.loads((out / "containment.json").read_text())["cleanup_verified"]
            try:
                clean = gpu_empty(uuid, out / "gpu-after.csv") and clean
            except (subprocess.SubprocessError, OSError):
                clean = False
            stamp(out / ("GPU_CLEANUP_VERIFIED" if clean else "QUARANTINED"))
            stamp(out / "TERMINAL")
        if not clean:
            rc = 125
        (out / "payload-exit.txt").write_text(str(rc) + "\n")
        if rc == 0:
            native = json.loads((out / "native/verified.json").read_text())
            assert native["tests"] > 0 and all(native[k] == 0 for k in ("failures", "errors", "skipped"))
            gpu = json.loads((out / "gpu/VERIFIED.json").read_text())
            # fdds_validate.sh owns exact XML/GDB/admission checks; retain that
            # verified summary and also check the actual full-suite XML here.
            import xml.etree.ElementTree as ET
            suites = list(ET.parse(out / "gpu/full-fdds.xml").getroot().iter("testsuite"))
            counts = {k: sum(int(s.get(k, "0")) for s in suites)
                      for k in ("tests", "failures", "errors", "skipped")}
            assert counts == {"tests": 112, "failures": 0, "errors": 0, "skipped": 0}, counts
            assert gpu["mode"] == "focused_then_full"
            assert gpu["executed_tests_including_repeats"] == 176
            assert gpu["unique_full_suite_tests"] == 112
            stamp(out / "COMPLETED")
        return rc


if __name__ == "__main__":
    sys.exit(main())
