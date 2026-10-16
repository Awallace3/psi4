#!/usr/bin/env python3
"""User-approved serial insertion in job14097678; never pause a chemistry worker.

Stop only the verified pilot controller, drain its live worker, insert small
checks, then always resume. A pipe watchdog resumes the controller on main-process
death. No source changes, job cancellation, or overlapping chemistry timings.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import tiny_grid_pilot as pilot

BASE = Path("/storage/project/r-cs207-0/awallace43/runs/saptdft-tiny-grid-v1/pilot-14097678")
CONTROLLER = "/storage/project/r-cs207-0/awallace43/gits/psi4.saptdft-cuest-tiny-grid-pilot-v1/devtools/benchmarks/tiny_grid_pilot.py"
EXTRAS = dict(tiny08=((75, 302), (20, 86)),
              tiny05=((75, 302), (15, 74)))


def proc(pid):
    directory = Path(f"/proc/{pid}")
    try:
        if directory.stat().st_uid != os.getuid():
            return None
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=pid, state=fields[0], ppid=int(fields[1]),
                    start=fields[19], cmd=(directory / "cmdline").read_bytes().split(b"\0"))
    except (OSError, ValueError, IndexError):
        return None


def matches_controller(p):
    return p is not None and os.fsencode(CONTROLLER) in p["cmd"] and b"campaign" in p["cmd"]


def find_controller():
    values = [p for d in Path("/proc").iterdir() if d.name.isdigit()
              and matches_controller(p := proc(int(d.name)))]
    assert len(values) == 1, f"Expected exactly one controller, found {len(values)}"
    return values[0]


def resume(controller):
    current = proc(controller["pid"])
    if matches_controller(current) and current["start"] == controller["start"]:
        os.kill(controller["pid"], signal.SIGCONT)


def remaining():
    runtime = dict(line.split("=", 1) for line in
                   (BASE / "metadata/runtime.env").read_text().splitlines() if "=" in line)
    return max(0., 7200.-(datetime.now().astimezone()-
                         datetime.fromisoformat(runtime["started_at"])).total_seconds())


def workers(controller):
    return [p for d in Path("/proc").iterdir() if d.name.isdigit()
            and (p := proc(int(d.name))) is not None and p["ppid"] == controller["pid"]
            and b"worker" in p["cmd"] and p["state"] not in ("Z", "X")]


def run(args):
    assert os.environ["SLURM_JOB_ID"] == "14097678"
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == ""
    assert subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[2]),
                                    "rev-parse", "HEAD"], text=True).strip() == args.pin
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    controller = find_controller()
    metadata = dict(job_id="14097678", controller=controller["pid"], approved_pause=True,
                    pin=args.pin, cases=[], overlap=False)
    pilot.grid.atomic(out / "manifest.json", metadata)
    read_pipe, write_pipe = os.pipe()
    watchdog = os.fork()
    if watchdog == 0:
        os.close(write_pipe)
        try:
            completed = os.read(read_pipe, 1)
            if not completed:
                resume(controller)
        finally:
            os._exit(0)
    os.close(read_pipe)

    def terminate(_signum, _frame):
        raise InterruptedError("Insertion interrupted")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        os.kill(controller["pid"], signal.SIGSTOP)
        for _ in range(40):
            if proc(controller["pid"])["state"] in ("T", "t"):
                break
            time.sleep(.05)
        assert proc(controller["pid"])["state"] in ("T", "t")
        print("CONTROLLER_PAUSED", controller["pid"], "remaining_s", round(remaining()), flush=True)
        # The normal controller enforces a 1500s worker timeout. Do not leave
        # that enforcement suspended indefinitely if a chemistry call stalls.
        deadline = time.monotonic()+min(1500., max(0., remaining()-180))
        while active := workers(controller):
            uptime = float(Path("/proc/uptime").read_text().split()[0])
            if any(uptime-int(p["start"])/os.sysconf("SC_CLK_TCK") >= 1495
                   for p in active):
                raise TimeoutError("Worker approaching its original timeout; resuming controller")
            if time.monotonic() >= deadline:
                raise TimeoutError("Active chemistry did not drain; resuming normal timeout enforcement")
            time.sleep(2)
        print("ACTIVE_WORKER_DRAINED; starting serial insertions", flush=True)
        for system in ("water", "peptide"):
            for arm in EXTRAS:
                if remaining() < 480:
                    metadata["stopped_for_headroom"] = True
                    break
                directory = out / f"{system}-{arm}"
                directory.mkdir()
                with tempfile.TemporaryDirectory(prefix="tiny-extra-", dir=os.environ["TMPDIR"]) as scratch:
                    command = [sys.executable, str(Path(__file__).resolve()), "worker",
                        "--system", system, "--arm", arm, "--output", str(directory),
                        "--expected-package", str(args.expected_package), "--variant", "cpu",
                        "--threads", str(args.threads), "--memory", "80 GiB"]
                    env = dict(os.environ, TMPDIR=scratch, SCRATCH=scratch, PSI_SCRATCH=scratch,
                               CUDA_VISIBLE_DEVICES="")
                    print("START", system, arm, flush=True)
                    with (directory / "console.log").open("w") as log:
                        child = subprocess.Popen(command, cwd=scratch, env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
                        try:
                            rc = child.wait(timeout=min(360., remaining()-120))
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                            rc = 124
                        except BaseException:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                            raise
                metadata["cases"].append(dict(system=system, arm=arm, returncode=rc,
                                               result=str(directory / "result.json")))
                pilot.grid.atomic(out / "manifest.json", metadata)
                print("END", system, arm, "rc", rc, flush=True)
        metadata["complete"] = len(metadata["cases"]) == 4
        pilot.grid.atomic(out / "manifest.json", metadata)
    finally:
        resume(controller)
        os.write(write_pipe, b"1")
        os.close(write_pipe)
        os.waitpid(watchdog, 0)
        print("CONTROLLER_RESUMED", controller["pid"], flush=True)


def main():
    pilot.CONFIGS.update(EXTRAS)
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        return pilot.main()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-package", type=Path, required=True)
    p.add_argument("--pin", required=True)
    p.add_argument("--threads", type=int, default=16)
    args = p.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
