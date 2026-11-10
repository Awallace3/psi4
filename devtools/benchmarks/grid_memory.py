#!/usr/bin/env python3
"""Separate, perturbative GPU-memory diagnostics; never a speedup denominator.

Run a Python case under an external per-process nvidia-smi sampler. Optional
Python-boundary hooks record JK initialization, SCF stages, form_G and grid
switches without changing the pinned Psi4 binary. C++ internal calls are not
intercepted. Sampled peaks are lower bounds, not allocator high-water marks.
"""
import argparse
from contextlib import contextmanager
import functools
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys
import threading
import time


def atomic(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def host_status(pid):
    """Linux KiB counters; VmHWM covers process lifetime, not a sampled phase."""
    try:
        fields = {}
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            name, _, value = line.partition(":")
            if name in ("VmRSS", "VmHWM"):
                fields[name + "_kib"] = int(value.split()[0])
        return fields
    except (OSError, ValueError):
        return {}


def process_tree(pid):
    """All currently visible descendants, not only the first generation."""
    found, todo = {pid}, [pid]
    while todo:
        current = todo.pop()
        try:
            tasks = list(Path(f"/proc/{current}/task").iterdir())
        except OSError:
            continue
        for task in tasks:
            try:
                children = (task / "children").read_text().split()
            except OSError:
                continue
            for child in map(int, children):
                if child not in found:
                    found.add(child)
                    todo.append(child)
    return found


def parse_compute_apps(text, pids):
    """No device-wide fallback; unrelated applications cannot inflate results."""
    used, refused = {}, []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 3:
            raise ValueError(f"Unexpected compute-app row: {line[:150]}")
        pid, value, uuid = fields
        if not pid.isdigit():
            raise ValueError(f"Invalid compute-app PID: {pid}")
        if int(pid) not in pids:
            continue
        try:
            amount = float(value)
            if not math.isfinite(amount) or amount < 0:
                raise ValueError
        except ValueError:
            refused.append(dict(pid=int(pid), uuid=uuid, value=value))
            continue
        used[uuid] = used.get(uuid, 0.0) + amount
    return used, refused


def summarize_samples(samples):
    groups = {}
    for row in samples:
        # Refused/failed samples must not masquerade as zero-memory observations.
        if row.get("error") or row.get("unavailable"):
            continue
        boundary = row.get("phase", {})
        phase = "/".join(boundary.get("stack") or [boundary.get("phase", "unlabelled")])
        if "initial_grid" in boundary:
            phase += ":initial" if boundary["initial_grid"] else ":final"
        if "nbf" in boundary:
            phase += f":nbf={boundary['nbf']}"
        group = groups.setdefault(phase, dict(samples=0, sampled_peak_mib=0.0,
                                             gpu_observed=False))
        group["samples"] += 1
        group["sampled_peak_mib"] = max(group["sampled_peak_mib"], row["total_mib"])
        group["gpu_observed"] |= bool(row["by_uuid_mib"])
    good = [row for row in samples if not row.get("error") and not row.get("unavailable")]
    return dict(
        sampled_peak_mib=max((row["total_mib"] for row in good), default=None),
        gpu_observed=any(row["by_uuid_mib"] for row in good),
        samples=len(samples), successful_samples=len(good),
        failed_samples=sum(bool(row.get("error")) for row in samples),
        unavailable_samples=sum(bool(row.get("unavailable")) for row in samples),
        phases=groups,
        caveat="Sampled per-process-tree NVML residency, not an exact allocation peak. "
               "Spikes shorter than the actual query interval can be missed. "
               "Python hooks do not separate internal C++ J/K from XC inside form_G.",
    )


class Boundaries:
    """Write scalars only: retaining a wavefunction would falsify its lifetime."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.stack = []
        self.sequence = 0
        self.originals = []

    @staticmethod
    def describe(wfn):
        data = {}
        if wfn is None:
            return data
        is_potential = hasattr(wfn, "initial_grid") and hasattr(wfn, "grid")
        if not is_potential and not hasattr(wfn, "basisset"):
            return data
        try:
            data["nbf"] = (wfn.basis() if is_potential else wfn.basisset()).nbf()
            if not is_potential:
                aux = wfn.get_basisset("DF_BASIS_SCF")
                data["naux"] = aux.nbf() if aux is not None else None
        except Exception as exc:
            data["basis_metadata_error"] = str(exc)
        try:
            potential = wfn if is_potential else wfn.V_potential()
            if potential is not None:
                grid = potential.grid()
                blocks = potential.nblocks()
                # GPU grid construction deliberately leaves the host block
                # arrays/counters empty. A host npoints()==0 is not an empty
                # integration grid and must not be reported as its real size.
                host_grid = blocks > 0
                data.update(initial_grid=potential.initial_grid(),
                            grid_points=grid.npoints() if host_grid else None,
                            grid_points_source="host-grid" if host_grid else "gpu-count-not-exposed",
                            host_grid_blocks=blocks,
                            max_host_block_points=grid.max_points() if host_grid else None,
                            max_host_block_functions=grid.max_functions() if host_grid else None,
                            host_collocation_size=grid.collocation_size() if host_grid else None)
        except Exception as exc:
            data["grid_metadata_error"] = str(exc)
        return data

    def emit(self, event, phase, obj=None, **extra):
        self.sequence += 1
        record = dict(sequence=self.sequence, monotonic_s=time.monotonic(),
                      event=event, phase=phase, stack=list(self.stack),
                      host=host_status(os.getpid()), **self.describe(obj), **extra)
        with (self.directory / "boundaries.jsonl").open("a") as out:
            out.write(json.dumps(record, allow_nan=False) + "\n")
            out.flush()
        atomic(self.directory / "phase.json", record)

    def wrap(self, owner, name, phase):
        inherited = name not in vars(owner)
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapped(obj, *args, **kwargs):
            label = f"{phase}:{kwargs.get('banner', '')}".rstrip(":")
            # The outer SCF helper's first argument is a method name, not a wfn.
            self.stack.append(label)
            try:
                self.emit("enter", label, obj)
                result = original(obj, *args, **kwargs)
                self.emit("leave", label, result if phase == "scf_helper" else obj)
                return result
            except BaseException as exc:
                self.emit("error", label, obj, error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                self.stack.pop()
                self.emit("resume", self.stack[-1] if self.stack else "between_scf")

        setattr(owner, name, wrapped)
        self.originals.append((owner, name, original, inherited))

    @contextmanager
    def installed(self, core, proc, sapt_proc):
        try:
            for owner, name, phase in (
                (core.HF, "initialize", "scf_initialize"),
                (core.HF, "initialize_jk", "scf_initialize_jk"),
                (core.JK, "initialize", "jk_initialize"),
                (core.HF, "finalize_energy", "scf_finalize"),
                (core.RHF, "form_G", "rhf_form_G"),
                (core.UHF, "form_G", "uhf_form_G"),
                (core.VBase, "set_grid", "xc_set_grid"),
                (proc, "scf_helper", "scf_helper"),
                (sapt_proc, "scf_helper", "scf_helper"),
            ):
                self.wrap(owner, name, phase)
            yield
        finally:
            for owner, name, original, inherited in reversed(self.originals):
                if inherited:
                    delattr(owner, name)
                else:
                    setattr(owner, name, original)
            self.originals.clear()


def instrument(args):
    directory = Path(os.environ["GRID_MEMORY_OUTPUT"])
    import psi4
    from psi4.driver.procrouting import proc
    from psi4.driver.procrouting.sapt import sapt_proc
    hooks = Boundaries(directory)
    hooks.emit("ready", "imports_complete", package=psi4.__file__)
    sys.argv = [args.script, *args.arguments]
    sys.path.insert(0, str(Path(args.script).resolve().parent))
    try:
        with hooks.installed(psi4.core, proc, sapt_proc):
            runpy.run_path(args.script, run_name="__main__")
    finally:
        hooks.emit("exit", "process_exit")


def run(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    script = Path(args.script).resolve()
    command = [sys.executable, str(Path(__file__).resolve()), "instrument",
               "--script", str(script), "--", *args.arguments]
    env = dict(os.environ, GRID_MEMORY_OUTPUT=str(out))
    start = time.monotonic()
    manifest = dict(command=command, job_id=os.getenv("SLURM_JOB_ID"),
                    script=str(script), interval_s=args.interval,
                    diagnostic_only=True, eligible_for_speedup_bars=False,
                    microprofiling="determined by child; keep off for memory runs",
                    boundary_hooks="Python-visible calls only; C++ internal calls are not intercepted")
    atomic(out / "manifest.json", manifest)
    smi = shutil.which("nvidia-smi")
    if smi is None:
        atomic(out / "summary.json", dict(ok=False, error="nvidia-smi unavailable", **manifest))
        return 1
    device = subprocess.run([smi, "--query-gpu=uuid,name,memory.total,driver_version",
                             "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=10)
    (out / "devices.txt").write_text(device.stdout + device.stderr)
    if device.returncode:
        atomic(out / "summary.json", dict(ok=False, error="GPU inventory failed", **manifest))
        return 1
    stop = threading.Event()
    samples = []
    with (out / "console.log").open("w") as console:
        child = subprocess.Popen(command, env=env, stdout=console, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        interrupted = {}
        def terminate(signum, frame):
            interrupted["signal"] = signum
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        old_handlers = {s: signal.signal(s, terminate) for s in (signal.SIGTERM, signal.SIGINT)}
        manifest["pid"] = child.pid
        atomic(out / "manifest.json", manifest)

        def sample():
            with (out / "samples.jsonl").open("w") as stream:
                while not stop.is_set():
                    stamp = time.monotonic()
                    record = dict(monotonic_s=stamp, elapsed_s=stamp-start,
                                  host=host_status(child.pid), phase={})
                    try:
                        record["phase"] = json.loads((out / "phase.json").read_text())
                    except FileNotFoundError:
                        pass
                    try:
                        query = subprocess.run(
                            [smi, "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
                             "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=5)
                        if query.returncode:
                            raise RuntimeError(query.stderr.strip()[:300])
                        used, refused = parse_compute_apps(query.stdout, process_tree(child.pid))
                        record.update(by_uuid_mib=used, total_mib=sum(used.values()),
                                      unavailable=refused)
                    except Exception as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"
                    record["query_duration_s"] = time.monotonic() - stamp
                    samples.append(record)
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    stop.wait(max(0.0, args.interval - record["query_duration_s"]))

        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        timed_out = False
        try:
            rc = child.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                rc = child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                rc = child.wait()
        finally:
            stop.set()
            thread.join(timeout=10)
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
    report = summarize_samples(samples)
    stamps = [r["monotonic_s"] for r in samples]
    report["max_observed_interval_s"] = max(
        (b-a for a, b in zip(stamps, stamps[1:])), default=None)
    report.update(ok=rc == 0 and not interrupted and report["gpu_observed"] and bool(report["successful_samples"])
                  and not report["unavailable_samples"] and not report["failed_samples"],
                  returncode=rc, timed_out=timed_out, interrupted=interrupted, **manifest)
    atomic(out / "summary.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    for name in ("run", "instrument"):
        sub = subs.add_parser(name)
        sub.add_argument("--script", required=True)
        sub.add_argument("arguments", nargs=argparse.REMAINDER)
        if name == "run":
            sub.add_argument("--output", required=True)
            sub.add_argument("--interval", type=float, default=0.2)
            sub.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.arguments and args.arguments[0] == "--":
        args.arguments.pop(0)
    if args.mode == "run":
        if not (math.isfinite(args.interval) and args.interval > 0 and
                math.isfinite(args.timeout) and args.timeout > 0):
            parser.error("interval and timeout must be finite and positive")
        return run(args)
    return instrument(args)


if __name__ == "__main__":
    sys.exit(main())
