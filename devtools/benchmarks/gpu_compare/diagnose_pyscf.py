"""Opt-in, bounded diagnostic of the existing profiled PySCF worker.

No GPU imports at module load. Run only in an explicitly assigned GPU window,
through the allocation owner's helper. This is not a benchmark or an SCF fix.
"""
import argparse
import contextlib
import faulthandler
import functools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import compare


class PhaseTrace:
    def __init__(self, path):
        self.path = path
        self.stream = None
        self.sequence = 0
        self.depth = 0

    def __enter__(self):
        self.stream = self.path.open("x", buffering=1)
        return self

    def __exit__(self, *_):
        self.stream.close()

    @contextlib.contextmanager
    def phase(self, name):
        self.sequence += 1
        sequence, depth = self.sequence, self.depth
        self.depth += 1
        def emit(event, **extra):
            self.stream.write(json.dumps(dict(
                event=event, phase=name, sequence=sequence, depth=depth,
                monotonic_s=time.monotonic(), pid=os.getpid(), **extra)) + "\n")
            self.stream.flush()
        emit("enter")
        try:
            yield
        except BaseException as exc:
            emit("error", exception=type(exc).__name__)
            raise
        else:
            emit("exit")
        finally:
            self.depth -= 1

    @contextlib.contextmanager
    def methods(self, mf):
        # Original direct dispatch MUST be validated before installing these
        # wrappers. The caller enters pyscf_timing_policy first.
        targets = [(mf, name, "mf." + name) for name in (
            "get_init_guess", "get_hcore", "get_ovlp", "eig", "get_veff",
            "get_j", "get_k", "get_jk")]
        targets += [(mf.grids, "build", "grids.build"),
                    (mf._numint, "nr_rks", "numint.nr_rks")]
        with contextlib.ExitStack() as stack:
            for obj, name, label in targets:
                original = getattr(obj, name)
                def traced(*args, _original=original, _label=label, **kwargs):
                    with self.phase(_label):
                        return _original(*args, **kwargs)
                stack.enter_context(patch.object(
                    obj, name, functools.wraps(original)(traced)))
            yield


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return None  # Lack of permission/proof is not evidence of absence.
    return True


def cleanup_group(child, grace, report):
    """Bounded group cleanup, including descendants of an exited leader."""
    deadline = time.monotonic() + grace
    term_deadline = deadline - min(1., grace / 2)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        # A second scheduler TERM must not interrupt cleanup before escalation.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        if child.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(child.pid, signal.SIGUSR1)
        def send(sig):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(child.pid, sig)
        def absent_before(bound):
            while True:
                reaped = child.poll() is not None  # WNOHANG; never bare wait().
                absent = group_exists(child.pid) is False
                if reaped and absent:
                    return True
                remaining = bound - time.monotonic()
                if remaining <= 0:
                    return False
                time.sleep(min(.02, remaining))
        # Check the group even if the direct child has already exited normally.
        if group_exists(child.pid) is not False:
            send(signal.SIGTERM)
        clean = absent_before(term_deadline)
        if not clean:
            send(signal.SIGKILL)
            clean = absent_before(deadline)
        report.update(
            cleanup_verified=clean, leader_reaped=child.poll() is not None,
            leader_returncode=child.returncode, pgid=child.pid,
            cleanup_scope="Direct child reaped AND process group absent",
            owner_containment_required=not clean)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@contextlib.contextmanager
def deferred_launch_interrupts():
    """Defer exceptions until a successfully created child has cleanup coverage."""
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    pending = []
    def defer(signum, frame):
        pending.append(signum)
    def restore():
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    try:
        for sig in previous:
            signal.signal(sig, defer)
        yield restore, pending, previous
    finally:
        restore()


def bounded_process(command, seconds, grace=5, cleanup_report=None):
    """One child group, no retries; terminate/reap it on timeout or interruption."""
    report = {} if cleanup_report is None else cleanup_report
    rc = None
    with deferred_launch_interrupts() as (restore, pending, previous):
        child = subprocess.Popen(command, start_new_session=True)
        try:
            # Restore/deliver only INSIDE the region protecting the live group.
            # Child exec does not inherit Python handlers or a blocked sigmask.
            restore()
            for signum in pending:
                handler = previous[signum]
                if callable(handler):
                    handler(signum, None)
                elif handler != signal.SIG_IGN:
                    raise KeyboardInterrupt(f"Signal {signum} during child launch")
            rc = child.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            rc = 124
        finally:
            cleanup_group(child, grace, report)
    return rc if report["cleanup_verified"] else 125


def require_tracked_diagnostic(repo, path, commit):
    relative = path.resolve().relative_to(repo.resolve()).as_posix()
    try:
        committed = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{commit}:{relative}"],
            stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Diagnostic must be tracked at the pinned HEAD") from exc
    import hashlib
    if hashlib.sha256(committed).hexdigest() != compare.digest(path):
        raise RuntimeError("Diagnostic bytes differ from the pinned HEAD")


def child_worker(args):
    """Exercise compare.worker unchanged; add trace hooks after route checks."""
    with (args.output / "python-stacks.log").open("x") as stacks, \
            PhaseTrace(args.output / "phases.jsonl") as trace:
        faulthandler.enable(file=stacks, all_threads=True)
        faulthandler.register(signal.SIGUSR1, file=stacks, all_threads=True)
        faulthandler.dump_traceback_later(
            args.stack_seconds, repeat=True, file=stacks)
        original_policy = compare.pyscf_timing_policy
        @contextlib.contextmanager
        def diagnosed_policy(policy, mf, timers, profile, direct):
            with original_policy(policy, mf, timers, profile, direct):
                with trace.methods(mf), trace.phase("scf.kernel"):
                    yield
        worker_args = SimpleNamespace(
            engine="pyscf", system=args.system, functional=args.functional,
            repeat=0, profile=True, export_grid=False, config=args.config,
            output=args.output / "worker", xyz=args.xyz,
            psi_package=args.psi_package, gpu_env=args.gpu_env)
        try:
            with patch.object(compare, "pyscf_timing_policy", diagnosed_policy), \
                    trace.phase("worker.imports_setup_and_scf"):
                return compare.worker(worker_args)
        finally:
            faulthandler.cancel_dump_traceback_later()
            faulthandler.unregister(signal.SIGUSR1)
            faulthandler.disable()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "xyz", "psi-package", "gpu-env", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--system", choices=compare.ATOMS, default="peptide")
    p.add_argument("--functional", default="pbe")
    p.add_argument("--seconds", type=float, default=180)
    p.add_argument("--stack-seconds", type=float, default=30)
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if not 0 < args.stack_seconds < args.seconds <= 300:
        p.error("Require 0 < stack-seconds < seconds <= 300; no implicit long run")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.child:
        return child_worker(args)
    repo = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise RuntimeError("Diagnostic harness must be committed and pristine")
    require_tracked_diagnostic(repo, Path(__file__).resolve(), commit)
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = [args.config, args.xyz, Path(__file__).resolve()]
    inputs += [compare.HERE / name for name in (
        "compare.py", "direct.py", "protocol.py", "grids.py", "memory.py")]
    hashes = {str(path): compare.digest(path) for path in inputs}
    compare.atomic_json(args.output / "diagnostic-inputs.json", dict(
        diagnostic_only=True, profile=True, headline_timing_eligible=False,
        harness_commit=commit, python=sys.executable,
        seconds=args.seconds, stack_seconds=args.stack_seconds,
        input_sha256=hashes, config=json.loads(args.config.read_text())))
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    rc, error, cleanup = None, None, {}
    try:
        rc = bounded_process(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child"],
            args.seconds, cleanup_report=cleanup)
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        unchanged = all(Path(path).exists() and compare.digest(Path(path)) == sha
                        for path, sha in hashes.items())
        compare.atomic_json(args.output / "diagnostic-terminal.json", dict(
            diagnostic_only=True, headline_timing_eligible=False, child_exit=rc,
            parent_error=error, input_hashes_unchanged=unchanged, cleanup=cleanup))
    return rc if unchanged else 1


if __name__ == "__main__":
    sys.exit(main())
