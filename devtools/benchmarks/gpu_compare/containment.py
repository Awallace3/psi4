"""Bounded Linux subreaper containment, including detached worker sessions."""
import ctypes
import os
from pathlib import Path
import signal
import subprocess
import time

from diagnose_pyscf import deferred_launch_interrupts


def subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Cannot enable child subreaper")
    enabled = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(enabled), 0, 0, 0) != 0 or enabled.value != 1:
        raise RuntimeError("Child subreaper not verified")


def children(pid):
    found = set()
    for path in (Path("/proc") / str(pid) / "task").glob("*/children"):
        try:
            found.update(int(x) for x in path.read_text().split())
        except FileNotFoundError:
            pass
    return found


def tree():
    pending = list(children(os.getpid()))
    found = set()
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.add(pid)
        pending.extend(children(pid))
        if len(found) > 10000:
            raise RuntimeError("Containment process limit exceeded")
    return found


def birth(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        return text.rsplit(")", 1)[1].split()[19]  # field22, Linux starttime
    except FileNotFoundError:
        return None


def run_contained(command, seconds, cleanup_report, grace=20):
    subreaper()
    started = time.monotonic()
    tracked = {}

    def collect():
        for pid in tree():
            token = birth(pid)
            if token is not None:
                tracked[pid] = token

    with deferred_launch_interrupts() as (restore, pending, previous):
        child = subprocess.Popen(command, start_new_session=True)
        try:
            restore()
            for signum in pending:
                handler = previous[signum]
                if callable(handler):
                    handler(signum, None)
                elif handler != signal.SIG_IGN:
                    raise KeyboardInterrupt("Interrupted during stage launch")
            while True:
                collect()
                rc = child.poll()
                if rc is not None:
                    return rc
                if time.monotonic() - started >= seconds:
                    return 124
                time.sleep(.05)
        finally:
            saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
            for sig in saved:
                signal.signal(sig, signal.SIG_IGN)
            deadline = time.monotonic() + grace
            clean = False
            try:
                while True:
                    collect()
                    child.poll()
                    # Reap adopted descendants, not Popen's directly-owned child.
                    for pid in children(os.getpid()) - {child.pid}:
                        try:
                            os.waitpid(pid, os.WNOHANG)
                        except ChildProcessError:
                            pass
                    live = {pid for pid, token in tracked.items() if birth(pid) == token}
                    if child.poll() is not None and not live and not tree():
                        clean = True
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    sig = signal.SIGTERM if remaining > min(2., grace/2) else signal.SIGKILL
                    for pid in sorted(live - {child.pid}) + [child.pid]:
                        if pid in tracked and birth(pid) == tracked[pid]:
                            try:
                                os.kill(pid, sig)
                            except ProcessLookupError:
                                pass
                    time.sleep(min(.05, remaining))
            finally:
                cleanup_report.update(
                    cleanup_verified=clean, leader_reaped=child.poll() is not None,
                    leader_returncode=child.returncode, subreaper_enabled=True,
                    descendants_absent=clean, observed_pids=sorted(tracked),
                    owner_containment_required=not clean,
                    cleanup_scope="Subreaper adoption/reaping plus detached descendant absence")
                for sig, handler in saved.items():
                    signal.signal(sig, handler)
