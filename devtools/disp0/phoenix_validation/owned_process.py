"""Bounded local child-tree ownership; no scheduler or foreign-process probes."""
import os
import ctypes
from contextlib import contextmanager
import signal
import subprocess
import time
from gpu_runtime import require_local_lease


def subreaper():
    # Orphans from this command become ours, allowing group-only reaping. No
    # enumeration or signalling of another agent's processes is involved.
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),"Cannot own/reap descendant cleanup")


def group_alive(pgid):
    while True:
        try:
            pid,_=os.waitpid(-pgid,os.WNOHANG)
            if pid==0:break
        except ChildProcessError:break
    try:os.killpg(pgid,0)
    except ProcessLookupError:return False
    return True


def unwind(child):
    """Stop/reap the entire *owned* group, even if its leader already exited."""
    if child.poll() is None:
        os.killpg(child.pid,signal.SIGINT)
    elif group_alive(child.pid):
        os.killpg(child.pid,signal.SIGINT)
    deadline=time.monotonic()+30
    while True:
        child.poll()
        if not group_alive(child.pid):break
        if time.monotonic()>=deadline:
            os.killpg(child.pid,signal.SIGKILL)
            break
        time.sleep(.05)
    child.wait()
    deadline=time.monotonic()+5
    while group_alive(child.pid):
        if time.monotonic()>=deadline:
            raise RuntimeError("Owned descendant group cleanup could not be proved")
        time.sleep(.05)


@contextmanager
def termination_signals():
    selected=(signal.SIGTERM,signal.SIGHUP)
    old={sig:signal.getsignal(sig) for sig in selected}
    def abort(signum,frame):
        for sig in selected:signal.signal(sig,signal.SIG_IGN)
        raise InterruptedError("Owned runner received termination signal "+str(signum))
    try:
        for sig in selected:signal.signal(sig,abort)
        yield
    finally:
        for sig,handler in old.items():signal.signal(sig,handler)


def execute(command,env,out,err,seconds,fd,policy,*,cwd=None):
    with termination_signals():
        return _execute(command,env,out,err,seconds,fd,policy,cwd=cwd)


def _execute(command,env,out,err,seconds,fd,policy,*,cwd=None):
    if require_local_lease(policy)!=fd:
        raise RuntimeError("Mismatched inherited common lease descriptor")
    subreaper()
    child=subprocess.Popen(command,env=env,stdout=out,stderr=err,cwd=cwd,
                           start_new_session=True,pass_fds=(fd,))
    deadline=time.monotonic()+seconds
    try:
        while True:
            try:
                code=child.wait(timeout=min(1,max(.001,deadline-time.monotonic())))
                if group_alive(child.pid):
                    unwind(child)
                    raise RuntimeError("Owned command left surviving descendants")
                require_local_lease(policy)
                return code
            except subprocess.TimeoutExpired:
                require_local_lease(policy)
                if time.monotonic()>=deadline:
                    raise TimeoutError("Bounded owned scientific child expired")
    except BaseException:
        old=signal.signal(signal.SIGINT,signal.SIG_IGN)
        try:unwind(child)
        finally:signal.signal(signal.SIGINT,old)
        raise
