"""Opt-in Linux process-owned byte proofs backed by revocable kernel read leases.

No CUDA imports, ELF loading, or import-time file-lease acquisition. This is
not an mtime hash cache. All files are hashed through leased descriptors once;
each subsequent proof requires the same live kernel lease, descriptor, pathname,
process, thread and signal owner. A break permanently revokes the whole proof.

An admitted owner is retained until PROCESS EXIT, not native-plan/session close:
upstream C++ owners and loaded ELF objects can outlive Python CUDA sessions.
There is deliberately no public close, renew, retry, or transfer operation.
Unadapted native entrypoints must use guarded_call(); adapted sessions also
check before/after synchronization, before publishing results. A forced lease
break can invalidate in-flight work: published-proof revocation immediately
terminates the explicitly dedicated worker, without Python/native finalizers.
"""
from dataclasses import dataclass
import ctypes as ct
import errno
import hashlib
import os
from pathlib import Path
import re
import signal
import stat
import sys
import threading

try:
    import fcntl
except ImportError:  # Keep ordinary non-Linux imports harmless.
    fcntl = None


_PROCESS_OWNER = None
_LEGACY_SESSION_STARTED = False
_ADMISSION_IN_PROGRESS = False
_STATE_LOCK = threading.Lock()
REVOKED_EXIT_CODE = 86
KERNEL_NOTIFICATION_EXIT_CODE = int(signal.SIGIO) if hasattr(signal,"SIGIO") else None


def _after_fork():
    # A lock held by a vanished parent thread must not hide the PID guard.
    # Retain the proof/FD state; never unlock inherited open file descriptions.
    global _STATE_LOCK
    _STATE_LOCK = threading.Lock()


if hasattr(os,"register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _terminate_worker(reason):
    # Revocation cannot safely unwind arbitrary native-owning Python results,
    # tracebacks, context managers or C++ finalizers. Exit only this opt-in worker.
    # Even diagnostic writes can block forever on a full stderr pipe. The exit
    # code is the diagnostic; do not touch an arbitrary inherited logging sink.
    os._exit(REVOKED_EXIT_CODE)


class _KernelSignal:
    """Linux x86-64 libc signal disposition, not Python's cached handler view.

    SIGIO calls libc _exit(int) DIRECTLY. Its signature matches a one-argument
    signal handler, and _exit is async-signal-safe. Python callbacks are too late:
    an exception's finally block can run before Python dispatches the callback.
    No Python/GIL/C++ finalizer is entered by this native signal disposition.
    """
    def __init__(self):
        if os.uname().machine != "x86_64" or ct.sizeof(ct.c_void_p) != 8:
            raise RuntimeError("Owned runtime signal adapter currently requires Linux x86-64")
        self.libc=ct.CDLL(None,use_errno=True)
        self.action=self.libc.sigaction
        self.action.argtypes=[ct.c_int,ct.c_void_p,ct.c_void_p]
        self.action.restype=ct.c_int
        self.change=self.libc.signal
        self.change.argtypes=[ct.c_int,ct.c_void_p]
        self.change.restype=ct.c_void_p
        self.exit=self.libc._exit
        self.exit.argtypes=[ct.c_int]
        self.exit.restype=None
        self.exit_address=ct.cast(self.exit,ct.c_void_p).value
        # On this explicit libc ABI the handler union is the first pointer.
        # Oversized storage avoids relying on the remaining mask/flag layout.
        self.storage=ct.create_string_buffer(1024)
        self.handler_pointer=ct.cast(self.storage,ct.POINTER(ct.c_void_p))

    def current(self):
        if self.action(signal.SIGIO,None,ct.byref(self.storage)) != 0:
            raise OSError(ct.get_errno(),"Cannot query actual kernel SIGIO disposition")
        return self.handler_pointer[0] or 0

    def set(self, address):
        previous=self.change(signal.SIGIO,ct.c_void_p(address))
        if previous == ct.c_void_p(-1).value:
            raise OSError(ct.get_errno(),"Cannot set actual kernel SIGIO disposition")
        return previous or 0


def _fingerprint(value):
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,
            value.st_ctime_ns,value.st_mode,value.st_uid,value.st_nlink)


def _normalized_pins(pins):
    if not isinstance(pins,dict) or not pins:
        raise ValueError("Explicit nonempty library path/SHA256 pins required")
    result = {}
    for name,checksum in pins.items():
        path = Path(name)
        if (not path.is_absolute() or not isinstance(checksum,str)
                or re.fullmatch(r"[0-9a-f]{64}",checksum) is None):
            raise ValueError("Absolute library paths and lowercase SHA256 pins required")
        canonical = str(path.resolve(strict=True))
        if canonical in result:
            raise ValueError("Duplicate canonical immutable-library path")
        result[canonical] = checksum
    return tuple(sorted(result.items()))


def _hash_descriptor(fd):
    value = hashlib.sha256()
    os.lseek(fd,0,os.SEEK_SET)
    for block in iter(lambda:os.read(fd,1024**2),b""):
        value.update(block)
    return value.hexdigest()


@dataclass(frozen=True)
class _PinnedFile:
    fd: int
    path: str
    sha256: str
    fingerprint: tuple


class _OwnedRuntime:
    """Created only by admit_owned_runtime(); never released after publication."""
    def _initialize(self, pins):
        self.pid = os.getpid()
        self.thread = threading.get_ident()
        self.native_thread = threading.get_native_id()
        self.snapshot = pins
        self._files = {}
        self._fds = []
        self.failed = False
        self.reason = None
        self.admitted = False
        self.published = False
        self.full_hash_calls = self.full_hash_bytes = 0
        self.live_checks = self.mapped_checks = 0
        self.previous_handler = None
        self.signal_installed = False
        self._rollback_complete = False
        self._signal_probe = None
        self.previous_kernel_handler = None
        if (not sys.platform.startswith("linux") or fcntl is None
                or not all(hasattr(fcntl,name) for name in (
                    "F_SETLEASE","F_GETLEASE","F_SETOWN","F_GETOWN","F_SETSIG","F_GETSIG"))
                or not hasattr(signal,"SIGIO") or not hasattr(os,"RTLD_NOLOAD")):
            raise RuntimeError("Owned immutable runtime requires Linux kernel file leases")
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Owned immutable runtime must be admitted on the main thread")
        self.previous_handler = signal.getsignal(signal.SIGIO)
        if self.previous_handler != signal.SIG_DFL:
            raise RuntimeError("Owned immutable runtime requires unclaimed process SIGIO")
        if signal.SIGIO in signal.pthread_sigmask(signal.SIG_BLOCK,set()):
            raise RuntimeError("Owned immutable runtime requires unblocked process SIGIO")
        self._signal_probe=_KernelSignal()
        if self._signal_probe.current() != 0:
            raise RuntimeError("Owned immutable runtime requires unclaimed actual kernel SIGIO")
        self.previous_kernel_handler=self._signal_probe.set(self._signal_probe.exit_address)
        self.signal_installed = True
        if self.previous_kernel_handler != 0:
            raise RuntimeError("Kernel SIGIO ownership changed during admission")
        for path,checksum in pins:
            fd = os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
            # Register immediately, including before validation/lease failures.
            self._fds.append(fd)
            before = os.fstat(fd)
            self._files[path] = _PinnedFile(fd,path,checksum,_fingerprint(before))
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                    or before.st_nlink < 1 or before.st_size < 1):
                raise RuntimeError("Immutable libraries must be nonempty current-user-owned regular files")
            if _fingerprint(Path(path).stat()) != _fingerprint(before):
                raise RuntimeError("Immutable library pathname changed during admission")
            fcntl.fcntl(fd,fcntl.F_SETOWN,self.pid)
            fcntl.fcntl(fd,fcntl.F_SETSIG,0)  # Explicit default SIGIO notification.
            try:
                fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_RDLCK)
            except OSError as error:
                raise RuntimeError("Cannot establish immutable read lease: "+path) from error
        # Acquire EVERY lease before hashing. Existing writers/shared writable
        # mappings reject acquisition; later write opens/truncations revoke it.
        for record in self._files.values():
            self._check_file(record)
            observed = _hash_descriptor(record.fd)
            self.full_hash_calls += 1
            self.full_hash_bytes += record.fingerprint[2]
            self._check_file(record)
            if observed != record.sha256:
                raise RuntimeError("Immutable library bytes differ from explicit pin: "+record.path)
        self.admitted = True
        self.check_live()

    def _fail(self, reason):
        self.invalidate(reason)
        raise RuntimeError("Owned immutable runtime invalidated; new process required: "+self.reason)

    def invalidate(self, reason):
        """Monotonic revocation; a published proof cannot safely unwind/recover."""
        self.failed = True
        if self.reason is None:
            self.reason = str(reason)
        if self.published:
            _terminate_worker(self.reason)

    def _check_caller(self):
        # Never unlock/close an inherited open-file description in a fork.
        if os.getpid() != self.pid:
            self._fail("Foreign process/fork cannot use immutable byte proof")
        if (threading.get_ident() != self.thread
                or threading.get_native_id() != self.native_thread):
            self._fail("Foreign thread cannot use immutable byte proof")
        if self.failed:
            self._fail(self.reason or "Previously invalidated")
        if (signal.getsignal(signal.SIGIO) != self.previous_handler
                or self._signal_probe.current() != self._signal_probe.exit_address):
            self._fail("Process SIGIO ownership changed")
        if signal.SIGIO in signal.pthread_sigmask(signal.SIG_BLOCK,set()):
            self._fail("Process SIGIO was blocked")
        if self.failed:
            self._fail(self.reason or "Previously invalidated")

    def _check_file(self, record):
        self._check_caller()
        try:
            # While a writer is still BLOCKED, F_GETLEASE already returns the
            # requested target F_UNLCK. Never wait for bytes to change or renew.
            if fcntl.fcntl(record.fd,fcntl.F_GETLEASE) != fcntl.F_RDLCK:
                self._fail("Kernel read lease lost or breaking: "+record.path)
            if (fcntl.fcntl(record.fd,fcntl.F_GETOWN) != self.pid
                    or fcntl.fcntl(record.fd,fcntl.F_GETSIG) != 0):
                self._fail("Kernel lease notification ownership changed: "+record.path)
            descriptor = os.fstat(record.fd)
            current = Path(record.path).stat()
            if (_fingerprint(descriptor) != record.fingerprint
                    or _fingerprint(current) != record.fingerprint):
                self._fail("Immutable library descriptor/path changed: "+record.path)
        except OSError as error:
            self._fail("Immutable library descriptor/path unavailable: "+record.path+": "+str(error))
        self._check_caller()

    def check_live(self):
        self.live_checks += 1
        self._check_caller()
        if not self.admitted:
            self._fail("Immutable byte admission is incomplete")
        for record in self._files.values():
            self._check_file(record)
        self._check_caller()

    def mapped_sha256(self, path, mappings):
        """Byte receipt valid only for the currently pinned mapped inode/device."""
        self.mapped_checks += 1
        self._check_caller()
        if not self.admitted:
            self._fail("Immutable byte admission is incomplete")
        record = self._files.get(str(path))
        if record is None:
            self._fail("Mapped library is outside the owned immutable set: "+str(path))
        self._check_file(record)
        observed = {(inode,device) for _,_,name,inode,device in mappings if name == record.path}
        expected = {(record.fingerprint[1],
                     (os.major(record.fingerprint[0]),os.minor(record.fingerprint[0])))}
        if observed != expected:
            self._fail("Mapped immutable inode/device differs from leased descriptor: "+record.path)
        self._check_caller()
        return record.sha256

    def require_library(self, path, checksum):
        """Bind a session's explicit library locator to this same owned set."""
        self.check_live()
        canonical = str(Path(path).resolve(strict=True))
        record = self._files.get(canonical)
        if record is None or record.sha256 != checksum:
            self._fail("Session library differs from owned immutable pins: "+canonical)
        self._check_file(record)
        self._check_caller()
        return record.sha256

    def load_library(self, path, checksum):
        """Load the admitted descriptor, never a separately reopened alias."""
        self.require_library(path,checksum)
        canonical = str(Path(path).resolve(strict=True))
        record = self._files.get(canonical)
        if record is None or record.sha256 != checksum:
            self._fail("Library locator changed before descriptor load")
        self._check_file(record)
        # /proc/self/fd changes $ORIGIN on a FIRST dlopen. Bootstrap must have
        # already loaded the full pinned closure through its canonical paths.
        # Never discover dependencies or execute first-load ELF constructors
        # through this descriptor adapter.
        from .sapt_cuest250_identity import mapped_rows
        rows = mapped_rows(Path("/proc/self/maps").read_text())
        for name in self._files:
            self.mapped_sha256(name,rows)
        try:
            library = ct.CDLL("/proc/self/fd/"+str(record.fd),
                              mode=os.RTLD_NOLOAD|os.RTLD_NOW)
        except OSError:
            self._fail("Pinned library is not already resident in the dynamic linker")
        self.check_live()
        return library, canonical

    def guarded_call(self, function, *args, **kwargs):
        """Synchronous publication gate for native work outside adapted sessions."""
        self.check_live()
        try:
            result = function(*args,**kwargs)
        except BaseException:
            # Ordinary errors retain their original exception. If identity has
            # been revoked, check_live terminates the worker instead of unwinding
            # any possibly native-owning result/exception/temporary objects.
            self.check_live()
            raise
        self.check_live()
        return result

    def report(self):
        # A healthy diagnostic receipt, never a replacement for execution checks.
        self.check_live()
        return {
            "contract":"process-owned-kernel-read-leases-v1",
            "pid":self.pid,"native_thread":self.native_thread,
            "lifetime":"process exit; no close/renew/retry/transfer",
            "revocation":"terminate dedicated worker without native unwinding",
            "revoked_exit_code":REVOKED_EXIT_CODE,
            "kernel_notification_exit_code":KERNEL_NOTIFICATION_EXIT_CODE,
            "admitted":self.admitted,"invalidated":self.failed,"reason":self.reason,
            "full_hash_calls":self.full_hash_calls,"full_hash_bytes":self.full_hash_bytes,
            "live_checks":self.live_checks,"mapped_checks":self.mapped_checks,
            "files":{path:{"sha256":record.sha256,"bytes":record.fingerprint[2]}
                     for path,record in self._files.items()},
        }

    def _discard_unpublished(self):
        """Admission rollback ONLY; never called on the published process owner."""
        errors = []
        unresolved = []
        for fd in self._fds:
            try:
                os.close(fd)  # No F_UNLCK on a possibly shared description.
            except BaseException as error:
                errors.append(error)
                try:
                    os.fstat(fd)
                except OSError as status:
                    if status.errno != errno.EBADF:  # Close demonstrably completed.
                        unresolved.append(fd)
                except BaseException:
                    unresolved.append(fd)
                else:
                    unresolved.append(fd)
        self._files.clear()
        self._fds.clear()
        if unresolved:
            # No safe way to retry an interrupted close or restore SIGIO while
            # an unpublished leased descriptor may still remain. Others have
            # already been attempted; process exit closes the remaining FDs.
            _terminate_worker("Immutable admission rollback left unresolved descriptors")
        if (self.signal_installed
                and self._signal_probe.current() == self._signal_probe.exit_address):
            # No leased descriptors remain, so drain our pending notifications
            # before restoring the prior signal disposition.
            previous_mask = None
            try:
                previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGIO})
                while signal.sigtimedwait({signal.SIGIO},0) is not None:
                    pass
            except BaseException as error:
                errors.append(error)
            finally:
                try:
                    self._signal_probe.set(self.previous_kernel_handler)
                except BaseException as error:
                    errors.append(error)
                if previous_mask is not None:
                    try:
                        signal.pthread_sigmask(signal.SIG_SETMASK,previous_mask)
                    except BaseException as error:
                        errors.append(error)
            if self._signal_probe.current() != self.previous_kernel_handler:
                _terminate_worker("Immutable admission rollback could not restore SIGIO")
        self._rollback_complete = True
        if errors:
            raise RuntimeError("Unpublished immutable-runtime rollback failed: "+str(errors))


def admit_owned_runtime(pins, *, dedicated_worker=False):
    """Explicitly opt this process into one immutable library set.

    dedicated_worker=True explicitly acknowledges worker termination on proof
    revocation (exit86 for live-check failure, exit SIGIO/29 for the native
    signal handler; no native finalizers). The caller reserves unblocked
    SIGIO and the admitting main thread for this lifetime.
    All unadapted native operations must use guarded_call(). Unsupported leases,
    existing writers, drift or revocation fail closed; no implicit fresh-hash
    fallback or proof renewal is performed.
    """
    global _PROCESS_OWNER, _ADMISSION_IN_PROGRESS
    if dedicated_worker is not True:
        raise ValueError("Explicit dedicated_worker=True is required; revocation terminates this worker")
    snapshot = _normalized_pins(pins)
    with _STATE_LOCK:
        if _PROCESS_OWNER is not None:
            _PROCESS_OWNER.check_live()
            if snapshot != _PROCESS_OWNER.snapshot:
                raise RuntimeError("Process already owns a different immutable runtime; new process required")
            return _PROCESS_OWNER
        if _ADMISSION_IN_PROGRESS:
            raise RuntimeError("Immutable runtime admission is already in progress")
        if _LEGACY_SESSION_STARTED:
            raise RuntimeError("Cannot upgrade a process after a legacy CUDA session; new process required")
        _ADMISSION_IN_PROGRESS = True
    owner = _OwnedRuntime()
    try:
        try:
            owner._initialize(snapshot)
        except BaseException as primary:
            try:
                owner._discard_unpublished()
            except BaseException as error:
                if not getattr(owner,"_rollback_complete",False):
                    _terminate_worker("Immutable admission rollback was interrupted/incomplete")
                primary.add_note("Immutable admission rollback also failed: "+str(error))
            raise
        # Raw descriptors remain process-owned through interpreter teardown.
        with _STATE_LOCK:
            owner.published = True
            _PROCESS_OWNER = owner
    finally:
        with _STATE_LOCK:
            _ADMISSION_IN_PROGRESS = False
    owner.check_live()
    return owner


def current_owned_runtime():
    """Return the explicit process owner, or None for legacy fresh-byte audits."""
    return _PROCESS_OWNER


def owner_for_session():
    """Select the adapter before a session's first native CUDA operation."""
    global _LEGACY_SESSION_STARTED
    with _STATE_LOCK:
        if _ADMISSION_IN_PROGRESS:
            raise RuntimeError("CUDA session selection is forbidden during immutable admission")
        owner = _PROCESS_OWNER
        if owner is None:
            _LEGACY_SESSION_STARTED = True
    if owner is not None:
        owner.check_live()
    return owner
