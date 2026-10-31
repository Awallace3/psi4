"""Explicit process-lifetime ELF snapshots, not a cache of mutable file proofs.

Stdlib-only support imported normally before psi4.__init__/native loading.
The scientific runtime adopts the exact prepared owner after package import;
there is no scientific-module injection or per-file installation overlay.
Each source is copied, SHA256 checked, irreversibly sealed, and checked again
before any ELF load. Loaded CDLLs and descriptors are retained until process
exit. Subsequent source-file changes cannot change these loaded images.

Unlike revocable file leases, sealed backing inodes cannot be overwritten or
truncated. Execution checks therefore validate caller/lifetime, not filesystem
metadata or SIGIO. Mapping/symbol admission remains fresh at loader/factory
seams. Only exact sealed inode/device/kernel-name matches may be translated
to their logical source locators; arbitrary deleted ELF mappings still fail.

No CUDA imports or native calls at import. Linux >=6.3, dedicated main-thread
bootstrap and a private diagnostic directory excluded from loader paths are
required. First loads and pinned dependencies are descriptor-bound; diagnostic
aliases never participate in ELF lookup.
"""
import ctypes as ct
import hashlib
import os
from pathlib import Path
import stat
import sys
import threading
import time

try:
    import fcntl
except ImportError:
    fcntl = None

CONTRACT = "process-owned-kernel-sealed-snapshots-v1"
F_ADD_SEALS, F_GET_SEALS = 1033, 1034
# WRITE|GROW|SHRINK|SEAL|EXEC. EXEC also implies FUTURE_WRITE on Linux.
REQUIRED_SEALS = 0x2f
MAX_SNAPSHOT_BYTES = 2 * 1024**3  # Canonical17-image closure is about1.76GB.
# These belong to the qualified readonly CPU/OS base, not the CUDA toolkit.
# Optional RDMA/other dependencies do not silently extend that trust base.
BASE_DEPENDENCIES = frozenset((
    "libc.so.6", "libm.so.6", "libdl.so.2", "libpthread.so.0", "librt.so.1",
    "libstdc++.so.6", "libgcc_s.so.1", "ld-linux-x86-64.so.2",
))
CUDA_PREFIXES = (
    "libcudart", "libcublas", "libcusolver", "libcusparse", "libnvJitLink",
    "libnvrtc", "libcufft", "libcurand", "libcutensor", "libcupti",
    "libnvfatbin", "libnvvm", "libnvptxcompiler", "libcudnn", "libcufile",
    "libnvjpeg", "libcuest",
)


def _memfd(name):
    flags = 0x01 | 0x02 | 0x10  # CLOEXEC|ALLOW_SEALING|EXEC.
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name, flags)
    # The supported conda Python omits the API despite libc/kernel support.
    libc = ct.CDLL(None, use_errno=True)
    create = libc.memfd_create
    create.argtypes, create.restype = [ct.c_char_p, ct.c_uint], ct.c_int
    fd = create(name.encode(), flags)
    if fd < 0:
        raise OSError(ct.get_errno(), "memfd_create")
    return fd


def _fingerprint(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode, value.st_uid, value.st_nlink)


def _identity(value):
    return value.st_ino, (os.major(value.st_dev), os.minor(value.st_dev))


def _descriptor_digest(fd):
    checksum = hashlib.sha256()
    offset = 0
    while block := os.pread(fd, 1024**2, offset):
        checksum.update(block)
        offset += len(block)
    return checksum.hexdigest()


def _mapped_basename(path):
    """POSIX kernel filenames, without constructing a Path for every segment.

    Preserve Path.name's normalization for the unusual empty/dot final fields;
    no filesystem lookup, resolved-path memo, or mapping receipt is involved.
    """
    tail = path.rsplit("/",1)[-1]
    return tail if tail and tail != "." else Path(path).name


class _SealedRuntime:
    """Private implementation of the existing process-owned runtime interface."""
    contract = CONTRACT

    def _initialize(self, pins, alias_directory):
        started = time.perf_counter()
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.native_thread = threading.get_native_id()
        self.snapshot, self._files, self._fds = pins, {}, []
        self._loaded = {}
        self._soname_handles = {}
        self._names = {}
        self._preload_complete = False
        self.failed, self.reason, self.published = False, None, False
        self.admitted = False
        self.full_hash_calls = self.full_hash_bytes = 0
        self._snapshot_bytes = 0
        self.live_checks = self.mapped_checks = 0
        self.prepare_seconds = self.preload_seconds = 0.0
        self._rollback_complete = False
        self._aliases_created = []
        if (not sys.platform.startswith("linux") or fcntl is None
                or os.uname().machine != "x86_64"):
            raise RuntimeError("Sealed snapshots require Linux x86-64 executable memfds")
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Sealed runtime must be admitted on the main thread")
        directory = Path(alias_directory)
        if (not directory.is_absolute() or directory.is_symlink()
                or directory.resolve(strict=True) != directory):
            raise ValueError("Absolute private startup alias directory required")
        self.alias_directory = str(directory)
        info = directory.stat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700 or any(directory.iterdir())):
            raise RuntimeError("Sealed startup alias directory must be empty, owned, mode0700")
        # This directory contains diagnostic links ONLY. It must never be a
        # mutable search namespace for first-load constructors/dependencies.
        if self.alias_directory in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
            raise RuntimeError("Diagnostic aliases must be excluded from the loader path")
        if os.environ.get("LD_PRELOAD"):
            raise RuntimeError("Sealed bootstrap forbids LD_PRELOAD")
        for line in Path("/proc/self/maps").read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6:
                name = fields[5]
                if (Path(name).name.startswith(CUDA_PREFIXES)
                        or name.endswith(" (deleted)") and ".so" in name):
                    raise RuntimeError("Sealed runtime must precede native CUDA/cuEST loading")
        names = [Path(path).name for path, _ in pins]
        if len(set(names)) != len(names):
            raise ValueError("Sealed source basenames must be unique")
        for path, checksum in pins:
            self._copy(path, checksum, directory)
        from ._elf import loader_metadata
        for record in self._files.values():
            record["elf"] = loader_metadata(record["fd"],record["bytes"])
            for name in {Path(record["path"]).name,record["elf"]["soname"]} - {None}:
                if name in self._names and self._names[name] is not record:
                    raise ValueError("Ambiguous pinned ELF basename/SONAME")
                self._names[name] = record
        self.admitted = True
        self.check_live()
        self.prepare_seconds = time.perf_counter() - started

    def _copy(self, path, expected, directory):
        fd = _memfd("psi4-sealed-" + Path(path).name)
        self._fds.append(fd)
        source = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            before = os.fstat(source)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                    or before.st_size <= 0 or before.st_nlink < 1):
                raise RuntimeError("Sealed sources must be nonempty owned regular files")
            if self._snapshot_bytes + before.st_size > MAX_SNAPSHOT_BYTES:
                raise MemoryError("Explicit sealed closure exceeds the2GiB snapshot budget")
            checksum = hashlib.sha256()
            copied = 0
            while block := os.read(source, 1024**2):
                copied += len(block)
                if copied > before.st_size:
                    raise RuntimeError("Sealed source changed size during bounded copying")
                checksum.update(block)
                pending = memoryview(block)
                while pending:
                    written = os.write(fd, pending)
                    if written <= 0:
                        raise OSError("Incomplete sealed snapshot write")
                    pending = pending[written:]
            if (_fingerprint(before) != _fingerprint(os.fstat(source))
                    or checksum.hexdigest() != expected):
                raise RuntimeError("Sealed source changed or differs from explicit SHA256")
        finally:
            os.close(source)
        os.fchmod(fd, 0o700)
        fcntl.fcntl(fd, F_ADD_SEALS, REQUIRED_SEALS)
        seals = fcntl.fcntl(fd, F_GET_SEALS)
        info = os.fstat(fd)
        if (seals & REQUIRED_SEALS != REQUIRED_SEALS
                or info.st_size != before.st_size or os.get_inheritable(fd)
                or _descriptor_digest(fd) != expected):
            raise RuntimeError("Executable snapshot seal/byte verification failed")
        self.full_hash_calls += 2
        self.full_hash_bytes += 2 * info.st_size
        self._snapshot_bytes += info.st_size
        kernel_path = os.readlink(f"/proc/self/fd/{fd}")
        record = {"fd": fd, "path": path, "sha256": expected, "bytes": info.st_size,
                  "identity": _identity(info), "kernel_path": kernel_path, "seals": seals}
        self._files[path] = record
        names = [Path(path).name]
        if names[0] == "libcuest.so.0.2.50":
            names.append("libcuest.so.0")  # Psi4 core's canonical DT_NEEDED.
        for name in names:
            alias = directory / name
            alias.symlink_to(f"/proc/{self.pid}/fd/{fd}")
            self._aliases_created.append(alias)

    def invalidate(self, reason):
        self.failed = True
        if self.reason is None:
            self.reason = str(reason)
        if self.published:
            os._exit(86)

    def _fail(self, reason):
        self.invalidate(reason)
        raise RuntimeError("Sealed runtime invalidated; new process required: " + self.reason)

    def check_live(self):
        """O(1) caller/lifetime gate; sealed loaded inodes need no file polling."""
        self.live_checks += 1
        if (self.failed or not self.admitted or os.getpid() != self.pid
                or threading.get_ident() != self.thread
                or threading.get_native_id() != self.native_thread):
            self._fail("Incomplete, invalidated, foreign-process/thread sealed runtime")

    def _record(self, path, checksum):
        self.check_live()
        # These are logical locators frozen at admission, not live aliases.
        record = self._files.get(os.fspath(path))
        if record is None or record["sha256"] != checksum:
            self._fail("Library differs from the explicitly sealed source set")
        self.check_live()
        return record

    def require_library(self, path, checksum):
        record = self._record(path, checksum)
        if record["path"] not in self._loaded:
            self._check_snapshot(record)
        self.check_live()
        return record["sha256"]

    def _check_snapshot(self, record):
        """Not a compute-path poll: used only before a retained load exists."""
        try:
            fd = record["fd"]
            if (_identity(os.fstat(fd)) != record["identity"]
                    or fcntl.fcntl(fd, F_GET_SEALS) != record["seals"]):
                self._fail("Snapshot descriptor changed before its first load")
        except OSError:
            self._fail("Snapshot descriptor unavailable before its first load")

    def preload(self, paths):
        """One descriptor-bound bootstrap, with explicit SONAME dependencies."""
        started = time.perf_counter()
        try:
            self._preload(paths)
        except BaseException:
            self._fail("Sealed bootstrap preload failed; no retry")
        self.preload_seconds += time.perf_counter() - started

    def _preload(self, paths):
        self.check_live()
        if self._preload_complete:
            self._fail("Sealed bootstrap loading is already complete")
        selected = set()
        for path in paths:
            record = self._files.get(os.fspath(path))
            if record is None:
                self._fail("Preload is outside the sealed source set")
            if record["path"] in selected:
                self._fail("Duplicate sealed preload")
            selected.add(record["path"])
            self._load(record,set())
        self._preload_complete = True
        self.check_live()

    def _load(self, record, visiting):
        path = record["path"]
        if path in self._loaded:
            return
        if path in visiting:
            self._fail("Cyclic pinned ELF dependencies are unsupported")
        visiting.add(path)
        for name in record["elf"]["needed"]:
            dependency = self._names.get(name)
            if dependency is not None:
                # An FD-loaded dependency must register the exact DT_NEEDED
                # SONAME so glibc reuses it instead of searching a pathname.
                if dependency["elf"]["soname"] != name:
                    self._fail("Pinned dependency lacks the required exact SONAME")
                self._load(dependency,visiting)
            elif name not in BASE_DEPENDENCIES:
                self._fail("Unpinned ELF dependency is outside the qualified base: "+name)
        visiting.remove(path)
        self._check_snapshot(record)
        soname = record["elf"]["soname"]
        if soname is not None:
            try:
                previous = ct.CDLL(soname,mode=os.RTLD_NOLOAD | os.RTLD_NOW)
            except OSError:
                pass
            else:
                self._fail("ELF SONAME was already loaded outside this snapshot owner: "+soname)
        native = ct.CDLL("/proc/self/fd/"+str(record["fd"]),
                         mode=os.RTLD_NOW | os.RTLD_GLOBAL)
        self._loaded[path] = native
        rows = self.mapped_rows(Path("/proc/self/maps").read_text())
        self.mapped_sha256(path,rows)
        if soname is not None:
            binding = ct.CDLL(soname,mode=os.RTLD_NOLOAD | os.RTLD_NOW)
            if binding._handle != native._handle:
                self._fail("ELF SONAME does not bind the exact descriptor-loaded object")
            self._soname_handles[soname] = binding

    def load_library(self, path, checksum):
        record = self._record(path, checksum)
        native = self._loaded.get(record["path"])
        if native is None:
            self._fail("Library was not explicitly preloaded by the sealed bootstrap")
        # Return the retained handle, not a new dlopen through a mutable locator.
        self.check_live()
        return native, record["path"]

    def mapped_rows(self, text):
        """Fresh kernel names; translate ONLY exact owned snapshot identities."""
        self.check_live()
        snapshots = {(r["kernel_path"], *r["identity"]): r["path"]
                     for r in self._files.values()}
        rows = []
        for line in text.splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            lo, hi = (int(value, 16) for value in fields[0].split("-"))
            inode = int(fields[4])
            device = tuple(int(value, 16) for value in fields[3].split(":"))
            name = fields[5]
            logical = snapshots.get((name, inode, device))
            if logical is not None:
                name = logical
            elif name.endswith(" (deleted)"):
                if ".so" in name:
                    self._fail("Foreign/deleted binary is not a sealed snapshot")
                continue
            elif _mapped_basename(name).startswith(CUDA_PREFIXES):
                self._fail("Unsealed/foreign mapped CUDA or cuEST library")
            rows.append((lo, hi, name, inode, device))
        self.check_live()
        return rows

    def mapped_sha256(self, path, mappings):
        self.mapped_checks += 1
        self.check_live()
        record = self._files.get(os.fspath(path))
        if record is None:
            self._fail("Mapped library is outside the sealed source set")
        if record["path"] not in self._loaded:
            self._fail("Mapped library has no process-retained loader handle")
        identities = {(inode, device) for _, _, name, inode, device in mappings
                      if name == record["path"]}
        if identities != {record["identity"]}:
            self._fail("Mapped inode/device differs from the sealed snapshot")
        self.check_live()
        return record["sha256"]

    def verify_symbols(self, path, names):
        """Fresh loader-seam audit; no reusable parsed mapping receipt."""
        self.check_live()
        record = self._files.get(os.fspath(path))
        if record is None:
            self._fail("Symbol library is outside the sealed set")
        native = self._loaded.get(record["path"])
        if native is None:
            self._fail("Symbol library is not process-retained")
        rows = self.mapped_rows(Path("/proc/self/maps").read_text())
        self.mapped_sha256(record["path"], rows)
        try:
            for name in names:
                address = ct.cast(getattr(native, name), ct.c_void_p).value
                owners = {p for lo, hi, p, _, _ in rows if lo <= address < hi}
                if owners != {record["path"]}:
                    self._fail("Native symbol is outside its sealed image: " + name)
        except BaseException:
            self._fail("Sealed native symbol admission failed")
        self.check_live()

    def guarded_call(self, function, *args, **kwargs):
        self.check_live()
        try:
            result = function(*args, **kwargs)
        except BaseException:
            self.check_live()
            raise
        self.check_live()
        return result

    def report(self):
        self.check_live()
        return {"contract": CONTRACT, "pid": self.pid, "native_thread": self.native_thread,
                "lifetime": "process exit; no close/renew/retry/transfer",
                "revocation": "caller/lifetime failure terminates dedicated worker",
                "revoked_exit_code": 86, "admitted": True, "invalidated": self.failed,
                "reason": self.reason, "full_hash_calls": self.full_hash_calls,
                "full_hash_bytes": self.full_hash_bytes, "live_checks": self.live_checks,
                "mapped_checks": self.mapped_checks,
                "sealed_snapshot_bytes": self._snapshot_bytes,
                "snapshot_prepare_seconds": self.prepare_seconds,
                "snapshot_preload_seconds": self.preload_seconds,
                "files": {path: {"sha256": r["sha256"], "bytes": r["bytes"],
                                 "seals": r["seals"], "kernel_path": r["kernel_path"],
                                 "elf_needed": r["elf"]["needed"],
                                 "elf_soname": r["elf"]["soname"]}
                          for path, r in self._files.items()}}

    def _discard_unpublished(self):
        if self.published or self._loaded:
            os._exit(86)  # No rollback/unload once native images may be alive.
        errors = []
        for alias in self._aliases_created:
            try:
                alias.unlink()
            except BaseException as error:
                errors.append(error)
        for fd in self._fds:
            try:
                os.close(fd)
            except BaseException:
                os._exit(86)  # Interrupted close cannot safely be retried.
        self._files.clear()
        self._fds.clear()
        self._rollback_complete = True
        if errors:
            raise RuntimeError("Sealed admission alias rollback failed: " + str(errors))
