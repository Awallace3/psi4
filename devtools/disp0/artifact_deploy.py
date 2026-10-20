"""Bounded, whole-artifact installation; never build, patch, overlay or retry."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import tempfile

if not __debug__:
    raise RuntimeError("Artifact validation requires assertions; optimized Python is forbidden")

P = Path("/storage/project/r-cs207-0/awallace43")
PARENT = P / "runs/saptdft-cuest-disp0"
SOURCE = PARENT / "build-587e172a33afaadb00eb7f574a5089533f7fc755-14196330"
EXPECTED = {
    "psi4-build/stage/lib/psi4/core.cpython-313-x86_64-linux-gnu.so":
        "b3cfe2f2261533df9a9018731d9fe2bffea41cfe70f53f14195caaee7de8697d",
    "einsums-stage/lib/einsums/_core.cpython-313-x86_64-linux-gnu.so":
        "8f23c69b79613f9b94eee95d4735e8f88f67be7c98867290f7546ca56bdc6833",
    "einsums-stage/lib/libEinsums-abi1.so.2.0.0":
        "9e31b5ff8cf28e796b04ee74853c4923f06acf03f1af24dfddd75b79de583cb7",
}


def digest(path, deadline):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            if time.monotonic() >= deadline:
                raise TimeoutError("Artifact I/O deadline exceeded; no automatic retry")
            result.update(chunk)
    return result.hexdigest()


def inventory(source, destination, deadline):
    origins = [(source / "psi4-build/stage", Path("psi4-build/stage")),
               (source / "einsums-build/lib/einsums", Path("einsums-stage/lib/einsums"))]
    origins += [(source / "einsums-build/lib" / name, Path("einsums-stage/lib") / name)
                for name in ("libEinsums-abi1.so", "libEinsums-abi1.so.1", "libEinsums-abi1.so.2.0.0")]
    records = {}
    for root, target in origins:
        assert root.exists(), f"Complete artifact missing: {root}"
        if root.is_dir():
            assert not root.is_symlink(), f"Symlinked artifact root needs review: {root}"
        paths = [root, *root.rglob("*")] if root.is_dir() else [root]
        for path in paths:
            if time.monotonic() >= deadline:
                raise TimeoutError("Artifact preflight exceeded deadline")
            relative = target / path.relative_to(root) if path != root else target
            key = str(relative)
            assert key not in records, f"Overlapping artifact selection: {key}"
            ancestor = path.parent
            while ancestor.is_relative_to(source):
                assert not ancestor.is_symlink(), f"Symlinked artifact ancestor: {ancestor}"
                ancestor = ancestor.parent
            entry = {"origin": str(path), "destination": str(destination / relative),
                     "mode": path.lstat().st_mode & 0o7777}
            if path.is_symlink():
                link = os.readlink(path)
                assert not Path(link).is_absolute(), f"Absolute artifact symlink needs review: {path}"
                assert path.resolve().is_relative_to(source), f"Artifact link escapes source run: {path}"
                entry.update(kind="symlink", target=link)
            elif path.is_dir():
                entry["kind"] = "directory"
            elif path.is_file():
                entry.update(kind="file", bytes=path.stat().st_size, sha256=digest(path, deadline))
            else:
                raise RuntimeError(f"Unsupported artifact type: {path}")
            records[key] = entry
    validate_relocated_links(records, destination)
    return records


def validate_relocated_links(records, destination):
    """Prove the complete destination graph is internal BEFORE any writes."""
    for relative in records:
        ancestor = Path(relative).parent
        while ancestor != Path("."):
            assert str(ancestor) not in records or records[str(ancestor)]["kind"] == "directory", \
                f"Artifact path would write through a symlink: {relative}"
            ancestor = ancestor.parent

    def follow(relative, seen):
        assert relative not in seen, f"Cyclic relocated artifact link: {relative}"
        entry = records[relative]
        if entry["kind"] != "symlink":
            return
        target = Path(os.path.normpath(str((destination / relative).parent / entry["target"])))
        assert target.is_relative_to(destination), f"Relocated link escapes new prefix: {relative}"
        key = str(target.relative_to(destination))
        assert key in records, f"Relocated link target was not selected: {relative} -> {key}"
        follow(key, {*seen, relative})

    for relative, entry in records.items():
        if entry["kind"] == "symlink":
            follow(relative, set())


class OwnedPrefix:
    """All writes use retained directory descriptors, never mutable paths."""

    def __init__(self, destination, fd):
        self.destination = destination
        info = os.fstat(fd)
        self.identity = info.st_dev, info.st_ino
        self.directories = {Path("."): fd}

    def directory(self, relative):
        relative = Path(relative)
        assert not relative.is_absolute() and ".." not in relative.parts
        if relative in self.directories:
            return self.directories[relative]
        parent = self.directory(relative.parent)
        os.mkdir(relative.name, mode=0o700, dir_fd=parent)  # never adopt an existing foreign directory
        fd = os.open(relative.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                     dir_fd=parent)
        self.directories[relative] = fd
        return fd

    def close(self):
        for fd in getattr(self, "directories", {}).values():
            os.close(fd)
        self.directories = {}

    def __del__(self):
        self.close()


class PrivateNamespace:
    """Trusted, unpublished 0700 namespace; its nonce is not a public target."""

    def __init__(self, parent, label):
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            info = os.fstat(parent_fd)
            assert info.st_uid == os.geteuid() and info.st_mode & 0o022 == 0, "Untrusted namespace parent"
            # Bootstrap through the trusted parent FD, not a mutable public
            # destination name. No artifact paths are disclosed until complete.
            created = tempfile.mkdtemp(prefix=label + "-", dir=f"/proc/self/fd/{parent_fd}")
            name = Path(created).name
            self.path = parent / name
            self.fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                              dir_fd=parent_fd)
            info = os.fstat(self.fd)
            assert info.st_uid == os.geteuid() and info.st_mode & 0o777 == 0o700
        finally:
            os.close(parent_fd)

    def __del__(self):
        if hasattr(self, "fd"):
            os.close(self.fd)


def create_prefix(destination, *, private_parent_fd):
    info = os.fstat(private_parent_fd)
    assert info.st_uid == os.geteuid() and info.st_mode & 0o777 == 0o700, \
        "Artifact acquisition requires an already trusted private namespace"
    parent_info = destination.parent.lstat()
    assert (parent_info.st_dev, parent_info.st_ino) == (info.st_dev, info.st_ino)
    os.mkdir(destination.name, mode=0o700, dir_fd=private_parent_fd)
    fd = os.open(destination.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                 dir_fd=private_parent_fd)
    return OwnedPrefix(destination, fd)


def verify_owned_prefix(destination, ownership):
    assert ownership is not None and not destination.is_symlink()
    info = destination.lstat()
    assert (info.st_dev, info.st_ino) == ownership.identity, "Deployment prefix ownership changed"


def install(records, destination, deadline, ownership):
    """Copy whole reviewed trees unchanged; never write a source artifact."""
    verify_owned_prefix(destination, ownership)
    for relative, entry in sorted(records.items(), key=lambda item: (item[0].count("/"), item[0])):
        if time.monotonic() >= deadline:
            raise TimeoutError("Artifact installation exceeded deadline; preserve partial prefix")
        relative = Path(relative)
        parent = ownership.directory(relative.parent)
        if entry["kind"] == "directory":
            ownership.directory(relative)
        elif entry["kind"] == "symlink":
            os.symlink(entry["target"], relative.name, dir_fd=parent)
        else:
            fd = os.open(relative.name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                         mode=0o600, dir_fd=parent)
            with os.fdopen(fd, "w+b") as output, open(entry["origin"], "rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Artifact copy exceeded deadline")
                    output.write(chunk)
                output.flush()
                os.fchmod(output.fileno(), entry["mode"])
                assert os.fstat(output.fileno()).st_size == entry["bytes"]
                output.seek(0)
                observed = hashlib.sha256()
                for chunk in iter(lambda: output.read(1024 * 1024), b""):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Artifact destination audit exceeded deadline")
                    observed.update(chunk)
                entry["destination_sha256"] = observed.hexdigest()
                assert entry["destination_sha256"] == entry["sha256"], f"Artifact byte mismatch: {relative}"
                entry["destination_mode"] = os.fstat(output.fileno()).st_mode & 0o7777
                assert entry["destination_mode"] == entry["mode"]
    verify_owned_prefix(destination, ownership)
    for relative, entry in records.items():
        relative = Path(relative)
        if entry["kind"] == "symlink":
            assert os.readlink(relative.name, dir_fd=ownership.directory(relative.parent)) == entry["target"]
    for relative, entry in sorted(records.items(), key=lambda item: item[0].count("/"), reverse=True):
        if entry["kind"] == "directory":
            os.fchmod(ownership.directory(relative), entry["mode"])


def verify_recipe(repository, script, pin):
    actual = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    assert actual == pin and len(pin) == 40
    relative = script.resolve().relative_to(repository.resolve())
    blob = subprocess.check_output(["git", "-C", str(repository), "show", f"{pin}:{relative.as_posix()}"])
    assert hashlib.sha256(blob).digest() == hashlib.sha256(script.read_bytes()).digest(), \
        "Executing installer differs from its pinned Git blob"
    subprocess.run(["git", "-C", str(repository), "diff", "HEAD", "--quiet"], check=True)
    subprocess.run(["git", "-C", str(repository), "diff", "--cached", "--quiet"], check=True)


def write_owned_manifest(destination, report, ownership):
    if ownership is None:
        return False  # failed exclusive mkdir: NEVER write into another deployment
    verify_owned_prefix(destination, ownership)
    fd = os.open("deployment-manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 mode=0o600, dir_fd=ownership.directory(Path(".")))
    with os.fdopen(fd, "w") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    verify_owned_prefix(destination, ownership)
    return True


def finalize_manifest(destination, report, ownership, timeout_seconds=10):
    """Hard finalization window; a stalled failure report cannot overrun120s."""
    def stop(signum, frame):
        os._exit(124)  # no recursive cleanup or blocking second manifest attempt
    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return write_owned_manifest(destination, report, ownership)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def run_bounded_worker(work, timeout_seconds=110):
    """Independent parent watches through stdout flush and worker termination."""
    if not all(hasattr(os, name) for name in ("fork", "waitid", "WNOWAIT", "setsid", "pipe2")):
        raise RuntimeError("A POSIX unreaped-child watchdog is required; no unbounded fallback")
    ready_read, ready_write = os.pipe2(os.O_CLOEXEC)
    child = os.fork()
    if child == 0:
        os.close(ready_write)
        os.setsid()  # the watchdog can kill only this worker and its own readelf descendants
        status = 0
        try:
            if os.read(ready_read, 1) != b"1":
                raise RuntimeError("Deployment watchdog was not armed")
            os.close(ready_read)
            work()
        except SystemExit as exc:
            status = exc.code if isinstance(exc.code, int) else 1
        except BaseException:
            traceback.print_exc()
            status = 1
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except BaseException:
            status = 1
        os._exit(status)
    os.close(ready_read)
    def stop(signum, frame):
        # waitid(WNOWAIT) retains the child PID, so its private process-group
        # identifier cannot be recycled into an unrelated group.
        try:
            os.killpg(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.kill(child, signal.SIGKILL)  # unreaped child PID cannot have been recycled
        except ProcessLookupError:
            pass
        os._exit(124)

    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        os.write(ready_write, b"1")
        os.close(ready_write)
    except BaseException:
        stop(signal.SIGTERM, None)
    result = os.waitid(os.P_PID, child, os.WEXITED | os.WNOWAIT)
    try:
        os.killpg(child, signal.SIGKILL)  # clean only private descendants on any completion
    except ProcessLookupError:
        pass
    status = result.si_status if result.si_code == os.CLD_EXITED else 128 + result.si_status
    os._exit(status)  # no parent output/atexit path can escape the live watchdog


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="new private runtime namespace label; path reported only after completion")
    parser.add_argument("--recipe-pin", required=True)
    parser.add_argument("--quota-headroom-bytes", type=int, required=True)
    parser.add_argument("--quota-receipt", required=True, help="fresh durable quota evidence reference")
    args = parser.parse_args()
    started = time.monotonic()
    deadline = started + 90  # leave margin below the explicit two-minute bound

    def abort(signum, frame):
        raise TimeoutError(f"Artifact deployment interrupted ({signum}); preserve partial prefix")

    signal.signal(signal.SIGALRM, abort)
    signal.signal(signal.SIGTERM, abort)
    signal.setitimer(signal.ITIMER_REAL, 90)
    assert args.label.startswith("runtime-587e172a33-") and all(
        character.isalnum() or character == "-" for character in args.label)
    # Layout-only placeholder for preflight; no caller-supplied/public runtime
    # directory is acquired. The real prefix is generated privately afterward.
    destination = PARENT / (args.label + "-unpublished") / "image"
    repository = Path(__file__).resolve().parents[2]
    pin = args.recipe_pin
    verify_recipe(repository, Path(__file__), pin)
    records = inventory(SOURCE, destination, deadline)
    for relative, expected in EXPECTED.items():
        assert records[relative]["sha256"] == expected, f"Unexpected approved binary: {relative}"
    total = sum(entry.get("bytes", 0) for entry in records.values())
    required = total * 2 + 1024**3
    assert shutil.disk_usage(PARENT).free >= required, "Destination free-space preflight failed"
    assert args.quota_headroom_bytes >= required, "Destination quota preflight failed"
    namespace = PrivateNamespace(PARENT, args.label)
    destination = namespace.path / "image"
    for relative, entry in records.items():
        entry["destination"] = str(destination / relative)
    validate_relocated_links(records, destination)
    report = {"status": "INSTALLING", "recipe_pin": pin, "source_run": str(SOURCE),
              "destination": str(destination), "source_job_status": "FAILED; never relabelled",
              "files_and_links": records, "selected_binary_sha256": EXPECTED, "artifact_bytes": total,
              "required_headroom_bytes": required, "quota_headroom_bytes": args.quota_headroom_bytes,
              "quota_receipt": args.quota_receipt, "abi": {},
              "scope": "artifact installation only; no import/API/numerical/device claim"}
    ownership = None
    try:
        ownership = create_prefix(destination, private_parent_fd=namespace.fd)
        install(records, destination, deadline, ownership=ownership)
        for relative in EXPECTED:
            pair = {}
            for label, path in (("origin", records[relative]["origin"]),
                                ("destination", str(destination / relative))):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Artifact ABI audit exceeded deadline")
                elf = subprocess.check_output(["readelf", "-d", path],
                                              text=True, timeout=min(5, remaining))
                pair[label] = [line.strip() for line in elf.splitlines()
                               if any(key in line for key in ("NEEDED", "SONAME", "RPATH", "RUNPATH"))]
            assert pair["origin"] == pair["destination"], f"ELF metadata changed: {relative}"
            report["abi"][relative] = pair
        report["status"] = "ARTIFACT_BYTES_VERIFIED"
    except BaseException as exc:
        report["status"] = "FAILED"
        report["exception"] = repr(exc)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        finalize_manifest(destination, report, ownership,
                          timeout_seconds=max(0.01, min(10, 110 - report["elapsed_seconds"])))
    print(json.dumps({key: report[key] for key in
                      ("status", "recipe_pin", "destination", "artifact_bytes", "elapsed_seconds")}))


if __name__ == "__main__":
    run_bounded_worker(main)
