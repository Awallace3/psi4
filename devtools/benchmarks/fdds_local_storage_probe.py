"""Bounded allocation-local storage diagnostic; no scientific settings changed."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def local_filesystem(kind):
    return kind.strip() in {"xfs", "ext4", "btrfs", "zfs"}


def write_read_probe(root, nbytes):
    if nbytes <= 0:
        raise ValueError("Probe size must be positive")
    path = root / "write-read-probe.bin"
    chunk = bytes(range(256)) * 4096
    expected = hashlib.sha256()
    left = nbytes
    with path.open("xb") as f:
        while left:
            data = chunk[:min(len(chunk), left)]
            f.write(data)
            expected.update(data)
            left -= len(data)
        f.flush()
        os.fsync(f.fileno())
    observed = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            observed.update(data)
    assert path.stat().st_size == nbytes
    assert expected.digest() == observed.digest()
    return dict(bytes=nbytes, sha256=observed.hexdigest(), fsync=True)


def main():
    job = int(os.environ["SLURM_JOB_ID"])
    assert job == 14214205
    root = Path(f"/scratch/{job}/awallace43-fdds-perf-v1")
    # Create only this fresh, job-owned namespace; no adoption or cleanup of
    # existing scratch, including the failed network-scratch attempt.
    os.umask(0o077)
    root.mkdir(parents=True, exist_ok=False)
    assert root.stat().st_uid == os.getuid()
    kind = subprocess.check_output(
        ["findmnt", "-n", "-o", "FSTYPE", "--target", str(root)],
        text=True, timeout=10).strip()
    assert local_filesystem(kind), kind
    usage = shutil.disk_usage(root)
    assert usage.free >= 160 * 1024**3, usage
    result = write_read_probe(root, 512 * 1024**2)
    result.update(job_id=job, root=str(root), filesystem=kind,
                  available_bytes=usage.free)
    tmp = root / ".STORAGE_VERIFIED.json.tmp"
    with tmp.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(root / "STORAGE_VERIFIED.json")
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
