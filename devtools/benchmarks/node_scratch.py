"""Select private writable disk scratch inside a compute allocation."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def select(candidates, job_id):
    for candidate in dict.fromkeys(candidates):
        if not candidate:
            continue
        base = Path(candidate)
        if not base.is_dir() or not os.access(base, os.W_OK | os.X_OK):
            continue
        try:
            filesystem = subprocess.check_output(
                ["stat", "-f", "-c", "%T", str(base)], text=True).strip()
            # Do not silently put Psi4 scratch on network storage or RAM disks.
            if filesystem not in ("xfs", "ext2/ext3", "btrfs", "zfs"):
                print(f"Reject scratch {base}: filesystem={filesystem}", file=sys.stderr)
                continue
            stat = os.statvfs(base)
            if stat.f_bavail * stat.f_frsize < 20 * 1024**3 or stat.f_favail < 1000:
                print(f"Reject scratch {base}: insufficient free disk/inodes", file=sys.stderr)
                continue
            target = Path(tempfile.mkdtemp(prefix=f"psi4-protein157-{job_id}-", dir=base))
            probe = target / ".write-probe"
            probe.write_bytes(b"scratch probe\n")
            probe.unlink()
            print(f"Selected scratch {target}, filesystem={filesystem}", file=sys.stderr)
            return target
        except OSError as exc:
            print(f"Reject scratch {base}: {exc}", file=sys.stderr)
    raise RuntimeError("No writable node-local disk scratch with >=20 GiB free")


if __name__ == "__main__":
    job = os.environ["SLURM_JOB_ID"]
    user = os.environ["USER"]
    print(select([os.environ.get("SLURM_TMPDIR", ""), os.environ.get("TMPDIR", ""),
                  f"/scratch/{user}", "/scratch", "/tmp"], job))
