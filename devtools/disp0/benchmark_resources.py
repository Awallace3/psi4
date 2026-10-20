"""Per-point RAM/quota checks; a zero-byte child verifies scratch ownership."""

import os
from pathlib import Path
import shutil
import re
import subprocess
import tempfile

if not __debug__:
    raise RuntimeError("Dispersion validation requires assertions; optimized Python is forbidden")

GiB = 1024**3


def effective_cgroup_memory(membership, mountinfo, read_text):
    """Use the process's memory cgroup and every visible ancestor, not root."""
    members = {}
    for line in membership.splitlines():
        _, controllers, path = line.split(":", 2)
        for controller in controllers.split(","):
            members[controller] = Path(path)
    bounds = []
    found = False

    def unescape(value):
        return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)

    for line in mountinfo.splitlines():
        before, after = line.split(" - ", 1)
        fields, fs = before.split(), after.split()
        if fs[0] == "cgroup2" and "" in members:
            member, limit_name, current_name = members[""], "memory.max", "memory.current"
        elif fs[0] == "cgroup" and "memory" in fs[2].split(",") and "memory" in members:
            member = members["memory"]
            limit_name, current_name = "memory.limit_in_bytes", "memory.usage_in_bytes"
        else:
            continue
        mount_root, mountpoint = Path(unescape(fields[3])), Path(unescape(fields[4]))
        # Do not guess a namespace root or accept a bind mount that hides
        # potentially tighter ancestors. Native Slurm's full hierarchy works.
        if member == Path("/") or mount_root != Path("/") or not member.is_relative_to(mount_root):
            continue
        current = mountpoint / member.relative_to(mount_root)
        found = True
        mount_bounds = []
        while True:
            try:
                limit = read_text(current / limit_name).strip()
            except FileNotFoundError:
                # A real cgroup-v2 hierarchy root lacks memory.max. A finite
                # visible descendant proves this controller is enabled; an
                # ordinary/hidden cgroup with missing controls is not accepted.
                if fs[0] == "cgroup2" and current == mountpoint and mount_bounds:
                    break
                raise
            if limit != "max" and int(limit) < 2**60:  # v1 unlimited sentinel
                used = int(read_text(current / current_name).strip())
                mount_bounds.append({"path": str(current), "limit": int(limit), "current": used,
                                     "available": int(limit) - used})
            if current == mountpoint:
                break
            current = current.parent
        bounds.extend(mount_bounds)
    if not found or not bounds:
        raise RuntimeError("Cannot establish effective finite process memory cgroup limits")
    return {"bounds": bounds, "available": min(row["available"] for row in bounds)}


def project_policy(parent_text, child_text):
    def parse(text):
        fields = text.split()
        assert len(fields) >= 3 and fields[1] in ("P", "-"), f"Unknown project policy: {text}"
        return int(fields[0]), fields[1] == "P"
    parent, inherits = parse(parent_text)
    child, _ = parse(child_text)
    if inherits:
        assert child == parent, "Project inheritance disagrees with new child inode"
    return {"parent_id": parent, "inheritance": inherits, "new_child_id": child}


def inspect_scratch_policy(root):
    parent = subprocess.check_output(["lfs", "project", "-d", str(root)], text=True)
    descriptor, name = tempfile.mkstemp(prefix="resource-proof-", dir=root)
    try:
        info = os.fstat(descriptor)
        child = subprocess.check_output(["lfs", "project", "-d", name], text=True)
        policy = project_policy(parent, child)
        return info, policy, parent, child
    finally:
        os.close(descriptor)
        os.unlink(name)


def quota_headroom(text):
    """Parse the byte/inode limits from lfs's default KiB quota table."""
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0].startswith("/"):
            fields = fields[1:]
        elif len(fields) != 8:
            continue
        if len(fields) != 8:
            continue
        try:
            used, soft, hard = (int(value.rstrip("*")) for value in fields[:3])
            files, inode_soft, inode_hard = (int(value.rstrip("*")) for value in fields[4:7])
        except ValueError:
            continue
        finite_bytes = [value for value in (soft, hard) if value]
        finite_inodes = [value for value in (inode_soft, inode_hard) if value]
        rows.append({"byte_headroom": (min(finite_bytes) - used) * 1024 if finite_bytes else None,
                     "inode_headroom": min(finite_inodes) - files if finite_inodes else None})
    assert len(rows) == 1, f"Ambiguous/missing Lustre quota row: {text}"
    return rows[0]


def check(system):
    root = Path(os.environ["PSI_SCRATCH"]).resolve()
    job = os.environ["SLURM_JOB_ID"]
    owner = Path("/storage/scratch1/3/awallace43/saptdft-cuest-disp0")
    assert root.is_relative_to(owner) and f"timing-{job}" in root.relative_to(owner).parts
    assert root.is_dir() and os.access(root, os.W_OK)
    needed_disk = {"peptide": 32, "nanotube": 64, "protein83": 200, "protein157": 600}[system] * GiB
    needed_ram = {"peptide": 4, "nanotube": 16, "protein83": 384, "protein157": 384}[system] * GiB
    meminfo = {line.split(":")[0]: int(line.split()[1]) * 1024
               for line in Path("/proc/meminfo").read_text().splitlines()
               if line.startswith("MemAvailable:")}
    available_ram = meminfo["MemAvailable"]
    cgroup = effective_cgroup_memory(Path("/proc/self/cgroup").read_text(),
                                    Path("/proc/self/mountinfo").read_text(),
                                    lambda path: path.read_text())
    available_ram = min(available_ram, cgroup["available"])
    assert available_ram >= needed_ram, f"Insufficient available RAM: {available_ram} < {needed_ram}"
    disk_free = shutil.disk_usage(root).free
    assert disk_free >= needed_disk, f"Insufficient filesystem free bytes: {disk_free} < {needed_disk}"
    # Inspect an actual newly created inode, including inherited group/project
    # policy. The private zero-byte probe is removed outside measured work.
    info, policy, parent_project, child_project = inspect_scratch_policy(root)
    user = subprocess.check_output(["lfs", "quota", "-u", str(info.st_uid), str(root)], text=True)
    gid = info.st_gid
    group = subprocess.check_output(["lfs", "quota", "-g", str(gid), str(root)], text=True)
    project_id = policy["new_child_id"]
    project = subprocess.check_output(["lfs", "quota", "-p", str(project_id), str(root)], text=True)
    quotas = {"user": quota_headroom(user), "group": quota_headroom(group),
              "project": quota_headroom(project)}
    for name, quota in quotas.items():
        assert quota["byte_headroom"] is None or quota["byte_headroom"] >= needed_disk, \
            f"Insufficient {name} scratch quota: {quota}"
        assert quota["inode_headroom"] is None or quota["inode_headroom"] >= 1000, \
            f"Insufficient {name} scratch inodes: {quota}"
    return {"status": "PASS", "scratch": str(root), "required_free_bytes": needed_disk,
            "filesystem_free_bytes": disk_free, "quotas": quotas, "quota_user_raw": user,
            "quota_group_raw": group, "effective_scratch_gid": gid, "project_id": project_id,
            "project_policy": policy, "project_parent_raw": parent_project,
            "project_child_raw": child_project, "quota_project_raw": project,
            "required_available_ram": needed_ram,
            "available_ram": available_ram, "cgroup": cgroup}
