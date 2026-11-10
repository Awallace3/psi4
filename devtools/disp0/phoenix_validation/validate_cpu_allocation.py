"""Cheap allocation-local admission before native imports or benchmark calls."""
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess


def fields(text):
    return dict(token.split("=",1) for token in shlex.split(text) if "=" in token)


def seconds(text):
    days,clock=(text.split("-",1) if "-" in text else ("0",text))
    h,m,s=map(int,clock.split(":"))
    return int(days)*86400+h*3600+m*60+s


def memory_bytes(text):
    match=re.fullmatch(r"(\d+(?:\.\d+)?)([KMGTP]?)",text)
    if not match:raise ValueError("Unknown Slurm memory format")
    amount,unit=match.groups()
    return int(float(amount)*1024**{"K":1,"M":2,"G":3,"T":4,"P":5,"":2}[unit])


def validate_job(job,manifest):
    if (job.get("Account")!=manifest["account"] or job.get("QOS")!=manifest["qos"]
            or set(job.get("Partition","").split(","))!={"cpu-small"}
            or job.get("NumNodes")!="1" or job.get("NumCPUs")!="24"
            or job.get("CPUs/Task")!="24" or job.get("NumTasks")!="1"):
        raise RuntimeError("Effective CPU allocation/account/QoS differs from approved manifest")
    if not (job.get("OverSubscribe")=="NO" or job.get("Shared")=="0"):
        raise RuntimeError("Exclusive CPU allocation was not established")
    if seconds(job["TimeLimit"])!=seconds(manifest["wall_time"]):
        raise RuntimeError("Effective CPU wall limit changed")
    if memory_bytes(job["MinMemoryNode"])!=manifest["memory_GiB"]*1024**3:
        raise RuntimeError("Effective CPU memory allocation changed")


def main():
    root=Path(os.environ["DISP0_JOB_RUN"])
    manifest=json.loads((Path(os.environ["DISP0_RUN_ROOT"])/"manifest.json").read_text())
    record={"status":"STARTED","job_id":os.environ["SLURM_JOB_ID"]}
    try:
        def query(argv):
            return subprocess.check_output(argv,text=True,timeout=30).strip()
        raw=query(["scontrol","show","job","-o",record["job_id"]])
        record["scheduler_job"]=raw
        validate_job(fields(raw),manifest)
        node=os.environ["SLURMD_NODENAME"]
        node_raw=query(["scontrol","show","node","-o",node])
        record["scheduler_node"]=node_raw
        node_fields=fields(node_raw)
        active=set(node_fields.get("ActiveFeatures",node_fields.get("AvailableFeatures","")).split(","))
        if not {"core24","gold6226","localNVMe"}<=active:
            raise RuntimeError("Actual node features differ from selected CPU baseline")
        hardware=json.loads(query(["lscpu","--json"]))
        model=next(v["data"] for v in hardware["lscpu"] if v["field"]=="Model name:")
        if not re.search(r"\bGold\s+6226\b",model):
            raise RuntimeError("Actual CPU is not the selected Xeon Gold6226")
        record["hardware"]=hardware
        record["cpu_affinity"]=sorted(os.sched_getaffinity(0))
        if len(record["cpu_affinity"])<24:
            raise RuntimeError("Fewer than24 allowed CPU threads")
        scratch=Path(os.environ["TMPDIR"]).resolve(strict=True)
        mount_raw=query(["findmnt","-T",str(scratch),"-n","-P","-o","SOURCE,FSTYPE,TARGET"])
        mount=fields(mount_raw)
        record["scratch_mount"]=mount
        if mount.get("FSTYPE") not in {"xfs","ext4","ext3","btrfs"}:
            raise RuntimeError("Benchmark scratch is not a verified node-local disk filesystem")
        disk=shutil.disk_usage(scratch)
        record["scratch_free_bytes_before_calls"]=disk.free
        record["scratch_total_bytes"]=disk.total
        if disk.free<256*1024**3:
            raise RuntimeError("Insufficient actual node-local scratch headroom")
        record["status"]="PASS_CPU_ALLOCATION_ONLY"
    except BaseException as error:
        record.update(status="FAILED",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        with (root/"allocation.json").open("x") as stream:
            json.dump(record,stream,indent=2,sort_keys=True)


if __name__=="__main__":main()
