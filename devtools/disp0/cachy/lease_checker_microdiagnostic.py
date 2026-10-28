"""Isolated real-kernel-lease microdiagnostic; NOT a Psi4/GPU benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types

ROOT=Path(__file__).resolve().parents[3]
SOURCE="psi4/driver/procrouting/sapt/sapt_cuest_owned_runtime.py"

def child(root):
    data=json.load(sys.stdin)
    module=types.ModuleType("_guard_micro")
    module.__file__=data["label"]
    sys.modules[module.__name__]=module
    exec(compile(data["source"],module.__file__,"exec"),module.__dict__)
    pins={}
    for index in range(17):
        path=root/f"library-{index}.so"
        path.write_bytes(b"CPU lease fixture, no ELF execution")
        pins[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    owner=module.admit_owned_runtime(pins,dedicated_worker=True)
    for _ in range(100):owner.check_live()
    samples=[]
    for _ in range(3):
        start=time.perf_counter()
        for _ in range(2000):owner.check_live()
        samples.append((time.perf_counter()-start)/2000)
    print(json.dumps({"scope":"CPU_LEASE_FIXTURE_NOT_GPU_EVIDENCE",
        "label":data["label"],"files":17,
        "seconds_per_call":samples,"source_sha256":hashlib.sha256(data["source"].encode()).hexdigest(),
        "full_hash_calls":owner.report()["full_hash_calls"]}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child",type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.child is not None:
        child(args.child)
        return
    base=subprocess.check_output(["git","show","250de9f44f:"+SOURCE],cwd=ROOT,text=True)
    for name,source in (("gated-base",base),("working-candidate",(ROOT/SOURCE).read_text())):
        with tempfile.TemporaryDirectory(prefix="disp0-guard-micro-",dir="/dev/shm") as temporary:
            result=subprocess.run([sys.executable,"-B",__file__,"--child",temporary],
                input=json.dumps({"label":name,"source":source}),capture_output=True,text=True,timeout=30)
            assert result.returncode==0,result.stderr
            print(result.stdout.strip(),flush=True)


if __name__=="__main__":
    main()
