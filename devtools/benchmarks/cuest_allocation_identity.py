"""Run as the owner's initial exclusive step; no context or package mutation."""
import ctypes as ct
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import uuid


def checked(status):
    if status != 0:
        raise RuntimeError(f"CUDA driver identity call returned {status}")


cuda = ct.CDLL("libcuda.so.1")
checked(cuda.cuInit(0))
device = ct.c_int()
checked(cuda.cuDeviceGet(ct.byref(device), 0))
raw = (ct.c_ubyte * 16)()
checked(cuda.cuDeviceGetUuid(ct.byref(raw), device))
name = ct.create_string_buffer(256)
checked(cuda.cuDeviceGetName(name, len(name), device))
identity = {
    "job_id": int(os.environ["SLURM_JOB_ID"]),
    "node": socket.gethostname(),
    "gpu_uuid": "GPU-" + str(uuid.UUID(bytes=bytes(raw))),
    "gpu_name": name.value.decode(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "slurm_step_gpus": os.environ.get("SLURM_STEP_GPUS"),
}
assert "H200" in identity["gpu_name"], identity
identity["nvml_identity"] = subprocess.check_output(
    ["nvidia-smi", "-i", identity["gpu_uuid"],
     "--query-gpu=uuid,name,driver_version,memory.total",
     "--format=csv,noheader,nounits"], text=True, timeout=15,
).strip()
path = Path(sys.argv[1])
path.write_text(json.dumps(identity, indent=2) + "\n")
print(json.dumps(identity, sort_keys=True))
