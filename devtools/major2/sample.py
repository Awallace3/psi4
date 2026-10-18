"""1Hz telemetry, supported optional fields only; no clock/cache/GPU state changes."""
import datetime
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

running = True


def stop(*_):
    global running
    running = False


if __name__ == "__main__":
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    path, uuid = sys.argv[1:]
    help_text = subprocess.check_output(["nvidia-smi", "--help-query-gpu"], text=True, timeout=15)
    required = ["uuid", "memory.used", "memory.total", "utilization.gpu"]
    optional = ["power.draw", "power.limit", "temperature.gpu", "clocks.sm", "clocks.mem",
                "clocks_event_reasons.active", "clocks_throttle_reasons.active"]
    supported = []
    for field in optional:
        if '"' + field + '"' in help_text:
            check = subprocess.run(["nvidia-smi", "--id="+uuid, "--query-gpu="+field,
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            if check.returncode == 0:
                supported.append(field)
    fields = required + supported
    with Path(path).open("x", buffering=1) as f:
        f.write(json.dumps(dict(fields=fields, unsupported_optional=sorted(set(optional)-set(fields)))) + "\n")
        while running:
            start = time.monotonic()
            result = subprocess.run(["nvidia-smi", "--id="+uuid, "--query-gpu="+",".join(fields),
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            optional_error = None
            sample_fields = fields
            if result.returncode:
                optional_error = result.stderr
                sample_fields = required
                result = subprocess.run(["nvidia-smi", "--id="+uuid, "--query-gpu="+",".join(required),
                                         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                if result.returncode:
                    raise RuntimeError("required telemetry query failed: " + result.stderr)
            values = [x.strip() for x in result.stdout.strip().split(",")]
            if not values or values[0] != uuid:
                raise RuntimeError("telemetry UUID changed")
            apps = subprocess.run(["nvidia-smi", "--id="+uuid, "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                                   "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            if apps.returncode:
                raise RuntimeError("compute-apps observation failed: " + apps.stderr)
            processes = []
            for line in apps.stdout.splitlines():
                fields_app = [x.strip() for x in line.split(",")]
                if len(fields_app) == 4:
                    processes.append(dict(zip(["gpu_uuid", "pid", "process_name", "used_memory_mib"], fields_app)))
            f.write(json.dumps(dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                    values=dict(zip(sample_fields, values)), compute_apps=processes,
                                    optional_query_error=optional_error)) + "\n")
            time.sleep(max(0, 1-(time.monotonic()-start)))
