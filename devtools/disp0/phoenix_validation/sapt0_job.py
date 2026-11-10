"""Allocated-node launcher. Build evidence and water gate precede timing."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from sapt0_timing import SYSTEMS, save, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build-root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--system", choices=SYSTEMS, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if int(os.environ["SLURM_CPUS_PER_TASK"]) != 8:
        raise RuntimeError("Matched eight-core comparison required")
    build_id = (args.build_root / "job-id.txt").read_text().strip()
    build_run = args.build_root / ("job-" + build_id)
    status = (build_run / "job-status.env").read_text()
    if "exit_code=0\n" not in status or "term_received=0\n" not in status:
        raise RuntimeError("Build was not completed cleanly")
    proof = json.loads((build_run / "result.json").read_text())
    if proof["status"] != "PASS":
        raise RuntimeError("Coherent build validation did not pass")
    original = build_run / "gpu-runtime/runtime-config.json"
    config = json.loads(original.read_text())
    if config["source_pin"] != proof["pin"]:
        raise RuntimeError("Different runtime and build pin")
    config.update(scope="MATCHED_CANONICAL_SAPT0_TIMING", threads=8,
                  host_memory="96 GiB", device_memory_limit_bytes=120 * 1024**3,
                  native_persistent_bytes=12 * 1024**3)
    config_path = args.output / "runtime-config.json"
    save(config_path, config)
    hardware = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total",
        "--format=csv,noheader,nounits"], text=True, timeout=20)
    save(args.output / "launch.json", {
        "hardware": hardware, "original_runtime_config": str(original),
        "original_runtime_sha256": sha(original), "build_proof_sha256": sha(build_run / "result.json"),
        "native_source_pin": proof["pin"], "helper_pin": os.environ["SAPT0_PIN"],
        "runtime_config_sha256": sha(config_path), "system": args.system,
        "input_hashes": {n: sha(args.inputs / (n + ".json")) for n in ("water", args.system)},
        "threads_each_arm": 8, "cpu_model": Path("/proc/cpuinfo").read_text(),
        "scheduling": {"job": os.environ["SLURM_JOB_ID"], "qos": os.environ.get("SLURM_JOB_QOS")}})
    # No launch of a larger case if the public CPU/GPU water gate fails.
    script = Path(__file__).with_name("sapt0_timing.py")
    for system, repeats in (("water", 1), (args.system, 3)):
        command = [sys.executable, "-B", str(script), "run",
                   "--config", str(config_path), "--config-sha256", sha(config_path),
                   "--case", str(args.inputs / (system + ".json")),
                   "--output", str(args.output / system), "--repeats", str(repeats),
                   "--timeout", "3600"]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
