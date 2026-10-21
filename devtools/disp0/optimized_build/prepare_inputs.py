"""One CPU-only input attempt: strict decoder + separately capped loader trace.

The trace worker is forked BEFORE lowering the decoder's AS hard limit. It
never decodes archives, provisions, imports scientific libraries or runs device
constructors. There is no worker respawn, retry or unbounded fallback.
"""
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import provision_sdk_inputs as inputs

LOADER_AS = 8 * 1024**3  # virtual mappings, not an 8GiB physical allocation
MAX_AUDIT_BYTES = 1024**2
MAX_FRAME = 64 * 1024
MAX_TRACES = len(inputs.contract.COMPONENTS)


def validate_request(request, vendor):
    argv, env = request["argv"], request["env"]
    inputs.contract.require(len(argv) == 2 and argv[0] == "/usr/bin/ldd", "Only loader tracing permitted")
    path = Path(argv[1])
    root = (vendor / "targets/x86_64-linux/lib").resolve(strict=True)
    inputs.contract.require(path.is_absolute() and path.is_file()
                            and path.resolve(strict=True) == path
                            and path.parent == root, "Unapproved loader target")
    inputs.contract.require(any(re.fullmatch(
        r"lib" + re.escape(family) + rf"\.so\.{major}(?:\.\d+)*", path.name)
        for family, major in inputs.contract.COMPONENTS.items()),
                            "Unapproved CUDA component family")
    expected = {"PATH": "/usr/bin:/bin", "LC_ALL": "C",
                "LD_LIBRARY_PATH": f"{root}:{vendor}/lib"}
    inputs.contract.require(env == expected, "Foreign loader environment")


def loader_child_limits():
    resource.setrlimit(resource.RLIMIT_AS, (LOADER_AS, LOADER_AS))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))


def bounded_stream(value):
    raw = value if isinstance(value, bytes) else (value or "").encode()
    return raw[:4096].decode(errors="replace"), len(raw), len(raw) > 4096


def trace_command(request):
    """Keep timeout partial output and explicit truncation, never admit either."""
    try:
        result = subprocess.run(
            request["argv"], env=request["env"], capture_output=True, text=True,
            timeout=10, preexec_fn=loader_child_limits)
        code, observed = result.returncode, result.returncode
        stdout, stderr = result.stdout, result.stderr
        status = "PASS" if code == 0 else "NONZERO"
    except subprocess.TimeoutExpired as exc:
        code, observed, status = -1, None, "TIMEOUT"
        stdout, stderr = exc.stdout, exc.stderr
    except Exception as exc:
        code, observed, status = -1, None, "ERROR"
        stdout, stderr = "", repr(exc)
    out, out_bytes, out_truncated = bounded_stream(stdout)
    err, err_bytes, err_truncated = bounded_stream(stderr)
    if out_truncated or err_truncated:
        code = -1
        status += "_OUTPUT_LIMIT"
    return {"argv": request["argv"], "exit": code, "observed_exit": observed,
            "status": status, "stdout": out, "stderr": err,
            "stdout_bytes": out_bytes, "stderr_bytes": err_bytes,
            "stdout_truncated": out_truncated, "stderr_truncated": err_truncated}


def loader_worker(connection, vendor, log, parent_connection):
    """Exactly one serial trace per component, within the existing 110s wall."""
    parent_connection.close()  # no inherited peer end that could hide decoder EOF
    resource.setrlimit(resource.RLIMIT_AS, (LOADER_AS, LOADER_AS))
    resource.setrlimit(resource.RLIMIT_CPU, (70, 70))
    import signal
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("loader wall deadline")))
    signal.alarm(110)
    written = 0
    seen = set()
    try:
        with log.open("xb") as stream:
            directory = os.open(log.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            for _ in range(MAX_TRACES):
                try:
                    request = json.loads(connection.recv_bytes(MAX_FRAME))
                except EOFError:
                    return
                start = time.monotonic()
                try:
                    validate_request(request, vendor)
                    inputs.contract.require(request["argv"][1] not in seen, "Duplicate loader request; no retry")
                    seen.add(request["argv"][1])
                    response = trace_command(request)
                except Exception as exc:
                    response = {"exit": -1, "stdout": "", "stderr": repr(exc)}
                response.update(pid=os.getpid(), address_space_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],
                                elapsed_seconds=time.monotonic() - start,
                                scope="CPU_LOAD_RESOLVE_ONLY_NO_DEVICE_OR_NUMERICS")
                data = (json.dumps(response, sort_keys=True) + "\n").encode()
                inputs.contract.require(len(data) <= MAX_FRAME and
                                        written + len(data) <= MAX_AUDIT_BYTES, "Loader evidence bound")
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())  # raw streams durable BEFORE acknowledging
                written += len(data)
                connection.send_bytes(data)
                if response["exit"] != 0:
                    return
    finally:
        connection.close()


class LoaderTrace:
    def __init__(self, connection, pid, audit_log=None):
        self.connection = connection
        self.audit_log = audit_log
        self.pid = pid
        self.evidence = {"kind": "SEPARATE_CPU_TRACE_WORKER", "pid": pid,
                         "address_space_bytes": LOADER_AS, "per_trace_timeout_seconds": 10,
                         "per_trace_cpu_seconds": 10, "worker_wall_seconds": 110, "records": []}

    def __call__(self, argv, *, env):
        inputs.contract.require(len(self.evidence["records"]) < MAX_TRACES, "Too many loader requests")
        request = json.dumps({"argv": argv, "env": env}).encode()
        inputs.contract.require(len(request) <= MAX_FRAME, "Oversized loader request")
        self.connection.send_bytes(request)
        if not self.connection.poll(15):
            raise TimeoutError("Loader response unavailable; no retry")
        response = json.loads(self.connection.recv_bytes(MAX_FRAME))
        self.evidence["records"].append(response)
        inputs.contract.require(response["exit"] == 0, f"Loader trace failed: {json.dumps(response)}")
        inputs.contract.require(response["address_space_bytes"] == LOADER_AS, "Wrong loader budget")
        inputs.contract.require(response["argv"] == argv and response["pid"] == self.pid
                                and response["scope"] == "CPU_LOAD_RESOLVE_ONLY_NO_DEVICE_OR_NUMERICS",
                                "Mismatched loader acknowledgement")
        return response["stdout"]

    def check_complete(self, count):
        inputs.contract.require(count == MAX_TRACES and len(self.evidence["records"]) == count,
                                "Incomplete mandatory loader evidence")
        inputs.contract.require(self.audit_log is not None and self.audit_log.is_file(),
                                "Missing durable loader evidence")
        self.evidence["audit_log"] = {"path": str(self.audit_log),
                                     "sha256": inputs.contract.digest(self.audit_log)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--vendor-environment", type=Path, required=True)
    parser.add_argument("--audit-log", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    inputs.contract.require(os.environ.get("SLURM_JOB_ID") and
                            os.environ.get("SLURM_CPUS_PER_TASK") == "8" and
                            not os.environ.get("SLURM_JOB_GPUS"), "Dedicated CPU build allocation required")
    inputs.contract.require(args.audit_log.is_absolute() and
                            args.audit_log.parent.resolve() == args.destination.parent.resolve(),
                            "Audit evidence must belong to new private run")
    inputs.contract.require(not args.audit_log.exists() and not args.audit_log.is_symlink(),
                            "Never overwrite/reuse loader evidence")
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    # Reserve metadata bytes without raising the original512MiB total ceiling.
    inputs.MAX_WRITE -= MAX_AUDIT_BYTES
    context = multiprocessing.get_context("fork")
    client, server = context.Pipe()
    worker = context.Process(target=loader_worker,
                             args=(server, args.vendor_environment.resolve(strict=True), args.audit_log, client))
    worker.start()  # wider parent budget is inherited ONLY by the tracing process
    server.close()
    try:
        inputs.apply_input_limits()  # decoder STILL has1GiB AS/110s/oneCPU
        proof = inputs.verify_remote_decoder()
        print(json.dumps({"decoder_proof_before_real_extraction": proof}), flush=True)
        result = inputs.provision(
            args.source, args.destination, args.lock, inputs.decoder(),
            args.vendor_environment.resolve(strict=True), proof,
            trace_runner=LoaderTrace(client, worker.pid, args.audit_log))
        client.close()
        worker.join(2)
        inputs.contract.require(worker.exitcode == 0, "Loader worker did not finish cleanly")
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        client.close()
        if worker.is_alive():
            worker.terminate()
        worker.join(2)
        if worker.is_alive():
            raise RuntimeError("Loader worker cleanup unverified")


if __name__ == "__main__":
    main()
