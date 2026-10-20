"""Untimed native diagnostics. Never enable a logger around measured work."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import uuid

if not __debug__:
    raise RuntimeError("Dispersion validation requires assertions; optimized Python is forbidden")


class NativeLogger:
    def __init__(self):
        self.library = ctypes.CDLL("libcublas.so.13")
        self.configure = self.library.cublasLoggerConfigure
        self.configure.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
        self.configure.restype = ctypes.c_int
        self.disable()

    def disable(self):
        status = self.configure(0, 0, 0, None)
        if status != 0:
            raise RuntimeError(f"Could not disable native logger: {status}")

    def capture(self, path, execute, synchronize):
        path = Path(path).resolve()
        assert not path.exists(), "Never append diagnostic logs"
        status = self.configure(1, 0, 0, os.fsencode(path))
        if status != 0:
            raise RuntimeError(f"Could not enable untimed native logger: {status}")
        try:
            execute()
            synchronize()
        finally:
            self.disable()
        text = path.read_text() if path.exists() else ""
        calls = len(re.findall(r"\bcublasDgemm(?:_v2)?(?:_64)?\b", text))
        return {"path": str(path), "dgemm_mentions": calls,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "scope": "untimed native logger; mention count is not an exact kernel count"}


def cuda_identity():
    runtime = ctypes.CDLL("libcudart.so.13")
    count, device, version = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    for name, output in (("cudaGetDeviceCount", count), ("cudaGetDevice", device),
                         ("cudaRuntimeGetVersion", version)):
        function = getattr(runtime, name)
        function.argtypes = [ctypes.POINTER(ctypes.c_int)]
        function.restype = ctypes.c_int
        status = function(ctypes.byref(output))
        if status != 0:
            raise RuntimeError(f"{name} failed: {status}")
    assert count.value == 1 and device.value == 0 and version.value == 13030
    raw_uuid = (ctypes.c_ubyte * 16)()
    function = runtime.cudaDeviceGetUuid
    function.argtypes = [ctypes.c_void_p, ctypes.c_int]
    function.restype = ctypes.c_int
    status = function(ctypes.byref(raw_uuid), device.value)
    if status != 0:
        raise RuntimeError(f"cudaDeviceGetUuid failed: {status}")
    identity = "GPU-" + str(uuid.UUID(bytes=bytes(raw_uuid)))
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"], text=True)
    assert any(row.split(",")[0].strip() == identity and "H200" in row
               for row in hardware.splitlines()), hardware
    return {"uuid": identity, "runtime_version": version.value, "logical_device": device.value}


def check_context(identity):
    text = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True)
    rows = [row.split(",") for row in text.splitlines() if "," in row]
    selected = [row for row in rows if row[1].strip() == identity["uuid"]]
    assert selected and all(row[0].strip() == str(os.getpid()) for row in selected), text
    return selected


def probe_dimensions(policy):
    """Only scalar policy metadata enters a fresh, strictly capped probe."""
    nr, ns = policy["compute_block_extents"]
    assert all(int(value) > 0 for value in
               (nr, ns, policy["occupied_A"], policy["occupied_B"], policy["naux"]))
    return {"nr": max(1, min(4, int(nr))), "ns": max(1, min(4, int(ns))),
            "na": max(1, min(8, int(policy["occupied_A"]))),
            "nb": max(1, min(8, int(policy["occupied_B"]))),
            "nq": max(1, min(64, int(policy["naux"])))}


def synthetic_policy_probe(logger, arm, out, ein, gpu, policy):
    """Fresh bounded SAME-kernel probe, NOT a trace/proof of timed dispatch.

    No timed arrays, adapters, graphs, DF scratch, or physical cache are inputs.
    Natural GPU placement stays enabled for the GPU policy; a small probe can
    legitimately stay on CPU. The separate positive control validates logging.
    """
    import numpy as np

    shape = probe_dimensions(policy)
    nq, nr, ns, na, nb = (shape[key] for key in ("nq", "nr", "ns", "na", "nb"))
    factors = [ein.create_zero_tensor("fresh-probe-factor", [2 * nq + 2, width])
               for width in (nr * na, ns * nb, ns * na, nr * nb)]
    outputs = [ein.create_zero_tensor("fresh-probe-output", list(dimensions)) for dimensions in
               ((nr * na, ns * nb), (nr * na, ns * nb), (ns * na, nr * nb))]
    rng = np.random.default_rng(19031)
    for factor in factors:
        np.asarray(factor)[...] = rng.normal(size=factor.shape) * 0.01
    adapter = gpu.BlockContractions(device="cpu" if arm == "cpu" else "cuda")
    observed = logger.capture(out / "synthetic-policy-cublas.log",
                              lambda: adapter.products(*factors, nq, *outputs),
                              ein.gpu.device_synchronize)
    graphs = [json.loads(graph.to_json()) for graph, _, _, _ in adapter._graphs.values()]
    targets = [node["target"] for graph in graphs for node in graph["nodes"] if node["kind"] == "Einsum"]
    assert len(targets) == 3
    if arm == "cpu":
        assert set(targets) == {"CPU"}, "CPU arm left GPU dispatch enabled"
    af, fa, bc, cb = [np.asarray(factor) for factor in factors]
    deltas = []
    for actual, expected in zip(outputs, (af[:nq].T @ fa[nq:2 * nq], af.T @ fa, bc.T @ cb)):
        delta = float(np.max(np.abs(np.asarray(actual) - expected)))
        deltas.append(delta)
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-11, atol=1e-12)
    if arm == "cpu" or "GPU" not in targets:
        assert observed["dgemm_mentions"] == 0
    else:
        assert observed["dgemm_mentions"] > 0

    # Naturally eligible shape already covered by the signed/device gate.
    # Separate bounded positive control, not a representative physical block.
    # Host tensor storage is ~355MiB plus native runtime/workspace overhead.
    positive = gpu.BlockContractions(device="cuda")
    nq, nr, na = 1024, 128, 15
    factors = [ein.create_zero_tensor("untimed-positive-factor", [2 * nq + 2, nr * na])
               for _ in range(4)]
    outputs = [ein.create_zero_tensor("untimed-positive-output", [nr * na, nr * na])
               for _ in range(3)]
    for index, factor in enumerate(factors):
        np.asarray(factor)[...] = (index + 1) * 0.001
    control = logger.capture(out / "untimed-positive-cublas.log",
                             lambda: positive.products(*factors, nq, *outputs),
                             ein.gpu.device_synchronize)
    assert control["dgemm_mentions"] > 0, "Native logger/positive CUDA control unproven"
    return {"status": "PASS", "policy_device": "cpu" if arm == "cpu" else "cuda",
            "source_policy_metadata_only": policy, "probe_shape": shape, "probe_max_abs_deltas": deltas,
            "targets": targets, "probe": observed, "positive_control": control,
            "fresh_synthetic_state": True, "reused_timed_graphs_or_scratch": False,
            "timed_work_was_traced": False, "actual_timed_gpu_dispatch_verified": False,
            "all_cpu_fallback": "GPU" not in targets}
