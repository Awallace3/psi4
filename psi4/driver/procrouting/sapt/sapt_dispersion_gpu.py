"""Kernel-local Einsums2 graph replay for blocked SAPT0 DF products.

Only eligible contractions are GPU work. Packing, denominators, division and
reductions are CPU work. Placement counts are *plans*, not device-execution
evidence; actual CUDA/cuBLAS traces are required for GPU validation.
"""

import json
import time
from pathlib import Path

import einsums as ein
import einsums.graph as cg
import einsums.sealed as sealed
import numpy as np


def gpu_runtime():
    """Reject CPU/mock/MPS builds instead of silently accepting a GPU request."""
    world = sealed.world()
    device = ein.gpu.device_name()
    maps = Path("/proc/self/maps").read_text()
    libraries = sorted({line.split()[-1] for line in maps.splitlines()
                        if "/" in line and any(name in line for name in
                                               ("libEinsums", "libcudart.so", "libcublas.so"))})
    if (world["version_major"] != 2 or not device or "mock" in device.lower()
            or "libcudart.so" not in maps or "libcublas.so" not in maps):
        raise RuntimeError("SAPT0 GPU contractions require a coherent CUDA-enabled Einsums2 runtime "
                           "and a usable CUDA device (CPU/mock/MPS runtimes are not supported)")
    mapped = sealed.mapped_einsums_libraries()
    if len(mapped) != 1:
        raise RuntimeError(f"Expected one coherent Einsums2 native library, found {mapped}")
    return {"world": world, "device": device, "libraries": libraries}


class BlockContractions:
    """Three packed FP64 DF products, cached per edge shape within one call.

    ``device="cpu"`` is the same default replay seam with GPU placement locally
    disabled, for differential tests. It is never labelled GPU execution.
    Graphs keep their owning inputs and outputs alive. Repacking on each call
    plus default replay's validity tracking avoids stale shadow reuse.
    """

    def __init__(self, *, device="cuda"):
        if device not in ("cuda", "cpu"):
            raise ValueError("BlockContractions device must be 'cuda' or 'cpu'")
        self.device = device
        self.runtime = gpu_runtime() if device == "cuda" else None
        self._graphs = {}
        self._packing = self._planning = self._replay = 0.0
        self._packed_bytes = 0
        self._calls = self._planned_gpu = self._planned_cpu = 0

    def products(self, afar, fabs, bcas, bcbr, nq, v, x, w):
        """V=Aar.Abs, X=AFar.FAbs, W=BCas.BCbr, all in packed factor layout."""
        if nq < 1 or any(a.shape[0] != 2 * nq + 2 for a in (afar, fabs, bcas, bcbr)):
            raise ValueError("Dispersion packed factors must have 2*nQ+2 rows")
        expected = [(afar.shape[1], fabs.shape[1]), (afar.shape[1], fabs.shape[1]),
                    (bcas.shape[1], bcbr.shape[1])]
        if any(tuple(a.shape) != shape for a, shape in zip((v, x, w), expected)):
            raise ValueError("Dispersion output shape does not match packed factors")
        key = (int(nq), *(tuple(a.shape) for a in (afar, fabs, bcas, bcbr)))
        if key not in self._graphs:
            t0 = time.perf_counter()
            shapes = [afar.shape, fabs.shape, bcas.shape, bcbr.shape,
                      (nq, afar.shape[1]), (nq, fabs.shape[1])]
            packs = [ein.create_zero_tensor(name, list(shape)) for name, shape in
                     zip(("AFar", "FAbs", "BCas", "BCbr", "Aar", "Abs"), shapes)]
            af, fa, bc, cb, a, b = packs
            g = cg.Graph("sapt0-dispersion-products")
            with cg.capture(g):
                ein.einsum("ij <- ki ; kj", v, a, b, c_pf=0.0, ab_pf=1.0)
                ein.einsum("ij <- ki ; kj", x, af, fa, c_pf=0.0, ab_pf=1.0)
                ein.einsum("ij <- ki ; kj", w, bc, cb, c_pf=0.0, ab_pf=1.0)
            pm = cg.default_pass_manager()
            # Pointer-array batching can hide supported rank-2 descriptors in
            # a CPU-only grouped descriptor at the audited E2 revision.
            pm.disable("GEMMBatching")
            if self.device == "cpu":
                pm.disable("GPUPlacement")
            g.apply(pm)
            plan = json.loads(g.to_json())
            targets = [node["target"] for node in plan["nodes"] if node["kind"] == "Einsum"]
            if len(targets) != 3:
                raise RuntimeError(f"Unexpected dispersion contraction graph: {plan}")
            self._graphs[key] = (g, packs, (v, x, w), targets)
            self._planning += time.perf_counter() - t0
        g, packs, outputs, targets = self._graphs[key]
        if any(a is not b for a, b in zip(outputs, (v, x, w))):
            raise RuntimeError("Dispersion replay output ownership changed for an existing block shape")
        t0 = time.perf_counter()
        sources = [afar, fabs, bcas, bcbr]
        for pack, source in zip(packs[:4], sources):
            np.asarray(pack)[...] = np.asarray(source)
        np.asarray(packs[4])[...] = np.asarray(afar)[:nq, :]
        np.asarray(packs[5])[...] = np.asarray(fabs)[nq:2 * nq, :]
        self._packing += time.perf_counter() - t0
        self._packed_bytes += sum(np.asarray(a).nbytes for a in packs)
        t0 = time.perf_counter()
        # No custom executor: default replay owns device-shadow validity,
        # transfers, fallback synchronization and final D2H flushing.
        g.execute()
        self._replay += time.perf_counter() - t0
        self._calls += 1
        self._planned_gpu += targets.count("GPU")
        self._planned_cpu += targets.count("CPU")

    def report(self, total_seconds):
        return {
            "scope": "GPU-eligible contractions with CPU elementwise/reductions"
                     if self.device == "cuda" else "CPU graph differential test",
            "runtime": self.runtime,
            "block_replays": self._calls,
            "cached_graph_shapes": len(self._graphs),
            "planned_gpu_contractions": self._planned_gpu,
            "planned_cpu_contractions": self._planned_cpu,
            "actual_gpu_execution_verified": False,
            "packed_host_bytes": self._packed_bytes,
            "packing_seconds": self._packing,
            "planning_seconds": self._planning,
            "replay_including_transfers_seconds": self._replay,
            "end_to_end_seconds": total_seconds,
            "other_host_seconds": max(0.0, total_seconds - self._packing - self._planning - self._replay),
            "note": "Placement is not CUDA execution proof; small/edge blocks may remain CPU. "
                    "Replay includes H2D/D2H and any dispatch fallback.",
        }
