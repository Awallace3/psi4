"""Caller-owned CuPy buffers for the existing cuEST DFMO Python API.

The caller supplies an independently audited full-Coulomb integral plan and
keeps its basis/plan/persistent-workspace owners alive. This module does NOT
guess basis conversion, metric normalization, auxiliary ordering or strides.
See devtools/disp0/cupy_backend.md for the still-required runtime gates.
"""
import ctypes
import time

import numpy as np

from .sapt_dispersion_cupy import require_cupy


class _Descriptor(ctypes.Structure):
    _fields_ = [("host_bytes", ctypes.c_size_t), ("device_bytes", ctypes.c_size_t)]


class _Workspace(ctypes.Structure):
    _fields_ = [("host", ctypes.c_void_p), ("host_bytes", ctypes.c_size_t),
                ("device", ctypes.c_void_p), ("device_bytes", ctypes.c_size_t)]


class CuESTDFMOProvider:
    """Borrow a plan; own all coefficient/output/temporary buffers in Python.

    output_order must be established against the exact SDK, not inferred from
    logical [naux,nleft,nright] shape. No native plan is destroyed here.
    native_persistent_bytes must include ALL retained native AO/RI basis,
    pair-list, plan and workspace allocations. This is a live-buffer estimate,
    NOT a whole-process peak-VRAM bound; library/context overhead needs tracing.
    """
    def __init__(self, *, handle, plan, primary_basis, auxiliary_basis, keepalive,
                 output_order, memory_limit_bytes, native_persistent_bytes, operator, metric_power,
                 variable_bytes=256 * 1024**2, expected_cudart=13030):
        cp = require_cupy()
        try:
            import cuest.bindings as ce
        except ImportError as exc:
            raise RuntimeError("CuPy DFMO needs matching optional cuest.bindings/native SDK") from exc
        if output_order not in ("C", "F"):
            raise ValueError("Exact SDK output order must be supplied explicitly")
        if operator != "coulomb" or metric_power != -0.5:
            raise ValueError("Canonical SAPT0 requires the full-Coulomb metric-transformed factor contract")
        for name, value in (("memory_limit_bytes", memory_limit_bytes),
                            ("native_persistent_bytes", native_persistent_bytes), ("variable_bytes", variable_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if memory_limit_bytes <= 0 or not keepalive:
            raise ValueError("A positive memory bound and retained native plan/workspace owners are required")
        if ctypes.sizeof(ctypes.c_size_t) != 8 or ctypes.sizeof(ctypes.c_void_p) != 8:
            raise RuntimeError("DFMO workspace ABI requires verified 64-bit size_t/pointers")
        if cp.cuda.runtime.runtimeGetVersion() != expected_cudart:
            raise RuntimeError("CuPy CUDA runtime differs from the explicitly selected SDK ABI")
        self.cp, self.ce = cp, ce
        self.handle, self.plan = handle, plan
        self.primary_basis, self.auxiliary_basis = primary_basis, auxiliary_basis
        self.keepalive = tuple(keepalive)
        self.nao, self.naux = int(primary_basis.nbf()), int(auxiliary_basis.nbf())
        if min(self.nao, self.naux) < 1:
            raise ValueError("Nonempty AO/RI bases required")
        self.output_order, self.device = output_order, cp.cuda.runtime.getDevice()
        self.memory_limit = memory_limit_bytes
        self.native_bytes, self.variable_bytes = native_persistent_bytes, variable_bytes
        self.calls = self.input_bytes = self.output_bytes = 0
        self.elapsed = 0.0
        self.closed = False
        self.failed = False

    def _check_device(self, *, allow_failed=False):
        if self.closed or (self.failed and not allow_failed) or self.cp.cuda.runtime.getDevice() != self.device:
            raise RuntimeError("Closed/invalidated DFMO provider or changed CUDA device; no retry")

    def validate_basis(self, primary, auxiliary):
        self._check_device()
        if self.primary_basis is not primary or self.auxiliary_basis is not auxiliary:
            raise ValueError("Plan factory must bind the exact caller AO/RI basis objects")

    def check_memory(self, requested):
        self._check_device()
        used = int(self.cp.get_default_memory_pool().used_bytes())
        reserved = int(self.cp.get_default_memory_pool().total_bytes())
        free, _ = self.cp.cuda.runtime.memGetInfo()
        # Cached pool blocks can be reused without acquiring new device memory.
        # Fragmentation may still cause the real allocator to fail; never retry.
        reusable = max(0, reserved - used)
        if (requested < 0 or used + self.native_bytes + requested > self.memory_limit
                or requested > free + reusable):
            raise MemoryError("CuPy/cuEST live-buffer estimate exceeded; shrink explicit block/workspace sizes")

    def _status(self, label, code):
        if code != self.ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError(f"cuEST {label} failed: {code}")

    def compute(self, pairs):
        """Host AO coefficient pairs -> owned device [Q,left,right] views.

        Only coefficient slabs are uploaded. DF factors are never downloaded.
        Device-wide synchronization is deliberately conservative at both native
        boundaries until an exact cuEST stream/event contract is established.
        Its cost is included in producer and whole-call timing.
        """
        self._check_device()
        cp, ce = self.cp, self.ce
        if not pairs or len(pairs) > 10:
            raise ValueError("DFMO expects one to ten independent coefficient pairs")
        names, nl, nr = [], [], []
        left, right = [], []
        for name, (cl, cr) in pairs.items():
            for coeff in (cl, cr):
                if (coeff.ndim != 2 or coeff.shape[0] != self.nao or coeff.shape[1] < 1
                        or coeff.dtype != np.float64 or not np.isfinite(coeff).all()):
                    raise ValueError(f"Invalid FP64 AO coefficient slab {name}")
            names.append(name)
            nl.append(cl.shape[1])
            nr.append(cr.shape[1])
            left.append(cl)
            right.append(cr)
        coeff_bytes = 8 * self.nao * (sum(nl) + sum(nr))
        output_elements = self.naux * sum(l * r for l, r in zip(nl, nr))
        # Includes intermediate per-pair uploads and their concatenated copies.
        self.check_memory(2 * coeff_bytes + 8 * output_elements)
        started = time.perf_counter()
        # NVIDIA's sample coefficients are concatenated [norb,nao] C-order.
        cl = cp.concatenate([cp.asarray(x.T, order="C") for x in left], axis=0)
        cr = cp.concatenate([cp.asarray(x.T, order="C") for x in right], axis=0)
        output = cp.empty(output_elements, dtype=cp.float64)
        # Complete coefficient uploads before a potentially separate native stream.
        params = ce.cuestDFMOIntegralsComputeParameters()
        kind = ce.CuestParametersType.CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS
        try:
            cp.cuda.runtime.deviceSynchronize()
            self._status("parameters create", ce.cuestParametersCreate(
                parametersType=kind, outParameters=params))
        except BaseException:
            self.failed = True
            raise
        primary = None
        try:
            variable = _Descriptor(0, self.variable_bytes)
            temporary = _Descriptor()
            kwargs = dict(handle=self.handle, plan=self.plan, parameters=params,
                          variableBufferSize=ctypes.addressof(variable),
                          numCoefficientMatrices=len(names), numLeftOrbitals=nl,
                          numRightOrbitals=nr, leftCoefficientMatrices=ce.Pointer(cl.data.ptr),
                          rightCoefficientMatrices=ce.Pointer(cr.data.ptr),
                          outTensors=ce.Pointer(output.data.ptr))
            self._status("workspace query", ce.cuestDFMOIntegralsComputeWorkspaceQuery(
                temporaryWorkspaceDescriptor=ctypes.addressof(temporary), **kwargs))
            self.check_memory(int(temporary.device_bytes))
            if temporary.host_bytes > 256 * 1024**2:
                raise MemoryError("DFMO host temporary workspace exceeds the explicit256MiB cap")
            host = np.empty(temporary.host_bytes, dtype=np.uint8)
            device = cp.empty(temporary.device_bytes, dtype=cp.uint8)
            workspace = _Workspace(host.ctypes.data if host.size else 0, host.size,
                                   device.data.ptr if device.size else 0, device.size)
            self._status("compute", ce.cuestDFMOIntegralsCompute(
                temporaryWorkspace=ctypes.addressof(workspace), **kwargs))
        except BaseException as error:
            primary = error
            self.failed = True
            raise
        finally:
            # Even on native failure, do not release inputs/workspace while a
            # launched native operation could still be using them.
            cleanup_errors = []
            for cleanup in (
                    lambda: cp.cuda.runtime.deviceSynchronize(),
                    lambda: self._status("parameters destroy", ce.cuestParametersDestroy(
                        parametersType=kind, parameters=params))):
                try:
                    cleanup()
                except BaseException as error:
                    self.failed = True
                    cleanup_errors.append(error)
            if cleanup_errors:
                target = primary if primary is not None else cleanup_errors[0]
                for error in cleanup_errors:
                    if error is not target:
                        target.add_note(f"DFMO cleanup also failed: {type(error).__name__}: {error}")
                if primary is None:
                    raise target
        factors, offset = {}, 0
        for name, l, r in zip(names, nl, nr):
            count = self.naux * l * r
            # CuPy views retain the single owning output allocation.
            factors[name] = output[offset:offset + count].reshape(
                (self.naux, l, r), order=self.output_order)
            offset += count
        self.calls += 1
        self.input_bytes += coeff_bytes
        self.output_bytes += 8 * output_elements
        self.elapsed += time.perf_counter() - started
        return factors

    def close(self):
        """Disable this borrowed-plan adapter; never destroy caller resources."""
        if self.closed:
            return
        try:
            self._check_device(allow_failed=True)
            self.cp.cuda.runtime.deviceSynchronize()
        finally:
            self.closed = True

    def __enter__(self):
        self._check_device()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        except BaseException as error:
            if exc is None:
                raise
            exc.add_note(f"DFMO context cleanup also failed: {type(error).__name__}: {error}")
        return False

    def report(self):
        return {
            "producer": "cuEST DFMO Python bindings; CuPy caller-owned buffers",
            "declared_operator": "coulomb", "declared_metric_power": -0.5,
            "calls": self.calls, "coefficient_h2d_bytes": self.input_bytes,
            "produced_device_factor_bytes": self.output_bytes,
            "producer_including_synchronization_seconds": self.elapsed,
            "device": self.device, "output_order": self.output_order,
            "live_buffer_estimate_limit_bytes": self.memory_limit,
            "declared_all_native_persistent_bytes": self.native_bytes,
            "variable_scratch_limit_bytes": self.variable_bytes,
            "pool_used_bytes": int(self.cp.get_default_memory_pool().used_bytes()),
            "pool_reserved_bytes": int(self.cp.get_default_memory_pool().total_bytes()),
            "basis_metric_layout_numerically_verified": False,
            "note": "Live-buffer estimates are not total/peak VRAM proof; "
                    "caller plan/basis/metric/layout require an independent physical gate",
        }
