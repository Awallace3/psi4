"""Lazy, explicit CUDA ownership/NVRTC/cuBLAS transport for SAPT0 Python.

Requires cuda-bindings, not CuPy. No context reset, global handles, import-time
CUDA calls, allocation retry or CPU fallback. A session is an explicit lifetime.
"""
from contextlib import contextmanager
import ctypes as ct
import hashlib
import math
from pathlib import Path

import numpy as np

from .sapt_dispersion_cuda_kernels import CUDA_SOURCE


def library(path, sha256):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    if not path.is_absolute() or digest.hexdigest() != sha256:
        raise RuntimeError("Native CUDA library identity does not match explicit admission")
    return ct.CDLL(str(path)), str(path)


def cleanup(primary, actions):
    """Attempt all cleanup and preserve the error from this call, not its caller."""
    errors = []
    for action in actions:
        try:
            action()
        except BaseException as error:
            errors.append(error)
    if errors:
        target = primary if primary is not None else errors[0]
        for error in errors:
            if error is not target:
                target.add_note(f"CUDA cleanup also failed: {type(error).__name__}: {error}")
        if primary is None:
            raise target


class DeviceArray:
    """Contiguous FP64/byte view retaining one explicitly owned allocation."""
    def __init__(self, allocation, shape, *, offset=0, order="C", dtype=np.float64):
        self.allocation, self.shape = allocation, tuple(map(int, shape))
        self.offset, self.order, self.dtype = int(offset), order, np.dtype(dtype)
        self.size = math.prod(self.shape)
        self.nbytes = self.size * self.dtype.itemsize
        if (any(x < 1 for x in self.shape) or order not in ("C", "F")
                or offset < 0 or offset % self.dtype.itemsize
                or offset + self.nbytes > allocation.nbytes):
            raise ValueError("Invalid device view extent/layout")

    @property
    def pointer(self):
        if self.allocation.freed:
            raise RuntimeError("Device allocation lifetime has ended")
        self.allocation.session.check()
        return self.allocation.pointer + self.offset

    def view(self, shape, *, offset=0, order="C"):
        if isinstance(offset,bool) or not isinstance(offset,int) or offset<0:
            raise ValueError("Device subview offset must be nonnegative")
        result = DeviceArray(self.allocation, shape, offset=self.offset+offset,
                             order=order, dtype=self.dtype)
        if offset + result.nbytes > self.nbytes:
            raise ValueError("Device view exceeds parent")
        return result


class _Allocation:
    def __init__(self, session, pointer, nbytes):
        self.session, self.pointer, self.nbytes = session, int(pointer), nbytes
        self.freed = False

    def free(self):
        if not self.freed:
            try:
                self.session.rt(self.session.runtime.cudaFree(self.pointer))
            except BaseException:
                self.session.failed = True
                raise
            self.freed = True
            self.session.live_bytes -= self.nbytes


class CUDASession:
    """One existing CUDA runtime context, one owned stream/BLAS handle/module.

    The caller supplies exact cuBLAS path/hash. CUDA runtime/NVRTC13.3 checks
    are ABI/preparation evidence only, not numerical/device-execution proof.
    """
    def __init__(self, *, cublas_library, cublas_sha256, memory_limit_bytes,
                 native_persistent_bytes, expected_cudart=13030):
        try:
            from cuda.bindings import runtime, driver, nvrtc
        except ImportError as error:
            raise RuntimeError("Optional cuda-bindings is required; CuPy is not used") from error
        if (isinstance(memory_limit_bytes, bool) or not isinstance(memory_limit_bytes, int)
                or memory_limit_bytes <= 0 or not isinstance(native_persistent_bytes, int)
                or native_persistent_bytes < 0):
            raise ValueError("Explicit live-buffer/native allocation budgets required")
        self.runtime, self.driver, self.nvrtc = runtime, driver, nvrtc
        self.allocations = []
        self.native_owners = []
        self.scope_depth = 0
        self.live_bytes = self.peak_live_bytes = 0
        self.limit, self.native_bytes = memory_limit_bytes, native_persistent_bytes
        self.failed = self.closed = False
        self.cleanup_complete = False
        self.stream = self.handle = self.module = None
        self.functions = {}
        self.blas, self.blas_path = library(cublas_library, cublas_sha256)
        self.blas_sha256 = cublas_sha256
        self.device = self.rt(runtime.cudaGetDevice())
        if self.rt(runtime.cudaGetDeviceCount()) < 1 or self.rt(runtime.cudaRuntimeGetVersion()) != expected_cudart:
            raise RuntimeError("An actual NVIDIA device and the pinned CUDA runtime are required")
        try:
            self.stream = self.rt(runtime.cudaStreamCreate())
            # Runtime stream creation establishes its primary context. Never
            # create, switch or release another driver's context.
            major = self.rt(runtime.cudaDeviceGetAttribute(
                runtime.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, self.device))
            minor = self.rt(runtime.cudaDeviceGetAttribute(
                runtime.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, self.device))
            self.arch = f"compute_{major}{minor}"
            self._setup_blas()
            self._compile()
        except BaseException as error:
            self.failed = True
            cleanup(error, [self.close])
            raise

    @staticmethod
    def rt(result):
        if int(result[0]) != 0:
            raise RuntimeError(f"CUDA API failed: {result[0]}")
        values = result[1:]
        return values[0] if len(values) == 1 else values

    def check(self):
        if self.closed or self.failed or self.rt(self.runtime.cudaGetDevice()) != self.device:
            raise RuntimeError("Closed/invalidated CUDA session or changed device; no retry")

    def check_memory(self, requested):
        self.check()
        free, _ = self.rt(self.runtime.cudaMemGetInfo())
        if (requested < 0 or self.live_bytes + self.native_bytes + requested > self.limit
                or requested > free):
            raise MemoryError("Explicit CUDA live-buffer estimate exceeded")

    def empty(self, shape, *, dtype=np.float64, zero=False):
        size = math.prod(shape) * np.dtype(dtype).itemsize
        self.check_memory(size)
        if size < 1:
            raise ValueError("Empty device allocations are not supported")
        try:
            allocation = _Allocation(self, self.rt(self.runtime.cudaMalloc(size)), size)
            self.allocations.append(allocation)
            self.live_bytes += size
            self.peak_live_bytes = max(self.peak_live_bytes, self.live_bytes)
            result = DeviceArray(allocation, shape, dtype=dtype)
            if zero:
                self.rt(self.runtime.cudaMemsetAsync(result.pointer, 0, size, self.stream))
            return result
        except BaseException:
            self.failed = True
            raise

    def upload(self, array):
        array = np.ascontiguousarray(array)
        if array.dtype != np.float64:
            raise ValueError("Only native FP64 host uploads are accepted")
        result = self.empty(array.shape)
        self.rt(self.runtime.cudaMemcpyAsync(
            result.pointer, array.ctypes.data, array.nbytes,
            self.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))
        # A pageable host temporary must not die before its upload completes.
        self.synchronize()
        return result

    def download(self, array):
        if array.order != "C":
            raise ValueError("Final downloads require explicit C-order layout")
        result = np.empty(array.shape, dtype=array.dtype)
        self.rt(self.runtime.cudaMemcpyAsync(
            result.ctypes.data, array.pointer, result.nbytes,
            self.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
        self.synchronize()
        return result

    def synchronize(self):
        try:
            self.rt(self.runtime.cudaStreamSynchronize(self.stream))
        except BaseException:
            self.failed = True
            raise

    def synchronize_device(self):
        try:
            self.rt(self.runtime.cudaDeviceSynchronize())
        except BaseException:
            self.failed = True
            raise

    @contextmanager
    def scope(self):
        self.check()
        start, primary = len(self.allocations), None
        self.scope_depth = getattr(self,"scope_depth",0) + 1
        try:
            yield
        except BaseException as error:
            primary = error
            self.failed = True
            raise
        finally:
            pending = self.allocations[start:]
            try:
                cleanup(primary, [self.synchronize_device,
                                  *(item.free for item in reversed(pending))])
            finally:
                # Retain failed frees for explicit final cleanup; never lose
                # their ownership or pretend their bytes were released.
                self.allocations[start:] = [item for item in pending if not item.freed]
                self.scope_depth -= 1

    def retain_native_owner(self, owner):
        """Keep native dependents alive before they borrow this session's buffers.

        Persistent native workspaces cannot be constructed in a temporary scope.
        Native owners must complete destruction before the session frees any
        allocations, module, BLAS handle or borrowed stream.
        """
        self.check()
        if getattr(self,"scope_depth",0):
            raise ValueError("Native owners must be created outside temporary CUDA scopes")
        if (not callable(getattr(owner,"close",None))
                or getattr(owner,"cleanup_complete",None) is not False):
            raise ValueError("An unfinished explicit native cleanup owner is required")
        if any(item is owner for item in self.native_owners):
            raise ValueError("Native owner is already retained")
        self.native_owners.append(owner)

    def release_native_owner(self, owner):
        # Cleanup-only retry is permitted after execution is closed/invalidated.
        if getattr(owner,"cleanup_complete",None) is not True:
            raise RuntimeError("Native owner has not completed cleanup")
        for index,item in enumerate(self.native_owners):
            if item is owner:
                del self.native_owners[index]
                return
        raise ValueError("Native owner is not retained by this session")

    def _setup_blas(self):
        specs = {
            "cublasCreate_v2": [ct.POINTER(ct.c_void_p)],
            "cublasDestroy_v2": [ct.c_void_p],
            "cublasSetStream_v2": [ct.c_void_p, ct.c_void_p],
            "cublasSetPointerMode_v2": [ct.c_void_p, ct.c_int],
            "cublasSetMathMode": [ct.c_void_p, ct.c_int],
            "cublasSetAtomicsMode": [ct.c_void_p, ct.c_int],
            "cublasDgemm_v2": [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
                               ct.POINTER(ct.c_double), ct.c_void_p, ct.c_int,
                               ct.c_void_p, ct.c_int, ct.POINTER(ct.c_double), ct.c_void_p, ct.c_int],
        }
        for name, args in specs.items():
            function = getattr(self.blas, name)
            function.argtypes, function.restype = args, ct.c_int
        handle = ct.c_void_p()
        self._blas("cublasCreate_v2", ct.byref(handle))
        self.handle = handle
        self._blas("cublasSetStream_v2", handle, int(self.stream))
        self._blas("cublasSetPointerMode_v2", handle, 0)  # Host scalar parameters.
        self._blas("cublasSetMathMode", handle, 0)       # Default/native FP64.
        self._blas("cublasSetAtomicsMode", handle, 0)    # No atomic algorithm.

    def _blas(self, name, *args):
        status = getattr(self.blas, name)(*args)
        if status != 0:
            self.failed = True
            raise RuntimeError(f"{name} failed: {status}")

    def _compile(self):
        nv, dr = self.nvrtc, self.driver
        major, minor = self.rt(nv.nvrtcVersion())
        if (major, minor) != (13, 3):
            raise RuntimeError("NVRTC differs from explicitly selected CUDA13.3")
        program = self.rt(nv.nvrtcCreateProgram(
            CUDA_SOURCE.encode(), b"sapt0_blocks.cu", 0, None, None))
        primary = None
        try:
            self.options = [b"--std=c++14", b"--fmad=false", f"--gpu-architecture={self.arch}".encode()]
            status = nv.nvrtcCompileProgram(program, len(self.options), self.options)
            if int(status[0]) != 0:
                length = self.rt(nv.nvrtcGetProgramLogSize(program))
                if length > 1024**2:
                    raise RuntimeError("NVRTC error log exceeds bounded capture")
                log = bytearray(length)
                self.rt(nv.nvrtcGetProgramLog(program, log))
                raise RuntimeError("NVRTC compilation failed: " + bytes(log).decode(errors="replace"))
            size = self.rt(nv.nvrtcGetPTXSize(program))
            if not 0 < size <= 16 * 1024**2:
                raise RuntimeError("Invalid/oversized NVRTC PTX")
            ptx = bytearray(size)
            self.rt(nv.nvrtcGetPTX(program, ptx))
            self.ptx_sha256 = hashlib.sha256(ptx).hexdigest()
            self.module = self.rt(dr.cuModuleLoadData(bytes(ptx)))
            for name in ("pack_factors", "reduce_dispersion"):
                self.functions[name] = self.rt(dr.cuModuleGetFunction(self.module, name.encode()))
        except BaseException as error:
            primary = error
            raise
        finally:
            cleanup(primary, [lambda: self.rt(nv.nvrtcDestroyProgram(program))])

    def launch(self, name, grid, threads, arguments):
        self.check()
        # Typed scalar holders MUST outlive cuLaunchKernel's parameter read.
        holders = [ct.c_void_p(value.pointer if value is not None else 0)
                   if isinstance(value, DeviceArray) or value is None
                   else ct.c_int(value) if isinstance(value, bool) else ct.c_longlong(value)
                   for value in arguments]
        parameters = (ct.c_void_p * len(holders))(
            *[ct.cast(ct.pointer(value), ct.c_void_p) for value in holders])
        try:
            self.rt(self.driver.cuLaunchKernel(
                self.functions[name], grid, 1, 1, threads, 1, 1, 0,
                int(self.stream), ct.addressof(parameters), 0))
        except BaseException:
            self.failed = True
            raise

    def gemm_at_b(self, left, right):
        """C-order[K,M]^T @ C-order[K,N], using column-major cuBLAS N×M."""
        self.check()
        if (len(left.shape) != 2 or len(right.shape) != 2
                or left.dtype != np.float64 or right.dtype != np.float64
                or left.allocation.session is not self or right.allocation.session is not self):
            raise ValueError("GEMM needs FP64 arrays owned by this CUDA session")
        k, m = left.shape
        kr, n = right.shape
        if k != kr or max(k, m, n) > 2**31-1 or left.order != "C" or right.order != "C":
            raise ValueError("Unsupported GEMM extent/layout")
        out = self.empty((m, n))
        alpha, beta = ct.c_double(1.0), ct.c_double(0.0)
        self._blas("cublasDgemm_v2", self.handle, 0, 1, n, m, k,
                   ct.byref(alpha), right.pointer, n, left.pointer, m,
                   ct.byref(beta), out.pointer, n)
        return out

    def close(self):
        if self.cleanup_complete:
            return
        # Execution is permanently disabled, even if an explicit cleanup-only
        # retry is later needed. Successfully destroyed resources are cleared.
        self.closed = True
        owners = tuple(getattr(self,"native_owners",()))
        if owners:
            try:
                cleanup(None,[owner.close for owner in reversed(owners)])
                if self.native_owners:
                    raise RuntimeError("Native dependents still retain CUDA session resources")
            except BaseException:
                # Never free persistent native workspaces or destroy borrowed
                # handles/stream after failed native cleanup. Owners remain
                # strongly retained for an explicit cleanup-only retry.
                self.failed = True
                raise
        actions = [self.synchronize_device,
                   *(allocation.free for allocation in reversed(self.allocations))]
        if self.handle is not None:
            def destroy_handle():
                self._blas("cublasDestroy_v2",self.handle)
                self.handle=None
            actions.append(destroy_handle)
        if self.module is not None:
            def destroy_module():
                self.rt(self.driver.cuModuleUnload(self.module))
                self.module=None
            actions.append(destroy_module)
        if self.stream is not None:
            def destroy_stream():
                self.rt(self.runtime.cudaStreamDestroy(self.stream))
                self.stream=None
            actions.append(destroy_stream)
        try:
            cleanup(None, actions)
        except BaseException:
            self.failed=True
            raise
        else:
            self.cleanup_complete=True
        finally:
            self.allocations[:]=[item for item in self.allocations if not item.freed]

    def __enter__(self):
        self.check()
        return self

    def __exit__(self, kind, error, tb):
        cleanup(error, [self.close])
        return False
