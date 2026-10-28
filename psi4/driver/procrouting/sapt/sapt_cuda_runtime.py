"""Lazy, explicit CUDA ownership/NVRTC/cuBLAS transport for SAPT0 Python.

Requires cuda-bindings, not CuPy. No context reset, global GPU handles, import-time
CUDA calls, allocation retry or CPU fallback. A session is an explicit lifetime.
"""
from contextlib import contextmanager
import ctypes as ct
import hashlib
import math
from pathlib import Path

import numpy as np

from .sapt_dispersion_cuda_kernels import CUDA_SOURCE


def _identity_guard(session):
    owner = getattr(session,"identity_owner",None)
    if owner is not None:
        owner.check_live()


def _argument_pointer(session, value):
    """Pure address packing, NOT an identity/context admission.

    Owned native entrypoints validate the complete proof/device again AFTER
    packing and before enqueue. Do not call DeviceArray.pointer recursively
    for every operand in that same operation. The public property and legacy
    adapter retain their existing full checks.
    """
    if getattr(session,"identity_owner",None) is None:
        return value.pointer
    if not isinstance(value,DeviceArray) or value.allocation.session is not session:
        raise ValueError("Native argument must belong to this CUDA session")
    if value.allocation.freed:
        raise RuntimeError("Device allocation lifetime has ended")
    return value.allocation.pointer+value.offset


def library(path, sha256, *, owner=None):
    if owner is not None:
        return owner.load_library(path,sha256)
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
                _identity_guard(self.session)
                self.session.rt(self.session.runtime.cudaFree(self.pointer))
                # Preserve the observed successful free even if subsequent
                # immutable-runtime validation fails. Never free it twice.
                self.freed = True
                self.session.live_bytes -= self.nbytes
                _identity_guard(self.session)
            except BaseException:
                self.session.failed = True
                raise


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
        self.blas_calls = {}
        self.kernel_launches = {}
        from .sapt_cuest_owned_runtime import owner_for_session
        self.identity_owner = owner_for_session()
        options = {"owner":self.identity_owner} if self.identity_owner is not None else {}
        self.blas, self.blas_path = library(cublas_library, cublas_sha256,**options)
        self.blas_sha256 = cublas_sha256
        _identity_guard(self)
        self.device = self.rt(runtime.cudaGetDevice())
        if self.rt(runtime.cudaGetDeviceCount()) < 1 or self.rt(runtime.cudaRuntimeGetVersion()) != expected_cudart:
            raise RuntimeError("An actual NVIDIA device and the pinned CUDA runtime are required")
        try:
            _identity_guard(self)
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
            _identity_guard(self)
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
        if self.closed or self.failed:
            raise RuntimeError("Closed/invalidated CUDA session or changed device; no retry")
        _identity_guard(self)
        if self.rt(self.runtime.cudaGetDevice()) != self.device:
            raise RuntimeError("Closed/invalidated CUDA session or changed device; no retry")
        _identity_guard(self)

    def check_memory(self, requested):
        self.check()
        free, _ = self.rt(self.runtime.cudaMemGetInfo())
        reasons = []
        if requested < 0:
            reasons.append("negative_request")
        if self.live_bytes + self.native_bytes + requested > self.limit:
            reasons.append("declared_budget")
        if requested > free:
            reasons.append("cuda_free")
        if reasons:
            # Capture the rejection snapshot, not the post-cleanup state.
            # Tracked bytes are not total VRAM and the native reserve is not
            # a measured peak. Preserve the admission policy and do not retry.
            raise MemoryError(
                "Explicit CUDA live-buffer estimate exceeded: "
                f"requested_bytes={requested}, explicit_live_bytes={self.live_bytes}, "
                f"native_reserved_bytes={self.native_bytes}, limit_bytes={self.limit}, "
                f"cuda_free_bytes={free}, rejected_by={'+'.join(reasons)}")

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
        return self._upload_array(array)

    def upload_indices(self, array):
        """Explicit int64 METADATA transport, not a numerical upload fallback."""
        array = np.ascontiguousarray(array)
        if (array.dtype != np.int64 or array.ndim != 1 or not array.size
                or np.any(array < 0)):
            raise ValueError("Nonempty nonnegative native int64 metadata indices required")
        return self._upload_array(array)

    def _upload_array(self, array):
        result = self.empty(array.shape,dtype=array.dtype)
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
            _identity_guard(self)
            self.rt(self.runtime.cudaStreamSynchronize(self.stream))
            _identity_guard(self)
        except BaseException:
            self.failed = True
            raise

    def synchronize_device(self):
        try:
            _identity_guard(self)
            self.rt(self.runtime.cudaDeviceSynchronize())
            _identity_guard(self)
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
                def release():
                    # Native compute may still borrow these buffers. Failed
                    # synchronization must retain them for cleanup-only retry.
                    self.synchronize_device()
                    cleanup(None, [item.free for item in reversed(pending)])
                cleanup(primary, [release])
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
            "cublasDgemmStridedBatched": [
                ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
                ct.POINTER(ct.c_double), ct.c_void_p, ct.c_int, ct.c_longlong,
                ct.c_void_p, ct.c_int, ct.c_longlong, ct.POINTER(ct.c_double),
                ct.c_void_p, ct.c_int, ct.c_longlong, ct.c_int],
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
        if (getattr(self,"identity_owner",None) is not None
                and name in ("cublasDgemm_v2","cublasDgemmStridedBatched")):
            # Operands were packed without recursive native admission. Check
            # the full proof AND current device at the actual compute seam.
            self.check()
        else:
            _identity_guard(self)
        status = getattr(self.blas, name)(*args)
        if status != 0:
            self.failed = True
            raise RuntimeError(f"{name} failed: {status}")
        self.blas_calls[name] = self.blas_calls.get(name,0)+1

    def _compile(self):
        nv, dr = self.nvrtc, self.driver
        def native(function, *args):
            _identity_guard(self)
            return function(*args)
        major, minor = self.rt(native(nv.nvrtcVersion))
        if (major, minor) != (13, 3):
            raise RuntimeError("NVRTC differs from explicitly selected CUDA13.3")
        program = self.rt(native(nv.nvrtcCreateProgram,
            CUDA_SOURCE.encode(), b"sapt0_blocks.cu", 0, None, None))
        primary = None
        try:
            self.options = [b"--std=c++14", b"--fmad=false", f"--gpu-architecture={self.arch}".encode()]
            status = native(nv.nvrtcCompileProgram,program,len(self.options),self.options)
            if int(status[0]) != 0:
                length = self.rt(native(nv.nvrtcGetProgramLogSize,program))
                if length > 1024**2:
                    raise RuntimeError("NVRTC error log exceeds bounded capture")
                log = bytearray(length)
                self.rt(native(nv.nvrtcGetProgramLog,program,log))
                raise RuntimeError("NVRTC compilation failed: " + bytes(log).decode(errors="replace"))
            size = self.rt(native(nv.nvrtcGetPTXSize,program))
            if not 0 < size <= 16 * 1024**2:
                raise RuntimeError("Invalid/oversized NVRTC PTX")
            ptx = bytearray(size)
            self.rt(native(nv.nvrtcGetPTX,program,ptx))
            self.ptx_sha256 = hashlib.sha256(ptx).hexdigest()
            self.module = self.rt(native(dr.cuModuleLoadData,bytes(ptx)))
            for name in ("pack_factors", "reduce_dispersion", "matrix_window",
                         "fragment_scale_factor", "apply_fragment_scale",
                         "linear_combination", "reduce_components", "permute_occupied_a",
                         "make_amplitudes", "rotation_error", "reduce_localized_dispersion",
                         "diagonal_factors", "scatter_matrix", "fragment_nuclear", "reduce_values",
                         "reduce_fragment_exchange", "make_single_amplitudes", "reduce_induction_source",
                         "matrix_error", "single_gap_error", "trace_factors",
                         "rhf_response_product", "response_dot", "response_precondition",
                         "response_accumulate_exchange", "response_finish",
                         "response_step", "response_direction", "response_metrics",
                         "inverse_sqrt_weights", "scale_eigenvector_rows", "gather_indexed_matrix",
                         "set_identity", "ibo_objective", "ibo_pair", "ibo_sweep_finish",
                         "ibo_fock_diagonal", "ibo_permutation", "ibo_charges", "ibo_population_error"):
                self.functions[name] = self.rt(native(dr.cuModuleGetFunction,self.module,name.encode()))
        except BaseException as error:
            primary = error
            raise
        finally:
            cleanup(primary, [lambda: self.rt(native(nv.nvrtcDestroyProgram,program))])

    def launch(self, name, grid, threads, arguments):
        self.check()
        owned=getattr(self,"identity_owner",None) is not None
        try:
            if owned:
                arguments=tuple(arguments)
                function=self.functions[name]
                grid,threads,stream=int(grid),int(threads),int(self.stream)
            # Typed scalar holders MUST outlive cuLaunchKernel's parameter read.
            holders = [ct.c_void_p(_argument_pointer(self,value) if value is not None else 0)
                       if isinstance(value, DeviceArray) or value is None
                       else ct.c_int(value) if isinstance(value, bool) else ct.c_longlong(value)
                       for value in arguments]
            parameters = (ct.c_void_p * len(holders))(
                *[ct.cast(ct.pointer(value), ct.c_void_p) for value in holders])
        except BaseException:
            # A converter/iterator may revoke the proof AND throw. Validate
            # before allowing caller/native cleanup to unwind. A healthy
            # proof preserves the original ordinary input exception.
            if owned:
                _identity_guard(self)
            raise
        try:
            if owned:
                # Scalar coercion/generator iteration may execute Python code.
                # Never enqueue with a proof, device, lifetime or address which
                # changed while arguments were being assembled.
                self.check()
                for value,holder in zip(arguments,holders):
                    if isinstance(value,DeviceArray) and holder.value != _argument_pointer(self,value):
                        raise RuntimeError("Device argument address changed during packing")
            self.rt(self.driver.cuLaunchKernel(
                function if owned else self.functions[name], grid, 1, 1, threads, 1, 1, 0,
                stream if owned else int(self.stream), ct.addressof(parameters), 0))
            self.kernel_launches[name] = self.kernel_launches.get(name,0)+1
        except BaseException:
            self.failed = True
            if owned:
                _identity_guard(self)
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
                   ct.byref(alpha), _argument_pointer(self,right), n, _argument_pointer(self,left), m,
                   ct.byref(beta), _argument_pointer(self,out), n)
        return out

    def gemm(self, left, right):
        """C-order[M,K] @ C-order[K,N], native FP64 and no host intermediates."""
        self.check()
        if (len(left.shape) != 2 or len(right.shape) != 2
                or left.dtype != np.float64 or right.dtype != np.float64
                or left.allocation.session is not self or right.allocation.session is not self):
            raise ValueError("GEMM needs FP64 arrays owned by this CUDA session")
        m, k = left.shape
        kr, n = right.shape
        if k != kr or max(k,m,n) > 2**31-1 or left.order != "C" or right.order != "C":
            raise ValueError("Unsupported GEMM extent/layout")
        out = self.empty((m,n))
        alpha, beta = ct.c_double(1.0), ct.c_double(0.0)
        self._blas("cublasDgemm_v2", self.handle, 0, 0, n, m, k,
                   ct.byref(alpha), _argument_pointer(self,right), n, _argument_pointer(self,left), k,
                   ct.byref(beta), _argument_pointer(self,out), n)
        return out

    def gemm_batched(self, left, right, *, transpose_left=False, transpose_right=False,
                     out=None):
        """FP64 C-order batched GEMM, with an optional rank-two broadcast.

        Three-dimensional operands have one contiguous matrix per batch.
        Rank-two operands broadcast with native cuBLAS stride zero. Transposes
        are descriptors, never host/device matrix copies. Output is [B,M,N].
        An optional nonaliasing output view enables bounded stream-ordered
        workspace reuse; beta=0 overwrites every output, including tail batches.
        """
        self.check()
        if type(transpose_left) is not bool or type(transpose_right) is not bool:
            raise ValueError("Batched GEMM transpose descriptors must be booleans")
        for value in (left, right):
            if (len(value.shape) not in (2, 3) or value.dtype != np.float64
                    or value.order != "C" or value.allocation.session is not self):
                raise ValueError("Batched GEMM needs owned C-order FP64 matrices")
            _argument_pointer(self,value)  # Reject ended lifetimes before allocating output.
        lb = left.shape[0] if len(left.shape) == 3 else None
        rb = right.shape[0] if len(right.shape) == 3 else None
        if lb is not None and rb is not None and lb != rb:
            raise ValueError("Batched GEMM batch extents differ")
        batches = lb if lb is not None else rb if rb is not None else 1
        lr, lc = left.shape[-2:]
        rr, rc = right.shape[-2:]
        m, k = (lc, lr) if transpose_left else (lr, lc)
        kr, n = (rc, rr) if transpose_right else (rr, rc)
        if k != kr or min(batches, m, n, k) < 1 or max(batches, m, n, k, lc, rc) > 2**31-1:
            raise ValueError("Unsupported batched GEMM extents")
        ls, rs, cs = (lr*lc if lb is not None else 0), (rr*rc if rb is not None else 0), m*n
        if max(ls, rs, cs)*batches > 2**63-1:
            raise ValueError("Batched GEMM stride exceeds the native ABI")
        shape = (batches, m, n)
        if out is None:
            out = self.empty(shape)
        else:
            if (not isinstance(out, DeviceArray) or out.shape != shape
                    or out.dtype != np.float64 or out.order != "C"
                    or out.allocation.session is not self):
                raise ValueError("Batched GEMM output needs an owned C-order FP64 extent")
            _argument_pointer(self,out)  # Reject ended lifetimes before passing native pointers.
            for value in (left, right):
                if (out.allocation is value.allocation
                        and out.offset < value.offset + value.nbytes
                        and value.offset < out.offset + out.nbytes):
                    raise ValueError("Batched GEMM output cannot overlap an input")
        alpha, beta = ct.c_double(1.0), ct.c_double(0.0)
        # C-order C.T = op(right).T @ op(left).T: reverse operands/dimensions.
        self._blas("cublasDgemmStridedBatched", self.handle,
                   int(transpose_right), int(transpose_left), n, m, k,
                   ct.byref(alpha), _argument_pointer(self,right), rc, rs,
                   _argument_pointer(self,left), lc, ls,
                   ct.byref(beta), _argument_pointer(self,out), n, cs, batches)
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
        # Do not continue to resource destruction after failed synchronization.
        # Execution stays disabled and all resources remain owned for retry.
        self.synchronize_device()
        actions = [allocation.free for allocation in reversed(self.allocations)]
        if self.handle is not None:
            def destroy_handle():
                _identity_guard(self)
                self._blas("cublasDestroy_v2",self.handle)
                self.handle=None
                _identity_guard(self)
            actions.append(destroy_handle)
        if self.module is not None:
            def destroy_module():
                _identity_guard(self)
                self.rt(self.driver.cuModuleUnload(self.module))
                self.module=None
                _identity_guard(self)
            actions.append(destroy_module)
        if self.stream is not None:
            def destroy_stream():
                _identity_guard(self)
                self.rt(self.runtime.cudaStreamDestroy(self.stream))
                self.stream=None
                _identity_guard(self)
            actions.append(destroy_stream)
        try:
            cleanup(None, actions)
            _identity_guard(self)
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
