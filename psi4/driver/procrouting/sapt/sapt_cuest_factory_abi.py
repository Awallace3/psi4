"""Explicit factory ABI/policy guards for canonical cuEST0.2.50.

No native calls at import. Admission checks mapped capabilities WITHOUT creating
a hidden context/version-query handle. This is not a live basis/plan factory
and cannot yield a production provider or admit numerical basis conventions.
"""
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path

from . import sapt_cuest250_identity as identity
from .sapt_cuda_runtime import library
from .sapt_cuest_dfmo_cuda import _Descriptor, _Workspace

H, U64 = ct.c_void_p, ct.c_uint64
PH, PU64, PD = ct.POINTER(H), ct.POINTER(U64), ct.POINTER(ct.c_double)
PQ, PW = ct.POINTER(_Descriptor), ct.POINTER(_Workspace)

# Canonical headers: context, parameter, shell, basis, pair-list, DF-plan, query.
# Device DFMO pointers retain the existing adapter's separately checked ABI.
FACTORY_ABI = {
    "cuestParametersConfigure": (ct.c_int,H,ct.c_int,H,ct.c_size_t),
    "cuestParametersQuery": (ct.c_int,H,ct.c_int,H,ct.c_size_t),
    "cuestCreate": (H,PH),
    "cuestDestroy": (H,),
    "cuestSetMathMode": (H,ct.c_int),
    "cuestQuery": (H,ct.c_int,H,ct.c_int,H,ct.c_size_t),
    "cuestAOShellCreate": (H,ct.c_int32,U64,U64,PD,PD,H,PH),
    "cuestAOShellDestroy": (H,),
    "cuestAOBasisCreateWorkspaceQuery": (H,U64,PU64,PH,H,PQ,PQ,PH),
    "cuestAOBasisCreate": (H,U64,PU64,PH,H,PW,PW,PH),
    "cuestAOBasisDestroy": (H,),
    "cuestAOPairListCreateWorkspaceQuery": (H,H,U64,PD,ct.c_double,H,PQ,PQ,PH),
    "cuestAOPairListCreate": (H,H,U64,PD,ct.c_double,H,PW,PW,PH),
    "cuestAOPairListDestroy": (H,),
    "cuestDFIntPlanCreateWorkspaceQuery": (H,H,H,H,H,PQ,PQ,PH),
    "cuestDFIntPlanCreate": (H,H,H,H,H,PW,PW,PH),
    "cuestDFIntPlanDestroy": (H,),
}

DF_PLAN_FIELDS = (
    (0, "fitting_cutoff", ct.c_double, 1e-12),
    (1, "relative_conditioning", ct.c_int32, 1),
    (2, "exchange_fraction", ct.c_double, 1.0),
    (3, "fitting_algorithm", ct.c_int, 1),  # MATRIXPOWER, never default QR0.
    (4, "lrc_exchange_fraction", ct.c_double, 0.0),
    (5, "lrc_omega", ct.c_double, 0.0),
    (6, "three_index_integral_direct", ct.c_int32, 0),
)


def configure_factory_abi(native):
    if (ct.sizeof(H) != 8 or ct.sizeof(ct.c_size_t) != 8
            or ct.sizeof(ct.c_int) != 4 or ct.sizeof(_Descriptor) != 16
            or ct.sizeof(_Workspace) != 32):
        raise RuntimeError("Unverified native factory enum/pointer/workspace ABI")
    identity.configure_abi(native)
    try:
        for name, args in FACTORY_ABI.items():
            function = getattr(native,name)
            function.argtypes, function.restype = list(args), ct.c_int
    except AttributeError as error:
        raise RuntimeError("Canonical0.2.50 native factory capability missing") from error
    return tuple(FACTORY_ABI)


def audit_factory_maps(native, proof, manifest, *, text=None):
    """Admit every constructor/configuration/cleanup symbol before first call."""
    names = configure_factory_abi(native)
    text = Path("/proc/self/maps").read_text() if text is None else text
    result = identity.audit_maps(native,proof,manifest,text=text)
    rows = identity.mapped_rows(text)
    for name in names:
        address = ct.cast(getattr(native,name),H).value
        owners = {path for lo,hi,path,_,_ in rows if lo <= address < hi}
        identity.require(owners == {proof["native_path"]},
                         f"cuEST factory symbol outside canonical0.2.50: {name}")
    return {**result, "factory_symbols":names}


def _handle(value):
    if isinstance(value,bool) or not isinstance(value,int) or not 0 < value < 2**64:
        raise ValueError("Explicit nonnull live native handle/address required")
    return H(value)


class _FactoryAPI:
    """Already admitted API; object/parameter lifetimes remain owner obligations."""
    def __init__(self, session, native, proof, manifest, maps):
        self.session, self.native, self.proof = session, native, proof
        self.manifest = json.loads(json.dumps(manifest))
        self.maps, self.pid, self.failed = maps, os.getpid(), False

    def check(self):
        if self.failed or self.pid != os.getpid():
            self.failed = True
            raise RuntimeError("Invalidated/foreign-process factory API; no computation retry")
        try:
            self.session.check()
        except BaseException:
            self.failed = True
            raise

    def status(self, label, code):
        if code != 0:
            self.failed = True
            raise RuntimeError(f"cuEST factory {label} failed: {code}")

    def _roundtrips(self, kind, parameters, fields):
        pointer = _handle(parameters)
        self.check()
        result = {}
        try:
            for attribute,name,ctype,expected in fields:
                value, observed = ctype(expected), ctype()
                self.status("parameters configure", self.native.cuestParametersConfigure(
                    kind,pointer,attribute,ct.byref(value),ct.sizeof(value)))
                self.status("parameters query", self.native.cuestParametersQuery(
                    kind,pointer,attribute,ct.byref(observed),ct.sizeof(observed)))
                if observed.value != expected:
                    raise RuntimeError(f"cuEST factory parameter readback differs: {name}")
                result[name] = observed.value
        except BaseException:
            self.failed = True
            raise
        return result

    def configure_context_parameters(self, parameters, *, stream, jit_compile_threads=1):
        """Borrow only the explicit stream; cuEST will own its BLAS/solver.

        Pointer attributes take the address OF a pointer value and pointer-size
        bytes. The owning constructor must retain that stream past cuestDestroy.
        """
        _handle(stream)
        if (isinstance(jit_compile_threads,bool) or not isinstance(jit_compile_threads,int)
                or not 1 <= jit_compile_threads <= 256):
            raise ValueError("JIT compile threads must be an explicit integer1..256")
        return self._roundtrips(0,parameters,(
            (0,"cuda_stream",H,stream),
            (1,"cublas",H,None),
            (2,"cusolver",H,None),
            (8,"jit_compile_threads",ct.c_int32,jit_compile_threads),
        ))

    def configure_df_plan_parameters(self, parameters):
        """Fixed initial policy; readback proves parameters, NOT the live plan.

        Metric/rank/eigenvalue behavior and native factor layout still require
        numerical admission. Direct-three-index1 is intentionally not enabled.
        """
        return self._roundtrips(5,parameters,DF_PLAN_FIELDS)

    def set_owned_context_fp64(self, handle, *, handle_pid):
        """Configure an explicitly owned live context, never an admission probe."""
        pointer = _handle(handle)
        self.check()
        if isinstance(handle_pid,bool) or not isinstance(handle_pid,int) or handle_pid != self.pid:
            raise ValueError("An owned same-process context is required")
        try:
            self.status("set native FP64",self.native.cuestSetMathMode(pointer,1))
            return identity.native_math_mode(self.native,handle)
        except BaseException:
            self.failed = True
            raise

    def assert_identity(self):
        self.check()
        try:
            self.maps = audit_factory_maps(self.native,self.proof,self.manifest)
        except BaseException:
            self.failed = True
            raise
        return self.maps


def admit_factory_api(*, session, sdk_root, archive, cuda_manifest, cuda_manifest_sha256):
    """File/map capability admission ONLY, using an already explicit CUDA session.

    No context, parameter, shell, basis or plan is constructed here, nor is any
    native version getter invoked without an actual owned live context.
    """
    canonical = json.dumps(cuda_manifest,sort_keys=True,separators=(",",":"),
                           allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != cuda_manifest_sha256:
        raise ValueError("Factory CUDA closure differs from frozen receipt hash")
    manifest = json.loads(canonical)
    session.check()
    proof = identity.verify_files(sdk_root,archive)
    native, _ = library(proof["native_path"],identity.NATIVE_SHA256)
    maps = audit_factory_maps(native,proof,manifest)
    if maps["cuda_mapped_sha256"].get(session.blas_path) != session.blas_sha256:
        raise RuntimeError("Factory and CUDA session use different cuBLAS closure")
    return _FactoryAPI(session,native,proof,manifest,maps)
