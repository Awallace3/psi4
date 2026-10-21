"""Canonical0.2.50 DFMO C-ABI adapter; no CuPy or cuest.bindings dependency.

An audited caller factory supplies native handles, exact Psi4 basis identities,
configured plan policy and persistent owners. No basis/plan is guessed here.
"""
import ctypes as ct
import hashlib
import json
import os
import time

import numpy as np

from . import sapt_cuest250_identity as identity
from .sapt_cuda_runtime import cleanup, library


class _Descriptor(ct.Structure):
    _fields_ = [("host_bytes",ct.c_size_t),("device_bytes",ct.c_size_t)]


class _Workspace(ct.Structure):
    _fields_ = [("host",ct.c_size_t),("host_bytes",ct.c_size_t),
                ("device",ct.c_size_t),("device_bytes",ct.c_size_t)]


POLICY = {"operator":"coulomb", "exchange_fraction":1.0, "lrc_exchange_fraction":0.0,
          "lrc_omega":0.0, "metric_power":-0.5, "fitting_algorithm":1,
          "relative_conditioning":1, "fitting_cutoff":1e-12}


class CuESTDFMOProvider:
    def __init__(self, *, session, sdk_root, archive, cuda_manifest, cuda_manifest_sha256,
                 handle, plan, handle_pid,
                 primary_basis, auxiliary_basis, keepalive, plan_policy,
                 output_order, variable_bytes=256*1024**2):
        if (isinstance(handle,bool) or not isinstance(handle,int) or not 0 < handle < 2**64
                or isinstance(plan,bool) or not isinstance(plan,int) or not 0 < plan < 2**64
                or handle_pid != os.getpid()
                or not keepalive or output_order not in ("C","F")):
            raise ValueError("Explicit live same-process owned-plan context/handles/layout required")
        manifest_data = json.dumps(cuda_manifest,sort_keys=True,separators=(",",":")).encode()
        if hashlib.sha256(manifest_data).hexdigest() != cuda_manifest_sha256:
            raise ValueError("CUDA closure manifest differs from the explicit frozen receipt hash")
        if any(plan_policy.get(key) != value for key,value in POLICY.items()):
            raise ValueError("DFMO needs explicit full-Coulomb MATRIXPOWER/relative1/cutoff1e-12 policy")
        if (isinstance(variable_bytes,bool) or not isinstance(variable_bytes,int) or variable_bytes < 0
                or ct.sizeof(ct.c_size_t) != 8 or ct.sizeof(ct.c_void_p) != 8):
            raise ValueError("Invalid variable workspace limit or unverified64-bit ABI")
        self.session = session
        self.proof = identity.verify_files(sdk_root,archive)
        self.proof["cuda_manifest_sha256"] = cuda_manifest_sha256
        self.native, _ = library(self.proof["native_path"],identity.NATIVE_SHA256)
        identity.configure_abi(self.native)
        self.handle, self.plan = handle, plan
        self.primary_basis, self.auxiliary_basis = primary_basis, auxiliary_basis
        self.keepalive = tuple(keepalive)
        self.cuda_manifest = json.loads(manifest_data)
        self.nao, self.naux = int(primary_basis.nbf()), int(auxiliary_basis.nbf())
        if min(self.nao,self.naux) < 1:
            raise ValueError("Nonempty exact AO/RI bases required")
        self.order, self.variable_bytes = output_order, variable_bytes
        self.closed = self.failed = False
        self.calls = self.input_bytes = self.output_bytes = 0
        self.elapsed = 0.0
        self.assert_identity()

    def assert_identity(self):
        self.session.check()
        self.maps = identity.audit_maps(self.native,self.proof,self.cuda_manifest)
        self.version = identity.native_version(self.native,self.handle)
        expected = self.maps["cuda_mapped_sha256"].get(self.session.blas_path)
        if expected != self.session.blas_sha256:
            raise RuntimeError("DFMO and cuBLAS session do not share the admitted CUDA closure")

    def validate_basis(self, primary, auxiliary):
        if self.closed or self.failed:
            raise RuntimeError("Closed/invalidated DFMO provider; no retry")
        if self.primary_basis is not primary or self.auxiliary_basis is not auxiliary:
            raise ValueError("Factory must retain the exact Psi4 AO/RI basis objects")
        self.assert_identity()

    def status(self, label, code):
        if code != 0:
            self.failed = True
            raise RuntimeError(f"cuEST0.2.50 {label} failed: {code}")

    def compute(self, pairs):
        if self.closed or self.failed:
            raise RuntimeError("Closed/invalidated DFMO provider; no retry")
        self.session.check()
        if not pairs or len(pairs) > 10:
            raise ValueError("DFMO needs1..10 independent coefficient pairs")
        names, nl, nr, left, right = [], [], [], [], []
        for name,(cl,cr) in pairs.items():
            for matrix in (cl,cr):
                if (matrix.ndim != 2 or matrix.shape[0] != self.nao or matrix.shape[1] < 1
                        or matrix.dtype != np.float64 or not np.isfinite(matrix).all()):
                    raise ValueError("Invalid native FP64 AO coefficient slab")
            names.append(name); nl.append(cl.shape[1]); nr.append(cr.shape[1])
            left.append(cl.T); right.append(cr.T)
        count = self.naux*sum(l*r for l,r in zip(nl,nr))
        started = time.perf_counter()
        output = self.session.empty((count,))
        with self.session.scope():
            cl = self.session.upload(np.concatenate(left,axis=0))
            cr = self.session.upload(np.concatenate(right,axis=0))
            self.session.synchronize_device()
            parameters = ct.c_void_p()
            # Exact0.2.50 cuest_parameter_types.h: DFMO parameters kind49.
            self.status("parameters create",self.native.cuestParametersCreate(49,ct.byref(parameters)))
            primary = None
            try:
                variable = _Descriptor(0,self.variable_bytes)
                temporary = _Descriptor()
                nl_values, nr_values = (ct.c_uint64*len(nl))(*nl),(ct.c_uint64*len(nr))(*nr)
                args = (ct.c_void_p(self.handle),ct.c_void_p(self.plan),parameters,
                        ct.byref(variable),None,len(names),nl_values,nr_values,
                        ct.c_void_p(cl.pointer),ct.c_void_p(cr.pointer),ct.c_void_p(output.pointer))
                query = (*args[:4],ct.byref(temporary),*args[5:])
                self.status("workspace query",self.native.cuestDFMOIntegralsComputeWorkspaceQuery(*query))
                if temporary.host_bytes > 256*1024**2:
                    raise MemoryError("DFMO host workspace exceeds explicit256MiB cap")
                host = np.empty(temporary.host_bytes,dtype=np.uint8)
                device = (self.session.empty((temporary.device_bytes,),dtype=np.uint8)
                          if temporary.device_bytes else None)
                workspace = _Workspace(host.ctypes.data if host.size else 0,host.size,
                                       device.pointer if device is not None else 0,temporary.device_bytes)
                compute = (*args[:4],ct.byref(workspace),*args[5:])
                self.status("compute",self.native.cuestDFMOIntegralsCompute(*compute))
            except BaseException as error:
                primary = error
                self.failed = True
                raise
            finally:
                try:
                    cleanup(primary,[self.session.synchronize_device,
                                     lambda:self.status("parameters destroy",
                                                        self.native.cuestParametersDestroy(49,parameters))])
                except BaseException:
                    self.failed = True
                    raise
        factors, offset = {}, 0
        for name,l,r in zip(names,nl,nr):
            size = self.naux*l*r
            factors[name] = output.view((self.naux,l,r),offset=8*offset,order=self.order)
            offset += size
        self.calls += 1
        self.input_bytes += 8*self.nao*(sum(nl)+sum(nr))
        self.output_bytes += 8*count
        self.elapsed += time.perf_counter()-started
        return factors

    def report(self):
        # Reaudit after native work to catch late-loaded wrong CUDA libraries.
        self.assert_identity()
        return {"producer":"cuEST0.2.50 C-ABI; Python-owned CUDA buffers", "calls":self.calls,
                "coefficient_h2d_bytes":self.input_bytes,"device_factor_bytes":self.output_bytes,
                "producer_seconds":self.elapsed,"sdk_file_proof":self.proof,
                "native_version":self.version,"loaded_maps":self.maps,"declared_plan_policy":POLICY,
                "basis_metric_rank_eigenvalue_semantics_verified":False,
                "note":"Policy declarations and capabilities are not DFMO numerical/rank proof"}

    def __enter__(self):
        if self.closed or self.failed:
            raise RuntimeError("Closed/invalidated DFMO provider")
        self.assert_identity()
        return self

    def close(self):
        if not self.closed:
            try:
                self.session.synchronize_device()
            finally:
                self.closed = True

    def __exit__(self, kind, error, tb):
        cleanup(error,[self.close])
        return False
