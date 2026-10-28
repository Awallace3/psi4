"""Canonical0.2.50 DFMO C-ABI adapter; no CuPy or cuest.bindings dependency.

An audited caller factory supplies native handles, exact Psi4 basis identities,
configured plan policy and persistent owners. No basis/plan is guessed here.
"""
import ctypes as ct
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from . import sapt_cuest250_identity as identity
from .sapt_cuda_runtime import cleanup, library
from .sapt_cuest_basis import require_spherical_basis
from .sapt_cuest_timing import PhaseTimings


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
        require_spherical_basis(primary_basis,role="primary")
        require_spherical_basis(auxiliary_basis,role="auxiliary")
        self.session = session
        self.pid = os.getpid()
        self.proof = identity.verify_files(sdk_root,archive)
        # verify_files already admitted the archive; preserve its locator for
        # fresh admission when a real sibling plan is requested.
        self.archive = Path(archive).resolve()
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
        self.device_timings = PhaseTimings()
        self._device_parameters = None
        self._device_pending = []
        self._device_owner = self._device_ambiguous = self.cleanup_complete = False
        self.device_input_bytes = 0
        self.assert_identity()

    def owned_plan(self):
        """Require the exact live typed parent, not guessed handle ownership.

        Borrowed providers remain usable for ordinary DFMO; shared native
        potentials/localization and sibling plans require this capability.
        """
        from .sapt_cuest_plan_owner import DiagnosticPlanOwner
        self.validate_basis(self.primary_basis,self.auxiliary_basis)
        owners = [owner for owner in self.keepalive
                  if isinstance(owner,DiagnosticPlanOwner)
                  and owner.session is self.session
                  and owner.primary_snapshot.owner is self.primary_basis
                  and owner.auxiliary_snapshot.owner is self.auxiliary_basis]
        if len(owners)!=1:
            raise ValueError("A single exact owned cuEST plan parent is required")
        owner = owners[0]
        if owner.handle!=self.handle or owner.plan!=self.plan or owner.pid!=self.pid:
            raise ValueError("DFMO provider handles differ from its owned parent")
        return owner

    @contextmanager
    def sibling_provider(self, auxiliary_basis):
        """Own another auxiliary plan on this SAME stream/session, then unwind.

        Prepare this provider first. The sibling's owner and parameters are
        registered later and closed first; device U/L can be shared unchanged.
        This creates a real needed DF plan, never a version-probing context.
        """
        from .sapt_cuest_factory_abi import admit_factory_api
        from .sapt_cuest_plan_owner import DiagnosticPlanOwner
        owner = self.owned_plan()
        if (not self._device_owner or self._device_ambiguous
                or self.session.scope_depth or not self.session.native_owners
                or self.session.native_owners[-1] is not self):
            raise RuntimeError("Prepare the original provider before sibling construction outside CUDA scopes")
        require_spherical_basis(auxiliary_basis,role="auxiliary")
        api = admit_factory_api(session=self.session,sdk_root=self.proof["sdk_root"],
            archive=self.archive,cuda_manifest=self.cuda_manifest,
            cuda_manifest_sha256=self.proof["cuda_manifest_sha256"])
        with DiagnosticPlanOwner(api,self.primary_basis,auxiliary_basis,
                coefficient_candidate=owner.COEFFICIENT_CANDIDATE,
                host_workspace_limit=owner.host_limit,
                jit_compile_threads=owner.jit_compile_threads) as sibling_owner:
            with CuESTDFMOProvider(session=self.session,sdk_root=self.proof["sdk_root"],
                    archive=self.archive,cuda_manifest=self.cuda_manifest,
                    cuda_manifest_sha256=self.proof["cuda_manifest_sha256"],
                    handle=sibling_owner.handle,plan=sibling_owner.plan,handle_pid=self.pid,
                    primary_basis=self.primary_basis,auxiliary_basis=auxiliary_basis,
                    keepalive=(sibling_owner,),plan_policy=dict(POLICY),
                    output_order=self.order,variable_bytes=self.variable_bytes) as sibling:
                sibling.prepare_device_compute()
                yield sibling

    def assert_identity(self):
        self.session.check()
        self.maps = identity.audit_maps(self.native,self.proof,self.cuda_manifest)
        self.version = identity.native_version(self.native,self.handle)
        self.math_mode = identity.native_math_mode(self.native,self.handle)
        expected = self.maps["cuda_mapped_sha256"].get(self.session.blas_path)
        if expected != self.session.blas_sha256:
            raise RuntimeError("DFMO and cuBLAS session do not share the admitted CUDA closure")

    def validate_basis(self, primary, auxiliary):
        if self.closed or self.failed:
            raise RuntimeError("Closed/invalidated DFMO provider; no retry")
        if self.primary_basis is not primary or self.auxiliary_basis is not auxiliary:
            raise ValueError("Factory must retain the exact Psi4 AO/RI basis objects")
        require_spherical_basis(primary,role="primary")
        require_spherical_basis(auxiliary,role="auxiliary")
        self.assert_identity()

    def status(self, label, code):
        if code != 0:
            self.failed = True
            raise RuntimeError(f"cuEST0.2.50 {label} failed: {code}")

    def compute(self, pairs):
        if self.closed or self.failed:
            raise RuntimeError("Closed/invalidated DFMO provider; no retry")
        self.session.check()
        try:
            # Borrowed handles can be reconfigured by their owner between calls.
            # Rejection permanently disables computation, never restores the mode.
            self.math_mode = identity.native_math_mode(self.native,self.handle)
        except BaseException:
            self.failed = True
            raise
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

    def prepare_device_compute(self):
        """Own persistent DFMO parameters BEFORE entering temporary CUDA scopes.

        The provider must be closed before its borrowed plan. The session lease
        enforces this when the plan owner was registered before this provider.
        Failure retains ownership; ambiguous native creation is quarantined.
        """
        if self.closed or self.failed or self._device_owner or self.pid != os.getpid():
            raise RuntimeError("Closed/invalidated or already prepared DFMO provider")
        if getattr(self.session,"scope_depth",0):
            raise ValueError("Prepare device DFMO outside temporary CUDA scopes")
        self.assert_identity()
        self._device_parameters = ct.c_void_p()
        self.session.retain_native_owner(self)
        self._device_owner = True
        observed = False
        try:
            status = self.native.cuestParametersCreate(49,ct.byref(self._device_parameters))
            observed = True
            if status:
                self._device_ambiguous = bool(self._device_parameters.value)
                self.status("device parameters create",status)
            if not self._device_parameters.value:
                self._device_ambiguous = True
                raise RuntimeError("Successful DFMO parameter creation returned null")
            self.session.synchronize_device()
        except BaseException:
            if not observed:
                self._device_ambiguous = True
            self.failed = True
            raise

    def compute_device(self, pairs):
        """DFMO from this session's device AO coefficients, with no host staging.

        Only metadata/descriptors and native host workspace live on the host.
        Pending call inputs are retained until successful synchronization, or
        native parameter teardown after a failed call. No execution retry.
        """
        with self.device_timings.measure("call_total"):
            return self._compute_device(pairs)

    def _compute_device(self, pairs):
        from .sapt_dispersion_cuda_setup import DeviceMath
        if (self.closed or self.failed or self.pid != os.getpid() or not self._device_owner
                or self._device_parameters is None or self._device_ambiguous):
            raise RuntimeError("Device DFMO requires a prepared live owned parameter object")
        self.session.check()
        if not pairs or len(pairs) > 10 or self.order != "C":
            raise ValueError("Device DFMO requires1..10 C-order output pairs")
        m = DeviceMath(self.session)
        names,nl,nr,left,right = [],[],[],[],[]
        for name,(cl,cr) in pairs.items():
            for value in (cl,cr):
                m.require(value)
                if len(value.shape) != 2 or value.shape[0] != self.nao:
                    raise ValueError("Invalid device FP64 AO coefficient shape")
            names.append(name); nl.append(cl.shape[1]); nr.append(cr.shape[1])
            left.append(cl); right.append(cr)
        started = time.perf_counter()
        pending = {"inputs":(tuple(left),tuple(right)), "host":None}
        # Register BEFORE any operation that can fail after native code borrows.
        self._device_pending.append(pending)
        try:
            with self.device_timings.measure("identity_and_abi"):
                self.assert_identity()
                # Factory map audits can reset ctypes descriptors. Rebind the
                # exact canonical eleven-argument DFMO ABI before use.
                identity.configure_abi(self.native)
            count = self.naux*sum(l*r for l,r in zip(nl,nr))
            with self.device_timings.measure("output_allocation"):
                output = self.session.empty((count,))
            pending["output"] = output
            with self.device_timings.measure("scope_including_cleanup"), self.session.scope():
                with self.device_timings.measure("coefficient_packing_enqueue"):
                    cl,cr = m.pack_coefficients(left),m.pack_coefficients(right)
                pending["packed"] = (cl,cr)
                nl_values = (ct.c_uint64*len(nl))(*nl)
                nr_values = (ct.c_uint64*len(nr))(*nr)
                variable,temporary = _Descriptor(0,self.variable_bytes),_Descriptor()
                pending["descriptors"] = (nl_values,nr_values,variable,temporary)
                with self.device_timings.measure("pre_query_synchronize"):
                    self.session.synchronize_device()
                args = (ct.c_void_p(self.handle),ct.c_void_p(self.plan),self._device_parameters,
                        ct.byref(variable),None,len(names),nl_values,nr_values,
                        ct.c_void_p(cl.pointer),ct.c_void_p(cr.pointer),ct.c_void_p(output.pointer))
                with self.device_timings.measure("workspace_query"):
                    self.status("device workspace query",self.native.cuestDFMOIntegralsComputeWorkspaceQuery(
                        *args[:4],ct.byref(temporary),*args[5:]))
                if temporary.host_bytes > 256*1024**2:
                    raise MemoryError("DFMO host workspace exceeds explicit256MiB cap")
                with self.device_timings.measure("workspace_allocation"):
                    host = np.empty(temporary.host_bytes,dtype=np.uint8)
                    pending["host"] = host
                    device = (self.session.empty((temporary.device_bytes,),dtype=np.uint8)
                              if temporary.device_bytes else None)
                workspace = _Workspace(host.ctypes.data if host.size else 0,host.size,
                                       device.pointer if device is not None else 0,temporary.device_bytes)
                pending["workspace"] = (device,workspace)
                with self.device_timings.measure("native_compute_enqueue"):
                    self.status("device compute",self.native.cuestDFMOIntegralsCompute(
                        *args[:4],ct.byref(workspace),*args[5:]))
                with self.device_timings.measure("completion_synchronize"):
                    self.session.synchronize_device()
            # Native work and scope cleanup both completed before host release.
            self._device_pending.remove(pending)
        except BaseException:
            self.failed = True
            raise
        factors,offset = {},0
        for name,l,r in zip(names,nl,nr):
            size = self.naux*l*r
            factors[name] = output.view((self.naux,l,r),offset=8*offset)
            offset += size
        self.calls += 1
        self.device_input_bytes += 8*self.nao*(sum(nl)+sum(nr))
        self.output_bytes += output.nbytes
        self.elapsed += time.perf_counter()-started
        return factors

    def report(self):
        # Reaudit after native work to catch late-loaded wrong CUDA libraries.
        self.assert_identity()
        return {"producer":"cuEST0.2.50 C-ABI; Python-owned CUDA buffers", "calls":self.calls,
                "coefficient_h2d_bytes":self.input_bytes,"device_factor_bytes":self.output_bytes,
                "device_coefficient_bytes":self.device_input_bytes,
                "producer_seconds":self.elapsed,"sdk_file_proof":self.proof,
                "device_host_phase_timings":self.device_timings.report(),
                "native_version":self.version,"loaded_maps":self.maps,"declared_plan_policy":POLICY,
                "native_math_mode":self.math_mode,
                "basis_metric_rank_eigenvalue_semantics_verified":False,
                "note":"Policy declarations and capabilities are not DFMO numerical/rank proof"}

    def __enter__(self):
        if self.closed or self.failed:
            raise RuntimeError("Closed/invalidated DFMO provider")
        self.assert_identity()
        return self

    def close(self):
        if self._device_owner and not self.cleanup_complete:
            self.closed = True
            try:
                if not self.session.native_owners or self.session.native_owners[-1] is not self:
                    raise RuntimeError("Later native leases must finish before DFMO parameter teardown")
                # Cleanup may run after the session disables execution. Admit
                # native destruction independently, never in a fork/on another
                # device or through an unadmitted mapped library.
                if self.pid != os.getpid():
                    raise RuntimeError("Native DFMO cleanup belongs to another process")
                current = self.session.rt(self.session.runtime.cudaGetDevice())
                if current != self.session.device:
                    raise RuntimeError("Native DFMO cleanup requires the original CUDA device")
                maps = identity.audit_maps(self.native,self.proof,self.cuda_manifest)
                if maps["cuda_mapped_sha256"].get(self.session.blas_path) != self.session.blas_sha256:
                    raise RuntimeError("Native DFMO cleanup CUDA closure differs from session")
                self.session.synchronize_device()
                if self._device_ambiguous:
                    raise RuntimeError("Ambiguous native DFMO parameter outcome; ownership quarantined")
                if self._device_parameters.value:
                    # A failed/exceptional destructor must not be retried blindly.
                    self._device_ambiguous = True
                    self.status("device parameters destroy",self.native.cuestParametersDestroy(
                        49,self._device_parameters))
                    self._device_parameters.value = None
                    self._device_ambiguous = False
                self.session.synchronize_device()
                self._device_pending.clear()
                self.cleanup_complete = True
                self.session.release_native_owner(self)
            except BaseException:
                self.failed = True
                raise
            return
        if not self.closed:
            try:
                self.session.synchronize_device()
            finally:
                self.closed = True

    def __exit__(self, kind, error, tb):
        cleanup(error,[self.close])
        return False
