"""Diagnostic-only owned cuEST context/bases/pair-list/DF plan.

This factory cannot yield a production factor provider. Its explicit candidate
coefficients are UNVALIDATED until independent normalization, component/unit,
layout, metric/rank and signed-component gates pass. No integrals or factors
are generated on the CPU and no DFHelper/fallback path exists.
"""
import ctypes as ct
from dataclasses import dataclass, field
import os

import numpy as np

from . import sapt_cuest_factory_abi as abi
from .sapt_cuda_runtime import cleanup
from .sapt_cuest_basis import snapshot_pair


@dataclass
class _Resource:
    label: str
    destroy: object = None
    handle: object = None
    keepalive: tuple = ()
    uncertain: bool = False
    uncertainty_stage: str = ""
    stats: dict = field(default_factory=dict)


def _cleanup_admission(api):
    # Cleanup is allowed after API/session execution is invalidated, but never
    # in a fork, on another device, or through an unadmitted mapped library.
    if api.pid != os.getpid():
        raise RuntimeError("Native cleanup belongs to another process")
    maps = abi.audit_factory_maps(api.native,api.proof,api.manifest)
    if maps["cuda_mapped_sha256"].get(api.session.blas_path) != api.session.blas_sha256:
        raise RuntimeError("Native cleanup CUDA closure differs from the session")
    current = api.session.rt(api.session.runtime.cudaGetDevice())
    if current != api.session.device:
        raise RuntimeError("Native cleanup requires the original current CUDA device")
    api.session.synchronize_device()
    api.maps = maps


class DiagnosticPlanOwner:
    """One explicit diagnostic plan; native owners precede borrowed CUDA release.

    Creation temporaries are conservatively retained for this initial small-case
    diagnostic implementation. Reverse registration order encodes dependencies:
    each native child is registered AFTER its parents, inputs and workspaces.
    Cleanup stops at the first failure, retaining that resource and every parent.
    A second automatic close cannot retry failed native destruction: callers must
    explicitly invoke retry_cleanup(). Ambiguous failed-create outputs never
    receive speculative destructor calls.
    """
    COEFFICIENT_CANDIDATE = "UNVALIDATED_PSI4_COEF"

    def __init__(self, api, primary, auxiliary, *, coefficient_candidate,
                 host_workspace_limit=256*1024**2):
        if not isinstance(api,abi._FactoryAPI):
            raise TypeError("An explicitly admitted canonical native factory API is required")
        if coefficient_candidate != self.COEFFICIENT_CANDIDATE:
            raise ValueError("Only explicit UNVALIDATED_PSI4_COEF diagnostics are implemented")
        if (isinstance(host_workspace_limit,bool) or not isinstance(host_workspace_limit,int)
                or host_workspace_limit <= 0):
            raise ValueError("An explicit positive host workspace bound is required")
        api.check()
        if getattr(api,"_plan_owner",None) is not None:
            raise RuntimeError("This native API already has a plan owner")
        p,a = snapshot_pair(primary,auxiliary)
        allowance = api.session.native_bytes
        if isinstance(allowance,bool) or not isinstance(allowance,int) or allowance <= 0:
            raise ValueError("Declare an opaque-native memory allowance; workspaces alone are not total VRAM")
        self.api,self.session,self.pid = api,api.session,os.getpid()
        self.primary_snapshot,self.auxiliary_snapshot = p,a
        self.host_limit = host_workspace_limit
        self.resources = []
        self.retired_waiting_for_sync = []
        self.closed = self.failed = self.cleanup_complete = self.cleanup_failed = False
        self.built = self.building = self.entered = False
        self._context = self._primary = self._auxiliary = self._pairs = self._plan = None
        self.version = self.context_parameters = self.plan_parameters = None
        self.session.retain_native_owner(self)  # BEFORE any native construction.
        api._plan_owner = self

    def _invalidate(self):
        self.closed = self.failed = True
        self.api.failed = self.session.failed = True

    def _check(self):
        if self.closed or self.failed or self.cleanup_complete or self.pid != os.getpid():
            raise RuntimeError("Closed/invalidated diagnostic native plan; no execution retry")
        self.api.check()

    def _check_construction(self):
        self._check()
        if getattr(self.session,"scope_depth",0):
            raise ValueError("Native construction must run outside temporary CUDA scopes")

    @staticmethod
    def _raw(resource):
        if resource is None or resource.uncertain or not resource.handle.value:
            raise RuntimeError("Native object is not live")
        return resource.handle

    def assert_live(self):
        self._check()
        try:
            self.primary_snapshot.assert_unchanged()
            self.auxiliary_snapshot.assert_unchanged()
        except BaseException:
            self._invalidate()
            raise

    @property
    def handle(self):
        self.assert_live()
        return self._raw(self._context).value

    @property
    def plan(self):
        self.assert_live()
        if not self.built:
            raise RuntimeError("Diagnostic native plan construction is incomplete")
        return self._raw(self._plan).value

    def _new_native(self, label, create, destroy, keepalive=()):
        self._check_construction()
        self.api.assert_identity()
        resource = _Resource(label,handle=ct.c_void_p(),keepalive=tuple(keepalive))
        resource.destroy = lambda:destroy(resource.handle)
        self.resources.append(resource)  # Retain the output slot BEFORE the call.
        status_observed = False
        try:
            status = create(ct.byref(resource.handle))
            status_observed = True
            if status != 0:
                resource.uncertain = bool(resource.handle.value)
                resource.uncertainty_stage = "create" if resource.uncertain else ""
                self.api.status(label+" create",status)
            if not resource.handle.value:
                resource.uncertain = True
                resource.uncertainty_stage = "create"
                raise RuntimeError("Successful native creation returned a null object")
            self.session.synchronize_device()
            return resource
        except BaseException:
            if not status_observed:
                resource.uncertain = True
                resource.uncertainty_stage = "create"
            if not resource.handle.value and not resource.uncertain:
                self.resources[:] = [item for item in self.resources if item is not resource]
            self._invalidate()
            raise

    def _parameters(self, kind):
        native = self.api.native
        return self._new_native(
            f"parameters{kind}",
            lambda out:native.cuestParametersCreate(kind,out),
            lambda handle:native.cuestParametersDestroy(kind,handle))

    def _workspace(self, label, descriptor, temporary):
        self._check_construction()
        host_bytes,device_bytes = int(descriptor.host_bytes),int(descriptor.device_bytes)
        known_host = sum(r.stats.get("host_bytes",0) for r in self.resources)
        if host_bytes + known_host > self.host_limit:
            raise MemoryError("Native host workspace subtotal exceeds declared bound")
        holder = {"host":None,"device":None}
        resource = _Resource(label,keepalive=(holder,),
                             stats={"host_bytes":0,"device_bytes":0,"temporary":temporary})
        def release():
            if holder["device"] is not None:
                holder["device"].allocation.free()  # Cleanup may follow session closure.
                holder["device"] = None
                resource.stats["device_bytes"] = 0
            holder["host"] = None
            resource.stats["host_bytes"] = 0
        resource.destroy = release
        self.resources.append(resource)
        holder["host"] = np.empty(host_bytes,dtype=np.uint8)
        resource.stats["host_bytes"] = host_bytes
        if device_bytes:
            holder["device"] = self.session.empty((device_bytes,),dtype=np.uint8)
            resource.stats["device_bytes"] = device_bytes
        workspace = abi._Workspace(
            holder["host"].ctypes.data if host_bytes else 0,host_bytes,
            holder["device"].pointer if device_bytes else 0,device_bytes)
        resource.keepalive += (workspace,)
        return workspace

    def _workspace_object(self, label, query, create, destroy, keepalive=()):
        self._check_construction()
        self.api.assert_identity()
        persistent,temporary = abi._Descriptor(),abi._Descriptor()
        self.api.status(label+" workspace query",query(ct.byref(persistent),ct.byref(temporary)))
        p = self._workspace(label+" persistent",persistent,False)
        t = self._workspace(label+" creation temporary",temporary,True)
        return self._new_native(
            label,lambda out:create(ct.byref(p),ct.byref(t),out),
            destroy,(*keepalive,p,t))

    def _basis(self, snapshot, shell_parameters, basis_parameters):
        native,context = self.api.native,self._raw(self._context)
        shells = []
        for shell in snapshot.shells:
            exps = np.asarray(shell.exponents,dtype=np.float64)
            # Deliberately a labeled candidate, NOT a native normalization claim.
            coefficients = np.asarray(shell.psi4_coefficients,dtype=np.float64)
            shells.append(self._new_native(
                f"{snapshot.role} shell{shell.index}",
                lambda out,s=shell,e=exps,c=coefficients:native.cuestAOShellCreate(
                    context,int(s.pure),s.am,len(s.exponents),
                    e.ctypes.data_as(abi.PD),c.ctypes.data_as(abi.PD),
                    self._raw(shell_parameters),out),
                native.cuestAOShellDestroy,(exps,coefficients)))
        counts = (ct.c_uint64*snapshot.ncenter)(*snapshot.shells_per_center)
        handles = (ct.c_void_p*len(shells))(*(self._raw(s).value for s in shells))
        args = (context,snapshot.ncenter,counts,handles,self._raw(basis_parameters))
        result = self._workspace_object(
            snapshot.role+" basis",
            lambda p,t:native.cuestAOBasisCreateWorkspaceQuery(*args,p,t,None),
            lambda p,t,out:native.cuestAOBasisCreate(*args,p,t,out),
            native.cuestAOBasisDestroy,(counts,handles,snapshot))
        expected = ((0,ct.c_int32,int(snapshot.shells[0].pure)),
                    (1,ct.c_int32,int(not snapshot.shells[0].pure)),
                    (2,ct.c_int32,0),(3,ct.c_uint64,snapshot.ncenter),
                    (4,ct.c_uint64,len(snapshot.shells)),(5,ct.c_uint64,snapshot.nbf),
                    (6,ct.c_uint64,snapshot.ncartesian),
                    (7,ct.c_uint64,sum(len(s.exponents) for s in snapshot.shells)),
                    (8,ct.c_uint64,max(s.am for s in snapshot.shells)))
        for attribute,ctype,value in expected:
            observed = ctype()
            self.api.status("basis dimension query",native.cuestQuery(
                context,2,self._raw(result),attribute,ct.byref(observed),ct.sizeof(observed)))
            if observed.value != value:
                raise RuntimeError("Native basis representation/dimensions disagree with Psi4 snapshot")
        return result

    def build(self):
        self._check_construction()
        if self.built or self.building:
            raise RuntimeError("Native construction is single-use")
        self.building = True
        try:
            native = self.api.native
            hp = self._parameters(0)
            self.context_parameters = self.api.configure_context_parameters(
                self._raw(hp).value,stream=int(self.session.stream),jit_compile_threads=1)
            self._context = self._new_native(
                "context",lambda out:native.cuestCreate(self._raw(hp),out),native.cuestDestroy)
            self.api.assert_identity()  # Include any lazy constructor-loaded libraries.
            self.version = abi.identity.native_version(native,self._raw(self._context).value)
            self.api.set_owned_context_fp64(self._raw(self._context).value,handle_pid=self.pid)
            sp,bp = self._parameters(1),self._parameters(2)
            self._primary = self._basis(self.primary_snapshot,sp,bp)
            self._auxiliary = self._basis(self.auxiliary_snapshot,sp,bp)
            pp = self._parameters(3)
            xyz = np.asarray(self.primary_snapshot.coordinates,dtype=np.float64).reshape(-1)
            pair_args = (self._raw(self._context),self._raw(self._primary),
                         self.primary_snapshot.ncenter,xyz.ctypes.data_as(abi.PD),0.0,self._raw(pp))
            self._pairs = self._workspace_object(
                "primary pair list",
                lambda p,t:native.cuestAOPairListCreateWorkspaceQuery(*pair_args,p,t,None),
                lambda p,t,out:native.cuestAOPairListCreate(*pair_args,p,t,out),
                native.cuestAOPairListDestroy,(xyz,))
            dp = self._parameters(5)
            self.plan_parameters = self.api.configure_df_plan_parameters(self._raw(dp).value)
            plan_args = (self._raw(self._context),self._raw(self._primary),
                         self._raw(self._auxiliary),self._raw(self._pairs),self._raw(dp))
            self._plan = self._workspace_object(
                "DF plan",
                lambda p,t:native.cuestDFIntPlanCreateWorkspaceQuery(*plan_args,p,t,None),
                lambda p,t,out:native.cuestDFIntPlanCreate(*plan_args,p,t,out),
                native.cuestDFIntPlanDestroy)
            self.session.synchronize_device()
            self.api.assert_identity()
            self.built = True
            return self
        except BaseException:
            self._invalidate()
            raise
        finally:
            self.building = False

    def close(self):
        if self.cleanup_complete:
            return
        if self.cleanup_failed:
            raise RuntimeError("Native cleanup previously failed; explicit retry_cleanup() required")
        self._close()

    def retry_cleanup(self):
        if self.cleanup_complete:
            return
        if not self.cleanup_failed:
            raise RuntimeError("No failed native cleanup to retry")
        self._close()

    def _release_retired_inputs(self):
        for resource in self.retired_waiting_for_sync:
            resource.keepalive = ()
            resource.destroy = None
        self.retired_waiting_for_sync.clear()

    def _close(self):
        self.closed = True
        self.api.failed = True
        try:
            if any(r.uncertain for r in self.resources):
                raise RuntimeError("Ambiguous native outcome retained; speculative cleanup is forbidden")
            _cleanup_admission(self.api)
            self._release_retired_inputs()
            while self.resources:
                resource = self.resources[-1]
                try:
                    status = resource.destroy()
                except BaseException as error:
                    if resource.handle is not None or not isinstance(error,Exception):
                        # No return status was observed. A native destructor
                        # might already have completed; never call it again on
                        # the assumption that its pointer is still live.
                        resource.uncertain = True
                        resource.uncertainty_stage = "destroy" if resource.handle is not None else "workspace"
                    raise
                if status is not None and status != 0:
                    raise RuntimeError(f"Native {resource.label} destroy failed: {status}")
                if resource.handle is not None:
                    resource.handle.value = None
                self.resources.pop()
                if resource.handle is not None:
                    # Successful native destruction must not be repeated if
                    # subsequent synchronization fails. Retain its host inputs
                    # until completion is established, and retain all parents.
                    self.retired_waiting_for_sync.append(resource)
                    self.session.synchronize_device()
                    self._release_retired_inputs()
                else:
                    resource.keepalive = ()
                    resource.destroy = None
            self.cleanup_complete = True
            self.session.release_native_owner(self)
            self.api._plan_owner = None
        except BaseException:
            self.cleanup_complete = False
            self.cleanup_failed = True
            self._invalidate()
            raise

    def report(self):
        return {
            "scope":"Diagnostic resource factory, NOT production/numerical admission",
            "pid":self.pid,"native_version":self.version,
            "coefficient_candidate":self.COEFFICIENT_CANDIDATE,
            "primary_fingerprint":self.primary_snapshot.fingerprint,
            "auxiliary_fingerprint":self.auxiliary_snapshot.fingerprint,
            "context_parameter_readback":self.context_parameters,
            "plan_parameter_readback":self.plan_parameters,
            "pair_thresholdPQ":0.0,"production_provider_admitted":False,
            "basis_layout_metric_rank_admitted":False,
            "known_persistent_device_workspace_bytes":sum(
                r.stats.get("device_bytes",0) for r in self.resources if not r.stats.get("temporary",False)),
            "known_retained_creation_temporary_device_bytes":sum(
                r.stats.get("device_bytes",0) for r in self.resources if r.stats.get("temporary",False)),
            "known_host_workspace_bytes":sum(r.stats.get("host_bytes",0) for r in self.resources),
            "declared_opaque_native_allowance_bytes":self.session.native_bytes,
            "opaque_native_internals_measured":False,
            "retained_resources":[r.label for r in self.resources],
            "retired_waiting_for_device_sync":[r.label for r in self.retired_waiting_for_sync],
            "ambiguous_create_outputs":[r.label for r in self.resources
                                        if r.uncertain and r.uncertainty_stage == "create"],
            "ambiguous_destroy_outcomes":[r.label for r in self.resources
                                         if r.uncertain and r.uncertainty_stage != "create"],
            "execution_closed":self.closed,"cleanup_complete":self.cleanup_complete}

    def __enter__(self):
        self._check()
        if self.entered:
            raise RuntimeError("Diagnostic native plan already has an active owner scope")
        try:
            if not self.built:
                self.build()
        except BaseException as error:
            cleanup(error,[self.close])
            raise
        self.entered = True
        return self

    def __exit__(self, kind, error, tb):
        try:
            cleanup(error,[self.close])
        finally:
            self.entered = False
        return False
