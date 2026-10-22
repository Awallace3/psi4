"""Owned canonical cuEST0.2.50 FP64 point potentials; no CPU integral fallback.

This experimental adapter extends an explicit DiagnosticPlanOwner's resource
graph. It creates no hidden CUDA context and has no independent teardown:
the parent graph retains its plans, parameters, workspaces and pending inputs.
"""
import ctypes as ct
from pathlib import Path

from . import sapt_cuest_factory_abi as abi
from . import sapt_cuest250_identity as identity
from .sapt_cuest_plan_owner import DiagnosticPlanOwner, _Resource
from .sapt_dispersion_cuda_setup import DeviceMath


class CuESTPointPotential:
    def __init__(self,owner,*,num_charges=1):
        if not isinstance(owner,DiagnosticPlanOwner):
            raise TypeError("An explicit owned diagnostic cuEST plan is required")
        if type(num_charges) is not int or not 1<=num_charges<2**64:
            raise ValueError("A positive uint64 point-charge count is required")
        owner._check_construction();owner.assert_live()
        self.owner,self.session=owner,owner.session
        self.num_charges=num_charges
        self.primary_basis=owner.primary_snapshot.owner
        self.nao=owner.primary_snapshot.nbf
        self.calls=0;self.failed=False;self.pending=[]
        self._bind()
        native,raw=owner.api.native,owner._raw
        plan_parameters=owner._parameters(4)
        args=(raw(owner._context),raw(owner._primary),raw(owner._pairs),raw(plan_parameters))
        self.plan=owner._workspace_object(
            "point potential OE plan",
            lambda p,t:native.cuestOEIntPlanCreateWorkspaceQuery(*args,p,t,None),
            lambda p,t,out:native.cuestOEIntPlanCreate(*args,p,t,out),
            native.cuestOEIntPlanDestroy,(owner._primary,owner._pairs,plan_parameters))
        self.parameters=owner._parameters(13)
        self.parameter_readback=owner.api._roundtrips(13,raw(self.parameters).value,(
            (0,"ffloat_usage",ct.c_int,2),(1,"jit_usage",ct.c_int,0)))
        descriptor=abi._Descriptor()
        owner.api.status("point potential workspace query",
            native.cuestPotentialComputeWorkspaceQuery(
                raw(owner._context),raw(self.plan),raw(self.parameters),
                ct.byref(descriptor),num_charges,None,None,None))
        self.workspace=owner._workspace("point potential temporary",descriptor,True)
        # Pending host/device arguments survive a failed call until synchronized
        # parent cleanup. This resource must follow its native dependencies.
        owner.resources.append(_Resource("point potential pending inputs",
            destroy=self.pending.clear,keepalive=(self,)))

    def _bind(self):
        owner=self.owner;owner.assert_live();owner.api.assert_identity()
        native=owner.api.native
        V,W,D=ct.c_void_p,ct.POINTER(abi._Workspace),ct.POINTER(abi._Descriptor)
        specs={
            "cuestOEIntPlanCreate":[V]*4+[W,W,ct.POINTER(V)],
            "cuestOEIntPlanCreateWorkspaceQuery":[V]*4+[D,D,ct.POINTER(V)],
            "cuestOEIntPlanDestroy":[V],
            "cuestPotentialCompute":[V]*3+[W,ct.c_uint64]+[abi.PD]*3,
            "cuestPotentialComputeWorkspaceQuery":[V]*3+[D,ct.c_uint64]+[abi.PD]*3}
        rows=identity.mapped_rows(Path("/proc/self/maps").read_text())
        self.symbols={}
        for name,types in specs.items():
            fn=getattr(native,name);address=ct.cast(fn,V).value
            owners={path for lo,hi,path,_,_ in rows if lo<=address<hi}
            identity.require(owners=={owner.api.proof["native_path"]},
                             "Point potential symbol outside canonical cuEST: "+name)
            fn.argtypes,fn.restype=types,ct.c_int
            self.symbols[name]=address

    def _precision(self):
        owner=self.owner;raw=owner._raw
        identity.native_math_mode(owner.api.native,owner.handle)
        # Read only: never silently restore precision after borrowed-state drift.
        for attribute,expected in ((0,2),(1,0)):
            observed=ct.c_int()
            owner.api.status("point potential precision query",
                owner.api.native.cuestParametersQuery(
                    13,raw(self.parameters),attribute,ct.byref(observed),ct.sizeof(observed)))
            if observed.value!=expected:
                raise RuntimeError("Native point potential FP64/JIT parameter drift")

    def compute_device(self,xyz,q):
        if self.failed:
            raise RuntimeError("Invalidated point potential; no computation retry")
        m=DeviceMath(self.session);m.require(xyz);m.require(q)
        if xyz.shape!=(self.num_charges,3) or q.shape!=(self.num_charges,):
            raise ValueError("Fixed-count device point coordinates/charges required")
        try:
            self._bind();self._precision()
            owner=self.owner;raw=owner._raw
            output=self.session.empty((self.nao,self.nao))
            pointers=tuple(ct.cast(v.pointer,abi.PD) for v in (xyz,q,output))
            self.pending.append((xyz,q,output,pointers))
            owner.api.status("point potential compute",
                owner.api.native.cuestPotentialCompute(
                    raw(owner._context),raw(self.plan),raw(self.parameters),
                    ct.byref(self.workspace),self.num_charges,*pointers))
            self.session.synchronize_device()
            owner.assert_live()
            self.pending.clear();self.calls+=1
            return output
        except BaseException:
            self.failed=True;self.owner._invalidate()
            raise
