"""Native spherical primary/MINAO overlap blocks for GPU IAO groundwork.

cuEST0.2.50 OE plans take one basis. A deterministic shell union computes the
full native overlap; integer metadata maps gather PP/PM/MM entirely on device.
The existing DFMO primary/auxiliary identities and plan are never replaced.
"""
import ctypes as ct
from dataclasses import dataclass,replace
import hashlib
import json
from pathlib import Path
import numpy as np

from . import sapt_cuest250_identity as identity
from . import sapt_cuest_factory_abi as abi
from .sapt_cuest_basis import snapshot_pair
from .sapt_cuest_plan_owner import DiagnosticPlanOwner,_Resource
from .sapt_dispersion_cuda_setup import DeviceMath


@dataclass(frozen=True)
class UnionBasisSnapshot:
    primary: object
    minimal: object
    shells: tuple
    primary_indices: tuple
    minimal_indices: tuple
    role: str="overlap-union"
    @property
    def ncenter(self):return self.primary.ncenter
    @property
    def nbf(self):return self.primary.nbf+self.minimal.nbf
    @property
    def ncartesian(self):return self.primary.ncartesian+self.minimal.ncartesian
    @property
    def coordinates(self):return self.primary.coordinates
    @property
    def shells_per_center(self):
        return tuple(sum(s.center==c for s in self.shells) for c in range(self.ncenter))
    @property
    def fingerprint(self):
        payload=("per-center-primary-then-minimal-v1",self.primary.fingerprint,self.minimal.fingerprint,
                 self.primary_indices,self.minimal_indices)
        return hashlib.sha256(json.dumps(payload,separators=(",",":")).encode()).hexdigest()
    def assert_unchanged(self):
        self.primary.assert_unchanged();self.minimal.assert_unchanged()


def union_snapshot(primary,minimal):
    if primary.ncenter!=minimal.ncenter or primary.coordinates!=minimal.coordinates:
        raise ValueError("Union overlaps require identical center/geometry ordering")
    maps=[[None]*primary.nbf,[None]*minimal.nbf];shells=[];offset=cartesian=0
    for center in range(primary.ncenter):
        for parent,index_map in zip((primary,minimal),maps):
            for shell in parent.shells:
                if shell.center!=center:continue
                shells.append(replace(shell,index=len(shells),function_offset=offset,cartesian_offset=cartesian))
                for k in range(shell.nfunction):index_map[shell.function_offset+k]=offset+k
                offset+=shell.nfunction;cartesian+=shell.ncartesian
    if any(i is None for indices in maps for i in indices):raise ValueError("Incomplete overlap union map")
    return UnionBasisSnapshot(primary,minimal,tuple(shells),*(tuple(indices) for indices in maps))


@dataclass(frozen=True)
class OverlapBlocksDevice:
    s11: object
    s12: object
    s22: object


def gather_overlap_blocks(session,full,primary_indices,minimal_indices,*,outputs=None):
    """Index maps are structural metadata; all matrix extraction stays on GPU."""
    m=DeviceMath(session);m.require(full)
    if len(full.shape)!=2 or full.shape[0]!=full.shape[1]:
        raise ValueError("Square full native union overlap required")
    maps=tuple(np.asarray(value) for value in (primary_indices,minimal_indices))
    for value in maps:
        if (value.dtype!=np.int64 or value.ndim!=1 or not value.size
                or np.any(value<0) or np.any(value>=full.shape[0])
                or np.unique(value).size!=value.size):
            raise ValueError("Unique in-range native int64 overlap metadata indices required")
    P,M=(len(value) for value in maps)
    if outputs is None:
        outputs=OverlapBlocksDevice(session.empty((P,P)),session.empty((P,M)),session.empty((M,M)))
    values=(outputs.s11,outputs.s12,outputs.s22)
    for value,shape in zip(values,((P,P),(P,M),(M,M))):
        m.require(value)
        if value.shape!=shape or value.allocation is full.allocation:
            raise ValueError("Wrong or aliased overlap output layout")
    if len({id(value.allocation) for value in values})!=3:
        raise ValueError("Overlap outputs require distinct allocations")
    with session.scope():
        p,q=(session.upload_indices(value) for value in maps)
        for rows,columns,output in ((p,p,outputs.s11),(p,q,outputs.s12),(q,q,outputs.s22)):
            session.launch("gather_indexed_matrix",min(65535,(output.size+255)//256),256,
                           [*output.shape,full.shape[1],rows,columns,full,output])
    return outputs


class CuESTOverlapBlocks:
    """Persistent native resources belong to the explicit parent's owner graph."""
    def __init__(self,owner,minimal_basis,*,retained_centers=None,max_union_nbf=None):
        if not isinstance(owner,DiagnosticPlanOwner):
            raise TypeError("An explicit admitted native plan owner is required")
        owner._check_construction();owner.assert_live()
        p,m=snapshot_pair(owner.primary_snapshot.owner,minimal_basis)
        if p.fingerprint!=owner.primary_snapshot.fingerprint:raise ValueError("Primary overlap basis changed")
        self.snapshot=union_snapshot(p,m)
        if max_union_nbf is None:max_union_nbf=2**31-1
        if (type(max_union_nbf) is not int or not 1<=max_union_nbf<=2**31-1
                or self.snapshot.nbf>max_union_nbf
                or max(s.am for s in self.snapshot.shells)>3):
            raise NotImplementedError("Spherical SPDF overlap union within the native ABI required")
        centers=tuple(range(p.ncenter)) if retained_centers is None else tuple(retained_centers)
        if (not centers or any(type(c) is not int or not 0<=c<p.ncenter for c in centers)
                or centers!=tuple(sorted(set(centers)))):
            raise ValueError("Ordered unique retained MINAO center metadata required")
        minimal_centers=tuple(s.center for s in m.shells for _ in range(s.nfunction))
        self.retained_centers=centers
        self.minimal_atom_centers=tuple(c for c in minimal_centers if c in centers)
        self.local_atom_indices=tuple(centers.index(c) for c in self.minimal_atom_centers)
        self.primary_indices=self.snapshot.primary_indices
        self.minimal_indices=tuple(i for i,c in zip(self.snapshot.minimal_indices,minimal_centers) if c in centers)
        self.owner,self.session=owner,owner.session
        P,M,U=len(self.primary_indices),len(self.minimal_indices),self.snapshot.nbf
        self.work_plan={"union_dimension":U,"primary_dimension":P,"minimal_dimension":M,
            "matrix_and_index_bytes":8*(U*U+P*P+P*M+M*M+P+M)}
        # Native queried workspaces are checked separately by owner._workspace;
        # matrix admission happens before creating the additional native plan.
        self.session.check_memory(self.work_plan["matrix_and_index_bytes"])
        self.calls=0;self.pending=[]
        self._bind()
        native,raw=owner.api.native,owner._raw;context=raw(owner._context)
        shell_params,basis_params=owner._parameters(1),owner._parameters(2)
        basis=owner._basis(self.snapshot,shell_params,basis_params)
        pair_params=owner._parameters(3)
        xyz=np.asarray(self.snapshot.coordinates,dtype=np.float64).reshape(-1)
        args=(context,raw(basis),self.snapshot.ncenter,xyz.ctypes.data_as(abi.PD),0.,raw(pair_params))
        pairs=owner._workspace_object("overlap union pair list",
            lambda p,t:native.cuestAOPairListCreateWorkspaceQuery(*args,p,t,None),
            lambda p,t,out:native.cuestAOPairListCreate(*args,p,t,out),
            native.cuestAOPairListDestroy,(xyz,basis,pair_params,self.snapshot))
        params=owner._parameters(4)
        args=(context,raw(basis),raw(pairs),raw(params))
        self.plan=owner._workspace_object("overlap union OE plan",
            lambda p,t:native.cuestOEIntPlanCreateWorkspaceQuery(*args,p,t,None),
            lambda p,t,out:native.cuestOEIntPlanCreate(*args,p,t,out),
            native.cuestOEIntPlanDestroy,(basis,pairs,params))
        self.parameters=owner._parameters(9);descriptor=abi._Descriptor()
        owner.api.status("overlap union workspace",native.cuestOverlapComputeWorkspaceQuery(
            context,raw(self.plan),raw(self.parameters),ct.byref(descriptor),None))
        self.workspace=owner._workspace("overlap union compute",descriptor,True)
        owner.resources.append(_Resource("overlap union pending output",
            destroy=self.pending.clear,keepalive=(self,)))

    def _bind(self):
        owner=self.owner;owner.assert_live();self.snapshot.assert_unchanged();owner.api.assert_identity()
        V,W,D=ct.c_void_p,ct.POINTER(abi._Workspace),ct.POINTER(abi._Descriptor)
        specs={"cuestOEIntPlanCreate":[V]*4+[W,W,ct.POINTER(V)],
            "cuestOEIntPlanCreateWorkspaceQuery":[V]*4+[D,D,ct.POINTER(V)],
            "cuestOEIntPlanDestroy":[V],"cuestOverlapCompute":[V]*3+[W,abi.PD],
            "cuestOverlapComputeWorkspaceQuery":[V]*3+[D,abi.PD]}
        rows=identity.mapped_rows(Path("/proc/self/maps").read_text())
        for name,args in specs.items():
            fn=getattr(owner.api.native,name);pointer=ct.cast(fn,V).value
            owners={path for lo,hi,path,_,_ in rows if lo<=pointer<hi}
            identity.require(owners=={owner.api.proof["native_path"]},"Overlap symbol outside canonical cuEST: "+name)
            fn.argtypes,fn.restype=args,ct.c_int

    def compute(self):
        owner,session=self.owner,self.session
        P,M=len(self.primary_indices),len(self.minimal_indices)
        session.check_memory(self.work_plan["matrix_and_index_bytes"])
        outputs=OverlapBlocksDevice(session.empty((P,P)),session.empty((P,M)),session.empty((M,M)))
        try:
            self._bind();identity.native_math_mode(owner.api.native,owner.handle)
            with session.scope():
                full=session.empty((self.snapshot.nbf,self.snapshot.nbf))
                pointer=ct.cast(full.pointer,abi.PD)
                self.pending.append((full,pointer))
                owner.api.status("native union overlap",owner.api.native.cuestOverlapCompute(
                    owner._raw(owner._context),owner._raw(self.plan),owner._raw(self.parameters),
                    ct.byref(self.workspace),pointer))
                session.synchronize_device();self.snapshot.assert_unchanged();owner.assert_live()
                self.pending.clear()
                gather_overlap_blocks(session,full,np.asarray(self.primary_indices,dtype=np.int64),
                    np.asarray(self.minimal_indices,dtype=np.int64),outputs=outputs)
            self.calls+=1
            return outputs
        except BaseException:
            owner._invalidate();raise
