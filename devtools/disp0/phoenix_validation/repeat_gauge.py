"""Strict repeat checks modulo orbital labels/signs, never energy rotations."""
import json
from pathlib import Path
import numpy as np
from fragment_gate import FIELDS,WATER_DIMENSIONS
from gpu_runtime import digest

FULL=("Elst_AB","Exch_AB","IndAB_AB","IndBA_AB","Disp_AB",
      "Ind20u_AB","Ind20u_BA","ExchInd20u_AB","ExchInd20u_BA",
      "Ind20r_AB","Ind20r_BA","ExchInd20r_AB","ExchInd20r_BA")
PAIRS=("Disp20 occupied pairs","Exch-Disp20 occupied pairs")
ATOL=RTOL=1e-9


def close(left,right,label,*,atol=ATOL,rtol=RTOL):
    if left.shape!=right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Incomplete/nonfinite repeat field: "+label)
    np.testing.assert_allclose(left,right,atol=atol,rtol=rtol,err_msg=label)
    return float(np.max(np.abs(left-right),initial=0.))


def compare_arrays(left,right,raw_left,raw_right,dimensions):
    if set(left)!=set(FIELDS) or set(right)!=set(FIELDS):
        raise ValueError("Every original repeat field must be compared")
    if (set(dimensions)!={"atoms","ao","occupied","frozen","active"}
            or any(type(dimensions[k]) is not int or dimensions[k]<1 for k in ("atoms","ao"))
            or any(not isinstance(dimensions[k],list) or len(dimensions[k])!=2
                   or any(type(n) is not int for n in dimensions[k])
                   for k in ("occupied","frozen","active"))):
        raise ValueError("Explicit independent case dimensions required")
    natoms=dimensions["atoms"];counts=dimensions["occupied"];frozen=dimensions["frozen"]
    active_counts=dimensions["active"]
    if any(not 0<=nf<count or count>dimensions["ao"] or active!=count-nf
           for count,nf,active in zip(counts,frozen,active_counts)):
        raise ValueError("Invalid independent frozen/active dimensions")
    s=np.asarray(raw_left["S"])
    if s.shape!=(dimensions["ao"],dimensions["ao"]):
        raise ValueError("Complete independently specified AO metric required")
    close(s,np.asarray(raw_right["S"]),"Repeat AO overlap",atol=1e-10,rtol=1e-10)
    close(s,s.T,"Symmetric AO metric",atol=1e-12,rtol=0)
    pair_shape=tuple(active_counts)
    errors={};mappings=[];signs=[];canonical_checks={}
    for side,count,nf in zip("AB",counts,frozen):
        for data,raw in ((left,raw_left),(right,raw_right)):
            co=np.asarray(raw["Cocc_"+side]);u=np.asarray(data["Uocc0"+side])
            loc=np.asarray(data["Locc_"+side])
            if co.shape!=(s.shape[0],count) or u.shape!=(count,count) or loc.shape!=co.shape:
                raise ValueError("Incomplete occupied coefficient/rotation dimensions")
            if data["Qocc0"+side].shape!=(natoms,count):
                raise ValueError("Complete atom/occupied populations required")
            close(co.T@s@co,np.eye(count),"Canonical occupied orthogonality",atol=1e-8,rtol=1e-8)
            close(u.T@u,np.eye(count),"IBO rotation orthogonality",atol=1e-8,rtol=1e-8)
            close(co@u,loc,"Independent Cocc @ Uocc reconstruction")
            close(u[:nf,nf:],np.zeros((nf,count-nf)),"Frozen/active IBO mixing",atol=1e-8,rtol=0)
            close(u[nf:,:nf],np.zeros((count-nf,nf)),"Active/frozen IBO mixing",atol=1e-8,rtol=0)
        overlap=left["Locc_"+side].T@s@right["Locc_"+side]
        mapping=np.argmax(np.abs(overlap),axis=0)
        if len(set(mapping.tolist()))!=count:
            raise RuntimeError("Repeat IBOs do not admit a unique orbital permutation")
        if np.any(mapping[:nf]>=nf) or np.any(mapping[nf:]<nf):
            raise RuntimeError("Repeat permutation mixes frozen/active spaces")
        selected=overlap[mapping,np.arange(count)]
        close(np.abs(selected),np.ones(count),"Unit matched IBO overlap")
        off=overlap.copy();off[mapping,np.arange(count)]=0
        close(off,np.zeros_like(off),"No continuous localized-orbital rotation",atol=ATOL,rtol=0)
        sign=np.sign(selected)
        mappings.append(mapping);signs.append(sign)
        errors["Locc_"+side]=close(left["Locc_"+side][:,mapping]*sign,
                                  right["Locc_"+side],"Signed/permuted repeat Locc_"+side)
        errors["Qocc0"+side]=close(left["Qocc0"+side][:,mapping],right["Qocc0"+side],
                                  "Permuted repeat populations "+side)
        # Canonical orbitals have their own phase/degenerate-space freedom.
        # Verify that frame change independently, then compare U coefficients
        # in ONE canonical frame. This never acts on an energy matrix.
        k=raw_left["Cocc_"+side].T@s@raw_right["Cocc_"+side]
        close(k.T@k,np.eye(count),"Same canonical occupied subspace",atol=1e-8,rtol=1e-8)
        close(k[:nf,nf:],np.zeros((nf,count-nf)),"Frozen/active canonical mixing",atol=1e-8,rtol=0)
        close(k[nf:,:nf],np.zeros((count-nf,nf)),"Active/frozen canonical mixing",atol=1e-8,rtol=0)
        el,er=raw_left["eps_occ_"+side],raw_right["eps_occ_"+side]
        if el.shape!=(count,) or er.shape!=(count,):raise ValueError("Full occupied energies required")
        canonical_checks[side]=close(k*er[None,:],el[:,None]*k,
                                    "Canonical energy/frame consistency "+side)
        errors["Uocc0"+side]=close(left["Uocc0"+side][:,mapping]*sign,
                                   k@right["Uocc0"+side],"Same-frame repeat Uocc0"+side)
    for name in ("ZA","ZB"):
        if left[name].shape!=(natoms,) or right[name].shape!=(natoms,):
            raise ValueError("Unchanged atomic layout required")
        errors[name]=close(left[name],right[name],"Repeat atomic charges "+name)
    indices=[np.r_[np.arange(natoms),natoms+mapping,natoms+count]
             for mapping,count in zip(mappings,counts)]
    shape=tuple(natoms+count+1 for count in counts)
    for name in FULL:
        if left[name].shape!=shape or right[name].shape!=shape:
            raise ValueError("Complete atom/occupied/external energy layout required")
        errors[name]=close(left[name][np.ix_(*indices)],right[name],"Permuted repeat energy "+name)
    active=[mapping[nf:]-nf for mapping,nf in zip(mappings,frozen)]
    for name in PAIRS:
        if left[name].shape!=pair_shape or right[name].shape!=pair_shape:
            raise ValueError("Matched complete active occupied pair components required")
        errors[name]=close(left[name][np.ix_(*active)],right[name],"Permuted repeat pairs "+name)
    if set(errors)!=set(FIELDS):raise RuntimeError("Repeat comparison dropped a field")
    return {"policy":"ORBITAL_LABELS_SIGNS_ONLY_NO_ENERGY_ROTATIONS",
        "left_indices_for_right_occupied":[m.tolist() for m in mappings],
        "left_orbital_signs_for_right":[s.tolist() for s in signs],"frozen_occupied":frozen,
        "maximum_absolute_field_errors":errors,"canonical_frame_energy_errors":canonical_checks,
        "rtol":RTOL,"atol":ATOL}


def compare_repeat(left_path,right_path):
    paths=[Path(left_path).resolve(strict=True),Path(right_path).resolve(strict=True)]
    if paths[0]==paths[1]:raise RuntimeError("Distinct fresh-process outputs required")
    docs=[json.loads((p/"result.json").read_text()) for p in paths]
    for document in docs:
        if (document["status"]!="EXECUTED_REQUIRES_INDEPENDENT_COMPARISON"
                or document["arm"]!="gpu"):
            raise RuntimeError("Completed fresh GPU repeat inputs required")
    if docs[0]["pid"]==docs[1]["pid"]:raise RuntimeError("Distinct fresh process identities required")
    for key in ("source_pin","runtime_config_sha256","geometry_sha256","scaled","orbital_dimensions"):
        if docs[0][key]!=docs[1][key]:raise RuntimeError("Unmatched GPU repeats: "+key)
    if docs[0]["orbital_dimensions"]!=WATER_DIMENSIONS:
        raise RuntimeError("Repeat dimensions differ from the fixed frozen water input")
    options=[]
    for path,document in zip(paths,docs):
        selected=dict(document["options"])
        if Path(selected.pop("fisapt_fsapt_filepath")).resolve()!=path/"fsapt":
            raise RuntimeError("Repeat output option is not its own fresh directory")
        options.append(selected)
    if options[0]!=options[1]:raise RuntimeError("Unmatched GPU repeat scientific options")
    for path,document in zip(paths,docs):
        for name in ("fragment-matrices.npz","raw-hf.npz"):
            if digest(path/name)!=document["artifact_sha256"][name]:
                raise RuntimeError("Completed repeat artifact changed")
    for name in ("energy_Eh","Disp20,u","Exch-Disp20,u"):
        close(np.asarray(docs[0][name]),np.asarray(docs[1][name]),"Signed repeat scalar "+name)
    with np.load(paths[0]/"fragment-matrices.npz",allow_pickle=False) as l, np.load(
            paths[1]/"fragment-matrices.npz",allow_pickle=False) as r, np.load(
            paths[0]/"raw-hf.npz",allow_pickle=False) as lh, np.load(
            paths[1]/"raw-hf.npz",allow_pickle=False) as rh:
        return compare_arrays(dict(l),dict(r),dict(lh),dict(rh),docs[0]["orbital_dimensions"])
