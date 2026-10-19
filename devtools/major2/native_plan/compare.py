"""Allocated-host-only full numerical comparison; tiny synthetic host tests allowed."""
import hashlib
import itertools
import math
from pathlib import Path
import numpy as np
import common as n


def array(path, size):
    p=Path(path)
    n.require(p.stat().st_size==size*size*8 and not p.stat().st_mode & 0o222,
              "readonly FP64 array size")
    a=np.fromfile(p,dtype="<f8").reshape(size,size)
    n.require(np.isfinite(a).all(),"nonfinite output")
    return a


def analyze(root, density, size=1786):
    root=Path(root)
    paths=[root/f"p{proc}"/(state+".bin") for proc in range(1,4) for state in n.STATES]
    n.require(set(root.glob("p*/*.bin"))==set(paths),"exact 18-output cohort")
    n.require(sum(p.stat().st_size for p in root.rglob("*") if p.is_file())<=n.CAP,"2GiB storage cap")
    d=array(density,size)
    outputs=[]
    for path in paths:
        j=array(path,size)
        scalar=float(np.sum(d*j.T,dtype=np.float64))
        sym=float(np.max(np.abs(j-j.T)))
        n.require(math.isfinite(scalar) and math.isfinite(sym),"finite scalar/symmetry")
        outputs.append(dict(label=f"{path.parent.name}/{path.stem}",sha256=n.sha(path),
                            scalar_TrDJ=scalar,symmetry_maxabs=sym))
    pairs=[]
    for ia,ib in itertools.combinations(range(18),2):
        aa,bb=outputs[ia],outputs[ib]
        a,b=array(paths[ia],size),array(paths[ib],size)
        delta=a-b
        scale=max(float(np.linalg.norm(a)),float(np.linalg.norm(b)),np.finfo(float).tiny)
        maxabs=float(np.max(np.abs(delta)))
        rel=float(np.linalg.norm(delta)/scale)
        same=aa["sha256"]==bb["sha256"]
        n.require(same==bool(np.array_equal(a.view(np.uint64),b.view(np.uint64))) and
                  math.isfinite(rel) and math.isfinite(maxabs),"comparison integrity")
        proc_a,state_a=aa["label"].split("/")
        proc_b,state_b=bb["label"].split("/")
        group=("fresh-process" if proc_a!=proc_b else
               "same-plan" if state_a[0]==state_b[0] else
               "DF-only" if {state_a[0],state_b[0]}=={"A","B"} else "upstream-rebuild")
        pairs.append(dict(a=aa["label"],b=bb["label"],group=group,bitwise_equal=same,
                          maxabs=maxabs,relative_F=rel,scalar_delta_TrDJ=bb["scalar_TrDJ"]-aa["scalar_TrDJ"]))
    n.require(len(pairs)==153,"all-pair count")
    groups={key:[p for p in pairs if p["group"]==key] for key in
            ("same-plan","DF-only","upstream-rebuild","fresh-process")}
    n.require({k:len(v) for k,v in groups.items()}==
              {"same-plan":9,"DF-only":12,"upstream-rebuild":24,"fresh-process":108},"comparison group counts")
    variation={k:any(not p["bitwise_equal"] for p in rows) for k,rows in groups.items()}
    verdict=("SAME_PLAN_VARIATION" if variation["same-plan"] else
             "DF_BOUNDARY_VARIATION" if variation["DF-only"] else
             "UPSTREAM_BOUNDARY_VARIATION" if variation["upstream-rebuild"] else
             "FRESH_PROCESS_ONLY_VARIATION" if variation["fresh-process"] else
             "MINIMIZATION_FAILED_NO_VARIATION")
    return dict(outputs=outputs,pairs=pairs,group_counts={k:len(v) for k,v in groups.items()},
                variation=variation,verdict=verdict,scalar="Tr(DJ)",
                findings_only=True,clears_wrapper=False,unique_cause=False,
                full_sapt_acceptance=False,performance_acceptance=False,followup="STOP for parent review")
