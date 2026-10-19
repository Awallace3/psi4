"""Allocation-only public Psi4 BasisSet export. No SCF, energy or JK."""
import os
from pathlib import Path
import sys
import common as n


def main():
    n.allocation_only()  # before numpy/Psi4 imports
    m = n.manifest(sys.argv[1],sys.argv[2])
    out = Path(sys.argv[3])
    n.durable(out/"export-effective-env.json",dict(numerical_overrides=n.check_env(os.environ),
                                                pid=os.getpid(),before_native_import=True))
    sys.path.insert(0,str(n.HERE.parent))
    import probe
    # Reuse exact reviewed core-first/PyEinsums runtime validation.
    sys.argv = [__file__,str(out/"base-import.json")]
    probe.main()
    import psi4
    import numpy as np
    import campaign
    import wire
    n.require(n.sha(psi4.core.__file__)==n.CORE_SHA and
              Path(psi4.core.__file__).resolve().is_relative_to(Path(n.STAGE)),
              "pinned BASE core")
    n.require(n.sha(Path(m["suite"])/"systems.py")==m["suite_sha256"],"suite bytes")
    geometry = campaign.literal_assignments(Path(m["suite"])/"systems.py",["SYSTEMS"])["SYSTEMS"]["protein157"]
    import hashlib
    n.require(hashlib.sha256(geometry.encode()).hexdigest()==m["geometry_sha256"],"geometry bytes")
    psi4.core.set_output_file(str(out/"basis.out"),False)
    mol=psi4.geometry(geometry)
    b=psi4.core.BasisSet.build(mol,"ORBITAL","6-31+g**",puream=True)
    a=psi4.core.BasisSet.build(mol,"DF_BASIS_SCF","def2-universal-jkfit","JKFIT","6-31+g**",puream=True)
    n.require((mol.natom(),b.nbf(),a.nbf())==(157,1786,7265) and
              b.has_puream() and a.has_puream() and mol.schoenflies_symbol()=="c1","basis/frame")
    xyz=[[mol.geometry().get(i,k) for k in range(3)] for i in range(mol.natom())]
    bases=[wire.snapshot(v,mol.natom()) for v in (b,a)]
    payload=wire.encode(xyz,bases)
    n.durable(out/"basis.bin",payload,binary=True)
    # Manifest carries shell snapshots too: JSON float representation roundtrips FP64.
    n.durable(out/"basis.json",dict(format="NDFPLAN1",units="bohr",xyz=xyz,bases=bases,
        bytes=len(payload),sha256=n.sha(out/"basis.bin"),natom=157,nbf=1786,naux=7265,
        shell_coordinates="snapshot; pairlist uses final primary molecule geometry",
        coefficients="GaussianShell.coef, verbatim normalized values",order="global shell/AO order"))
    saved=n.load(out/"basis.json")
    n.require(wire.encode(saved["xyz"],saved["bases"])==payload,"JSON/binary exact float64 roundtrip")
    for name,digest in n.FACTOR_FILES.items():
        p=Path(n.FACTOR)/name
        n.require(n.sha(p)==digest and not p.stat().st_mode & 0o222,"frozen original factor")
    d=np.load(Path(n.FACTOR)/"D.npy",allow_pickle=False)
    n.require(d.shape==(1786,1786) and d.dtype==np.dtype("<f8") and
              d.flags.c_contiguous and np.isfinite(d).all(),"frozen density shape/dtype")
    data=d.tobytes(order="C")
    n.require(hashlib.sha256(data).hexdigest()==n.D_SHA,"original density array hash")
    n.durable(out/"D.bin",data,binary=True)
    reread=np.fromfile(out/"D.bin",dtype="<f8").reshape(d.shape)
    n.require(np.array_equal(reread.view(np.uint64),d.view(np.uint64)),"D extraction byte proof")
    n.durable(out/"INPUT-COMPLETE.json",dict(basis_sha256=n.sha(out/"basis.bin"),
        density_sha256=n.sha(out/"D.bin"),frozen_factor=n.FACTOR_FILES,rank=465,
        D_is_not_C_reconstructed=True,computed_from_C=False,scf=0,energy=0,jk=0,
        core_sha256=n.CORE_SHA,pid=os.getpid(),job=os.environ["SLURM_JOB_ID"]))
    psi4.core.close_outfile()


if __name__ == "__main__":
    main()
