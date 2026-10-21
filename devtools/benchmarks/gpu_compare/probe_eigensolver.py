#!/usr/bin/env python3
"""Compute-node-only actual-molecule eigensolver gate, not a timing benchmark."""
import argparse
import json
import os
from pathlib import Path
import sys

from compare import (atomic_json, capture_geometry, cuda_identity, digest,
                     molecular_geometry, runtime_libraries, versions, xyz)
from eigensolver import eigensolver_policy
from protocol import configure_pyscf_overlap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--xyz", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("native", "cupy-canonical"), required=True)
    a = p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        p.error("Use an allocated compute node")
    a.output.mkdir(parents=True, exist_ok=False)
    import cupy as cp
    import numpy as np
    from scipy.linalg import eigh
    from pyscf import gto, lib
    from gpu4pyscf.dft.rks import RKS
    import gpu4pyscf
    c = json.loads(a.config.read_text())
    c["pyscf_eigensolver"] = a.mode
    lib.num_threads(c["threads"])
    cp.cuda.Device(0).use()
    r = dict(config=c, diagnostic_only=True, headline_timing_eligible=False,
             config_sha256=digest(a.config), geometry_sha256=digest(a.xyz),
             gpu=cuda_identity(cp.cuda.runtime), versions=versions(), basis_cases=[])
    atom = xyz(a.xyz, "peptide")
    coordinates, unit = molecular_geometry(atom, c)
    configure_pyscf_overlap(c, r)
    for basis in ("cc-pvtz", "aug-cc-pvtz"):
        mol = gto.M(atom=coordinates, unit=unit, basis=basis, charge=0, spin=0,
                    cart=False, verbose=0, max_memory=c["memory_gib"] * 1024)
        capture_geometry(mol.atom_coords(unit="Bohr"), atom, r)
        # Independent CPU integrals and LAPACK are a correctness control only.
        h = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
        s = mol.intor_symmetric("int1e_ovlp")
        sv, su = eigh(s)
        keep = sv > c["overlap_eigenvalue_cutoff"]
        x = su[:, keep] / np.sqrt(sv[keep])
        reference, _ = eigh(x.T @ h @ x)
        mf = RKS(mol, xc="PBE")
        case = dict(basis=basis, nbf=int(mol.nao_nr()), nmo=int(keep.sum()))
        with eigensolver_policy(mf, c, case, profile=True):
            ev, coeff = mf.eig(cp.asarray(h), cp.asarray(s))
            if not all(isinstance(v, cp.ndarray) for v in (ev, coeff)):
                raise RuntimeError("Eigensolver did not return GPU matrices")
            cp.cuda.runtime.deviceSynchronize()
            values, orbitals = cp.asnumpy(ev), cp.asnumpy(coeff)
        if values.shape != reference.shape:
            raise RuntimeError("GPU/CPU reference retained ranks disagree")
        np.testing.assert_allclose(values, reference, atol=1e-7, rtol=1e-8)
        orth = float(np.max(np.abs(orbitals.T @ s @ orbitals - np.eye(len(values)))))
        residual = x.T @ (h @ orbitals - (s @ orbitals) * values)
        relative = float(np.linalg.norm(residual) /
                         max(1., np.linalg.norm(x.T @ h @ orbitals)))
        if orth > 1e-7 or relative > 1e-7:
            raise RuntimeError(f"Failed eigenpair residual/orthogonality: {relative}, {orth}")
        case.update(passed=True, maximum_eigenvalue_error=float(np.max(np.abs(values-reference))),
                    s_orthogonality_error=orth, projected_relative_residual=relative)
        r["basis_cases"].append(case)
        atomic_json(a.output / "progress.json", r)
    root = Path(gpu4pyscf.__file__).resolve().parent
    r.update(passed=True, runtime_libraries=runtime_libraries(),
             installed_source_sha256={str(root / name): digest(root / name) for name in
                 ("lib/cusolver.py", "lib/cupy_helper.py", "scf/hf.py")})
    atomic_json(a.output / "result.json", r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
