#!/usr/bin/env python3
"""Generate a matched-sample water ISA-A update oracle, NOT converged ISA water.

Run with the staged Psi4 in p4_ci:
  python make_isa_fit_fixture.py --camcasp /path/to/CamCASP

The oracle extracts executable RHS and metric-modification loops from local
CamCASP source. Adapter stubs supply samples/weighted overlap instead of invoking
CamCASP basis/grid/DF construction. No extracted source or binary is distributed.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
import psi4


def routine(text, name):
    pattern = rf"^subroutine {name}\(.*?^end subroutine {name}\b"
    match = re.search(pattern, text, re.I | re.M | re.S)
    if not match:
        raise RuntimeError(f"Cannot locate {name}")
    return match.group()


def oracle_program(root):
    rhs_text = (root / "src/num_integrals.F90").read_text()
    metric_text = (root / "src/stockholder.F90").read_text()
    rhs = routine(rhs_text, "Chi_rho_w_overlap")
    rhs = rhs[rhs.index("    if (wEps>0.0_dp) then"):rhs.index("    st_pt = end_pt + 1")]
    metric = routine(metric_text, "make_Stilde_stockholder_A")
    metric = metric[metric.index("st_k  = 0"):metric.index("Stilde%onfile = .false.")]
    # Fail loudly if a different source revision removes the branches we target.
    assert "eta*WaValues(pt)" in rhs and "MaxExp = 230.0_dp" in rhs
    assert "PositiveW_Auto" in metric and "alpha <= PositiveW_MaxAlpha" in metric
    prefix = '''module adapter
implicit none
integer, parameter :: dp=kind(1d0)
type aux_type
  integer :: nshells
  integer, allocatable :: symm(:), first(:), last(:)
  real(dp), allocatable :: expon(:)
end type
type vector_type
  real(dp), allocatable :: vector(:)
end type
type atom_type
  real(dp) :: coord(3)
  type(aux_type) :: aux
  type(vector_type) :: D
end type
type mol_type
  type(atom_type) :: atoms(1)
end type
type matrix_type
  real(dp), allocatable :: matrix(:,:)
end type
real(dp), allocatable :: samples(:,:)
contains
subroutine evaluate_gtos_shell(shell,r,chi,ncomp,aux)
integer, intent(in) :: shell
real(dp), intent(in) :: r(:,:)
real(dp), intent(out) :: chi(:,:)
integer, intent(out) :: ncomp
type(aux_type), intent(in) :: aux
ncomp=1
chi(:,1)=samples(:,shell)
end subroutine
subroutine inquire_shell_size(aux,shell,ncomp)
type(aux_type), intent(in) :: aux
integer, intent(in) :: shell
integer, intent(out) :: ncomp
ncomp=1
end subroutine
end module
program driver
use adapter
implicit none
type(mol_type) :: mol
type(matrix_type) :: S, Stilde
integer :: np,nf,i,j,atomindx,info,iblock,iauto
integer :: pt,ashell,st_a,end_a,l_a,aindx,NumComp_a,npoints_in_batch,negative_w
integer :: st_k,end_k,st_l,end_l,k_shell,l_shell,k_symm,l_symm,num_comp_k,num_comp_l,first,last
integer, allocatable :: piv(:)
real(dp) :: wEps,eta,EtaDamp,PositiveW_Lambda,PositiveW_MaxAlpha,L_wSum_eps,MaxExp,Aa(3),alpha,tot_rho_site
logical :: wEps_S_block_only,PositiveW_Auto
real(dp), allocatable :: wt(:),RhoValues(:),WaValues(:),WsumValues(:),r(:,:),rA2(:),wExpEps(:)
real(dp), allocatable :: WStockValues(:),RhoWValues(:),Chi_a(:,:),lu(:,:),solution(:,:)
read(*,*) np,nf
read(*,*) wEps,eta,PositiveW_Lambda,PositiveW_MaxAlpha,L_wSum_eps,iblock,iauto
wEps_S_block_only=iblock==1
PositiveW_Auto=iauto==1
EtaDamp=eta
atomindx=1
read(*,*) mol%atoms(1)%coord
allocate(mol%atoms(1)%aux%symm(nf),mol%atoms(1)%aux%first(nf),mol%atoms(1)%aux%last(nf))
allocate(mol%atoms(1)%aux%expon(nf),mol%atoms(1)%D%vector(nf))
mol%atoms(1)%aux%nshells=nf
read(*,*) mol%atoms(1)%aux%symm
read(*,*) mol%atoms(1)%aux%expon
read(*,*) mol%atoms(1)%D%vector
do i=1,nf
  mol%atoms(1)%aux%first(i)=i
  mol%atoms(1)%aux%last(i)=i
enddo
allocate(S%matrix(nf,1),Stilde%matrix(nf,nf),samples(np,nf),wt(np),RhoValues(np),WaValues(np),WsumValues(np))
allocate(r(3,np),rA2(np),wExpEps(np),WStockValues(np),RhoWValues(np),Chi_a(np,1),lu(nf,nf),solution(nf,1),piv(nf))
do i=1,nf
  read(*,*) Stilde%matrix(i,:)
enddo
do i=1,np
  read(*,*) r(:,i),wt(i),RhoValues(i),WaValues(i),WsumValues(i),samples(i,:)
enddo
S%matrix=0d0
negative_w=0
tot_rho_site=0d0
npoints_in_batch=np
'''
    suffix = '''
lu=Stilde%matrix
solution=S%matrix
call dgesv(nf,1,lu,nf,piv,solution,nf,info)
if(info/=0) stop 2
open(20,file='result.dat',status='replace')
write(20,'(es26.17e3)') tot_rho_site
do i=1,nf
  write(20,'(*(es26.17e3,1x))') Stilde%matrix(i,:),S%matrix(i,1),solution(i,1)
enddo
close(20)
end program
'''
    return (prefix + "call calculate\n" + suffix.replace("end program", "contains\nsubroutine calculate")
            + rhs + metric + "\nend subroutine\nend program\n")


def metric(exponents, angular, eps, s_only):
    out = np.zeros((len(exponents), len(exponents)))
    for i, ai in enumerate(exponents):
        for j, aj in enumerate(exponents):
            if angular[i] != angular[j]:
                continue
            beta = ai + aj - (eps if angular[i] == 0 or not s_only else 0.0)
            if beta <= 0:
                raise ValueError("Nonintegrable metric")
            if angular[i] == 0:
                out[i, j] = (np.pi / beta) ** 1.5
            elif i == j:  # x, y, z single p components
                out[i, j] = 0.5 * np.pi**1.5 / beta**2.5
    return out


def run_oracle(tmp, option, centre, angular, exponent, previous, overlap, points, weights, density, shape, shape_sum, phi):
    stream = io.StringIO()
    stream.write(f"{len(points)} {len(exponent)}\n")
    stream.write(" ".join(f"{x:.17e}" for x in option[:5]) + f" {int(option[5])} {int(option[6])}\n")
    for row in [centre, angular, exponent, previous]:
        np.savetxt(stream, row[None, :], fmt="%d" if row is angular else "%.17e")
    np.savetxt(stream, overlap, fmt="%.17e")
    np.savetxt(stream, np.column_stack([points, weights, density, shape, shape_sum, phi]), fmt="%.17e")
    subprocess.run([str(tmp / "oracle")], input=stream.getvalue(), text=True, cwd=tmp,
                   check=True, stdout=subprocess.DEVNULL)
    lines = (tmp / "result.dat").read_text().splitlines()
    return float(lines[0]), np.loadtxt(io.StringIO("\n".join(lines[1:])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camcasp", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    root = args.camcasp.resolve()
    program = oracle_program(root)
    args.output.mkdir(parents=True, exist_ok=True)
    psi4.set_num_threads(1)
    psi4.core.set_output_file(str(args.output / "isa_fit_generation.log"), False)
    molecule = psi4.geometry("""units bohr
no_com
no_reorient
symmetry c1
O 0 0 0
H -1.45365196 0 -1.12168732
H  1.45365196 0 -1.12168732
""")
    psi4.set_options({"basis": "sto-3g", "scf_type": "pk", "e_convergence": 1e-12, "d_convergence": 1e-11})
    energy, wfn = psi4.energy("hf", molecule=molecule, return_wfn=True)
    opts = psi4.core.IsaGridOptions()
    opts.radial_points, opts.spherical_points = 16, 110
    grid = psi4.core.IsaGrid(molecule, opts)
    points = np.column_stack([grid.x(), grid.y(), grid.z()])
    weights = np.array(grid.w())
    ao = np.array([wfn.basisset().compute_phi(*p) for p in points])
    density = np.einsum("pi,ij,pj->p", ao, wfn.Da().np + wfn.Db().np, ao)
    centres = molecule.geometry().np.copy()
    exponent = np.array([1., .8, .8, .8, .2])
    angular = np.array([0, 1, 1, 1, 0])
    previous = np.array([1., .1, -.1, .05, -.01])
    radius2, phi, shape = [], [], []
    for centre in centres:
        delta = points - centre
        r2 = (delta**2).sum(axis=1)
        values = np.exp(-r2[:, None] * exponent)
        values[:, 1:4] *= delta
        radius2.append(r2)
        phi.append(values)
        shape.append(np.maximum(values[:, [0, 4]] @ previous[[0, 4]], 0.))
    radius2, phi, shape = map(np.array, (radius2, phi, shape))
    shape_sum = shape.sum(axis=0)
    # w_eps, damping, lambda, max_alpha, cutoff, s_block_only, positive_auto
    options = np.array([[0., 0., 0., .2, 1e-36, 1, 1],
                        [.17, .03, .001, .2, 1e-36, 1, 1],
                        [.17, .03, .001, .2, 1e-36, 0, 0]])
    overlaps, results, populations = [], [], []
    with tempfile.TemporaryDirectory(prefix="isa-fit-oracle-") as tmp:
        tmp = Path(tmp)
        (tmp / "oracle.f90").write_text(program)
        subprocess.run(["gfortran", "-O0", "-ffp-contract=off", "-ffree-line-length-none",
                        "oracle.f90", "-llapack", "-lblas", "-o", "oracle"], cwd=tmp, check=True)
        for option in options:
            overlap = metric(exponent, angular, option[0], bool(option[5]))
            overlaps.append(overlap)
            atom_results, atom_populations = [], []
            for atom in range(3):
                population, result = run_oracle(tmp, option, centres[atom], angular, exponent, previous,
                                               overlap, points, weights, density, shape[atom], shape_sum, phi[atom])
                atom_populations.append(population)
                atom_results.append(result)
            results.append(atom_results)
            populations.append(atom_populations)
        # Synthetic supplied-input stress cases exercise branches that nonnegative
        # water shapes cannot. This is an algebraic checkpoint, not a physical basis.
        edges = dict(points=np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.], [1e4, 0., 0.]]),
                     weights=np.array([.2, .4, .3, .1, 1e-100]),
                     density=np.array([2., 1., -.01, .5, .7]),
                     shape=np.array([.5, .2, -.1, .3, .4]),
                     shape_sum=np.array([1., 0., -.5, 1e-36, .8]),
                     basis_values=np.array([[1., .5, .3], [.4, .8, .2], [.2, .1, .5], [.1, .2, .3], [.5, .6, .2]]),
                     overlap=np.array([[2., 0., .4], [0., 3., 0.], [.4, 0., 1.]]),
                     previous=np.array([1., -.2, -.1]), angular_momenta=np.array([0, 1, 0]),
                     exponents=np.array([1., .1, .2]))
        edges["radius_squared"] = (edges["points"]**2).sum(axis=1)
        edge_options, edge_population, edge_results = [], [], []
        for s_only in [0, 1]:
            for auto in [0, 1]:
                option = np.array([.17, .03, .5, 1., 1e-36, s_only, auto])
                population, result = run_oracle(tmp, option, np.zeros(3), edges["angular_momenta"],
                    edges["exponents"], edges["previous"], edges["overlap"], edges["points"], edges["weights"],
                    edges["density"], edges["shape"], edges["shape_sum"], edges["basis_values"])
                edge_options.append(option)
                edge_population.append(population)
                edge_results.append(result)
        edges.update(options=edge_options, population=edge_population, result=edge_results)
        edge_target = args.output / "camcasp_isa_fit_edges.npz"
        np.savez_compressed(edge_target, **edges)
    target = args.output / "camcasp_isa_fit_water.npz"
    np.savez_compressed(target, points=points, centres=centres, weights=weights, density=density,
                        shape=shape, shape_sum=shape_sum, radius_squared=radius2, basis_values=phi,
                        exponents=exponent, angular_momenta=angular, previous=previous,
                        options=options, overlap=overlaps, result=results, population=populations)
    sources = ["src/num_integrals.F90", "src/stockholder.F90", "src/parameters.f90"]
    manifest = {
        "schema_version": 1, "claim": "matched-sample frozen ISA-A update arithmetic, not converged ISA or Drho-C parity",
        "geometry_units": "bohr", "site_order": ["O", "H1", "H2"], "energy_hartree": energy,
        "density": "Psi4 RHF/STO-3G, total Da+Db, PK, e_conv=1e-12, d_conv=1e-11; not CamCASP Drho-C",
        "psi4_version": psi4.__version__, "psi4_core": psi4.core.__file__,
        "basis": "Diagnostic unnormalized [exp(-r²), x exp(-.8r²), y exp(-.8r²), z exp(-.8r²), exp(-.2r²)]; not production AtomAux",
        "shape": "max(phi_s @ previous_s, 0), identical frozen samples in both implementations",
        "metric": "Analytic same-centre diagnostic Gaussian overlap supplied to both; W-Eps weighting before source ridge/damping loops",
        "grid": {"radial_points": 16, "spherical_points": 110},
        "adapter": "Exact executable RHS and metric-modification loops; sampled basis evaluator returns one supplied component per adapter shell. LU uses LAPACK DGESV.",
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "compiler_flags": "-O0 -ffp-contract=off -ffree-line-length-none -llapack -lblas",
        "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
        "oracle_program_sha256": hashlib.sha256(program.encode()).hexdigest(),
        "fixture_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "edge_fixture_sha256": hashlib.sha256(edge_target.read_bytes()).hexdigest(),
        "edge_fixture": "Synthetic supplied-input algebra: signed sums, nonzero shape at/exceeding cutoff, auto positive-coefficient eligibility, nonzero exponent-cap contributions. Not a physical basis or density.",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "regenerate": "python tests/pytests/data_isapol/oracle/make_isa_fit_fixture.py --camcasp /path/to/CamCASP",
    }
    (args.output / "camcasp_isa_fit_water.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {target} ({target.stat().st_size} bytes); SCF energy {energy:.14f}")


if __name__ == "__main__":
    main()
