#
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2024 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

"""
Embedded SCF for I-SAPT (Intramolecular SAPT).

This module implements embedded SCF calculations for I-SAPT, where monomers A and B
are computed in the electrostatic embedding potential of fragment C. This is the
Python analog of the C++ FISAPTSCF class, but with support for DFT functionals.

The key equation is:
    F = H + W + 2*J - K + V_xc  (for DFT)
    F = H + W + 2*J - K         (for HF)

where:
    H = T + V (kinetic + nuclear attraction for monomer)
    W = V_C + 2*J_C - K_C (embedding potential from fragment C)
    J, K = Coulomb and exchange for the monomer's occupied orbitals
    V_xc = exchange-correlation potential (DFT only)

References:
    - Parrish et al., J. Chem. Phys. 141, 044115 (2014) - ISAPT0
    - Parrish et al., J. Chem. Theory Comput. 11, 2087 (2015) - ISAPT
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any

from psi4 import core

from ...p4util import solvers


def _get_matrix_dims(M):
    """Get (rows, cols) from a Matrix, handling both dimension-aware and regular types."""
    if hasattr(M, 'rowspi'):
        return M.rowspi()[0], M.colspi()[0]
    else:
        return M.rows(), M.cols()


class EmbeddedSCF:
    """
    Embedded SCF solver for I-SAPT.
    
    Performs SCF on a monomer (A or B) in the presence of an embedding potential
    from fragment C. The orbital space is restricted to exclude C orbitals.
    
    Parameters
    ----------
    jk : core.JK
        JK object for computing Coulomb and exchange integrals.
    enuc : float
        Nuclear repulsion energy for this monomer.
    S : core.Matrix
        Overlap matrix in AO basis.
    X : core.Matrix
        Restricted basis (columns span the allowed orbital space).
        Typically [LoccA_or_B, Cvir] excluding C orbitals.
    T : core.Matrix
        Kinetic energy matrix in AO basis.
    V : core.Matrix
        Nuclear attraction matrix for this monomer (V_A or V_B).
    W : core.Matrix
        Embedding potential from fragment C: W = V_C + 2*J_C - K_C.
    C_guess : core.Matrix
        Initial guess for occupied orbitals.
    functional : str, optional
        DFT functional name. If None or 'HF', pure Hartree-Fock is used.
    V_potential : core.VBase, optional
        V potential object for DFT calculations. Required if functional is not HF.
    options : dict, optional
        Dictionary of SCF options (maxiter, e_convergence, d_convergence, etc.)
    
    Attributes
    ----------
    converged : bool
        Whether SCF converged.
    energy : float
        Final SCF energy.
    Cocc : core.Matrix
        Occupied orbital coefficients.
    Cvir : core.Matrix
        Virtual orbital coefficients.
    eps_occ : core.Vector
        Occupied orbital energies.
    eps_vir : core.Vector
        Virtual orbital energies.
    J : core.Matrix
        Final Coulomb matrix.
    K : core.Matrix
        Final exchange matrix.
    """
    
    def __init__(
        self,
        jk: core.JK,
        enuc: float,
        S: core.Matrix,
        X: core.Matrix,
        T: core.Matrix,
        V: core.Matrix,
        W: core.Matrix,
        C_guess: core.Matrix,
        functional: Optional[str] = None,
        V_potential: Optional[core.VBase] = None,
        options: Optional[Dict[str, Any]] = None
    ):
        self.jk = jk
        self.enuc = enuc
        self.S = S
        self.X = X
        self.T = T
        self.V = V
        self.W = W
        self.C_guess = C_guess
        
        # Determine if we're doing DFT
        self.functional = functional
        self.is_dft = functional is not None and functional.upper() != 'HF'
        self.V_potential = V_potential
        
        if self.is_dft and self.V_potential is None:
            raise ValueError("V_potential must be provided for DFT calculations")
        
        # Parse options
        opts = options or {}
        self.maxiter = opts.get('maxiter', 100)
        self.e_convergence = opts.get('e_convergence', 1e-8)
        self.d_convergence = opts.get('d_convergence', 1e-8)
        self.diis_max_vecs = opts.get('diis_max_vecs', 6)
        self.print_level = opts.get('print_level', 1)
        
        # Results (populated by compute_energy)
        self.converged = False
        self.energy = 0.0
        self.Cocc = None
        self.Cvir = None
        self.eps_occ = None
        self.eps_vir = None
        self.J = None
        self.K = None
        self.F = None
        
    def compute_energy(self) -> float:
        """
        Run the embedded SCF procedure.
        
        Returns
        -------
        float
            The converged SCF energy.
        """
        # Sizing
        nbf, nmo = _get_matrix_dims(self.X)
        _, nocc = _get_matrix_dims(self.C_guess)
        nvir = nmo - nocc
        
        if self.print_level > 0:
            core.print_out(f"\n    Embedded SCF Calculation\n")
            core.print_out(f"    {'='*40}\n")
            core.print_out(f"    Basis functions:      {nbf:6d}\n")
            core.print_out(f"    Restricted MOs:       {nmo:6d}\n")
            core.print_out(f"    Occupied orbitals:    {nocc:6d}\n")
            core.print_out(f"    Virtual orbitals:     {nvir:6d}\n")
            core.print_out(f"    DFT functional:       {self.functional if self.is_dft else 'HF'}\n")
            core.print_out(f"    Maxiter:              {self.maxiter:6d}\n")
            core.print_out(f"    E convergence:        {self.e_convergence:11.3E}\n")
            core.print_out(f"    D convergence:        {self.d_convergence:11.3E}\n\n")
        
        # Build one-electron Hamiltonian: H = T + V
        H = self.T.clone()
        H.name = "H"
        H.add(self.V)
        
        # Initialize Fock matrix
        F = self.T.clone()
        F.name = "F"
        
        # Copy initial guess for occupied orbitals
        Cocc = self.C_guess.clone()
        Cocc.name = "Cocc"
        
        # DIIS setup
        diis = solvers.DIIS(max_vec=self.diis_max_vecs)
        
        # SCF iteration
        Eold = 0.0
        if self.print_level > 0:
            core.print_out(f"    {'Iter':>4s} {'Energy':>24s} {'dE':>12s} {'|D|':>12s} {'DIIS':>6s}\n")
        
        for iteration in range(1, self.maxiter + 1):
            # Build density matrix: D = C_occ @ C_occ.T
            D = core.doublet(Cocc, Cocc, False, True)
            
            # Compute J and K
            self.jk.C_left_add(Cocc)
            self.jk.C_right_add(Cocc)
            self.jk.compute()
            
            J = self.jk.J()[0]
            K = self.jk.K()[0]
            
            # Clear JK for next iteration
            self.jk.C_clear()
            
            # Build Fock matrix: F = H + W + 2*J - K
            F.copy(H)
            F.add(self.W)
            F.axpy(2.0, J)
            F.axpy(-1.0, K)
            
            # Add V_xc for DFT
            if self.is_dft:
                self.V_potential.set_D([D])
                self.V_potential.compute_V([F])
            
            # Compute energy: E = E_nuc + tr(D*H) + tr(D*F) + tr(D*W)
            # Note: For HF, this simplifies to E = E_nuc + tr(D*(H+F)) 
            # For DFT, we need the XC energy separately
            E_one = D.vector_dot(H)
            E_two = D.vector_dot(F)
            E_embed = D.vector_dot(self.W)
            
            if self.is_dft:
                # DFT energy: E = E_nuc + tr(D*H) + tr(D*(J - 0.5*K)) + E_xc + tr(D*W)
                # But F already includes V_xc, and V_potential has E_xc
                E_xc = self.V_potential.quadrature_values()["FUNCTIONAL"]
                # Correct energy formula for DFT with embedding
                E = self.enuc + E_one + D.vector_dot(J) - 0.5 * D.vector_dot(K) + E_xc + E_embed
            else:
                # HF energy: E = E_nuc + tr(D*H) + tr(D*F) + tr(D*W)
                E = self.enuc + E_one + E_two + E_embed
            
            dE = E - Eold
            
            # Compute orbital gradient: G = F*D*S - S*D*F (transformed to MO basis)
            FDS = core.triplet(F, D, self.S, False, False, False)
            SDF = FDS.transpose()
            G = FDS.clone()
            G.subtract(SDF)
            G.transform(self.X)
            Gnorm = G.rms()
            
            # Add to DIIS - state is F, error is G
            diis.add(F, G)
            diised = len(diis.state) >= 2  # Can extrapolate after 2 vectors
            diis_str = "DIIS" if diised else ""
            
            if self.print_level > 0:
                core.print_out(f"    {iteration:4d} {E:24.16E} {dE:12.3E} {Gnorm:12.3E} {diis_str:>6s}\n")
            
            # Check convergence
            if abs(dE) < self.e_convergence and Gnorm < self.d_convergence:
                self.converged = True
                break
            
            Eold = E
            
            # DIIS extrapolation
            if diised:
                F = diis.extrapolate()
            
            # Diagonalize Fock matrix in restricted basis
            F_mo = core.triplet(self.X, F, self.X, True, False, False)
            eps = core.Vector("eps", nmo)
            U = core.Matrix("U", nmo, nmo)
            F_mo.diagonalize(U, eps, core.DiagonalizeOrder.Ascending)
            
            # Transform back to AO basis
            C = core.doublet(self.X, U, False, False)
            
            # Extract occupied orbitals
            Cocc_new = core.Matrix("Cocc", nbf, nocc)
            Cp = C.np
            Cop = Cocc_new.np
            for m in range(nbf):
                for i in range(nocc):
                    Cop[m, i] = Cp[m, i]
            Cocc = Cocc_new
        
        if self.print_level > 0:
            core.print_out("\n")
            if self.converged:
                core.print_out("    Embedded SCF Converged.\n\n")
            else:
                core.print_out("    Embedded SCF Failed to Converge.\n\n")
            core.print_out(f"    Final SCF Energy: {E:24.16E} [Eh]\n\n")
        
        # Store final results
        self.energy = E
        self.J = J.clone()
        self.K = K.clone()
        self.F = F.clone()
        
        # Get final orbitals from last diagonalization
        F_mo = core.triplet(self.X, F, self.X, True, False, False)
        eps = core.Vector("eps", nmo)
        U = core.Matrix("U", nmo, nmo)
        F_mo.diagonalize(U, eps, core.DiagonalizeOrder.Ascending)
        C = core.doublet(self.X, U, False, False)
        
        # Split into occupied and virtual
        self.Cocc = core.Matrix("Cocc", nbf, nocc)
        self.Cvir = core.Matrix("Cvir", nbf, nvir)
        self.eps_occ = core.Vector("eps_occ", nocc)
        self.eps_vir = core.Vector("eps_vir", nvir)
        
        Cp = C.np
        ep = eps.np
        
        for m in range(nbf):
            for i in range(nocc):
                self.Cocc.np[m, i] = Cp[m, i]
            for a in range(nvir):
                self.Cvir.np[m, a] = Cp[m, nocc + a]
        
        for i in range(nocc):
            self.eps_occ.np[i] = ep[i]
        for a in range(nvir):
            self.eps_vir.np[a] = ep[nocc + a]
        
        return self.energy


def compute_fragment_nuclear_potential(
    molecule: core.Molecule,
    basisset: core.BasisSet,
    fragment_charges: core.Vector
) -> core.Matrix:
    """
    Compute the nuclear attraction potential matrix for a fragment.
    
    This computes the one-electron integrals <mu|V_frag|nu> where V_frag
    is the nuclear potential from atoms with the specified charges.
    
    Parameters
    ----------
    molecule : core.Molecule
        The full molecule.
    basisset : core.BasisSet
        The basis set.
    fragment_charges : core.Vector
        Nuclear charges for each atom (0 for atoms not in this fragment).
    
    Returns
    -------
    core.Matrix
        The nuclear attraction potential matrix.
    """
    natom = molecule.natom()
    nbf = basisset.nbf()
    Zp = fragment_charges.np
    
    # Count non-zero charges
    n_charges = sum(1 for A in range(natom) if abs(Zp[A]) > 1e-14)
    
    # If no charges, return zero matrix
    if n_charges == 0:
        return core.Matrix("V_frag", nbf, nbf)
    
    # Use ExternalPotential to compute nuclear attraction potential
    # This is the same pattern used in F-SAPT electrostatics
    ext_pot = core.ExternalPotential()
    for A in range(natom):
        if abs(Zp[A]) > 1e-14:
            ext_pot.addCharge(Zp[A], molecule.x(A), molecule.y(A), molecule.z(A))
    
    # Compute potential matrix
    # ExternalPotential.computePotentialMatrix() with positive charges gives a negative 
    # potential matrix (attractive for electrons). This is consistent with psi4's 
    # mints.ao_potential() which also returns negative values for nuclear attraction.
    V_frag = ext_pot.computePotentialMatrix(basisset)
    V_frag.name = "V_frag"
    
    return V_frag


def compute_fragment_nuclear_repulsion(
    molecule: core.Molecule,
    ZA: core.Vector,
    ZB: core.Vector,
    ZC: core.Vector
) -> Tuple[float, float, float, float]:
    """
    Compute nuclear repulsion energies between fragments.
    
    Parameters
    ----------
    molecule : core.Molecule
        The full molecule.
    ZA, ZB, ZC : core.Vector
        Nuclear charges assigned to each fragment (length = natom).
    
    Returns
    -------
    Tuple[float, float, float, float]
        (E_AA, E_BB, E_CC, E_total) - self-energies and total nuclear repulsion.
    
    Notes
    -----
    The nuclear repulsion matrix E is computed as:
        E[i,j] = 0.5 * sum_A,B Z_i[A] * Z_j[B] / R_AB  (for A != B)
    
    For I-SAPT, we typically need:
        E_A = E[A,A] (self-energy of A)
        E_B = E[B,B] (self-energy of B)
        E_AC = E[A,C] + E[C,A] (interaction of A with C)
        E_BC = E[B,C] + E[C,B] (interaction of B with C)
    """
    natom = molecule.natom()
    
    ZAp = ZA.np
    ZBp = ZB.np
    ZCp = ZC.np
    
    # Build inverse distance matrix
    Rinv = np.zeros((natom, natom))
    for A in range(natom):
        for B in range(natom):
            if A != B:
                dx = molecule.x(A) - molecule.x(B)
                dy = molecule.y(A) - molecule.y(B)
                dz = molecule.z(A) - molecule.z(B)
                R = np.sqrt(dx*dx + dy*dy + dz*dz)
                Rinv[A, B] = 1.0 / R
    
    # Build charge matrix: Zs[A, frag] = Z_frag[A]
    Zs = np.column_stack([ZAp, ZBp, ZCp])  # (natom, 3)
    
    # Compute nuclear repulsion matrix: E[i,j] = 0.5 * Zi.T @ Rinv @ Zj
    E_nuc = 0.5 * Zs.T @ Rinv @ Zs  # (3, 3)
    
    E_AA = E_nuc[0, 0]
    E_BB = E_nuc[1, 1]
    E_CC = E_nuc[2, 2]
    
    # Total nuclear repulsion
    E_total = np.sum(E_nuc)
    
    return E_AA, E_BB, E_CC, E_total


def compute_embedding_potential(
    V_C: core.Matrix,
    J_C: core.Matrix,
    K_C: core.Matrix
) -> core.Matrix:
    """
    Compute the embedding potential from fragment C.
    
    Parameters
    ----------
    V_C : core.Matrix
        Nuclear attraction potential from C nuclei.
    J_C : core.Matrix
        Coulomb matrix from C electrons.
    K_C : core.Matrix
        Exchange matrix from C electrons.
    
    Returns
    -------
    core.Matrix
        Embedding potential W_C = V_C + 2*J_C - K_C.
    """
    W_C = V_C.clone()
    W_C.name = "W_C"
    W_C.axpy(2.0, J_C)
    W_C.axpy(-1.0, K_C)
    return W_C


def compute_JK_C(
    jk: core.JK,
    Locc_C: core.Matrix,
    nbf: int
) -> Tuple[core.Matrix, core.Matrix]:
    """
    Compute Coulomb and exchange matrices for fragment C.
    
    Parameters
    ----------
    jk : core.JK
        JK object for integral computation.
    Locc_C : core.Matrix
        Localized occupied orbitals of fragment C.
    nbf : int
        Number of basis functions.
    
    Returns
    -------
    Tuple[core.Matrix, core.Matrix]
        (J_C, K_C) Coulomb and exchange matrices.
    """
    _, nocc_C = _get_matrix_dims(Locc_C)
    
    # Handle empty fragment C
    if nocc_C == 0:
        J_C = core.Matrix("J_C", nbf, nbf)
        K_C = core.Matrix("K_C", nbf, nbf)
        return J_C, K_C
    
    # Compute J and K for C
    jk.C_clear()  # Ensure JK is clean before adding
    jk.C_left_add(Locc_C)
    jk.C_right_add(Locc_C)
    jk.compute()
    
    J_C = jk.J()[0].clone()
    K_C = jk.K()[0].clone()
    J_C.name = "J_C"
    K_C.name = "K_C"
    
    jk.C_clear()
    
    return J_C, K_C


def build_restricted_basis(
    Locc_monomer: core.Matrix,
    Cvir: core.Matrix
) -> core.Matrix:
    """
    Build the restricted basis for embedded SCF.
    
    The restricted basis X spans the allowed orbital space, which includes
    the localized occupied orbitals of the monomer and the virtual orbitals
    (shared between monomers but orthogonal to C).
    
    Parameters
    ----------
    Locc_monomer : core.Matrix
        Localized occupied orbitals of the monomer (A or B).
    Cvir : core.Matrix
        Virtual orbitals (orthogonal to all occupied).
    
    Returns
    -------
    core.Matrix
        Restricted basis matrix X = [Locc_monomer | Cvir].
    """
    nbf, nocc = _get_matrix_dims(Locc_monomer)
    _, nvir = _get_matrix_dims(Cvir)
    nmo = nocc + nvir
    
    X = core.Matrix("X", nbf, nmo)
    Xp = X.np
    
    # Copy occupied orbitals
    Locc_p = Locc_monomer.np
    for m in range(nbf):
        for i in range(nocc):
            Xp[m, i] = Locc_p[m, i]
    
    # Copy virtual orbitals
    Cvir_p = Cvir.np
    for m in range(nbf):
        for a in range(nvir):
            Xp[m, nocc + a] = Cvir_p[m, a]
    
    return X


def build_restricted_basis_isapt(
    Locc_A: core.Matrix,
    Locc_B: core.Matrix,
    Cvir: core.Matrix
) -> core.Matrix:
    """
    Build the restricted basis for I-SAPT embedded SCF (C excluded).
    
    For I-SAPT, the restricted basis includes orbitals from BOTH monomers
    A and B plus the virtual orbitals. This matches the C++ FISAPT::scf()
    implementation where XC = [LoccA | LoccB | Cvir].
    
    The key insight is that while each monomer's SCF is solved independently,
    both need to be able to mix within the same orbital space (excluding C).
    
    Parameters
    ----------
    Locc_A : core.Matrix
        Localized occupied orbitals of monomer A.
    Locc_B : core.Matrix
        Localized occupied orbitals of monomer B.
    Cvir : core.Matrix
        Virtual orbitals (orthogonal to all occupied, including C).
    
    Returns
    -------
    core.Matrix
        Restricted basis matrix X = [Locc_A | Locc_B | Cvir].
    """
    nbf, nocc_A = _get_matrix_dims(Locc_A)
    _, nocc_B = _get_matrix_dims(Locc_B)
    _, nvir = _get_matrix_dims(Cvir)
    nmo = nocc_A + nocc_B + nvir
    
    X = core.Matrix("XC", nbf, nmo)
    Xp = X.np
    
    # Copy Locc_A
    Locc_Ap = Locc_A.np
    for m in range(nbf):
        for i in range(nocc_A):
            Xp[m, i] = Locc_Ap[m, i]
    
    # Copy Locc_B
    Locc_Bp = Locc_B.np
    for m in range(nbf):
        for i in range(nocc_B):
            Xp[m, nocc_A + i] = Locc_Bp[m, i]
    
    # Copy virtual orbitals
    Cvir_p = Cvir.np
    for m in range(nbf):
        for a in range(nvir):
            Xp[m, nocc_A + nocc_B + a] = Cvir_p[m, a]
    
    return X


def run_embedded_scf(
    jk: core.JK,
    molecule: core.Molecule,
    basisset: core.BasisSet,
    mints: core.MintsHelper,
    Locc_monomer: core.Matrix,
    Locc_C: core.Matrix,
    Cvir: core.Matrix,
    Z_monomer: core.Vector,
    Z_C: core.Vector,
    functional: Optional[str] = None,
    V_potential: Optional[core.VBase] = None,
    options: Optional[Dict[str, Any]] = None
) -> EmbeddedSCF:
    """
    Run embedded SCF for a monomer in the presence of fragment C.
    
    This is the main entry point for I-SAPT embedded SCF calculations.
    
    Parameters
    ----------
    jk : core.JK
        JK object for integral computation.
    molecule : core.Molecule
        The full molecule.
    basisset : core.BasisSet
        The basis set.
    mints : core.MintsHelper
        Mints helper for one-electron integrals.
    Locc_monomer : core.Matrix
        Localized occupied orbitals of the monomer (A or B).
    Locc_C : core.Matrix
        Localized occupied orbitals of fragment C.
    Cvir : core.Matrix
        Virtual orbitals (orthogonal to all occupied).
    Z_monomer : core.Vector
        Nuclear charges for the monomer (length = natom).
    Z_C : core.Vector
        Nuclear charges for fragment C (length = natom).
    functional : str, optional
        DFT functional name.
    V_potential : core.VBase, optional
        V potential object for DFT.
    options : dict, optional
        SCF options.
    
    Returns
    -------
    EmbeddedSCF
        The converged embedded SCF object.
    """
    nbf = basisset.nbf()
    
    # Get one-electron integrals
    S = mints.ao_overlap()
    T = mints.ao_kinetic()
    
    # Compute nuclear potentials for monomer and C
    V_monomer = compute_fragment_nuclear_potential(molecule, basisset, Z_monomer)
    V_C = compute_fragment_nuclear_potential(molecule, basisset, Z_C)
    
    # Compute J_C and K_C from C orbitals
    J_C, K_C = compute_JK_C(jk, Locc_C, nbf)
    
    # Build embedding potential
    W_C = compute_embedding_potential(V_C, J_C, K_C)
    
    # Build restricted basis
    X = build_restricted_basis(Locc_monomer, Cvir)
    
    # Compute nuclear repulsion for monomer
    # For I-SAPT, we need the monomer's self-energy
    natom = molecule.natom()
    Z_dummy = core.Vector("Z_dummy", natom)  # zeros
    E_monomer, _, _, _ = compute_fragment_nuclear_repulsion(
        molecule, Z_monomer, Z_dummy, Z_dummy
    )
    
    # Create and run embedded SCF
    scf = EmbeddedSCF(
        jk=jk,
        enuc=E_monomer,
        S=S,
        X=X,
        T=T,
        V=V_monomer,
        W=W_C,
        C_guess=Locc_monomer,
        functional=functional,
        V_potential=V_potential,
        options=options
    )
    scf.compute_energy()
    
    return scf
