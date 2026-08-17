#
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2025 The Psi4 Developers.
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

"""Driver orchestration for the native atomic-polarizability pipeline.

The native pipeline needs three converged SCFs at one geometry in one basis:

* the **neutral precursor**, an ordinary calculation whose HOMO energy and total energy
  enter the GRAC shift;
* the **cation**, whose total energy fixes the vertical ionization potential;
* the **GRAC-corrected reference**, rerun with that shift applied.

Multi-SCF workflows belong in the driver by Psi4 convention, so this module runs them and
hands the triple to :py:class:`psi4.core.OEProp`, which remains the publication entry
point. A bare ``OEProp`` call on a single wavefunction keeps failing closed.

See ``docs/superpowers/specs/2026-08-17-end-to-end-wiring.md``.
"""

__all__ = [
    "PIPELINE_MEMORY_BYTES",
    "atomic_polarizabilities",
    "atomic_polarizability_scf_triple",
    "publish_atomic_polarizabilities",
]

from typing import NamedTuple, Optional

from psi4 import core

from ..p4util.exceptions import ValidationError

#: Memory the pipeline configures for itself, in bytes.
#:
#: The WSM refinement builds one dense row per unordered fit-point pair, and its resource
#: gate refuses to run when the estimated peak exceeds *half* the configured memory. With
#: the default 407-point fit grid and the 170-variable C2v(Z) mask for water the fit-stage
#: SVD peak is about 0.45 GB, so the gate needs roughly 0.9 GB configured; Psi4's 500 MB
#: default admits only about 125 points and fails closed on the default grid. 4 GB is
#: chosen to leave headroom for the point-response and ISA stages at production basis sets.
#: This is deliberately explicit rather than incidental: raise it if you raise
#: ``ATOMIC_POLARIZABILITY_FIT_MAX_POINTS`` toward its 500-point architectural ceiling.
PIPELINE_MEMORY_BYTES = 4 * 1024**3


class AtomicPolarizabilitySCFTriple(NamedTuple):
    """The three converged SCFs the native pipeline requires, in constructor order."""

    grac_wfn: core.Wavefunction
    neutral_precursor_wfn: core.Wavefunction
    cation_wfn: core.Wavefunction


def atomic_polarizability_scf_triple(
    molecule: Optional[core.Molecule] = None,
    functional: str = "pbe0",
) -> AtomicPolarizabilitySCFTriple:
    """Run the neutral precursor, cation, and GRAC-corrected SCFs at one geometry.

    Parameters
    ----------
    molecule
        Neutral target. Defaults to the active molecule. Must be closed-shell neutral; the
        cation is derived from it by removing one electron at the same geometry, which is
        the vertical protocol.
    functional
        Ground-state functional. The reviewed protocol is PBE0. The response kernel is
        fixed at 25 percent CHF plus 75 percent ALDA independently of this choice.

    Returns
    -------
    AtomicPolarizabilitySCFTriple
        ``(grac_wfn, neutral_precursor_wfn, cation_wfn)``.

    Notes
    -----
    ``DFT_GRAC_SHIFT`` is restored to its incoming value before returning, so a caller's
    option state is not silently mutated.
    """
    from psi4.driver import energy

    if molecule is None:
        molecule = core.get_active_molecule()
    molecule.update_geometry()

    if molecule.molecular_charge() != 0:
        raise ValidationError(
            "atomic polarizabilities: the target molecule must be neutral; the cation "
            f"precursor is derived from it, but its charge is {molecule.molecular_charge()}"
        )
    if molecule.multiplicity() != 1:
        raise ValidationError(
            "atomic polarizabilities: the target molecule must be closed shell, but its "
            f"multiplicity is {molecule.multiplicity()}"
        )

    incoming_reference = core.get_global_option("REFERENCE")
    incoming_shift = core.get_global_option("DFT_GRAC_SHIFT")
    try:
        # The precursor and the GRAC reference must both be the closed-shell neutral, and
        # the shift must be zero while the precursor is converged or it would be circular.
        core.set_global_option("REFERENCE", "RHF")
        core.set_global_option("DFT_GRAC_SHIFT", 0.0)
        _, precursor = energy(functional, molecule=molecule, return_wfn=True)

        cation = molecule.clone()
        cation.set_molecular_charge(1)
        cation.set_multiplicity(2)
        cation.update_geometry()
        core.set_global_option("REFERENCE", "UHF")
        _, cation_wfn = energy(functional, molecule=cation, return_wfn=True)

        homo = max(precursor.epsilon_a_subset("SO", "OCC").to_array().ravel())
        shift = cation_wfn.energy() - precursor.energy() + homo

        core.set_global_option("REFERENCE", "RHF")
        core.set_global_option("DFT_GRAC_SHIFT", shift)
        _, grac_wfn = energy(functional, molecule=molecule, return_wfn=True)
    finally:
        core.set_global_option("REFERENCE", incoming_reference)
        core.set_global_option("DFT_GRAC_SHIFT", incoming_shift)

    return AtomicPolarizabilitySCFTriple(grac_wfn, precursor, cation_wfn)


def publish_atomic_polarizabilities(
    grac_wfn: Optional[core.Wavefunction],
    neutral_precursor_wfn: Optional[core.Wavefunction],
    cation_wfn: Optional[core.Wavefunction],
    memory: Optional[int] = PIPELINE_MEMORY_BYTES,
) -> core.Wavefunction:
    """Chain the native stages and publish the seven array variables.

    Parameters
    ----------
    grac_wfn, neutral_precursor_wfn, cation_wfn
        The SCF triple. Any missing or structurally inconsistent member raises an
        ``AtomicPolarizabilityPrerequisiteError``, and nothing is published.
    memory
        Process memory in bytes to configure for the pipeline, restored afterwards. Pass
        ``None`` to leave the current setting alone, which will fail closed on the default
        fit grid; see :data:`PIPELINE_MEMORY_BYTES`.

    Returns
    -------
    psi4.core.Wavefunction
        ``grac_wfn``, now carrying the seven arrays.
    """
    if grac_wfn is None:
        raise RuntimeError(
            "AtomicPolarizabilityPrerequisiteError: the GRAC-corrected reference "
            "wavefunction is missing"
        )

    properties = core.OEProp(grac_wfn)
    if neutral_precursor_wfn is not None and cation_wfn is not None:
        properties.set_atomic_polarizability_references(neutral_precursor_wfn, cation_wfn)
    properties.add("ATOMIC_POLARIZABILITIES")

    incoming_memory = core.get_memory()
    if memory is not None:
        core.set_memory_bytes(int(memory), True)
    try:
        properties.compute()
    finally:
        if memory is not None:
            core.set_memory_bytes(int(incoming_memory), True)
    return grac_wfn


def atomic_polarizabilities(
    molecule: Optional[core.Molecule] = None,
    functional: str = "pbe0",
    memory: Optional[int] = PIPELINE_MEMORY_BYTES,
) -> core.Wavefunction:
    """Run the three SCFs and publish the native atomic-polarizability outputs.

    This is the single entry point for the pipeline. It publishes
    ``ATOMIC POLARIZABILITIES`` ``(natom, 6)``, ``ATOMIC DYNAMIC POLARIZABILITIES``
    ``(nfreq * natom, 6)``, ``ATOMIC POLARIZABILITY FREQUENCIES`` ``(nfreq, 1)``, and
    ``ATOMIC C6``/``C8``/``C10``/``C12`` ``(natom, natom)``, on both the returned
    wavefunction and the global QCVariable store. Either all seven appear or none do.

    Grid quality is not inherited silently: pin ``DFT_SPHERICAL_POINTS``,
    ``DFT_RADIAL_POINTS``, and the ``ATOMIC_POLARIZABILITY_ISA_*`` keywords explicitly. The
    measured floor for the ``1e-4`` parity gate is a 302/50 DFT grid with ISA 60/18/24
    *without* diffuse functions; the SCF test fixtures are orders of magnitude too coarse
    and LW localization rejects them above ``1e-2``. With diffuse functions the DFT grid,
    not the ISA grid, becomes binding: aug-cc-pVDZ sticks at an LW charge-sum residual of
    ``1.2e-05`` on 302/50 no matter how dense the ISA grid is, and needs 590/99.
    """
    triple = atomic_polarizability_scf_triple(molecule=molecule, functional=functional)
    return publish_atomic_polarizabilities(*triple, memory=memory)
