#
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2026 The Psi4 Developers.
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
"""Module with property-related helper functions."""

import time
from typing import Any, Dict, List, Optional, Tuple

import psi4

from . import free_atom_cache, optproc
from .exceptions import ValidationError

__all__ = ['free_atom_volumes', 'compute_free_atom_volume']

# Re-entrancy guard for free_atom_volumes. The atomic reference computations it runs go through
# the ordinary energy() path, which calls back into this routine whenever MBIS_VOLUME_RATIOS is in
# SCF_PROPERTIES. The flag -- rather than "is this a single atom?" -- is what stops that recursion,
# so that a single-atom *input* still gets a reference computation and hence a volume ratio.
_computing_free_atom_volumes = False

# Reference number of *unpaired electrons* per element, indexed by Z. Note that this is not the
# same as the total spin of the ground-state atom. Index 0 is for ghost atoms, whose Z() is 0.
_REFERENCE_S = [
    0, 1, 0, 1, 0, 1, 2, 3, 2, 1, 0, 1, 0, 1, 2, 3, 2, 1, 0, 1, 0, 1, 2, 3, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0,
    1, 0, 1, 2, 5, 6, 5, 4, 3, 0, 1, 0, 1, 2, 3, 2, 1, 0, 1, 0, 1, 0, 3, 4, 5, 6, 7, 8, 5, 4, 3, 2, 1, 0, 1, 2, 3,
    4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0
]

# Recorded alongside the volume in a cache entry. Only <R^3> is used to form volume ratios; the
# siblings come from the same MBIS pass for free and make an entry interpretable on its own, which
# is worth a great deal when the question is "is this number the one I think it is?".
_COMPANION_QUANTITIES = (
    'MBIS RADIAL MOMENTS <R^2>',
    'MBIS RADIAL MOMENTS <R^3>',
    'MBIS RADIAL MOMENTS <R^4>',
    'MBIS VALENCE WIDTHS',
    'MBIS VALENCE CHARGES',
    'MBIS CHARGES',
)


def free_atom_volumes(wfn: psi4.core.Wavefunction, **kwargs):
    """
    Computes free-atom volumes using MBIS density partitioning.
    The free-atom volumes are computed for all unique (inc. basis set)
    atoms in a molecule and stored as wavefunction variables, :psivar:`MBIS FREE ATOM n VOLUME`.
    Free-atom densities are computed at the same level of theory as the molecule,
    and we use unrestricted references as needed in computing the ground-state.

    The free-atom volumes are used to compute volume ratios in routine MBIS computations.

    Volumes already present on `wfn`, and volumes found in the persistent cache described in
    :py:mod:`psi4.driver.p4util.free_atom_cache`, are not recomputed.  Since a free-atom volume
    depends on the element, the level of theory, and the basis that element is given -- never on
    the molecule -- that is pure saving, and for small molecules it is 8-25% of the wall time.

    Parameters
    ----------
    wfn
        The wave function associated with the molecule, method, and basis for
        atomic computations
    """

    # Break the recursion described at _computing_free_atom_volumes: this call is coming from
    # inside one of our own atomic reference computations, which needs no volumes of its own.
    global _computing_free_atom_volumes
    if _computing_free_atom_volumes:
        return 0

    theory = _detect_theory(wfn)
    mol = wfn.molecule()
    salt = str(kwargs.get('salt', ''))

    groups = _group_atoms(wfn, mol)

    # Resolve what can be resolved without computing anything, before touching any option, so the
    # cache key describes the settings the user asked for rather than the ones we are about to
    # impose on the atomic computations.
    pending: List[Tuple[_AtomGroup, Optional[Dict[str, Any]]]] = []
    hits = 0
    for group in groups:
        key = free_atom_cache.build_key(group.symbol, group.Z, group.multiplicity, group.reference, theory,
                                        group.basis_hash, salt=salt)
        record = free_atom_cache.lookup(key)
        if record is None:
            pending.append((group, key))
        else:
            hits += 1
            _deposit(wfn, group, record["value"])

    if groups:
        psi4.core.print_out(f"  Free-atom volumes: {len(groups)} unique atoms, {hits} from cache, "
                            f"{len(pending)} to compute\n")

    # The atomic reference computations get their MBIS pass from the explicit oeprop() call below,
    # so the properties the driver would otherwise run on them are stripped: that pass is redundant
    # work, and asking it for MBIS_VOLUME_RATIOS would be asking an atom for its own volume ratio.
    #
    # This save/restore, and the activate() that re-selects the parent molecule, run even when
    # every volume came from the cache and the loop below does nothing.  Making them conditional
    # would mean the observable side effects of asking for volume ratios depended on whether some
    # earlier job had happened to populate a cache -- a difference no caller could reason about.
    optstash = optproc.OptionsState(["SCF", 'REFERENCE'], ["SCF", 'SCF_PROPERTIES'])
    psi4.core.set_local_option("SCF", "SCF_PROPERTIES", [])
    _computing_free_atom_volumes = True
    try:
        for group, key in pending:
            volume = _run_free_atom(group.labels[0], group.symbol, group.multiplicity, group.reference,
                                    group.blockname, theory, group.basis_hash, key)
            _deposit(wfn, group, volume)
            psi4.core.clean()
            psi4.core.clean_variables()
    finally:
        _computing_free_atom_volumes = False
        # reset mol and reference to original
        optstash.restore()
        mol.update_geometry()
        psi4.molutil.activate(mol)

    return 0


def compute_free_atom_volume(element: str, theory: str, basis: Optional[str] = None, salt: str = "") -> float:
    """Compute the free-atom volume of `element`, consulting and populating the cache.

    Standalone counterpart to :py:func:`free_atom_volumes`, for filling a cache ahead of a
    campaign rather than as a side effect of one; see
    :py:func:`psi4.driver.p4util.free_atom_cache.prewarm`.  All other settings are taken from the
    current psi4 state, so set them as the campaign will before calling.
    """
    global _computing_free_atom_volumes

    import qcelemental as qcel

    symbol = str(element).upper()
    Z = int(qcel.periodictable.to_Z(symbol))
    multiplicity = int(1 + _REFERENCE_S[Z])
    reference = "RHF" if _REFERENCE_S[Z] == 0 else "UHF"
    blockname = basis if basis is not None else psi4.core.get_global_option("BASIS")

    # There is no parent molecule to take the basis from, so it is identified by building it for
    # the atom directly rather than by waiting for the SCF to build one.
    atom = _atomic_molecule(symbol, symbol, multiplicity)
    basis_hash = free_atom_cache.basis_content_hash(
        psi4.core.BasisSet.build(atom, "ORBITAL", blockname, quiet=True), 0)
    key = free_atom_cache.build_key(symbol, Z, multiplicity, reference, theory, basis_hash, salt=salt)

    record = free_atom_cache.lookup(key)
    if record is not None:
        return record["value"]

    optstash = optproc.OptionsState(["SCF", 'REFERENCE'], ["SCF", 'SCF_PROPERTIES'])
    psi4.core.set_local_option("SCF", "SCF_PROPERTIES", [])
    _computing_free_atom_volumes = True
    try:
        volume = _run_free_atom(symbol, symbol, multiplicity, reference, blockname, theory, basis_hash, key)
    finally:
        _computing_free_atom_volumes = False
        optstash.restore()

    psi4.core.clean()
    psi4.core.clean_variables()
    return volume


# ==> Internals <==


class _AtomGroup:
    """The atoms of one molecule that share a single free-atom reference computation.

    Grouping is by element *and by the contracted functions that element is given*, not by the
    name of the basis: ``assign O1 aug-cc-pvtz`` in a ``basis {}`` block makes two oxygens differ
    physically while reporting the same block name for both, and the volumes they should be
    divided by genuinely differ.  The atoms' input labels are carried along because the volume is
    deposited under each of them -- that is the name oeprop.cc looks it up by.
    """

    __slots__ = ("symbol", "Z", "blockname", "basis_hash", "labels")

    def __init__(self, symbol: str, Z: int, blockname: str, basis_hash: Optional[str]):
        self.symbol = symbol
        self.Z = Z
        self.blockname = blockname
        self.basis_hash = basis_hash
        self.labels: List[str] = []

    @property
    def multiplicity(self) -> int:
        return int(1 + _REFERENCE_S[self.Z])

    @property
    def reference(self) -> str:
        # make sure we do UHF/UKS if we're not a singlet
        return "RHF" if _REFERENCE_S[self.Z] == 0 else "UHF"


def _detect_theory(wfn: psi4.core.Wavefunction) -> str:
    """Recover the level of theory the molecule was treated at from `wfn`'s scalar variables.

    It is not stored anywhere directly, and the wavefunction may have been preloaded, so the
    method is identified as whichever ``<method> TOTAL ENERGY`` variable agrees with
    ``CURRENT ENERGY``.
    """
    current_en = wfn.scalar_variable('CURRENT ENERGY')
    total_keys = [k for k in wfn.scalar_variables().keys() if ('TOTAL ENERGY' in k and 'SCF' not in k)]
    total_energy_diffs = sorted([[abs(wfn.scalar_variable(k) - current_en), k] for k in total_keys],
                                key=lambda x: x[0])
    if len(total_energy_diffs) == 0 or total_energy_diffs[0][0] > 1e-8:
        raise ValidationError(
            "No valid 'method TOTAL ENERGY' found in wavefunction scalar variables. Needed for MBIS free atoms.")

    theory = total_energy_diffs[0][1].split()[0]
    if theory == 'DFT':
        theory = wfn.functional().name()
    return theory


def _group_atoms(wfn: psi4.core.Wavefunction, mol: psi4.core.Molecule) -> List[_AtomGroup]:
    """Which free-atom references this molecule still needs, one entry per reference."""
    basisset = wfn.basisset()
    groups: Dict[Tuple[str, int, str, Optional[str]], _AtomGroup] = {}

    for atom in range(mol.natom()):
        label = mol.label(atom)
        # A volume already on the wavefunction is taken as given, which lets a driver that knows
        # these numbers -- from a previous displacement, or from its own table -- inject them.
        if wfn.has_scalar_variable(f"MBIS FREE ATOM {label} VOLUME"):
            continue
        symbol = mol.symbol(atom)
        Z = int(mol.Z(atom))
        blockname = mol.basis_on_atom(atom)
        identity = (symbol, Z, blockname, free_atom_cache.basis_content_hash(basisset, atom))
        if identity not in groups:
            groups[identity] = _AtomGroup(*identity)
        groups[identity].labels.append(label)

    return list(groups.values())


def _deposit(wfn: psi4.core.Wavefunction, group: _AtomGroup, volume: float) -> None:
    """Record `volume` under every label in `group`, which is how oeprop.cc finds it."""
    for label in group.labels:
        wfn.set_variable(f"MBIS FREE ATOM {label} VOLUME", volume)
        # set_variable("MBIS FREE ATOM n VOLUME")  # P::e OEPROP


def _atomic_molecule(label: str, symbol: str, multiplicity: int) -> psi4.core.Molecule:
    """A single atom at the origin, neutral and in its reference spin state.

    Built from the *label* so that a label-assigned basis (``assign O1 aug-cc-pvtz``) resolves to
    the functions that atom actually carries in the parent molecule.  The charge is 0 regardless
    of the parent's: the reference for a volume ratio is the neutral free atom, which is exactly
    what makes an ion's ratio differ from unity.
    """
    try:
        return psi4.core.Molecule.from_arrays(geom=[0, 0, 0],
                                              elbl=[label],
                                              molecular_charge=0,
                                              molecular_multiplicity=multiplicity)
    except Exception:
        # An exotic label qcelemental will not parse. The element still gets its volume; only the
        # label-specific basis assignment is lost, and the hash check in _run_free_atom will
        # notice if that changed the basis.
        return psi4.core.Molecule.from_arrays(geom=[0, 0, 0],
                                              elem=[symbol],
                                              molecular_charge=0,
                                              molecular_multiplicity=multiplicity)


def _run_free_atom(label: str, symbol: str, multiplicity: int, reference: str, blockname: str, theory: str,
                   basis_hash: Optional[str], key: Optional[Dict[str, Any]]) -> float:
    """Run one atomic reference computation and return its volume, caching it if `key` holds up."""
    psi4.core.set_local_option("SCF", "REFERENCE", reference)

    # Set the molecule, here just an atom
    a_mol = _atomic_molecule(label, symbol, multiplicity)
    a_mol.update_geometry()
    psi4.molutil.activate(a_mol)

    method = theory + "/" + blockname

    start = time.time()
    at_e, at_wfn = psi4.energy(method, return_wfn=True)

    # Now, re-run mbis for the atomic density, grabbing only the volume
    psi4.oeprop(at_wfn, 'MBIS_CHARGES', title=label + " " + method, free_atom=True)

    volume = at_wfn.array_variable('MBIS RADIAL MOMENTS <R^3>').get(0, 0)  # P::e OEPROP
    walltime = time.time() - start

    if _basis_as_promised(at_wfn, label, basis_hash):
        companions = {}
        for quantity in _COMPANION_QUANTITIES:
            if at_wfn.has_array_variable(quantity):
                companions[quantity] = at_wfn.array_variable(quantity).get(0, 0)
        free_atom_cache.store(key, volume, extras=companions, walltime_s=walltime)

    return volume


def _basis_as_promised(at_wfn: psi4.core.Wavefunction, label: str, basis_hash: Optional[str]) -> bool:
    """Whether the atom got the basis its cache key claims, and so whether it may be cached.

    The key is built from the parent molecule's basis, but the number is produced by a separately
    constructed one-atom molecule, and nothing structural guarantees the two resolve identically --
    a label the atomic molecule could not carry, or a ``basis {}`` body whose assignments depend on
    position, would break the correspondence.  Rather than trust it, compare.  A mismatch does not
    invalidate the volume, which is a good number for the basis it was actually computed in; it
    invalidates only our claim to know which basis that was, so the entry is not stored.
    """
    if basis_hash is None:
        return False
    if free_atom_cache.basis_content_hash(at_wfn.basisset(), 0) == basis_hash:
        return True
    psi4.core.print_out(
        f"  Warning: the free-atom computation for {label} resolved to a different basis than {label} carries\n"
        f"           in the molecule. The volume is used as computed but is not cached.\n")
    return False
