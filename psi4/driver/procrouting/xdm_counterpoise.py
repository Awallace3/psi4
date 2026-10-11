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

"""Counterpoise treatment of ghost atoms for XDM dispersion.

XDM partitions the density with Hirshfeld weights built from a promolecule of
free atoms. Ghost centres stay in that promolecule, so a counterpoise monomer
and its parent cluster share the grid and the weights, and the monomer's real
atoms keep only the share of the density their dimer counterparts would get.

That alone does not make the interaction energy additive. In the cluster, atom
*a* of monomer A also picks up the partner's density, :math:`\\int h_a \\rho_B`,
weighted by up to :math:`r^6` in :math:`\\langle M_3^2\\rangle`. That pickup
inflates C10, hence Rc and the BJ damping radius of every A-A pair, and it is
absent from the counterpoise monomer. The resulting intramonomer energy shift
decays far more slowly than :math:`R^{-6}` and changes the sign of the
dispersion interaction energy at long range for large-``a1`` parameter sets.

The partner's pickup is exactly what the partner's own counterpoise job
deposits on its ghost centres. :func:`superpose_ghost_moments` adds those
ghost-centre moments to each counterpoise subsystem's real atoms and
re-evaluates its XDM energy, so the subtraction removes the superposition
artefact and leaves only the intermolecular pairs and the true density
deformation.
"""

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from psi4 import core

from ..p4util.exceptions import ValidationError

#: First column of ``XDM ATOMIC PROPERTIES`` that is additive over fragment densities
#: (charge, volume, <M1^2>, <M2^2>, <M3^2>, polarizability).
_ADDITIVE_COLUMNS = slice(2, None)


def _xdm_task_data(result) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    qcvars = (result.extras or {}).get("qcvars", {})
    props = qcvars.get("XDM ATOMIC PROPERTIES")
    params = qcvars.get("XDM DAMPING PARAMETERS")
    if props is None or params is None:
        return None
    coords = np.asarray(result.molecule.geometry, dtype=float).reshape(-1, 3)
    props = np.asarray(props, dtype=float)
    if props.shape[0] != coords.shape[0]:
        raise ValidationError("XDM counterpoise: per-atom properties do not match the task molecule.")
    return coords, props, np.asarray(params, dtype=float).ravel()


def ghost_superposition_correction(xdm: core.XDMDispersion, coords: np.ndarray, props: np.ndarray,
                                   partner_props: Sequence[np.ndarray]) -> float:
    """XDM energy change [Eh] from adding partner jobs' ghost-centre moments to ``props``.

    ``props`` and every entry of ``partner_props`` are ``XDM ATOMIC PROPERTIES`` arrays
    on the same centres (same atom order, bohr ``coords``); each partner is the
    counterpoise job of one ghost fragment of ``props``'s job, in the same basis.
    """
    superposed = props.copy()
    for partner in partner_props:
        if partner.shape != props.shape or not np.array_equal(partner[:, 0], props[:, 0]):
            raise ValidationError("XDM counterpoise: partner properties are not on the same centres.")
        superposed[:, _ADDITIVE_COLUMNS] += partner[:, _ADDITIVE_COLUMNS]
    coords_m = core.Matrix.from_array(coords)
    e_bare = xdm.energy_from_properties(coords_m, core.Matrix.from_array(props))
    e_superposed = xdm.energy_from_properties(coords_m, core.Matrix.from_array(superposed))
    return e_superposed - e_bare


def superpose_ghost_moments(component_results: Mapping[str, Any],
                            component_properties: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """Correct the XDM energy of every subsystem computed with ghost fragments.

    For a subsystem with real fragments S in the basis of fragments T, the
    real atoms' moments are augmented by the ghost-centre moments of each
    single-fragment task ({F}, T), F in T \\ S, and the XDM pair sum is
    re-evaluated. The difference is added to the task's energy in
    ``component_properties`` before many-body assembly.

    Returns
    -------
    dict
        Energy correction [Eh] per corrected task label.
    """
    from qcmanybody.utils import delabeler

    tasks = {}
    for label, result in component_results.items():
        data = _xdm_task_data(result)
        if data is not None:
            mc, frag, bas = delabeler(label)
            tasks[label] = (mc, tuple(frag), tuple(bas), data)
    by_key = {(mc, frag, bas): label for label, (mc, frag, bas, _) in tasks.items()}

    corrections = {}
    for label, (mc, frag, bas, (coords, props, params)) in tasks.items():
        ghosts = sorted(set(bas) - set(frag))
        if not ghosts:
            continue
        partner_props = []
        for ghost in ghosts:
            partner = by_key.get((mc, (ghost, ), bas))
            if partner is None:
                raise ValidationError(f"XDM counterpoise: task {label} needs fragment {ghost} computed alone in the "
                                      f"basis of fragments {list(bas)} to superpose its density, and none was run.")
            partner_props.append(tasks[partner][3][1])

        xdm = core.XDMDispersion(float(params[0]), float(params[1]), "")
        correction = ghost_superposition_correction(xdm, coords, props, partner_props)
        component_properties[label]["energy"] = component_properties[label]["energy"] + correction
        corrections[label] = correction

    if corrections:
        core.print_out("\n  ==> XDM Counterpoise: Superposed Ghost-Centre Moments <==\n\n")
        core.print_out(f"    {'Task':<50} {'dE_XDM [Eh]':>18}\n")
        for label, correction in corrections.items():
            core.print_out(f"    {label:<50} {correction:18.12f}\n")
        core.print_out("\n")
    return corrections
