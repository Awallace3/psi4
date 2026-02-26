"""XDM Becke-Johnson damping parameters managed on Python side."""

from __future__ import annotations

from typing import Dict, Tuple


_XDM_BJ_PARAMS_ANGSTROM: Dict[str, Tuple[float, float]] = {
    # B3LYP - /nocp performs a little worse at dz and smaller
    "b3lyp/aug-cc-pvtz": (0.538965, 1.707159),
    "b3lyp/aug-cc-pvtz/nocp": (0.299469, 2.454713),
    "b3lyp/aug-cc-pvdz": (0.541310, 1.707761),
    "b3lyp/aug-cc-pvdz/nocp": (0.594163, 1.708002),
    "b3lyp/6-31+g*": (0.539265, 1.706567),
    "b3lyp/6-31+g*/nocp": (0.249120, 2.690583),
    "b3lyp/6-31+g**": (0.544086, 1.707329),
    "b3lyp/6-31+g**/nocp": (0.226145, 2.762253),
    "b3lyp/6-311+g(2d,2p)": (0.541432, 1.706280),
    "b3lyp/6-311+g(2d,2p)/nocp": (0.564665, 1.708375),
    "b3lyp/cc-pvdz": (0.502835, 1.705159),
    "b3lyp/cc-pvdz/nocp": (0.200048, 3.125447),
    "b3lyp/cc-pvtz": (0.533051, 1.708554),
    "b3lyp/cc-pvtz/nocp": (0.474763, 2.033757),

    # PBE0
    "pbe0/6-31+g*": (0.665768, 1.708279),
    "pbe0/6-31+g*/nocp": (0.000000, 3.942401),
    "pbe0/6-31+g**": (0.672756, 1.709480),
    "pbe0/6-31+g**/nocp": (0.000000, 3.957889),
    "pbe0/aug-cc-pvdz": (0.675533, 1.709020),
    "pbe0/aug-cc-pvdz/nocp": (0.757220, 1.709831),
    "pbe0/cc-pvdz": (0.648251, 1.707199),
    "pbe0/cc-pvdz/nocp": (0.000016, 4.226151),
    "pbe0/cc-pvtz": (0.662432, 1.709395),
    "pbe0/cc-pvtz/nocp": (0.337012, 2.867459),

    # PBE
    "pbe/6-31+g*": (0.638224, 1.705838),
    "pbe/6-31+g*/nocp": (0.699062, 1.710117),
    "pbe/6-31+g**": (0.644670, 1.707056),
    "pbe/6-31+g**/nocp": (0.704316, 1.709007),
    "pbe/aug-cc-pvdz": (0.646009, 1.706796),
    "pbe/aug-cc-pvdz/nocp": (0.719266, 1.705258),
    "pbe/aug-cc-pvtz": (0.641172, 1.707118),
    "pbe/aug-cc-pvtz/nocp": (0.658991, 1.708286),
}


def get_xdm_bj_params(functional_name: str, basis_name: str, *, nocp: bool = False) -> Tuple[float, float]:
    """Return (a1, a2_angstrom) for a functional/basis XDM model.

    Parameters
    ----------
    functional_name
        XC functional name (without ``-xdm`` suffix).
    basis_name
        Orbital basis name.
    nocp
        If ``True``, prefer fitted ``/nocp`` parameterization.
    """

    func = functional_name.lower()
    basis = basis_name.lower()
    key = f"{func}/{basis}"

    if nocp:
        key_nocp = f"{key}/nocp"
        if key_nocp in _XDM_BJ_PARAMS_ANGSTROM:
            return _XDM_BJ_PARAMS_ANGSTROM[key_nocp]

    if key in _XDM_BJ_PARAMS_ANGSTROM:
        return _XDM_BJ_PARAMS_ANGSTROM[key]

    raise KeyError(key)
