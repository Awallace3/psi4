"""XDM Becke-Johnson damping parameters managed on Python side."""

from __future__ import annotations

from typing import Dict, Tuple


_XDM_BJ_PARAMS_ANGSTROM: Dict[str, Tuple[float, float]] = {
    # B3LYP - /nocp performs a little worse at dz and smaller
    "b3lyp/aug-cc-pvtz/cp": (0.538965, 1.707159),
    "b3lyp/aug-cc-pvtz": (0.299469, 2.454713),
    "b3lyp/aug-cc-pvdz/cp": (0.541310, 1.707761),
    "b3lyp/aug-cc-pvdz": (0.594163, 1.708002),
    "b3lyp/6-31+g*/cp": (0.539265, 1.706567),
    "b3lyp/6-31+g*": (0.249120, 2.690583),
    "b3lyp/6-31+g**/cp": (0.544086, 1.707329),
    "b3lyp/6-31+g**": (0.226145, 2.762253),
    "b3lyp/6-311+g(2d,2p)/cp": (0.541432, 1.706280),
    "b3lyp/6-311+g(2d,2p)": (0.564665, 1.708375),
    "b3lyp/cc-pvdz/cp": (0.502835, 1.705159),
    "b3lyp/cc-pvdz": (0.200048, 3.125447),
    "b3lyp/cc-pvtz/cp": (0.533051, 1.708554),
    "b3lyp/cc-pvtz": (0.474763, 2.033757),
    # PBE0
    "pbe0/6-31+g*/cp": (0.665768, 1.708279),
    "pbe0/6-31+g*": (0.000000, 3.942401),
    "pbe0/6-31+g**/cp": (0.672756, 1.709480),
    "pbe0/6-31+g**": (0.000000, 3.957889),
    "pbe0/aug-cc-pvdz/cp": (0.675533, 1.709020),
    "pbe0/aug-cc-pvdz": (0.757220, 1.709831),
    "pbe0/cc-pvdz/cp": (0.648251, 1.707199),
    "pbe0/cc-pvdz": (0.000016, 4.226151),
    "pbe0/cc-pvtz/cp": (0.662432, 1.709395),
    "pbe0/cc-pvtz": (0.337012, 2.867459),
    # PBE
    "pbe/6-31+g*/cp": (0.638224, 1.705838),
    "pbe/6-31+g*": (0.699062, 1.710117),
    "pbe/6-31+g**/cp": (0.644670, 1.707056),
    "pbe/6-31+g**": (0.704316, 1.709007),
    "pbe/aug-cc-pvdz/cp": (0.646009, 1.706796),
    "pbe/aug-cc-pvdz": (0.719266, 1.705258),
    "pbe/aug-cc-pvtz/cp": (0.641172, 1.707118),
    "pbe/aug-cc-pvtz": (0.658991, 1.708286),
}

_XDM_LOS_II_PARAMS_ANGSTROM: Dict[str, Tuple[float, float]] = {
    # Powell optimizations to minimize error of DFT IE to estimated CCSD(T)/CBS
    # reference on 4568 dimers from doi.org/10.1063/5.0275311

    # b3lyp/aug-cc-pvdz/cp MAE: 0.137372 kcal/mol on LoS-II
    "b3lyp/aug-cc-pvdz/cp": (0.315041, 2.359816),
    # b3lyp/aug-cc-pvdz MAE: 0.199973 kcal/mol on LoS-II
    "b3lyp/aug-cc-pvdz": (0.362043, 2.452093),
    # b3lyp/aug-cc-pvtz/cp MAE: 0.134453 kcal/mol on LoS-II
    "b3lyp/aug-cc-pvtz/cp": (0.352501, 2.241051),
    # b3lyp/aug-cc-pvtz MAE: 0.136507 kcal/mol on LoS-II
    "b3lyp/aug-cc-pvtz": (0.344500, 2.304241),
    # pbe0/aug-cc-pvdz/cp MAE: 0.157610 kcal/mol on LoS-II
    "pbe0/aug-cc-pvdz/cp": (0.137099, 3.344830),
    # pbe0/aug-cc-pvdz MAE: 0.208978 kcal/mol on LoS-II
    "pbe0/aug-cc-pvdz": (0.068267, 3.963874),
}


def normalize_xdm_model(model: str) -> str:
    """Normalize XDM damping model names to canonical internal labels."""

    normalized = model.strip().lower()
    aliases = {
        "": "kb49",
        "kb49": "kb49",
        "los-ii": "los-ii",
    }
    if normalized not in aliases:
        raise KeyError(normalized)
    return aliases[normalized]


def get_xdm_bj_params(
    functional_name: str,
    basis_name: str,
    *,
    cp: bool = False,
    model: str = "kb49",
) -> Tuple[float, float]:
    """Return (a1, a2_angstrom) for a functional/basis XDM model.

    Parameters
    ----------
    functional_name
        XC functional name (without ``-xdm`` suffix).
    basis_name
        Orbital basis name.
    cp
        If ``True``, use fitted ``/cp`` parameterization.
    model
        XDM damping-parameter model. Supported values are ``kb49`` and
        ``los-ii``.
    """

    func = functional_name.lower()
    basis = basis_name.lower()
    key = f"{func}/{basis}"
    normalized_model = normalize_xdm_model(model)

    if cp:
        key_cp = f"{key}/cp"
        if key_cp in _XDM_BJ_PARAMS_ANGSTROM:
            if normalized_model == "los-ii":
                return _XDM_LOS_II_PARAMS_ANGSTROM[key_cp]
            return _XDM_BJ_PARAMS_ANGSTROM[key_cp]

    if key in _XDM_BJ_PARAMS_ANGSTROM:
        if normalized_model == "los-ii":
            return _XDM_LOS_II_PARAMS_ANGSTROM[key]
        return _XDM_BJ_PARAMS_ANGSTROM[key]

    raise KeyError(key)
