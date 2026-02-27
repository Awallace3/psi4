"""XDM Becke-Johnson damping parameters managed on Python side."""

from __future__ import annotations

from typing import Dict, Tuple


_XDM_BJ_PARAMS_ANGSTROM: Dict[str, Tuple[float, float]] = {
    # ---- B3LYP ----
    # CP (prior fitting)
    "b3lyp/aug-cc-pvtz/cp": (0.538965, 1.707159),
    "b3lyp/aug-cc-pvdz/cp": (0.541310, 1.707761),
    "b3lyp/cc-pvtz/cp": (0.533051, 1.708554),
    "b3lyp/cc-pvdz/cp": (0.502835, 1.705159),
    "b3lyp/6-31+g*/cp": (0.539265, 1.706567),
    "b3lyp/6-31+g**/cp": (0.544086, 1.707329),
    "b3lyp/6-311+g(2d,2p)/cp": (0.541432, 1.706280),
    # noCP (KB49 RMSP fit)
    "b3lyp/aug-cc-pvtz": (0.5049, 1.8215),  # MAPD: 6.94%, MAE: 0.225 kcal/mol
    "b3lyp/aug-cc-pvdz": (0.7259, 1.3140),  # MAPD: 8.15%, MAE: 0.273 kcal/mol
    "b3lyp/cc-pvtz": (0.5397, 1.8048),  # MAPD: 15.64%, MAE: 0.516 kcal/mol
    "b3lyp/cc-pvdz": (0.3934, 2.4519),  # MAPD: 38.69%, MAE: 1.293 kcal/mol
    # noCP Pople (prior fitting)
    "b3lyp/6-31+g*": (0.249120, 2.690583),
    "b3lyp/6-31+g**": (0.226145, 2.762253),
    "b3lyp/6-311+g(2d,2p)": (0.564665, 1.708375),

    # ---- B3P86 ----
    "b3p86/aug-cc-pvtz": (1.0391, 0.2422),  # MAPD: 13.65%, MAE: 0.486 kcal/mol
    "b3p86/aug-cc-pvdz": (1.1622, 0.0000),  # MAPD: 12.62%, MAE: 0.468 kcal/mol
    "b3p86/cc-pvtz": (0.6134, 1.6075),  # MAPD: 14.21%, MAE: 0.533 kcal/mol
    "b3p86/cc-pvdz": (0.5413, 1.9809),  # MAPD: 31.17%, MAE: 1.120 kcal/mol

    # ---- B3PW91 ----
    "b3pw91/aug-cc-pvtz": (0.2745, 2.2471),  # MAPD: 15.58%, MAE: 0.471 kcal/mol
    "b3pw91/aug-cc-pvdz": (1.0810, 0.0000),  # MAPD: 14.28%, MAE: 0.487 kcal/mol
    "b3pw91/cc-pvtz": (0.5767, 1.4443),  # MAPD: 13.20%, MAE: 0.459 kcal/mol
    "b3pw91/cc-pvdz": (0.5415, 1.7270),  # MAPD: 28.85%, MAE: 0.986 kcal/mol

    # ---- B86BPBE ----
    "b86bpbe/aug-cc-pvtz": (0.9003, 0.8168),  # MAPD: 10.85%, MAE: 0.355 kcal/mol
    "b86bpbe/aug-cc-pvdz": (0.9428, 0.8603),  # MAPD: 13.60%, MAE: 0.412 kcal/mol
    "b86bpbe/cc-pvtz": (0.5183, 2.0817),  # MAPD: 20.02%, MAE: 0.626 kcal/mol
    "b86bpbe/cc-pvdz": (0.2617, 3.0913),  # MAPD: 41.18%, MAE: 1.368 kcal/mol

    # ---- B97-1 ----
    "b97-1/aug-cc-pvtz": (0.4762, 2.5787),  # MAPD: 11.46%, MAE: 0.383 kcal/mol
    "b97-1/aug-cc-pvdz": (0.2667, 3.5271),  # MAPD: 16.19%, MAE: 0.459 kcal/mol
    "b97-1/cc-pvtz": (0.3583, 3.0287),  # MAPD: 19.86%, MAE: 0.620 kcal/mol
    "b97-1/cc-pvdz": (0.0000, 4.3169),  # MAPD: 37.59%, MAE: 1.248 kcal/mol

    # ---- BHandHLYP ----
    "bhandhlyp/aug-cc-pvtz": (0.2823, 2.7786),  # MAPD: 6.85%, MAE: 0.276 kcal/mol
    "bhandhlyp/aug-cc-pvdz": (0.2731, 3.0268),  # MAPD: 8.88%, MAE: 0.343 kcal/mol
    "bhandhlyp/cc-pvtz": (0.5132, 2.1560),  # MAPD: 16.31%, MAE: 0.585 kcal/mol
    "bhandhlyp/cc-pvdz": (0.2962, 3.0123),  # MAPD: 34.10%, MAE: 1.205 kcal/mol

    # ---- BLYP ----
    "blyp/aug-cc-pvtz": (0.5769, 1.3424),  # MAPD: 10.56%, MAE: 0.303 kcal/mol
    "blyp/aug-cc-pvdz": (0.8980, 0.5261),  # MAPD: 10.24%, MAE: 0.312 kcal/mol
    "blyp/cc-pvtz": (0.5381, 1.5455),  # MAPD: 19.77%, MAE: 0.563 kcal/mol
    "blyp/cc-pvdz": (0.4409, 2.0840),  # MAPD: 45.42%, MAE: 1.443 kcal/mol

    # ---- BP86 ----
    "bp86/aug-cc-pvtz": (1.0461, 0.0196),  # MAPD: 19.30%, MAE: 0.634 kcal/mol
    "bp86/aug-cc-pvdz": (1.0865, 0.0000),  # MAPD: 18.26%, MAE: 0.620 kcal/mol
    "bp86/cc-pvtz": (0.5913, 1.4541),  # MAPD: 18.13%, MAE: 0.609 kcal/mol
    "bp86/cc-pvdz": (0.5526, 1.7320),  # MAPD: 36.18%, MAE: 1.223 kcal/mol

    # ---- CAM-B3LYP ----
    "cam-b3lyp/aug-cc-pvtz": (0.3892, 2.5727),  # MAPD: 7.16%, MAE: 0.329 kcal/mol
    "cam-b3lyp/aug-cc-pvdz": (0.4083, 2.7387),  # MAPD: 11.66%, MAE: 0.447 kcal/mol
    "cam-b3lyp/cc-pvtz": (0.4410, 2.5056),  # MAPD: 19.49%, MAE: 0.711 kcal/mol
    "cam-b3lyp/cc-pvdz": (0.1203, 3.7350),  # MAPD: 41.51%, MAE: 1.498 kcal/mol

    # ---- HF ----
    "hf/aug-cc-pvtz": (0.3456, 2.0087),  # MAPD: 13.88%, MAE: 0.472 kcal/mol
    "hf/aug-cc-pvdz": (0.4302, 1.9813),  # MAPD: 13.89%, MAE: 0.538 kcal/mol
    "hf/cc-pvtz": (0.5442, 1.4730),  # MAPD: 13.46%, MAE: 0.424 kcal/mol
    "hf/cc-pvdz": (0.5217, 1.7061),  # MAPD: 20.51%, MAE: 0.583 kcal/mol

    # ---- HSE06 ----
    "hse06/aug-cc-pvtz": (0.5987, 2.0371),  # MAPD: 10.98%, MAE: 0.425 kcal/mol (48/49 valid)
    "hse06/aug-cc-pvdz": (0.4495, 2.7766),  # MAPD: 14.33%, MAE: 0.505 kcal/mol
    "hse06/cc-pvtz": (0.4717, 2.5112),  # MAPD: 19.51%, MAE: 0.701 kcal/mol
    "hse06/cc-pvdz": (0.1149, 3.8501),  # MAPD: 36.89%, MAE: 1.319 kcal/mol

    # ---- LC-wPBE ----
    "lc-wpbe/aug-cc-pvtz": (1.0292, 0.5824),  # MAPD: 9.01%, MAE: 0.269 kcal/mol
    "lc-wpbe/aug-cc-pvdz": (1.3046, 0.0000),  # MAPD: 8.74%, MAE: 0.276 kcal/mol
    "lc-wpbe/cc-pvtz": (0.5280, 2.1846),  # MAPD: 11.24%, MAE: 0.316 kcal/mol
    "lc-wpbe/cc-pvdz": (0.2790, 3.2148),  # MAPD: 26.01%, MAE: 0.880 kcal/mol

    # ---- PBE ----
    # CP (prior fitting)
    "pbe/aug-cc-pvtz/cp": (0.641172, 1.707118),
    "pbe/aug-cc-pvdz/cp": (0.646009, 1.706796),
    "pbe/6-31+g*/cp": (0.638224, 1.705838),
    "pbe/6-31+g**/cp": (0.644670, 1.707056),
    # noCP (KB49 RMSP fit)
    "pbe/aug-cc-pvtz": (0.7110, 1.6463),  # MAPD: 13.12%, MAE: 0.446 kcal/mol
    "pbe/aug-cc-pvdz": (0.5556, 2.3610),  # MAPD: 17.93%, MAE: 0.544 kcal/mol
    "pbe/cc-pvtz": (0.4041, 2.6998),  # MAPD: 24.16%, MAE: 0.784 kcal/mol
    "pbe/cc-pvdz": (0.0320, 4.0862),  # MAPD: 44.82%, MAE: 1.532 kcal/mol
    # noCP Pople (prior fitting)
    "pbe/6-31+g*": (0.699062, 1.710117),
    "pbe/6-31+g**": (0.704316, 1.709007),

    # ---- PBE0 ----
    # CP (prior fitting)
    "pbe0/aug-cc-pvdz/cp": (0.675533, 1.709020),
    "pbe0/cc-pvtz/cp": (0.662432, 1.709395),
    "pbe0/cc-pvdz/cp": (0.648251, 1.707199),
    "pbe0/6-31+g*/cp": (0.665768, 1.708279),
    "pbe0/6-31+g**/cp": (0.672756, 1.709480),
    # noCP (KB49 RMSP fit)
    "pbe0/aug-cc-pvtz": (0.5884, 2.0642),  # MAPD: 9.89%, MAE: 0.368 kcal/mol
    "pbe0/aug-cc-pvdz": (0.4681, 2.7183),  # MAPD: 13.08%, MAE: 0.442 kcal/mol
    "pbe0/cc-pvtz": (0.4823, 2.4796),  # MAPD: 18.49%, MAE: 0.645 kcal/mol
    "pbe0/cc-pvdz": (0.1392, 3.7712),  # MAPD: 35.34%, MAE: 1.246 kcal/mol
    # noCP Pople (prior fitting)
    "pbe0/6-31+g*": (0.000000, 3.942401),
    "pbe0/6-31+g**": (0.000000, 3.957889),

    # ---- PW86PBE ----
    "pw86pbe/aug-cc-pvtz": (1.1094, 0.3190),  # MAPD: 10.89%, MAE: 0.359 kcal/mol
    "pw86pbe/aug-cc-pvdz": (1.1333, 0.4261),  # MAPD: 14.76%, MAE: 0.433 kcal/mol
    "pw86pbe/cc-pvtz": (0.5056, 2.2769),  # MAPD: 21.24%, MAE: 0.654 kcal/mol
    "pw86pbe/cc-pvdz": (0.1616, 3.5581),  # MAPD: 43.61%, MAE: 1.437 kcal/mol

    # ---- TPSS ----
    "tpss/aug-cc-pvtz": (0.6362, 1.4995),  # MAPD: 10.19%, MAE: 0.337 kcal/mol
    "tpss/aug-cc-pvdz": (0.6643, 1.6121),  # MAPD: 12.93%, MAE: 0.400 kcal/mol
    "tpss/cc-pvtz": (0.5212, 1.9238),  # MAPD: 16.74%, MAE: 0.556 kcal/mol
    "tpss/cc-pvdz": (0.3357, 2.7252),  # MAPD: 35.99%, MAE: 1.210 kcal/mol
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
