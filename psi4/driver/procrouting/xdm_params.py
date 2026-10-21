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

"""XDM Becke-Johnson damping parameters managed on Python side."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from ..p4util.exceptions import ValidationError

# "-xdm", "-xdm(kb49)", "-xdm(los-ii)" at the end of a functional name
_XDM_SUFFIX = re.compile(r"-xdm(?:\([^)]*\))?$", re.IGNORECASE)


def is_xdm_dispersion(dispersion: Any) -> bool:
    """Whether a functional dictionary's ``dispersion`` entry requests XDM."""

    return isinstance(dispersion, dict) and str(dispersion.get("type", "")).lower() == "xdm"


def has_xdm_suffix(method: str) -> bool:
    """Whether ``method`` names an XDM functional, e.g. ``b3lyp-xdm`` or ``b3lyp-xdm(los-ii)/sto-3g``."""

    return bool(_XDM_SUFFIX.search(method.split("/", 1)[0]))


def strip_xdm_suffix(name: str) -> str:
    """Return ``name`` without its ``-xdm`` / ``-xdm(<model>)`` suffix."""

    return _XDM_SUFFIX.sub("", name)


_XDM_BJ_PARAMS_ANGSTROM: Dict[str, Tuple[float, float]] = {
    # ---- B3LYP ----
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
    "hse06/aug-cc-pvtz": (0.5886, 2.0647),  # MAPD: 10.85%, MAE: 0.421 kcal/mol
    "hse06/aug-cc-pvdz": (0.4495, 2.7766),  # MAPD: 14.33%, MAE: 0.505 kcal/mol
    "hse06/cc-pvtz": (0.4717, 2.5112),  # MAPD: 19.51%, MAE: 0.701 kcal/mol
    "hse06/cc-pvdz": (0.1149, 3.8501),  # MAPD: 36.89%, MAE: 1.319 kcal/mol

    # ---- LC-wPBE ----
    "lc-wpbe/aug-cc-pvtz": (1.0292, 0.5824),  # MAPD: 9.01%, MAE: 0.269 kcal/mol
    "lc-wpbe/aug-cc-pvdz": (1.3046, 0.0000),  # MAPD: 8.74%, MAE: 0.276 kcal/mol
    "lc-wpbe/cc-pvtz": (0.5280, 2.1846),  # MAPD: 11.24%, MAE: 0.316 kcal/mol
    "lc-wpbe/cc-pvdz": (0.2790, 3.2148),  # MAPD: 26.01%, MAE: 0.880 kcal/mol

    # ---- PBE ----
    # noCP (KB49 RMSP fit)
    "pbe/aug-cc-pvtz": (0.7110, 1.6463),  # MAPD: 13.12%, MAE: 0.446 kcal/mol
    "pbe/aug-cc-pvdz": (0.5556, 2.3610),  # MAPD: 17.93%, MAE: 0.544 kcal/mol
    "pbe/cc-pvtz": (0.4041, 2.6998),  # MAPD: 24.16%, MAE: 0.784 kcal/mol
    "pbe/cc-pvdz": (0.0320, 4.0862),  # MAPD: 44.82%, MAE: 1.532 kcal/mol
    # noCP Pople (prior fitting)
    "pbe/6-31+g*": (0.699062, 1.710117),
    "pbe/6-31+g**": (0.704316, 1.709007),

    # ---- PBE0 ----
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

    # b3lyp/aug-cc-pvdz MAE: 0.199973 kcal/mol on LoS-II
    "b3lyp/aug-cc-pvdz": (0.362043, 2.452093),
    # b3lyp/aug-cc-pvtz MAE: 0.136507 kcal/mol on LoS-II
    "b3lyp/aug-cc-pvtz": (0.344500, 2.304241),
}

_XDM_LOS_II_CP_PARAMS_ANGSTROM: Dict[str, Tuple[float, float]] = {
    # Fitted to counterpoise-corrected DFT IE with ghost-centre moment
    # superposition (bsse_type="cp") against the same LoS-II reference;
    # least squares on per-dimer residuals, 4556 (b3lyp) / 4558 (pbe0) dimers.

    # b3lyp/aug-cc-pvdz RMSE 0.262, MAE: 0.144 kcal/mol on LoS-II
    "b3lyp/aug-cc-pvdz": (0.3880, 2.1781),
    # pbe0/aug-cc-pvdz RMSE 0.347, MAE: 0.179 kcal/mol on LoS-II
    "pbe0/aug-cc-pvdz": (0.1592, 3.3865),
}

_XDM_KB49_CP_PARAMS_ANGSTROM: Dict[str, Tuple[float, float]] = {
    # Fitted to counterpoise-corrected DFT IE with ghost-centre moment superposition
    # (bsse_type="cp") against the KB49 reference; least squares on per-dimer relative
    # errors (RMSP), 49 dimers each.

    # ---- B3LYP ----
    # b3lyp/aug-cc-pvtz RMSP: 9.24%, MAE: 0.260 kcal/mol on KB49
    "b3lyp/aug-cc-pvtz": (0.6568, 1.3771),
    # b3lyp/aug-cc-pvdz RMSP: 9.09%, MAE: 0.254 kcal/mol on KB49
    "b3lyp/aug-cc-pvdz": (0.5915, 1.5540),
    # b3lyp/cc-pvtz RMSP: 10.52%, MAE: 0.290 kcal/mol on KB49
    "b3lyp/cc-pvtz": (0.6819, 1.2902),
    # b3lyp/cc-pvdz RMSP: 14.68%, MAE: 0.378 kcal/mol on KB49
    "b3lyp/cc-pvdz": (0.6541, 1.2844),

    # ---- B3P86 ----
    # b3p86/aug-cc-pvtz RMSP: 15.71%, MAE: 0.402 kcal/mol on KB49
    "b3p86/aug-cc-pvtz": (1.1158, 0.0000),
    # b3p86/aug-cc-pvdz RMSP: 16.11%, MAE: 0.392 kcal/mol on KB49
    "b3p86/aug-cc-pvdz": (1.1073, 0.0000),
    # b3p86/cc-pvtz RMSP: 15.97%, MAE: 0.404 kcal/mol on KB49
    "b3p86/cc-pvtz": (1.1201, 0.0000),
    # b3p86/cc-pvdz RMSP: 18.75%, MAE: 0.466 kcal/mol on KB49
    "b3p86/cc-pvdz": (1.0972, 0.0000),

    # ---- B3PW91 ----
    # b3pw91/aug-cc-pvtz RMSP: 17.83%, MAE: 0.404 kcal/mol on KB49
    "b3pw91/aug-cc-pvtz": (1.0341, 0.0000),
    # b3pw91/aug-cc-pvdz RMSP: 17.70%, MAE: 0.406 kcal/mol on KB49
    "b3pw91/aug-cc-pvdz": (1.0284, 0.0000),
    # b3pw91/cc-pvtz RMSP: 17.59%, MAE: 0.403 kcal/mol on KB49
    "b3pw91/cc-pvtz": (1.0357, 0.0000),
    # b3pw91/cc-pvdz RMSP: 18.89%, MAE: 0.468 kcal/mol on KB49
    "b3pw91/cc-pvdz": (1.0195, 0.0000),

    # ---- B86BPBE ----
    # b86bpbe/aug-cc-pvtz RMSP: 14.82%, MAE: 0.387 kcal/mol on KB49
    "b86bpbe/aug-cc-pvtz": (0.8202, 1.0699),
    # b86bpbe/aug-cc-pvdz RMSP: 14.45%, MAE: 0.382 kcal/mol on KB49
    "b86bpbe/aug-cc-pvdz": (0.8289, 1.0214),
    # b86bpbe/cc-pvtz RMSP: 16.13%, MAE: 0.416 kcal/mol on KB49
    "b86bpbe/cc-pvtz": (0.8571, 0.9527),
    # b86bpbe/cc-pvdz RMSP: 19.35%, MAE: 0.504 kcal/mol on KB49
    "b86bpbe/cc-pvdz": (0.8861, 0.7862),

    # ---- B97-1 ----
    # b97-1/aug-cc-pvtz RMSP: 16.97%, MAE: 0.443 kcal/mol on KB49
    "b97-1/aug-cc-pvtz": (0.1733, 3.5698),
    # b97-1/aug-cc-pvdz RMSP: 17.10%, MAE: 0.459 kcal/mol on KB49
    "b97-1/aug-cc-pvdz": (0.1732, 3.5597),
    # b97-1/cc-pvtz RMSP: 18.85%, MAE: 0.463 kcal/mol on KB49
    "b97-1/cc-pvtz": (0.3327, 3.0255),
    # b97-1/cc-pvdz RMSP: 20.61%, MAE: 0.516 kcal/mol on KB49
    "b97-1/cc-pvdz": (0.5250, 2.3053),

    # ---- BHandHLYP ----
    # bhandhlyp/aug-cc-pvtz RMSP: 8.61%, MAE: 0.288 kcal/mol on KB49
    "bhandhlyp/aug-cc-pvtz": (0.2902, 2.7884),
    # bhandhlyp/aug-cc-pvdz RMSP: 8.53%, MAE: 0.282 kcal/mol on KB49
    "bhandhlyp/aug-cc-pvdz": (0.2191, 3.0018),
    # bhandhlyp/cc-pvtz RMSP: 10.17%, MAE: 0.296 kcal/mol on KB49
    "bhandhlyp/cc-pvtz": (0.3067, 2.7149),
    # bhandhlyp/cc-pvdz RMSP: 13.06%, MAE: 0.337 kcal/mol on KB49
    "bhandhlyp/cc-pvdz": (0.3321, 2.5480),

    # ---- BLYP ----
    # blyp/aug-cc-pvtz RMSP: 13.88%, MAE: 0.330 kcal/mol on KB49
    "blyp/aug-cc-pvtz": (0.7677, 0.7472),
    # blyp/aug-cc-pvdz RMSP: 14.11%, MAE: 0.343 kcal/mol on KB49
    "blyp/aug-cc-pvdz": (0.6860, 0.9671),
    # blyp/cc-pvtz RMSP: 15.06%, MAE: 0.377 kcal/mol on KB49
    "blyp/cc-pvtz": (0.8118, 0.6000),
    # blyp/cc-pvdz RMSP: 20.22%, MAE: 0.535 kcal/mol on KB49
    "blyp/cc-pvdz": (0.7639, 0.6602),

    # ---- BP86 ----
    # bp86/aug-cc-pvtz RMSP: 22.72%, MAE: 0.494 kcal/mol on KB49
    "bp86/aug-cc-pvtz": (1.0324, 0.0000),
    # bp86/aug-cc-pvdz RMSP: 23.20%, MAE: 0.489 kcal/mol on KB49
    "bp86/aug-cc-pvdz": (1.0232, 0.0000),
    # bp86/cc-pvtz RMSP: 23.09%, MAE: 0.491 kcal/mol on KB49
    "bp86/cc-pvtz": (1.0343, 0.0000),
    # bp86/cc-pvdz RMSP: 25.86%, MAE: 0.571 kcal/mol on KB49
    "bp86/cc-pvdz": (1.0065, 0.0000),

    # ---- CAM-B3LYP ----
    # cam-b3lyp/aug-cc-pvtz RMSP: 8.70%, MAE: 0.331 kcal/mol on KB49
    "cam-b3lyp/aug-cc-pvtz": (0.2880, 2.9175),
    # cam-b3lyp/aug-cc-pvdz RMSP: 8.62%, MAE: 0.322 kcal/mol on KB49
    "cam-b3lyp/aug-cc-pvdz": (0.2712, 2.9595),
    # cam-b3lyp/cc-pvtz RMSP: 10.07%, MAE: 0.329 kcal/mol on KB49
    "cam-b3lyp/cc-pvtz": (0.3130, 2.8140),
    # cam-b3lyp/cc-pvdz RMSP: 13.09%, MAE: 0.376 kcal/mol on KB49
    "cam-b3lyp/cc-pvdz": (0.4263, 2.3465),

    # ---- HF ----
    # hf/aug-cc-pvtz RMSP: 17.96%, MAE: 0.486 kcal/mol on KB49
    "hf/aug-cc-pvtz": (0.3400, 2.2291),
    # hf/aug-cc-pvdz RMSP: 18.11%, MAE: 0.503 kcal/mol on KB49
    "hf/aug-cc-pvdz": (0.3572, 2.1785),
    # hf/cc-pvtz RMSP: 19.29%, MAE: 0.519 kcal/mol on KB49
    "hf/cc-pvtz": (0.4473, 1.8743),
    # hf/cc-pvdz RMSP: 19.95%, MAE: 0.511 kcal/mol on KB49
    "hf/cc-pvdz": (0.6722, 1.1393),

    # ---- HSE06 ----
    # hse06/aug-cc-pvtz RMSP: 14.38%, MAE: 0.430 kcal/mol on KB49
    "hse06/aug-cc-pvtz": (0.4365, 2.5872),
    # hse06/aug-cc-pvdz RMSP: 13.96%, MAE: 0.413 kcal/mol on KB49
    "hse06/aug-cc-pvdz": (0.4503, 2.5495),
    # hse06/cc-pvtz RMSP: 15.81%, MAE: 0.437 kcal/mol on KB49
    "hse06/cc-pvtz": (0.5115, 2.3306),
    # hse06/cc-pvdz RMSP: 18.28%, MAE: 0.482 kcal/mol on KB49
    "hse06/cc-pvdz": (0.6545, 1.8252),

    # ---- LC-wPBE ----
    # lc-wpbe/aug-cc-pvtz RMSP: 8.68%, MAE: 0.228 kcal/mol on KB49
    "lc-wpbe/aug-cc-pvtz": (1.0635, 0.4477),
    # lc-wpbe/aug-cc-pvdz RMSP: 9.65%, MAE: 0.238 kcal/mol on KB49
    "lc-wpbe/aug-cc-pvdz": (1.2114, 0.0000),
    # lc-wpbe/cc-pvtz RMSP: 8.69%, MAE: 0.224 kcal/mol on KB49
    "lc-wpbe/cc-pvtz": (1.0880, 0.3783),
    # lc-wpbe/cc-pvdz RMSP: 11.77%, MAE: 0.304 kcal/mol on KB49
    "lc-wpbe/cc-pvdz": (1.2034, 0.0000),

    # ---- PBE ----
    # pbe/aug-cc-pvtz RMSP: 17.94%, MAE: 0.492 kcal/mol on KB49
    "pbe/aug-cc-pvtz": (0.4310, 2.5633),
    # pbe/aug-cc-pvdz RMSP: 17.63%, MAE: 0.486 kcal/mol on KB49
    "pbe/aug-cc-pvdz": (0.3917, 2.6789),
    # pbe/cc-pvtz RMSP: 19.56%, MAE: 0.508 kcal/mol on KB49
    "pbe/cc-pvtz": (0.5101, 2.2838),
    # pbe/cc-pvdz RMSP: 22.49%, MAE: 0.582 kcal/mol on KB49
    "pbe/cc-pvdz": (0.5292, 2.1268),

    # ---- PBE0 ----
    # pbe0/aug-cc-pvtz RMSP: 13.53%, MAE: 0.388 kcal/mol on KB49
    "pbe0/aug-cc-pvtz": (0.4952, 2.3693),
    # pbe0/aug-cc-pvdz RMSP: 13.07%, MAE: 0.365 kcal/mol on KB49
    "pbe0/aug-cc-pvdz": (0.5170, 2.3020),
    # pbe0/cc-pvtz RMSP: 14.98%, MAE: 0.399 kcal/mol on KB49
    "pbe0/cc-pvtz": (0.5565, 2.1585),
    # pbe0/cc-pvdz RMSP: 17.28%, MAE: 0.441 kcal/mol on KB49
    "pbe0/cc-pvdz": (0.7092, 1.6276),

    # ---- PW86PBE ----
    # pw86pbe/aug-cc-pvtz RMSP: 14.89%, MAE: 0.399 kcal/mol on KB49
    "pw86pbe/aug-cc-pvtz": (0.9122, 0.9244),
    # pw86pbe/aug-cc-pvdz RMSP: 14.75%, MAE: 0.393 kcal/mol on KB49
    "pw86pbe/aug-cc-pvdz": (0.9444, 0.8007),
    # pw86pbe/cc-pvtz RMSP: 16.42%, MAE: 0.417 kcal/mol on KB49
    "pw86pbe/cc-pvtz": (0.9360, 0.8543),
    # pw86pbe/cc-pvdz RMSP: 19.84%, MAE: 0.503 kcal/mol on KB49
    "pw86pbe/cc-pvdz": (0.9550, 0.6942),

    # ---- TPSS ----
    # tpss/aug-cc-pvtz RMSP: 14.10%, MAE: 0.378 kcal/mol on KB49
    "tpss/aug-cc-pvtz": (0.6512, 1.4752),
    # tpss/aug-cc-pvdz RMSP: 13.28%, MAE: 0.362 kcal/mol on KB49
    "tpss/aug-cc-pvdz": (0.6762, 1.3937),
    # tpss/cc-pvtz RMSP: 15.01%, MAE: 0.402 kcal/mol on KB49
    "tpss/cc-pvtz": (0.7458, 1.1681),
    # tpss/cc-pvdz RMSP: 17.84%, MAE: 0.474 kcal/mol on KB49
    "tpss/cc-pvdz": (0.8590, 0.7829),
}

_XDM_MODEL_ALIASES: Dict[str, str] = {
    "": "kb49",
    "kb49": "kb49",
    "los-ii": "los-ii",
}

# Each model holds separate fits for plain (no-CP) and counterpoise energies; the
# counterpoise table is used for every task of a bsse_type="cp"/"vmfc" computation.
_XDM_PARAM_TABLES: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {
    "kb49": {
        "nocp": _XDM_BJ_PARAMS_ANGSTROM,
        "cp": _XDM_KB49_CP_PARAMS_ANGSTROM
    },
    "los-ii": {
        "nocp": _XDM_LOS_II_PARAMS_ANGSTROM,
        "cp": _XDM_LOS_II_CP_PARAMS_ANGSTROM
    },
}


def _bsse_label(counterpoise: bool) -> str:
    return "cp" if counterpoise else "nocp"


def normalize_xdm_model(model: str) -> str:
    """Normalize XDM damping model names to canonical internal labels."""

    normalized = str(model).strip().lower()
    if normalized not in _XDM_MODEL_ALIASES:
        supported = sorted(m for m in _XDM_MODEL_ALIASES if m)
        raise ValidationError(f"XDM: Unrecognized damping model ({model}). Supported models are ({supported}).")
    return _XDM_MODEL_ALIASES[normalized]


def available_xdm_functionals(model: str = "kb49") -> List[str]:
    """Functional names with fitted XDM damping parameters (CP or no-CP) under ``model``."""

    tables = _XDM_PARAM_TABLES[normalize_xdm_model(model)]
    return sorted({key.split("/", 1)[0] for table in tables.values() for key in table})


def available_xdm_bases(functional_name: str, model: str = "kb49", *, counterpoise: bool = False) -> List[str]:
    """Basis names with fitted (a1, a2) for ``functional_name`` under ``model``."""

    func = functional_name.lower()
    table = _XDM_PARAM_TABLES[normalize_xdm_model(model)][_bsse_label(counterpoise)]
    return sorted(key.split("/", 1)[1] for key in table if key.split("/", 1)[0] == func)


def get_xdm_bj_params(
    functional_name: str,
    basis_name: str,
    *,
    model: str = "kb49",
    counterpoise: bool = False,
) -> Tuple[float, float]:
    """Return (a1, a2_angstrom) for a functional/basis XDM model.

    Parameters
    ----------
    functional_name
        XC functional name (without ``-xdm`` suffix).
    basis_name
        Orbital basis name.
    model
        XDM damping-parameter model. Supported values are ``kb49`` and
        ``los-ii``.
    counterpoise
        Select the parameters fitted to counterpoise-corrected interaction
        energies instead of the no-CP fit.
    """

    key = f"{functional_name.lower()}/{basis_name.lower()}"
    table = _XDM_PARAM_TABLES[normalize_xdm_model(model)][_bsse_label(counterpoise)]
    if key in table:
        return table[key]

    raise KeyError(key)
