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
    # Fitted to no-CP DFT IE (DFT_GRID_ORIENTATION=INPUT) against the KB49 reference;
    # least squares on per-dimer relative errors (RMSP), 49 dimers each. Pople-basis
    # entries are from a prior fit and carry no KB49 statistics.

    # ---- B3LYP ----
    # b3lyp/aug-cc-pvtz RMSP: 8.18%, MAE: 0.248 kcal/mol on KB49
    "b3lyp/aug-cc-pvtz": (0.6339, 1.5205),
    # b3lyp/aug-cc-pvdz RMSP: 11.96%, MAE: 0.313 kcal/mol on KB49
    "b3lyp/aug-cc-pvdz": (0.5294, 2.0268),
    # b3lyp/cc-pvtz RMSP: 24.00%, MAE: 0.497 kcal/mol on KB49
    "b3lyp/cc-pvtz": (0.0000, 3.6107),
    # b3lyp/cc-pvdz RMSP: 60.47%, MAE: 1.292 kcal/mol on KB49
    "b3lyp/cc-pvdz": (0.0000, 3.8589),
    # Pople bases (prior fit)
    "b3lyp/6-31+g*": (0.249120, 2.690583),
    "b3lyp/6-31+g**": (0.226145, 2.762253),
    "b3lyp/6-311+g(2d,2p)": (0.564665, 1.708375),

    # ---- B3P86 ----
    # b3p86/aug-cc-pvtz RMSP: 17.37%, MAE: 0.439 kcal/mol on KB49
    "b3p86/aug-cc-pvtz": (1.0076, 0.4279),
    # b3p86/aug-cc-pvdz RMSP: 15.79%, MAE: 0.410 kcal/mol on KB49
    "b3p86/aug-cc-pvdz": (1.2013, 0.0000),
    # b3p86/cc-pvtz RMSP: 20.93%, MAE: 0.510 kcal/mol on KB49
    "b3p86/cc-pvtz": (0.3856, 2.4239),
    # b3p86/cc-pvdz RMSP: 46.78%, MAE: 1.030 kcal/mol on KB49
    "b3p86/cc-pvdz": (0.0000, 3.8310),

    # ---- B3PW91 ----
    # b3pw91/aug-cc-pvtz RMSP: 19.16%, MAE: 0.424 kcal/mol on KB49
    "b3pw91/aug-cc-pvtz": (0.5852, 1.4540),
    # b3pw91/aug-cc-pvdz RMSP: 16.25%, MAE: 0.386 kcal/mol on KB49
    "b3pw91/aug-cc-pvdz": (1.1043, 0.0323),
    # b3pw91/cc-pvtz RMSP: 18.55%, MAE: 0.412 kcal/mol on KB49
    "b3pw91/cc-pvtz": (0.2926, 2.4368),
    # b3pw91/cc-pvdz RMSP: 43.02%, MAE: 0.850 kcal/mol on KB49
    "b3pw91/cc-pvdz": (0.0000, 3.5779),

    # ---- B86BPBE ----
    # b86bpbe/aug-cc-pvtz RMSP: 14.74%, MAE: 0.386 kcal/mol on KB49
    "b86bpbe/aug-cc-pvtz": (0.7452, 1.3878),
    # b86bpbe/aug-cc-pvdz RMSP: 18.22%, MAE: 0.461 kcal/mol on KB49
    "b86bpbe/aug-cc-pvdz": (0.6669, 1.8586),
    # b86bpbe/cc-pvtz RMSP: 29.81%, MAE: 0.647 kcal/mol on KB49
    "b86bpbe/cc-pvtz": (0.0000, 3.8535),
    # b86bpbe/cc-pvdz RMSP: 64.82%, MAE: 1.400 kcal/mol on KB49
    "b86bpbe/cc-pvdz": (0.0000, 4.1417),

    # ---- B97-1 ----
    # b97-1/aug-cc-pvtz RMSP: 17.08%, MAE: 0.432 kcal/mol on KB49
    "b97-1/aug-cc-pvtz": (0.2134, 3.5101),
    # b97-1/aug-cc-pvdz RMSP: 22.90%, MAE: 0.504 kcal/mol on KB49
    "b97-1/aug-cc-pvdz": (0.0000, 4.5536),
    # b97-1/cc-pvtz RMSP: 31.70%, MAE: 0.670 kcal/mol on KB49
    "b97-1/cc-pvtz": (0.0000, 4.3236),
    # b97-1/cc-pvdz RMSP: 61.85%, MAE: 1.332 kcal/mol on KB49
    "b97-1/cc-pvdz": (0.0000, 4.6313),

    # ---- BHandHLYP ----
    # bhandhlyp/aug-cc-pvtz RMSP: 8.65%, MAE: 0.309 kcal/mol on KB49
    "bhandhlyp/aug-cc-pvtz": (0.3924, 2.5277),
    # bhandhlyp/aug-cc-pvdz RMSP: 13.31%, MAE: 0.371 kcal/mol on KB49
    "bhandhlyp/aug-cc-pvdz": (0.1382, 3.5494),
    # bhandhlyp/cc-pvtz RMSP: 23.38%, MAE: 0.571 kcal/mol on KB49
    "bhandhlyp/cc-pvtz": (0.0000, 3.8599),
    # bhandhlyp/cc-pvdz RMSP: 55.40%, MAE: 1.236 kcal/mol on KB49
    "bhandhlyp/cc-pvdz": (0.0000, 4.0987),

    # ---- BLYP ----
    # blyp/aug-cc-pvtz RMSP: 11.83%, MAE: 0.243 kcal/mol on KB49
    "blyp/aug-cc-pvtz": (0.6130, 1.3105),
    # blyp/aug-cc-pvdz RMSP: 12.18%, MAE: 0.284 kcal/mol on KB49
    "blyp/aug-cc-pvdz": (0.6772, 1.2885),
    # blyp/cc-pvtz RMSP: 28.08%, MAE: 0.511 kcal/mol on KB49
    "blyp/cc-pvtz": (0.0000, 3.3546),
    # blyp/cc-pvdz RMSP: 68.68%, MAE: 1.386 kcal/mol on KB49
    "blyp/cc-pvdz": (0.0000, 3.6424),

    # ---- BP86 ----
    # bp86/aug-cc-pvtz RMSP: 25.05%, MAE: 0.522 kcal/mol on KB49
    "bp86/aug-cc-pvtz": (0.7387, 1.0014),
    # bp86/aug-cc-pvdz RMSP: 21.54%, MAE: 0.471 kcal/mol on KB49
    "bp86/aug-cc-pvdz": (1.1152, 0.0000),
    # bp86/cc-pvtz RMSP: 25.01%, MAE: 0.550 kcal/mol on KB49
    "bp86/cc-pvtz": (0.3401, 2.3142),
    # bp86/cc-pvdz RMSP: 51.54%, MAE: 1.072 kcal/mol on KB49
    "bp86/cc-pvdz": (0.0000, 3.5912),

    # ---- CAM-B3LYP ----
    # cam-b3lyp/aug-cc-pvtz RMSP: 9.10%, MAE: 0.358 kcal/mol on KB49
    "cam-b3lyp/aug-cc-pvtz": (0.3550, 2.7695),
    # cam-b3lyp/aug-cc-pvdz RMSP: 15.26%, MAE: 0.472 kcal/mol on KB49
    "cam-b3lyp/aug-cc-pvdz": (0.1743, 3.5935),
    # cam-b3lyp/cc-pvtz RMSP: 29.86%, MAE: 0.732 kcal/mol on KB49
    "cam-b3lyp/cc-pvtz": (0.0000, 4.0242),
    # cam-b3lyp/cc-pvdz RMSP: 67.15%, MAE: 1.555 kcal/mol on KB49
    "cam-b3lyp/cc-pvdz": (0.0000, 4.3310),

    # ---- HF ----
    # hf/aug-cc-pvtz RMSP: 16.75%, MAE: 0.444 kcal/mol on KB49
    "hf/aug-cc-pvtz": (0.5964, 1.5149),
    # hf/aug-cc-pvdz RMSP: 18.69%, MAE: 0.581 kcal/mol on KB49
    "hf/aug-cc-pvdz": (0.3937, 2.3362),
    # hf/cc-pvtz RMSP: 17.94%, MAE: 0.427 kcal/mol on KB49
    "hf/cc-pvtz": (0.2274, 2.7048),
    # hf/cc-pvdz RMSP: 33.02%, MAE: 0.518 kcal/mol on KB49
    "hf/cc-pvdz": (0.0000, 3.6151),

    # ---- HSE06 ----
    # hse06/aug-cc-pvtz RMSP: 14.32%, MAE: 0.439 kcal/mol on KB49
    "hse06/aug-cc-pvtz": (0.3803, 2.8371),
    # hse06/aug-cc-pvdz RMSP: 18.20%, MAE: 0.518 kcal/mol on KB49
    "hse06/aug-cc-pvdz": (0.1554, 3.8738),
    # hse06/cc-pvtz RMSP: 28.13%, MAE: 0.708 kcal/mol on KB49
    "hse06/cc-pvtz": (0.0000, 4.1504),
    # hse06/cc-pvdz RMSP: 58.65%, MAE: 1.372 kcal/mol on KB49
    "hse06/cc-pvdz": (0.0000, 4.4662),

    # ---- LC-wPBE ----
    # lc-wpbe/aug-cc-pvtz RMSP: 9.03%, MAE: 0.235 kcal/mol on KB49
    "lc-wpbe/aug-cc-pvtz": (1.0101, 0.6984),
    # lc-wpbe/aug-cc-pvdz RMSP: 11.31%, MAE: 0.280 kcal/mol on KB49
    "lc-wpbe/aug-cc-pvdz": (1.3343, 0.0000),
    # lc-wpbe/cc-pvtz RMSP: 18.09%, MAE: 0.316 kcal/mol on KB49
    "lc-wpbe/cc-pvtz": (0.0000, 3.9450),
    # lc-wpbe/cc-pvdz RMSP: 47.80%, MAE: 0.910 kcal/mol on KB49
    "lc-wpbe/cc-pvdz": (0.0000, 4.2676),

    # ---- PBE ----
    # pbe/aug-cc-pvtz RMSP: 17.91%, MAE: 0.490 kcal/mol on KB49
    "pbe/aug-cc-pvtz": (0.4326, 2.6285),
    # pbe/aug-cc-pvdz RMSP: 22.66%, MAE: 0.582 kcal/mol on KB49
    "pbe/aug-cc-pvdz": (0.2236, 3.5855),
    # pbe/cc-pvtz RMSP: 36.45%, MAE: 0.829 kcal/mol on KB49
    "pbe/cc-pvtz": (0.0000, 4.1576),
    # pbe/cc-pvdz RMSP: 72.35%, MAE: 1.616 kcal/mol on KB49
    "pbe/cc-pvdz": (0.0000, 4.5106),
    # Pople bases (prior fit)
    "pbe/6-31+g*": (0.699062, 1.710117),
    "pbe/6-31+g**": (0.704316, 1.709007),

    # ---- PBE0 ----
    # pbe0/aug-cc-pvtz RMSP: 13.36%, MAE: 0.394 kcal/mol on KB49
    "pbe0/aug-cc-pvtz": (0.4335, 2.6326),
    # pbe0/aug-cc-pvdz RMSP: 17.06%, MAE: 0.460 kcal/mol on KB49
    "pbe0/aug-cc-pvdz": (0.2308, 3.5971),
    # pbe0/cc-pvtz RMSP: 27.10%, MAE: 0.654 kcal/mol on KB49
    "pbe0/cc-pvtz": (0.0000, 4.1133),
    # pbe0/cc-pvdz RMSP: 56.91%, MAE: 1.303 kcal/mol on KB49
    "pbe0/cc-pvdz": (0.0000, 4.4230),
    # Pople bases (prior fit)
    "pbe0/6-31+g*": (0.000000, 3.942401),
    "pbe0/6-31+g**": (0.000000, 3.957889),

    # ---- PW86PBE ----
    # pw86pbe/aug-cc-pvtz RMSP: 14.91%, MAE: 0.394 kcal/mol on KB49
    "pw86pbe/aug-cc-pvtz": (0.8925, 1.0867),
    # pw86pbe/aug-cc-pvdz RMSP: 19.59%, MAE: 0.484 kcal/mol on KB49
    "pw86pbe/aug-cc-pvdz": (0.8200, 1.5582),
    # pw86pbe/cc-pvtz RMSP: 31.70%, MAE: 0.670 kcal/mol on KB49
    "pw86pbe/cc-pvtz": (0.0000, 4.0182),
    # pw86pbe/cc-pvdz RMSP: 68.84%, MAE: 1.481 kcal/mol on KB49
    "pw86pbe/cc-pvdz": (0.0000, 4.3216),

    # ---- TPSS ----
    # tpss/aug-cc-pvtz RMSP: 13.63%, MAE: 0.368 kcal/mol on KB49
    "tpss/aug-cc-pvtz": (0.5851, 1.7527),
    # tpss/aug-cc-pvdz RMSP: 16.55%, MAE: 0.426 kcal/mol on KB49
    "tpss/aug-cc-pvdz": (0.4797, 2.3152),
    # tpss/cc-pvtz RMSP: 25.55%, MAE: 0.580 kcal/mol on KB49
    "tpss/cc-pvtz": (0.0000, 3.6975),
    # tpss/cc-pvdz RMSP: 58.17%, MAE: 1.244 kcal/mol on KB49
    "tpss/cc-pvdz": (0.0000, 3.9895),
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
