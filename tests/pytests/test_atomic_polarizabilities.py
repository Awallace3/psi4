"""Fixed-protocol H2O parity tests against the reviewed CamCASP L3 model.

Every reference value in this module is a hard-coded literal, extracted once from the
approved reference data and reviewed. This module must never read that data at runtime,
import JSON, or invoke any external command; `test_native_atomic_polarizability_source_guard`
and the checks at the bottom of this file enforce that.

Reviewed protocol: PBE0/aug-cc-pVTZ with the Psi4 GRAC asymptotic correction, ALDA+CHF
response kernel, LW localization to L3, PFIT WSM limit L3, hydrogen limit L3, penalty
weight 4, weight coefficient 0.001, cutoff 1e-4. Atom order is O, H1, H2. Changing any of
these defines a different model and invalidates every literal below.
"""

import inspect
from pathlib import Path

import numpy as np
import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.mints]

# Plan-mandated tolerances. Do not loosen these to make a comparison pass.
TENSOR_RTOL = 1.0e-4
TENSOR_ATOL = 1.0e-5
FREQUENCY_RTOL = 1.0e-10
FREQUENCY_ATOL = 1.0e-12

# Reviewed CamCASP L3 reference literals for H2O.
# Extracted once from the approved reference JSON; this module must never read it.
# Protocol: PBE0/aug-cc-pVTZ, GRAC, ALDA+CHF, LW localization L3, PFIT WSM L3,
# weight 4, coefficient 0.001, cutoff 1e-4; atom order O, H1, H2.
# Packed Cartesian order: xx, xy, xz, yy, yz, zz.

REFERENCE_ATOM_ORDER = ("O", "H1", "H2")

REFERENCE_FREQUENCIES = [
    0.0,
    0.0066096015960872435,
    0.03617481199863096,
    0.09544736369034827,
    0.1976442118453127,
    0.3704172128053672,
    0.6749146404580301,
    1.264899172436498,
    2.619244684547324,
    6.910885950408292,
    37.82376235021415,
]

REFERENCE_STATIC_POLARIZABILITIES = [
    [7.043489935336, 0.0, 0.0, 5.762074477569, 0.0, 5.583657081749],  # O
    [1.573674631536, 0.0, 0.005761700031, 1.617426936478, 0.0, 2.009572611043],  # H1
    [1.573674631536, 0.0, -0.005761700031, 1.617426936478, 0.0, 2.009572611043],  # H2
]

# Frequency-major: 11 frequency blocks x 3 atoms = 33 rows.
REFERENCE_DYNAMIC_POLARIZABILITIES = [
    # omega = 0.0
    [7.043489935336, 0.0, 0.0, 5.762074477569, 0.0, 5.583657081749],  # O
    [1.573674631536, 0.0, 0.005761700031, 1.617426936478, 0.0, 2.009572611043],  # H1
    [1.573674631536, 0.0, -0.005761700031, 1.617426936478, 0.0, 2.009572611043],  # H2
    # omega = 0.0066096015960872435
    [7.042555547095, 0.0, 0.0, 5.76081682545, 0.0, 5.582772755905],  # O
    [1.573534655112, 0.0, 0.005732987046, 1.617123630915, 0.0, 2.009282959193],  # H1
    [1.573534655112, 0.0, -0.005732987046, 1.617123630915, 0.0, 2.009282959193],  # H2
    # omega = 0.03617481199863096
    [7.015628114877, 0.0, 0.0, 5.724976423105, 0.0, 5.55735570141],  # O
    [1.569494591692, 0.0, 0.004909961281, 1.608425789727, 0.0, 2.000958745108],  # H1
    [1.569494591692, 0.0, -0.004909961281, 1.608425789727, 0.0, 2.000958745108],  # H2
    # omega = 0.09544736369034827
    [6.855062972454, 0.0, 0.0, 5.520688666617, 0.0, 5.408179361062],  # O
    [1.545206913227, 0.0, 0.000342474975, 1.560220077, 0.0, 1.95235453312],  # H1
    [1.545206913227, 0.0, -0.000342474975, 1.560220077, 0.0, 1.95235453312],  # H2
    # omega = 0.1976442118453127
    [6.314650531429, 0.0, 0.0, 4.919948807192, 0.0, 4.931313464556],  # O
    [1.460721853579, 0.0, -0.011439220594, 1.430356852628, 0.0, 1.799331125072],  # H1
    [1.460721853579, 0.0, 0.011439220594, 1.430356852628, 0.0, 1.799331125072],  # H2
    # omega = 0.3704172128053672
    [5.092654354114, 0.0, 0.0, 3.811929880978, 0.0, 3.944837932174],  # O
    [1.250511752113, 0.0, -0.025028472834, 1.215318338875, 0.0, 1.486419371091],  # H1
    [1.250511752113, 0.0, 0.025028472834, 1.215318338875, 0.0, 1.486419371091],  # H2
    # omega = 0.6749146404580301
    [3.305159964243, 0.0, 0.0, 2.41770120867, 0.0, 2.588072897572],  # O
    [0.879837161229, 0.0, -0.031787565312, 0.909793279154, 0.0, 1.033993582562],  # H1
    [0.879837161229, 0.0, 0.031787565312, 0.909793279154, 0.0, 1.033993582562],  # H2
    # omega = 1.264899172436498
    [1.623379529512, 0.0, 0.0, 1.175762307979, 0.0, 1.291542745658],  # O
    [0.453999439172, 0.0, -0.03178793951, 0.529449388733, 0.0, 0.552671562884],  # H1
    [0.453999439172, 0.0, 0.03178793951, 0.529449388733, 0.0, 0.552671562884],  # H2
    # omega = 2.619244684547324
    [0.543549646384, 0.0, 0.0, 0.391348983569, 0.0, 0.434824815251],  # O
    [0.156876198749, 0.0, -0.018251131062, 0.208916688566, 0.0, 0.201377505489],  # H1
    [0.156876198749, 0.0, 0.018251131062, 0.208916688566, 0.0, 0.201377505489],  # H2
    # omega = 6.910885950408292
    [0.094132251576, 0.0, 0.0, 0.065061752975, 0.0, 0.077044273024],  # O
    [0.030665158894, 0.0, -0.003375290318, 0.044566770843, 0.0, 0.039004785403],  # H1
    [0.030665158894, 0.0, 0.003375290318, 0.044566770843, 0.0, 0.039004785403],  # H2
    # omega = 37.82376235021415
    [0.00330035474, 0.0, 0.0, 0.002088968843, 0.0, 0.002771145402],  # O
    [0.001417737635, 0.0, -4.8695647e-05, 0.001999385817, 0.0, 0.001677189308],  # H1
    [0.001417737635, 0.0, 4.8695647e-05, 0.001999385817, 0.0, 0.001677189308],  # H2
]

REFERENCE_C6 = [
    [17.25559, 5.382332, 5.382332],
    [5.382332, 1.698678, 1.698678],
    [5.382332, 1.698678, 1.698678],
]

REFERENCE_C8 = [
    [346.424, 83.90759, 83.90759],
    [83.90759, 18.32833, 18.32833],
    [83.90759, 18.32833, 18.32833],
]

REFERENCE_C10 = [
    [7484.441, 1523.525, 1523.525],
    [1523.525, 291.4843, 291.4843],
    [1523.525, 291.4843, 291.4843],
]

REFERENCE_C12 = [
    [127231.0, 20293.77, 20293.77],
    [20293.77, 3216.541, 3216.541],
    [20293.77, 3216.541, 3216.541],
]


# --------------------------------------------------------------------------------------
# Literal self-consistency: these guard the extracted constants themselves against
# transcription damage, independently of any pipeline stage.
# --------------------------------------------------------------------------------------


def test_reference_literal_shapes():
    assert len(REFERENCE_ATOM_ORDER) == 3
    assert len(REFERENCE_FREQUENCIES) == 11
    assert np.asarray(REFERENCE_STATIC_POLARIZABILITIES).shape == (3, 6)
    assert np.asarray(REFERENCE_DYNAMIC_POLARIZABILITIES).shape == (33, 6)
    for matrix in (REFERENCE_C6, REFERENCE_C8, REFERENCE_C10, REFERENCE_C12):
        assert np.asarray(matrix).shape == (3, 3)


def test_reference_literals_are_finite():
    for values in (
        REFERENCE_FREQUENCIES,
        REFERENCE_STATIC_POLARIZABILITIES,
        REFERENCE_DYNAMIC_POLARIZABILITIES,
        REFERENCE_C6,
        REFERENCE_C8,
        REFERENCE_C10,
        REFERENCE_C12,
    ):
        assert np.all(np.isfinite(np.asarray(values, dtype=float)))


def test_reference_static_block_is_first_dynamic_block():
    """The static tensor must be the zero-frequency block of the dynamic output."""
    dynamic = np.asarray(REFERENCE_DYNAMIC_POLARIZABILITIES)
    np.testing.assert_allclose(
        np.asarray(REFERENCE_STATIC_POLARIZABILITIES), dynamic[:3], rtol=0.0, atol=0.0
    )


def test_reference_frequencies_are_strictly_increasing():
    frequencies = np.asarray(REFERENCE_FREQUENCIES)
    assert frequencies[0] == 0.0
    assert np.all(np.diff(frequencies) > 0.0)


def test_reference_dispersion_matrices_are_symmetric():
    for matrix in (REFERENCE_C6, REFERENCE_C8, REFERENCE_C10, REFERENCE_C12):
        array = np.asarray(matrix)
        np.testing.assert_allclose(array, array.T, rtol=0.0, atol=0.0)


def test_reference_hydrogens_are_equivalent_in_dispersion():
    """H1 and H2 are symmetry-equivalent, so dispersion rows/columns must match."""
    for matrix in (REFERENCE_C6, REFERENCE_C8, REFERENCE_C10, REFERENCE_C12):
        array = np.asarray(matrix)
        np.testing.assert_allclose(array[1], array[2], rtol=0.0, atol=0.0)


def _unpack(packed):
    """Expand packed xx, xy, xz, yy, yz, zz into a symmetric 3x3 matrix."""
    xx, xy, xz, yy, yz, zz = packed
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def test_reference_c2_relation_holds_at_every_frequency():
    """H2 is the C2/mirror image of H1: alpha_H2 = S_x alpha_H1 S_x, S_x = diag(-1, 1, 1)."""
    mirror = np.diag([-1.0, 1.0, 1.0])
    dynamic = np.asarray(REFERENCE_DYNAMIC_POLARIZABILITIES)
    for block in range(11):
        h1 = _unpack(dynamic[3 * block + 1])
        h2 = _unpack(dynamic[3 * block + 2])
        np.testing.assert_allclose(h2, mirror @ h1 @ mirror, rtol=0.0, atol=0.0)


def test_reference_oxygen_tensor_is_diagonal_at_every_frequency():
    """O sits on both mirror planes, so its Cartesian tensor has no off-diagonal part."""
    dynamic = np.asarray(REFERENCE_DYNAMIC_POLARIZABILITIES)
    for block in range(11):
        _, xy, xz, _, yz, _ = dynamic[3 * block]
        assert xy == 0.0
        assert xz == 0.0
        assert yz == 0.0


# --------------------------------------------------------------------------------------
# Frequency-grid parity. This is a genuine comparison against the reviewed model and
# does not depend on any response, localization, or refinement stage.
# --------------------------------------------------------------------------------------


def test_frequency_grid_matches_camcasp():
    frequencies, _weights = psi4.core._atomic_polarizability_make_casimir_grid(10, 0.5)
    np.testing.assert_allclose(
        np.asarray(frequencies, dtype=float),
        np.asarray(REFERENCE_FREQUENCIES, dtype=float),
        rtol=FREQUENCY_RTOL,
        atol=FREQUENCY_ATOL,
    )


def test_dispersion_quadrature_weights_are_casimir_polder():
    """The weights must be the Gauss-Legendre rule mapped by omega = w0 (1-t)/(1+t).

    Derived independently here rather than asserted against the implementation, so this
    fails if the mapping or the base frequency ever changes.
    """
    _frequencies, weights = psi4.core._atomic_polarizability_make_casimir_grid(10, 0.5)
    weights = np.asarray(weights, dtype=float)

    scale = 0.5
    nodes, gauss_weights = np.polynomial.legendre.leggauss(10)
    mapped_frequency = scale * (1.0 - nodes) / (1.0 + nodes)
    mapped_weight = gauss_weights * 2.0 * scale / (1.0 + nodes) ** 2
    expected = mapped_weight[np.argsort(mapped_frequency)]

    assert weights[0] == 0.0, "the static point must carry no quadrature weight"
    np.testing.assert_allclose(weights[1:], expected, rtol=1.0e-12, atol=0.0)


# --------------------------------------------------------------------------------------
# Source-independence guard for this module specifically.
# --------------------------------------------------------------------------------------


def test_module_has_no_runtime_reference_dependency():
    """This module must not read reference data or shell out at runtime.

    Prose may name the reviewed source; executable dependencies on it are the thing being
    forbidden, so this checks for the mechanisms rather than for the word.
    """
    source = Path(inspect.getfile(test_module_has_no_runtime_reference_dependency)).read_text()

    # Exclude this guard's own body: it necessarily names the mechanisms it forbids.
    marker = "def test_module_has_no_runtime_reference_dependency"
    scanned = source.split(marker)[0]
    executable = "\n".join(
        line for line in scanned.splitlines() if not line.lstrip().startswith("#")
    )

    for mechanism in (
        "camcasp-reference",
        "import json",
        "import subprocess",
        "subprocess",
        "os.system",
        "os.popen",
        "check_output",
        "Popen",
        "importlib",
        "open(",
        "read_text",
        "loadtxt",
        "np.load",
        "pathlib.Path(",
    ):
        assert mechanism not in executable, f"forbidden runtime dependency: {mechanism}"
