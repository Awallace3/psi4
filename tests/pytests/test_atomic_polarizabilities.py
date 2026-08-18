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
import os
from pathlib import Path

import numpy as np
import pytest

import psi4
from psi4.driver.procrouting import atomic_polarizability as native_driver


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
# End-to-end publication (Task 7).
#
# The seven public array variables. Every one must appear, with the shape recorded in
# docs/superpowers/specs/2026-08-17-end-to-end-wiring.md, or none may appear at all.
# --------------------------------------------------------------------------------------

PUBLISHED_VARIABLES = (
    "ATOMIC POLARIZABILITIES",
    "ATOMIC DYNAMIC POLARIZABILITIES",
    "ATOMIC POLARIZABILITY FREQUENCIES",
    "ATOMIC C6",
    "ATOMIC C8",
    "ATOMIC C10",
    "ATOMIC C12",
)

PUBLISHED_SHAPES = {
    "ATOMIC POLARIZABILITIES": (3, 6),
    "ATOMIC DYNAMIC POLARIZABILITIES": (33, 6),
    "ATOMIC POLARIZABILITY FREQUENCIES": (11, 1),
    "ATOMIC C6": (3, 3),
    "ATOMIC C8": (3, 3),
    "ATOMIC C10": (3, 3),
    "ATOMIC C12": (3, 3),
}

# The reviewed geometry: C2 axis along z, molecule in the xz plane, O at the origin.
# `symmetry c1`/`no_com`/`no_reorient` is the reviewed protocol, and the PDef mask is
# derived geometrically, so C2v(Z) is still detected under the declared C1.
REVIEWED_GEOMETRY = """
0 1
O  0.00000000  0.0  0.00000000
H  1.45365196  0.0 -1.12168732
H -1.45365196  0.0 -1.12168732
symmetry c1
no_com
no_reorient
units bohr
"""

# Wiring verification protocol. Deliberately NOT the reviewed parity protocol: a smaller
# orbital basis, so its published numbers differ from the reviewed literals by basis and
# it gates shapes, symmetry, and fail-closed behaviour only, never the literals.
#
# The DFT grid is 590/99 rather than the wiring spec's 302/50, because the spec's grid
# table was measured without diffuse functions: with aug-cc-pVDZ the LW charge-sum residual
# sticks at 1.2e-05 on a 302/50 grid no matter how dense the ISA grid is, and only a 590/99
# grid brings it inside 1e-6. Densifying the ISA grid past 60/18/24 changes C6 by 4e-05
# relative here, so the DFT grid, not the ISA grid, is the binding constraint.
WIRING_PROTOCOL = {
    "basis": "aug-cc-pvdz",
    "scf_type": "pk",
    "dft_spherical_points": 590,
    "dft_radial_points": 99,
    "dft_density_tolerance": 1.0e-12,
    "atomic_polarizability_isa_radial_points": 60,
    "atomic_polarizability_isa_angular_polar_points": 18,
    "atomic_polarizability_isa_angular_azimuthal_points": 24,
    "atomic_polarizability_localization_tolerance": 1.0e-6,
}

# Fail-closed behaviour depends on no grid or basis quality at all, so its SCF triple is
# built as cheaply as possible.
FAIL_CLOSED_PROTOCOL = {
    "basis": "sto-3g",
    "scf_type": "pk",
    "dft_spherical_points": 50,
    "dft_radial_points": 12,
    "dft_density_tolerance": 1.0e-12,
}

# The reviewed parity protocol. Running it is Task 8, not Task 7; it is expensive, so the
# six literal comparisons below are skipped unless it is explicitly requested. They are
# reported as skipped rather than passed precisely so an unexercised comparison can never
# be mistaken for a satisfied one.
#
# Convergence is pinned rather than inherited. At this basis the cation UKS lands only just
# inside Psi4's 1e-6 defaults (final Delta E -3.08e-07, RMS |[F,P]| 8.92e-07), and the
# response provenance seal re-derives convergence from its own last observed iteration rather
# than trusting the SCF's verdict, so it refuses the state and the whole pipeline fails closed.
# Tightening the SCF gives the seal margin to work with.
#
# The localization tolerance is 1e-6, matching the wiring protocol, because that is what the
# grid actually delivers: the measured LW charge-sum residual here is 5.39e-07. A tighter 1e-8
# is not attainable at 590/99 and only made localize_lw fail its own postcondition.
PARITY_PROTOCOL = {
    "basis": "aug-cc-pvtz",
    "scf_type": "pk",
    "e_convergence": 1.0e-10,
    "d_convergence": 1.0e-9,
    "dft_spherical_points": 590,
    "dft_radial_points": 99,
    "dft_density_tolerance": 1.0e-12,
    "atomic_polarizability_isa_radial_points": 100,
    "atomic_polarizability_isa_angular_polar_points": 24,
    "atomic_polarizability_isa_angular_azimuthal_points": 32,
    "atomic_polarizability_localization_tolerance": 1.0e-6,
}

PARITY_SKIP_REASON = (
    "the reviewed aug-cc-pVTZ/GRAC parity protocol is Task 8; set "
    "PSI4_ATOMIC_POLARIZABILITY_PARITY=1 to exercise the reviewed-literal comparisons. "
    "Those literals come from a CamCASP run that partitions the FDDS by constrained density "
    "fitting, not ISA, so an ISA pipeline cannot satisfy them; see the comment above them"
)


def _published(wfn):
    """Collect the seven published arrays from a wavefunction as NumPy arrays."""
    return {name: np.asarray(wfn.array_variable(name)) for name in PUBLISHED_VARIABLES}


def _run_protocol(protocol):
    psi4.core.clean_variables()
    psi4.core.be_quiet()
    psi4.set_options(protocol)
    molecule = psi4.geometry(REVIEWED_GEOMETRY)
    wfn = native_driver.atomic_polarizabilities(molecule=molecule)
    return _published(wfn)


@pytest.fixture(scope="module")
def wiring_published():
    return _run_protocol(WIRING_PROTOCOL)


@pytest.fixture(scope="module")
def parity_published():
    if os.environ.get("PSI4_ATOMIC_POLARIZABILITY_PARITY") != "1":
        pytest.skip(PARITY_SKIP_REASON)
    return _run_protocol(PARITY_PROTOCOL)


@pytest.mark.scf
def test_published_shapes(wiring_published):
    for name, shape in PUBLISHED_SHAPES.items():
        assert wiring_published[name].shape == shape, name


@pytest.mark.scf
def test_published_values_are_finite(wiring_published):
    for name, array in wiring_published.items():
        assert np.all(np.isfinite(array)), name


@pytest.mark.scf
def test_published_static_block_is_the_zero_frequency_dynamic_block(wiring_published):
    static = wiring_published["ATOMIC POLARIZABILITIES"]
    dynamic = wiring_published["ATOMIC DYNAMIC POLARIZABILITIES"]
    np.testing.assert_allclose(static, dynamic[:3], rtol=0.0, atol=0.0)


@pytest.mark.scf
def test_published_frequencies_are_the_protocol_grid(wiring_published):
    frequencies = wiring_published["ATOMIC POLARIZABILITY FREQUENCIES"].ravel()
    expected, _weights = psi4.core._atomic_polarizability_make_casimir_grid(10, 0.5)
    np.testing.assert_allclose(
        frequencies,
        np.asarray(expected, dtype=float),
        rtol=FREQUENCY_RTOL,
        atol=FREQUENCY_ATOL,
    )


@pytest.mark.scf
def test_published_tensors_are_exactly_symmetric_at_every_frequency(wiring_published):
    """Every per-atom global Cartesian tensor is exactly symmetric.

    The packed form is symmetric by construction, so the substance is that publication
    happened at all: the packer rejects an asymmetric tensor rather than symmetrizing it,
    so a successful publication is the assertion. The round trip below additionally pins
    that the packed order really is xx, xy, xz, yy, yz, zz.
    """
    dynamic = wiring_published["ATOMIC DYNAMIC POLARIZABILITIES"]
    for row in range(dynamic.shape[0]):
        tensor = _unpack(dynamic[row])
        np.testing.assert_allclose(tensor, tensor.T, rtol=0.0, atol=0.0)
        repacked = psi4.core._atomic_polarizability_pack_symmetric_tensor(
            psi4.core.Matrix.from_array(tensor)
        )
        np.testing.assert_allclose(np.asarray(repacked), dynamic[row], rtol=0.0, atol=0.0)


@pytest.mark.scf
def test_published_oxygen_tensor_is_diagonal_at_every_frequency(wiring_published):
    """O sits on both mirror planes, so C2v forbids every off-diagonal Cartesian element.

    This is a prediction of the derived PDef mask rather than a packing identity: without
    the mask the under-determined L3 fit invents exactly this kind of anisotropy.
    """
    dynamic = wiring_published["ATOMIC DYNAMIC POLARIZABILITIES"]
    for block in range(11):
        _, xy, xz, _, yz, _ = dynamic[3 * block]
        assert abs(xy) <= TENSOR_ATOL, (block, xy)
        assert abs(xz) <= TENSOR_ATOL, (block, xz)
        assert abs(yz) <= TENSOR_ATOL, (block, yz)


@pytest.mark.scf
def test_published_output_satisfies_the_h2o_c2_relation(wiring_published):
    """alpha_H2 = S_x alpha_H1 S_x on published output, at every frequency.

    This is the observable consequence of the PDef symmetry mask and the H2-copies-H1
    equality rows, so it fails if the mask is derived in the wrong frame.
    """
    mirror = np.diag([-1.0, 1.0, 1.0])
    dynamic = wiring_published["ATOMIC DYNAMIC POLARIZABILITIES"]
    for block in range(11):
        h1 = _unpack(dynamic[3 * block + 1])
        h2 = _unpack(dynamic[3 * block + 2])
        np.testing.assert_allclose(h2, mirror @ h1 @ mirror, rtol=TENSOR_RTOL, atol=TENSOR_ATOL)


@pytest.mark.scf
def test_published_dispersion_is_symmetric_with_equivalent_hydrogens(wiring_published):
    for name in ("ATOMIC C6", "ATOMIC C8", "ATOMIC C10", "ATOMIC C12"):
        array = wiring_published[name]
        np.testing.assert_allclose(array, array.T, rtol=0.0, atol=0.0, err_msg=name)
        np.testing.assert_allclose(
            array[1], array[2], rtol=TENSOR_RTOL, atol=TENSOR_ATOL, err_msg=name
        )


# --------------------------------------------------------------------------------------
# Molecular-polarizability conservation.
#
# A distributed polarizability model must reproduce the molecular dipole polarizability
# when its site tensors are summed. Nothing else in this suite asserts that: the LW
# localization and WSM refinement stage tests each verify conservation only relative to
# their own input, and the published-output tests check symmetry, diagonality and decay,
# never magnitude. A uniform deficit therefore used to pass everything -- and did: the
# published sum once recovered 64 percent of Psi4's own molecular value (xx 0.77,
# yy 0.51, zz 0.64) because the WSM fit points sat inside the charge density where a
# rank-3 multipole model cannot represent the point-to-point response.
#
# Two independent references are used, because they fail for different reasons:
#
#  * Psi4's own coupled-perturbed molecular DIPOLE POLARIZABILITY at the identical
#    functional, basis and DFT grid. This is a genuinely external number, but it uses
#    PBE0's own xc kernel while the pipeline deliberately uses 25 percent CHF plus
#    75 percent ALDA, so a few percent of disagreement is expected by construction.
#  * The pipeline's own ISA-Pol site-pair response, contracted here -- outside the
#    pipeline -- with the exact molecular dipole operator. Summing the rank-1 site
#    blocks together with rank-0 times the site position is algebraically exact for any
#    partition of unity, so this reference is available at every frequency and isolates
#    the localization and refinement stages from the response kernel.
#
# Neither gate may be loosened to make a comparison pass.
# --------------------------------------------------------------------------------------

#: Largest fraction by which the summed distributed model may fall short of, or exceed,
#: the site-pair response it was derived from. The only legitimate sources of difference
#: are the rank-3 truncation of the site model and the finite fit grid; measured at
#: 0.5 percent (xx), 2.4 percent (yy) and 1.9 percent (zz) on this protocol.
RESPONSE_CONSERVATION_TOLERANCE = 0.05

#: Largest fraction by which the summed distributed model may differ from Psi4's own
#: molecular value. Wider than the gate above because the reference kernel differs from
#: PBE0's by construction; the site-pair response itself sits 3 percent below Psi4's
#: molecular value for that reason alone.
MOLECULAR_CONSERVATION_TOLERANCE = 0.15

#: Imaginary frequency used for the dynamic conservation check (grid index 5).
CONSERVATION_FREQUENCY_INDEX = 5

# Index of the rank-1 component in the 16-component 00/rank-1/rank-2/rank-3 site-pair
# block, per axis, and the matching Cartesian axis index.
_SITE_PAIR_DIPOLE_ROW = {"x": 2, "y": 3, "z": 1}
_CARTESIAN_AXIS = {"x": 0, "y": 1, "z": 2}


def _molecular_polarizability_from_site_pairs(response):
    """Contract a site-pair response block set into the molecular dipole polarizability.

    The site-pair blocks carry ranks 0 to 3 about each site. Translating every site's
    contribution to a common origin makes the total dipole operator
    ``sum_s [Q_s,1m + R_s,m Q_s,00]``, which equals the molecular dipole operator exactly
    whenever the site weights partition unity -- independently of the partition, the rank
    truncation or the basis. So this is an exact reconstruction, not an approximation.
    """
    positions = np.asarray(response["positions"])
    blocks = [np.asarray(block) for block in response["blocks"]]
    site_count = positions.shape[0]
    assert len(blocks) == site_count * site_count

    def operator(axis, site):
        vector = np.zeros(16)
        vector[_SITE_PAIR_DIPOLE_ROW[axis]] = 1.0
        vector[0] = positions[site][_CARTESIAN_AXIS[axis]]
        return vector

    alpha = np.zeros((3, 3))
    for first, first_axis in enumerate("xyz"):
        for second, second_axis in enumerate("xyz"):
            total = 0.0
            for source in range(site_count):
                left = operator(first_axis, source)
                for sink in range(site_count):
                    right = operator(second_axis, sink)
                    total += left @ blocks[source * site_count + sink] @ right
            alpha[first, second] = total
    return alpha


@pytest.fixture(scope="module")
def wiring_conservation():
    """Published output, the site-pair response behind it, and Psi4's molecular value.

    One SCF triple feeds all three so the comparison is at a single electronic state.
    """
    psi4.core.clean_variables()
    psi4.core.be_quiet()
    psi4.set_options(WIRING_PROTOCOL)
    molecule = psi4.geometry(REVIEWED_GEOMETRY)
    triple = native_driver.atomic_polarizability_scf_triple(molecule=molecule)

    published = _published(native_driver.publish_atomic_polarizabilities(*triple))
    frequencies = published["ATOMIC POLARIZABILITY FREQUENCIES"].ravel()
    imaginary = float(frequencies[CONSERVATION_FREQUENCY_INDEX])

    incoming_memory = psi4.core.get_memory()
    psi4.core.set_memory_bytes(int(native_driver.PIPELINE_MEMORY_BYTES), True)
    try:
        context = psi4.core._atomic_polarizability_make_frozen_response_context(*triple)
        provider = psi4.core._atomic_polarizability_make_native_response_provider(
            context,
            {
                "radial_points": WIRING_PROTOCOL["atomic_polarizability_isa_radial_points"],
                "angular_polar_points":
                    WIRING_PROTOCOL["atomic_polarizability_isa_angular_polar_points"],
                "angular_azimuthal_points":
                    WIRING_PROTOCOL["atomic_polarizability_isa_angular_azimuthal_points"],
            },
        )
        site_pairs = provider.compute([0.0, imaginary], [0.0, 1.0])
    finally:
        psi4.core.set_memory_bytes(int(incoming_memory), True)

    response_reference = {
        0: _molecular_polarizability_from_site_pairs(site_pairs[0]),
        CONSERVATION_FREQUENCY_INDEX: _molecular_polarizability_from_site_pairs(site_pairs[1]),
    }

    # Psi4's own molecular CPKS polarizability, uncorrected and unshifted, at the same
    # functional, basis and DFT grid. This is the external reference.
    psi4.core.clean_variables()
    psi4.set_options(dict(WIRING_PROTOCOL, dft_grac_shift=0.0, reference="rhf"))
    psi4.properties("pbe0", properties=["DIPOLE_POLARIZABILITIES"],
                    molecule=psi4.geometry(REVIEWED_GEOMETRY))
    molecular = np.array([
        psi4.variable("DIPOLE POLARIZABILITY XX"),
        psi4.variable("DIPOLE POLARIZABILITY YY"),
        psi4.variable("DIPOLE POLARIZABILITY ZZ"),
    ])
    return published, response_reference, molecular


def _summed_distributed_tensor(published, frequency_index):
    """Sum the published per-atom Cartesian tensors at one frequency."""
    dynamic = published["ATOMIC DYNAMIC POLARIZABILITIES"]
    site_count = published["ATOMIC POLARIZABILITIES"].shape[0]
    start = frequency_index * site_count
    packed = dynamic[start:start + site_count].sum(axis=0)
    return _unpack(packed)


@pytest.mark.scf
@pytest.mark.parametrize("frequency_index", [0, CONSERVATION_FREQUENCY_INDEX])
def test_published_atomic_sum_conserves_the_site_pair_response(
    wiring_conservation, frequency_index
):
    """The summed distributed model must reproduce the response it was derived from.

    This is the assertion the suite was missing. LW localization is an exact
    redistribution and WSM refinement is a refinement, not a rescaling, so the summed
    rank-1 site blocks must return the site-pair response contracted with the molecular
    dipole operator, to within the rank-3 truncation of the site model.
    """
    published, response_reference, _ = wiring_conservation
    summed = _summed_distributed_tensor(published, frequency_index)
    expected = response_reference[frequency_index]
    for axis in range(3):
        reference = expected[axis, axis]
        assert reference > 0.0, (frequency_index, axis, reference)
        deficit = abs(summed[axis, axis] - reference) / reference
        assert deficit <= RESPONSE_CONSERVATION_TOLERANCE, (
            f"frequency index {frequency_index}, axis {'xyz'[axis]}: distributed sum "
            f"{summed[axis, axis]:.6f} against site-pair response {reference:.6f} "
            f"({deficit:.1%} away)"
        )
    isotropic = np.trace(summed) / 3.0
    expected_isotropic = np.trace(expected) / 3.0
    deficit = abs(isotropic - expected_isotropic) / expected_isotropic
    assert deficit <= RESPONSE_CONSERVATION_TOLERANCE, (
        f"frequency index {frequency_index}: isotropic distributed sum {isotropic:.6f} "
        f"against site-pair response {expected_isotropic:.6f} ({deficit:.1%} away)"
    )


@pytest.mark.scf
def test_published_atomic_sum_conserves_psi4s_molecular_dipole_polarizability(
    wiring_conservation
):
    """The static summed model must reproduce Psi4's own molecular CPKS polarizability.

    Independent of the pipeline entirely: a separate coupled-perturbed calculation at the
    same functional, basis and DFT grid. The reference kernel differs from PBE0's by
    construction, which is why this gate is wider than the site-pair gate; it is still far
    tighter than the 36 percent deficit it exists to catch.
    """
    published, _, molecular = wiring_conservation
    summed = _summed_distributed_tensor(published, 0)
    for axis in range(3):
        deficit = abs(summed[axis, axis] - molecular[axis]) / molecular[axis]
        assert deficit <= MOLECULAR_CONSERVATION_TOLERANCE, (
            f"axis {'xyz'[axis]}: distributed sum {summed[axis, axis]:.6f} against Psi4's "
            f"molecular {molecular[axis]:.6f} ({deficit:.1%} away)"
        )
    isotropic = np.trace(summed) / 3.0
    deficit = abs(isotropic - molecular.mean()) / molecular.mean()
    assert deficit <= MOLECULAR_CONSERVATION_TOLERANCE, (
        f"isotropic distributed sum {isotropic:.6f} against Psi4's molecular "
        f"{molecular.mean():.6f} ({deficit:.1%} away)"
    )


# --------------------------------------------------------------------------------------
# Wiring preconditions that need no SCF. These pin the two things the wiring is easiest to
# get quietly wrong: the frame the PDef mask is expressed in, and the memory the WSM stage
# needs on the default fit grid.
# --------------------------------------------------------------------------------------


def test_pdef_mask_is_derived_geometrically_not_from_the_declared_point_group():
    """`symmetry c1` must not disable the mask: detection is geometric.

    The reviewed protocol declares C1 with no_com/no_reorient. If the mask were keyed off
    the declared point group it would silently go fully active (360 variables), the L3 fit
    would be under-determined, and it would invent hydrogen dipole anisotropy.
    """
    molecule = psi4.geometry(REVIEWED_GEOMETRY)
    molecule.update_geometry()
    assert molecule.point_group().symbol() == "c1"

    derived = psi4.core._atomic_polarizability_derive_pdef_constraints(molecule)
    assert derived["molecular_point_group"] == "C2v(Z)"
    assert derived["variable_count"] == 360
    assert derived["active_variable_count"] == 170
    assert derived["equality_row_count"] == 66
    assert derived["independent_variable_count"] == 104
    assert [site["point_group"] for site in derived["sites"]] == ["C2v(Z)", "Cs(Y)", "Cs(Y)"]
    assert [len(site["active_pairs"]) for site in derived["sites"]] == [38, 66, 66]
    # H2 is a symmetry copy of H1, not an independent fit.
    assert [site["symmetry_source"] for site in derived["sites"]] == [0, 1, 1]


def test_local_axes_move_the_mask_out_of_the_frame_refine_wsm_needs():
    """A non-identity local frame yields a mask indexed in that frame, not the molecular one.

    refine_wsm's harmonics are global, so it must be handed the empty-site_axes mask. This
    pins the hazard: the same molecule with rotated local axes produces a *different* mask,
    which would look plausible and be wrong.
    """
    molecule = psi4.geometry(REVIEWED_GEOMETRY)
    molecule.update_geometry()
    molecular = psi4.core._atomic_polarizability_derive_pdef_constraints(molecule)

    swap_xy = psi4.core.Matrix.from_array(
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    )
    rotated = psi4.core._atomic_polarizability_derive_pdef_constraints(
        molecule, [swap_xy, swap_xy, swap_xy]
    )
    assert list(rotated["active_variables"]) != list(molecular["active_variables"])


def test_default_fit_grid_needs_more_than_psi4s_default_memory():
    """The 329-point default grid does not fit the 500 MB default; the driver must raise it.

    The WSM design matrix carries one dense row per unordered fit-point pair and the stage
    gate reserves half of configured memory, so this is a hard architectural constraint,
    not a tuning preference.
    """
    molecule = psi4.geometry(REVIEWED_GEOMETRY)
    molecule.update_geometry()
    points = np.asarray(psi4.core._atomic_polarizability_wsm_fit_points(molecule)["points"])
    assert points.shape == (329, 3)

    active, constraints = 170, 66
    with pytest.raises(RuntimeError, match="exceeds half the reserved memory"):
        psi4.core._atomic_polarizability_plan_wsm_refinement(
            points.shape[0], 3, active, constraints, 500 * 1000 * 1000
        )
    plan = psi4.core._atomic_polarizability_plan_wsm_refinement(
        points.shape[0], 3, active, constraints, native_driver.PIPELINE_MEMORY_BYTES
    )
    assert plan["estimated_bytes"] > 500 * 1000 * 1000 // 2
    assert 2 * plan["estimated_bytes"] <= native_driver.PIPELINE_MEMORY_BYTES


@pytest.fixture(scope="module")
def fail_closed_triple():
    psi4.core.be_quiet()
    psi4.set_options(FAIL_CLOSED_PROTOCOL)
    molecule = psi4.geometry(REVIEWED_GEOMETRY)
    return list(native_driver.atomic_polarizability_scf_triple(molecule=molecule))


@pytest.mark.scf
@pytest.mark.parametrize("missing", [0, 1, 2])
def test_missing_wavefunction_fails_closed_and_publishes_nothing(fail_closed_triple, missing):
    """Any absent member of the SCF triple must raise and publish nothing at all."""
    psi4.core.clean_variables()
    triple = list(fail_closed_triple)
    reference = triple[0]
    triple[missing] = None

    with pytest.raises(RuntimeError, match="AtomicPolarizabilityPrerequisiteError"):
        native_driver.publish_atomic_polarizabilities(*triple)

    for name in PUBLISHED_VARIABLES:
        assert not psi4.core.has_variable(name), name
        assert not reference.has_array_variable(name), name


@pytest.mark.scf
def test_mismatched_basis_fails_closed_and_publishes_nothing(fail_closed_triple):
    """An SCF triple that does not share one basis must raise and publish nothing."""
    psi4.core.clean_variables()
    psi4.core.be_quiet()
    grac, precursor, _cation = fail_closed_triple
    molecule = psi4.geometry(REVIEWED_GEOMETRY)
    cation = molecule.clone()
    cation.set_molecular_charge(1)
    cation.set_multiplicity(2)
    cation.update_geometry()
    psi4.set_options({"basis": "6-31g", "scf_type": "pk", "reference": "uhf"})
    _, wrong_basis = psi4.energy("scf", molecule=cation, return_wfn=True)
    psi4.set_options({"reference": "rhf"})

    with pytest.raises(RuntimeError, match="AtomicPolarizabilityPrerequisiteError"):
        native_driver.publish_atomic_polarizabilities(grac, precursor, wrong_basis)

    for name in PUBLISHED_VARIABLES:
        assert not psi4.core.has_variable(name), name
        assert not grac.has_array_variable(name), name


@pytest.mark.scf
def test_bare_oeprop_on_a_single_wavefunction_fails_closed():
    """A bare OEProp call without the SCF triple must never publish partial output."""
    psi4.core.be_quiet()
    psi4.set_options({"basis": "sto-3g", "scf_type": "pk"})
    molecule = psi4.geometry(REVIEWED_GEOMETRY)
    _, wfn = psi4.energy("scf", molecule=molecule, return_wfn=True)

    properties = psi4.core.OEProp(wfn)
    properties.add("ATOMIC_POLARIZABILITIES")
    with pytest.raises(RuntimeError, match="AtomicPolarizabilityPrerequisiteError"):
        properties.compute()

    for name in PUBLISHED_VARIABLES:
        assert not wfn.has_array_variable(name), name


# --------------------------------------------------------------------------------------
# The six reviewed-literal comparisons. These are the Task 8 acceptance gate and are
# only meaningful under the reviewed protocol; see PARITY_SKIP_REASON.
#
# They cannot pass as things stand, for an understood reason that is not a defect in this
# code. The literals come from a CamCASP run whose distributed-polarizability algorithm is
# "DF : density-fitting-based partitioning of the FDDS", while this pipeline partitions the
# response by real-space ISA. Those are different models of the same molecular response:
# they agree on the total and disagree on how it is split between sites. Regenerating the
# reference with grid ISA, holding the wavefunction, auxiliary basis and fit points
# byte-identical, reproduces this pipeline to 2-5% on six of seven per-site dipole
# components and 1-10% per pair on C6, against errors up to a factor of 113 here.
#
# Measured against these literals: 11 of 18 static entries fall outside the gate, by up to
# 6e4 times the tolerance, and the Cn matrices by up to 9e4 times. Do not resolve that by
# loosening the gate or rewriting the literals -- they are a faithful record of a
# DF-partitioned calculation, and the plan's constraints forbid both. Either regenerate
# them from a matching partition or keep them as the DF oracle they are.
#
# See docs/superpowers/specs/2026-08-18-isa-grid-oracle.md.
# --------------------------------------------------------------------------------------


@pytest.mark.scf
def test_parity_static_polarizabilities_match_camcasp(parity_published):
    np.testing.assert_allclose(
        parity_published["ATOMIC POLARIZABILITIES"],
        np.asarray(REFERENCE_STATIC_POLARIZABILITIES),
        rtol=TENSOR_RTOL,
        atol=TENSOR_ATOL,
    )


@pytest.mark.scf
def test_parity_dynamic_polarizabilities_match_camcasp(parity_published):
    np.testing.assert_allclose(
        parity_published["ATOMIC DYNAMIC POLARIZABILITIES"],
        np.asarray(REFERENCE_DYNAMIC_POLARIZABILITIES),
        rtol=TENSOR_RTOL,
        atol=TENSOR_ATOL,
    )


@pytest.mark.scf
@pytest.mark.parametrize(
    "name,reference",
    [
        ("ATOMIC C6", REFERENCE_C6),
        ("ATOMIC C8", REFERENCE_C8),
        ("ATOMIC C10", REFERENCE_C10),
        ("ATOMIC C12", REFERENCE_C12),
    ],
)
def test_parity_dispersion_coefficients_match_camcasp(parity_published, name, reference):
    np.testing.assert_allclose(
        parity_published[name],
        np.asarray(reference),
        rtol=TENSOR_RTOL,
        atol=TENSOR_ATOL,
    )


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
