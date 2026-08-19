"""Pure-evaluator tests for the constrained density-fitting partition core.

``auxiliary_multipole_moments`` computes the analytic Racah regular real
solid-harmonic moments ``Q_t[k]`` of every auxiliary basis function about its
assigned site. It is a caller-supplied-data pure evaluator with no wavefunction,
no grid and no process options.

The moment tests carry two independent oracles: closed-form literals recorded in
the resolved-open-questions research, and a product Gauss-Hermite quadrature
written here from the published solid-harmonic definitions. Gauss-Hermite is
*exact* for a polynomial times a Gaussian, so the quadrature is an oracle rather
than an approximation.
"""

import math
import uuid

import numpy as np
import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.mints]


_COMPONENT_LABELS = (
    "00",
    "10",
    "11c",
    "11s",
    "20",
    "21c",
    "21s",
    "22c",
    "22s",
    "30",
    "31c",
    "31s",
    "32c",
    "32s",
    "33c",
    "33s",
)

_COMPONENT_INDEX = {label: index for index, label in enumerate(_COMPONENT_LABELS)}


def _regular_harmonics(x, y, z):
    """Racah-normalised regular real solid harmonics, ranks 0 through 3.

    Written from the published definitions, in the pipeline's component order
    ``00; 10 11c 11s; 20 21c 21s 22c 22s; 30 31c 31s 32c 32s 33c 33s``. Note the
    rank-1 block is ordered ``z, x, y``.
    """
    r2 = x * x + y * y + z * z
    return np.stack(
        [
            np.ones_like(x),
            z,
            x,
            y,
            (3.0 * z * z - r2) / 2.0,
            math.sqrt(3.0) * x * z,
            math.sqrt(3.0) * y * z,
            math.sqrt(3.0) * (x * x - y * y) / 2.0,
            math.sqrt(3.0) * x * y,
            (5.0 * z * z * z - 3.0 * z * r2) / 2.0,
            math.sqrt(3.0 / 8.0) * x * (5.0 * z * z - r2),
            math.sqrt(3.0 / 8.0) * y * (5.0 * z * z - r2),
            math.sqrt(15.0) * z * (x * x - y * y) / 2.0,
            math.sqrt(15.0) * x * y * z,
            math.sqrt(10.0) * x * (x * x - 3.0 * y * y) / 4.0,
            math.sqrt(10.0) * y * (3.0 * x * x - y * y) / 4.0,
        ],
        axis=-1,
    )


def _cartesian_powers(am):
    """Psi4's Cartesian ordering within a shell: a descending, then b descending."""
    return [(a, b, am - a - b) for a in range(am, -1, -1) for b in range(am - a, -1, -1)]


def _double_factorial(n):
    result = 1.0
    while n > 1:
        result *= n
        n -= 2
    return result


def _gaussian_moment_1d(power, exponent):
    """``integral x**power exp(-exponent x**2) dx`` over the whole line."""
    if power % 2 == 1:
        return 0.0
    return (
        _double_factorial(power - 1)
        * (2.0 * exponent) ** (-power / 2.0)
        * math.sqrt(math.pi / exponent)
    )


def _psi4_cartesian_shell_normalisation(am, exponent):
    """Psi4 normalises a shell so its ``x**l`` component has unit self-overlap."""
    self_overlap = (
        _gaussian_moment_1d(2 * am, 2.0 * exponent)
        * _gaussian_moment_1d(0, 2.0 * exponent) ** 2
    )
    return 1.0 / math.sqrt(self_overlap)


def _numeric_moments(shell_parameters, powers, shell_center, site, nodes=14):
    """Product Gauss-Hermite quadrature of ``chi_k(r) R_t(r - site)``.

    Exact for a polynomial times a Gaussian, so this is an independent oracle and
    not an approximation. ``shell_parameters`` is a sequence of
    ``(exponent, coefficient)`` primitive pairs.
    """
    a, b, c = powers
    total = np.zeros(16)
    for exponent, coefficient in shell_parameters:
        raw, weights = np.polynomial.hermite.hermgauss(nodes)
        scale = 1.0 / math.sqrt(exponent)
        u = raw * scale
        w = weights * scale
        ux, uy, uz = np.meshgrid(u, u, u, indexing="ij")
        wx, wy, wz = np.meshgrid(w, w, w, indexing="ij")
        amplitude = coefficient * (ux**a) * (uy**b) * (uz**c) * wx * wy * wz
        harmonics = _regular_harmonics(
            shell_center[0] + ux - site[0],
            shell_center[1] + uy - site[1],
            shell_center[2] + uz - site[2],
        )
        total += np.einsum("ijk,ijkt->t", amplitude, harmonics)
    return total


def _single_shell_basis(shell_letter, exponent, centers, puream=0):
    """Build a one-shell-per-atom basis with the given angular momentum."""
    label = "CDFAUX" + uuid.uuid4().hex[:8].upper()
    geometry = "\n".join(
        "He {:.17g} {:.17g} {:.17g}".format(*center) for center in centers
    )
    molecule = psi4.geometry(
        f"""
0 1
{geometry}
units bohr
no_reorient
no_com
symmetry c1
"""
    )
    psi4.basis_helper(
        f"""
assign {label}
[{label}]
He     0
{shell_letter}   1   1.00
      {exponent:.10f}       1.0000000
****
""",
        name=label,
        key="BASIS",
        set_option=False,
    )
    basis = psi4.core.BasisSet.build(molecule, "BASIS", label, puream=puream)
    return molecule, basis


def _moments(basis, sites, function_to_site):
    return np.array(
        psi4.core._atomic_polarizability_test_auxiliary_multipole_moments(
            basis, psi4.core.Matrix.from_array(np.asarray(sites, dtype=float)),
            list(function_to_site),
        )["moments"]
    )


# --------------------------------------------------------------------------
# A1 -- auxiliary_multipole_moments
# --------------------------------------------------------------------------

# Closed-form moments of the *unnormalised* Cartesian primitive
# ``x**a y**b z**c exp(-alpha r**2)`` centred on its own site, recorded in the
# resolved-open-questions research and validated there against independent
# quadrature to 1.6e-12. Psi4 folds a shell normalisation into
# GaussianShell::coef, so a test against these literals must divide it out.
_PRIMITIVE_MOMENT_LITERALS = (
    ("S", 0.7, (0, 0, 0), {"00": 9.507749897101e00}),
    (
        "D",
        0.55,
        (2, 0, 0),
        {
            "00": 1.241046601525e01,
            "20": -1.128224183205e01,
            "22c": 1.954141607639e01,
        },
    ),
    (
        "F",
        0.6,
        (3, 0, 0),
        {
            "11c": 2.496069629392e01,
            "31c": -2.547540397695e01,
            "33c": 3.288860511355e01,
        },
    ),
    (
        "G",
        0.5,
        (4, 0, 0),
        {
            "00": 4.724882983717e01,
            "20": -9.449765967433e01,
            "22c": 1.636747477523e02,
        },
    ),
    ("G", 0.95, (2, 1, 1), {"21s": 1.518585419043e00}),
)


def test_cartesian_shell_normalisation_model_matches_psi4s_own_overlap():
    """Guard the normalisation used to scale the closed-form literals."""
    for letter, am, exponent in (("S", 0, 0.7), ("D", 2, 0.55), ("F", 3, 0.6), ("G", 4, 0.5)):
        _, basis = _single_shell_basis(letter, exponent, [(0.0, 0.0, 0.0)])
        assert basis.has_puream() is False
        overlap = np.array(psi4.core.MintsHelper(basis).ao_overlap())
        expected = np.array(
            [
                _double_factorial(2 * a - 1)
                * _double_factorial(2 * b - 1)
                * _double_factorial(2 * c - 1)
                / _double_factorial(2 * am - 1)
                for (a, b, c) in _cartesian_powers(am)
            ]
        )
        assert np.allclose(np.diag(overlap), expected, rtol=0.0, atol=1.0e-13)
        assert math.isclose(
            basis.shell(0).coef(0),
            _psi4_cartesian_shell_normalisation(am, exponent),
            rel_tol=1.0e-13,
        )


@pytest.mark.parametrize(
    "letter,exponent,powers,literals", _PRIMITIVE_MOMENT_LITERALS,
    ids=[f"{entry[0]}{entry[2]}" for entry in _PRIMITIVE_MOMENT_LITERALS],
)
def test_auxiliary_multipole_moments_match_the_analytic_closed_form(
    letter, exponent, powers, literals
):
    """Centred Cartesian primitives against the recorded closed-form literals.

    The D and G cases are the Cartesian-contaminant tests: a pure d shell carries
    no rank-0 charge, a Cartesian one does. They fail for a spherical auxiliary
    basis and so catch a Cartesian/spherical mix-up.
    """
    _, basis = _single_shell_basis(letter, exponent, [(0.0, 0.0, 0.0)])
    am = basis.shell(0).am
    normalisation = basis.shell(0).coef(0)
    component = _cartesian_powers(am).index(powers)
    moments = _moments(basis, [(0.0, 0.0, 0.0)], [0] * basis.nbf())

    expected = np.zeros(16)
    for label, value in literals.items():
        expected[_COMPONENT_INDEX[label]] = normalisation * value
    assert np.allclose(moments[component], expected, rtol=1.0e-10, atol=1.0e-12)


def test_auxiliary_multipole_moments_match_independent_quadrature_for_every_shell():
    """Every Cartesian component of s through g against exact Gauss-Hermite."""
    for letter, exponent in (("S", 0.7), ("P", 1.2), ("D", 0.55), ("F", 0.6), ("G", 0.95)):
        _, basis = _single_shell_basis(letter, exponent, [(0.0, 0.0, 0.0)])
        am = basis.shell(0).am
        parameters = [(basis.shell(0).exp(0), basis.shell(0).coef(0))]
        moments = _moments(basis, [(0.0, 0.0, 0.0)], [0] * basis.nbf())
        for component, powers in enumerate(_cartesian_powers(am)):
            reference = _numeric_moments(
                parameters, powers, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
            )
            assert np.allclose(moments[component], reference, rtol=1.0e-9, atol=1.0e-11)


def test_rank_zero_auxiliary_moment_is_the_function_charge():
    """``Q_00[k]`` must be ``integral chi_k dr``, independent of the site used."""
    for letter, exponent in (("S", 0.7), ("P", 1.2), ("D", 0.55), ("F", 0.6), ("G", 0.95)):
        _, basis = _single_shell_basis(letter, exponent, [(0.0, 0.0, 0.0)])
        am = basis.shell(0).am
        alpha = basis.shell(0).exp(0)
        coefficient = basis.shell(0).coef(0)
        moments = _moments(basis, [(0.4, -0.7, 0.9)], [0] * basis.nbf())
        for component, (a, b, c) in enumerate(_cartesian_powers(am)):
            charge = (
                coefficient
                * _gaussian_moment_1d(a, alpha)
                * _gaussian_moment_1d(b, alpha)
                * _gaussian_moment_1d(c, alpha)
            )
            assert math.isclose(moments[component, 0], charge, rel_tol=1.0e-12, abs_tol=1.0e-13)


def test_cartesian_d_and_g_shells_carry_rank_zero_charge_but_pure_shells_do_not():
    """The structural fact that makes the Cartesian auxiliary basis different."""
    _, cartesian = _single_shell_basis("D", 0.55, [(0.0, 0.0, 0.0)], puream=0)
    moments = _moments(cartesian, [(0.0, 0.0, 0.0)], [0] * cartesian.nbf())
    charged = np.flatnonzero(np.abs(moments[:, 0]) > 1.0e-12)
    # xx, yy and zz of a Cartesian d shell each carry charge; xy, xz, yz do not.
    assert charged.tolist() == [0, 3, 5]

    _, spherical = _single_shell_basis("D", 0.55, [(0.0, 0.0, 0.0)], puream=1)
    assert spherical.has_puream() is True
    with pytest.raises(RuntimeError, match=r"Cartesian"):
        _moments(spherical, [(0.0, 0.0, 0.0)], [0] * spherical.nbf())


def test_offsite_s_moments_obey_the_mean_value_identity_and_rotate_covariantly():
    """An s Gaussian at P has moments ``q * R_t(P - site)`` exactly.

    ``R_t`` is harmonic, so the spherical mean-value property makes this exact.
    Rotating the displacement must therefore rotate the moments: the norm of each
    rank block is a rotational invariant.
    """
    exponent = 0.83
    displacement = np.array([0.61, -1.27, 0.44])
    angle = 0.7
    axis = np.array([1.0, 2.0, -0.5])
    axis = axis / np.linalg.norm(axis)
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    rotation = (
        np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)
    )
    rotated = rotation @ displacement

    blocks = []
    for point in (displacement, rotated):
        _, basis = _single_shell_basis("S", exponent, [tuple(point)])
        charge = basis.shell(0).coef(0) * (math.pi / exponent) ** 1.5
        moments = _moments(basis, [(0.0, 0.0, 0.0)], [0])
        expected = charge * _regular_harmonics(*point)
        assert np.allclose(moments[0], expected, rtol=1.0e-11, atol=1.0e-13)
        blocks.append(moments[0])

    for start, stop in ((0, 1), (1, 4), (4, 9), (9, 16)):
        assert math.isclose(
            float(np.linalg.norm(blocks[0][start:stop])),
            float(np.linalg.norm(blocks[1][start:stop])),
            rel_tol=1.0e-10,
        )


def test_auxiliary_multipole_moments_fail_closed_on_an_invalid_site_map():
    _, basis = _single_shell_basis("P", 1.1, [(0.0, 0.0, 0.0)])
    with pytest.raises(RuntimeError, match=r"function-to-site map must cover"):
        _moments(basis, [(0.0, 0.0, 0.0)], [0] * (basis.nbf() - 1))
    with pytest.raises(RuntimeError, match=r"site that does not exist"):
        _moments(basis, [(0.0, 0.0, 0.0)], [1] * basis.nbf())
