"""Pure-evaluator tests for the constrained density-fitting partition core.

Two evaluators are covered here, both of which are caller-supplied-data pure
functions with no wavefunction, no grid and no process options:

* ``auxiliary_multipole_moments`` -- analytic Racah regular real solid-harmonic
  moments ``Q_t[k]`` of every auxiliary basis function about its assigned site;
* ``solve_constrained_density_fit`` -- the auxiliary-space normal-equation solve
  under either a finite quadratic constraint penalty or a hard equality
  constraint.

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


# --------------------------------------------------------------------------
# A3 -- solve_constrained_density_fit
# --------------------------------------------------------------------------


def _solve(metric, rhs, constraints, targets, **options):
    result = psi4.core._atomic_polarizability_test_constrained_density_fit(
        psi4.core.Matrix.from_array(np.asarray(metric, dtype=float)),
        psi4.core.Matrix.from_array(np.asarray(rhs, dtype=float)),
        psi4.core.Matrix.from_array(np.asarray(constraints, dtype=float)),
        list(np.asarray(targets, dtype=float)),
        options,
    )
    return np.array(result["coefficients"]), result


def _spd_metric(eigenvalues, seed):
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.standard_normal((len(eigenvalues), len(eigenvalues))))[0]
    return basis @ np.diag(np.asarray(eigenvalues, dtype=float)) @ basis.T


def _well_conditioned_problem(seed=20260818, size=12, transitions=3, rows=2):
    rng = np.random.default_rng(seed)
    metric = _spd_metric(rng.uniform(0.5, 4.0, size), seed + 1)
    constraints = rng.standard_normal((rows, size))
    targets = rng.standard_normal(rows)
    exact = np.zeros((size, transitions))
    particular = np.linalg.lstsq(constraints, targets, rcond=None)[0]
    null = np.linalg.svd(constraints)[2][rows:]
    for column in range(transitions):
        exact[:, column] = particular + null.T @ rng.standard_normal(size - rows)
    multipliers = rng.standard_normal((rows, transitions))
    rhs = metric @ exact + constraints.T @ multipliers
    return metric, rhs, constraints, targets, exact


def test_hard_constraint_recovers_the_exact_solution_in_the_auxiliary_span():
    """A KKT point built by construction must be reproduced exactly."""
    metric, rhs, constraints, targets, exact = _well_conditioned_problem()
    solution, result = _solve(
        metric, rhs, constraints, targets, constraints_policy="hard"
    )
    assert np.allclose(solution, exact, rtol=1.0e-10, atol=1.0e-11)
    assert result["policy"] == "hard-constraint"
    assert result["constraint_count"] == constraints.shape[0]
    assert result["discarded_directions"] == 0


def test_hard_constraint_is_satisfied_to_machine_precision():
    metric, rhs, constraints, targets, _ = _well_conditioned_problem()
    solution, result = _solve(
        metric, rhs, constraints, targets, constraints_policy="hard"
    )
    residual = constraints @ solution - targets[:, None]
    assert np.max(np.abs(residual)) < 1.0e-12
    assert result["max_constraint_residual"] < 1.0e-12


def test_the_penalty_leaves_a_finite_constraint_residual_that_falls_as_one_over_lambda():
    """The reviewed reference protocol is a soft penalty, not a hard constraint."""
    metric, rhs, constraints, targets, _ = _well_conditioned_problem()
    residuals = []
    for penalty in (1.0, 1.0e2, 1.0e4):
        solution, result = _solve(
            metric, rhs, constraints, targets,
            constraints_policy="penalty", constraint_penalty=penalty,
        )
        assert result["policy"] == "quadratic-penalty"
        residuals.append(np.max(np.abs(constraints @ solution - targets[:, None])))
    assert residuals[0] > 1.0e-3
    for previous, current in zip(residuals, residuals[1:]):
        assert current < previous / 50.0


def test_the_penalty_reproduces_the_hard_constraint_solution_as_lambda_grows():
    metric, rhs, constraints, targets, _ = _well_conditioned_problem()
    hard, _ = _solve(metric, rhs, constraints, targets, constraints_policy="hard")
    errors = []
    for penalty in (1.0, 1.0e2, 1.0e4):
        solution, _ = _solve(
            metric, rhs, constraints, targets,
            constraints_policy="penalty", constraint_penalty=penalty,
        )
        errors.append(np.max(np.abs(solution - hard)))
    for previous, current in zip(errors, errors[1:]):
        assert current < previous / 50.0
    assert errors[-1] < 1.0e-4

    # Beyond this the penalty matrix itself is the accuracy limit, not the model:
    # its condition number grows linearly in the penalty weight, so the approach to
    # the hard-constraint solution stalls at the double-precision floor near 2e-07.
    tight, result = _solve(
        metric, rhs, constraints, targets,
        constraints_policy="penalty", constraint_penalty=1.0e6,
    )
    assert np.max(np.abs(tight - hard)) < 1.0e-6
    assert result["condition_number"] > 1.0e7


def test_general_multi_row_constraints_are_supported():
    metric, rhs, constraints, targets, exact = _well_conditioned_problem(
        seed=99, size=15, transitions=4, rows=5
    )
    solution, result = _solve(
        metric, rhs, constraints, targets, constraints_policy="hard"
    )
    assert result["constraint_count"] == 5
    assert np.allclose(solution, exact, rtol=1.0e-10, atol=1.0e-11)
    assert np.max(np.abs(constraints @ solution - targets[:, None])) < 1.0e-12


def test_an_empty_constraint_set_reduces_to_the_plain_metric_solve():
    metric, rhs, _, _, _ = _well_conditioned_problem()
    solution, result = _solve(metric, rhs, np.zeros((0, metric.shape[0])), [])
    assert result["constraint_count"] == 0
    assert np.allclose(metric @ solution, rhs, rtol=1.0e-10, atol=1.0e-11)


def test_the_metric_cutoff_is_relative_and_never_an_absolute_magnitude():
    """Rescaling the metric must not change which directions are retained.

    An absolute cutoff would keep both directions of the unscaled problem and
    discard both of the scaled one; a relative cutoff discards exactly the same
    direction in each. The solution must scale as ``1/s`` and nothing else.
    """
    size = 6
    spectrum = [1.0, 0.8, 0.6, 0.4, 0.2, 1.0e-4]
    metric = _spd_metric(spectrum, seed=7)
    rng = np.random.default_rng(11)
    rhs = rng.standard_normal((size, 2))
    empty = np.zeros((0, size))

    plain, plain_result = _solve(
        metric, rhs, empty, [], metric_relative_cutoff=1.0e-2
    )
    scale = 1.0e-6
    scaled, scaled_result = _solve(
        scale * metric, rhs, empty, [], metric_relative_cutoff=1.0e-2
    )

    assert plain_result["discarded_directions"] == 1
    assert scaled_result["discarded_directions"] == 1
    assert plain_result["retained_rank"] == scaled_result["retained_rank"] == size - 1
    assert np.allclose(scaled, plain / scale, rtol=1.0e-9, atol=1.0e-9)
    assert math.isclose(
        plain_result["condition_number"], scaled_result["condition_number"],
        rel_tol=1.0e-9,
    )
    assert math.isclose(
        plain_result["effective_cutoff"] / scaled_result["effective_cutoff"],
        1.0 / scale,
        rel_tol=1.0e-9,
    )


def test_the_default_gates_admit_the_reviewed_reference_conditioning():
    """The reviewed normal matrix is measured at 7.8e12; 1e12 would reject it."""
    size = 8
    spectrum = [1.0393e03 / 7.798e12] + [1.0] * (size - 2) + [1.0393e03]
    metric = _spd_metric(spectrum, seed=13)
    rng = np.random.default_rng(17)
    rhs = rng.standard_normal((size, 2))
    _, result = _solve(metric, rhs, np.zeros((0, size)), [])
    # The eigendecomposition of a 7.8e+12 matrix cannot resolve its own condition
    # number to better than a few parts in 1e5; the point of the assertion is that
    # the gate admits it, not that the diagnostic is exact.
    assert math.isclose(result["condition_number"], 7.798e12, rel_tol=1.0e-4)
    assert result["discarded_directions"] == 0
    assert result["maximum_condition_number"] == pytest.approx(1.0e14)
    assert result["metric_relative_cutoff"] == pytest.approx(1.0e-14)


def test_an_ill_conditioned_metric_fails_closed():
    size = 6
    metric = _spd_metric([1.0e-16] + [1.0] * (size - 1), seed=23)
    rng = np.random.default_rng(29)
    rhs = rng.standard_normal((size, 1))
    with pytest.raises(RuntimeError, match=r"condition number exceeds explicit threshold"):
        _solve(metric, rhs, np.zeros((0, size)), [])


def test_inconsistent_and_ambiguous_constraints_fail_closed():
    metric, rhs, constraints, targets, _ = _well_conditioned_problem()
    # A repeated row with a consistent target is ambiguous, not infeasible.
    duplicated = np.vstack([constraints, constraints[0]])
    with pytest.raises(RuntimeError, match=r"constraints are ambiguous \(linearly dependent\)"):
        _solve(
            metric, rhs, duplicated, list(targets) + [targets[0]],
            constraints_policy="hard",
        )
    # A repeated row with a contradictory target is infeasible, and is reported as
    # such before the ambiguity check, exactly as the refinement solver orders them.
    with pytest.raises(RuntimeError, match=r"constraints are inconsistent"):
        _solve(
            metric, rhs, duplicated, list(targets) + [targets[0] + 1.0],
            constraints_policy="hard",
        )
    with pytest.raises(RuntimeError, match=r"constraints are inconsistent"):
        _solve(
            metric, rhs, np.vstack([constraints, np.zeros(constraints.shape[1])]),
            list(targets) + [1.0], constraints_policy="hard",
        )


def test_a_nonsymmetric_metric_fails_closed():
    metric, rhs, _, _, _ = _well_conditioned_problem()
    broken = metric.copy()
    broken[0, 1] += 1.0
    with pytest.raises(RuntimeError, match=r"metric must be symmetric"):
        _solve(broken, rhs, np.zeros((0, metric.shape[0])), [])


# --------------------------------------------------------------------------
# A5 -- plan_cdf_partition, the Coulomb metric, and the localisation form
# --------------------------------------------------------------------------


def _cdf_plan(nbf=92, naux=246, nocc=5, nvir=87, site_count=3, memory_bytes=4 * 1024**3):
    return psi4.core._atomic_polarizability_estimate_cdf_partition(
        nbf, naux, nocc, nvir, site_count, memory_bytes
    )


def test_cdf_partition_plan_accounts_for_every_retained_payload():
    """The reviewed protocol's own dimensions, gated inside half the configured memory."""
    plan = _cdf_plan()
    assert plan["naux"] == 246
    assert plan["transition_count"] == 5 * 87
    # Three retained naux x naux matrices, one streamed auxiliary shell block of
    # three-index integrals, four naux x nov coefficient blocks, the moments, and the
    # site-major output. Nothing may be missing from the aggregate.
    parts = (
        plan["metric_bytes"]
        + plan["three_index_bytes"]
        + plan["coefficient_bytes"]
        + plan["moment_bytes"]
        + plan["projection_bytes"]
    )
    assert parts < plan["estimated_bytes"]
    assert plan["metric_bytes"] == 3 * 246 * 246 * 8
    assert plan["moment_bytes"] == 246 * 16 * 8
    assert plan["projection_bytes"] == 3 * 16 * 5 * 87 * 8
    assert plan["coefficient_bytes"] == 4 * 246 * 5 * 87 * 8
    assert plan["reserved_memory_bytes"] == (4 * 1024**3) // 2
    assert plan["estimated_bytes"] <= plan["reserved_memory_bytes"]
    assert plan["work_terms"] == 246 * 92 * 92 * 92
    assert plan["work_terms"] < plan["max_work_terms"]
    assert plan["algorithm"] == (
        "AUXILIARY_SHELL_STREAMED_THREE_INDEX_TRANSFORM_DENSE_NORMAL_SOLVE"
    )


def test_cdf_partition_plan_fails_closed_before_allocating():
    with pytest.raises(RuntimeError, match=r"dimensions must be nonzero"):
        _cdf_plan(naux=0)
    with pytest.raises(RuntimeError, match=r"auxiliary function count exceeds"):
        _cdf_plan(naux=100000)
    with pytest.raises(RuntimeError, match=r"site count exceeds"):
        _cdf_plan(site_count=65)
    with pytest.raises(RuntimeError, match=r"transition count exceeds"):
        _cdf_plan(nocc=8, nvir=100)
    with pytest.raises(RuntimeError, match=r"storage exceeds reserved memory"):
        _cdf_plan(memory_bytes=8 * 1024**2)


def _reviewed_auxiliary_basis(puream=0):
    molecule = psi4.geometry(
        """
0 1
O  0.00000000  0.0  0.00000000
H -1.45365196  0.0 -1.12168732
H  1.45365196  0.0 -1.12168732
symmetry c1
no_com
no_reorient
units bohr
"""
    )
    return molecule, psi4.core.BasisSet.build(
        molecule, "DF_BASIS_ATOMIC_POLARIZABILITY", "AUG-CC-PVTZ-RI",
        puream=puream, quiet=True,
    )


def test_the_reviewed_auxiliary_basis_has_the_recorded_structural_fingerprint():
    """246 Cartesian functions in 56 shells, not the 198 the spherical form gives.

    The difference is 48 Cartesian contaminant functions and it is not cosmetic: they
    are what makes d and g components carry rank-0 charge, so the spherical form is a
    different auxiliary space with a different partition.
    """
    _, cartesian = _reviewed_auxiliary_basis(puream=0)
    assert (cartesian.nbf(), cartesian.nshell(), cartesian.has_puream()) == (246, 56, False)
    per_center = {}
    for index in range(cartesian.nshell()):
        shell = cartesian.shell(index)
        per_center.setdefault(shell.ncenter, {}).setdefault(shell.am, 0)
        per_center[shell.ncenter][shell.am] += 1
    assert per_center[0] == {0: 9, 1: 7, 2: 6, 3: 4, 4: 2}
    assert per_center[1] == per_center[2] == {0: 5, 1: 4, 2: 3, 3: 2}

    _, spherical = _reviewed_auxiliary_basis(puream=1)
    assert (spherical.nbf(), spherical.nshell(), spherical.has_puream()) == (198, 56, True)


def test_the_reviewed_auxiliary_basis_charge_vector_is_dense_over_sixty_seven_functions():
    """A structural assertion no spherical basis can satisfy.

    A pure auxiliary basis puts charge only on its s functions -- 19 of these 56 shells.
    The Cartesian basis puts it on 19 s functions, the xx/yy/zz component of each of 12
    d shells, and six components of each of oxygen's two g shells: 67 in all. The
    charge-penalty vector is therefore dense, and the rank-0 projection draws on d- and
    g-type coefficients as well as s-type ones.
    """
    molecule, cartesian = _reviewed_auxiliary_basis(puream=0)
    sites = [
        [molecule.x(atom), molecule.y(atom), molecule.z(atom)]
        for atom in range(molecule.natom())
    ]
    function_to_site = [cartesian.function_to_center(k) for k in range(cartesian.nbf())]
    moments = _moments(cartesian, sites, function_to_site)
    assert int(np.count_nonzero(moments[:, 0])) == 67


def _coulomb_metric(basis):
    return np.array(psi4.core._atomic_polarizability_test_auxiliary_coulomb_metric(basis))


def test_the_auxiliary_coulomb_metric_matches_psi4s_own_two_centre_integrals():
    """Independent oracle for J_kl = (chi_k || chi_l) on the reviewed basis."""
    _, cartesian = _reviewed_auxiliary_basis(puream=0)
    metric = _coulomb_metric(cartesian)
    zero = psi4.core.BasisSet.zero_ao_basis_set()
    reference = np.array(
        psi4.core.MintsHelper(cartesian).ao_eri(cartesian, zero, cartesian, zero)
    ).reshape(cartesian.nbf(), cartesian.nbf())
    assert np.allclose(metric, reference, rtol=0.0, atol=1.0e-13)
    assert np.allclose(metric, metric.T, rtol=0.0, atol=0.0)
    assert np.all(np.diag(metric) > 0.0)


def _normal_matrix(metric, function_to_site, site_count, localisation, weight):
    return np.array(
        psi4.core._atomic_polarizability_test_cdf_localised_normal_matrix(
            psi4.core.Matrix.from_array(np.asarray(metric, dtype=float)),
            [int(site) for site in function_to_site],
            int(site_count),
            localisation,
            float(weight),
        )
    )


def _recovered_localisation_masks(metric, function_to_site, site_count, weight=0.5):
    """Recover K_inter and K_self from the assembler rather than reimplementing them."""
    inter = _normal_matrix(metric, function_to_site, site_count, "inter-site", weight)
    self_form = _normal_matrix(
        metric, function_to_site, site_count, "site-self-repulsion", weight
    )
    return (metric - inter) / weight, (self_form - metric) / weight


def _reviewed_metric_and_sites():
    _, cartesian = _reviewed_auxiliary_basis(puream=0)
    function_to_site = np.array(
        [cartesian.function_to_center(k) for k in range(cartesian.nbf())]
    )
    return _coulomb_metric(cartesian), function_to_site, 3


def test_the_localisation_masks_split_the_coulomb_metric_exactly():
    """K_self + K_inter must be J itself, which is what makes the form free of integrals.

    Both masks are recovered from the assembler rather than assumed: the inter-site form
    is J - eta K_inter and the site form is J + eta K_self, so dividing the difference
    from J by eta gives each mask back. They must then add to J, and each must equal J
    masked by site coincidence.
    """
    metric, function_to_site, site_count = _reviewed_metric_and_sites()
    # Recovered at a weight of order one. Dividing by the reviewed 5e-4 would amplify
    # double-precision round-off by 2000x, which is a property of this arithmetic and
    # not of the assembler; the reviewed weight is checked below against the mask
    # recovered here, which needs no division at all.
    k_inter, k_self = _recovered_localisation_masks(metric, function_to_site, site_count)
    same_site = function_to_site[:, None] == function_to_site[None, :]
    assert np.allclose(k_inter, np.where(same_site, 0.0, metric), rtol=0.0, atol=1.0e-13)
    assert np.allclose(k_self, np.where(same_site, metric, 0.0), rtol=0.0, atol=1.0e-13)
    assert np.allclose(k_self + k_inter, metric, rtol=0.0, atol=1.0e-13)

    reviewed = _normal_matrix(metric, function_to_site, site_count, "inter-site", 5.0e-4)
    assert np.allclose(reviewed, metric - 5.0e-4 * k_inter, rtol=1.0e-13, atol=1.0e-15)


def test_the_localisation_form_is_the_published_site_blocked_repulsion():
    """d^T K d must be the published sum over site-block Coulomb repulsions.

    The published localisation term is built from the site-resolved fitted densities,
    E^ab = ( rho~^a || rho~^b ) with rho~^a = sum_{k on a} d_k chi_k. Because the Coulomb
    metric *is* the pairwise repulsion of the auxiliary functions -- checked against
    Psi4's own two-centre integrals above -- E^ab is d_a^T J[a, b] d_b, and the two
    quadratic forms must agree term for term. Each site's self-repulsion must also be
    strictly positive, which a mask built the wrong way round would break.
    """
    metric, function_to_site, site_count = _reviewed_metric_and_sites()
    k_inter, k_self = _recovered_localisation_masks(metric, function_to_site, site_count)

    coefficients = np.random.default_rng(20260819).standard_normal(metric.shape[0])
    blocks = [np.flatnonzero(function_to_site == site) for site in range(site_count)]
    self_repulsion = 0.0
    cross_repulsion = 0.0
    for first in range(site_count):
        for second in range(site_count):
            energy = (
                coefficients[blocks[first]]
                @ metric[np.ix_(blocks[first], blocks[second])]
                @ coefficients[blocks[second]]
            )
            if first == second:
                assert energy > 0.0
                self_repulsion += energy
            else:
                cross_repulsion += energy

    assert math.isclose(coefficients @ k_self @ coefficients, self_repulsion, rel_tol=1.0e-12)
    assert math.isclose(coefficients @ k_inter @ coefficients, cross_repulsion, rel_tol=1.0e-12)
    # And the two together are the total Coulomb self-repulsion of the fitted density.
    assert math.isclose(
        self_repulsion + cross_repulsion, coefficients @ metric @ coefficients, rel_tol=1.0e-12
    )


def test_the_localisation_form_reduces_to_the_bare_metric_when_it_is_switched_off():
    metric, function_to_site, site_count = _reviewed_metric_and_sites()
    for localisation, weight in (
        ("none", 5.0e-4), ("inter-site", 0.0), ("site-self-repulsion", 0.0),
    ):
        assembled = _normal_matrix(metric, function_to_site, site_count, localisation, weight)
        assert np.allclose(assembled, metric, rtol=0.0, atol=0.0)


def test_the_localisation_form_fails_closed_on_an_unusable_weight_or_site_map():
    metric, function_to_site, site_count = _reviewed_metric_and_sites()
    with pytest.raises(RuntimeError, match=r"smaller than one in magnitude"):
        _normal_matrix(metric, function_to_site, site_count, "inter-site", 1.0)
    with pytest.raises(RuntimeError, match=r"function-to-site map must cover"):
        _normal_matrix(metric, function_to_site[:-1], site_count, "inter-site", 5.0e-4)
    with pytest.raises(RuntimeError, match=r"site that does not exist"):
        _normal_matrix(metric, function_to_site, 2, "inter-site", 5.0e-4)


# --------------------------------------------------------------------------
# A5 -- project_transition_multipoles_cdf
# --------------------------------------------------------------------------

#: Small, cheap protocol. The auxiliary partition is being tested for layout, algebra
#: and fail-closed behaviour here, none of which depends on basis or grid quality; the
#: reviewed protocol comparison lives with the reviewed literals.
_CONTEXT_PROTOCOL = {
    "basis": "sto-3g",
    "scf_type": "pk",
    "reference": "rhf",
    "dft_spherical_points": 50,
    "dft_radial_points": 12,
    "dft_density_tolerance": 1.0e-12,
    "dft_grac_shift": 0.0,
}

_CONTEXT_GEOMETRY = """
0 1
O  0.00000000  0.0  0.00000000
H -1.45365196  0.0 -1.12168732
H  1.45365196  0.0 -1.12168732
symmetry c1
no_com
no_reorient
units bohr
"""

_AUXILIARY_KEY = "DF_BASIS_ATOMIC_POLARIZABILITY"


@pytest.fixture(scope="module")
def scf_triple():
    psi4.core.be_quiet()
    psi4.set_options(_CONTEXT_PROTOCOL)
    neutral = psi4.geometry(_CONTEXT_GEOMETRY)
    _, precursor = psi4.energy("pbe0", molecule=neutral, return_wfn=True)

    cation = neutral.clone()
    cation.set_molecular_charge(1)
    cation.set_multiplicity(2)
    cation.update_geometry()
    psi4.set_options({"reference": "uhf"})
    _, cation_wfn = psi4.energy("pbe0", molecule=cation, return_wfn=True)

    homo = max(precursor.epsilon_a_subset("SO", "OCC").to_array().ravel())
    shift = cation_wfn.energy() - precursor.energy() + homo
    psi4.set_options({"reference": "rhf", "dft_grac_shift": shift})
    _, grac = psi4.energy("pbe0", molecule=neutral, return_wfn=True)
    psi4.set_options({"dft_grac_shift": 0.0})
    return grac, precursor, cation_wfn


def _context_with_auxiliary(triple, auxiliary_name="cc-pvdz-ri", puream=0):
    grac, precursor, cation = triple
    auxiliary = psi4.core.BasisSet.build(
        grac.molecule(), _AUXILIARY_KEY, auxiliary_name, puream=puream, quiet=True
    )
    grac.set_basisset(_AUXILIARY_KEY, auxiliary)
    context = psi4.core._atomic_polarizability_make_frozen_response_context(
        grac, precursor, cation, _AUXILIARY_KEY
    )
    return context, auxiliary


def _project_cdf(context, **options):
    return psi4.core._atomic_polarizability_test_project_transition_multipoles_cdf(
        context, options
    )


def test_cdf_projection_layout_is_identical_to_the_real_space_producer(scf_triple):
    """Same rows, same columns, same transition order. Only the definition differs."""
    context, auxiliary = _context_with_auxiliary(scf_triple)
    cdf = _project_cdf(context)
    grac, precursor, cation = scf_triple
    plain = psi4.core._atomic_polarizability_make_frozen_response_context(
        grac, precursor, cation
    )
    point_count = plain.summary()["grid_point_count"]
    site_count = plain.summary()["site_count"]
    isa = psi4.core._atomic_polarizability_test_project_transition_multipoles_context(
        plain, plain, [1.0 / site_count] * (point_count * site_count)
    )

    assert cdf["component_order"] == isa["component_order"]
    assert cdf["transition_order"] == isa["transition_order"]
    assert cdf["transitions"] == isa["transitions"]
    cdf_values = np.array(cdf["values"])
    isa_values = np.array(isa["values"])
    assert cdf_values.shape == isa_values.shape == (site_count * 16, len(cdf["transitions"]))
    assert np.all(np.isfinite(cdf_values))
    # And they are genuinely different partitions of the same response, not copies.
    assert not np.allclose(cdf_values, isa_values, rtol=1.0e-3, atol=1.0e-6)
    assert cdf["auxiliary_count"] == auxiliary.nbf()


def test_cdf_rank_zero_rows_sum_to_the_fitted_charge_and_not_to_machine_zero(scf_triple):
    """Partition conservation, gated at the precision the penalty model actually has.

    Summing every site's rank-0 row gives sum_k q_k d_k for that transition, which is
    the fitted charge of the transition density. Every pair consumed here is
    occupied-virtual, so the target is the overlap <i|a> = 0. Under a hard Lagrange
    constraint this would be machine zero; under the finite quadratic penalty the
    reference actually used it is small and nonzero, and the reference's own log records
    its violation at about 1e-2. So the invariant is asserted against the measured
    residual, and separately shown to be *not* machine zero -- which is the assertion
    that distinguishes the model being reproduced from the one the specification
    originally described.
    """
    context, _ = _context_with_auxiliary(scf_triple)
    result = _project_cdf(context, localisation="inter-site", localisation_weight=5.0e-4,
                          constraints_policy="penalty", constraint_penalty=1.0)
    values = np.array(result["values"])
    site_count = values.shape[0] // 16
    charge = values[[site * 16 for site in range(site_count)], :].sum(axis=0)

    measured = float(result["max_charge_residual"])
    assert measured == pytest.approx(float(np.max(np.abs(charge))), rel=1.0e-9, abs=1.0e-14)
    assert 0.0 < measured <= result["charge_residual_bound"]
    assert np.max(np.abs(charge)) <= measured * (1.0 + 1.0e-9)
    assert result["fit_policy"] == "quadratic-penalty"
    assert result["localisation"] == "inter-site"
    assert result["localisation_weight"] == pytest.approx(5.0e-4)


def test_a_hard_charge_constraint_drives_the_conservation_residual_to_machine_zero(scf_triple):
    """The same projection under the specification's original model, for contrast.

    This is the measurement that shows the two models are different partitions rather
    than two evaluations of one. It is not the reviewed protocol and is not the default.
    """
    context, _ = _context_with_auxiliary(scf_triple)
    penalty = _project_cdf(context, constraints_policy="penalty", constraint_penalty=1.0)
    hard = _project_cdf(context, constraints_policy="hard")
    assert hard["max_charge_residual"] < 1.0e-12
    assert penalty["max_charge_residual"] > 1.0e6 * hard["max_charge_residual"]
    assert not np.allclose(
        np.array(hard["values"]), np.array(penalty["values"]), rtol=1.0e-6, atol=1.0e-9
    )


def test_cdf_projection_reports_the_measured_conditioning_and_retains_every_direction(scf_triple):
    context, auxiliary = _context_with_auxiliary(scf_triple)
    result = _project_cdf(context)
    assert result["retained_rank"] == auxiliary.nbf()
    assert result["discarded_directions"] == 0
    assert result["condition_number"] > 1.0
    assert result["max_stationarity_residual"] < 1.0e-8
    assert 0 < result["charged_auxiliary_count"] <= auxiliary.nbf()


def test_cdf_projection_gates_resources_before_allocating(scf_triple):
    context, _ = _context_with_auxiliary(scf_triple)
    incoming = psi4.core.get_memory()
    try:
        psi4.core.set_memory_bytes(1024**2, True)
        with pytest.raises(RuntimeError, match=r"storage exceeds reserved memory"):
            _project_cdf(context)
    finally:
        psi4.core.set_memory_bytes(incoming, True)


def test_cdf_projection_requires_a_sealed_auxiliary_basis(scf_triple):
    grac, precursor, cation = scf_triple
    plain = psi4.core._atomic_polarizability_make_frozen_response_context(
        grac, precursor, cation
    )
    with pytest.raises(RuntimeError, match=r"carries no sealed auxiliary basis"):
        _project_cdf(plain)


def test_the_frozen_context_rejects_an_unattached_auxiliary_key(scf_triple):
    grac, precursor, cation = scf_triple
    with pytest.raises(RuntimeError, match=r"is not attached to the GRAC"):
        psi4.core._atomic_polarizability_make_frozen_response_context(
            grac, precursor, cation, "DF_BASIS_NOT_ATTACHED"
        )


def test_cdf_projection_rejects_a_spherical_auxiliary_basis(scf_triple):
    """A spherical auxiliary space is a different space, so this fails rather than runs."""
    context, auxiliary = _context_with_auxiliary(scf_triple, puream=1)
    assert auxiliary.has_puream() is True
    with pytest.raises(RuntimeError, match=r"must use Cartesian functions"):
        _project_cdf(context)


def test_cdf_projection_rejects_an_auxiliary_basis_that_is_not_the_sealed_one(scf_triple):
    """Asking for one auxiliary space and attaching another must not run silently."""
    context, _ = _context_with_auxiliary(scf_triple)
    assert _project_cdf(context, auxiliary_basis="cc-pVDZ-RI")["auxiliary_count"] > 0
    with pytest.raises(RuntimeError, match=r"does not match the basis sealed"):
        _project_cdf(context, auxiliary_basis="aug-cc-pVTZ-RI")
