# Psi4: Copyright (c) 2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Native point-charge response prerequisite; requires the parent's rebuilt Psi4.

Every numerical claim here is checked against an oracle written in this file or
against a different Psi4 code path, never against a stored CamCASP/ORIENT
number, a reference file or a tolerance imported from another track. The
analytic electrostatic-potential oracle uses only ``math.erf`` and covers every
shell of the fixture basis (s and p); it is what pins the sign and the
normalization of the AO kernel.

``core.ExternalPotential`` is **not** an independent kernel: its
``computePotentialMatrix`` reaches the same ``libint2::Operator::nuclear``
integrals that ``point_response.cc`` drives, with the same charge-inclusive
convention. Layer 2 below therefore certifies the AO->MO transform and the
``t = a*nocc+i`` packing, not the kernel itself.

Two normalizations are deliberately **not** closed here. (a) The overall factor
4 in the shared full-OV right-hand side is written identically in the code and
in layer 3's re-derivation, and cancels exactly in layer 4's ratio, so no test
in this file would notice a global rescaling of the response; an absolute
oracle (Psi4's own CPHF dipole polarizability against a matched exchange/kernel
configuration) is the missing check. (b) No shell of angular momentum above p
is exercised, because the fixture basis has none.

This file certifies the *prerequisite* only: a direct-OV point-charge response
of an actual native wavefunction. It makes no claim about the historical
constrained-NN/distributed fitted-propagator target, its point lattice, frame,
anchoring or refinement, and the PFIT test below is an interface check with an
explicitly synthetic caller-declared model.
"""
import math
from contextlib import contextmanager
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import psi4
from psi4 import core
from psi4.driver.p4util import OptionsState
from psi4.driver.procrouting.isapol_native_response import native_response_from_wavefunction
from psi4.driver.procrouting import isapol_native_point_response as npr


@contextmanager
def scf_settings(basis, puream=True):
    saved = OptionsState(["BASIS"], ["PUREAM"], ["SCF_TYPE"], ["SCF", "REFERENCE"],
                         ["SCF", "E_CONVERGENCE"], ["SCF", "D_CONVERGENCE"])
    try:
        psi4.set_options({"basis": basis, "puream": puream, "scf_type": "pk",
                          "reference": "rhf", "e_convergence": 1.e-11,
                          "d_convergence": 1.e-11})
        yield
    finally:
        saved.restore()


def _water(z):
    return psi4.geometry(f"""
    0 1
    O 0 0 0
    H 0.757 0 {z}
    H -0.757 0 {z}
    symmetry c1
    no_reorient
    no_com
    """)


@pytest.fixture(scope="module")
def water():
    with scf_settings("sto-3g"):
        _, wfn = psi4.energy("hf", molecule=_water(0.586), return_wfn=True)
    return wfn


@pytest.fixture(scope="module")
def other_water():
    """Same dimensions, different orbitals: an invalidated response context."""
    with scf_settings("sto-3g"):
        _, wfn = psi4.energy("hf", molecule=_water(0.606), return_wfn=True)
    return wfn


@pytest.fixture(scope="module")
def response(water):
    return native_response_from_wavefunction(water, caller_converged=True, kernel="no_local",
                                             exact_exchange=0.0, local_scale=0.0)


#: Three points, deliberately asymmetric and off every symmetry element.
POINTS = np.array([[3., 0., 0.], [0., 3., 0.], [0., 0., -3.]])


# ----------------------------------------------------------------------------
# Layer 1: analytic AO electrostatic-potential oracle
# ----------------------------------------------------------------------------
def boys_f(m, x):
    """F_m(x) = int_0^1 t^(2m) exp(-x t^2) dt, from ``math.erf`` only.

    The alternating series is used for x <= 1, where the upward recursion
    F_(m+1) = ((2m+1) F_m - exp(-x)) / 2x loses digits to cancellation; above
    that the recursion is stable for these small m.
    """
    if x <= 1.:
        return sum(((-x) ** k / math.factorial(k)) / (2 * m + 2 * k + 1) for k in range(30))
    values = [math.sqrt(math.pi / (4. * x)) * math.erf(math.sqrt(x))]
    decay = math.exp(-x)
    for lower in range(m):
        values.append(((2 * lower + 1) * values[lower] - decay) / (2. * x))
    return values[m]


def _analytic_primitive_block(am_a, a, a_xyz, am_b, b, b_xyz, centre):
    """Primitive POSITIVE-kernel ESP block for angular momenta in {s, p}.

    From the Gaussian product theorem plus one Obara-Saika step, with
    p = a+b, P = (aA+bB)/p, K = (2 pi/p) exp(-(ab/p)|A-B|^2), T = p|P-C|^2:
        (s|V|s)      = K F0
        (p_i|V|s)    = K [ (P_i-A_i) F0 - (P_i-C_i) F1 ]
        (p_i|V|p_j)  = (P_i-A_i)(s|V|p_j)^(0) - (P_i-C_i)(s|V|p_j)^(1)
                       + delta_ij K (F0 - F1) / 2p
    Cartesian p order (x, y, z); the pure/Cartesian p functions differ only by
    ordering at this angular momentum.
    """
    p = a + b
    centroid = (a * a_xyz + b * b_xyz) / p
    kernel = (2. * math.pi / p) * math.exp(-(a * b / p) * float(np.dot(a_xyz - b_xyz,
                                                                      a_xyz - b_xyz)))
    f = [boys_f(m, p * float(np.dot(centroid - centre, centroid - centre))) for m in range(3)]
    if am_a == 0 and am_b == 0:
        return np.array([[kernel * f[0]]])
    if am_a == 1 and am_b == 0:
        return np.array([[kernel * ((centroid[i] - a_xyz[i]) * f[0]
                                    - (centroid[i] - centre[i]) * f[1])] for i in range(3)])
    if am_a == 0 and am_b == 1:
        return np.array([[kernel * ((centroid[j] - b_xyz[j]) * f[0]
                                    - (centroid[j] - centre[j]) * f[1]) for j in range(3)]])
    out = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            value = (centroid[i] - a_xyz[i]) * kernel * (
                (centroid[j] - b_xyz[j]) * f[0] - (centroid[j] - centre[j]) * f[1])
            value -= (centroid[i] - centre[i]) * kernel * (
                (centroid[j] - b_xyz[j]) * f[1] - (centroid[j] - centre[j]) * f[2])
            if i == j:
                value += kernel * (f[0] - f[1]) / (2. * p)
            out[i, j] = value
    return out


#: Psi4 orders pure l=1 functions m = 0, +1, -1, i.e. (z, x, y). A wrong
#: assumption here fails the Cartesian-versus-pure comparison loudly.
PURE_P_FROM_CARTESIAN = (2, 0, 1)


def analytic_ao_matrix(basis, centre):
    """Full AO matrix of +int chi chi /|r-C| dr for a basis of s and p shells."""
    matrix = np.zeros((basis.nbf(), basis.nbf()))
    for ish in range(basis.nshell()):
        sh_a = basis.shell(ish)
        a_xyz = np.array([sh_a.coord(k) for k in range(3)])
        assert sh_a.am <= 1, "oracle covers s and p shells only"
        for jsh in range(basis.nshell()):
            sh_b = basis.shell(jsh)
            b_xyz = np.array([sh_b.coord(k) for k in range(3)])
            block = np.zeros((3 if sh_a.am else 1, 3 if sh_b.am else 1))
            for k in range(sh_a.nprimitive):
                for l in range(sh_b.nprimitive):
                    block += sh_a.coef(k) * sh_b.coef(l) * _analytic_primitive_block(
                        sh_a.am, sh_a.exp(k), a_xyz, sh_b.am, sh_b.exp(l), b_xyz, centre)
            if sh_a.am == 1 and sh_a.is_pure():
                block = block[list(PURE_P_FROM_CARTESIAN), :]
            if sh_b.am == 1 and sh_b.is_pure():
                block = block[:, list(PURE_P_FROM_CARTESIAN)]
            matrix[sh_a.function_index:sh_a.function_index + block.shape[0],
                   sh_b.function_index:sh_b.function_index + block.shape[1]] = block
    return matrix


def external_potential_ao(basis, centre):
    """Independent AO potential matrix through the external-charge code path.

    ``addCharge`` is charge-inclusive, so charge -1 yields the POSITIVE Coulomb
    kernel +int chi chi /|r-C| dr published by IsaPointChargeOperators.
    """
    potential = core.ExternalPotential()
    potential.addCharge(-1., float(centre[0]), float(centre[1]), float(centre[2]))
    return np.asarray(potential.computePotentialMatrix(basis))


def test_analytic_oracle_matches_engine_over_the_whole_basis(water):
    """Pins the sign and normalization of the AO kernel against pure analysis.

    Covers every shell of the fixture basis, s and p, not just the s block.
    """
    basis = water.basisset()
    assert sorted({basis.shell(i).am for i in range(basis.nshell())}) == [0, 1]
    assert all(basis.shell(i).is_pure() for i in range(basis.nshell()) if basis.shell(i).am)
    for centre in POINTS:
        actual = external_potential_ao(basis, centre)
        assert actual.shape == (basis.nbf(), basis.nbf())
        assert np.min(np.diag(actual)) > 0.  # positive kernel, not charge-inclusive
        np.testing.assert_allclose(actual, analytic_ao_matrix(basis, centre),
                                   rtol=0., atol=1.e-13)


def test_analytic_oracle_matches_engine_without_a_pure_ordering_assumption(water):
    """Same oracle on a Cartesian basis, where no l=1 ordering choice is made."""
    basis = core.BasisSet.build(water.molecule(), "BASIS", "sto-3g", puream=False, quiet=True)
    assert not any(basis.shell(i).is_pure() for i in range(basis.nshell()))
    for centre in POINTS:
        np.testing.assert_allclose(external_potential_ao(basis, centre),
                                   analytic_ao_matrix(basis, centre), rtol=0., atol=1.e-13)


# ----------------------------------------------------------------------------
# Layer 2: the owned operator against an independent MO transformation
# ----------------------------------------------------------------------------
def independent_legs(wfn, points):
    """W[a*nocc+i][p] from ExternalPotential AO matrices and an explicit transform.

    Same underlying Libint ``nuclear`` integrals as the C++ (see the module
    docstring), so this certifies the AO->MO transform and the OV packing, not
    the kernel; the kernel is layer 1's job.
    """
    c = np.asarray(wfn.Ca())
    nocc, nmo = wfn.nalpha(), c.shape[1]
    nvir = nmo - nocc
    legs = np.zeros((nocc * nvir, len(points)))
    for p, centre in enumerate(points):
        ov = c[:, :nocc].T @ external_potential_ao(wfn.basisset(), centre) @ c[:, nocc:]
        for a in range(nvir):
            for i in range(nocc):
                legs[a * nocc + i, p] = ov[i, a]
    return legs


def test_operators_match_independent_mo_transform(water, response):
    out = npr.native_point_charge_response(response, water, POINTS)
    operators = out.operators
    assert operators.representation == npr.REPRESENTATION
    assert operators.ov_order == "occupied_fast: t=a*nocc+i"
    assert "positive Coulomb kernel" in operators.convention
    assert "no 1/2, no nuclear term" in operators.convention
    legs = operators.operators().to_array()
    expected = independent_legs(water, POINTS)
    assert legs.shape == (water.nalpha() * (water.nmo() - water.nalpha()), len(POINTS))
    np.testing.assert_allclose(legs, expected, rtol=0., atol=1.e-13)
    np.testing.assert_allclose(operators.points().to_array(), POINTS, rtol=0., atol=0.)
    assert operators.maximum_absolute_element == pytest.approx(np.max(np.abs(expected)))
    # Diagnostics are recorded, and recorded honestly; nothing filters on them.
    assert operators.minimum_nuclear_distance_bohr > 0.
    assert operators.minimum_point_separation_bohr == pytest.approx(
        min(np.linalg.norm(POINTS[i] - POINTS[j]) for i in range(3) for j in range(i)))
    assert operators.planned_bytes > 0


def test_single_point_has_no_separation_diagnostic(water, response):
    out = npr.native_point_charge_response(response, water, POINTS[:1])
    assert out.operators.npoint == 1
    assert out.operators.minimum_point_separation_bohr == 0.
    assert out.responses[0].shape == (1, 1)
    assert out.packed_targets[0].shape == (1,)


# ----------------------------------------------------------------------------
# Layer 3: the bounded npoint-RHS solve against the full nov-RHS response
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("omega", [0., 0.3])
def test_point_legs_solve_equals_full_response_contraction(water, response, omega):
    """The bounded route must reproduce -W^T C W built from the full solve.

    The npoint-RHS route must agree with the nov-RHS one exactly; it saves the
    nov x nov right-hand-side and solution blocks, not the O(nov^3)
    factorization. It is an equivalence check, not a relaxation of the native
    response work guard, which ``response`` already passed.
    """
    out = npr.native_point_charge_response(response, water, POINTS, frequencies=(omega,))
    legs = out.operators.operators().to_array()
    full = np.asarray(response.at_frequency(omega).raw_coupled)
    np.testing.assert_allclose(out.responses[0], -(legs.T @ full @ legs),
                               rtol=2.e-9, atol=2.e-12)
    # And against the explicit linear algebra, independent of the shared solver.
    h1 = response.provider.h1().to_array()
    h2 = response.provider.h2().to_array()
    direct = np.linalg.solve(h2 @ h1 + omega**2 * np.eye(len(h1)), -4. * h2 @ legs)
    np.testing.assert_allclose(out.responses[0], -(legs.T @ direct), rtol=2.e-9, atol=2.e-12)


def test_frequency_batching_is_independent_of_grouping(water, response):
    grouped = npr.native_point_charge_response(response, water, POINTS, frequencies=(0., 0.3))
    for k, omega in enumerate(grouped.frequencies_au):
        single = npr.native_point_charge_response(response, water, POINTS, frequencies=(omega,))
        np.testing.assert_array_equal(grouped.responses[k], single.responses[0])
        np.testing.assert_array_equal(grouped.packed_targets[k], single.packed_targets[0])
    assert grouped.frequencies_au == (0., 0.3)
    # Static response is strictly the largest magnitude on the imaginary axis.
    assert grouped.maximum_absolute_values[0] > grouped.maximum_absolute_values[1]


# ----------------------------------------------------------------------------
# Layer 4: far-field multipole limit with measured 1/R convergence
# ----------------------------------------------------------------------------
def dipole_legs(wfn):
    """d[a*nocc+i][x] = <i|r_x|a>, in the same occupied-fast OV order."""
    c = np.asarray(wfn.Ca())
    nocc, nmo = wfn.nalpha(), c.shape[1]
    nvir = nmo - nocc
    dip = [np.asarray(m) for m in core.MintsHelper(wfn.basisset()).ao_dipole()]
    out = np.zeros((nocc * nvir, 3))
    for x in range(3):
        ov = c[:, :nocc].T @ dip[x] @ c[:, nocc:]
        for a in range(nvir):
            for i in range(nocc):
                out[a * nocc + i, x] = ov[i, a]
    return out


@pytest.mark.parametrize("omega", [0., 0.3])
def test_far_field_limit_and_first_order_convergence(water, response, omega):
    """v_pq -> (R_p . alpha . R_q)/(R_p^3 R_q^3) with error O(r_mol/R).

    alpha is built here from independent dipole OV legs and the same full
    response, so this simultaneously pins the target sign (alpha > 0 requires
    v_pp > 0) and the leading multipole content of W.
    """
    legs = dipole_legs(water)
    full = np.asarray(response.at_frequency(omega).raw_coupled)
    alpha = -legs.T @ full @ legs
    assert np.min(np.diag(alpha)) > 0.
    directions = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                           [1., 1., 1.] / np.sqrt(3.)])
    errors = []
    for radius in (50., 100., 200.):
        points = directions * radius
        out = npr.native_point_charge_response(response, water, points, frequencies=(omega,))
        value = out.responses[0]
        norms = np.linalg.norm(points, axis=1)
        approx = (points @ alpha @ points.T) / np.outer(norms**3, norms**3)
        errors.append(np.max(np.abs(value - approx)) / np.max(np.abs(approx)))
        assert np.min(np.diag(value)) > 0.
    assert errors[0] < 0.05
    for coarse, fine in zip(errors, errors[1:]):
        # Doubling R must halve the relative error: the residue is the O(1/R)
        # quadrupole/penetration term, not an unconverged or mis-signed solve.
        assert coarse / fine == pytest.approx(2., rel=0.02)


# ----------------------------------------------------------------------------
# Diagnostics: reciprocity and positivity are reported, never repaired
# ----------------------------------------------------------------------------
def test_reciprocity_positivity_and_packing(water, response):
    out = npr.native_point_charge_response(response, water, POINTS, frequencies=(0., 0.3))
    n = len(POINTS)
    for k, value in enumerate(out.responses):
        assert out.reciprocity_defects[k] == pytest.approx(
            float(np.max(np.abs(value - value.T))), abs=0.)
        assert out.reciprocity_defects[k] < 1.e-12
        # Reported, and NOT symmetrized: the stored triangle is the computed one.
        packed = out.packed_targets[k]
        assert packed.shape == (n * (n + 1) // 2,)
        for i in range(n):
            for j in range(i + 1):
                assert packed[i * (i + 1) // 2 + j] == value[i, j]
        assert out.minimum_diagonals[k] == pytest.approx(float(np.min(np.diag(value))), abs=0.)
        assert out.minimum_diagonals[k] > 0.  # alpha positive definite => v_pp > 0
        assert out.maximum_absolute_values[k] == pytest.approx(float(np.max(np.abs(value))), abs=0.)
    assert out.convention == npr.CONVENTION
    assert "-dphi" in out.convention or "-d(phi_induced" in out.convention
    assert out.representation == npr.REPRESENTATION
    assert "not fitted, not constrained-NN, not refined" in out.generation_record
    assert "v=-W^T C W" in out.generation_record
    assert out.caller_converged is True
    assert out.convergence_evidence
    assert len(out.context_sha256) == 64


def test_context_hash_separates_points_frequencies_and_response_model(water, response):
    """The kernel/model identity enters the digest through H1/H2 as well as the
    recorded scalars, so a hybrid fraction cannot alias the pure-kernel context."""
    base = npr.native_point_charge_response(response, water, POINTS)
    same = npr.native_point_charge_response(response, water, POINTS)
    assert base.context_sha256 == same.context_sha256
    shifted = npr.native_point_charge_response(response, water, POINTS + 1.e-6)
    other_omega = npr.native_point_charge_response(response, water, POINTS, frequencies=(0.3,))
    assert len({base.context_sha256, shifted.context_sha256, other_omega.context_sha256}) == 3
    hybrid = native_response_from_wavefunction(water, caller_converged=True, kernel="no_local",
                                               exact_exchange=0.25, local_scale=0.0)
    assert npr.native_point_charge_response(hybrid, water, POINTS).context_sha256 \
        != base.context_sha256


# ----------------------------------------------------------------------------
# Immutability and ownership
# ----------------------------------------------------------------------------
def test_returned_state_is_immutable_and_copied(water, response):
    out = npr.native_point_charge_response(response, water, POINTS)
    with pytest.raises(FrozenInstanceError):
        out.convention = "other"
    assert out.points_bohr.flags.writeable is False
    assert out.responses[0].flags.writeable is False
    assert out.packed_targets[0].flags.writeable is False
    assert isinstance(out.responses, tuple) and isinstance(out.packed_targets, tuple)
    first, second = out.operators.operators(), out.operators.operators()
    assert first is not second
    first.set(0, 0, 12345.)
    np.testing.assert_array_equal(second.to_array(), out.operators.operators().to_array())
    points_copy = out.operators.points()
    points_copy.set(0, 0, -99.)
    np.testing.assert_allclose(out.operators.points().to_array(), POINTS, rtol=0., atol=0.)


def test_supplied_points_array_is_not_aliased(water, response):
    supplied = POINTS.copy()
    out = npr.native_point_charge_response(response, water, supplied)
    supplied[0, 0] = -7.
    np.testing.assert_allclose(out.points_bohr, POINTS, rtol=0., atol=0.)


# ----------------------------------------------------------------------------
# Guards: inputs, resources, context invalidation
# ----------------------------------------------------------------------------
def test_point_input_guards(water, response):
    for bad in (POINTS[:, :2], POINTS.ravel(), np.zeros((0, 3)),
                np.array([[np.nan, 0., 0.]]), POINTS + 1.j):
        with pytest.raises(ValueError):
            npr.native_point_charge_response(response, water, bad)
    with pytest.raises(ValueError, match="duplicate"):
        npr.native_point_charge_response(response, water, np.vstack([POINTS, POINTS[1]]))
    with pytest.raises(ValueError, match="resource limit"):
        npr.native_point_charge_response(response, water, POINTS, max_points=2)
    with pytest.raises(ValueError, match="resource limit"):
        npr.native_point_charge_response(response, water, POINTS, max_bytes=8)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            npr.native_point_charge_response(response, water, POINTS, max_points=bad)
        with pytest.raises(ValueError, match="positive integer"):
            npr.native_point_charge_response(response, water, POINTS, max_bytes=bad)


def test_frequency_guards(water, response):
    for bad in ((), (-1.e-9,), (np.nan,), (0., 0.), 0.3, np.zeros((2, 1)),
                tuple(float(i) for i in range(65))):
        with pytest.raises(ValueError):
            npr.native_point_charge_response(response, water, POINTS, frequencies=bad)


def test_response_and_context_guards(water, other_water, response):
    with pytest.raises(ValueError, match="native wavefunction response producer"):
        npr.native_point_charge_response(object(), water, POINTS)
    with pytest.raises(ValueError, match="orbitals differ"):
        npr.native_point_charge_response(response, other_water, POINTS)
    with pytest.raises(ValueError):
        npr.native_point_charge_response(response, None, POINTS)


def test_cpp_operator_guards_are_authoritative(water):
    matrix = core.Matrix.from_array(POINTS)
    with pytest.raises((ValueError, RuntimeError),
                       match="caller convergence declaration required"):
        core.IsaPointChargeOperators(water, False, matrix, 512 * 1024**2, 512)
    with pytest.raises((ValueError, RuntimeError), match="byte resource limit"):
        core.IsaPointChargeOperators(water, True, matrix, 8, 512)
    with pytest.raises((ValueError, RuntimeError), match="source point resource limit"):
        core.IsaPointChargeOperators(water, True, matrix, 512 * 1024**2, 2)
    with pytest.raises((ValueError, RuntimeError), match="duplicate"):
        core.IsaPointChargeOperators(water, True, core.Matrix.from_array(np.vstack([POINTS, POINTS[0]])),
                                     512 * 1024**2, 512)
    with pytest.raises((ValueError, RuntimeError)):
        core.IsaPointChargeOperators(water, True, core.Matrix.from_array(POINTS[:, :2]),
                                     512 * 1024**2, 512)


# ----------------------------------------------------------------------------
# PFIT wiring: interface acceptance only, with a synthetic declared model
# ----------------------------------------------------------------------------
def synthetic_problem(out, frequency_index=0):
    """A one-channel, one-parameter model DECLARED BY THIS TEST.

    No channel, site, parameter count or model convention is taken from the
    historical target, from a final Cn, or from any other track. This test
    proves the packed targets and provenance are accepted by the owned solver;
    it is not a physical charge/polarizability model.
    """
    n = out.npoint
    fields = np.linalg.norm(out.points_bohr, axis=1).reshape(n, 1)**-3
    problem = core.IsaPfitProblem()
    model = core.IsaPfitModel()
    model.channel_labels = ["synthetic_isotropic_channel"]
    model.parameter_labels = ["synthetic_scale"]
    model.parameter_units = ["atomic_units"]
    tensor = core.IsaPfitMatrix()
    tensor.rows, tensor.cols = 1, 1
    tensor.values = [1.]
    model.parameter_tensors = [tensor]
    model.fixed = [False]
    model.fixed_values = [0.]
    model.provenance = ("test-only synthetic single isotropic channel declared by "
                        "test_isapol_native_point_response; not a physical model")
    problem.model = model
    problem.batches = [out.batch("native_direct_point_targets", frequency_index, fields)]
    penalty = core.IsaPfitMatrixPenalty()
    zero = core.IsaPfitMatrix()
    zero.rows, zero.cols = 1, 1
    zero.values = [0.]
    penalty.matrix = zero
    penalty.anchor = [0.]
    problem.penalty = penalty
    problem.target_provenance = out.target_provenance("native direct-OV point response, this test")
    return problem


def test_pfit_accepts_native_direct_point_targets(water, response):
    out = npr.native_point_charge_response(response, water, POINTS)
    problem = synthetic_problem(out)
    provenance = problem.target_provenance
    assert provenance.origin == core.IsaPfitTargetOrigin.NativeDirectActualPointResponse
    assert provenance.convention == (
        core.IsaPfitTargetConvention.NegativeInducedPotentialPerUnitSourceChargeAtomicUnits)
    assert provenance.response_representation == npr.REPRESENTATION
    assert provenance.auxiliary_basis_id == ""
    options = core.IsaPfitOptions()
    options.solver = core.IsaPfitSolver.NormalEquationsDSYSV
    result = core.isa_pfit_solve(problem, options)
    assert result.status == core.IsaPfitStatus.Solved
    assert result.diagnostics.data_rows == out.npoint * (out.npoint + 1) // 2
    assert result.diagnostics.objective_available
    assert not result.diagnostics.native_verified  # synthetic model, not a native claim
    assert np.isfinite(result.parameters).all()
    assert np.asarray(problem.batches[0].targets) == pytest.approx(
        np.asarray(out.packed_targets[0]), abs=0.)


def test_pfit_rejects_false_native_direct_provenance(water, response):
    out = npr.native_point_charge_response(response, water, POINTS)
    options = core.IsaPfitOptions()
    options.solver = core.IsaPfitSolver.NormalEquationsDSYSV
    for mutate in (lambda p: setattr(p, "auxiliary_basis_id", "aug-cc-pvtz-ri"),
                   lambda p: setattr(p, "response_representation", "fitted_density_coefficients"),
                   lambda p: setattr(p, "response_representation", "")):
        problem = synthetic_problem(out)
        mutated = problem.target_provenance
        mutate(mutated)
        problem.target_provenance = mutated
        with pytest.raises((ValueError, RuntimeError)):
            core.isa_pfit_solve(problem, options)


def test_target_provenance_and_batch_input_guards(water, response):
    out = npr.native_point_charge_response(response, water, POINTS, frequencies=(0., 0.3))
    for bad in ("", "   ", None, 5):
        with pytest.raises(ValueError):
            out.target_provenance(bad)
    fields = np.ones((out.npoint, 1))
    for bad in (-1, 2, 1.5, True, "0"):
        with pytest.raises(ValueError):
            out.batch("label", bad, fields)
    for bad in ("", "  ", None):
        with pytest.raises(ValueError):
            out.batch(bad, 0, fields)
    for bad in (np.ones(out.npoint), np.ones((out.npoint + 1, 1)), np.ones((out.npoint, 0)),
                np.full((out.npoint, 1), np.nan), np.ones((out.npoint, 1)) * 1.j):
        with pytest.raises(ValueError, match="design matrix"):
            out.batch("label", 0, bad)
    assert out.batch("label", 1, fields).targets == pytest.approx(
        list(out.packed_targets[1]), abs=0.)
