"""Independent supplied PFIT mathematics, not native molecular acceptance.

No reference-file parser, target reconstruction from the fitted tensors, or
availability skip: parent must build the new API before running this file.
"""
import math

import numpy as np
import pytest
import psi4

c = psi4.core


def matrix(a):
    a = np.asarray(a, dtype=float)
    m = c.IsaPfitMatrix()
    m.rows, m.cols = a.shape
    m.values = a.ravel().tolist()
    return m


def array(m):
    return np.array(m.values).reshape(m.rows, m.cols)


def batch(label, fields, targets):
    b = c.IsaPfitBatch()
    b.label = label
    b.points_bohr = [[float(i), 0., 0.] for i in range(len(fields))]
    b.fields = matrix(fields)
    b.targets = targets
    return b


def problem(tensors=None, batches=None, penalty=None, anchor=None, fixed=None):
    tensors = np.array([[[1.]]]) if tensors is None else np.asarray(tensors)
    n, nc, _ = tensors.shape
    p = c.IsaPfitProblem()
    m = c.IsaPfitModel()
    m.channel_labels = [f"channel{i}" for i in range(nc)]
    m.parameter_labels = [f"parameter{i}" for i in range(n)]
    m.parameter_units = ["atomic_units"] * n
    m.parameter_tensors = [matrix(t) for t in tensors]
    m.fixed = [False] * n if fixed is None else fixed
    m.fixed_values = [3.] * n
    m.provenance = "test-only explicit global Cartesian channels, full listed tensors; no localization"
    p.model = m
    p.batches = [batch("one", [[1.]], [2.])] if batches is None else batches
    prior = c.IsaPfitMatrixPenalty()
    prior.matrix = matrix(np.zeros((n, n)) if penalty is None else penalty)
    prior.anchor = [0.] * n if anchor is None else anchor
    p.penalty = prior
    provenance = c.IsaPfitTargetProvenance()
    provenance.origin = c.IsaPfitTargetOrigin.SyntheticAnalyticTest
    provenance.convention = c.IsaPfitTargetConvention.NegativeInducedPotentialPerUnitSourceChargeAtomicUnits
    provenance.source_id = "independent analytic test"
    provenance.generation_record = "synthetic canonical -dphi_induced/dq, Eh/e^2; no energy half or bare electrostatics"
    p.target_provenance = provenance
    return p


def lc(t, a, s=1.):
    l = c.IsaPfitLinearPenalty()
    l.coefficients, l.target, l.strength = t, a, s
    return l


@pytest.fixture(params=["NormalEquationsDSYSV", "StreamingQR"])
def options(request):
    o = c.IsaPfitOptions()
    o.solver = getattr(c.IsaPfitSolver, request.param)
    return o


def solve(p, o):
    r = c.isa_pfit_solve(p, o)
    assert r.status in (c.IsaPfitStatus.Solved, c.IsaPfitStatus.AllFixed)
    assert r.diagnostics.objective_available
    assert not r.diagnostics.native_verified
    return r


def test_pair_oracle(options):
    p = problem(batches=[batch("A", [[1.], [1.]], [1., 2., 3.]), batch("B", [[1.]], [10.])])
    r = solve(p, options)
    assert r.diagnostics.data_rows == 4
    assert [b.rows for b in r.diagnostics.batches] == [3, 1]
    np.testing.assert_allclose(array(r.normal_matrix), [[4.]])
    np.testing.assert_allclose(r.normal_rhs, [16.])
    np.testing.assert_allclose(r.parameters, [4.])
    assert r.diagnostics.data_sse == pytest.approx(50.)
    assert r.diagnostics.data_rms == pytest.approx(math.sqrt(12.5))
    assert r.diagnostics.data_max_residual == pytest.approx(6.)
    assert [b.sse for b in r.diagnostics.batches] == pytest.approx([14., 36.])


def test_lc_rhs_and_constants(options):
    p = problem()
    p.linear_penalties = [lc([2.], 5., 3.)]
    r = solve(p, options)
    np.testing.assert_allclose(array(r.normal_matrix), [[13.]])
    np.testing.assert_allclose(r.normal_rhs, [32.])
    np.testing.assert_allclose(r.parameters, [32/13])
    np.testing.assert_allclose(r.lc_penalty_rhs, [30.])
    np.testing.assert_allclose(array(r.lc_penalty_matrix), [[12.]])
    np.testing.assert_allclose(array(r.effective_penalty_matrix), [[12.]])
    np.testing.assert_allclose(r.effective_penalty_rhs, [30.])
    p.linear_penalties = [lc([1.], 1.), lc([1.], 3.), lc([0.], 4., 2.)]
    r = solve(p, options)
    np.testing.assert_allclose(r.parameters, [2.])
    assert r.diagnostics.lc_objective == pytest.approx(34.)
    assert r.diagnostics.total_objective == pytest.approx(34.)


def test_fixed_correlated_prior(options):
    p = problem([np.diag([1., 0.]), np.diag([0., 1.])],
                [batch("x", [[1., 0.]], [2.]), batch("y", [[0., 1.]], [5.])],
                [[2., 1.], [1., 2.]], [1., -1.], [True, False])
    r = solve(p, options)
    assert r.diagnostics.free_indices == [1]
    np.testing.assert_allclose(array(r.normal_matrix), [[3.]])
    np.testing.assert_allclose(r.normal_rhs, [1.], atol=1e-14)
    np.testing.assert_allclose(r.parameters, [3., 1/3])
    np.testing.assert_allclose(r.matrix_penalty_rhs, [1., -1.], atol=1e-14)
    assert r.parameters[0] == 3.


def test_channel_symmetry_not_pair_weight(options):
    p = problem([[[0., 1.], [1., 0.]]], [batch("xy", [[1., 0.], [0., 1.]], [0., 7., 0.])])
    r = solve(p, options)
    np.testing.assert_allclose(array(r.normal_matrix), [[1.]])
    np.testing.assert_allclose(r.parameters, [7.])
    assert r.diagnostics.data_sse == pytest.approx(0.)


@pytest.mark.parametrize("sign,success", [(-1., True), (1., False)])
def test_singular_prior(options, sign, success):
    p = problem([[[1.]], [[sign]]], [batch("one", [[1.]], [2.])], [[1., 1.], [1., 1.]])
    r = c.isa_pfit_solve(p, options)
    if success:
        assert r.status == c.IsaPfitStatus.Solved
        np.testing.assert_allclose(r.parameters, [1., -1.], atol=1e-12)
    else:
        assert r.status == c.IsaPfitStatus.RankDeficient
        assert r.diagnostics.numerical_rank == 1
        with pytest.raises(RuntimeError, match="unavailable"):
            _ = r.parameters


def test_all_fixed(options):
    p = problem(fixed=[True], penalty=[[2.]], anchor=[1.])
    p.linear_penalties = [lc([0.], 4., 2.)]
    r = solve(p, options)
    assert r.status == c.IsaPfitStatus.AllFixed
    assert r.parameters == [3.]
    assert r.normal_matrix.rows == 0
    assert r.diagnostics.data_sse == pytest.approx(1.)
    assert r.diagnostics.matrix_objective == pytest.approx(8.)
    assert r.diagnostics.lc_objective == pytest.approx(32.)
    assert r.diagnostics.total_objective == pytest.approx(41.)


@pytest.mark.parametrize("chunk", [1, 2, 4, 7, 100])
def test_independent_dense_augmented_oracle(options, chunk):
    # Literal independent design for diag K0, diag K1, symmetric offdiag K2.
    # Physical batches: three points [(1,0),(0,1),(1,1)], then two [(2,1),(1,-1)].
    A = np.array([[1,0,0], [0,0,1], [0,1,0], [1,0,1], [0,1,1], [1,1,2],
                  [4,1,4], [2,-1,-1], [1,1,-2]], dtype=float)
    y = np.array([1,2,3,4,5,6,2,-1,3.])
    B = np.array([[1,2,0], [0,1,1], [1,0,1.]])
    P = B.T @ B
    h = np.array([1., -1., 2.])
    t = np.array([2., 1., -1.])
    aug = np.vstack([A, B, 2*t, np.zeros(3)])
    target = np.r_[y, B@h, 6., 5.]
    # Fixed p0=3. Eliminate from the independent augmented system.
    expect = np.r_[3., np.linalg.lstsq(aug[:, 1:], target-aug[:, 0]*3., rcond=None)[0]]
    p = problem([np.diag([1., 0.]), np.diag([0., 1.]), [[0., 1.], [1., 0.]]],
                [batch("a", [[1,0], [0,1], [1,1]], y[:6].tolist()),
                 batch("b", [[2,1], [1,-1]], y[6:].tolist())], P, h.tolist(), [True,False,False])
    p.linear_penalties = [lc(t.tolist(), 3., 4.), lc([0.,0.,0.], 5.)]
    options.qr_chunk_rows = chunk
    options.retain_pair_predictions = True
    r = solve(p, options)
    np.testing.assert_allclose(r.parameters, expect, atol=2e-13)
    np.testing.assert_allclose(array(r.normal_matrix), aug[:, 1:].T@aug[:, 1:], atol=2e-13)
    np.testing.assert_allclose(r.normal_rhs, aug[:, 1:].T@(target-aug[:, 0]*3), atol=2e-13)
    effective_p = array(r.matrix_penalty_matrix)
    np.testing.assert_array_equal(effective_p, effective_p.T)
    np.testing.assert_allclose(effective_p, P, atol=2e-13)
    assert r.diagnostics.penalty_correction_max == np.max(np.abs(effective_p-P))
    np.testing.assert_allclose(r.matrix_penalty_rhs, P@h, atol=2e-13)
    np.testing.assert_allclose(array(r.lc_penalty_matrix), 4*np.outer(t, t), atol=2e-13)
    np.testing.assert_allclose(r.lc_penalty_rhs, 12*t, atol=2e-13)
    np.testing.assert_allclose(array(r.effective_penalty_matrix), P+4*np.outer(t, t), atol=2e-13)
    np.testing.assert_allclose(r.effective_penalty_rhs, P@h+12*t, atol=2e-13)
    np.testing.assert_allclose(np.concatenate(r.predictions), A@expect, atol=2e-13)
    d = r.diagnostics
    assert d.data_sse == pytest.approx(np.linalg.norm(A@expect-y)**2)
    assert d.matrix_objective == pytest.approx(np.linalg.norm(B@(expect-h))**2)
    assert d.lc_objective == pytest.approx(4*(t@expect-3)**2+25)
    assert d.total_objective == pytest.approx(np.linalg.norm(aug@expect-target)**2)
    assert d.backward_residual < 1e-13
    if options.solver == c.IsaPfitSolver.StreamingQR:
        assert d.qr_discarded_rhs_sse == pytest.approx(d.total_objective)
        assert "singular values" in d.rank_method
        assert d.qr_r_rcond > 0
    else:
        assert "normal H" in d.rank_method
        assert d.normal_h_rcond > 0
        assert d.normal_h_norm1 == pytest.approx(np.linalg.norm(array(r.normal_matrix), 1))
        assert d.condition_estimate_available


def test_ownership(options):
    p = problem()
    original = matrix([[1.]])
    m = p.model
    m.parameter_tensors = [original]
    p.model = m
    original.values = [99.]
    m.fixed = [True]
    options.retain_pair_predictions = True
    r = solve(p, options)
    p.batches = [batch("changed", [[1.]], [50.])]
    snap = r.normal_matrix
    snap.values = [100.]
    pred = r.predictions
    pred[0][0] = -20.
    diag = r.diagnostics
    diag.data_sse = 123.
    prov = r.target_provenance
    prov.source_id = "changed"
    assert r.parameters == pytest.approx([2.])
    assert r.predictions == [[2.]]
    assert r.normal_matrix.values == [1.]
    assert r.diagnostics.data_sse == 0.
    assert r.target_provenance.source_id != "changed"


def test_nearly_dependent_no_hidden_fallback(options):
    eps = 1e-7
    p = problem([[[1.,0.],[0.,1.]], [[1.,0.],[0.,1.+eps]]],
                [batch("a", [[1.,0.]], [1.]), batch("b", [[0.,1.]], [1.])])
    options.minimum_solver_rcond = 1e-5
    r = c.isa_pfit_solve(p, options)
    assert r.status in (c.IsaPfitStatus.RankDeficient, c.IsaPfitStatus.IllConditioned)
    assert r.normal_matrix.rows == 2  # never pruned
    assert r.diagnostics.rank_smallest < 1e-6*r.diagnostics.rank_largest
    with pytest.raises(RuntimeError):
        _ = r.parameters


@pytest.mark.parametrize("bad", ["nan", "inf", "overflow", "packed", "empty_targets", "fields", "tensor", "tensor_asym", "symmetry", "indefinite",
                                  "negative_strength", "empty", "origin", "convention", "source", "generation",
                                  "fitted_basis", "fitted_representation", "frequency", "labels", "anchor", "lc_shape"])
def test_invalid_inputs(options, bad):
    p = problem()
    if bad in ("nan", "inf", "overflow", "packed", "empty_targets", "fields"):
        b = p.batches[0]
        if bad == "nan": b.targets = [float("nan")]
        if bad == "inf": b.points_bohr = [[float("inf"), 0., 0.]]
        if bad == "overflow": b.fields = matrix([[1e308]])
        if bad == "packed": b.targets = [1., 2.]
        if bad == "empty_targets": b.targets = []
        if bad == "fields": b.fields = matrix([[1., 2.]])
        p.batches = [b]
    elif bad == "tensor":
        m = p.model; m.parameter_tensors = [matrix([[1.,2.]])]; p.model = m
    elif bad == "tensor_asym":
        p = problem([[[1., 1e-20], [0., 1.]]], [batch("a", [[1., 0.]], [2.])])
    elif bad in ("symmetry", "indefinite"):
        if bad == "symmetry":
            p = problem([[[1.]], [[2.]]], penalty=[[1., 1e-20], [0., 1.]])
        else:
            prior = p.penalty; prior.matrix = matrix([[-1e-200]]); p.penalty = prior
    elif bad == "negative_strength": p.linear_penalties = [lc([1.], 2., -1.)]
    elif bad == "empty": p.batches = []
    elif bad == "frequency": p.frequency_au = -1.
    elif bad == "labels":
        m = p.model; m.channel_labels = [""]; p.model = m
    elif bad == "anchor":
        prior = p.penalty; prior.anchor = []; p.penalty = prior
    elif bad == "lc_shape": p.linear_penalties = [lc([], 1.)]
    else:
        pr = p.target_provenance
        if bad == "origin": pr.origin = c.IsaPfitTargetOrigin.Unspecified
        if bad == "convention": pr.convention = c.IsaPfitTargetConvention.Unspecified
        if bad == "source": pr.source_id = " "
        if bad == "generation": pr.generation_record = ""
        if bad.startswith("fitted"):
            pr.origin = c.IsaPfitTargetOrigin.SuppliedFittedPropagatorPointResponse
            pr.response_representation = "wrong" if bad == "fitted_representation" else "fitted_density_coefficients"
            pr.auxiliary_basis_id = "basis" if bad == "fitted_representation" else ""
        p.target_provenance = pr
    with pytest.raises((ValueError, RuntimeError), match="PFIT"):
        c.isa_pfit_solve(p, options)


@pytest.mark.parametrize("setting,value", [("maximum_work_bytes", 1), ("qr_chunk_rows", 0),
    ("qr_chunk_rows", 2**63), ("rank_relative_tolerance", float("nan")), ("minimum_solver_rcond", -1.)])
def test_bad_resources(options, setting, value):
    setattr(options, setting, value)
    with pytest.raises((ValueError, RuntimeError), match="PFIT"):
        c.isa_pfit_solve(problem(), options)


def test_dimension_arithmetic_overflow(options):
    p = problem()
    m = p.penalty.matrix
    m.rows = 2**63
    m.cols = 2**63
    prior = p.penalty; prior.matrix = m; p.penalty = prior
    with pytest.raises((ValueError, RuntimeError), match="PFIT"):
        c.isa_pfit_solve(p, options)


def test_synthetic_actual_gaussian_targets_not_truncated_model(options):
    # Analytic diffuse Cartesian p_z Gaussian chi=z exp(-alpha*r^2).
    # Its Coulomb potential on +z: W(R)=pi^(3/2)/(2 alpha^(5/2) R^2)
    # * [erf(sqrt(alpha)R)-2 sqrt(alpha)R exp(-alpha R^2)/sqrt(pi)].
    # Synthetic coefficient response C=-gamma gives canonical v=gamma W_i W_j.
    # Truncated point dipole model instead uses f=1/R^2: finite-size penetration
    # makes targets genuinely outside that model, with no native acceptance claim.
    alpha, gamma = .7, .4
    radii = np.array([.5, 1., 2., 4.])
    w = np.array([math.pi**1.5/(2*alpha**2.5*r*r) *
                  (math.erf(math.sqrt(alpha)*r)-2*math.sqrt(alpha)*r*math.exp(-alpha*r*r)/math.sqrt(math.pi))
                  for r in radii])
    targets = [gamma*w[i]*w[j] for i in range(4) for j in range(i+1)]
    b = batch("gaussian", (1/radii**2)[:,None], targets)
    b.points_bohr = [[0.,0.,r] for r in radii]
    p = problem(batches=[b])
    pr = p.target_provenance
    pr.source_id = "synthetic analytic diffuse p Gaussian actual point samples"
    pr.generation_record = "analytic p Gaussian Coulomb W, C=-0.4; v=-W C W.T independent of point-dipole K"
    p.target_provenance = pr
    r = solve(p, options)
    A = np.array([1/(radii[i]**2*radii[j]**2) for i in range(4) for j in range(i+1)])
    expect = A@targets/(A@A)
    assert r.parameters == pytest.approx([expect])
    assert r.diagnostics.data_sse == pytest.approx(np.linalg.norm(A*expect-targets)**2)
    assert r.diagnostics.data_sse > .01
    assert r.target_provenance.origin == c.IsaPfitTargetOrigin.SyntheticAnalyticTest
    assert not r.diagnostics.native_verified


def test_small_positive_prior_mode_is_not_truncated(options):
    p = problem([np.diag([1., 0.]), np.diag([0., 1.])],
                [batch("x", [[1., 0.]], [2.]), batch("y", [[0., 1.]], [5.])],
                [[1., 0.], [0., 1e-24]], [0., 1e12])
    r = solve(p, options)
    assert array(r.matrix_penalty_matrix)[1, 1] == pytest.approx(1e-24, rel=1e-14, abs=0.)
    assert r.diagnostics.penalty_min_eigenvalue == pytest.approx(1e-24, rel=1e-14, abs=0.)
    assert r.diagnostics.matrix_objective > .99


def test_fitted_origin_is_only_caller_declaration(options):
    p = problem()
    pr = p.target_provenance
    pr.origin = c.IsaPfitTargetOrigin.SuppliedFittedPropagatorPointResponse
    pr.response_representation = "fitted_density_coefficients"
    pr.auxiliary_basis_id = "explicit synthetic AUX identity"
    p.target_provenance = pr
    r = solve(p, options)
    assert r.target_provenance.auxiliary_basis_id == pr.auxiliary_basis_id
    assert not r.diagnostics.native_verified


def test_prediction_and_chunk_budget_accounting(options):
    p = problem()
    options.qr_chunk_rows = 1
    a = solve(p, options).diagnostics.work_budget_bytes
    options.retain_pair_predictions = True
    b = solve(p, options).diagnostics.work_budget_bytes
    assert b == a + 8  # exactly one retained numerical prediction
    options.qr_chunk_rows = 11
    d = solve(p, options).diagnostics.work_budget_bytes
    if options.solver == c.IsaPfitSolver.StreamingQR:
        assert d == b + 10 * 2 * 8
    else:
        assert d == b  # no unused chunk allocation in the normal path
