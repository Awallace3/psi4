# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Complete-cloud design rows; small independent COPY-aware algebra."""
import numpy as np
import pytest


def row_fixture():
    from psi4 import core

    def matrix(values):
        values = np.asarray(values, dtype=float)
        result = core.IsaPfitMatrix()
        result.rows, result.cols = values.shape
        result.values = values.ravel().tolist()
        return result

    problem = core.IsaPfitRowProblem()
    model = core.IsaPfitRowModel()
    model.channel_labels = ["supplied-channel"]
    model.parameter_labels = ["a", "b"]
    model.parameter_units = ["atomic_units", "atomic_units"]
    model.fixed, model.fixed_values = [False, False], [0., 0.]
    model.provenance = "independent synthetic design; no native certification"
    problem.model = model
    cloud = core.IsaPfitCloudRows()
    cloud.label, cloud.points, cloud.full_row_count = "complete-cloud", 3, 6
    cloud.maximum_block_rows = 2
    problem.cloud = cloud
    prior = core.IsaPfitMatrixPenalty()
    prior.matrix, prior.anchor = matrix([[2., .5], [.5, 1.]]), [.1, -.2]
    problem.penalty = prior
    provenance = core.IsaPfitTargetProvenance()
    provenance.origin = core.IsaPfitTargetOrigin.SyntheticAnalyticTest
    provenance.convention = core.IsaPfitTargetConvention.NegativeInducedPotentialPerUnitSourceChargeAtomicUnits
    provenance.source_id = "small supplied complete cloud"
    provenance.generation_record = "independent canonical targets, no energy half or bare electrostatics"
    problem.target_provenance = provenance
    design = np.array([[1., 0.], [2., 1.], [0., 2.], [3., -1.], [-1., 1.], [1., 2.]])
    targets = np.array([1., .5, -2., 4., .1, -1.])
    return problem, design, targets


@pytest.mark.parametrize("solver", ["NormalEquationsDSYSV", "StreamingQR"])
def test_global_row_solver_matches_independent_penalized_equation(solver):
    from psi4 import core
    problem, design, targets = row_fixture()
    passes = []

    def blocks():
        passes.append(len(passes))
        for start in range(0, 6, 2):
            yield start, design[start:start+2].copy(), targets[start:start+2].copy()

    options = core.IsaPfitOptions()
    options.solver = getattr(core.IsaPfitSolver, solver)
    result = core.isa_pfit_solve_rows(problem, blocks, options)
    penalty = np.array([[2., .5], [.5, 1.]])
    expected = np.linalg.solve(design.T @ design+penalty,
                               design.T @ targets+penalty @ np.array([.1, -.2]))
    np.testing.assert_allclose(result.parameters, expected, atol=1e-12, rtol=1e-12)
    assert passes == [0, 1]
    assert result.diagnostics.data_rows == 6
    assert len(result.diagnostics.batches) == 1
    np.testing.assert_allclose(result.diagnostics.data_sse,
                               np.sum((design @ expected-targets)**2), rtol=1e-12)


@pytest.mark.parametrize("change", ["design", "target", "partition"])
def test_global_row_solver_checks_replay_content_not_block_partition(change):
    from psi4 import core
    problem, design, targets = row_fixture()
    passes = []

    def blocks():
        replay = bool(passes)
        passes.append(1)
        a, b = design.copy(), targets.copy()
        if replay and change == "design":
            a[0, 0] += .1
        if replay and change == "target":
            b[-1] += .1
        step = 1 if replay else 2
        for start in range(0, 6, step):
            yield start, a[start:start+step], b[start:start+step]

    if change == "partition":
        assert core.isa_pfit_solve_rows(problem, blocks).diagnostics.data_rows == 6
    else:
        with pytest.raises(Exception, match="replay content changed"):
            core.isa_pfit_solve_rows(problem, blocks)


@pytest.mark.parametrize("kind", ["gap", "early", "extra", "oversized",
                                  "empty", "dtype", "strided", "nan", "bool", "negative"])
def test_global_row_solver_rejects_malformed_stream(kind):
    from psi4 import core
    problem, design, targets = row_fixture()

    def blocks():
        if kind == "early":
            return
        if kind == "oversized":
            yield 0, design[:3], targets[:3]
            return
        if kind == "empty":
            yield 0, design[:0], targets[:0]
            return
        for start in range(0, 6, 2):
            a, b = design[start:start+2].copy(), targets[start:start+2].copy()
            index = start
            if start == 0:
                if kind == "gap":
                    index = 1
                elif kind == "dtype":
                    a = a.astype(np.float32)
                elif kind == "strided":
                    a = a[:, ::-1]
                elif kind == "nan":
                    b[0] = np.nan
                elif kind == "bool":
                    index = False
                elif kind == "negative":
                    index = -1
            yield index, a, b
        if kind == "extra":
            yield 6, design[:1], targets[:1]

    with pytest.raises(Exception):
        core.isa_pfit_solve_rows(problem, blocks)


def test_global_row_solver_snapshots_options_and_admits_before_callbacks():
    from psi4 import core
    problem, design, targets = row_fixture()
    options = core.IsaPfitOptions()
    passes = []

    def blocks():
        passes.append(1)
        model = problem.model
        model.fixed = []
        problem.model = model
        options.retain_pair_predictions = True
        for start in range(0, 6, 2):
            yield start, design[start:start+2], targets[start:start+2]

    options.maximum_work_bytes = 1
    with pytest.raises(Exception, match="maximum_work_bytes"):
        core.isa_pfit_solve_rows(problem, blocks, options)
    assert not passes
    options.maximum_work_bytes = 256*1024**2
    result = core.isa_pfit_solve_rows(problem, blocks, options)
    assert result.status == core.IsaPfitStatus.Solved
    assert not result.settings.retain_pair_predictions
    assert len(passes) == 2


@pytest.mark.parametrize("block_rows", [1, 4, 8])
def test_complete_cloud_design_keeps_cross_block_pairs_and_copy_columns(block_rows):
    from psi4.driver.procrouting import isapol_refine as R
    from psi4.driver.procrouting.isapol_pfit_stream import PackedDesignRows
    frame = tuple(map(tuple, np.eye(3)))
    sites = (R.RefinementSite("H1", "H", (0., 0., 0.), frame, 1),
             R.RefinementSite("H2", "H", (1., 0., 0.), frame, 1))
    anchor = np.zeros((4, 4))
    anchor[1, 1], anchor[2, 2] = 1., 2.
    anchor[1, 2] = anchor[2, 1] = .3
    model = R.refinement_model(sites, [anchor, anchor.copy()], cutoff=1.e-4,
                               weight_type=4, weight_coefficient=1.e-5,
                               provenance="small explicit COPY row test")
    assert model.parameter_count == 3
    fields = (np.arange(40., dtype=float).reshape(5, 8)-11)/8
    expected = []
    for i in range(5):
        for j in range(i+1):
            expected.append([
                fields[i, 1]*fields[j, 1]+fields[i, 5]*fields[j, 5],
                fields[i, 1]*fields[j, 2]+fields[i, 2]*fields[j, 1]+
                fields[i, 5]*fields[j, 6]+fields[i, 6]*fields[j, 5],
                fields[i, 2]*fields[j, 2]+fields[i, 6]*fields[j, 6]])
    source = PackedDesignRows(model, fields, block_rows=block_rows)
    assert source.row_count == 15
    fields[:] = 0.  # Replay must own its source fields.
    for _ in range(2):
        blocks = list(source.blocks())
        assert [start for start, _ in blocks] == list(range(0, 15, block_rows))
        assert all(len(values) <= block_rows for _, values in blocks)
        np.testing.assert_allclose(np.concatenate([values for _, values in blocks]), expected)
    # Two independent clouds of sizes 2 and 3 would supply only 3+6 rows.
    assert source.row_count > 2*3//2+3*4//2


def test_design_admission_replay_and_benzene_pair_count():
    from psi4.driver.procrouting import isapol_refine as R
    from psi4.driver.procrouting.isapol_pfit_stream import PackedDesignRows
    frame = tuple(map(tuple, np.eye(3)))
    sites = (R.RefinementSite("C1", "C", (0., 0., 0.), frame, 1),)
    anchor = np.zeros((4, 4))
    anchor[1, 1] = 1.
    model = R.refinement_model(sites, [anchor], cutoff=1.e-4, weight_type=4,
                               weight_coefficient=1.e-5, provenance="row budget test")
    fields = np.ones((5, 4))
    source = PackedDesignRows(model, fields, block_rows=4)
    with pytest.raises(ValueError, match="byte resource"):
        PackedDesignRows(model, fields, block_rows=4, max_bytes=source.planned_bytes-1)
    exact = PackedDesignRows(model, fields, block_rows=4, max_bytes=source.planned_bytes)
    first, second = exact.blocks(), exact.blocks()
    np.testing.assert_array_equal(next(first)[1], next(second)[1])
    assert exact.passes_used == 2
    with pytest.raises(RuntimeError, match="pass budget"):
        exact.blocks()
    # Admission only: do not materialize the two-million-row design in-repo.
    full = PackedDesignRows(model, np.ones((2000, 4)), block_rows=4)
    assert full.row_count == 2_001_000
    assert full.planned_bytes < 200_000


def test_design_chunked_products_replay_bit_for_bit():
    from psi4.driver.procrouting import isapol_refine as R
    from psi4.driver.procrouting.isapol_pfit_stream import PackedDesignRows
    frame = tuple(map(tuple, np.eye(3)))
    sites = tuple(R.RefinementSite(f"C{k}", "C", (float(k), 0., 0.), frame, 2) for k in range(2))
    anchor = np.diag(np.linspace(2., 1., 9))
    anchor[0, 4] = anchor[4, 0] = anchor[1, 3] = anchor[3, 1] = .5
    model = R.refinement_model(sites, [anchor, anchor.copy()], cutoff=1.e-4,
                               provenance="chunked replay test")
    fields = np.random.default_rng(7).normal(size=(70, model.channel_count))
    pairs = [(i, j) for i in range(70) for j in range(i+1)]
    expected = np.zeros((len(pairs), model.parameter_count))
    for p, entries in enumerate(model.parameter_entries):
        for site, a, b in entries:
            a, b = model.channel_offsets[site]+a, model.channel_offsets[site]+b
            for k, (i, j) in enumerate(pairs):
                expected[k, p] += fields[i, a]*fields[j, b]+(fields[i, b]*fields[j, a] if a != b else 0.)
    source = PackedDesignRows(model, fields, block_rows=256)
    assert 1 < source._chunk < 70  # several products, each spanning several blocks
    first, second = list(source.blocks()), list(source.blocks())
    for (s1, d1), (s2, d2) in zip(first, second):
        assert s1 == s2 and d1.tobytes() == d2.tobytes()
    np.testing.assert_allclose(np.concatenate([d for _, d in first]), expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("block_rows", [0, True, 4097])
def test_design_refuses_invalid_block_sizes(block_rows):
    from psi4.driver.procrouting import isapol_refine as R
    from psi4.driver.procrouting.isapol_pfit_stream import PackedDesignRows
    site = R.RefinementSite("H", "H", (0., 0., 0.), tuple(map(tuple, np.eye(3))), 1)
    model = R.refinement_model([site], [np.eye(4)], cutoff=1.e-4, weight_type=4,
                               weight_coefficient=1.e-5, provenance="invalid block test")
    with pytest.raises(ValueError, match="block_rows"):
        PackedDesignRows(model, np.ones((3, 4)), block_rows=block_rows)


def test_design_rejects_malformed_model_tables():
    from dataclasses import replace
    from psi4.driver.procrouting import isapol_refine as R
    from psi4.driver.procrouting.isapol_pfit_stream import PackedDesignRows
    site = R.RefinementSite("H", "H", (0., 0., 0.), tuple(map(tuple, np.eye(3))), 1)
    anchor = np.zeros((4, 4))
    anchor[1, 1] = 1.
    model = R.refinement_model([site], [anchor], cutoff=1.e-4, weight_type=4,
                               weight_coefficient=1.e-5, provenance="model refusal test")
    for changes in (
            {"channel_offsets": (.5,)}, {"channel_offsets": (1,)},
            {"channel_offsets": ()}, {"parameter_entries": ()},
            {"parameter_entries": ((),)}, {"parameter_entries": (((0, 1),),)},
            {"parameter_entries": (((2, 1, 1),),)},
            {"parameter_entries": (((0, 1, 4),),)}):
        with pytest.raises(ValueError):
            PackedDesignRows(replace(model, **changes), np.ones((3, 4)))


def test_design_work_is_global_and_failed_pass_is_not_refunded(monkeypatch):
    from psi4.driver.procrouting import isapol_refine as R
    from psi4.driver.procrouting.isapol_pfit_stream import PackedDesignRows
    site = R.RefinementSite("H", "H", (0., 0., 0.), tuple(map(tuple, np.eye(3))), 1)
    anchor = np.zeros((4, 4))
    anchor[1, 1] = 1.
    model = R.refinement_model([site], [anchor], cutoff=1.e-4, weight_type=4,
                               weight_coefficient=1.e-5, provenance="global work test")
    fields = np.ones((5, 4))
    planned = PackedDesignRows(model, fields).planned_work
    monkeypatch.setattr(PackedDesignRows, "MAX_WORK", planned)
    for block in (1, 4, 4096):
        assert PackedDesignRows(model, fields, block_rows=block).planned_work == planned
    monkeypatch.setattr(PackedDesignRows, "MAX_WORK", planned-1)
    for block in (1, 4, 4096):
        with pytest.raises(ValueError, match="replay work"):
            PackedDesignRows(model, fields, block_rows=block)
    source = PackedDesignRows(model, np.full((1, 4), 1.e308), max_passes=1)
    with pytest.raises(FloatingPointError):
        next(source.blocks())
    assert source.passes_used == 1
    with pytest.raises(RuntimeError, match="pass budget"):
        source.blocks()


@pytest.mark.parametrize("solver", ["NormalEquationsDSYSV", "StreamingQR"])
def test_shared_design_problems_match_their_lone_fits(solver):
    from psi4 import core
    problem, design, targets = row_fixture()
    second, _, _ = row_fixture()
    second.frequency_au = .4
    prior = second.penalty
    prior.anchor = [.3, .05]
    second.penalty = prior
    stacked = np.column_stack((targets, 2*targets[::-1]+.3))
    options = core.IsaPfitOptions()
    options.solver = getattr(core.IsaPfitSolver, solver)
    options.retain_pair_predictions = True
    passes = []

    def blocks():
        passes.append(1)
        for start in range(0, 6, 2):
            yield start, design[start:start+2].copy(), stacked[start:start+2].copy()

    results = core.isa_pfit_solve_rows_multi([problem, second], blocks, options)
    assert len(passes) == 2 and [r.frequency_au for r in results] == [0., .4]
    for t, (p, result) in enumerate(zip([problem, second], results)):
        def lone():
            for start in range(0, 6, 2):
                yield start, design[start:start+2].copy(), stacked[start:start+2, t].copy()
        alone = core.isa_pfit_solve_rows(p, lone, options)
        np.testing.assert_allclose(result.parameters, alone.parameters, rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(result.predictions, alone.predictions, rtol=1e-14, atol=1e-15)
        assert result.diagnostics.data_sse == pytest.approx(alone.diagnostics.data_sse, rel=1e-13)
        assert result.diagnostics.total_objective == pytest.approx(alone.diagnostics.total_objective, rel=1e-13)
    assert not np.allclose(results[0].parameters, results[1].parameters)
    model = second.model
    model.parameter_labels = ["a", "c"]
    second.model = model
    with pytest.raises(Exception, match="share one model"):
        core.isa_pfit_solve_rows_multi([problem, second], blocks, options)
    with pytest.raises(Exception):
        core.isa_pfit_solve_rows_multi([problem, problem], lambda: iter([(0, design, targets)]), options)


def _constrained_pair(calls=None):
    """Two RHS sharing one design, with one fixed parameter and one LC row."""
    from psi4 import core
    problem, design, targets = row_fixture()
    model = problem.model
    model.fixed, model.fixed_values = [True, False], [.25, 0.]
    problem.model = model
    constraint = core.IsaPfitLinearPenalty()
    constraint.coefficients, constraint.target, constraint.strength = [1., 2.], .5, 3.
    problem.linear_penalties = [constraint]
    second, _, _ = row_fixture()
    second.model, second.linear_penalties = model, [constraint]
    prior = second.penalty
    prior.anchor = [-.4, .7]
    second.penalty = prior
    stacked = np.column_stack((targets, targets[::-1]-1.))

    def blocks():
        if calls is not None:
            calls.append(1)
        for start in range(0, 6, 2):
            yield start, design[start:start+2].copy(), stacked[start:start+2].copy()

    return problem, second, design, stacked, blocks


@pytest.mark.parametrize("solver", ["NormalEquationsDSYSV", "StreamingQR"])
def test_rows_multi_fixed_parameter_and_linear_constraint(solver):
    """Fixed values and LC rows on the streamed path, against one augmented lstsq per RHS."""
    from psi4 import core
    problem, second, design, stacked, blocks = _constrained_pair()
    options = core.IsaPfitOptions()
    options.solver = getattr(core.IsaPfitSolver, solver)
    results = core.isa_pfit_solve_rows_multi([problem, second], blocks, options)
    root = np.linalg.cholesky(np.array([[2., .5], [.5, 1.]])).T
    lc = np.sqrt(3.)*np.array([1., 2.])
    for t, (anchor, result) in enumerate(zip(([.1, -.2], [-.4, .7]), results)):
        rows = np.vstack([design, root, lc])
        rhs = np.r_[stacked[:, t], root @ np.array(anchor), np.sqrt(3.)*.5]
        free = np.linalg.lstsq(rows[:, 1:], rhs-rows[:, 0]*.25, rcond=None)[0]
        assert result.status == core.IsaPfitStatus.Solved
        np.testing.assert_allclose(result.parameters, [.25, free[0]], rtol=1e-12, atol=1e-13)
        assert result.parameters[0] == .25


@pytest.mark.parametrize("chunk", [256, 100000])
def test_streaming_qr_charges_the_live_column_copy(chunk):
    """StreamingQR keeps its shared chunk buffer, chunk*(nf+nt), alive while each
    problem's column copy, chunk*(nf+1), solves; both are charged before any row
    is requested.  Here nf = 1 (one fixed parameter) and nt = 2."""
    from psi4 import core
    calls = []
    problem, second, _, _, blocks = _constrained_pair(calls)
    options = core.IsaPfitOptions()
    options.solver = core.IsaPfitSolver.StreamingQR
    options.qr_chunk_rows = 1
    reference = core.isa_pfit_solve_rows_multi([problem, second], blocks, options)
    options.qr_chunk_rows = chunk
    results = core.isa_pfit_solve_rows_multi([problem, second], blocks, options)
    required = results[0].diagnostics.work_budget_bytes
    assert required == reference[0].diagnostics.work_budget_bytes+8*(chunk-1)*((1+2)+(1+1))
    for got, want in zip(results, reference):  # chunking buffers rows; it never changes the arithmetic
        assert got.status == core.IsaPfitStatus.Solved
        assert got.diagnostics.work_budget_bytes == required
        assert list(got.parameters) == list(want.parameters)
    options.maximum_work_bytes = required
    exact = core.isa_pfit_solve_rows_multi([problem, second], blocks, options)
    assert [list(r.parameters) for r in exact] == [list(r.parameters) for r in results]
    options.maximum_work_bytes = required-1
    with pytest.raises(Exception, match="maximum_work_bytes"):
        core.isa_pfit_solve_rows_multi([problem, second], blocks, options)
    # What the omitted column-copy buffer used to admit is now refused up front.
    del calls[:]
    options.maximum_work_bytes = required-8*chunk*(1+1)
    with pytest.raises(Exception, match="kernel numerical buffers exceed maximum_work_bytes"):
        core.isa_pfit_solve_rows_multi([problem, second], blocks, options)
    assert not calls


def _water_like_models(frequencies=(0., .6)):
    """O to rank 2 and two COPY-equivalent hydrogens to rank 1, one model per node."""
    from psi4.driver.procrouting import isapol_refine as R
    identity = tuple(map(tuple, np.eye(3)))
    mirror = ((-1., 0., 0.), (0., -1., 0.), (0., 0., 1.))
    sites = (R.RefinementSite("O", "O", (0., 0., 0.), identity, 2),
             R.RefinementSite("H1", "H", (-1.45365196, 0., -1.12168732), mirror, 1),
             R.RefinementSite("H2", "H", (1.45365196, 0., -1.12168732), identity, 1))
    rng = np.random.default_rng(5)
    base_o = rng.normal(size=(9, 9))
    base_h = rng.normal(size=(4, 4))
    models, sources = [], []
    for omega in frequencies:
        scale = 1./(1.+omega*omega)
        anchor_o = scale*(base_o @ base_o.T/9+np.eye(9))
        anchor_h = scale*(base_h @ base_h.T/4+.5*np.eye(4))
        models.append(R.refinement_model(sites, [anchor_o, anchor_h, anchor_h.copy()],
                                         frequency_au=omega, weight_type=3, weight_coefficient=1e-3,
                                         provenance="streamed-vs-dense synthetic"))
        sources.append([anchor_o*1.1, anchor_h*.9, anchor_h*.9])
    return models, sources


def _points(count):
    rng = np.random.default_rng(17)
    v = rng.normal(size=(count, 3))
    return v/np.linalg.norm(v, axis=1)[:, None]*rng.uniform(4., 7., size=(count, 1))


@pytest.mark.parametrize("block_rows", [1, 7, 4096])
def test_streamed_refine_matches_dense_refine_and_an_independent_oracle(block_rows):
    from psi4 import core
    from psi4.driver.procrouting import isapol_refine as R
    from psi4.driver.procrouting.isapol_pfit_stream import refine_streamed
    models, sources = _water_like_models()
    points = _points(23)
    fields = R.channel_fields(points, models[0])
    packed = np.column_stack([R.pack_lower_triangle(R.point_to_point_response(fields, m, s))
                              for m, s in zip(models, sources)])
    declared = dict(target_origin=core.IsaPfitTargetOrigin.SyntheticAnalyticTest,
                    source_id="forward map of a perturbed local model",
                    generation_record="synthetic -dphi/dq, Eh/e^2")
    streamed = refine_streamed(models, points, packed, block_rows=block_rows, **declared)
    # Independent design: each variable's pair row, built from fields and the COPY entries.
    i, j = np.tril_indices(len(points))
    design = np.zeros((len(i), models[0].parameter_count))
    for k, entries in enumerate(models[0].parameter_entries):
        for site, a, b in entries:
            a, b = models[0].channel_offsets[site]+a, models[0].channel_offsets[site]+b
            design[:, k] += fields[i, a]*fields[j, b]+(fields[i, b]*fields[j, a] if a != b else 0.)
    for t, (model, fit) in enumerate(zip(models, streamed)):
        dense = R.refine(model, points, packed[:, t], fields=fields, **declared)
        np.testing.assert_allclose(fit.parameters, dense.parameters, rtol=1e-11, atol=1e-12)
        strengths, anchors = np.diag(model.strengths), np.array(model.anchors)
        oracle = np.linalg.solve(design.T @ design+strengths,
                                 design.T @ packed[:, t]+strengths @ anchors)
        np.testing.assert_allclose(fit.parameters, oracle, rtol=1e-10, atol=1e-11)
        assert fit.status == core.IsaPfitStatus.Solved and fit.model is model
        assert fit.result.frequency_au == model.frequency_au
        assert np.array_equal(fit.refined_tensors[1], fit.refined_tensors[2])
        assert not fit.refined_tensors[0].flags.writeable
        assert fit.result.parameter_units == dense.result.parameter_units
    assert not np.allclose(streamed[0].parameters, streamed[1].parameters)


def test_streamed_refine_admissions_are_exact_and_refuse_before_solving():
    from psi4 import core
    from psi4.driver.procrouting import isapol_refine as R
    from psi4.driver.procrouting.isapol_pfit_stream import PackedDesignRows, refine_streamed
    models, sources = _water_like_models((.3,))
    points = _points(9)
    fields = R.channel_fields(points, models[0])
    packed = R.pack_lower_triangle(R.point_to_point_response(fields, models[0], sources[0]))[:, None]
    declared = dict(target_origin=core.IsaPfitTargetOrigin.SyntheticAnalyticTest,
                    source_id="budget", generation_record="budget boundary")
    # Row-producer plan plus points (24 B each), computed fields and block copies (17 B/row/RHS).
    planned = (PackedDesignRows(models[0], fields, block_rows=8).planned_bytes
               + 24*len(points)+fields.nbytes+17*8)
    refine_streamed(models, points, packed, block_rows=8, max_bytes=planned, **declared)
    with pytest.raises(ValueError, match="byte resource"):
        refine_streamed(models, points, packed, block_rows=8, max_bytes=planned-1, **declared)
    options = core.IsaPfitOptions()
    options.solver = core.IsaPfitSolver.NormalEquationsDSYSV
    required = refine_streamed(models, points, packed, block_rows=8, options=options,
                               **declared)[0].diagnostics.work_budget_bytes
    options.maximum_work_bytes = required
    refine_streamed(models, points, packed, block_rows=8, options=options, **declared)
    options.maximum_work_bytes = required-1
    with pytest.raises(Exception, match="maximum_work_bytes"):
        refine_streamed(models, points, packed, block_rows=8, options=options, **declared)


def test_streamed_refine_refuses_inconsistent_inputs_and_unsolved_fits():
    from dataclasses import replace
    from psi4 import core
    from psi4.driver.procrouting import isapol_refine as R
    from psi4.driver.procrouting.isapol_pfit_stream import refine_streamed
    models, _ = _water_like_models()
    points = _points(5)
    rows = 15
    declared = dict(target_origin=core.IsaPfitTargetOrigin.SyntheticAnalyticTest,
                    source_id="refusals", generation_record="refusals")
    good = np.zeros((rows, 2))
    moved = replace(models[1], sites=(models[1].sites[0], models[1].sites[2], models[1].sites[1]))
    for bad, args in (("share sites", ([models[0], moved], points, good)),
                      ("shape", (models, points, np.zeros((rows, 1)))),
                      ("float64", (models, points, good.astype(np.float32))),
                      ("finite", (models, points, np.full((rows, 2), np.nan))),
                      ("npoint x 3", (models, points[:, :2], good)),
                      ("RefinementModel", ([], points, good))):
        with pytest.raises(ValueError, match=bad):
            refine_streamed(*args, **declared)
    with pytest.raises(ValueError, match="source_id"):
        refine_streamed(models, points, good, **dict(declared, source_id=" "))
    with pytest.raises(ValueError, match="auxiliary_basis_id"):
        refine_streamed(models, points, good, response_representation="fitted_density_coefficients",
                        **dict(declared, target_origin=core.IsaPfitTargetOrigin.NativeFittedPointResponse))
    # weight_type 0: no penalty, so one point cannot determine the variables.
    free = [R.refinement_model(m.sites, [np.eye(9), np.eye(4), np.eye(4)], weight_type=0,
                               provenance="unpenalized") for m in models[:1]]
    with pytest.raises(ValueError, match="did not solve"):
        refine_streamed(free, points[:1], np.ones((1, 1)), **declared)


def _s_auxiliary():
    from psi4 import core
    shells = []
    for centre, exponent in ((0, .7), (1, 1.3), (1, .4)):
        shell = core.IsaGaussianShell()
        shell.centre, shell.l = centre, 0
        shell.exponents, shell.coefficients = [exponent], [1.]
        shells.append(shell)
    return core.IsaExplicitBasis(core.IsaBasisRole.MolecularAux, core.IsaBasisRepresentation.Cartesian,
                                 [[0., 0., 0.], [1.2, -.3, .5]], shells)


def test_fitted_point_targets_are_minus_ptcp_with_analytic_potentials():
    """Unnormalized s shells: P(k, i) = (pi/a)^1.5 erf(sqrt(a) r)/r, independent of Libint."""
    from math import erf, pi, sqrt
    from psi4.driver.procrouting.isapol_pfit_stream import fitted_point_targets
    points = _points(40)
    centres, exponents = np.array([[0., 0., 0.], [1.2, -.3, .5], [1.2, -.3, .5]]), [.7, 1.3, .4]
    analytic = np.array([[(pi/a)**1.5*erf(sqrt(a)*np.linalg.norm(p-c))/np.linalg.norm(p-c)
                          for p in points] for c, a in zip(centres, exponents)])
    rng = np.random.default_rng(3)
    responses = [rng.normal(size=(3, 3)) for _ in range(2)]
    packed = fitted_point_targets(_s_auxiliary(), points, responses)
    i, j = np.tril_indices(len(points))
    for t, response in enumerate(responses):
        dense = -analytic.T @ response @ analytic
        # Row-major packing i*(i+1)//2+j, j <= i, of the lower triangle as given.
        np.testing.assert_allclose(packed[:, t], dense[i, j], rtol=1e-12, atol=1e-14)
    assert packed.shape == (40*41//2, 2) and packed.flags.c_contiguous


def test_native_fitted_targets_streamed_fit_equals_dense_fit():
    """fitted_point_targets -> refine_streamed per node equals refine on the same targets."""
    from psi4 import core
    from psi4.driver.procrouting import isapol_refine as R
    from psi4.driver.procrouting.isapol_pfit_stream import fitted_point_targets, refine_streamed
    site = R.RefinementSite('A', 'A', (0., 0., 0.), np.eye(3), 1)
    points = _points(19)+[4., 4., 4.]
    models = []
    for omega, scale in ((0., 1.), (.6, .7)):
        models.append(R.refinement_model([site], [scale*np.diag([0., 2., 3., 4.])], frequency_au=omega,
                                         weight_type=4, weight_coefficient=1e-5,
                                         declared_variables=models[0].parameter_labels if models else None,
                                         provenance='synthetic identity'))
    responses = [-np.eye(3), -np.diag([.9, .6, 1.2])]
    packed = fitted_point_targets(_s_auxiliary(), points, responses)
    declared = dict(target_origin=core.IsaPfitTargetOrigin.NativeFittedPointResponse,
                    source_id='synthetic-state', auxiliary_basis_id='synthetic-aux',
                    response_representation='fitted_density_coefficients',
                    generation_record='synthetic-equation')
    streamed = refine_streamed(models, points, packed, **declared)
    for t, (model, fit) in enumerate(zip(models, streamed)):
        dense = R.refine(model, points, packed[:, t], **declared)
        np.testing.assert_allclose(fit.parameters, dense.parameters, rtol=1e-11, atol=1e-11)
        assert fit.result.target_provenance.auxiliary_basis_id == 'synthetic-aux'
    assert not np.allclose(streamed[0].parameters, streamed[1].parameters)


def test_fitted_point_targets_admission_and_refusals():
    from psi4.driver.procrouting.isapol_pfit_stream import fitted_point_targets
    aux, points = _s_auxiliary(), _points(5)
    responses = [np.eye(3)]
    planned = 8*(3*5+3*5+2*3*5+5*5+3*5+15*1)
    np.testing.assert_array_equal(fitted_point_targets(aux, points, responses, max_bytes=planned),
                                  fitted_point_targets(aux, points, responses))
    with pytest.raises(ValueError, match='byte resource'):
        fitted_point_targets(aux, points, responses, max_bytes=planned-1)
    for bad in ([np.eye(2)], [np.eye(3, dtype=np.float32)], [np.full((3, 3), np.nan)], [],
                [np.eye(3).tolist()]):
        with pytest.raises(ValueError, match='coefficient response'):
            fitted_point_targets(aux, points, bad)
    with pytest.raises(ValueError, match='IsaExplicitBasis'):
        fitted_point_targets(None, points, responses)
    with pytest.raises(ValueError, match='npoint x 3'):
        fitted_point_targets(aux, points[:, :2], responses)
