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


@pytest.mark.parametrize("kind", ["gap", "duplicate", "early", "extra", "oversized",
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
            if kind == "duplicate" and start == 2:
                index = 0
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
