"""Mock scientific-state boundaries, exercise the real request/oeprop interface."""
from types import SimpleNamespace as NS

import numpy as np
import pytest


def test_request3_workspace_reaches_bounded_driver_through_all_public_layers(monkeypatch, tmp_path):
    import psi4
    from psi4 import core
    from psi4.driver.procrouting import isapol_schema as schema
    from psi4.driver.procrouting import isapol_molecular as molecular
    from psi4.driver.procrouting import isapol_bounded_oeprop as oeprop
    core.clean_options()
    molecule = psi4.geometry(
        '0 1\nO 0 0 0\nH 1.4 0 1\nH -1.4 0 1\n'
        'symmetry c1\nno_com\nno_reorient\nunits bohr')
    declaration = molecular.MolecularDeclaration(
        molecular.PROFILE, (np.eye(3),)*3, ((0, 1), (0, 2)))
    cap = 1024**3+137
    raw = dict(schema='native-monomer-request/3', point_group='c1',
        declaration=declaration.as_dict(), pfit_maximum_work_bytes=cap,
        resources=dict(max_bytes=4*1024**3, max_work=100*10**12,
                       max_io_bytes=64*1024**3, capacity='large'),
        max_replay_work=10**12, implementation_sha256=dict(core='0'*64, isapol_python='1'*64))
    request = schema.NativeMonomerRequest.from_dict(raw)
    class Wfn:
        def molecule(self): return molecule
        def basisset(self):
            return NS(name=lambda: 'aug-cc-pVTZ', has_puream=lambda: True)
        def scalar_variables(self): return {}
        def array_variables(self): return {}
    class Grid:
        def __init__(self, *args): pass
        def x(self): return [0.]
        y = z = x
        def w(self): return [1.]
    class ReachedBounded(Exception):
        pass
    captured = {}
    def backend(wfn, recipe, **kwargs):
        captured.update(kwargs)
        raise ReachedBounded
    monkeypatch.setattr(core, 'Wavefunction', Wfn)
    monkeypatch.setattr(core, 'IsaGrid', Grid)
    monkeypatch.setattr(core, 'get_memory', lambda: 16*1024**3)
    monkeypatch.setattr(core, 'IOManager',
        NS(shared_object=lambda: NS(get_default_path=lambda: str(tmp_path))))
    monkeypatch.setattr(molecular, 'require_scf_seal', lambda *args: None)
    monkeypatch.setattr(molecular, 'validate_correction', lambda *args, **kwargs: None)
    monkeypatch.setattr(molecular, 'native_monomer_recipe',
                        lambda *args, **kwargs: NS(auxiliary=object()))
    monkeypatch.setattr(schema, 'validate_native_request',
                        lambda *args, **kwargs: (request, {'sha256': dict(request.implementation_sha256)}))
    monkeypatch.setattr(oeprop, 'bounded_properties', backend)
    # Do not bypass run_molecular_properties, util.oeprop or bounded_oeprop.run:
    # only the live-state producers and final scientific dispatch are mocked.
    with pytest.raises(ReachedBounded):
        schema.execute_native_request(Wfn(), request)
    assert captured['pfit_maximum_work_bytes'] == cap
    assert captured['max_replay_work'] == 10**12
    assert captured['max_parameters'] == 108
    assert len(captured['declared_variables']) == 108
    assert captured['quadrature'].frequencies[0] == 0.
    assert len(captured['quadrature'].frequencies) == 11
    assert captured['lattice_options'].npoints == 2000


@pytest.mark.parametrize('solver', ['NormalEquationsDSYSV', 'StreamingQR'])
@pytest.mark.parametrize('predictions', [False, True])
def test_planner_native_base_boundary_includes_all_source_and_solver_buffers(solver, predictions):
    from psi4 import core
    from psi4.driver.procrouting.isapol_pfit_plan import native_row_plan
    n, k, t, block, chunk = 3, 2, 2, 4096, 17
    options = core.IsaPfitOptions()
    options.solver = getattr(core.IsaPfitSolver, solver)
    options.retain_pair_predictions = predictions
    options.qr_chunk_rows = chunk
    planned = native_row_plan(n, k, t, block_rows=block, solver=solver,
                              qr_chunk_rows=chunk, retain_pair_predictions=predictions)
    model = core.IsaPfitRowModel()
    model.channel_labels, model.parameter_labels = ['synthetic'], ['a', 'b']
    model.parameter_units, model.fixed, model.fixed_values = ['au', 'au'], [False]*2, [0.]*2
    model.provenance = 'synthetic native planner boundary; no molecular data'
    cloud = core.IsaPfitCloudRows()
    cloud.label, cloud.points, cloud.full_row_count = 'tiny complete', n, 6
    cloud.maximum_block_rows = block
    matrix = core.IsaPfitMatrix()
    matrix.rows, matrix.cols, matrix.values = 2, 2, [1., 0., 0., 1.]
    penalty = core.IsaPfitMatrixPenalty()
    penalty.matrix, penalty.anchor = matrix, [0., 0.]
    provenance = core.IsaPfitTargetProvenance()
    provenance.origin = core.IsaPfitTargetOrigin.SyntheticAnalyticTest
    provenance.convention = (
        core.IsaPfitTargetConvention.NegativeInducedPotentialPerUnitSourceChargeAtomicUnits)
    provenance.source_id, provenance.generation_record = 'synthetic', 'workspace boundary test'
    problems = []
    for i in range(t):
        problem = core.IsaPfitRowProblem()
        problem.model, problem.cloud, problem.penalty = model, cloud, penalty
        problem.target_provenance, problem.frequency_au = provenance, float(i)
        problems.append(problem)
    design = np.arange(12, dtype=float).reshape(6, 2)/7
    targets = np.ones((6, 2))
    calls = []
    def blocks():
        calls.append(1)
        yield 0, design, targets
    options.maximum_work_bytes = planned['native_base_bytes']-1
    with pytest.raises(Exception, match='kernel numerical buffers exceed maximum_work_bytes'):
        core.isa_pfit_solve_rows_multi(problems, blocks, options)
    assert not calls
    options.maximum_work_bytes += 1
    # At the exact base, only a queried-workspace refusal is permitted.
    try:
        core.isa_pfit_solve_rows_multi(problems, blocks, options)
    except Exception as error:
        assert 'workspace' in str(error) and 'kernel numerical buffers exceed' not in str(error)
    calls.clear()
    options.maximum_work_bytes = 1024**2
    results = core.isa_pfit_solve_rows_multi(problems, blocks, options)
    assert calls == [1, 1]
    assert all(r.diagnostics.work_budget_bytes >= planned['native_base_bytes'] for r in results)
    assert all(r.settings.maximum_work_bytes == 1024**2 for r in results)
