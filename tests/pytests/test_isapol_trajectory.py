"""Negative acceptance tests on synthetic states, not molecular trajectory parity."""
import copy
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('trajectory_compare', Path(__file__).parent/'data_isapol/oracle/compare_isa_trajectory.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


def fixture():
    atoms = [dict(vectors={'D': [1.], 'W0': [1.]}, charges=[1., 1.],
                  flags=[1, 1, 1], tail=[1.5, 2.5, 1., 2.]) for _ in range(3)]
    reference = dict(iteration=1, floats=[.17], ints=[1],
                     deltas=[0., 0., 0.], max_delta=0.,
                     pre={'atoms': copy.deepcopy(atoms)},
                     post={'atoms': atoms, 'active': [0., 0., 0., 1.]})
    final = dict(atomic_coefficients=[[1.], [1.], [1.]], shape_coefficients=[[1.], [1.], [1.]],
                 saved_shape_charges=[1., 1., 1.], max_delta=0., active_w_eps=0.,
                 active_positive_lambda=0., apply_tails=False,
                 tails=[dict(defined=True, amplitude=1., exponent=2., cutoff=1.5) for _ in range(3)])
    last = dict(iteration=1, converged=True, atom_converged=[True]*3, deltas=[0.]*3,
                max_delta=0., active_w_eps=0., active_positive_lambda=0.,
                apply_tails=False, shape_charges=[1.]*3)
    return dict(iterations=1, converged=True, final=final, history=[last]), reference, copy.deepcopy(reference)


def test_exact_synthetic_trajectory():
    report = tool.compare(*fixture())
    assert report['passed'] and report['iteration_count_matches'] and report['flags_match']
    assert all(e['max_absolute'] == 0 for e in report['errors'].values())


@pytest.mark.parametrize('defect', ['iterations', 'atom_flag', 'delta', 'max_delta', 'cutoff',
                                    'tail_defined', 'history', 'cutoff_invariance', 'config'])
def test_trajectory_rejects_omitted_acceptance_checks(defect):
    actual, reference, initial = fixture()
    if defect == 'iterations':
        actual['iterations'] = 2
        actual['history'].append(copy.deepcopy(actual['history'][0]))
        actual['history'][-1]['iteration'] = 2
    elif defect == 'atom_flag': actual['history'][-1]['atom_converged'][0] = False
    elif defect == 'delta': actual['history'][-1]['deltas'][0] = .1
    elif defect == 'max_delta': actual['final']['max_delta'] = .1
    elif defect == 'cutoff': actual['final']['tails'][0]['cutoff'] = 1.6
    elif defect == 'tail_defined': actual['final']['tails'][0]['defined'] = False
    elif defect == 'history': actual['history'][-1]['iteration'] = 3
    elif defect == 'cutoff_invariance': initial['post']['atoms'][0]['tail'][0] = 1.6
    elif defect == 'config': initial['floats'][0] = .2
    assert not tool.compare(actual, reference, initial)['passed']


def test_scalar_parameter_diagnostics_do_not_replace_legacy_joint_gate():
    actual, reference, initial = fixture()
    actual['final']['tails'][0]['amplitude'] = 1000.
    reference['post']['atoms'][0]['tail'][2] = 1000.
    actual['final']['tails'][0]['exponent'] += 1e-8
    report = tool.compare(actual, reference, initial)
    assert report['passed']  # Existing joint denominator is deliberately retained.
    assert report['parameter_diagnostics']['atom1_exponent']['max_scaled'] > 1e-9


def test_nonfinite_trajectory_rejected():
    actual, reference, initial = fixture()
    actual['final']['atomic_coefficients'][0][0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        tool.compare(actual, reference, initial)
