"""Negative acceptance tests on synthetic states, not molecular trajectory parity."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
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


@pytest.mark.parametrize('tolerance,defect', [(1e-10, 5e-10), (1e-6, 5e-7), (.01, .005)])
def test_legacy_custom_tolerance_without_profile_certification(tolerance, defect):
    actual, reference, initial = fixture()
    actual['final']['tails'][0]['exponent'] += defect
    report = tool.compare(actual, reference, initial, tolerance)
    expected = all(e['max_scaled'] <= tolerance for e in report['errors'].values())
    assert report['passed'] == expected
    assert report['scaled_tolerance'] == tolerance
    assert report['comparison_mode'] == 'legacy-custom-tolerance'
    assert report['profile_certified'] is False
    for name in ('strict_passed', 'provisional_passed', 'selected_profile', 'selected_profile_passed'):
        assert name not in report
    # Legacy calls do not redefine or affect subsequent fixed certificates.
    fixed = tool.compare(actual, reference, initial, profile='provisional-1e-3')
    assert fixed['checks']['atom1_tail']['strict_threshold'] == 1e-9
    assert fixed['checks']['atom1_tail']['provisional_threshold'] == 1e-3
    actual['iterations'] += 1
    assert not tool.compare(actual, reference, initial, tolerance)['passed']


@pytest.mark.parametrize('tolerance', [0., -1., float('nan'), float('inf')])
def test_invalid_legacy_tolerance(tolerance):
    with pytest.raises(ValueError, match='Invalid tolerance'):
        tool.compare(*fixture(), tolerance)


def test_provisional_joint_tail_only_and_mixed_sites():
    actual, reference, initial = fixture()
    actual['final']['tails'][0]['exponent'] += 2 * 2.3588804665973028e-8
    strict = tool.compare(actual, reference, initial)
    assert not strict['passed'] and not strict['selected_profile_passed']
    report = tool.compare(actual, reference, initial, profile='provisional-1e-3')
    assert not report['passed'] and report['provisional_passed'] and report['selected_profile_passed']
    assert report['checks']['atom1_tail']['provisional_threshold'] == 1e-3
    assert report['parameter_diagnostics']['atom1_exponent']['strict_threshold'] == 1e-9
    # A different site's strict density defect cannot hide behind tail eligibility.
    actual['final']['atomic_coefficients'][1][0] += 1e-8
    assert not tool.compare(actual, reference, initial, profile='provisional-1e-3')['provisional_passed']


def test_provisional_retains_joint_reference_denominator():
    actual, reference, initial = fixture()
    actual['final']['tails'][0]['amplitude'] = 1000.
    reference['post']['atoms'][0]['tail'][2] = 1000.
    actual['final']['tails'][0]['exponent'] += .5
    report = tool.compare(actual, reference, initial, profile='provisional-1e-3')
    assert report['checks']['atom1_tail']['actual'] == .0005
    assert report['provisional_passed'] and not report['passed']
    assert report['parameter_diagnostics']['atom1_exponent']['max_scaled'] == .25
    assert not report['parameter_diagnostics']['atom1_exponent']['strict_passed']


@pytest.mark.parametrize('defect', ['control', 'cutoff', 'charge', 'history_control', 'structural', 'config', 'endpoint'])
def test_provisional_does_not_relax_strict_controller_requirements(defect):
    actual, reference, initial = fixture()
    if defect == 'control': actual['final']['active_w_eps'] = 1e-8
    elif defect == 'history_control': actual['history'][0]['active_positive_lambda'] = 1e-8
    elif defect == 'cutoff': actual['final']['tails'][0]['cutoff'] += 1e-8
    elif defect == 'charge': actual['final']['saved_shape_charges'][0] += 1e-8
    elif defect == 'structural': actual['history'][0]['atom_converged'][0] = False
    elif defect == 'config': initial['floats'][0] += 1e-12
    elif defect == 'endpoint': initial['post']['atoms'][0]['tail'][0] += 1e-12
    report = tool.compare(actual, reference, initial, profile='provisional-1e-3')
    assert not report['passed'] and not report['provisional_passed']


@pytest.mark.parametrize('defect', ['unknown_profile', 'missing_check', 'shape', 'nan_config', 'bad_flag', 'custom_tolerance'])
def test_provisional_malformed_trajectory_fails_closed(defect):
    actual, reference, initial = fixture()
    kwargs = {'profile': 'provisional-1e-3'}
    if defect == 'unknown_profile': kwargs['profile'] = 'loose'
    elif defect == 'missing_check': del actual['final']['saved_shape_charges']
    elif defect == 'shape': actual['final']['atomic_coefficients'][0].append(1.)
    elif defect == 'nan_config': initial['floats'][0] = float('nan')
    elif defect == 'bad_flag': actual['history'][0]['converged'] = 1
    elif defect == 'custom_tolerance': kwargs['tolerance'] = .01
    with pytest.raises((ValueError, KeyError)):
        tool.compare(actual, reference, initial, **kwargs)


@pytest.mark.parametrize('profile', ['strict', 'provisional-1e-3'])
def test_trajectory_cli_selected_exit_and_exclusive_provenance(profile, tmp_path, monkeypatch):
    actual, reference, initial = fixture()
    actual['final']['tails'][0]['exponent'] += 1e-7
    initial_dir, final_dir = tmp_path/'initial', tmp_path/'final'
    initial_dir.mkdir(); final_dir.mkdir()
    for directory in (initial_dir, final_dir):
        (directory/'isapol-sweep-state.dat').write_text('synthetic reader test double')
    monkeypatch.setattr(tool.sweep, 'read_state', lambda path: initial if path.parent == initial_dir else reference)
    trajectory, report_path = tmp_path/'trajectory.json', tmp_path/'report.json'
    trajectory.write_text(json.dumps(actual))
    argv = ['compare_isa_trajectory.py', str(trajectory), '--initial', str(initial_dir), '--reference', str(final_dir), '--report', str(report_path)]
    if profile != 'strict': argv.extend(['--profile', profile])
    monkeypatch.setattr(sys, 'argv', argv)
    if profile == 'strict':
        with pytest.raises(SystemExit): tool.main()
    else:
        tool.main()
    saved = json.loads(report_path.read_text())
    assert saved['selected_profile_passed'] is (profile != 'strict')
    assert not saved['passed'] and saved['provisional_passed']
    assert saved['provenance']['before'] == saved['provenance']['after']
    assert str(trajectory) in saved['source_sha256']
    assert 'no controller execution' in saved['execution_kind']
    original = report_path.read_bytes()
    with pytest.raises(FileExistsError): tool.main()
    assert report_path.read_bytes() == original
