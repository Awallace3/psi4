"""Source-load task18 helper; shared FDDS comes only from installed/staged Psi4.

Synthetic controls do not certify a retained numerical capture or native integrals.
"""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HELPER = Path(__file__).parent / 'data_isapol/oracle/compare_native_response_legs.py'
spec = importlib.util.spec_from_file_location('native_response_legs_under_test', HELPER)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def basis(n, rep):
    return dict(nfunction=n, representation=rep, centres=[[0., 0., 0.]], labels=['X'], charges=[1.],
                shells=[[1, 0, i + 1, i + 1] for i in range(n)],
                exponents=list(np.arange(n) + 1.), contractions=[[1.] for _ in range(n)])


def identity():
    return dict(dimensions=[3, 2, 1, 2], representation=h.REPRESENTATION, order=h.ORDER, spin_factor=1,
                fit=dict(lambda_=1., eta=0., gamma=0., fourth=0.),
                full_C=h.numeric_identity(np.eye(3)), diagonal_energies=h.numeric_identity([-1., .5, 1.]),
                MAIN=h.descriptor_identity(basis(3, 'S')), AUX=h.descriptor_identity(basis(2, 'C')))


def direct(h1, h2, d, om2):
    # Independent unsymmetrized solve, not the shared provider or helper oracle.
    operator = np.matmul(h2, h1) - om2 * np.eye(len(h1))
    z = np.linalg.solve(operator, -4 * np.matmul(h2, d))
    return np.matmul(d.T, z)


def inputs(native=True):
    h10 = np.array([[2., .4], [-.2, 3.]])
    h2 = np.array([[1.2, -.3], [.1, 1.7]])
    k = np.array([[-.08, .03], [-.01, .02]])
    d = np.array([[.3, .2], [-.1, .4]])
    a = .25
    h1 = h10 + 4 * d @ ((1 - a) * k @ d.T)
    omega2 = np.array([0., -.04, -4., -1.e12])
    i = identity()
    return dict(h1=h1, h10=h10, h2=h2, kernel=k, exchange=a, reference_d=d,
                native_d=d + np.array([[.02, -.01], [.03, .01]]) if native else d.copy(),
                omega2=omega2, reference_cdf=np.asarray([direct(h1, h2, d, w) for w in omega2]),
                identity=i, native_identity=copy.deepcopy(i))


def test_two_distinct_semantics_and_independent_direct_controls():
    p = inputs()
    report = h.compare(**p)
    assert not report['passed']
    outputs = []
    for name in h.EXPERIMENTS:
        experiment = report['experiments'][name]
        assert len(experiment['frequencies']) == 4
        for row in experiment['frequencies']:
            control, native = [row['controls'][label] for label in ('captured_D', 'native_D')]
            assert control['passed'] and native['equivalence_passed']
            assert native['coupling_denominator']['retained_rank'] == 2
            assert native['direct_backward_residual'] < 1.e-14
            assert native['coupling_backward_residual'] < 1.e-14
            d = p['native_d']
            h1 = p['h1'] if name == h.EXPERIMENTS[0] else p['h10'] + 4 * d @ ((1 - p['exchange']) * p['kernel']) @ d.T
            expected = direct(h1, p['h2'], d, row['omega2'])
            np.testing.assert_allclose(native['raw_cdf'], expected, atol=1.e-13, rtol=1.e-12)
            np.testing.assert_allclose(native['direct_cdf'], expected, atol=1.e-13, rtol=1.e-12)
        first = experiment['frequencies'][0]['controls']['native_D']
        assert not first['forward_passed'] and first['reference_error']['max_scaled'] > 1.e-9
        assert first['reciprocity']['max_scaled'] > 1.e-4  # Never silently symmetrized.
        outputs.append(first['raw_cdf'])
    assert np.max(np.abs(outputs[0] - outputs[1])) > 1.e-4
    assert report['tolerance'] == 1.e-9 and report['pseudoinverse_rcond'] == 1.e-13


def test_captured_d_controls_pass_and_large_frequency_decays():
    p = inputs(native=False)
    report = h.compare(**p)
    assert report['passed']
    for exp in report['experiments'].values():
        rows = exp['frequencies']
        for row in rows:
            assert row['passed']
            assert row['controls']['native_D']['captured_control_error']['max_absolute'] == 0.
        assert rows[0]['xi'] == 0. and rows[-1]['xi'] == 1.e6
        assert np.max(np.abs(rows[-1]['controls']['native_D']['raw_cdf'])) < 1.e-11


def test_mapping_proxy_identity_is_recursively_owned():
    from types import MappingProxyType
    p = inputs(native=False)
    original = p['identity']
    p['identity'] = MappingProxyType(original)
    report = h.compare(**p)
    original['fit']['lambda_'] = 999.
    original['dimensions'][0] = 999
    assert report['identity']['fit']['lambda_'] == 1.
    assert report['identity']['dimensions'][0] == 3
    with pytest.raises(TypeError):
        report['identity']['fit']['lambda_'] = 2.
    with pytest.raises(TypeError):
        report['identity']['dimensions'][0] = 4


def test_inputs_and_results_are_immutable_and_owned():
    p = inputs(native=False)
    before = copy.deepcopy(p)
    report = h.compare(**p)
    for key, value in p.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, before[key])
    result = report['experiments'][h.EXPERIMENTS[0]]['frequencies'][0]['controls']['native_D']['raw_cdf']
    expected = result.copy()
    p['native_d'][:] = 999
    p['identity']['order'] = 'broken'
    np.testing.assert_array_equal(result, expected)
    with pytest.raises(TypeError):
        report['passed'] = True
    with pytest.raises(ValueError):
        result[:] = 0
    with pytest.raises(ValueError):
        result.setflags(write=True)
    assert report['identity']['order'] == h.ORDER
    json.dumps(h.plain(report), allow_nan=False)


@pytest.mark.parametrize('key,value', [
    ('h1', np.eye(3)), ('h10', np.zeros((2, 3))), ('h2', np.ones((2, 2), complex)),
    ('kernel', np.full((2, 2), np.inf)), ('reference_d', np.zeros((3, 2))),
    ('native_d', np.full((2, 2), np.nan)), ('native_d', np.zeros((2, 3))),
    ('omega2', [1.]), ('omega2', [np.nan]), ('omega2', []), ('omega2', [1j]),
    ('reference_cdf', np.zeros((1, 2, 2))), ('exchange', True), ('exchange', -1.),
    ('exchange', np.nan), ('exchange', 1j)])
def test_structural_inputs_fail_closed(key, value):
    p = inputs()
    p[key] = value
    with pytest.raises(ValueError):
        h.compare(**p)


@pytest.mark.parametrize('key,value', [
    ('representation', 'fdds_coulomb_auxiliary'), ('order', 'virtual-fast'), ('spin_factor', 2),
    ('dimensions', [3, 2, 2, 2]), ('fit', dict(lambda_=1., eta=.01, gamma=0., fourth=0.))])
def test_identity_contract_rejected_even_when_both_labels_agree(key, value):
    p = inputs()
    p['identity'][key] = value
    p['native_identity'] = copy.deepcopy(p['identity'])
    with pytest.raises(ValueError):
        h.compare(**p)


@pytest.mark.parametrize('key', ['MAIN', 'AUX', 'full_C', 'diagonal_energies', 'fit'])
def test_mismatched_native_input_identities(key):
    p = inputs()
    if key in ('MAIN', 'AUX'):
        p['native_identity'][key]['descriptor']['centres'][0][0] = .1
    elif key == 'fit':
        p['native_identity'][key]['lambda_'] = 0.
    else:
        p['native_identity'][key]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='identity mismatch'):
        h.compare(**p)


@pytest.mark.parametrize('field', ['centres', 'shells', 'contractions', 'exponents'])
def test_cannot_silently_retag_exact_basis(field):
    p = inputs()
    b = p['identity']['MAIN']['descriptor']
    if field == 'exponents':
        b[field][0] += .1
    else:
        b[field][0][0] += .1
    p['native_identity'] = copy.deepcopy(p['identity'])
    with pytest.raises(ValueError, match='descriptor/hash'):
        h.compare(**p)


def test_singular_baselines_record_failure_then_continue_all_experiments():
    p = inputs(native=False)
    p['h1'] = p['h10'] = np.zeros((2, 2))
    p['kernel'][:] = 0
    report = h.compare(**p)
    assert not report['passed']
    for exp in report['experiments'].values():
        assert len(exp['frequencies']) == 4
        for label in ('captured_D', 'native_D'):
            assert 'LinAlgError' in exp['frequencies'][0]['controls'][label]['failure']
            assert exp['frequencies'][1]['controls'][label]['finite']
    json.dumps(h.plain(report), allow_nan=False)


@pytest.mark.parametrize('epsilon', [0., 1.e-14])
def test_rank_truncation_is_not_direct_equivalence(epsilon):
    p = inputs(native=False)
    p.update(h10=np.eye(2), h2=np.eye(2), native_d=np.eye(2), reference_d=np.eye(2), exchange=0.)
    p['kernel'] = np.diag([(-1. + epsilon) / 4., 0.])
    p['h1'] = p['h10'] + 4 * p['kernel']
    p['reference_cdf'][:] = 0.  # Deliberately not accepted reference; characterize policy only.
    report = h.compare(**p)
    exp = report['experiments'][h.EXPERIMENTS[1]]
    first = exp['frequencies'][0]['controls']['native_D']
    assert first['coupling_denominator']['retained_rank'] == 1
    assert not first['equivalence_passed'] and not first['passed']
    assert 'raw_cdf' in first  # Retain inherited pinv output even if full solve is singular.
    assert exp['frequencies'][1]['controls']['native_D']['coupling_denominator']['retained_rank'] == 2
    json.dumps(h.plain(report), allow_nan=False)


def test_overflow_is_failed_diagnostic_not_hidden_fallback():
    p = inputs()
    p['h2'][:] = 1.e308
    report = h.compare(**p)
    for exp in report['experiments'].values():
        assert len(exp['frequencies']) == 4
        assert all(not row['passed'] for row in exp['frequencies'])
        assert all('failure' in row['controls']['native_D'] for row in exp['frequencies'])
    json.dumps(h.plain(report), allow_nan=False)


def test_numeric_hash_is_common_values_not_storage():
    a = np.arange(12.).reshape(3, 4)
    assert h.numeric_identity(a) == h.numeric_identity(np.asfortranarray(a))
    assert h.numeric_identity(a) == h.numeric_identity(a.astype('>f8'))
    assert h.numeric_identity(a) == h.numeric_identity(a.astype('i8'))
    assert h.numeric_identity(a) != h.numeric_identity(a.T)


def test_source_only_import_needs_no_psi4_or_reference_files(monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert not name.startswith('psi4')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    monkeypatch.setattr(Path, 'glob', lambda *a, **k: pytest.fail('reference glob at import'))
    spec = importlib.util.spec_from_file_location('standalone_native_legs', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.measure) and callable(module.compare)


def capture_fixture():
    i = identity()
    c = np.eye(92)
    energy = np.arange(92.)
    orb = dict(kind='ORBITALS', header=(1, 1, 92, 246, 5, 87, 10), c=c, energies=energy)
    state = dict(kind='DIAGONAL_ENERGIES', energies=energy,
                 bases=dict(MAIN=basis(92, 'S'), AUX=basis(246, 'C')))
    d = dict(start=2, end=436, parent=('NN', 'water'), matrix=np.zeros((435, 246)))
    fit = ((1, 0, 1, 1), (1., 0., 0., 0.), (1, 0))
    projection = dict(event=dict(fit=fit), d=d)
    snapshots = [dict(fit=fit, d=copy.deepcopy(d)) for _ in range(11)]
    replay = dict(passed=True, frequencies=[{}] * 11, raw_fits=[{}], hessian_reconstruction={})
    return [orb, state], snapshots, projection, dict(controls=(0, 0, 1, 1, 0, 1)), replay


def test_capture_contract_valid():
    _, _, i = h.capture_inputs(*capture_fixture())
    assert i['dimensions'] == [92, 246, 5, 87]


@pytest.mark.parametrize('mutation', ['energy', 'generation', 'D', 'parent', 'fit', 'retag', 'replay', 'raw', 'hessian', 'frequency', 'dimension'])
def test_capture_association_gates(mutation):
    args = capture_fixture()
    events, snapshots, projection, policy, replay = args
    if mutation == 'energy':
        events[0]['energies'] = events[0]['energies'] + .01
    elif mutation == 'generation':
        snapshots[-1]['d']['start'] += 1
    elif mutation == 'D':
        snapshots[-1]['d']['matrix'][0, 0] = 1.
    elif mutation == 'parent':
        snapshots[-1]['d']['parent'] = ('OTHER', 'water')
    elif mutation in ('fit', 'retag'):
        snapshots[-1]['fit'] = ((1, 0, 3, 1), (1., 0., 0., 0.), (1, 0))
    elif mutation == 'replay':
        replay['passed'] = False
    elif mutation == 'raw':
        replay['raw_fits'] = []
    elif mutation == 'hessian':
        replay['hessian_reconstruction'] = None
    elif mutation == 'frequency':
        replay['frequencies'] = [{}]
    elif mutation == 'dimension':
        events[0]['header'] = (1, 1, 91, 246, 5, 86, 10)
    with pytest.raises(ValueError):
        h.capture_inputs(*args)


def raw_fit_fixture():
    fit = ((1, 0, 1, 1), (1., 0., 0., 0.), (0, 0))
    subset = ((1, 0, 3, 1), (1., 0., 0., 0.), (0, 0))
    parent = ('NN original', 'file')
    raw = dict(begin=dict(fit=fit, x_identity=parent), end=dict(header=(7,)), ov=np.eye(2))
    projection = dict(event=dict(fit=subset), d=dict(parent=parent, start=9, end=10, matrix=np.eye(2)))
    events = [dict(kind='OV_ROW', parent=parent, header=(9,), ar=1)]
    return events, [raw], projection


def test_explicit_nn_to_ov_subset_association_preserves_original_metadata():
    events, fits, projection = raw_fit_fixture()
    result = h.match_raw_fit(events, fits, projection)
    assert result is fits[0]
    assert result['begin']['fit'][0][2] == 1
    assert projection['event']['fit'][0][2] == 3  # Never silently overwritten.


@pytest.mark.parametrize('mutation', ['flags', 'lambda', 'type', 'parent', 'generation', 'D', 'duplicate'])
def test_raw_fit_association_cannot_relabel_old_cache(mutation):
    events, fits, projection = raw_fit_fixture()
    if mutation == 'flags':
        projection['event']['fit'] = ((1, 0, 3, 1), (1., 0., 0., 0.), (1, 0))
    elif mutation == 'lambda':
        projection['event']['fit'] = ((1, 0, 3, 1), (0., 0., 0., 0.), (0, 0))
    elif mutation == 'type':
        fits[0]['begin']['fit'] = ((1, 0, 2, 1), (1., 0., 0., 0.), (0, 0))
    elif mutation == 'parent':
        projection['d']['parent'] = ('other', 'file')
    elif mutation == 'generation':
        projection['d']['start'] += 1
    elif mutation == 'D':
        projection['d']['matrix'][0, 0] += .01
    else:
        fits.append(copy.deepcopy(fits[0]))
    with pytest.raises(ValueError, match='matching original raw fit'):
        h.match_raw_fit(events, fits, projection)


def test_exclusive_cli_outputs_and_failed_gate_are_retained(tmp_path, monkeypatch):
    capture = tmp_path / 'capture'
    capture.mkdir()
    rp, op = tmp_path / 'report.json', tmp_path / 'operands.npz'
    measured = h.compare(**inputs())
    monkeypatch.setattr(h, 'measure', lambda directory: (measured, {'D': np.eye(2)}))
    result = h.run(capture, rp, op)
    assert not result['passed'] and not json.loads(rp.read_text())['passed']
    with np.load(op, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['D'], np.eye(2))
    with pytest.raises(ValueError, match='exists'):
        h.run(capture, rp, op)
    with pytest.raises(ValueError, match='inside capture'):
        h.run(capture, capture / 'report.json', tmp_path / 'other.npz')


def test_cli_failure_exit_and_explicit_arguments(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(h, 'run', lambda *args: {'passed': False})
    monkeypatch.setattr('sys.argv', ['compare_native_response_legs.py', str(tmp_path), '--report',
                                   str(tmp_path / 'out.json'), '--operands', str(tmp_path / 'out.npz')])
    assert h.main() == 1
    assert not json.loads(capsys.readouterr().out)['passed']
