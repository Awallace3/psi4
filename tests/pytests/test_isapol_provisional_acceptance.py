"""Pure comparison-contract tests; no native measurements or production imports."""
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ORACLE = Path(__file__).parent / 'data_isapol/oracle'


def load(name):
    spec = importlib.util.spec_from_file_location(name, ORACLE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p = load('provisional_acceptance')
pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


def inputs(stage='native-drho'):
    metrics, _, checks = p.contract(stage)
    return {k: 0. for k in metrics}, {k: True for k in checks}


@pytest.mark.parametrize('stage', ['native-drho', 'native-ov'])
def test_default_strict_and_exact_boundary(stage):
    metrics, checks = inputs(stage)
    assert p.evaluate(stage, metrics, checks)['selected_profile'] == 'strict'
    for value, expected in [(1e-3, True), (math.nextafter(1e-3, math.inf), False)]:
        metrics['coefficients'] = value
        report = p.evaluate(stage, metrics, checks)
        assert not report['passed'] and not report['strict_passed'] and not report['selected_profile_passed']
        assert report['provisional_passed'] is expected
        selected = p.evaluate(stage, metrics, checks, profile='provisional-1e-3')
        assert selected['selected_profile_passed'] is expected
        detail = selected['checks']['coefficients']
        assert detail['actual'] == value and detail['strict_threshold'] == 1e-9 and detail['provisional_threshold'] == 1e-3
        assert selected['original_target'] == 1e-9 and 'TODO' in selected['tightening_todo']


def test_higher_precision_boundary_is_not_rounded_down():
    metrics, checks = inputs()
    value = np.nextafter(np.longdouble(1e-3), np.longdouble(np.inf))
    metrics['coefficients'] = value
    if value == float(value):  # Platforms whose longdouble is binary64.
        assert not p.evaluate('native-drho', metrics, checks, profile='provisional-1e-3')['provisional_passed']
    else:
        with pytest.raises(ValueError, match='exactly representable'):
            p.evaluate('native-drho', metrics, checks, profile='provisional-1e-3')


def test_actual_historical_drho_coefficient_cannot_be_hidden_by_density():
    metrics, checks = inputs()
    metrics['coefficients'] = 0.0012802188514176112
    report = p.evaluate('native-drho', metrics, checks, profile='provisional-1e-3')
    assert not any(report[k] for k in ('passed', 'strict_passed', 'provisional_passed', 'selected_profile_passed'))
    assert report['checks']['sampled_density']['provisional_passed']


def test_drho_only_exception_and_boundary():
    metrics, checks = inputs()
    for value, expected in [(0.0012802188514176112, True), (1e-2, True),
                            (math.nextafter(1e-2, math.inf), False)]:
        metrics['coefficients'] = value
        report = p.evaluate('native-drho', metrics, checks, profile=p.DRHO_PROFILE)
        assert report['selected_profile_passed'] is expected
        assert report['provisional_passed'] is expected
        assert not report['passed'] and not report['strict_passed']
        assert report['checks']['coefficients']['provisional_threshold'] == 1e-2
        assert not p.evaluate('native-drho', metrics, checks, profile='provisional-1e-3')['provisional_passed']
    metrics, checks = inputs()
    required, relaxed, structural = p.contract('native-drho')
    for key in required-relaxed:
        bad = dict(metrics, **{key: 1e-8})
        report = p.evaluate('native-drho', bad, checks, profile=p.DRHO_PROFILE)
        assert not report['selected_profile_passed']
        assert report['checks'][key]['provisional_threshold'] == 1e-9
    for key in structural:
        assert not p.evaluate('native-drho', metrics, dict(checks, **{key: False}),
                              profile=p.DRHO_PROFILE)['selected_profile_passed']
    assert not p.evaluate('native-drho', metrics, checks, profile=p.DRHO_PROFILE,
                          missing_implementations=['localization'])['selected_profile_passed']


@pytest.mark.parametrize('stage', ['native-ov', 'trajectory'])
def test_drho_exception_cannot_leak_to_other_stages(stage):
    with pytest.raises(ValueError, match='Drho-C-only'):
        p.evaluate(stage, {}, {}, profile=p.DRHO_PROFILE)


def test_historical_ov_candidate():
    metrics, checks = inputs('native-ov')
    metrics.update(coefficients=2.3019798321950356e-5, sampled_transition_density=5.0303437185125854e-8,
                   pointwise_scaled_density_error=2.2143328081905677e-6,
                   relative_abs_weighted_density_l2=6.665561522827979e-8)
    report = p.evaluate('native-ov', metrics, checks, profile='provisional-1e-3')
    assert not report['passed'] and report['provisional_passed'] and report['selected_profile_passed']


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf, -1e-30, True, [0.], '0'])
def test_invalid_metric_fails_closed(value):
    metrics, checks = inputs()
    metrics['coefficients'] = value
    with pytest.raises(ValueError):
        p.evaluate('native-drho', metrics, checks, profile='provisional-1e-3')


@pytest.mark.parametrize('defect', ['profile', 'stage', 'missing_metric', 'unknown_metric', 'missing_check', 'unknown_check', 'numeric_boolean'])
def test_invalid_contract_fails_closed(defect):
    metrics, checks = inputs()
    stage, profile = 'native-drho', 'provisional-1e-3'
    if defect == 'profile': profile = 'provisional-1e-2'
    if defect == 'stage': stage = 'native-response'
    if defect == 'missing_metric': del metrics['coefficients']
    if defect == 'unknown_metric': metrics['coefficient_rms'] = 0.
    if defect == 'missing_check': del checks['checkpoint_identity']
    if defect == 'unknown_check': checks['arbitrary_pass'] = True
    if defect == 'numeric_boolean': checks['checkpoint_identity'] = 1
    with pytest.raises(ValueError):
        p.evaluate(stage, metrics, checks, profile=profile)


@pytest.mark.parametrize('stage', ['native-drho', 'native-ov'])
def test_every_nonallowlisted_metric_and_structure_stays_strict(stage):
    required, relaxed, structural = p.contract(stage)
    for key in required - relaxed:
        metrics, checks = inputs(stage)
        metrics[key] = 1e-8
        report = p.evaluate(stage, metrics, checks, profile='provisional-1e-3')
        assert not report['selected_profile_passed']
        assert report['checks'][key]['provisional_threshold'] == 1e-9
    for key in structural:
        metrics, checks = inputs(stage)
        checks[key] = False
        assert not p.evaluate(stage, metrics, checks, profile='provisional-1e-3')['provisional_passed']


@pytest.mark.parametrize('missing', [('production C++ OV API',), ('published localization',), ('native response operator producers',)])
def test_numerical_pass_cannot_promote_missing_implementation(missing):
    metrics, checks = inputs('native-ov')
    report = p.evaluate('native-ov', metrics, checks, profile='provisional-1e-3', missing_implementations=missing)
    assert report['provisional_numerical_passed'] and report['strict_numerical_passed']
    assert not report['passed'] and not report['provisional_passed'] and not report['selected_profile_passed']
    assert report['status'] == 'missing-implementation'


def test_exclusive_outputs_preflight_all_and_preserve_existing(tmp_path):
    first, second = tmp_path/'first', tmp_path/'second'
    second.write_bytes(b'historical')
    with pytest.raises(FileExistsError):
        with p.exclusive_outputs([first, second]):
            pytest.fail('Must fail before opening first')
    assert not first.exists() and second.read_bytes() == b'historical'
    with pytest.raises(ValueError):
        p.preflight([first, first])
    with p.exclusive_outputs([first]) as streams:
        streams[0].write(p.json_bytes({'passed': False, 'provisional_passed': True}))
    assert json.loads(first.read_text())['passed'] is False
    with pytest.raises(FileExistsError):
        with p.exclusive_outputs([first]):
            pass


def test_exclusive_outputs_concurrent_creator_not_overwritten(tmp_path, monkeypatch):
    first, second = tmp_path/'first', tmp_path/'second'
    original_open = Path.open
    def racing_open(path, *args, **kwargs):
        if path == second and args == ('xb',):
            with original_open(path, 'wb') as stream:
                stream.write(b'concurrent')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', racing_open)
    with pytest.raises(FileExistsError):
        with p.exclusive_outputs([first, second]):
            pass
    assert not first.exists() and second.read_bytes() == b'concurrent'


@pytest.mark.parametrize('stage', ['drho', 'ov'])
def test_native_runner_preflights_before_measurement(stage, tmp_path, monkeypatch):
    runner = load('compare_native_' + stage)
    monkeypatch.setattr(runner, 'measure', lambda *a, **k: pytest.fail('No measurement permitted'))
    report, coefficients = tmp_path/'report.json', tmp_path/'coefficients'
    coefficients.write_bytes(b'historical')
    extra = {'reference_replay': tmp_path/'replay'} if stage == 'ov' else {}
    with pytest.raises(FileExistsError):
        runner.run(tmp_path/'inputs', tmp_path/'checkpoint', report, coefficients, **extra)
    assert not report.exists() and coefficients.read_bytes() == b'historical'


@pytest.mark.parametrize('stage', ['drho', 'ov'])
def test_mocked_native_wrapper_status_provenance_and_exclusive_report(stage, tmp_path, monkeypatch):
    # Explicit test doubles: this tests the wrapper, NOT native parity.
    runner = load('compare_native_' + stage)
    directory = tmp_path/'inputs'
    directory.mkdir()
    (directory/'record').write_text('input')
    checkpoint, extension = tmp_path/'checkpoint', tmp_path/'core.so'
    checkpoint.write_text('checkpoint'); extension.write_bytes(b'test double')
    monkeypatch.setattr(runner.profiles, 'production_psi4', lambda: SimpleNamespace(core=SimpleNamespace(__file__=str(extension))))
    monkeypatch.setattr(runner.profiles, 'load', lambda name: SimpleNamespace(replay=lambda d: {'passed': True}))
    metrics, _ = inputs('native-' + stage)
    metrics['coefficients'] = 2e-5
    error_keys = ['metric', 'coefficients', 'sampled_density', 'rhs', 'fitted_electrons'] if stage == 'drho' else ['metric', 'coefficients', 'sampled_transition_density']
    raw = {k: v for k, v in metrics.items() if k not in error_keys}
    raw.update(passed=False, errors={k: {'max_absolute': metrics[k], 'max_scaled': metrics[k]} for k in error_keys})
    coeff = {'coefficients': [1.], 'charges': [1.]} if stage == 'drho' else np.array([[1.]])
    monkeypatch.setattr(runner, '_measure', lambda *args: (raw.copy(), coeff))
    extra = {}
    if stage == 'ov':
        replay = tmp_path/'reference-replay.json'
        replay.write_text('{"passed":true}')
        extra['reference_replay'] = replay
    report_path, coefficients_path = tmp_path/'fresh.json', tmp_path/'fresh-coefficients'
    report = runner.run(directory, checkpoint, report_path, coefficients_path, profile='provisional-1e-3', **extra)
    saved = json.loads(report_path.read_text())
    assert not saved['passed'] and saved['provisional_passed'] and saved['selected_profile_passed']
    assert saved['provenance']['before'] == saved['provenance']['after']
    assert str(extension) in saved['provenance']['before'] and str(checkpoint) in saved['provenance']['before']
    assert saved['provenance']['original_measurement_sha256'] == runner.ORIGINAL_SHA256
    assert saved['outputs']['coefficients_sha256'] == p.hashes([coefficients_path])[str(coefficients_path)]
    assert saved['broader_pipeline']['status'] == 'missing-implementation' and not saved['broader_pipeline']['passed']
    assert 'deferred' in saved['broader_pipeline']['response_transition_leg_diagnostic']
    assert report['selected_profile'] == 'provisional-1e-3'
    with pytest.raises(FileExistsError):
        runner.run(directory, checkpoint, report_path, coefficients_path, **extra)


@pytest.mark.parametrize('stage', ['drho', 'ov'])
@pytest.mark.parametrize('profile', ['strict', 'provisional-1e-3'])
def test_native_cli_exit_uses_selected_not_legacy_status(stage, profile, monkeypatch):
    runner = load('compare_native_' + stage)
    def fake_run(*args, **kwargs):
        assert kwargs['profile'] == profile
        return dict(passed=False, strict_passed=False, provisional_passed=True,
                    selected_profile=profile, selected_profile_passed=profile != 'strict')
    monkeypatch.setattr(runner, 'run', fake_run)
    argv = ['native', 'inputs', '--checkpoint', 'checkpoint', '--report', 'fresh-report', '--coefficients', 'fresh-coefficients']
    if stage == 'ov': argv.extend(['--reference-replay', 'retained-reference-control'])
    if profile != 'strict': argv.extend(['--profile', profile])
    monkeypatch.setattr(p.sys, 'argv', argv)
    if profile == 'strict':
        with pytest.raises(SystemExit): runner.main()
    else:
        runner.main()


def test_production_import_filters_source_and_empty_path(monkeypatch, tmp_path):
    extension = tmp_path / ('core' + p.importlib.machinery.EXTENSION_SUFFIXES[0])
    extension.write_bytes(b'test')
    old = ['', str(p.ROOT), str(p.ROOT/'psi4'), str(tmp_path)]
    monkeypatch.setattr(p.sys, 'path', old[:])
    def importer(name):
        assert p.sys.path == [str(tmp_path)]
        return SimpleNamespace(core=SimpleNamespace(__file__=str(extension)))
    monkeypatch.setattr(p.importlib, 'import_module', importer)
    assert p.production_psi4().core.__file__ == str(extension)
    assert p.sys.path == old
