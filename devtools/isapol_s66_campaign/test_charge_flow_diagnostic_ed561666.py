"""Synthetic frozen-response algebra only; no local quantum calculation."""
import importlib.util
import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('charge_flow_diag',
    Path(__file__).with_name('charge_flow_diagnostic_ed561666.py'))
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def operands():
    rng = np.random.default_rng(123)
    q = rng.normal(size=(50, 3))
    charge = np.array([1., 2., 3.])
    q[25] = charge-q[0]+np.array([2e-7, -3e-7, 4e-7])
    c = rng.normal(size=(4, 3))
    b = c @ q.T
    x = -4*b
    return q, charge, c, x, b, -b.T @ x


def test_decomposition_uses_actual_signed_sums_not_absolute_sum():
    q, charge, c, x, b, alpha = operands()
    result = diag.analyze_charge_flow(q, charge, c, x, b, alpha, 2)
    selector = np.zeros(50)
    selector[::25] = 1.
    retained = np.r_[0:9, 25:34]
    sums = (alpha.T @ selector)[retained]
    assert result['transpose_sum_maximum'] == pytest.approx(np.max(np.abs(sums)))
    point = result['transpose_maximum']
    assert point['fit']+point['integration'] == pytest.approx(point['signed_sum'])
    assert result['q_charge_row_error'] == pytest.approx(4e-7)
    assert result['reconstruction_error'] < 1e-12


def test_exact_quadrature_does_not_remove_finite_fit_charge():
    q, charge, c, _, _, _ = operands()
    q[25] = charge-q[0]
    b = c @ q.T
    x = -4*b
    result = diag.analyze_charge_flow(q, charge, c, x, b, -b.T @ x, 2)
    assert result['integration_response_contribution_maximum'] < 1e-12
    assert result['fit_response_contribution_maximum'] > 1.


def test_nonsymmetric_response_does_not_assume_slot_reciprocity():
    q, charge, c, _, b, _ = operands()
    x = np.random.default_rng(22).normal(size=b.shape)
    alpha = -b.T @ x
    result = diag.analyze_charge_flow(q, charge, c, x, b, alpha, 2)
    assert result['reconstruction_error'] < 1e-12
    assert result['first_slot_maximum'] != pytest.approx(result['transpose_sum_maximum'])


def test_exact_anchor_neutrality_does_not_remove_quadrature_defect():
    q, charge, c, _, _, _ = operands()
    # Synthetic exact-neutral columns, without modifying a scientific result.
    c = c-(c @ charge)[:, None]*charge[None, :]/np.dot(charge, charge)
    b = c @ q.T
    x = -4*b
    result = diag.analyze_charge_flow(q, charge, c, x, b, -b.T @ x, 2)
    assert result['fit_response_contribution_maximum'] < 1e-12
    assert result['integration_response_contribution_maximum'] > 1e-8


def test_rank_truncation_and_both_index_slots_match_lw_definition():
    alpha = np.zeros((50, 50))
    alpha[10, 0] = 2.
    alpha[1, 25] = 3e-6
    alpha[0, 26] = -4e-6
    row, column, _ = diag.charge_vectors(alpha, 2, 2)
    assert np.max(np.abs(row)) == 3e-6
    assert np.max(np.abs(column)) == 4e-6
    assert diag.charge_vectors(alpha, 2, 4)[0].max() == 2.


@pytest.mark.parametrize('which', ['q', 'c', 'x', 'b', 'alpha'])
def test_mismatched_dimensions_or_operands_are_refused(which):
    arrays = list(operands())
    index = {'q': 0, 'c': 2, 'x': 3, 'b': 4, 'alpha': 5}[which]
    arrays[index] = arrays[index].copy()
    arrays[index].flat[0] += .1
    with pytest.raises(ValueError, match='reconstruct'):
        diag.analyze_charge_flow(*arrays, 2)


def test_capture_is_exclusive_exact_and_does_not_change_inputs(tmp_path):
    capture = diag.Capture(tmp_path)
    array = np.array([[1., np.nextafter(.5, 1.)], [.5, 2.]])
    before = array.tobytes()
    capture.save('nonlocal_response', array)
    assert array.tobytes() == before and array.flags.writeable
    saved = capture.load('nonlocal_response')
    assert saved.tobytes() == before and not saved.flags.writeable
    with pytest.raises(ValueError, match='exclusive'):
        capture.save('nonlocal_response', array)
    (tmp_path/'nonlocal_response.npy').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        capture.load('nonlocal_response')


def test_capture_caps_and_nonfinite_data_fail_before_publication(tmp_path, monkeypatch):
    capture = diag.Capture(tmp_path)
    monkeypatch.setattr(diag, 'MAX_ARRAY_BYTES', 1)
    with pytest.raises(ValueError, match='before write'):
        capture.save('q', np.ones((1, 1)))
    with pytest.raises(ValueError, match='float64'):
        capture.save('q', np.array([np.nan]))
    assert list(tmp_path.iterdir()) == []


def test_real_profile_events_capture_without_replacing_numeric_functions(tmp_path):
    arrays = operands()
    q, charge, c, x, b, alpha = arrays
    def q_function(auxiliary):
        return None, q, float(np.max(np.abs(q[::25].sum(axis=0)-charge)))
    def fit_function(*, offsite_metric_damping, charge_penalty):
        charges = charge
        return NS(coefficients=c)
    class Solver:
        p = 1
        legs = np.column_stack((np.zeros((4, 1)), b))
        def solve(self, omega):
            solved = np.column_stack((np.zeros((4, 1)), x))
            return np.eye(1), alpha, 1e-15
    def lw_function(*, residual_policy, localization_rank_limit):
        raise ValueError('localize_lw: postcondition exceeds residual tolerance')
    def refinement():
        raise AssertionError('must never execute')
    observer = diag.Observer(diag.Capture(tmp_path), q_function=q_function,
        fit_function=fit_function, solve_function=Solver.solve, lw_function=lw_function,
        refinement_function=refinement)
    functions_before = (q_function.__code__, fit_function.__code__, Solver.solve.__code__,
                        lw_function.__code__)
    try:
        sys.setprofile(observer.profile)
        q_function('SYNTHETIC_AUX')
        fit_function(offsite_metric_damping=.1, charge_penalty=1000.)
        fit_function(offsite_metric_damping=0., charge_penalty=1000.)
        Solver().solve(0.)
        with pytest.raises(ValueError, match='postcondition'):
            lw_function(residual_policy='production', localization_rank_limit=2)
    finally:
        sys.setprofile(None)
    assert functions_before == (q_function.__code__, fit_function.__code__,
                                Solver.solve.__code__, lw_function.__code__)
    assert observer.validate()['input_sum_rule'] > 0.
    assert not observer.details.get('lw_passed', False)


def test_passing_lw_still_stops_before_model_fit(tmp_path):
    def q_function(auxiliary): pass
    def fit_function(): pass
    def solve_function(): pass
    def lw_function(*, residual_policy, localization_rank_limit): return 'SYNTHETIC_PASS'
    def refinement(): raise AssertionError('must not run')
    observer = diag.Observer(diag.Capture(tmp_path), q_function=q_function,
        fit_function=fit_function, solve_function=solve_function, lw_function=lw_function,
        refinement_function=refinement)
    try:
        sys.setprofile(observer.profile)
        with pytest.raises(diag.DiagnosticStop, match='before refinement'):
            lw_function(residual_policy='production', localization_rank_limit=2)
    finally:
        sys.setprofile(None)
    assert observer.details['lw_passed'] is True


def test_only_exact_approved_embers_shape_is_admitted():
    allocation = dict(Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small',
        NumCPUs='12', NumNodes='1', MinMemoryNode='64G', TimeLimit='04:00:00')
    diag.admit_allocation(allocation)
    for k, v in [('QOS', 'inferno'), ('TimeLimit', '08:00:00'),
                 ('MinMemoryNode', '144G'), ('NumCPUs', '24')]:
        with pytest.raises(ValueError, match='embers'):
            diag.admit_allocation(allocation | {k: v})


def test_executable_runtime_omits_dataset_references_and_preserves_native_request(monkeypatch):
    value = dict(entry=dict(molecule={'synthetic_geometry': True},
        attributes={'sapt_reference': 'must not execute'}, additional_keywords={}),
        specification=dict(driver='properties', method='pbe0', basis='aug-cc-pvtz',
                           keywords={'function_kwargs': {'native_monomer': 'SYNTHETIC_REQUEST'}}))
    raw = json.dumps(value).encode()
    expected = dict(molecule={'synthetic_geometry': True}, specification=dict(driver='properties',
        model=dict(method='pbe0', basis='aug-cc-pvtz'),
        keywords=value['specification']['keywords']))
    expected_raw = json.dumps(expected, allow_nan=False).encode()
    monkeypatch.setattr(diag, 'CASE_SHA256', diag.sha(raw))
    monkeypatch.setattr(diag, 'RUNTIME_SHA256', diag.sha(expected_raw))
    assert diag.bound_runtime(raw) == expected_raw
    assert b'sapt_reference' not in expected_raw
    with pytest.raises(ValueError, match='exact unchanged'):
        diag.bound_runtime(raw+b' ')


def test_help_is_standard_library_only_and_no_server_or_retry_apis_exist():
    result = subprocess.run([sys.executable, '-I', '-S', '-B', diag.__file__, '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0 and '--worker' in result.stdout
    tree = ast.parse(Path(diag.__file__).read_text())
    calls = {ast.unparse(node.func).split('.')[-1] for node in ast.walk(tree)
             if isinstance(node, ast.Call)}
    assert not calls & {'PortalClient', 'ComputeManager', 'reset_records', 'modify_records',
                        'cancel_records', 'submit', 'compute', 'add_entries', 'add_singlepoints'}
    batch = Path(diag.__file__).with_suffix('.sbatch').read_text()
    assert '#SBATCH --qos=embers' in batch and '#SBATCH --no-requeue' in batch
    assert '#SBATCH --time=04:00:00' in batch
    assert batch.index('ccasp_wait_child "$child" 20') < batch.index('CHARGE_DIAGNOSTIC_COMPLETE')


def test_anchor_caller_and_original_penalty_are_not_inferred_from_call_count(tmp_path):
    def q(auxiliary): pass
    def fit(*, offsite_metric_damping, charge_penalty):
        raise AssertionError('wrong call must be refused before executing')
    def solve(): pass
    def lw(): pass
    def refinement(): pass
    def prepare():
        name = 'target'  # Synthetic ordering error: not the anchor.
        fit(offsite_metric_damping=0., charge_penalty=1000.)
    observer = diag.Observer(diag.Capture(tmp_path), q_function=q, fit_function=fit,
        solve_function=solve, lw_function=lw, refinement_function=refinement,
        prepare_function=prepare)
    try:
        sys.setprofile(observer.profile)
        with pytest.raises(ValueError, match='anchor-then-target'):
            prepare()
    finally:
        sys.setprofile(None)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('outcome', ['lw_refusal', 'wrapped_stop', 'unrelated_failure',
                                    'runtime_changed', 'source_changed'])
def test_worker_classifies_schema_results_without_retry_and_always_clears_profiler(
        tmp_path, monkeypatch, outcome):
    payload = tmp_path/'payload'
    payload.mkdir()
    runtime = tmp_path/'runtime.json'
    raw = json.dumps(dict(molecule={'synthetic': True}, specification=dict(
        driver='properties', model=dict(method='pbe0', basis='aug-cc-pvtz'), keywords={}))).encode()
    runtime.write_bytes(raw)
    monkeypatch.setattr(diag, 'RUNTIME_SHA256', diag.sha(raw))
    core_file = tmp_path/'synthetic-core'
    core_file.write_bytes(b'not a scientific installation')
    effects = []
    def production_function(): pass
    observer = NS(functions=(production_function,), profile=lambda *args: None,
        validate=lambda: {'synthetic_analysis': True}, counts=dict(q=1, fit=2, solve=1, lw=1),
        details=dict(lw_passed=outcome == 'wrapped_stop'),
        capture=NS(records={}, total_bytes=0))
    monkeypatch.setattr(diag, 'observer_for_installation', lambda *args: (observer, {}))
    def schema(input_data, *, return_dict):
        assert sys.getprofile() == observer.profile
        effects.append('one scientific invocation')
        assert input_data['schema_name'] == 'qcschema_input' and return_dict is True
        if outcome == 'runtime_changed':
            runtime.write_bytes(b'changed')
        if outcome == 'source_changed':
            def different(): return 1
            production_function.__code__ = different.__code__
        kind = 'DiagnosticStop' if outcome == 'wrapped_stop' else 'ValueError'
        message = ('unrelated failure' if outcome == 'unrelated_failure' else
                   'localize_lw: postcondition exceeds residual tolerance')
        return dict(success=False, error=dict(error_type=kind, error_message=message),
                    input_data=dict(stdout='SYNTHETIC retained scientific log\n'))
    psi4 = NS(core=NS(__file__=str(core_file)), set_memory=lambda *args, **kw: None,
              set_num_threads=lambda *args, **kw: None)
    driver = NS(schema_wrapper=NS(run_qcschema=schema))
    monkeypatch.setitem(sys.modules, 'psi4', psi4)
    monkeypatch.setitem(sys.modules, 'psi4.driver', driver)
    args = NS(run=tmp_path, runtime=runtime, commit='SYNTHETIC_OPS', smoke=False)
    if outcome in ('lw_refusal', 'wrapped_stop'):
        diag.run_worker(args)
        report = json.loads((payload/'charge-diagnostic.json').read_bytes())
        expected = ('FIRST_NODE_LW_PASSED_NO_MODEL_FIT' if outcome == 'wrapped_stop'
                    else 'FIRST_NODE_PRODUCTION_LW_REFUSAL_CAPTURED')
        assert report['discriminator'] == expected
        assert report['stdout_available'] is True
        assert report['stdout_sha256'] == diag.file_sha(payload/'scientific-output.log')
    else:
        with pytest.raises(ValueError):
            diag.run_worker(args)
        assert not (payload/'charge-diagnostic.json').exists()
    assert sys.getprofile() is None
    assert effects == ['one scientific invocation']
    assert not (payload/'diagnostic-receipt.json').exists()  # parent alone publishes this


@pytest.mark.parametrize('failure', ['smoke', 'post_runtime', 'post_source', 'none'])
def test_parent_never_retries_or_publishes_completion_after_unexpected_failure(
        tmp_path, monkeypatch, failure):
    monkeypatch.setattr(diag, 'RUNS', tmp_path)
    run = tmp_path/'ccasp-native-s66-ed561666fc-charge-diagnostic001'
    (run/'logs').mkdir(parents=True)
    build = tmp_path/'build'
    build.mkdir()
    manifest = build/'build-identity.json'
    manifest.write_text(json.dumps(dict(commit=diag.SCIENTIFIC_COMMIT)))
    case = tmp_path/'case.json'
    case.write_bytes(b'{"synthetic_case":true}')
    monkeypatch.setattr(diag, 'BUILD', build)
    monkeypatch.setattr(diag, 'BUILD_SHA256', diag.file_sha(manifest))
    monkeypatch.setattr(diag, 'CASE', case)
    monkeypatch.setattr(diag, 'CASE_SHA256', diag.file_sha(case))
    monkeypatch.setattr(diag, 'RUNTIME_SHA256', diag.sha(b'{}'))
    monkeypatch.setattr(diag, 'bound_runtime', lambda raw: b'{}')
    monkeypatch.setattr(diag, 'qualified_receipts', lambda: {'synthetic': True})
    monkeypatch.setattr(diag, 'MANAGER', Path(sys.executable).resolve().parents[1])
    scratch = tmp_path/'scratch'
    scratch.mkdir()
    monkeypatch.setattr(diag, 'SCRATCH', scratch)
    monkeypatch.setenv('SLURM_JOB_ID', 'SYNTHETIC_JOB')
    sources = []
    def source(*args):
        sources.append(True)
        return {'source': 'changed' if failure == 'post_source' and len(sources) == 2 else 'same'}
    monkeypatch.setattr(diag, 'source_identities', source)
    verified = []
    def verify(*args):
        verified.append(True)
        if failure == 'post_runtime' and len(verified) == 2:
            raise ValueError('synthetic post-run runtime change')
    allocation = dict(Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small',
        NumCPUs='12', NumNodes='1', MinMemoryNode='64G', TimeLimit='04:00:00')
    checks = NS(allocation=lambda: allocation, verify_runtime=verify,
        installed_python_command=lambda *args: ['synthetic-child'])
    common = NS(child_environment=lambda *args: {})
    pilot = NS(scratch_snapshot=lambda *args, **kw: {'synthetic': True})
    monkeypatch.setattr(diag, 'helper', lambda name: {
        'phoenix_installed_checks': checks, 'phoenix_descriptor_bank_cb63305': common,
        'phoenix_pilot_cb63305': pilot, 'descriptor_scratch_cb63305': None}[name])
    calls = []
    def child(command, **kw):
        calls.append(command)
        if '--smoke' in command:
            if failure == 'smoke':
                raise subprocess.CalledProcessError(1, command)
            return
        diag.write_json(run/'payload/charge-diagnostic.json', dict(
            status='COMPLETE_DIAGNOSTIC_NOT_A_MODEL', operational_commit='SYNTHETIC_OPS',
            scientific_commit=diag.SCIENTIFIC_COMMIT, runtime_input_sha256=diag.RUNTIME_SHA256,
            production_tolerance=1e-6, production_cap_unchanged=32,
            discriminator='SYNTHETIC_NO_QUANTUM_EXECUTION'))
    monkeypatch.setattr(diag.subprocess, 'run', child)
    if failure == 'none':
        diag.run_parent(NS(run=run, commit='SYNTHETIC_OPS'))
        assert (run/'payload/diagnostic-receipt.json').exists()
    else:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            diag.run_parent(NS(run=run, commit='SYNTHETIC_OPS'))
        assert not (run/'payload/diagnostic-receipt.json').exists()
    assert len(calls) == (1 if failure == 'smoke' else 2)
    assert sum('--smoke' not in call for call in calls) <= 1
