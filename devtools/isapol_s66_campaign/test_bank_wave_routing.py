"""Explicit wave branches, shell grammar and session guards; no job execution."""
import copy
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS

import pytest


def load(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_wave_routing_test_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = '/storage/project/r-cs207-0/awallace43/runs'
D = ROOT+'/ccasp-native-s66-grid400-1aaa9e0a30-waves001/wave-0025-0033'
MID = 'monomer-0026'
K = D+'/cases/'+MID+'.json'
F = D+'/members/'+MID+'/worker/payload/worker-preflight.json'


def wrapper(role, *, registration_sha=True):
    name = dict(worker='phoenix_worker_cb63305', science='phoenix_pilot_cb63305',
                audit='runtime_audit_cb63305')[role]
    source = Path(__file__).with_name(name+'.sbatch').read_text()
    split = ('ROOT=/storage/project/r-cs207-0/awallace43/runs' if role == 'worker'
             else 'MANAGER=/storage/project/')
    prefix = source.split(split, 1)[0]
    argv = ['/unit/source', 'a'*40, K]
    if role != 'worker':
        argv.append(F)
    if role == 'audit':
        argv.append(D+'/members/'+MID+'/science')
    argv += [D+'/members/'+MID+'/'+role, '--grid400-bank-wave',
             D+'/plan.json', 'b'*64, '-' if role == 'worker' else D+'/registration.json']
    if role == 'audit':
        if registration_sha:
            argv += ['--wave-registration-sha256', 'c'*64]
        argv += ['--wave-science-barrier', D+'/science-barrier.json',
                 '--wave-science-barrier-sha256', 'd'*64]
    return prefix, argv


def invoke(prefix, argv):
    return subprocess.run(['bash', '--noprofile', '--norc', '-c',
        prefix+'\nprintf "%s\\n" "${EXTRA[@]}" "$PRODUCER"', 'unit', *argv],
        capture_output=True, text=True)


@pytest.mark.parametrize('role', ['worker', 'science', 'audit'])
def test_exact_controller_wrapper_argv_forwarded_without_advancement(role):
    prefix, argv = wrapper(role)
    result = invoke(prefix, argv)
    assert result.returncode == 0, result.stderr
    flags = result.stdout.splitlines()
    assert flags[:5] == ['--producer', 'grid400', '--campaign-scope',
                         'eligible-wave', '--remaining-case']
    assert flags[flags.index('--wave-plan')+1] == D+'/plan.json'
    assert '--advance-checkpoint' not in flags and '--advance-bank-checkpoint' not in flags
    assert ('--wave-registration' in flags) is (role != 'worker')
    assert ('--wave-science-barrier' in flags) is (role == 'audit')
    assert flags[-1] == 'grid400-1aaa9e0a30-bank'


def test_audit_optional_registration_hash_does_not_make_barrier_optional():
    prefix, argv = wrapper('audit', registration_sha=False)
    assert invoke(prefix, argv).returncode == 0
    for bad in (argv[:-2], argv[:-4], argv+['--wave-registration-sha256', 'c'*64]*2):
        assert invoke(prefix, bad).returncode != 0


@pytest.mark.parametrize('role', ['worker', 'science', 'audit'])
@pytest.mark.parametrize('change', ['plan_hash', 'case', 'run', 'root', 'extra', 'mixed',
                                  'registration', 'alias'])
def test_wrapper_rejects_ambiguous_foreign_or_mixed_wave_arguments(role, change):
    prefix, argv = wrapper(role)
    if change == 'plan_hash':
        argv[argv.index('b'*64)] = 'b'*63
    elif change == 'case':
        argv[2] = K.replace(MID, 'monomer-0027')
    elif change == 'run':
        index = {'worker': 3, 'science': 4, 'audit': 5}[role]
        argv[index] = argv[index]+'/nested'
    elif change == 'root':
        argv = [arg.replace('waves001', 'waves002') for arg in argv]
    elif change == 'extra':
        argv += ['--wave-plan', D+'/plan.json']
    elif change == 'mixed':
        argv += ['--advance-bank-checkpoint']
    elif change == 'registration':
        index = argv.index('--grid400-bank-wave')+3
        argv[index] = '/unit/foreign' if role != 'worker' else D+'/registration.json'
    else:
        index = argv.index('--grid400-bank-wave')+1
        argv[index] = D+'/./plan.json'
    assert invoke(prefix, argv).returncode != 0


@pytest.mark.parametrize('profile_name', ['s20', 's40', 's60', 's80', 's120'])
def test_wave_reuses_exact_worker_science_profiles_without_widening_old_default(profile_name):
    worker, science = load('phoenix_worker_cb63305'), load('phoenix_pilot_cb63305')
    profile = worker.helper('grid400_bank_policy').execution_profile(profile_name)
    allocation = dict(Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small',
        NumCPUs=str(profile['cores']), NumNodes='1',
        MinMemoryNode=str(profile['job_memory_GiB'])+'G', TimeLimit='00:45:00')
    worker.admit_allocation(allocation, producer='grid400', campaign_scope='eligible-wave',
                            execution_profile=profile)
    allocation['TimeLimit'] = profile['time_limit']
    science.admit_allocation(allocation, producer='grid400', campaign_scope='eligible-wave',
                             execution_profile=profile)
    with pytest.raises(ValueError):
        worker.source_names('grid400', remaining_case=True)
    with pytest.raises(ValueError):
        science.admit_allocation(allocation | {'QOS': 'inferno'}, producer='grid400',
            campaign_scope='eligible-wave', execution_profile=profile)


def preflight():
    fixture = load('test_bank_science')
    science, record, draft, kwargs = fixture.preflight(True)
    for doc in (record, draft):
        doc.update(campaign_scope='eligible-wave', dispatch_protocol='registered-wave/1',
            wave_id='wave-0025-0033', bootstrap_context_sha256='a'*64,
            checkpoint_sha256='b'*64, predecessor_snapshot_sha256='c'*64)
    record['status'] = 'PASS_SELECTED_WAVE_CASE_WORKER_PREFLIGHT_ONLY'
    kwargs['campaign_scope'] = 'eligible-wave'
    return science, record, draft, kwargs


def test_wave_worker_proof_cannot_be_consumed_by_serial_preflight():
    science, record, draft, kwargs = preflight()
    science.validate_preflight(record, draft, **kwargs)
    with pytest.raises(ValueError):
        science.validate_preflight(record, draft, **(kwargs | {'campaign_scope': 'eligible-bank'}))


@pytest.mark.parametrize('field,value', [
    ('dispatch_protocol', 'serial'), ('wave_id', 'wave-0033-0041'),
    ('checkpoint_sha256', 'f'*64), ('predecessor_snapshot_sha256', 'f'*64),
    ('runtime_input_sha256', 'f'*64), ('operational_sources_sha256', {}),
    ('descriptor_verification', {'status': 'REFUSED_DESCRIPTOR_PLAN'}),
    ('task_config', {'ncores': 24, 'memory_GiB': 120, 'internal_retries': 1}),
])
def test_wave_preflight_keeps_all_original_runtime_source_descriptor_gates(field, value):
    science, record, draft, kwargs = preflight()
    record[field] = value
    with pytest.raises(ValueError):
        science.validate_preflight(record, draft, **kwargs)


@pytest.mark.parametrize('flag', ['advance_checkpoint', 'advance_bank_checkpoint'])
def test_wave_audit_rejects_advancement_before_allocation_or_scientific_reads(monkeypatch, flag):
    audit = load('runtime_audit_cb63305')
    arguments = NS(producer='grid400', campaign_scope='eligible-wave', remaining_case=True,
        wave_plan=Path('/unit/plan.json'), wave_plan_sha256='a'*64,
        wave_registration=Path('/unit/registration.json'), case=Path('/unit/science'),
        wave_science_barrier=Path('/unit/science-barrier.json'),
        wave_science_barrier_sha256='b'*64, **{flag: True})
    monkeypatch.setattr(audit.argparse.ArgumentParser, 'parse_args', lambda self: arguments)
    original = audit.helper
    monkeypatch.setattr(audit, 'helper', lambda name: original(name)
        if name in ('phoenix_worker_cb63305', 'bank_wave_runtime') else pytest.fail(name))
    with pytest.raises(ValueError, match='advance singleton'):
        audit.main()


def test_wave_session_missing_registration_stops_before_manager_import(tmp_path):
    session = load('single_record_cb63305')
    fixture = load('test_grid400_bank_execution')
    _, case, kwargs = fixture.binding()
    case.update(campaign_scope='eligible-wave', dataset_mode='append')
    with pytest.raises(ValueError, match='registered wave binding'):
        session.run_session(NS(), **kwargs, producer='grid400', case=case,
            campaign_scope='eligible-wave', payload=tmp_path, save=lambda *a: None)


@pytest.mark.parametrize('scope', ['default', 'eligible-bank'])
def test_wave_flags_cannot_be_silently_ignored_in_legacy_scope(scope):
    worker = load('phoenix_worker_cb63305')
    with pytest.raises(ValueError, match='explicit eligible-wave'):
        worker.selected_scope(NS(producer='grid400', campaign_scope=scope,
                                 wave_plan=Path('/unit/plan.json')))


def test_nested_wave_paths_are_exact_and_role_bound(monkeypatch):
    runtime = load('bank_wave_runtime')
    root = Path(D)
    binding = dict(plan_path=root/'plan.json', monomer_id=MID,
        plan=dict(source=str(Path(__file__).resolve().parents[2]), operational_commit='a'*40))
    monkeypatch.setattr(runtime, 'helper', lambda name: NS(
        load_binding=lambda args, **kw: binding))
    arguments = NS(run=root/'members'/MID/'audit', worker_preflight=Path(F),
                   case=root/'members'/MID/'science', commit='a'*40)
    runtime.validate_paths(arguments, role='audit')
    for key, value in [('run', arguments.run/'nested'),
                       ('worker_preflight', Path(F.replace(MID, 'monomer-0027'))),
                       ('case', root/'members'/MID/'worker')]:
        changed = copy.copy(arguments)
        setattr(changed, key, value)
        with pytest.raises(ValueError):
            runtime.validate_paths(changed, role='audit')
