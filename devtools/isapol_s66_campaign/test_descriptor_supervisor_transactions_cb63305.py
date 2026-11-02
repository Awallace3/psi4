"""Mocked supervisor transactions; never native execution or real admission."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest


def load():
    spec = importlib.util.spec_from_file_location(
        'mock_descriptor_supervisor', Path(__file__).with_name('phoenix_descriptor_bank_cb63305.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transaction(monkeypatch, tmp_path, case):
    module = load()
    build, run = tmp_path/'build', tmp_path/'ccasp-native-s66-transaction'
    build.mkdir()
    run.mkdir()
    (run/'logs').mkdir()
    scratch = tmp_path/'scratch'
    scratch.mkdir()
    qroot = tmp_path/'qualification'
    qroot.mkdir()
    package = str(build/'env/lib/python3.13/site-packages/psi4')
    identity = dict(package=package+'/__init__.py', core=package+'/core.so',
        version='SYNTHETIC_UNIT_ONLY', implementation=dict(sha256=dict(core='0'*64, isapol_python='1'*64)))
    original_helper = module.helper
    planner = original_helper('resource_plan_cb63305')
    manifest = dict(commit=planner.SCIENTIFIC_COMMIT, installed_package=package)
    request = dict(schema='native-monomer-request/3',
        resources=dict(max_bytes=12*1024**3, max_work=6*10**12, max_io_bytes=16*1024**3, capacity='large'))
    runtime = dict(molecule=dict(symbols=['O', 'H', 'H']),
        specification=dict(keywords=dict(function_kwargs=dict(native_monomer=request))))
    raw = json.dumps(runtime, sort_keys=True, allow_nan=False).encode()
    rows = [dict(monomer_id=f'monomer-{i:04d}', intended_memory_bytes=20*1024**3,
        runtime_input=copy.deepcopy(runtime), runtime_input_sha256=module.digest(raw)) for i in range(1, 132)]
    bank = dict(monomers=rows, qualification='SYNTHETIC_UNIT_ONLY_NOT_ACCEPTANCE')
    paths = dict(build_identity=build/'build-identity.json',
        qualification=qroot/'installed-validation.json',
        imported_identity=qroot/'imported-identity.json', junit=qroot/'junit.xml')
    values = dict(build_identity=manifest, imported_identity=identity,
                  qualification={'status': 'SYNTHETIC_UNIT_ONLY'}, junit={'not': 'real JUnit'})
    for name in ('inventory', 'entries', 'declarations', 'input_validation', 'resources', 'bank'):
        paths[name] = tmp_path/(name+'.json')
        values[name] = bank if name == 'bank' else {'synthetic_role': name}
    for name, path in paths.items():
        path.write_text(json.dumps(values[name]))
    ops_commit = 'c'*40
    allocation = dict(JobId='123', Account='gts-cs207-chemx', QOS='embers',
        Partition='cpu-small', NumNodes='1', NumCPUs='16', MinMemoryNode='96G', TimeLimit='00:45:00')
    checks = original_helper('phoenix_installed_checks')
    children, runtime_checks, statuses = [], [], []
    def git(source, *args):
        if args == ('rev-parse', 'HEAD'):
            return ops_commit
        if args == ('status', '--porcelain'):
            statuses.append(1)
            return ' M synthetic-source-drift' if case == 'source_drift' and len(statuses) > 1 else ''
        if args[0] == 'diff':
            return 'devtools/isapol_s66_campaign/synthetic-operation.py'
        raise AssertionError('unexpected Git command: '+str(args))
    def verify_runtime(*args):
        runtime_checks.append(len(children))
        if case == 'runtime_drift' and len(runtime_checks) > 1:
            raise RuntimeError('synthetic runtime drift')
    checks.git, checks.allocation, checks.verify_runtime = git, lambda: allocation, verify_runtime
    profile = original_helper('qualified_cb63305')
    profile.verify_receipts = lambda _: dict(status='SYNTHETIC_UNIT_RECEIPTS_ONLY')
    prepare = original_helper('prepare_descriptor_bank_cb63305')
    prepare.prepare_bank = lambda *args: copy.deepcopy(bank)
    admission = original_helper('descriptor_admission_cb63305')
    # Measurement/binding is independently tested. This deliberately returns
    # the supplied report ID so coverage must independently reject duplicates.
    admission.verify_descriptor = lambda report, row, identity, **kw: dict(
        monomer_id=report['monomer_id'], status=report['status'])
    helpers = dict(phoenix_installed_checks=checks, qualified_cb63305=profile,
        prepare_descriptor_bank_cb63305=prepare, descriptor_admission_cb63305=admission,
        resource_plan_cb63305=planner)
    monkeypatch.setattr(module, 'helper', lambda name: helpers.get(name) or original_helper(name))
    monkeypatch.setattr(module, 'RUNS', tmp_path)
    monkeypatch.setattr(module, 'SCRATCH', scratch)
    monkeypatch.setattr(module, 'quota_snapshot', lambda *args: {'status': 'SYNTHETIC_SNAPSHOT_ONLY'})
    monkeypatch.setattr(module, 'policy_binding', lambda *args: planner.FILES)
    monkeypatch.setattr(module, 'operation_hashes', lambda: {name: 'a'*64 for name in module.OP_FILES})
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    def child(command, **kwargs):
        assert '-I' in command and '-B' in command and '-X' in command
        assert kwargs['stdin'] == subprocess.DEVNULL
        monomer = command[command.index('--monomer-id')+1]
        children.append(monomer)
        if case == 'child_failure' and len(children) == 5:
            raise subprocess.CalledProcessError(1, command)
        if case == 'interrupted' and len(children) == 5:
            raise KeyboardInterrupt
        if case == 'missing_report' and len(children) == 5:
            return NS(returncode=0)
        if case == 'input_drift' and len(children) == 1:
            paths['inventory'].write_bytes(paths['inventory'].read_bytes()+b' ')
        report = dict(monomer_id=('monomer-0001' if case == 'duplicate_report'
                                  and len(children) == 131 else monomer),
                      status=('REFUSED_DESCRIPTOR_PLAN' if case == 'refused'
                              and len(children) == 5 else 'PASS_DESCRIPTOR_PLAN_ONLY'))
        Path(command[command.index('--output')+1]).write_text(json.dumps(report))
        return NS(returncode=0)
    monkeypatch.setattr(module.subprocess, 'run', child)
    args = ['supervisor', '--build', str(build), '--qualification', str(paths['qualification']),
        '--bank', str(paths['bank']), '--resources', str(paths['resources']),
        '--inventory', str(paths['inventory']), '--entries', str(paths['entries']),
        '--declarations', str(paths['declarations']), '--input-validation', str(paths['input_validation']),
        '--run', str(run), '--commit', ops_commit,
        '--expected-bank-sha256', hashlib.sha256(paths['bank'].read_bytes()).hexdigest()]
    monkeypatch.setattr(sys, 'argv', args)
    return module, run, children, runtime_checks


def test_all131_mock_children_and_before_after_runtime_checks_required(monkeypatch, tmp_path):
    module, run, children, checks = transaction(monkeypatch, tmp_path, 'success')
    module.main()
    report = json.loads((run/'payload/descriptor-allocation.json').read_text())
    assert report['status'] == 'PASS_ALL_131_DESCRIPTOR_PLANS_ONLY'
    assert len(report['reports']) == len(set(children)) == 131
    assert checks == [0, 131]
    assert report['producer']['status'] == 'SYNTHETIC_UNIT_RECEIPTS_ONLY'


def test_refused_measurement_is_not_dropped_or_relabelled_as_pass(monkeypatch, tmp_path):
    module, run, children, _ = transaction(monkeypatch, tmp_path, 'refused')
    module.main()
    report = json.loads((run/'payload/descriptor-allocation.json').read_text())
    assert report['status'] == 'REFUSED_DESCRIPTOR_PLANS' and len(children) == 131
    assert sum(r['status'] == 'REFUSED_DESCRIPTOR_PLAN' for r in report['reports']) == 1


@pytest.mark.parametrize('case', ['missing_report', 'duplicate_report', 'child_failure',
                                  'input_drift', 'source_drift', 'runtime_drift', 'interrupted'])
def test_failure_drift_or_interruption_cannot_publish_success(monkeypatch, tmp_path, case):
    module, run, children, _ = transaction(monkeypatch, tmp_path, case)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError,
                        subprocess.CalledProcessError, KeyboardInterrupt)):
        module.main()
    assert not (run/'payload/descriptor-allocation.json').exists()
    report = json.loads((run/'payload/descriptor-allocation-failure.json').read_text())
    assert report['status'] == 'FAILED_DESCRIPTOR_ALLOCATION'
    if case in ('child_failure', 'missing_report', 'interrupted'):
        assert len(children) == 5
