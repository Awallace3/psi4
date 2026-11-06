"""Real closed metadata loader against temporary synthetic wave files only."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

_spec = importlib.util.spec_from_file_location('_wave_campaign_fixture',
                                              Path(__file__).with_name('test_bank_wave.py'))
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
load, fixture = _fixture.load, _fixture.fixture


def files(monkeypatch, tmp_path, *, registered=True, audit=False):
    f, campaign = fixture(tmp_path), load('bank_wave_campaign')
    monkeypatch.setattr(campaign, 'WAVE_ROOT', tmp_path)
    bank = campaign.helper('bank_campaign')
    monkeypatch.setattr(bank, 'authority', lambda: {})
    monkeypatch.setattr(bank, 'load_pool', lambda *args: ({}, {}))
    monkeypatch.setattr(bank, 'prepared_cases', lambda *args: f.prepared)
    monkeypatch.setattr(campaign, '_load_predecessor', lambda plan: (
        f.binding['context_raw'], f.binding['checkpoint_raw'], f.binding['proof_blobs']))
    store = campaign.helper('bank_store')
    monkeypatch.setattr(store, 'INTENTS', tmp_path/'intents')
    monkeypatch.setenv('SLURM_JOB_ID', '8000')
    args = NS(producer='grid400', campaign_scope='eligible-wave', remaining_case=True,
        wave_plan=f.binding['plan_path'], wave_plan_sha256=f.binding['plan_sha256'],
        pilot=Path(f.plan['members'][f.plan['member_ids'][0]]['case_path']),
        inputs=bank.INPUTS, build=bank.BUILD_IDENTITY.parent,
        qualification=bank.QUALIFICATION, descriptors=bank.DESCRIPTORS,
        wave_registration=f.directory/'registration.json')
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        store._write_once(path, value)
    save(f.binding['plan_path'], f.binding['plan_raw'])
    for mid, case in f.cases.items():
        save(Path(f.plan['members'][mid]['case_path']), f.core.encode(case))
    if registered:
        intent = store._reserve_wave_intent(**campaign.intent_fields(f.binding, f.reg['members'],
                                                                   f.reg['registration_role']))
        f.reg['dispatch_intent_sha256'] = intent['sha256']
        reg_raw = f.core.encode(f.reg)
        f.binding.update(registration_raw=reg_raw, registration_sha256=f.core.digest(reg_raw))
        permit = f.binding['release_permit']
        permit.update(registration_sha256=f.core.digest(reg_raw), dispatch_intent_sha256=intent['sha256'])
        f.binding.update(release_permit_raw=f.core.encode(permit),
                         release_permit_sha256=f.core.digest(f.core.encode(permit)))
        save(f.directory/'registration.json', reg_raw)
        save(f.directory/'release-permit.json', f.binding['release_permit_raw'])
        save(f.directory/'registration/dispatch-intent.json', f.core.encode(intent))
        save(f.directory/'registration/registered-snapshot.json', f.core.encode(f.registered))
        for mid in f.plan['member_ids']:
            for role in ('before', 'entry', 'record'):
                save(Path(f.reg['members'][mid]['captures'][role]['path']), f.core.encode(f.captures[mid][role]))
        if audit:
            completed = _fixture.complete(f)
            for mid in f.plan['member_ids']:
                run = f.directory/'members'/mid/'science'
                save(run/'accounting.txt', completed.sci_accounting[mid])
                save(run/'payload/pilot-outcome.json', completed.outcomes[mid])
                save(run/'payload/owned-after-delivery.json', f.core.encode(completed.deliveries[mid]))
            save(f.directory/'science-barrier.json', f.binding['science_barrier_raw'])
            args.case = f.directory/'members'/f.binding['monomer_id']/'science'
            args.wave_science_barrier = f.directory/'science-barrier.json'
            args.wave_science_barrier_sha256 = f.binding['science_barrier_sha256']
    return f, campaign, args


def test_worker_reconstructs_before_registration_and_ignores_future_files(monkeypatch, tmp_path):
    f, campaign, args = files(monkeypatch, tmp_path, registered=False)
    args.case = f.directory/'members'/f.binding['monomer_id']/'science'
    args.wave_science_barrier = f.directory/'science-barrier.json'
    args.wave_science_barrier_sha256 = 'f'*64
    binding = campaign.load_binding(args, require_registration=False)
    assert binding['registration'] is None and binding['science_barrier'] is None
    assert campaign.verify_prepared_from_paths(args, args.build/'build-identity.json') == f.cases[binding['monomer_id']]
    campaign.revalidate_binding(binding)
    assert not (f.directory/'registration.json').exists()


def test_registration_permit_and_durable_shared_intent_are_revalidated(monkeypatch, tmp_path):
    f, campaign, args = files(monkeypatch, tmp_path)
    binding = campaign.load_binding(args, require_registration=True)
    assert binding['registration'] == f.reg
    assert binding['release_permit_sha256'] == f.binding['release_permit_sha256']
    campaign.revalidate_binding(binding)
    permit_path = f.directory/'release-permit.json'
    permit_path.write_bytes(permit_path.read_bytes()+b' ')
    with pytest.raises(ValueError, match='changed'):
        campaign.revalidate_binding(binding)


@pytest.mark.parametrize('change', ['permit_job', 'registration_alias', 'capture', 'intent',
                                   'explicit_sha', 'case', 'plan', 'old_args'])
def test_loader_rejects_changed_or_foreign_inputs(monkeypatch, tmp_path, change):
    f, campaign, args = files(monkeypatch, tmp_path)
    if change == 'permit_job':
        path = f.directory/'release-permit.json'
        value = f.core.decoded(path.read_bytes())
        value['members'][f.binding['monomer_id']]['science_job_id'] = '99999'
        path.write_bytes(f.core.encode(value))
    elif change == 'registration_alias':
        args.wave_registration = f.directory/'alias.json'
        args.wave_registration.symlink_to(f.directory/'registration.json')
    elif change == 'capture':
        path = Path(f.reg['members'][f.binding['monomer_id']]['captures']['entry']['path'])
        value = f.core.decoded(path.read_bytes())
        value['metadata']['name'] = 'wrong'
        path.write_bytes(f.core.encode(value))
    elif change == 'intent':
        target = campaign.helper('bank_store').INTENTS/'357'/f.plan['predecessor_snapshot_sha256']/'intent.json'
        target.write_bytes(target.read_bytes()+b' ')
    elif change == 'explicit_sha':
        args.wave_registration_sha256 = 'f'*64
    elif change == 'case':
        args.pilot.write_bytes(args.pilot.read_bytes()+b' ')
    elif change == 'plan':
        args.wave_plan.write_bytes(args.wave_plan.read_bytes()+b' ')
    else:
        args.bank_checkpoint = Path('/old')
    with pytest.raises((ValueError, KeyError)):
        campaign.load_binding(args, require_registration=True)


def test_audit_requires_real_all_sci_accounting_and_outcomes(monkeypatch, tmp_path):
    f, campaign, args = files(monkeypatch, tmp_path, audit=True)
    binding = campaign.load_binding(args, require_registration=True)
    assert binding['science_barrier_sha256'] == f.binding['science_barrier_sha256']
    campaign.revalidate_binding(binding)
    peer = f.plan['member_ids'][-1]
    (f.directory/'members'/peer/'science/accounting.txt').write_bytes(b'123|RUNNING|0:0\n')
    with pytest.raises(ValueError):
        campaign.load_binding(args, require_registration=True)


def test_missing_permit_or_barrier_never_falls_back_to_serial(monkeypatch, tmp_path):
    f, campaign, args = files(monkeypatch, tmp_path)
    args.case = f.directory/'members'/f.binding['monomer_id']/'science'
    with pytest.raises(ValueError, match='barrier'):
        campaign.load_binding(args, require_registration=True)
    del args.case
    (f.directory/'release-permit.json').unlink()
    with pytest.raises(ValueError):
        campaign.load_binding(args, require_registration=True)


def test_original_production_cutover_pins_are_closed():
    campaign = load('bank_wave_campaign')
    with pytest.raises(ValueError):
        campaign._load_predecessor(dict(bootstrap_context_path='/tmp/bank-context.json',
            bootstrap_context_sha256='a'*64, predecessor_path='/tmp/bank-checkpoint.json',
            predecessor_sha256='b'*64, generation_before=25))
