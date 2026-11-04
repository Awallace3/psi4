"""Fresh singleton construction with mocked preceding admission, not a model."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).with_name('pilot_cb63305.py')
spec = importlib.util.spec_from_file_location('_fresh_grid400_pilot_test', path)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
GIB = 1024**3


def fixture(monkeypatch):
    prepare = pilot.helper('prepare_descriptor_bank_cb63305')
    molecule = dict(symbols=['C', 'H']*6, geometry=[0.]*36,
                    fix_com=True, fix_orientation=True, fix_symmetry='c1')
    request = dict(schema='native-monomer-request/3', point_group='c1',
        resources=dict(max_bytes=48*GIB, max_work=75*10**12, max_io_bytes=16*GIB, capacity='large'),
        max_replay_work=10**12, pfit_maximum_work_bytes=2*GIB,
        implementation_sha256=dict(core='a'*64, isapol_python='b'*64))
    runtime = dict(molecule=copy.deepcopy(molecule), specification=dict(driver='properties',
        model=dict(method='pbe0', basis='aug-cc-pvtz'), keywords=dict(reference='rhf',
            function_kwargs=dict(native_monomer=request, properties=['dipole']))))
    sha = prepare.json_digest(runtime)
    entry = dict(name='synthetic-selected-monomer', molecule=molecule, additional_keywords={},
        attributes=dict(monomer_id='monomer-0008', sapt_reference='metadata only'))
    inputs = dict(inventory=json.dumps(dict(monomers=[
        dict(id='monomer-0008', formula='SYNTHETIC_ONLY')])).encode(),
        entries=json.dumps(dict(entries=[entry])).encode())
    row = dict(monomer_id='monomer-0008', entry_name=entry['name'], runtime_input=runtime,
               runtime_input_sha256=sha, intended_memory_bytes=60*GIB)
    bank = dict(monomers=[row], producer=dict(status='SYNTHETIC_UNIT_RECEIPTS_ONLY'),
                input_sha256={}, resource_declarations_sha256='f'*64)
    report = dict(monomer_id='monomer-0008', runtime_input_sha256=sha,
                  descriptor_report_sha256='d'*64, dimensions=dict(sites=12))
    cohort = pilot.helper('grid400_cohort')
    cohort.verify_admitted_bank = lambda *args: (bank, dict(reports=[report]))
    original = pilot.helper
    monkeypatch.setattr(pilot, 'helper', lambda name: cohort if name == 'grid400_cohort' else original(name))
    pins = pilot.producer_pins('grid400') | dict(runtime_input_sha256=sha, selected_report_sha256='d'*64)
    old_pins = pilot.producer_pins
    monkeypatch.setattr(pilot, 'producer_pins', lambda producer='cb63305':
                        copy.deepcopy(pins) if producer == 'grid400' else old_pins(producer))
    return inputs, row, report, entry


def construct(inputs):
    return pilot.prepare_pilot(inputs, {}, b'unit resources', b'unit bank',
                               b'unit allocation', {}, producer='grid400')


def test_fresh_grid400_draft_preserves_attributes_and_has_no_append_target(monkeypatch):
    inputs, row, report, entry = fixture(monkeypatch)
    draft = construct(inputs)
    assert draft['producer_profile'] == 'grid400' and draft['dataset_mode'] == 'fresh'
    assert draft['monomer_id'] == 'monomer-0008'
    assert draft['formula'] == 'SYNTHETIC_ONLY'  # Not a scientific fixture.
    assert draft['execution_profile']['name'] == 'grid400-s60'
    assert draft['execution_profile']['time_limit'] == '04:00:00'
    assert (draft['queue_policy']['cpus'], draft['queue_policy']['psi4_memory_GiB'],
            draft['queue_policy']['job_memory_GiB']) == (16, 60, 96)
    assert draft['queue_policy']['maximum_records'] == draft['queue_policy']['concurrency'] == 1
    assert draft['queue_policy']['find_existing'] is draft['queue_policy']['existing_ok'] is False
    assert draft['entry']['attributes'] == entry['attributes']
    assert draft['runtime_input_sha256'] == row['runtime_input_sha256']
    assert draft['descriptor_report_sha256'] == report['descriptor_report_sha256']
    assert 'dataset_id' not in draft and 'owned_checkpoint' not in draft
    assert 'sapt_reference' not in json.dumps(pilot.helper('phoenix_worker_probe').runtime_input(draft))


@pytest.mark.parametrize('change', ['geometry', 'runtime', 'report', 'point_group',
                                   'monomer', 'budget', 'memory'])
def test_changed_geometry_receipts_or_finite_contract_cannot_make_a_draft(monkeypatch, change):
    inputs, row, report, _ = fixture(monkeypatch)
    if change == 'geometry':
        entries = json.loads(inputs['entries'])
        entries['entries'][0]['molecule']['symbols'][0] = 'H'
        inputs['entries'] = json.dumps(entries).encode()
    elif change == 'runtime':
        row['runtime_input_sha256'] = 'c'*64
    elif change == 'report':
        report['descriptor_report_sha256'] = 'e'*64
    elif change == 'monomer':
        row['monomer_id'] = 'monomer-0009'
    elif change == 'memory':
        row['intended_memory_bytes'] = 80*GIB
    else:
        request = row['runtime_input']['specification']['keywords']['function_kwargs']['native_monomer']
        if change == 'point_group':
            request['point_group'] = 'd2h'
        else:
            request['resources']['max_bytes'] = 64*GIB
    with pytest.raises(ValueError):
        construct(inputs)
