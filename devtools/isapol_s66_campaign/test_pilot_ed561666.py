"""Fresh-pilot construction seams only; no real descriptor/model acceptance."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).with_name('pilot_cb63305.py')
spec = importlib.util.spec_from_file_location('fresh_cap32_pilot_test', path)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def fixture(monkeypatch):
    cases = pilot.helper('case_cb63305')
    molecule = dict(symbols=['C', 'H']*6, geometry=[0.]*36,
                    fix_com=True, fix_orientation=True, fix_symmetry='c1')
    request = dict(schema='native-monomer-request/3', point_group='c1',
                   resources=dict(max_bytes=32*1024**3),
                   implementation_sha256=dict(core='a'*64, isapol_python='b'*64))
    runtime = dict(molecule=copy.deepcopy(molecule), specification=dict(driver='properties',
        model=dict(method='pbe0', basis='aug-cc-pvtz'), keywords=dict(reference='rhf',
            function_kwargs=dict(native_monomer=request, properties=['dipole']))))
    sha = cases.json_digest(runtime)
    entry = dict(name='synthetic-benzene', molecule=molecule, additional_keywords={},
        attributes=dict(monomer_id='monomer-0047', sapt_reference='metadata only',
                        user_context='preserve unchanged'))
    inputs = {
        'inventory': json.dumps(dict(monomers=[dict(id='monomer-0047', formula='C6H6',
            representative=dict(dimer_id='24_Benzene-Benzene_pi-pi_0.90', role='A',
                                R=.9, fragment_indices=list(range(12))))])).encode(),
        'entries': json.dumps(dict(entries=[entry])).encode()}
    row = dict(monomer_id='monomer-0047', entry_name=entry['name'],
               runtime_input=runtime, runtime_input_sha256=sha, intended_memory_bytes=40*1024**3)
    bank = dict(monomers=[row], producer=dict(status='SYNTHETIC_UNIT_RECEIPTS_ONLY'),
                input_sha256={}, resource_declarations_sha256='f'*64)
    report = dict(monomer_id='monomer-0047', runtime_input_sha256=sha,
                  descriptor_report_sha256='d'*64, dimensions=dict(sites=12))
    allocation = dict(reports=[report])
    # Mock the preceding receipt/descriptor gate, not any scientific calculation.
    cases.verify_admitted_bank = lambda *a, **k: (bank, allocation)
    original = pilot.helper
    monkeypatch.setattr(pilot, 'helper', lambda name: cases if name == 'case_cb63305' else original(name))
    pins = pilot.producer_pins('ed561666') | dict(runtime_input_sha256=sha, selected_report_sha256='d'*64)
    original_pins = pilot.producer_pins
    monkeypatch.setattr(pilot, 'producer_pins',
                        lambda producer='cb63305': copy.deepcopy(pins) if producer == 'ed561666'
                        else original_pins(producer))
    return inputs, row, report, entry


def construct(inputs):
    return pilot.prepare_pilot(inputs, {}, b'unit resources', b'unit bank',
                               b'unit allocation', {}, producer='ed561666')


def test_fresh_s40_pilot_preserves_attributes_and_does_not_become_owned354(monkeypatch):
    inputs, row, report, original_entry = fixture(monkeypatch)
    draft = construct(inputs)
    assert draft['schema'] == 'native-selected-molecular-pilot/1'
    assert draft['producer_profile'] == 'ed561666' and draft['dataset_mode'] == 'fresh'
    assert draft['dataset_name'] == 'S66x8-native-CCASP-ed561666fc-request3-v1'
    assert draft['execution_profile']['name'] == 's40'
    assert (draft['execution_profile']['cores'], draft['execution_profile']['science_memory_GiB']) == (12, 40)
    assert draft['entry']['attributes'] == original_entry['attributes']
    assert draft['runtime_input_sha256'] == row['runtime_input_sha256']
    assert draft['descriptor_report_sha256'] == report['descriptor_report_sha256']
    assert draft['queue_policy']['find_existing'] is False
    assert draft['queue_policy']['existing_ok'] is False and draft['queue_policy']['maximum_records'] == 1
    assert 'dataset_id' not in draft and 'required_first_pilot_audit_sha256' not in draft
    assert 'sapt_reference' not in json.dumps(pilot.helper('phoenix_worker_probe').runtime_input(draft))


@pytest.mark.parametrize('target', ['geometry', 'runtime_hash', 'descriptor', 'point_group'])
def test_reconstructed_fresh_pilot_refuses_changed_inputs(monkeypatch, target):
    inputs, row, report, _ = fixture(monkeypatch)
    if target == 'geometry':
        entries = json.loads(inputs['entries'])
        entries['entries'][0]['molecule']['symbols'][0] = 'H'
        inputs['entries'] = json.dumps(entries).encode()
    elif target == 'runtime_hash':
        row['runtime_input_sha256'] = 'c'*64
    elif target == 'descriptor':
        report['descriptor_report_sha256'] = 'e'*64
    else:
        row['runtime_input']['specification']['keywords']['function_kwargs']['native_monomer']['point_group'] = 'd2h'
    with pytest.raises(ValueError):
        construct(inputs)


def test_closed_profile_protects_prior_datasets_records_and_rejects_unknown_selector():
    pins = pilot.producer_pins('ed561666')
    assert pins['protected_datasets'] == (353, 354)
    assert pins['protected_records'] == (5234314, 5234315, 5234316, 5234317)
    assert pins['descriptor_job'] == '14992173'
    for value in (None, True, 'latest', '../caller_module'):
        with pytest.raises(ValueError, match='closed'):
            pilot.producer_pins(value)
    cases = pilot.helper('case_cb63305')
    with pytest.raises(ValueError, match='exact actual'):
        cases.verify_admitted_bank({}, {}, b'', b'not pinned', b'', {}, producer='ed561666')
