"""Selected raw bookkeeping with synthetic receipt fixtures, not admission."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('_selected_preparation_fixtures',
    Path(__file__).with_name('test_prepare_descriptor_bank_cb63305.py'))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
prepare = fixtures.prepare
GIB = 1024**3


def selected_inputs(monkeypatch):
    docs, contracts = fixtures.fixture()
    raw = fixtures.bind_inputs(monkeypatch, docs)
    original = prepare.helper

    def helper(name):
        if name == 'qualified_grid400':
            return SimpleNamespace(verify_receipts=lambda _: dict(
                status='SYNTHETIC_UNIT_RECEIPTS_ONLY',
                implementation_sha256={'core': '1'*64, 'isapol_python': '2'*64}))
        return original(name)

    monkeypatch.setattr(prepare, 'helper', helper)
    contract = copy.deepcopy(contracts['monomer-0008'])
    contract['resources'] = dict(max_bytes=48*GIB, max_work=75*10**12,
                                max_io_bytes=16*GIB, capacity='large')
    contract['intended_memory_bytes'] = 60*GIB
    contract['pfit_maximum_work_bytes'] = 2*GIB
    contract['max_replay_work'] = 10**12
    resources = dict(schema='native-s66x8-selected-resource-declarations/1',
        status='DECLARED_NOT_ADMITTED', inventory_sha256=prepare.INPUT_SHA256['inventory'],
        policy_id='1aaa9e0a30-reference-h2h1-grid400-ledger-v4',
        selected_monomers=['monomer-0008'], monomers={'monomer-0008': contract})
    return raw, resources


def compute(raw, resources):
    return prepare.prepare_grid400_pilot_bank(raw, {}, fixtures.encode(resources))


def test_exact_one_selected_raw_runtime_not_all131_or_a_model(monkeypatch):
    raw, resources = selected_inputs(monkeypatch)
    before = copy.deepcopy((raw, resources))
    bank = compute(raw, resources)
    assert bank['schema'] == 'native-s66x8-selected-descriptor-preparation/1'
    assert bank['status'] == 'PREPARED_SELECTED_MONOMER_FOR_DESCRIPTORS_ONLY'
    assert bank['selected_monomers'] == ['monomer-0008']
    assert bank['monomer_count'] == len(bank['monomers']) == 1
    assert bank['fragment_contexts'] == 8
    assert (bank['inventory_monomers'], bank['inventory_fragment_contexts']) == (131, 1056)
    row = bank['monomers'][0]
    assert row['monomer_id'] == 'monomer-0008' and row['intended_memory_bytes'] == 60*GIB
    request = row['runtime_input']['specification']['keywords']['function_kwargs']['native_monomer']
    assert request['resources'] == resources['monomers']['monomer-0008']['resources']
    assert request['implementation_sha256'] == {'core': '1'*64, 'isapol_python': '2'*64}
    assert request['schema'] == 'native-monomer-request/3'
    assert request['max_replay_work'] == 10**12
    assert request['pfit_maximum_work_bytes'] == 2*GIB
    assert 'sapt_reference' not in json.dumps(row['runtime_input'])
    assert 'Resource declarations are not admission' in bank['qualification']
    assert (raw, resources) == before


@pytest.mark.parametrize('change', ['missing', 'extra', 'wrong_selected', 'duplicate_selected',
                                  'old_schema', 'old_policy', 'implicit_all131'])
def test_selection_cannot_be_widened_or_reinterpreted(monkeypatch, change):
    raw, resources = selected_inputs(monkeypatch)
    if change == 'missing':
        resources['monomers'] = {}
    elif change == 'extra':
        resources['monomers']['monomer-0009'] = copy.deepcopy(resources['monomers']['monomer-0008'])
    elif change == 'wrong_selected':
        resources['selected_monomers'] = ['monomer-0009']
    elif change == 'duplicate_selected':
        resources['selected_monomers'] *= 2
    elif change == 'old_schema':
        resources['schema'] = 'native-s66x8-resource-declarations/2'
    elif change == 'old_policy':
        resources['policy_id'] = 'ed561666-reference-h2h1-ledger-v3'
    else:
        del resources['selected_monomers']
    with pytest.raises(ValueError, match='selected monomer0008 resource declarations'):
        compute(raw, resources)


@pytest.mark.parametrize('field,value', [
    ('max_bytes', 32*GIB), ('max_bytes', 64*GIB),
    ('max_work', 100*10**12), ('max_io_bytes', 32*GIB), ('capacity', 'extended'),
    ('intended_memory_bytes', 80*GIB)])
def test_approved_budget_is_exact_and_never_inferred_or_promoted(monkeypatch, field, value):
    raw, resources = selected_inputs(monkeypatch)
    contract = resources['monomers']['monomer-0008']
    if field == 'intended_memory_bytes':
        contract[field] = value
    else:
        contract['resources'][field] = value
    before = copy.deepcopy(resources)
    # A 64GiB numeric budget also exceeds the unchanged60GiB science limit,
    # so the generic memory-cover guard correctly refuses before the exact
    # selected-profile guard. Both are refusal, never resource promotion.
    with pytest.raises(ValueError, match='exact approved fresh|intended Psi4 memory must cover'):
        compute(raw, resources)
    assert resources == before


@pytest.mark.parametrize('field,value', [
    ('max_replay_work', 2*10**12), ('max_replay_work', 10**12-1),
    ('max_replay_work', float(10**12)), ('pfit_maximum_work_bytes', GIB),
    ('pfit_maximum_work_bytes', 4*GIB), ('pfit_maximum_work_bytes', float(2*GIB))])
def test_original_replay_and_native_workspace_limits_are_preserved(monkeypatch, field, value):
    raw, resources = selected_inputs(monkeypatch)
    resources['monomers']['monomer-0008'][field] = value
    with pytest.raises(ValueError, match='exact original replay/native workspace'):
        compute(raw, resources)


def test_selected_candidate_cannot_enter_legacy_all131_entry_point(monkeypatch):
    raw, resources = selected_inputs(monkeypatch)
    with pytest.raises(ValueError, match='closed installed producer'):
        prepare.prepare_bank(raw, {}, fixtures.encode(resources), producer='grid400')
    with pytest.raises(ValueError, match='explicit131-monomer resource declarations'):
        prepare.prepare_bank(raw, {}, fixtures.encode(resources))


def test_full_immutable_raw_inputs_still_required_for_selected_preparation(monkeypatch):
    raw, resources = selected_inputs(monkeypatch)
    raw['entries'] += b' '
    with pytest.raises(ValueError, match='raw input hash differs'):
        compute(raw, resources)


def test_unqualified_candidate_receipts_cannot_be_replaced_by_old_receipts(monkeypatch):
    raw, resources = selected_inputs(monkeypatch)
    original = prepare.helper

    def helper(name):
        if name == 'qualified_grid400':
            def refuse(_):
                raise ValueError('candidate installed receipts required')
            return SimpleNamespace(verify_receipts=refuse)
        return original(name)

    monkeypatch.setattr(prepare, 'helper', helper)
    with pytest.raises(ValueError, match='candidate installed receipts required'):
        compute(raw, resources)
