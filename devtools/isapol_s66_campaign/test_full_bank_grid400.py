"""Closed full-bank descriptor preparation; no native or server work."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixtures = load('_full_grid400_inputs', 'test_prepare_descriptor_bank_cb63305.py')
prepare = fixtures.prepare
profiles = load('_full_grid400_profiles', 'descriptor_profiles.py')
admission = load('_full_grid400_admission', 'descriptor_admission_cb63305.py')
supervisor = load('_full_grid400_supervisor', 'phoenix_descriptor_bank_cb63305.py')


def inputs(monkeypatch):
    docs, contracts = fixtures.fixture()
    raw = fixtures.bind_inputs(monkeypatch, docs)
    original = prepare.helper
    monkeypatch.setattr(prepare, 'helper', lambda name: (
        NS(verify_receipts=lambda _: dict(status='SYNTHETIC_ONLY',
            implementation_sha256=dict(core='a'*64, isapol_python='b'*64)))
        if name == 'qualified_grid400' else original(name)))
    resource = json.loads(fixtures.resources(raw, contracts))
    resource['policy_id'] = '1aaa9e0a30-reference-h2h1-grid400-ledger-v4'
    return raw, resource


def test_explicit_full_scope_never_changes_default_singleton():
    old = profiles.profile('grid400')
    full = profiles.profile('grid400', scope='full-bank')
    assert old['selected_monomers'] == ('monomer-0008',)
    assert full['selected_monomers'] is None
    assert {key: value for key, value in full.items() if key != 'selected_monomers'} == {
        key: value for key, value in old.items() if key != 'selected_monomers'}
    with pytest.raises(ValueError):
        profiles.profile('ed561666', scope='full-bank')
    with pytest.raises(ValueError):
        profiles.profile('grid400', scope='arbitrary-selection')


def test_all131_raw_geometries_and1056_contexts_are_retained(monkeypatch):
    raw, resource = inputs(monkeypatch)
    before = copy.deepcopy((raw, resource))
    bank = prepare.prepare_grid400_full_bank(raw, {}, fixtures.encode(resource))
    assert bank['campaign_scope'] == 'full-bank'
    assert bank['schema'] == 'native-s66x8-descriptor-preparation/2'
    assert bank['status'] == 'PREPARED_FOR_DESCRIPTORS_ONLY'
    assert bank['monomer_count'] == len(bank['monomers']) == 131
    assert bank['fragment_contexts'] == 1056
    assert {row['monomer_id'] for row in bank['monomers']} == {
        f'monomer-{i:04d}' for i in range(1, 132)}
    assert all('sapt_reference' not in json.dumps(row['runtime_input']) for row in bank['monomers'])
    assert (raw, resource) == before


@pytest.mark.parametrize('change', ['missing', 'extra', 'old_policy', 'selected_schema', 'raw_bytes'])
def test_full_scope_requires_exact_complete_inputs_and_explicit_resources(monkeypatch, change):
    raw, resource = inputs(monkeypatch)
    if change == 'missing':
        del resource['monomers']['monomer-0001']
    elif change == 'extra':
        resource['monomers']['monomer-0132'] = copy.deepcopy(resource['monomers']['monomer-0001'])
    elif change == 'old_policy':
        resource['policy_id'] = 'ed561666-reference-h2h1-ledger-v3'
    elif change == 'selected_schema':
        resource['schema'] = 'native-s66x8-selected-resource-declarations/1'
    else:
        raw['entries'] += b' '
    with pytest.raises(ValueError):
        prepare.prepare_grid400_full_bank(raw, {}, fixtures.encode(resource))


def test_full_scope_and_singleton_cannot_authorize_each_other(monkeypatch):
    raw, resource = inputs(monkeypatch)
    with pytest.raises(ValueError):
        prepare.prepare_grid400_pilot_bank(raw, {}, fixtures.encode(resource))
    with pytest.raises(ValueError):
        prepare.prepare_bank(raw, {}, fixtures.encode(resource), producer='grid400')
    one = [dict(monomer_id='monomer-0008', status='PASS_DESCRIPTOR_PLAN_ONLY')]
    with pytest.raises(ValueError):
        admission.verify_coverage(one, {'monomer-0008'})


def test_full_descriptor_route_is_closed_and_separate():
    route = dict(Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small',
                 NumCPUs='24', NumNodes='1', TimeLimit='00:45:00')
    assert supervisor.admit_route(route, scope='full-bank')['cpus'] == 24
    with pytest.raises(ValueError):
        supervisor.admit_route(route)
    with pytest.raises(ValueError):
        supervisor.admit_route(route | {'QOS': 'inferno'}, scope='full-bank')
