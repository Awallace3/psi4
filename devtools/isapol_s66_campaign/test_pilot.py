"""Synthetic preparation tests; no SCF, network or server writes."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('s66_pilot', Path(__file__).with_name('pilot.py'))
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def inputs():
    molecule = dict(symbols=['O', 'H', 'H'], geometry=[0., 0., 0., 1., 0., 0., 0., 1., 0.],
                    molecular_charge=0., molecular_multiplicity=1,
                    fix_com=True, fix_orientation=True, fix_symmetry='c1')
    geometry_sha = pilot.json_digest(molecule)
    entries = dict(inventory_sha256='a'*64, entries=[dict(
        name='monomer-0001-test', molecule=molecule, additional_keywords={},
        attributes=dict(monomer_id='monomer-0001', geometry_sha256=geometry_sha,
                        dimer_contexts=[{'sapt_reference_components_kcal_mol': {'DISP': -2.}}]))])
    declarations = dict(inventory_sha256='a'*64, monomers=[dict(
        monomer_id='monomer-0001', geometry_sha256=geometry_sha,
        declaration=dict(schema='native-monomer-declaration/1',
                         profile='s66-chno-isa-a-v1', frames=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]*3,
                         bonds=[[0, 1], [0, 2]]))])
    qualification = dict(status='passed', runtime_unchanged=True, build_commit='b'*40,
                         build_tree='c'*40, test_source_commit='d'*40, expected_test_count=197,
                         test_summary=dict(tests=197, failures=0, errors=0, skipped=0))
    pins = dict(core='e'*64, isapol_python='f'*64)
    resources = dict(max_bytes=12*1024**3, max_work=6_000_000_000_000,
                     max_io_bytes=16*1024**3, capacity='large')
    return entries, declarations, qualification, pins, resources


def test_water_preparation_binds_execution_but_keeps_references_in_attributes():
    values = inputs()
    original = copy.deepcopy(values)
    result = pilot.prepare_pilot(*values, monomer_id='monomer-0001',
                                 max_replay_work=1_000_000_000_000)
    assert values == original
    assert result['status'] == 'PREPARED_NOT_REGISTERED_OR_SUBMITTED'
    entry = result['entry']
    request = entry['additional_keywords']['function_kwargs']['native_monomer']
    assert request['schema'] == 'native-monomer-request/2'
    assert request['point_group'] == 'c1'
    assert request['implementation_sha256'] == values[3]
    assert request['declaration'] == values[1]['monomers'][0]['declaration']
    assert entry['attributes']['dimer_contexts'] == values[0]['entries'][0]['attributes']['dimer_contexts']
    runtime = json.dumps(dict(molecule=entry['molecule'], keywords=entry['additional_keywords'],
                              specification=result['specification']))
    assert 'sapt_reference' not in runtime and 'DISP' not in runtime and '-2.0' not in runtime
    assert result['submission'] == dict(find_existing=False, maximum_concurrency=1, qos='embers')
    assert result['specification']['program'] == 'psi4'
    assert result['specification']['driver'] == 'properties'
    assert result['specification']['method'] == 'pbe0'
    assert result['specification']['basis'] == 'aug-cc-pvtz'


@pytest.mark.parametrize('changes', [
    {'status': 'failed'}, {'runtime_unchanged': False},
    {'expected_test_count': 196},
    {'test_summary': dict(tests=196, failures=0, errors=0, skipped=0)},
    {'test_summary': dict(tests=197, failures=0, errors=0, skipped=1)},
])
def test_partial_or_unsealed_installed_results_cannot_prepare_pilot(changes):
    values = inputs()
    values[2].update(changes)
    with pytest.raises(ValueError, match='qualification'):
        pilot.prepare_pilot(*values, monomer_id='monomer-0001', max_replay_work=10**12)


@pytest.mark.parametrize('changed', ['geometry', 'declaration', 'inventory', 'runtime', 'duplicate'])
def test_input_identity_changes_are_not_silently_joined(changed):
    entries, declarations, qualification, pins, resources = inputs()
    if changed == 'geometry':
        entries['entries'][0]['molecule']['geometry'][0] = .01
    elif changed == 'declaration':
        declarations['monomers'][0]['geometry_sha256'] = '0'*64
    elif changed == 'inventory':
        declarations['inventory_sha256'] = '0'*64
    elif changed == 'runtime':
        entries['entries'][0]['additional_keywords']['guess'] = 'read'
    else:
        entries['entries'].append(copy.deepcopy(entries['entries'][0]))
    with pytest.raises(ValueError):
        pilot.prepare_pilot(entries, declarations, qualification, pins, resources,
                            monomer_id='monomer-0001', max_replay_work=10**12)


@pytest.mark.parametrize('work', [True, float('inf'), 0, 2_000_000_000_001])
def test_replay_allowance_is_an_explicit_bounded_integer(work):
    with pytest.raises(ValueError, match='resource'):
        pilot.prepare_pilot(*inputs(), monomer_id='monomer-0001', max_replay_work=work)
