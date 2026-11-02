"""Policy-label transformation only; no descriptor or molecular admission."""
import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).with_name('prepare_ed561666_inputs.py')
spec = importlib.util.spec_from_file_location('cap32_inputs_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def resources():
    return dict(schema='native-s66x8-resource-declarations/2', status='DECLARED_NOT_ADMITTED',
        inventory_sha256=module.prepare_module().INPUT_SHA256['inventory'],
        policy_id='cb63305-reference-h2h1-ledger-v2',
        monomers={f'monomer-{i:04d}': {'synthetic_resource_only': i} for i in range(1, 132)})


def test_only_policy_label_changes_not_contracts_or_admission_status():
    old = resources()
    converted = json.loads(module.rebind_resources(json.dumps(old).encode()))
    assert converted.pop('policy_id') == 'ed561666-reference-h2h1-ledger-v3'
    old.pop('policy_id')
    assert converted == old
    assert converted['status'] == 'DECLARED_NOT_ADMITTED'


@pytest.mark.parametrize('change', [
    dict(policy_id='latest'), dict(status='ADMITTED'), dict(monomers={}),
    dict(inventory_sha256='a'*64), dict(extra='caller override'),
])
def test_unknown_partial_or_already_admitted_source_is_refused(change):
    with pytest.raises(ValueError):
        module.rebind_resources(json.dumps(resources() | change).encode())
