"""Pure attempt/endpoint guards, not checkpoint acceptance or live HTTP."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def load(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_bank_transport_test_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TAG = 'ccasp-native-grid400-bank-monomer-0001-123'


def fixture(mode='fresh', *, module=None):
    module = load('bank_transport') if module is None else module
    cases, inputs, pool = load('test_grid400_bank_case').fixture()
    case = cases.prepare_case(inputs, pool, 'monomer-0001')
    raw = None
    if mode == 'append':
        case = cases.prepare_case(inputs, pool, 'monomer-0002')
        snapshot = dict(metadata=dict(id=9999))
        checkpoint = dict(schema='native-audited-bank-checkpoint/1',
            status='READ_ONLY_INITIAL_BASELINE', campaign=case['campaign'], dataset_id=9999,
            eligibility_sha256=case['eligibility_sha256'], bootstrap_context_sha256='b'*64,
            generation=1, snapshot=snapshot,
            snapshot_sha256=module.helper('dataset_guard').digest(snapshot))
        raw = json.dumps(checkpoint).encode()
        case.update(dataset_mode='append', dataset_id=9999,
            bootstrap_context_sha256='b'*64, checkpoint_sha256=hashlib.sha256(raw).hexdigest())
    tag = TAG.replace('monomer-0001', case['monomer_id'])
    scope = module.RequestScope(case, copy.deepcopy(case), tag, checkpoint_raw=raw)
    return module, scope, case, raw


def test_fresh_scope_consumes_each_write_once_and_binds_only_creation_response():
    _, scope, _, _ = fixture()
    scope.request('post', 'api/v1/datasets/singlepoint')
    scope.bind_created(9999)
    for suffix in ('entries/bulkCreate', 'specifications', 'submit'):
        endpoint = 'api/v1/datasets/singlepoint/9999/'+suffix
        scope.request('post', endpoint)
        with pytest.raises(ValueError, match='already attempted'):
            scope.request('post', endpoint)
    assert scope.attempted == {'create', 'entry', 'specification', 'submit'}
    with pytest.raises(ValueError):
        scope.bind_created(9998)


def test_append_scope_has_no_dataset_creation_or_specification_permission():
    _, scope, _, _ = fixture('append')
    for endpoint in ('api/v1/datasets/singlepoint', 'api/v1/datasets/singlepoint/9999/specifications'):
        with pytest.raises(ValueError):
            scope.request('post', endpoint)
    scope.request('post', 'api/v1/datasets/singlepoint/9999/entries/bulkCreate')
    scope.request('post', 'api/v1/datasets/singlepoint/9999/submit')
    assert scope.attempted == {'entry', 'submit'}


@pytest.mark.parametrize('identifier', [353, 354, 355, 356, 0, True])
def test_protected_or_invalid_creation_identity_cannot_bind(identifier):
    _, scope, _, _ = fixture()
    scope.request('post', 'api/v1/datasets/singlepoint')
    with pytest.raises(ValueError):
        scope.bind_created(identifier)


@pytest.mark.parametrize('method,endpoint', [
    ('post', 'api/v1/records/bulkModify'), ('post', 'api/v1/records/bulkDelete'),
    ('post', 'api/v1/datasets/singlepoint/356/submit'),
    ('post', 'api/v1/datasets/singlepoint/9998/entries/bulkCreate'),
    ('post', 'api/v1/datasets/singlepoint/09999/submit'),
    ('post', 'api/v1/datasets/singlepoint/9999/copy_from'),
    ('post', 'api/v1/datasets/singlepoint/9999/background_submit'),
    ('delete', 'api/v1/datasets/singlepoint/9999/entries'),
    ('patch', 'api/v1/datasets/singlepoint/9999/specifications'),
    ('post', 'api/v1/datasets/singlepoint/9999/../355/submit'),
    ('post', '/api/v1/datasets/singlepoint/9999/submit'),
    ('post', 'api/v1/datasets/singlepoint/9999/submit?query=1')])
def test_mutation_alias_or_foreign_target_is_refused(method, endpoint):
    _, scope, _, _ = fixture('append')
    with pytest.raises(ValueError):
        scope.request(method, endpoint)
    assert not scope.attempted


@pytest.mark.parametrize('change', ['case', 'target', 'checkpoint_bytes', 'context',
                                  'eligibility', 'snapshot', 'queue'])
def test_case_and_checkpoint_binding_must_match_reconstruction(change):
    module, _, case, raw = fixture('append')
    reproduced = copy.deepcopy(case)
    if change == 'case':
        case['formula'] = 'changed'
    elif change == 'target':
        case['dataset_id'] = reproduced['dataset_id'] = 356
    elif change == 'checkpoint_bytes':
        raw += b' '
    elif change == 'context':
        case['bootstrap_context_sha256'] = reproduced['bootstrap_context_sha256'] = 'c'*64
    elif change == 'eligibility':
        case['eligibility_sha256'] = reproduced['eligibility_sha256'] = 'c'*64
    elif change == 'queue':
        case['queue_policy']['find_existing'] = reproduced['queue_policy']['find_existing'] = True
    else:
        checkpoint = json.loads(raw)
        checkpoint['snapshot']['metadata']['id'] = 9998
        raw = json.dumps(checkpoint).encode()
        case['checkpoint_sha256'] = reproduced['checkpoint_sha256'] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        module.RequestScope(case, reproduced, TAG.replace('0001', '0002'), checkpoint_raw=raw)
