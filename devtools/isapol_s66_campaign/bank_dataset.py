"""Closed eligible-bank adapter around the shared one-shot dataset transactions.

No client is constructed here implicitly. Callers must reproduce cases and
validate append checkpoints/intents from the closed campaign authority first.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

_MODULES = {}


def helper(name):
    if name not in ('bank_transport', 'dataset_fresh_cb63305', 'grid400_bank_case','qomp_campaign'):
        raise ValueError('closed eligible bank dataset helpers required')
    if name not in _MODULES:
        path = Path(__file__).with_name(name+'.py')
        spec = importlib.util.spec_from_file_location('_bank_dataset_'+name, path)
        module = importlib.util.module_from_spec(spec)
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
        _MODULES[name] = module
    return _MODULES[name]


def prepare_transaction(case, reproduced_case, compute_tag, *, checkpoint_raw=None,producer='grid400'):
    return helper('bank_transport').RequestScope(case, reproduced_case, compute_tag,
                                                checkpoint_raw=checkpoint_raw,producer=producer)


def client_class(scope):
    return helper('bank_transport').client_class(scope)


def transport_binding(*,producer='grid400'):
    return helper('bank_transport').transport_binding(**(dict(producer=producer) if producer!='grid400' else {}))


def _validate(client, scope, draft):
    helper('bank_transport').require_client(client, scope)
    if hashlib.sha256(json.dumps(draft, sort_keys=True, allow_nan=False).encode()).hexdigest() != scope.case_sha256:
        raise ValueError('entire reproduced bank draft changed after transport binding')
    cases = helper('qomp_campaign' if scope.producer=='qomp' else 'grid400_bank_case')
    return SimpleNamespace(DATASET_NAME=cases.DATASET_NAME, SPECIFICATION_NAME=cases.SPECIFICATION_NAME)


def register_once(client, scope, draft, save):
    pins = _validate(client, scope, draft)
    if scope.mode != 'fresh' or scope.dataset_id is not None:
        raise ValueError('only the first eligible fresh transaction may register a dataset')
    return helper('dataset_fresh_cb63305')._register_verified(
        client, draft, scope.compute_tag, save, pins=pins, protected=scope.protected,
        description=('Fresh QOMP missing106 cohort; original25 remain separate original provenance. '
            'One ordinary pilot only, no scientific/cohort acceptance.'
            if scope.producer=='qomp' else
            'Fresh measured grid400 eligible monomer subset, ordinary audit required per model; '
            'held descriptor refusals excluded, no whole-set or scientific accuracy claim.'),
        bind=scope.bind_created)


def submit_once(client, scope, dataset, draft, save):
    pins = _validate(client, scope, draft)
    if scope.mode != 'fresh' or scope.dataset_id != dataset.id:
        raise ValueError('exact newly created eligible bank bootstrap dataset required')
    return helper('dataset_fresh_cb63305')._submit_verified(
        client, dataset, draft, scope.compute_tag, save, pins=pins, protected=scope.protected)
