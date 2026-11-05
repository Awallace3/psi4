"""One-attempt dataset transport for a reconstructed eligible-bank transaction.

The caller supplies a case reproduced from closed measured inputs. Append
callers must independently validate the complete ordinary-audit checkpoint
and reserve a one-writer intent before this transport can be used to write.
This module does not confer that authority, repair state, or retry requests.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re


def helper(name):
    if name not in ('grid400_bank_policy', 'grid400_bank_case', 'dataset_fresh_cb63305',
                    'phoenix_worker_probe', 'dataset_guard', 'phoenix_pilot'):
        raise ValueError('closed eligible bank transport helpers required')
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_bank_transport_'+name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def _encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


class RequestScope:
    """Consume mutation attempts before HTTP, including failed/lost responses."""

    def __init__(self, case, reproduced_case, compute_tag, *, checkpoint_raw=None):
        policy, cases = helper('grid400_bank_policy'), helper('grid400_bank_case')
        if (type(case) is not dict or _encoded(case) != _encoded(reproduced_case)
                or case.get('schema') != 'native-selected-molecular-pilot/1'
                or case.get('producer_profile') != 'grid400'
                or case.get('status') != 'PREPARED_NOT_REGISTERED_OR_SUBMITTED'
                or case.get('campaign') != policy.CAMPAIGN
                or case.get('campaign_scope') != 'eligible-bank'
                or case.get('dataset_name') != cases.DATASET_NAME
                or case.get('specification_name') != cases.SPECIFICATION_NAME
                or case.get('dataset_mode') not in ('fresh', 'append')
                or type(case.get('monomer_id')) is not str
                or re.fullmatch(r'monomer-[0-9]{4}', case['monomer_id']) is None
                or not 1 <= int(case['monomer_id'][-4:]) <= 131
                or re.fullmatch(re.escape('ccasp-native-grid400-bank-'+case['monomer_id']+'-')+
                                '[1-9][0-9]*', compute_tag) is None):
            raise ValueError('entire reconstructed eligible-bank transaction required')
        profile = policy.validate_profile(case.get('execution_profile'))
        runtime = helper('phoenix_worker_probe').runtime_input(case)
        declared = policy.validate_execution_profile(dict(runtime_input=runtime,
            intended_memory_bytes=case.get('intended_memory_bytes')))
        if (_encoded(profile) != _encoded(declared)
                or hashlib.sha256(_encoded(runtime)).hexdigest() != case.get('runtime_input_sha256')):
            raise ValueError('exact declared bank runtime and profile required')
        queue = case.get('queue_policy', {})
        if (queue.get('find_existing') is not False or queue.get('existing_ok') is not False
                or queue.get('maximum_records') != 1 or queue.get('concurrency') != 1
                or queue.get('cpus') != profile['cores']
                or queue.get('psi4_memory_GiB') != profile['science_memory_GiB']
                or queue.get('job_memory_GiB') != profile['job_memory_GiB']
                or type(case.get('eligibility_sha256')) is not str
                or re.fullmatch('[0-9a-f]{64}', case['eligibility_sha256']) is None):
            raise ValueError('one finite no-reuse eligible-bank transaction required')
        self.mode, self.dataset_id = case['dataset_mode'], None
        self._draft = json.loads(_encoded(case))
        self.case_sha256 = hashlib.sha256(_encoded(case)).hexdigest()
        self.compute_tag, self.protected = compute_tag, policy.PROTECTED_DATASETS
        self.attempted = set()
        self.client_type = None
        if self.mode == 'fresh':
            if checkpoint_raw is not None or any(key in case for key in (
                    'dataset_id', 'bootstrap_context_sha256', 'checkpoint_sha256')):
                raise ValueError('fresh bank creation cannot bind an existing dataset or checkpoint')
        else:
            identifier = case.get('dataset_id')
            if (type(identifier) is not int or identifier <= 0 or identifier in self.protected
                    or type(case.get('bootstrap_context_sha256')) is not str
                    or re.fullmatch('[0-9a-f]{64}', case['bootstrap_context_sha256']) is None
                    or type(checkpoint_raw) is not bytes
                    or hashlib.sha256(checkpoint_raw).hexdigest() != case.get('checkpoint_sha256')):
                raise ValueError('exact verified fresh-bank append checkpoint bytes required')
            checkpoint = json.loads(checkpoint_raw)
            if (checkpoint.get('schema') != 'native-audited-bank-checkpoint/1'
                    or checkpoint.get('status') not in ('READ_ONLY_INITIAL_BASELINE', 'READ_ONLY_AUDITED_ADVANCE')
                    or checkpoint.get('campaign') != policy.CAMPAIGN
                    or checkpoint.get('dataset_id') != identifier
                    or checkpoint.get('eligibility_sha256') != case.get('eligibility_sha256')
                    or checkpoint.get('bootstrap_context_sha256') != case.get('bootstrap_context_sha256')
                    or type(checkpoint.get('generation')) is not int
                    or not 1 <= checkpoint['generation'] < case.get('eligible_monomer_count', 0)
                    or checkpoint.get('snapshot', {}).get('metadata', {}).get('id') != identifier
                    or checkpoint.get('snapshot_sha256') != helper('dataset_guard').digest(
                        checkpoint['snapshot'])):
                raise ValueError('verified bank checkpoint target/eligibility/context differs')
            self.dataset_id = identifier

    def bind_created(self, identifier):
        if (self.mode != 'fresh' or 'create' not in self.attempted or self.dataset_id is not None
                or type(identifier) is not int or identifier <= 0 or identifier in self.protected):
            raise ValueError('one returned fresh bank dataset identity required')
        self.dataset_id = identifier

    def request(self, method, endpoint):
        if (type(method) is not str or type(endpoint) is not str or not endpoint
                or not endpoint.isascii() or endpoint.startswith('/') or '%' in endpoint
                or '?' in endpoint or '#' in endpoint or '://' in endpoint or '//' in endpoint
                or any(part in ('.', '..') for part in endpoint.split('/'))):
            raise ValueError('plain bank transport method and endpoint required')
        method = method.lower()
        if method in ('get', 'head'):
            return
        if method != 'post':
            raise ValueError('only exact one-shot bank POST operations are allowed')
        if endpoint in ('auth/v1/login', 'auth/v1/refresh', 'api/v1/managers/query',
                        'api/v1/molecules/bulkGet') or re.fullmatch(
                r'api/v1/records/(?:singlepoint/)?(?:bulkGet|query)', endpoint):
            return
        if endpoint == 'api/v1/datasets/singlepoint':
            if self.mode != 'fresh' or self.dataset_id is not None:
                raise ValueError('append transport cannot create a dataset')
            self._consume('create')
            return
        match = re.fullmatch(r'api/v1/datasets/singlepoint/([1-9][0-9]*)/(.*)', endpoint)
        if not match or int(match[1]) != self.dataset_id or int(match[1]) in self.protected:
            raise ValueError('only the transaction-bound new bank dataset may be addressed')
        suffix = match[2]
        if suffix in ('entries/bulkFetch', 'records/bulkFetch', 'specifications/bulkFetch'):
            return
        if suffix == 'entries/bulkCreate':
            self._consume('entry')
        elif suffix == 'specifications' and self.mode == 'fresh':
            self._consume('specification')
        elif suffix == 'submit':
            if 'entry' not in self.attempted or self.mode == 'fresh' and 'specification' not in self.attempted:
                raise ValueError('entry/specification transaction must precede submission')
            self._consume('submit')
        else:
            raise ValueError('closed bank dataset mutation allowlist required')

    def _consume(self, role):
        if role in self.attempted:
            raise ValueError('bank '+role+' was already attempted; no replay')
        self.attempted.add(role)


def transport_binding():
    return dict(helper('dataset_fresh_cb63305').transport_binding(producer='grid400'),
        bank_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bank_mutation_policy='reconstructed-fresh-or-audited-append-one-attempt/1')


def client_class(scope):
    """Dedicated reviewed QCPortal class; one scope shared by any construction."""
    if type(scope) is not RequestScope or scope.client_type is not None:
        raise ValueError('one unused reconstructed bank request scope required')
    transport_binding()
    from qcportal import PortalClient
    from qcportal.exceptions import AuthenticationFailure
    from qcportal.serialization import serialize
    from qcportal.singlepoint import QCSpecification, SinglepointDatasetNewEntry
    from qcportal.singlepoint.dataset_models import SinglepointDatasetSpecification
    from qcportal.dataset_models import DatasetSubmitBody
    draft = scope._draft
    expected = {
        'entries/bulkCreate': [SinglepointDatasetNewEntry(**draft['entry'])],
        'specifications': [SinglepointDatasetSpecification(name=draft['specification_name'],
            specification=QCSpecification(**helper('phoenix_pilot').execution_specification(
                draft['specification'])), description=None)],
        'submit': DatasetSubmitBody(entry_names=[draft['entry']['name']],
            specification_names=[draft['specification_name']], compute_tag=scope.compute_tag,
            find_existing=False),
    }
    models = {'entries/bulkCreate': SinglepointDatasetNewEntry,
              'specifications': SinglepointDatasetSpecification, 'submit': DatasetSubmitBody}

    def normalize_payload(role, body):
        # QCPortal may omit unset optional fields or explicitly send null.
        # Expand the same reviewed metadata-model defaults on both sides;
        # raw native kwargs and entry attributes remain bound as JSON data.
        model = models[role]
        if role != 'submit':
            if type(body) is not list or len(body) != 1 or type(body[0]) is not dict:
                raise ValueError('exact one-entry bank mutation payload required')
            plain = [model(**body[0]).dict()]
        else:
            if type(body) is not dict:
                raise ValueError('exact one-entry bank mutation payload required')
            plain = model(**body).dict()
        return _encoded(json.loads(serialize(plain, 'application/json')))

    expected = {key: normalize_payload(key, json.loads(serialize(value, 'application/json')))
                for key, value in expected.items()}

    class BankClient(PortalClient):
        _bank_scope = scope

        def _send_request(self, req, allow_retries=True):
            # Constructor login and proactive token renewal bypass _request
            # in reviewed QCPortal0.63. Cover that lower-level path as well.
            return super()._send_request(req, allow_retries=(
                allow_retries if req.method.lower() in ('get', 'head') else False))

        def _get_JWT_token(self):
            if getattr(self, '_bank_refresh_in_progress', False):
                raise AuthenticationFailure('bank token refresh failed; no automatic login fallback')
            return super()._get_JWT_token()

        def _refresh_JWT_token(self):
            self._bank_refresh_in_progress = True
            try:
                return super()._refresh_JWT_token()
            finally:
                self._bank_refresh_in_progress = False

        def _request(self, method, endpoint, **kwargs):
            scope.request(method, endpoint)
            if method.lower() not in ('get', 'head'):
                if endpoint.startswith('api/v1/datasets/'):
                    if (self.encoding != 'application/json' or kwargs.get('file_data') is not None
                            or kwargs.get('url_params') is not None
                            or type(kwargs.get('body')) not in (str, bytes)):
                        raise ValueError('exact serialized JSON bank write payload required')
                    body = json.loads(kwargs['body'])
                    if endpoint == 'api/v1/datasets/singlepoint':
                        if (type(body) is not dict or body.get('name') != draft['dataset_name']
                                or body.get('existing_ok') is not False
                                or body.get('default_compute_tag') != scope.compute_tag
                                or _encoded(body.get('provenance')) != _encoded(dict(
                                    producer=draft['producer'],
                                    descriptor_bank_sha256=draft['descriptor_bank_sha256']))):
                            raise ValueError('exact fresh bank creation payload required')
                    else:
                        suffix = endpoint.split('/'+str(scope.dataset_id)+'/', 1)[-1]
                        if suffix in expected and normalize_payload(suffix, body) != expected[suffix]:
                            raise ValueError('exact one-entry bank mutation payload required')
                kwargs['allow_retries'] = False
                kwargs['internal_retry'] = False
            return super()._request(method, endpoint, **kwargs)

    scope.client_type = BankClient
    return BankClient


def require_client(client, scope):
    if (type(scope) is not RequestScope or scope.client_type is None
            or type(client) is not scope.client_type or getattr(client, '_bank_scope', None) is not scope):
        raise ValueError('transaction-owned reviewed one-shot bank client required')
