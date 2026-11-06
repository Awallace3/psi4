"""Registration-only one-shot transport and separately owned read-only clients."""
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

_MODULES, _READ_CLIENTS = {}, {}
# QCPortal 0.63 SinglepointRecord result/manager lifecycle fields. Unknown
# fields, user metadata, identity, molecule and specification remain hashed.
LIFECYCLE_FIELDS = frozenset(('status', 'manager_name', 'modified_on', 'properties', 'extras',
                             'compute_history_', 'task_', 'native_files_', 'wavefunction_'))


def helper(name):
    if name not in ('bank_wave', 'bank_wave_campaign', 'bank_transport', 'dataset_guard',
                    'phoenix_pilot', 'dataset_fresh_cb63305'):
        raise ValueError('closed wave transport helper required')
    if name not in _MODULES:
        path = Path(__file__).with_name(name+'.py')
        spec = importlib.util.spec_from_file_location('_wave_transport_'+name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
        _MODULES[name] = module
    return _MODULES[name]


def transport_binding():
    core = helper('bank_wave')
    return dict(helper('bank_transport').transport_binding(),
        wave_adapter_sha256=core.digest(Path(__file__).read_bytes()),
        wave_mutation_policy='registered-wave/1', wave_identity_policy='closed-record-lifecycle/1',
        wave_lifecycle_fields=sorted(LIFECYCLE_FIELDS))


def stable_record_identity(body):
    if type(body) is not dict:
        raise ValueError('full actual record body required')
    return helper('dataset_guard').digest(
        {key: value for key, value in body.items() if key not in LIFECYCLE_FIELDS})


def _snapshot_dataset(client, plan, predecessor_snapshot=None):
    """Enrich the SAME actual record objects captured by dataset_guard.

    No second read is spliced into the observation. The proxy only observes
    the base helper's iterator and leaves prefix record representations intact.
    """
    base = helper('dataset_guard')
    bodies = {}
    class Dataset:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def iterate_records(self, *args, **kwargs):
            for entry, spec, record in self.wrapped.iterate_records(*args, **kwargs):
                bodies[record.id] = (entry, base.plain(record))
                yield entry, spec, record

    class Client:
        def __getattr__(self, name):
            return getattr(client, name)

        def get_dataset_by_id(self, identifier):
            return Dataset(client.get_dataset_by_id(identifier))

    snapshot = base.snapshot_dataset(Client(), dataset_id=357, max_entries=len(plan['eligible_ids']))
    declared = {Path(plan['members'][mid]['case_path']).stem: mid for mid in plan['member_ids']}
    for identifier, (name, body) in bodies.items():
        record = snapshot['records'][str(identifier)]
        if record['body_sha256'] != base.digest(body):
            raise ValueError('record body changed during wave snapshot')
        attributes = snapshot['entries'][name]['attributes']
        prior = (predecessor_snapshot or {}).get('records', {}).get(str(identifier), {})
        if attributes.get('monomer_id') in declared or 'wave_identity_sha256' in prior:
            record['wave_identity_sha256'] = stable_record_identity(body)
    return snapshot


def snapshot_dataset(client, *, binding):
    require_readonly_client(client, binding, monomer_id=binding['monomer_id'])
    return _snapshot_dataset(client, binding['plan'],
        helper('bank_wave').decoded(binding['checkpoint_raw'])['snapshot'])


def _endpoint(method, endpoint, dataset_id):
    if (type(method) is not str or type(endpoint) is not str or not endpoint
            or not endpoint.isascii() or endpoint.startswith('/') or '%' in endpoint
            or '?' in endpoint or '#' in endpoint or '://' in endpoint or '//' in endpoint
            or any(part in ('.', '..') for part in endpoint.split('/'))):
        raise ValueError('plain wave endpoint required')
    method = method.lower()
    if method in ('get', 'head'):
        return 'read'
    if method != 'post':
        raise ValueError('wave permits only exact one-shot POSTs')
    if endpoint in ('auth/v1/login', 'auth/v1/refresh', 'api/v1/managers/query',
                    'api/v1/molecules/bulkGet') or re.fullmatch(
                    r'api/v1/records/(?:singlepoint/)?(?:bulkGet|query)', endpoint):
        return 'read'
    prefix = f'api/v1/datasets/singlepoint/{dataset_id}/'
    if endpoint.startswith(prefix):
        suffix = endpoint[len(prefix):]
        if suffix in ('entries/bulkFetch', 'records/bulkFetch', 'specifications/bulkFetch'):
            return 'read'
        if suffix in ('entries/bulkCreate', 'submit'):
            return suffix
    raise ValueError('closed wave dataset operation allowlist required')


def _engine(check):
    """Same reviewed auth/attempt rules as bank_transport; no serial subclass."""
    from qcportal import PortalClient
    from qcportal.exceptions import AuthenticationFailure
    class WaveClient(PortalClient):
        def _send_request(self, req, allow_retries=True):
            return super()._send_request(req, allow_retries=(
                allow_retries if req.method.lower() in ('get', 'head') else False))

        def _get_JWT_token(self):
            if getattr(self, '_wave_refresh_in_progress', False):
                raise AuthenticationFailure('wave refresh failed; no automatic login fallback')
            return super()._get_JWT_token()

        def _refresh_JWT_token(self):
            self._wave_refresh_in_progress = True
            try:
                return super()._refresh_JWT_token()
            finally:
                self._wave_refresh_in_progress = False

        def _request(self, method, endpoint, **kwargs):
            check(self, method, endpoint, kwargs)
            if method.lower() not in ('get', 'head'):
                kwargs['allow_retries'] = False
                kwargs['internal_retry'] = False
            return super()._request(method, endpoint, **kwargs)
    return WaveClient


def _authority(binding):
    helper('bank_wave_campaign').revalidate_binding(binding)
    if binding['registration'] is None or binding['release_permit'] is None:
        raise ValueError('complete registered/permit-bound authority required')
    core = helper('bank_wave')
    core.decoded(binding['plan_raw'], binding['plan_sha256'])
    core.decoded(binding['registration_raw'], binding['registration_sha256'])
    core.validate_release_permit(binding['plan'], binding['plan_sha256'], binding['registration'],
        binding['registration_sha256'], binding['release_permit_raw'])


def _read_key(binding, mid):
    if mid != binding['monomer_id'] or mid not in binding['plan']['member_ids']:
        raise ValueError('exact bound read-only wave member required')
    return tuple(binding[key] for key in (
        'plan_sha256', 'registration_sha256', 'release_permit_sha256'))+(mid,)


def readonly_client_class(binding, *, monomer_id):
    _authority(binding)
    transport_binding()
    key = _read_key(binding, monomer_id)
    if key not in _READ_CLIENTS:
        def check(client, method, endpoint, kwargs):
            if _endpoint(method, endpoint, 357) != 'read':
                raise ValueError('wave compute/audit dataset client is read-only')
        cls = _engine(check)
        cls._wave_read_authority = key
        _READ_CLIENTS[key] = cls
    return _READ_CLIENTS[key]


def require_readonly_client(client, binding, *, monomer_id):
    key = _read_key(binding, monomer_id)
    if (key not in _READ_CLIENTS or type(client) is not _READ_CLIENTS[key]
            or getattr(client, '_wave_read_authority', None) != key):
        raise ValueError('exact factory-owned read-only wave client required')


class WaveScope:
    """One registration allocation, one typed entry/submit attempt per slot."""
    def __init__(self, binding, cases, members, registration_role):
        helper('bank_wave_campaign').revalidate_binding(binding)
        core = helper('bank_wave')
        if (binding['registration'] is not None or set(cases) != set(binding['plan']['member_ids'])
                or set(members) != set(cases) or registration_role.get('kind') != 'allocation'
                or registration_role.get('job_id') != os.environ.get('SLURM_JOB_ID')):
            raise ValueError('actual single-writer registration role required')
        core.helper('native_controller').admit_audit_allocation(
            registration_role['allocation'], registration_role['job_id'])
        for mid in cases:
            if core.digest(core.encode(cases[mid])) != binding['plan']['members'][mid]['case_sha256']:
                raise ValueError('entire wave registration case differs')
        # Validate held identities and real registration role, without reserving.
        core.helper('bank_store')._wave_body(**helper('bank_wave_campaign').intent_fields(
            binding, members, registration_role), require_job=True)
        self.binding, self.cases, self.members = binding, cases, members
        self.registration_role = registration_role
        self.attempted, self.finished = set(), []
        self.client_type, self.active = None, None

    def select(self, mid):
        if (self.active is not None or len(self.finished) >= len(self.cases)
                or self.binding['plan']['member_ids'][len(self.finished)] != mid):
            raise ValueError('sequential single-entry wave registration required')
        self.active = mid

    def finish(self, mid):
        if self.active != mid or (mid, 'submit') not in self.attempted:
            raise ValueError('verified slot submission required before next entry')
        self.finished.append(mid)
        self.active = None

    def request(self, method, endpoint):
        role = _endpoint(method, endpoint, 357)
        if role == 'read':
            return role
        if self.active is None:
            raise ValueError('one active wave registration slot required')
        token = self.active, role
        if token in self.attempted:
            raise ValueError('wave operation already attempted; never replay')
        if role == 'submit' and (self.active, 'entries/bulkCreate') not in self.attempted:
            raise ValueError('wave entry must precede submit')
        self.attempted.add(token)  # Before validation, auth, HTTP or a lost reply.
        return role


def client_class(scope):
    if type(scope) is not WaveScope or scope.client_type is not None:
        raise ValueError('one unused exact registration scope required')
    transport_binding()
    from qcportal.serialization import serialize
    from qcportal.singlepoint import SinglepointDatasetNewEntry
    from qcportal.dataset_models import DatasetSubmitBody
    core = helper('bank_wave')
    def normalized(role, body):
        if role == 'entries/bulkCreate':
            if type(body) is not list or len(body) != 1:
                raise ValueError('exactly one wave entry required')
            value = [SinglepointDatasetNewEntry(**body[0]).dict()]
        else:
            value = DatasetSubmitBody(**body).dict()
        return core.encode(json.loads(serialize(value, 'application/json')))

    def check(client, method, endpoint, kwargs):
        role = scope.request(method, endpoint)
        if role == 'read':
            return
        if (client.encoding != 'application/json' or kwargs.get('file_data') is not None
                or kwargs.get('url_params') is not None or type(kwargs.get('body')) not in (str, bytes)):
            raise ValueError('typed serialized one-entry wave JSON required')
        case, slot = scope.cases[scope.active], scope.members[scope.active]
        if role == 'entries/bulkCreate':
            expected = [case['entry']]
        else:
            expected = dict(entry_names=[case['entry']['name']],
                specification_names=[case['specification_name']],
                compute_tag=slot['compute_tag'], find_existing=False)
        if normalized(role, json.loads(kwargs['body'])) != normalized(role, expected):
            raise ValueError('exact registered slot payload required')
    cls = _engine(check)
    cls._wave_registration_scope = scope
    scope.client_type = cls
    return cls


def require_registration_client(client, scope):
    if (type(scope) is not WaveScope or scope.client_type is None
            or type(client) is not scope.client_type
            or getattr(client, '_wave_registration_scope', None) is not scope):
        raise ValueError('exact registration-owned wave client required')
