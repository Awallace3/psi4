"""Fresh cb/request3 singleton writes, never historical dataset353 appends.

All mutating HTTP requests disable both retries in the reviewed QCPortal0.63
stack. Lost responses and expired tokens are hard stops, not retry permission.
"""
import hashlib
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
import re
from types import SimpleNamespace

_CLIENT_CLASS = None
_ED_CLIENT_CLASS = None
_GRID400_CLIENT_CLASS = None


def helper(name):
    if name not in ('dataset_append', 'pilot_cb63305', 'phoenix_pilot',
                    'grid400_policy', 'phoenix_worker_probe'):
        raise ValueError('closed fresh-dataset helpers required')
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_fresh_dataset_'+name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def transport_binding(*, producer='cb63305'):
    # Reuse only the reviewed ordinary client/source-version check. Historical
    # append endpoints, owner gates and acf54 receipts are not reused.
    if type(producer) is not str or producer not in ('cb63305', 'ed561666', 'grid400'):
        raise ValueError('closed fresh transport producer required')
    binding = helper('dataset_append').transport_binding()
    result = dict(version=binding['version'], source_sha256=binding['source_sha256'],
                  adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if producer in ('ed561666', 'grid400'):
        import qcportal
        path = Path(qcportal.__file__).resolve().with_name('manager_client.py')
        manager_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if manager_sha != 'fd4147c5cd163c4431d24b7afd5b1e2edd398d452aa3cff6fc79450ed127e6b3':
            raise RuntimeError('reviewed singleton ManagerClient transport source changed')
        result.update(producer=producer, mutation_policy=(
                      'fresh-ed-protect353-354-no-record-writes/1' if producer == 'ed561666'
                      else 'fresh-grid400-protect353-354-355-no-record-writes/1'),
                      manager_source_sha256={'manager_client.py': manager_sha},
                      manager_mutation_policy='no-claim-or-delivery-http-retries/1')
        if producer == 'grid400':
            result['manager_activation_policy'] = 'owned-one-shot-client-factory-including-activation/1'
            result['manager_delivery_policy'] = 'one-result-return-attempt-no-deferred-or-shutdown-replay/1'
            result['dataset_scope_policy'] = 'one-created-dataset-exact-method-endpoint-allowlist/2'
    return result


def guard_manager_client(client):
    """Close both HTTP retry paths on this owned manager before its first claim.

    ComputeManager constructs/activates its client before returning. This
    instance-only adapter covers subsequent claim, return and deactivation;
    it neither changes the deployed package nor the historical cb manager.
    """
    transport_binding(producer='ed561666')
    from qcportal.manager_client import ManagerClient
    if not isinstance(client, ManagerClient) or '_request' in client.__dict__:
        raise ValueError('unmodified owned ManagerClient instance required')
    request = client._request

    def one_shot(method, endpoint, **kwargs):
        if method.lower() not in ('get', 'head'):
            kwargs['allow_retries'] = False
            kwargs['internal_retry'] = False
        return request(method, endpoint, **kwargs)

    client._request = one_shot


def _guard_ed_mutation(endpoint):
    return _guard_selected_mutation(endpoint, protected=(353, 354))


def _protected_datasets(producer):
    if producer == 'cb63305':
        return (353,)
    if producer == 'ed561666':
        return (353, 354)
    if producer == 'grid400':
        return helper('grid400_policy').PROTECTED_DATASETS
    raise ValueError('closed protected dataset profile required')


def _guard_selected_mutation(endpoint, *, protected):
    # This fresh-dataset client never needs record state mutation, for ANY
    # record. Denying those endpoints also protects bulk IDs carried in bodies.
    if type(endpoint) is not str:
        raise ValueError('plain explicit fresh mutation endpoint required')
    route = endpoint.split('?', 1)[0].split('#', 1)[0]
    if (not route or not route.isascii() or route.startswith('/') or '%' in route
            or '://' in route or '//' in route or any(p in ('.', '..') for p in route.split('/'))):
        raise ValueError('plain explicit fresh mutation endpoint required')
    if protected not in ((353, 354), (353, 354, 355)):
        raise ValueError('closed protected dataset tuple required')
    dataset = re.match(r'^api/v1/datasets/(?:singlepoint/)?([0-9]+)(?:/|$)', route)
    if dataset and int(dataset[1]) in protected:
        raise ValueError('protected datasets'+'/'.join(map(str, protected))+
                         ' are never fresh-producer mutation targets')
    if re.match(r'^api/v1/records(?:/|$)', route) and re.fullmatch(
            r'api/v1/records/(?:singlepoint/)?(?:bulkGet|query)', route) is None:
        raise ValueError('record mutations are forbidden in the fresh-producer client')


def _guard_grid400_mutation(client, method, endpoint):
    """Only create once, then address the returned new dataset, or read."""
    _guard_selected_mutation(endpoint, protected=_protected_datasets('grid400'))
    route = endpoint.split('?', 1)[0].split('#', 1)[0]
    if method != 'post' or route != endpoint:
        raise ValueError('closed fresh grid400 POST method and endpoint required')
    if route == 'api/v1/datasets/singlepoint':
        if getattr(client, '_grid400_create_attempted', False):
            raise ValueError('only one fresh grid400 dataset creation attempt is allowed')
        client._grid400_create_attempted = True
        return
    dataset = re.fullmatch(r'api/v1/datasets/(?:singlepoint/)?([0-9]+)(?:/(.*))?', route)
    if dataset:
        identifier = int(dataset[1])
        if (dataset[1] != str(identifier)
                or getattr(client, '_grid400_dataset_id', None) != identifier):
            raise ValueError('only the newly bound grid400 dataset may be mutated')
        # ID binding is necessary but not sufficient: never allow copy,
        # background_submit, record reset/cancel/retag, or deletion beneath it.
        suffix = dataset[2]
        writes = {'entries/bulkCreate': '_grid400_entry_attempted',
                  'specifications': '_grid400_specification_attempted',
                  'submit': '_grid400_submit_attempted'}
        if route.startswith('api/v1/datasets/singlepoint/') and suffix in writes:
            flag = writes[suffix]
            if getattr(client, flag, False):
                raise ValueError('only one fresh grid400 '+suffix+' attempt is allowed')
            setattr(client, flag, True)
            return
        if route.startswith('api/v1/datasets/singlepoint/') and suffix in (
                'entries/bulkFetch', 'records/bulkFetch', 'specifications/bulkFetch'):
            return
        raise ValueError('closed fresh grid400 dataset method and endpoint required')
    if (route in ('auth/v1/login', 'auth/v1/refresh', 'api/v1/managers/query',
                  'api/v1/molecules/bulkGet')
            or re.fullmatch(r'api/v1/records/(?:singlepoint/)?(?:bulkGet|query)', route)):
        return
    raise ValueError('closed fresh grid400 mutation/read endpoint required')


def _bind_grid400_dataset(client, identifier):
    require_client(client, producer='grid400')
    if (type(identifier) is not int or identifier <= 0 or identifier in _protected_datasets('grid400')
            or getattr(client, '_grid400_create_attempted', False) is not True
            or getattr(client, '_grid400_dataset_id', None) is not None):
        raise ValueError('one successfully created fresh grid400 dataset identity required')
    client._grid400_dataset_id = identifier


def client_class(*, producer='cb63305'):
    global _CLIENT_CLASS, _ED_CLIENT_CLASS, _GRID400_CLIENT_CLASS
    if type(producer) is not str or producer not in ('cb63305', 'ed561666', 'grid400'):
        raise ValueError('closed fresh client producer required')
    if producer in ('ed561666', 'grid400'):
        transport_binding(producer=producer)
        cached = _ED_CLIENT_CLASS if producer == 'ed561666' else _GRID400_CLIENT_CLASS
        if cached is None:
            from qcportal import PortalClient
            protected = _protected_datasets(producer)

            class FreshSelectedClient(PortalClient):
                def _request(self, method, endpoint, **kwargs):
                    if method.lower() not in ('get', 'head'):
                        if producer == 'grid400':
                            _guard_grid400_mutation(self, method.lower(), endpoint)
                        else:
                            _guard_selected_mutation(endpoint, protected=protected)
                        kwargs['allow_retries'] = False
                        kwargs['internal_retry'] = False
                    return super()._request(method, endpoint, **kwargs)

            cached = FreshSelectedClient
            if producer == 'ed561666':
                _ED_CLIENT_CLASS = cached
            else:
                _GRID400_CLIENT_CLASS = cached
        return cached
    transport_binding()
    if _CLIENT_CLASS is None:
        from qcportal import PortalClient

        class FreshNativeClient(PortalClient):
            def _request(self, method, endpoint, **kwargs):
                if method.lower() not in ('get', 'head'):
                    if re.match(r'^api/v1/datasets/(?:singlepoint/)?353(?:/|$)', endpoint):
                        raise ValueError('historical dataset353 is never a fresh-pilot mutation target')
                    kwargs['allow_retries'] = False
                    kwargs['internal_retry'] = False
                return super()._request(method, endpoint, **kwargs)

        _CLIENT_CLASS = FreshNativeClient
    return _CLIENT_CLASS


@contextmanager
def grid400_manager_client_factory():
    """Guard even activation during construction of this one owned manager.

    The dedicated process has one manager. Substitute its constructor's client
    class only during construction and restore it immediately, including on
    failure. No deployed files or other processes/managers are modified.
    """
    transport_binding(producer='grid400')
    from qcportal.manager_client import ManagerClient
    import qcfractalcompute.compute_manager as compute
    if compute.ManagerClient is not ManagerClient:
        raise ValueError('unmodified manager client constructor binding required')

    class OneShotManagerClient(ManagerClient):
        def return_finished(self, results_compressed):
            # HTTP retry flags do not stop ComputeManager's deferred-result
            # queue. Fail outside its AllowedConnectionExceptions on the first
            # ambiguous return, and retain the attempt bit through shutdown.
            if getattr(self, '_grid400_delivery_attempted', False):
                raise RuntimeError('grid400 result delivery was already attempted; no replay')
            if type(results_compressed) is not dict or len(results_compressed) != 1:
                raise ValueError('one fresh grid400 task result required')
            self._grid400_delivery_attempted = True
            try:
                return super().return_finished(results_compressed)
            except Exception as exc:
                raise RuntimeError('grid400 result delivery failed or is ambiguous; no replay') from exc

        def _request(self, method, endpoint, **kwargs):
            if method.lower() not in ('get', 'head'):
                kwargs['allow_retries'] = False
                kwargs['internal_retry'] = False
            return super()._request(method, endpoint, **kwargs)

    compute.ManagerClient = OneShotManagerClient
    try:
        yield OneShotManagerClient
    finally:
        compute.ManagerClient = ManagerClient


def require_client(client, *, producer='cb63305'):
    selected = client_class() if producer == 'cb63305' else client_class(producer=producer)
    if not isinstance(client, selected):
        raise ValueError('fresh singleton writes require reviewed one-shot transport')


def validate_draft(draft, compute_tag, *, producer='cb63305'):
    if type(producer) is not str or producer not in ('cb63305', 'ed561666', 'grid400'):
        raise ValueError('closed fresh draft producer required')
    pins = helper('pilot_cb63305')
    if producer in ('ed561666', 'grid400'):
        selected = pins.producer_pins(producer)
        if (draft.get('schema') != selected['schema']
                or draft.get('producer_profile') != producer or draft.get('dataset_mode') != 'fresh'
                or draft.get('status') != 'PREPARED_NOT_REGISTERED_OR_SUBMITTED'
                or draft.get('monomer_id') != selected['monomer']
                or draft.get('dataset_name') != selected['dataset_name']
                or draft.get('specification_name') != selected['specification_name']
                or draft.get('runtime_input_sha256') != selected['runtime_input_sha256']
                or draft.get('descriptor_bank_sha256') != selected['bank_sha256']
                or draft.get('descriptor_allocation_sha256') != selected['allocation_sha256']
                or any(k in draft for k in ('dataset_id', 'required_first_pilot_audit_sha256',
                                           'owned_checkpoint', 'checkpoint_sha256'))
                or draft.get('queue_policy', {}).get('find_existing') is not False
                or draft['queue_policy'].get('existing_ok') is not False
                or draft['queue_policy'].get('concurrency') != 1
                or draft['queue_policy'].get('maximum_records') != 1
                or re.fullmatch(re.escape(selected['namespace']+selected['monomer']+'-')+
                                r'[1-9][0-9]*', compute_tag) is None):
            raise ValueError('exact reconstructed fresh cap32 pilot and tag required')
        if producer == 'grid400':
            policy = helper('grid400_policy')
            profile = policy.validate_profile(draft.get('execution_profile'))
            runtime = helper('phoenix_worker_probe').runtime_input(draft)
            policy.validate_execution_profile(dict(monomer_id=draft['monomer_id'],
                runtime_input=runtime, intended_memory_bytes=draft.get('intended_memory_bytes')))
            if (hashlib.sha256(json.dumps(runtime, sort_keys=True, allow_nan=False).encode()).hexdigest()
                    != selected['runtime_input_sha256']
                    or draft['queue_policy'].get('cpus') != profile['cores']
                    or draft['queue_policy'].get('psi4_memory_GiB') != profile['science_memory_GiB']
                    or draft['queue_policy'].get('job_memory_GiB') != profile['job_memory_GiB']):
                raise ValueError('exact fresh grid400 runtime and approved singleton shape required')
        return SimpleNamespace(DATASET_NAME=selected['dataset_name'],
                               SPECIFICATION_NAME=selected['specification_name'])
    if (draft.get('schema') != 'native-expanded-molecular-pilot/1'
            or draft.get('status') != 'PREPARED_NOT_REGISTERED_OR_SUBMITTED'
            or draft.get('monomer_id') != pins.MONOMER
            or draft.get('dataset_name') != pins.DATASET_NAME
            or draft.get('specification_name') != pins.SPECIFICATION_NAME
            or draft.get('queue_policy', {}).get('find_existing') is not False
            or draft['queue_policy'].get('existing_ok') is not False
            or draft['queue_policy'].get('concurrency') != 1
            or draft['queue_policy'].get('maximum_records') != 1
            or re.fullmatch(r'ccasp-native-cb63305-monomer-0006-[0-9]+', compute_tag) is None):
        raise ValueError('exact first expanded-pilot draft and dedicated tag required')
    return pins


def register_once(client, draft, compute_tag, save, *, producer='cb63305'):
    """Create one new dataset, entry and spec; no recovery/overwrite behavior."""
    if type(producer) is not str or producer not in ('cb63305', 'ed561666', 'grid400'):
        raise ValueError('closed fresh registration producer required')
    selection = {} if producer == 'cb63305' else dict(producer=producer)
    require_client(client, **selection)
    pins = validate_draft(draft, compute_tag, **selection)
    from qcportal.singlepoint import QCSpecification, SinglepointDatasetNewEntry
    entry = SinglepointDatasetNewEntry(**draft['entry'])
    specification = QCSpecification(**helper('phoenix_pilot').execution_specification(draft['specification']))
    save('dataset-create-attempt.json', dict(name=pins.DATASET_NAME, tag=compute_tag,
                                            existing_ok=False, transport_retries=False))
    dataset = client.add_dataset('singlepoint', pins.DATASET_NAME, existing_ok=False,
        default_tag=compute_tag,
        description=('Fresh cb63305 native PBE0/aug-cc-pVTZ request3 monomers; '
                     'first methylamine pilot only; no whole-set/accuracy/parity claim.'
                     if producer == 'cb63305' else
                     'Fresh grid400 native PBE0/aug-cc-pVTZ request3 monomer0008 pilot only; '
                     'not whole-set molecular acceptance or accuracy/parity.'
                     if producer == 'grid400' else
                     'Fresh ed561666 native PBE0/aug-cc-pVTZ request3 benzene pilot only; '
                     'not whole-set molecular acceptance or accuracy/parity.'),
        provenance=dict(producer=draft['producer'], descriptor_bank_sha256=draft['descriptor_bank_sha256']))
    protected = _protected_datasets(producer)
    if (type(dataset.id) is not int or dataset.id <= 0 or dataset.id in protected
            or dataset.name != pins.DATASET_NAME or getattr(dataset, '_client', None) is not client):
        raise RuntimeError('fresh dataset returned an invalid or historical identity; do not retry')
    if producer == 'grid400':
        _bind_grid400_dataset(client, dataset.id)
    save('dataset-created.json', dict(dataset_id=dataset.id, name=pins.DATASET_NAME))
    save('entry-insert-attempt.json', dict(dataset_id=dataset.id, entry=entry.name,
                                         transport_retries=False))
    meta = dataset.add_entries([entry])
    if not meta.success or meta.n_inserted != 1 or meta.n_existing != 0:
        raise RuntimeError('entry insertion was not exactly one fresh entry; do not retry')
    save('specification-insert-attempt.json', dict(dataset_id=dataset.id,
        specification=pins.SPECIFICATION_NAME, transport_retries=False))
    meta = dataset.add_specification(pins.SPECIFICATION_NAME, specification)
    if not meta.success or meta.n_inserted != 1 or meta.n_existing != 0:
        raise RuntimeError('specification insertion was not exactly one fresh specification; do not retry')
    stored = dataset.get_entry(entry.name, force_refetch=True)
    if (stored.attributes != entry.attributes or stored.additional_keywords != entry.additional_keywords
            or dataset.get_record(entry.name, pins.SPECIFICATION_NAME) is not None):
        raise RuntimeError('stored singleton entry changed or already has a record')
    return dataset, specification


def submit_once(client, dataset, draft, compute_tag, save, *, producer='cb63305'):
    if type(producer) is not str or producer not in ('cb63305', 'ed561666', 'grid400'):
        raise ValueError('closed fresh submission producer required')
    selection = {} if producer == 'cb63305' else dict(producer=producer)
    require_client(client, **selection)
    pins = validate_draft(draft, compute_tag, **selection)
    protected = _protected_datasets(producer)
    if (type(dataset.id) is not int or dataset.id <= 0 or dataset.id in protected
            or dataset.name != pins.DATASET_NAME or getattr(dataset, '_client', None) is not client):
        raise ValueError('only the newly bound expanded-pilot dataset may be submitted')
    if producer == 'grid400' and getattr(client, '_grid400_dataset_id', None) != dataset.id:
        raise ValueError('submission requires the exact newly created grid400 dataset binding')
    save('record-submit-attempt.json', dict(dataset_id=dataset.id, entry=draft['entry']['name'],
        specification=pins.SPECIFICATION_NAME, compute_tag=compute_tag,
        find_existing=False, transport_retries=False))
    meta = dataset.submit(entry_names=[draft['entry']['name']],
        specification_names=[pins.SPECIFICATION_NAME], compute_tag=compute_tag, find_existing=False)
    if not meta.success or meta.n_inserted != 1 or meta.n_existing != 0:
        raise RuntimeError('record insertion was not exactly one fresh record; do not retry')
    return meta
