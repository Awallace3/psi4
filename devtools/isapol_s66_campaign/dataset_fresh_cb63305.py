"""Fresh cb/request3 singleton writes, never historical dataset353 appends.

All mutating HTTP requests disable both retries in the reviewed QCPortal0.63
stack. Lost responses and expired tokens are hard stops, not retry permission.
"""
import hashlib
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace

_CLIENT_CLASS = None
_ED_CLIENT_CLASS = None


def helper(name):
    if name not in ('dataset_append', 'pilot_cb63305', 'phoenix_pilot'):
        raise ValueError('closed fresh-dataset helpers required')
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_fresh_dataset_'+name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def transport_binding(*, producer='cb63305'):
    # Reuse only the reviewed ordinary client/source-version check. Historical
    # append endpoints, owner gates and acf54 receipts are not reused.
    if type(producer) is not str or producer not in ('cb63305', 'ed561666'):
        raise ValueError('closed fresh transport producer required')
    binding = helper('dataset_append').transport_binding()
    result = dict(version=binding['version'], source_sha256=binding['source_sha256'],
                  adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if producer == 'ed561666':
        import qcportal
        path = Path(qcportal.__file__).resolve().with_name('manager_client.py')
        manager_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if manager_sha != 'fd4147c5cd163c4431d24b7afd5b1e2edd398d452aa3cff6fc79450ed127e6b3':
            raise RuntimeError('reviewed singleton ManagerClient transport source changed')
        result.update(producer=producer, mutation_policy='fresh-ed-protect353-354-no-record-writes/1',
                      manager_source_sha256={'manager_client.py': manager_sha},
                      manager_mutation_policy='no-claim-or-delivery-http-retries/1')
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
    # This fresh-dataset client never needs record state mutation, for ANY
    # record. Denying those endpoints also protects bulk IDs carried in bodies.
    if type(endpoint) is not str:
        raise ValueError('plain explicit fresh mutation endpoint required')
    route = endpoint.split('?', 1)[0].split('#', 1)[0]
    if (not route or not route.isascii() or route.startswith('/') or '%' in route
            or '://' in route or '//' in route or any(p in ('.', '..') for p in route.split('/'))):
        raise ValueError('plain explicit fresh mutation endpoint required')
    if re.match(r'^api/v1/datasets/(?:singlepoint/)?(?:353|354)(?:/|$)', route):
        raise ValueError('protected datasets353/354 are never fresh-producer mutation targets')
    if re.match(r'^api/v1/records(?:/|$)', route) and re.fullmatch(
            r'api/v1/records/(?:singlepoint/)?(?:bulkGet|query)', route) is None:
        raise ValueError('record mutations are forbidden in the fresh-producer client')


def client_class(*, producer='cb63305'):
    global _CLIENT_CLASS, _ED_CLIENT_CLASS
    if type(producer) is not str or producer not in ('cb63305', 'ed561666'):
        raise ValueError('closed fresh client producer required')
    if producer == 'ed561666':
        transport_binding(producer=producer)
        if _ED_CLIENT_CLASS is None:
            from qcportal import PortalClient

            class FreshCap32Client(PortalClient):
                def _request(self, method, endpoint, **kwargs):
                    if method.lower() not in ('get', 'head'):
                        _guard_ed_mutation(endpoint)
                        kwargs['allow_retries'] = False
                        kwargs['internal_retry'] = False
                    return super()._request(method, endpoint, **kwargs)

            _ED_CLIENT_CLASS = FreshCap32Client
        return _ED_CLIENT_CLASS
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


def require_client(client, *, producer='cb63305'):
    selected = client_class() if producer == 'cb63305' else client_class(producer=producer)
    if not isinstance(client, selected):
        raise ValueError('fresh singleton writes require reviewed one-shot transport')


def validate_draft(draft, compute_tag, *, producer='cb63305'):
    if type(producer) is not str or producer not in ('cb63305', 'ed561666'):
        raise ValueError('closed fresh draft producer required')
    pins = helper('pilot_cb63305')
    if producer == 'ed561666':
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
    if type(producer) is not str or producer not in ('cb63305', 'ed561666'):
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
                     'Fresh ed561666 native PBE0/aug-cc-pVTZ request3 benzene pilot only; '
                     'not whole-set molecular acceptance or accuracy/parity.'),
        provenance=dict(producer=draft['producer'], descriptor_bank_sha256=draft['descriptor_bank_sha256']))
    protected = (353,) if producer == 'cb63305' else (353, 354)
    if (type(dataset.id) is not int or dataset.id <= 0 or dataset.id in protected
            or dataset.name != pins.DATASET_NAME or getattr(dataset, '_client', None) is not client):
        raise RuntimeError('fresh dataset returned an invalid or historical identity; do not retry')
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
    if type(producer) is not str or producer not in ('cb63305', 'ed561666'):
        raise ValueError('closed fresh submission producer required')
    selection = {} if producer == 'cb63305' else dict(producer=producer)
    require_client(client, **selection)
    pins = validate_draft(draft, compute_tag, **selection)
    protected = (353,) if producer == 'cb63305' else (353, 354)
    if (type(dataset.id) is not int or dataset.id <= 0 or dataset.id in protected
            or dataset.name != pins.DATASET_NAME or getattr(dataset, '_client', None) is not client):
        raise ValueError('only the newly bound expanded-pilot dataset may be submitted')
    save('record-submit-attempt.json', dict(dataset_id=dataset.id, entry=draft['entry']['name'],
        specification=pins.SPECIFICATION_NAME, compute_tag=compute_tag,
        find_existing=False, transport_retries=False))
    meta = dataset.submit(entry_names=[draft['entry']['name']],
        specification_names=[pins.SPECIFICATION_NAME], compute_tag=compute_tag, find_existing=False)
    if not meta.success or meta.n_inserted != 1 or meta.n_existing != 0:
        raise RuntimeError('record insertion was not exactly one fresh record; do not retry')
    return meta
