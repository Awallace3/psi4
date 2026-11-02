"""Fresh cb/request3 singleton writes, never historical dataset353 appends.

All mutating HTTP requests disable both retries in the reviewed QCPortal0.63
stack. Lost responses and expired tokens are hard stops, not retry permission.
"""
import hashlib
import importlib.util
from pathlib import Path
import re

_CLIENT_CLASS = None


def helper(name):
    if name not in ('dataset_append', 'pilot_cb63305', 'phoenix_pilot'):
        raise ValueError('closed fresh-dataset helpers required')
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_fresh_dataset_'+name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def transport_binding():
    # Reuse only the reviewed ordinary client/source-version check. Historical
    # append endpoints, owner gates and acf54 receipts are not reused.
    binding = helper('dataset_append').transport_binding()
    return dict(version=binding['version'], source_sha256=binding['source_sha256'],
                adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def client_class():
    global _CLIENT_CLASS
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


def require_client(client):
    if not isinstance(client, client_class()):
        raise ValueError('fresh singleton writes require reviewed one-shot transport')


def validate_draft(draft, compute_tag):
    pins = helper('pilot_cb63305')
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


def register_once(client, draft, compute_tag, save):
    """Create one new dataset, entry and spec; no recovery/overwrite behavior."""
    require_client(client)
    pins = validate_draft(draft, compute_tag)
    from qcportal.singlepoint import QCSpecification, SinglepointDatasetNewEntry
    entry = SinglepointDatasetNewEntry(**draft['entry'])
    specification = QCSpecification(**helper('phoenix_pilot').execution_specification(draft['specification']))
    save('dataset-create-attempt.json', dict(name=pins.DATASET_NAME, tag=compute_tag,
                                            existing_ok=False, transport_retries=False))
    dataset = client.add_dataset('singlepoint', pins.DATASET_NAME, existing_ok=False,
        default_tag=compute_tag,
        description='Fresh cb63305 native PBE0/aug-cc-pVTZ request3 monomers; '
                    'first methylamine pilot only; no whole-set/accuracy/parity claim.',
        provenance=dict(producer=draft['producer'], descriptor_bank_sha256=draft['descriptor_bank_sha256']))
    if (type(dataset.id) is not int or dataset.id <= 0 or dataset.id == 353
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


def submit_once(client, dataset, draft, compute_tag, save):
    require_client(client)
    pins = validate_draft(draft, compute_tag)
    if (type(dataset.id) is not int or dataset.id <= 0 or dataset.id == 353
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
