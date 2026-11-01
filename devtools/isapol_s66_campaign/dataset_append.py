"""Exactly two singleton mutation boundaries, with transport retries disabled."""
import hashlib
from pathlib import Path
import re

TRANSPORT_SOURCES = {
    'client_base.py': 'f62aefc6abf3ff6c3349322346ac043f3787b5ad7824ad1abeea2d54f5506f13',
    'client.py': '1aa088a8147be9edcfb5dd247f7ab8ffe886d89912ab0e2866b315fce0ad8d2f',
}
APPEND_ENDPOINTS = frozenset((
    'api/v1/datasets/singlepoint/353/entries/bulkCreate',
    'api/v1/datasets/singlepoint/353/submit'))
_CLIENT_CLASS = None


def transport_binding():
    import qcportal
    if str(qcportal.__version__) != '0.63':
        raise RuntimeError('one-shot append transport requires reviewed QCPortal0.63')
    root = Path(qcportal.__file__).resolve().parent
    observed = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                for name in TRANSPORT_SOURCES}
    if observed != TRANSPORT_SOURCES:
        raise RuntimeError('reviewed QCPortal transport source changed')
    return dict(version='0.63', source_sha256=observed,
                adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def client_class():
    """Keep real model serialization, but never resend an expired-token POST."""
    global _CLIENT_CLASS
    transport_binding()
    if _CLIENT_CLASS is None:
        from qcportal import PortalClient

        class OneShotAppendClient(PortalClient):
            def _request(self, method, endpoint, **kwargs):
                if method.lower() == 'post' and endpoint in APPEND_ENDPOINTS:
                    # QCPortal0.63's expired-token recursion drops allow_retries.
                    # Disable that recursion too: HTTP401 is a hard stop.
                    kwargs['internal_retry'] = False
                    kwargs['allow_retries'] = False
                return super()._request(method, endpoint, **kwargs)

        _CLIENT_CLASS = OneShotAppendClient
    return _CLIENT_CLASS


def require_one_shot_client(client):
    if not isinstance(client, client_class()):
        raise ValueError('singleton writes require the reviewed one-shot transport client')


def insert_entry_once(client, entry):
    require_one_shot_client(client)
    from qcportal.metadata_models import InsertMetadata
    from qcportal.singlepoint import SinglepointDatasetNewEntry
    model = SinglepointDatasetNewEntry(**entry)
    if model.attributes.get('monomer_id') != 'monomer-0004':
        raise ValueError('only the next checkpoint-bound methanol entry')
    meta = client.make_request(
        'post', 'api/v1/datasets/singlepoint/353/entries/bulkCreate',
        InsertMetadata, body=[model], allow_retries=False)
    if not meta.success or meta.n_inserted != 1 or meta.n_existing != 0:
        raise RuntimeError('entry insertion was not exactly one fresh entry; do not retry')
    return meta


def submit_record_once(client, *, entry_name, compute_tag):
    require_one_shot_client(client)
    from qcportal.dataset_models import DatasetSubmitBody
    from qcportal.metadata_models import InsertCountsMetadata
    if re.fullmatch(r'ccasp-native-s66-monomer-0004-[0-9]+', compute_tag) is None:
        raise ValueError('one exact owned methanol compute tag required')
    body = DatasetSubmitBody(
        entry_names=[entry_name], specification_names=['pbe0-atz-native-request2'],
        compute_tag=compute_tag, find_existing=False)
    meta = client.make_request(
        'post', 'api/v1/datasets/singlepoint/353/submit', InsertCountsMetadata,
        body=body, allow_retries=False)
    if not meta.success or meta.n_inserted != 1 or meta.n_existing != 0:
        raise RuntimeError('record submission was not exactly one fresh record; do not retry')
    return meta
