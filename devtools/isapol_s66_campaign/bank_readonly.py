"""Read-only bank client using the reviewed transaction client's auth guards.

It cannot be passed to the mutation adapter (the exact client type differs),
and only explicit read POSTs survive the HTTP allowlist. This does not create
case/checkpoint authority or accept results.
"""
import re


def readonly_client_class(dataset_adapter, scope, dataset_id):
    if (type(dataset_id) is not int or dataset_id <= 0 or dataset_id in (353, 354, 355, 356)
            or scope.dataset_id is not None and scope.dataset_id != dataset_id):
        raise ValueError('one independently bound new bank dataset for read-only access required')
    parent = scope.client_type or dataset_adapter.client_class(scope)
    if getattr(parent, '_bank_scope', None) is not scope:
        raise ValueError('reviewed case-bound parent transport required')
    from qcportal import PortalClient

    class ReadOnlyBankClient(parent):
        _readonly_bank_dataset_id = dataset_id

        def _request(self, method, endpoint, **kwargs):
            if (type(method) is not str or type(endpoint) is not str or not endpoint
                    or not endpoint.isascii() or endpoint.startswith('/') or '%' in endpoint
                    or '?' in endpoint or '#' in endpoint or '://' in endpoint or '//' in endpoint
                    or any(part in ('.', '..') for part in endpoint.split('/'))):
                raise ValueError('plain read-only bank HTTP request required')
            method = method.lower()
            if method not in ('get', 'head'):
                allowed = endpoint in ('auth/v1/login', 'auth/v1/refresh',
                    'api/v1/managers/query', 'api/v1/molecules/bulkGet') or re.fullmatch(
                    r'api/v1/records/(?:singlepoint/)?(?:bulkGet|query)', endpoint) is not None
                match = re.fullmatch(
                    r'api/v1/datasets/singlepoint/([1-9][0-9]*)/(entries|records|specifications)/bulkFetch',
                    endpoint)
                allowed = allowed or match is not None and int(match[1]) == dataset_id
                if method != 'post' or not allowed:
                    raise ValueError('bank audit client is read-only; mutations are forbidden')
                kwargs['allow_retries'] = False
                kwargs['internal_retry'] = False
            # Deliberately use the reviewed base read implementation, not the
            # parent's mutation scope. Parent _send_request/auth guards remain.
            return PortalClient._request(self, method, endpoint, **kwargs)

    return ReadOnlyBankClient
