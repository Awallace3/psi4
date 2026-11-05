"""Exact native-envelope transport identity, including JSON number semantics.

Only the informational Drho metric condition may change float/integer spelling.
No epsilon comparison, numerical solver tolerance, input, producer hash, tensor
artifact, or other metadata is relaxed.
"""
import copy
from decimal import Decimal
import hashlib
import json

POLICY = 'exact-json-diagnostic-number/1'
PATH = ('result', 'provenance', 'partition', 'diagnostics', 'drho_metric_condition')


def _encode(value):
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError('finite JSON native envelope required') from exc


def verify_native_envelope_identity(local, server):
    """Return a reproducible identity witness or raise ValueError on mismatch."""
    for envelope in (local, server):
        if (type(envelope) is not dict or envelope.get('schema') != 'native-monomer-artifacts/2'
                or any(type(envelope.get(key)) is not dict for key in (
                    'request', 'implementation', 'artifacts', 'result'))):
            raise ValueError('complete native envelope2 dictionaries required')
    left_raw, right_raw = _encode(local), _encode(server)
    normalized = None
    if left_raw != right_raw:
        left, right = copy.deepcopy(local), copy.deepcopy(server)
        for key in PATH[:-1]:
            if type(left.get(key)) is not dict or type(right.get(key)) is not dict:
                raise ValueError('native envelope differs beyond diagnostic JSON number spelling')
            left, right = left[key], right[key]
        x, y = left.pop(PATH[-1], None), right.pop(PATH[-1], None)
        if type(x) not in (int, float) or type(y) not in (int, float):
            raise ValueError('plain finite numeric diagnostic required')
        a, b = Decimal(_encode(x).decode()), Decimal(_encode(y).decode())
        if not a.is_finite() or not b.is_finite() or a < 1 or a != b:
            raise ValueError('diagnostic JSON numeric value differs')
        # Remove only that proven-equivalent number, then compare all remaining
        # JSON values/structures exactly, including numeric spellings elsewhere.
        left, right = copy.deepcopy(local), copy.deepcopy(server)
        for envelope in (left, right):
            node = envelope
            for key in PATH[:-1]:
                node = node[key]
            del node[PATH[-1]]
        if _encode(left) != _encode(right):
            raise ValueError('native envelope differs outside diagnostic JSON number spelling')
        normalized = dict(path='/'.join(PATH), local_type=type(x).__name__,
            server_type=type(y).__name__, local_json=_encode(x).decode(),
            server_json=_encode(y).decode(), decimal_value=format(a, 'f'))
    return dict(status='PASS_NATIVE_ENVELOPE_IDENTITY', policy=POLICY,
        local_json_sha256=hashlib.sha256(left_raw).hexdigest(),
        server_json_sha256=hashlib.sha256(right_raw).hexdigest(),
        diagnostic_json_number_equivalence=normalized, epsilon_tolerance_used=False)
