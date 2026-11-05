"""Private durable one-writer intents and content-addressed audit proof bytes.

Only metadata is written. Existing/partial intents are never cleared or
repaired; aliases of the same canonical dataset snapshot share one intent.
These are ordinary-OS guards, not atomic QCArchive locks or host attestation.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat

# Verified owned home750, unlike the existing group-writable project runs775.
# Only new private children are created; shared hierarchies are never chmodded.
ROOT = Path('/storage/home/hcoda1/3/awallace43')
INTENTS = ROOT/'ccasp-native-s66-grid400-1aaa9e0a30-bank-dispatch-intents'
PROOFS = ROOT/'ccasp-native-s66-grid400-1aaa9e0a30-bank-audit-proofs'
PROOF_ROLES = frozenset(('audit', 'outcome', 'case', 'envelope'))
MAX_PROOF_BYTES = 512*1024**2


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('exact bank metadata SHA256 required')
    return value


def _sync(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _directory(path, *, private):
    info = path.lstat()
    mode = stat.S_IMODE(info.st_mode)
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or (mode & 0o777 != 0o700 if private else bool(mode & 0o022))):
        raise ValueError('owned private bank metadata directory and non-writable container required')


def _private_root(path):
    # The containing directory must already be owned and non-writable
    # by other accounts. Never create or chmod an existing shared hierarchy.
    _directory(path.parent, private=False)
    try:
        path.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError:
        pass
    _directory(path, private=True)
    # Sync even an existing directory's link: it may have been created by a
    # concurrent same-owner setup process which has not yet synced its parent.
    _sync(path.parent)


def _write_once(path, raw):
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _sync(path.parent)


def prepare_storage():
    """Validate both private stores before discovery or scientific execution.

    This creates no writer reservation, QCA object or proof and repairs nothing.
    """
    for path in (INTENTS, PROOFS):
        _private_root(path)


def _read_regular(path, *, maximum, durable=False):
    """Bound the opened regular descriptor, not a racy pathname size check."""
    if type(maximum) is not int or maximum < 0:
        raise ValueError('finite bank metadata read bound required')
    try:
        fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise ValueError('plain bounded regular bank metadata file required') from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ValueError('bounded regular bank metadata descriptor required')
        chunks, size = [], 0
        while True:
            block = os.read(fd, min(1024**2, maximum-size+1))
            if not block:
                break
            size += len(block)
            if size > maximum:
                raise ValueError('bank metadata grew beyond its read bound')
            chunks.append(block)
        after = os.fstat(fd)
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if (size != after.st_size
                or any(getattr(before, key) != getattr(after, key) for key in fields)):
            raise ValueError('bank metadata changed while being read')
        if durable:
            os.fsync(fd)
        return b''.join(chunks)
    finally:
        os.close(fd)


def _body(*, dataset_id, checkpoint_sha256, snapshot_sha256, context_sha256,
          case_sha256, eligibility_sha256, monomer_id, compute_tag, require_job):
    if (type(monomer_id) is not str or re.fullmatch('monomer-[0-9]{4}', monomer_id) is None
            or not 1 <= int(monomer_id[-4:]) <= 131
            or type(compute_tag) is not str or re.fullmatch(
                re.escape('ccasp-native-grid400-bank-'+monomer_id+'-')+'[1-9][0-9]{0,19}', compute_tag) is None):
        raise ValueError('one exact bank monomer and allocation tag required')
    job = compute_tag.rsplit('-', 1)[-1]
    if require_job and os.environ.get('SLURM_JOB_ID') != job:
        raise ValueError('dispatch intent must belong to the actual science allocation')
    for sha in (case_sha256, eligibility_sha256):
        _sha(sha)
    if dataset_id is None:
        if any(value is not None for value in (checkpoint_sha256, snapshot_sha256, context_sha256)):
            raise ValueError('bootstrap intent has no prior dataset/checkpoint')
        key = ('bootstrap', eligibility_sha256)
    else:
        if type(dataset_id) is not int or not 0 < dataset_id < 2**63 or dataset_id in (353, 354, 355, 356):
            raise ValueError('only the independently audited fresh bank may be an intent target')
        for sha in (checkpoint_sha256, snapshot_sha256, context_sha256):
            _sha(sha)
        # Canonical live-state identity, NOT raw checkpoint/context spelling.
        key = (str(dataset_id), snapshot_sha256)
    body = dict(schema='native-bank-dispatch-intent/1', status='RESERVED_BEFORE_CREATE' if dataset_id is None
        else 'RESERVED_BEFORE_ENTRY', dataset_id=dataset_id, checkpoint_sha256=checkpoint_sha256,
        snapshot_sha256=snapshot_sha256, bootstrap_context_sha256=context_sha256,
        case_sha256=case_sha256, eligibility_sha256=eligibility_sha256,
        monomer_id=monomer_id, compute_tag=compute_tag, job_id=job)
    return key, body


def reserve_intent(**fields):
    key, body = _body(**fields, require_job=True)
    _private_root(INTENTS)
    parent = INTENTS/key[0]
    _private_root(parent)
    target = parent/key[1]
    target.mkdir(mode=0o700, exist_ok=False)
    _directory(target, private=True)
    _sync(parent)
    raw = json.dumps(body, sort_keys=True, indent=2, allow_nan=False).encode()
    _write_once(target/'intent.json', raw)
    return dict(body, sha256=hashlib.sha256(raw).hexdigest())


def verify_intent(saved, **fields):
    key, body = _body(**fields, require_job=False)
    path = INTENTS/key[0]/key[1]/'intent.json'
    _directory(INTENTS.parent, private=False)
    for directory in (INTENTS, path.parent.parent, path.parent):
        _directory(directory, private=True)
    raw = _read_regular(path, maximum=16*1024)
    expected = dict(body, sha256=hashlib.sha256(raw).hexdigest())
    if json.loads(raw) != body or saved != expected:
        raise ValueError('bank dispatch intent bytes or predecessor changed')
    return expected['sha256']


def store_receipts(receipts):
    if (type(receipts) is not dict or set(receipts) != PROOF_ROLES
            or any(type(raw) is not bytes or len(raw) > 32*1024**2 for raw in receipts.values())):
        raise ValueError('exact bounded ordinary audit receipt bytes required')
    _private_root(PROOFS)
    result = {}
    for role, raw in receipts.items():
        sha = hashlib.sha256(raw).hexdigest()
        path = PROOFS/(sha+'.json')
        if path.is_symlink():
            raise ValueError('proof aliases are forbidden')
        try:
            _write_once(path, raw)
        except FileExistsError:
            if _read_regular(path, maximum=32*1024**2, durable=True) != raw:
                raise ValueError('existing content-addressed bank proof differs')
            # An interrupted writer may have left identical bytes before
            # syncing the file/link. Reuse must finish both durability steps.
            _sync(PROOFS)
        result[role] = sha
    return result


def read_proofs(required):
    if type(required) is not set or len(required) > 4*131:
        raise ValueError('bounded exact content-addressed bank proof inventory required')
    for sha in required:
        _sha(sha)
    if required:
        _directory(PROOFS.parent, private=False)
        _directory(PROOFS, private=True)
    paths = {sha: PROOFS/(sha+'.json') for sha in required}
    result, used = {}, 0
    for sha, path in sorted(paths.items()):
        raw = _read_regular(path, maximum=min(32*1024**2, MAX_PROOF_BYTES-used))
        used += len(raw)
        result[sha] = raw
    if any(hashlib.sha256(raw).hexdigest() != sha for sha, raw in result.items()):
        raise ValueError('stored bank audit proof bytes changed')
    return result
