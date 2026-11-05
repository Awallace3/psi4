"""Temporary metadata only: immutable intents and exact proof bytes."""
import hashlib
import importlib.util
import os
from pathlib import Path
import stat

import pytest


def fixture(monkeypatch, tmp_path):
    path = Path(__file__).with_name('bank_store.py')
    spec = importlib.util.spec_from_file_location('_bank_store_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'INTENTS', tmp_path/'intents')
    monkeypatch.setattr(module, 'PROOFS', tmp_path/'proofs')
    monkeypatch.setenv('SLURM_JOB_ID', '502')
    fields = dict(dataset_id=9999, checkpoint_sha256='a'*64, snapshot_sha256='b'*64,
        context_sha256='c'*64, case_sha256='d'*64, eligibility_sha256='e'*64,
        monomer_id='monomer-0002', compute_tag='ccasp-native-grid400-bank-monomer-0002-502')
    return module, fields


def test_intent_is_durable_exact_and_verifiable_from_a_distinct_audit_job(monkeypatch, tmp_path):
    store, fields = fixture(monkeypatch, tmp_path)
    saved = store.reserve_intent(**fields)
    monkeypatch.setenv('SLURM_JOB_ID', '999')
    assert store.verify_intent(saved, **fields) == saved['sha256']
    with pytest.raises(ValueError, match='actual science allocation'):
        store.reserve_intent(**fields)


@pytest.mark.parametrize('alias', ['checkpoint_sha256', 'context_sha256', 'case_sha256'])
def test_alternate_byte_commitments_cannot_reserve_same_canonical_state(monkeypatch, tmp_path, alias):
    store, fields = fixture(monkeypatch, tmp_path)
    store.reserve_intent(**fields)
    fields[alias] = 'f'*64
    with pytest.raises(FileExistsError):
        store.reserve_intent(**fields)


def test_partial_intent_is_never_repaired_or_cleared(monkeypatch, tmp_path):
    store, fields = fixture(monkeypatch, tmp_path)
    target = store.INTENTS/'9999'/fields['snapshot_sha256']
    store.INTENTS.mkdir(mode=0o700)
    target.parent.mkdir(mode=0o700)
    target.mkdir(mode=0o700)
    with pytest.raises(FileExistsError):
        store.reserve_intent(**fields)
    assert not (target/'intent.json').exists()


def test_bootstrap_has_a_separate_single_creation_intent(monkeypatch, tmp_path):
    store, fields = fixture(monkeypatch, tmp_path)
    fields.update(dataset_id=None, checkpoint_sha256=None, snapshot_sha256=None, context_sha256=None)
    saved = store.reserve_intent(**fields)
    assert saved['status'] == 'RESERVED_BEFORE_CREATE'
    assert store.verify_intent(saved, **fields) == saved['sha256']
    with pytest.raises(FileExistsError):
        store.reserve_intent(**fields)


@pytest.mark.parametrize('identifier', [353, 354, 355, 356, True, -1])
def test_old_or_invalid_dataset_cannot_reserve_an_intent(monkeypatch, tmp_path, identifier):
    store, fields = fixture(monkeypatch, tmp_path)
    fields['dataset_id'] = identifier
    with pytest.raises(ValueError):
        store.reserve_intent(**fields)
    assert not store.INTENTS.exists()


def test_changed_intent_bytes_or_commitments_are_refused(monkeypatch, tmp_path):
    store, fields = fixture(monkeypatch, tmp_path)
    saved = store.reserve_intent(**fields)
    with pytest.raises(ValueError):
        store.verify_intent(dict(saved, sha256='f'*64), **fields)
    path = store.INTENTS/'9999'/fields['snapshot_sha256']/'intent.json'
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):
        store.verify_intent(saved, **fields)


def test_proof_storage_reuses_only_identical_content_and_reads_exact_requested_hashes(monkeypatch, tmp_path):
    store, _ = fixture(monkeypatch, tmp_path)
    receipts = {role: ('synthetic-'+role).encode() for role in store.PROOF_ROLES}
    hashes = store.store_receipts(receipts)
    assert hashes == store.store_receipts(receipts)
    assert store.read_proofs(set(hashes.values())) == {sha: receipts[role] for role, sha in hashes.items()}
    with pytest.raises(ValueError):
        store.read_proofs({'../escape'})
    one = next(iter(hashes.values()))
    (store.PROOFS/(one+'.json')).write_bytes(b'changed')
    with pytest.raises(ValueError):
        store.read_proofs({one})
    with pytest.raises(ValueError):
        store.store_receipts(receipts)


def test_symlink_intent_or_proof_alias_is_not_followed(monkeypatch, tmp_path):
    store, fields = fixture(monkeypatch, tmp_path)
    actual = tmp_path/'other'
    actual.mkdir()
    store.INTENTS.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError):
        store.reserve_intent(**fields)
    store.PROOFS.mkdir(mode=0o700)
    raw = b'synthetic-proof'
    sha = hashlib.sha256(raw).hexdigest()
    target = tmp_path/'plain.json'
    target.write_bytes(raw)
    (store.PROOFS/(sha+'.json')).symlink_to(target)
    with pytest.raises(ValueError):
        store.read_proofs({sha})


def test_all_new_directory_links_are_synced_before_reservation_returns(monkeypatch, tmp_path):
    store, fields = fixture(monkeypatch, tmp_path)
    events = []
    sync, fsync = store._sync, store.os.fsync
    def synced(path):
        events.append(('directory', path))
        sync(path)
    def file_synced(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            events.append(('file', None))
        fsync(fd)
    monkeypatch.setattr(store, '_sync', synced)
    monkeypatch.setattr(store.os, 'fsync', file_synced)
    store.reserve_intent(**fields)
    target = store.INTENTS/'9999'/fields['snapshot_sha256']
    assert events == [('directory', tmp_path), ('directory', store.INTENTS),
        ('directory', store.INTENTS/'9999'), ('file', None), ('directory', target)]
    assert stat.S_IMODE((target/'intent.json').stat().st_mode) == 0o600


@pytest.mark.parametrize('failure', [1, 2, 3, 4])
def test_directory_sync_failure_is_not_acknowledged_or_repaired(monkeypatch, tmp_path, failure):
    store, fields = fixture(monkeypatch, tmp_path)
    sync, count = store._sync, []
    def fails(path):
        count.append(path)
        if len(count) == failure:
            raise OSError('synthetic directory sync failure')
        sync(path)
    monkeypatch.setattr(store, '_sync', fails)
    with pytest.raises(OSError):
        store.reserve_intent(**fields)
    if failure >= 3:
        target = store.INTENTS/'9999'/fields['snapshot_sha256']
        assert target.is_dir()
        monkeypatch.setattr(store, '_sync', sync)
        with pytest.raises(FileExistsError):
            store.reserve_intent(**fields)


@pytest.mark.parametrize('where', ['root', 'dataset', 'proofs', 'container'])
def test_existing_permissive_directories_are_rejected_not_chmod_repaired(monkeypatch, tmp_path, where):
    store, fields = fixture(monkeypatch, tmp_path)
    if where == 'container':
        tmp_path.chmod(0o770)
        target = tmp_path
    elif where == 'proofs':
        store.PROOFS.mkdir(mode=0o755)
        target = store.PROOFS
    else:
        store.INTENTS.mkdir(mode=0o755 if where == 'root' else 0o700)
        target = store.INTENTS
        if where == 'dataset':
            target = store.INTENTS/'9999'
            target.mkdir(mode=0o755)
    before = stat.S_IMODE(target.stat().st_mode)
    try:
        with pytest.raises(ValueError, match='private'):
            if where == 'proofs':
                store.store_receipts({role: b'unit' for role in store.PROOF_ROLES})
            else:
                store.reserve_intent(**fields)
        assert stat.S_IMODE(target.stat().st_mode) == before
    finally:
        if where == 'container':
            tmp_path.chmod(0o700)


def test_wrong_directory_owner_is_rejected(monkeypatch, tmp_path):
    store, fields = fixture(monkeypatch, tmp_path)
    uid = os.geteuid()
    monkeypatch.setattr(store.os, 'geteuid', lambda: uid+1)
    with pytest.raises(ValueError, match='owned private'):
        store.reserve_intent(**fields)
    assert not store.INTENTS.exists()


def test_fifo_proof_is_rejected_without_reading_or_blocking(monkeypatch, tmp_path):
    store, _ = fixture(monkeypatch, tmp_path)
    store.PROOFS.mkdir(mode=0o700)
    sha = 'a'*64
    os.mkfifo(store.PROOFS/(sha+'.json'), 0o600)
    monkeypatch.setattr(store.os, 'read', lambda *args: pytest.fail('nonregular proof was read'))
    with pytest.raises(ValueError, match='regular'):
        store.read_proofs({sha})


def test_oversized_existing_proof_is_never_read_or_overwritten(monkeypatch, tmp_path):
    store, _ = fixture(monkeypatch, tmp_path)
    receipts = {role: ('unit-'+role).encode() for role in store.PROOF_ROLES}
    hashes = store.store_receipts(receipts)
    path = store.PROOFS/(hashes['audit']+'.json')
    with path.open('r+b') as stream:
        stream.truncate(32*1024**2+1)
    read, inode = store.os.read, path.stat().st_ino
    def checked(fd, count):
        if os.fstat(fd).st_ino == inode:
            pytest.fail('oversized proof was read')
        return read(fd, count)
    monkeypatch.setattr(store.os, 'read', checked)
    with pytest.raises(ValueError, match='bounded regular'):
        store.store_receipts(receipts)
    assert path.stat().st_size == 32*1024**2+1


def test_opened_file_growth_is_bounded_and_rejected(monkeypatch, tmp_path):
    store, _ = fixture(monkeypatch, tmp_path)
    path = tmp_path/'growing.json'
    path.write_bytes(b'1')
    read, calls = store.os.read, []
    def grows(fd, count):
        calls.append(count)
        if len(calls) == 1:
            with path.open('ab') as stream:
                stream.write(b'23456789')
        return read(fd, count)
    monkeypatch.setattr(store.os, 'read', grows)
    with pytest.raises(ValueError, match='read bound'):
        store._read_regular(path, maximum=4)
    assert calls == [5]


def test_actual_aggregate_proof_reads_cannot_exceed_total_budget(monkeypatch, tmp_path):
    store, _ = fixture(monkeypatch, tmp_path)
    receipts = {role: ('unit-'+role).encode() for role in store.PROOF_ROLES}
    hashes = store.store_receipts(receipts)
    monkeypatch.setattr(store, 'MAX_PROOF_BYTES', 2)
    with pytest.raises(ValueError, match='bounded regular'):
        store.read_proofs(set(hashes.values()))


def test_identical_unsynced_proof_is_file_and_link_synced_before_reuse(monkeypatch, tmp_path):
    store, _ = fixture(monkeypatch, tmp_path)
    store.PROOFS.mkdir(mode=0o700)
    receipt = b'synthetic interrupted writer bytes'
    sha = hashlib.sha256(receipt).hexdigest()
    # Simulate the bytes left before the original writer's file/link fsync.
    (store.PROOFS/(sha+'.json')).write_bytes(receipt)
    events, fsync = [], store.os.fsync
    def synced(fd):
        events.append('file' if stat.S_ISREG(os.fstat(fd).st_mode) else 'directory')
        fsync(fd)
    monkeypatch.setattr(store.os, 'fsync', synced)
    assert store.store_receipts({role: receipt for role in store.PROOF_ROLES}) == {
        role: sha for role in store.PROOF_ROLES}
    assert events == ['directory']+['file', 'directory']*4


def test_failed_sync_of_reused_proof_is_never_acknowledged(monkeypatch, tmp_path):
    store, _ = fixture(monkeypatch, tmp_path)
    receipts = {role: b'synthetic proof' for role in store.PROOF_ROLES}
    store.store_receipts(receipts)
    fsync = store.os.fsync
    def fail(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError('synthetic reused proof fsync failure')
        fsync(fd)
    monkeypatch.setattr(store.os, 'fsync', fail)
    with pytest.raises(OSError, match='reused proof'):
        store.store_receipts(receipts)


def test_storage_preflight_creates_only_private_empty_stores_not_intents(monkeypatch, tmp_path):
    store, _ = fixture(monkeypatch, tmp_path)
    store.prepare_storage()
    store.prepare_storage()
    for path in (store.INTENTS, store.PROOFS):
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
        assert not list(path.iterdir())
    store.PROOFS.chmod(0o755)
    with pytest.raises(ValueError, match='private'):
        store.prepare_storage()
    assert stat.S_IMODE(store.PROOFS.stat().st_mode) == 0o755
