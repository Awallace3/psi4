"""Synthetic transport-only tests: no real client, records, credentials or SCF."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

spec = importlib.util.spec_from_file_location('dataset_append', Path(__file__).with_name('dataset_append.py'))
append = importlib.util.module_from_spec(spec)
spec.loader.exec_module(append)


@pytest.fixture(autouse=True)
def models(monkeypatch):
    # Model construction is checked against the installed API in the allocation;
    # these fixtures exercise transport retry/singleton control flow only.
    for name in ('qcportal', 'qcportal.metadata_models', 'qcportal.singlepoint', 'qcportal.dataset_models'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules['qcportal.metadata_models'].InsertMetadata = type('InsertMetadata', (), {})
    sys.modules['qcportal.metadata_models'].InsertCountsMetadata = type('InsertCountsMetadata', (), {})
    sys.modules['qcportal.singlepoint'].SinglepointDatasetNewEntry = NS
    sys.modules['qcportal.dataset_models'].DatasetSubmitBody = NS
    # Actual HTTP behavior is separately mandatory in transport_tests.py under
    # the real version-pinned p4qca environment, not this fixture layer.
    monkeypatch.setattr(append, 'require_one_shot_client', lambda client: None)


class Client:
    def __init__(self, result=None, error=None):
        self.calls = []
        self.result = result or NS(success=True, n_inserted=1, n_existing=0)
        self.error = error

    def make_request(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.result


def test_only_two_exact_post_endpoints_no_retries_no_other_mutation():
    client = Client()
    append.insert_entry_once(client, dict(name='new', attributes={'monomer_id': 'monomer-0004'}))
    append.submit_record_once(client, entry_name='new', compute_tag='ccasp-native-s66-monomer-0004-123')
    assert [c[0][:2] for c in client.calls] == [
        ('post', 'api/v1/datasets/singlepoint/353/entries/bulkCreate'),
        ('post', 'api/v1/datasets/singlepoint/353/submit')]
    assert all(c[1]['allow_retries'] is False for c in client.calls)
    assert len(client.calls[0][1]['body']) == 1
    body = client.calls[1][1]['body']
    assert body.entry_names == ['new'] and body.specification_names == ['pbe0-atz-native-request2']
    assert body.find_existing is False and body.compute_tag == 'ccasp-native-s66-monomer-0004-123'


@pytest.mark.parametrize('stage', ['entry', 'record'])
@pytest.mark.parametrize('failure', ['transport', 'existing', 'multiple', 'rejected'])
def test_ambiguous_or_nonfresh_writes_stop_without_retry_or_rollback(stage, failure):
    result = NS(success=failure != 'rejected', n_inserted=2 if failure == 'multiple' else 1,
                n_existing=1 if failure == 'existing' else 0)
    client = Client(result=result, error=TimeoutError() if failure == 'transport' else None)
    with pytest.raises((RuntimeError, TimeoutError)):
        if stage == 'entry':
            append.insert_entry_once(client, dict(attributes={'monomer_id': 'monomer-0004'}))
        else:
            append.submit_record_once(client, entry_name='new', compute_tag='ccasp-native-s66-monomer-0004-123')
    assert len(client.calls) == 1


def test_no_other_monomer_or_wildcard_tag_can_write():
    client = Client()
    with pytest.raises(ValueError):
        append.insert_entry_once(client, dict(attributes={'monomer_id': 'monomer-0006'}))
    for tag in ('*', 'ccasp-native-s66-water-123', 'ccasp-native-s66-monomer-0006-123'):
        with pytest.raises(ValueError):
            append.submit_record_once(client, entry_name='new', compute_tag=tag)
    assert client.calls == []
