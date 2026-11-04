"""Measured pin constants and synthetic receipt-shape guards, no quantum work."""
import importlib.util
from pathlib import Path

import pytest


def load(filename):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location('_grid400_test_'+path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qualified = load('qualified_grid400.py')
fixtures = load('test_qualified_ed561666.py')
legacy = fixtures.new


def test_candidate_pins_are_actual_distinct_measured461case_installation():
    assert qualified.SCIENTIFIC_COMMIT == '1aaa9e0a306373b42a3decc25c7dfa106dbc8ea2'
    assert qualified.SCIENTIFIC_TREE == '8f4705983f81bc5acf42625192ece9b0379c8209'
    assert qualified.VERSION == '1.12a1.dev180'
    assert qualified.TEST_SUMMARY == dict(tests=461, failures=0, errors=0, skipped=0)
    assert qualified.IMPLEMENTATION != legacy.IMPLEMENTATION
    assert all(qualified.RECEIPT_SHA256[key] != legacy.RECEIPT_SHA256[key]
               for key in qualified.RECEIPT_SHA256)
    assert qualified.PFIT_SHA256 == legacy.PFIT_SHA256
    assert qualified.RECEIPT_SHA256['build_identity'] == (
        'd2b4049c2e5818e462ad3c6ac36f93d6a9623a6a205c701b885887ec2a9f3bfd')


def test_synthetic_candidate_wrapper_has_no_descriptor_or_model_claim(monkeypatch):
    blobs = fixtures.synthetic(monkeypatch, profile=qualified)
    result = qualified.verify_receipts(blobs)
    assert result['schema'] == 'native-grid400-qualified-producer-pins/1'
    assert result['status'] == 'VERIFIED_INSTALLED_PRODUCER_RECEIPTS_ONLY'
    assert result['implementation_sha256'] == qualified.IMPLEMENTATION
    assert 'no current-runtime' in result['qualification']
    with pytest.raises(ValueError):
        legacy.verify_receipts(blobs)
    with pytest.raises(ValueError):
        qualified.verify_receipts(blobs | {'junit': blobs['junit']+b' '})
    with pytest.raises(TypeError):
        qualified.verify_receipts(blobs, profile='ed561666')


@pytest.mark.parametrize('change', [
    ('installed_qualification', dict(installed_suite='legacy')),
    ('installed_qualification', dict(expected_test_count=459)),
    ('installed_qualification', dict(test_summary=legacy.TEST_SUMMARY)),
    ('installed_qualification', dict(runtime_unchanged=False)),
    ('build_identity', dict(commit=legacy.SCIENTIFIC_COMMIT)),
    ('imported_identity', dict(version=legacy.VERSION)),
    ('imported_identity', dict(implementation=dict(sha256=legacy.IMPLEMENTATION))),
    ('junit', b'<testsuites><testsuite tests="461" failures="0" errors="0" skipped="1"/></testsuites>')])
def test_crosswired_or_relaxed_candidate_receipts_refuse(monkeypatch, change):
    blobs = fixtures.synthetic(monkeypatch, profile=qualified, change=change)
    with pytest.raises(ValueError):
        qualified.verify_receipts(blobs)
