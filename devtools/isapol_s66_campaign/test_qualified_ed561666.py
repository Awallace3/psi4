"""Cross-producer and synthetic-shape checks; not installed/molecular tests."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def load(filename):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location('_pins_test_'+path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


new = load('qualified_ed561666.py')
old = load('qualified_cb63305.py')


def synthetic(monkeypatch, *, change=None, profile=None):
    target = new if profile is None else profile
    package = '/synthetic/cap32/psi4'
    identity = dict(version=target.VERSION, core=package+'/core.so', package=package+'/__init__.py',
        implementation=dict(sha256=dict(target.IMPLEMENTATION),
            python_files={'procrouting/isapol_pfit_plan.py': target.PFIT_SHA256}))
    manifest = dict(schema='fresh-native-build-identity/1', commit=target.SCIENTIFIC_COMMIT,
        tree=target.SCIENTIFIC_TREE, clean_source=True, fresh_build=True,
        installed_core=identity['core'], installed_package=package,
        installed_files={'driver/procrouting/isapol_pfit_plan.py': target.PFIT_SHA256})
    count = target.TEST_SUMMARY['tests']
    report = dict(status='passed', runtime_unchanged=True, build_commit=target.SCIENTIFIC_COMMIT,
        build_tree=target.SCIENTIFIC_TREE, harness_commit=target.TEST_COMMIT,
        test_source_commit=target.TEST_COMMIT, expected_test_count=count,
        test_summary=dict(target.TEST_SUMMARY), test_only_revision_changes=[])
    if target.PROFILE_ID == 'grid400':
        report['installed_suite'] = 'grid400'
    data = dict(build_identity=manifest, imported_identity=identity, installed_qualification=report,
        junit=(f'<testsuites><testsuite tests="{count}" failures="0" errors="0" skipped="0"/>'
               '</testsuites>').encode())
    if change:
        role, values = change
        if type(values) is bytes:
            data[role] = values
        else:
            data[role].update(copy.deepcopy(values))
    blobs = {key: value if type(value) is bytes else json.dumps(value, sort_keys=True).encode()
             for key, value in data.items()}
    for key, role in (('build_identity_sha256', 'build_identity'),
                     ('imported_identity_sha256', 'imported_identity'), ('junit_sha256', 'junit')):
        report[key] = hashlib.sha256(blobs[role]).hexdigest()
    blobs['installed_qualification'] = json.dumps(report, sort_keys=True).encode()
    monkeypatch.setattr(target, 'RECEIPT_SHA256',
                        {key: hashlib.sha256(raw).hexdigest() for key, raw in blobs.items()})
    return blobs


def test_measured_pin_sets_and_test_counts_are_distinct():
    assert new.SCIENTIFIC_COMMIT == 'ed561666fcac212fffa774429fc3bd8f8cfc903d'
    assert old.SCIENTIFIC_COMMIT == 'cb63305f4115526417a830829a0a30c36b85b4e8'
    assert new.TEST_SUMMARY['tests'] == 459 and old.TEST_SUMMARY['tests'] == 430
    assert new.IMPLEMENTATION != old.IMPLEMENTATION
    assert all(new.RECEIPT_SHA256[key] != old.RECEIPT_SHA256[key] for key in new.RECEIPT_SHA256)
    assert new.PFIT_SHA256 == old.PFIT_SHA256  # This source did not change.


def test_new_public_wrapper_only_verifies_its_installed_receipts(monkeypatch):
    result = new.verify_receipts(synthetic(monkeypatch))
    assert result['schema'] == 'native-ed561666-qualified-producer-pins/1'
    assert result['status'] == 'VERIFIED_INSTALLED_PRODUCER_RECEIPTS_ONLY'
    assert result['scientific_commit'] == new.SCIENTIFIC_COMMIT
    assert 'no current-runtime' in result['qualification']
    assert 'models' not in result and 'descriptor_admission' not in result


@pytest.mark.parametrize('change', [
    ('build_identity', dict(commit=old.SCIENTIFIC_COMMIT)),
    ('build_identity', dict(fresh_build=False)),
    ('installed_qualification', dict(expected_test_count=430)),
    ('installed_qualification', dict(test_summary=old.TEST_SUMMARY)),
    ('installed_qualification', dict(runtime_unchanged=False)),
    ('installed_qualification', dict(harness_commit=old.TEST_COMMIT)),
    ('installed_qualification', dict(test_only_revision_changes=['undeclared.py'])),
    ('imported_identity', dict(version=old.VERSION)),
    ('imported_identity', dict(implementation=dict(sha256=old.IMPLEMENTATION))),
    ('junit', b'<testsuites><testsuite tests="458" failures="0" errors="0" skipped="0"/></testsuites>'),
])
def test_cross_wired_or_incomplete_synthetic_shapes_are_rejected(monkeypatch, change):
    with pytest.raises(ValueError):
        new.verify_receipts(synthetic(monkeypatch, change=change))


def test_receipt_bytes_roles_hashes_and_public_override_refused(monkeypatch):
    blobs = synthetic(monkeypatch)
    for changed in (blobs | {'junit': blobs['junit']+b' '},
                    blobs | {'junit': bytearray(blobs['junit'])},
                    {key: value for key, value in blobs.items() if key != 'build_identity'},
                    blobs | {'caller_profile': b'not permitted'}):
        with pytest.raises(ValueError):
            new.verify_receipts(changed)
    with pytest.raises(TypeError):
        new.verify_receipts(blobs, profile='cb63305')
    with pytest.raises(ValueError):
        old.verify_receipts(blobs)
