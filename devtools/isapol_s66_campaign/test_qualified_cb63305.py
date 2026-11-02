"""Synthetic receipt-shape adversaries; never installed/molecular acceptance."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'qualified_cb63305', Path(__file__).with_name('qualified_cb63305.py'))
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def encode(value):
    return json.dumps(value, sort_keys=True).encode()


def fixture():
    package = '/synthetic/private/psi4'
    identity = dict(version=profile.VERSION, core=package+'/core.so',
        package=package+'/__init__.py', implementation=dict(
            sha256=dict(profile.IMPLEMENTATION),
            python_files={'procrouting/isapol_pfit_plan.py': profile.PFIT_SHA256}))
    manifest = dict(schema='fresh-native-build-identity/1', commit=profile.SCIENTIFIC_COMMIT,
        tree=profile.SCIENTIFIC_TREE, clean_source=True, fresh_build=True,
        installed_core=identity['core'], installed_package=package,
        installed_files={'driver/procrouting/isapol_pfit_plan.py': profile.PFIT_SHA256})
    qualification = dict(status='passed', runtime_unchanged=True,
        build_commit=profile.SCIENTIFIC_COMMIT, build_tree=profile.SCIENTIFIC_TREE,
        harness_commit=profile.TEST_COMMIT, test_source_commit=profile.TEST_COMMIT,
        expected_test_count=430, test_summary=dict(profile.TEST_SUMMARY),
        test_only_revision_changes=sorted(profile.TEST_ONLY_CHANGES))
    return dict(build_identity=manifest, imported_identity=identity,
                installed_qualification=qualification,
                junit=b'<testsuites><testsuite tests="430" failures="0" errors="0" skipped="0"/></testsuites>')


def bind_synthetic(monkeypatch, data):
    """Mock only immutable byte pins so shape guards can be tested separately."""
    blobs = {k: v if type(v) is bytes else encode(v) for k, v in data.items()}
    report = copy.deepcopy(data['installed_qualification'])
    for key, role in (('build_identity_sha256', 'build_identity'),
                      ('imported_identity_sha256', 'imported_identity'), ('junit_sha256', 'junit')):
        report[key] = hashlib.sha256(blobs[role]).hexdigest()
    blobs['installed_qualification'] = encode(report)
    monkeypatch.setattr(profile, 'RECEIPT_SHA256',
                        {role: hashlib.sha256(raw).hexdigest() for role, raw in blobs.items()})
    return blobs


def test_profile_constants_identify_actual_new_receipts_not_legacy_or_latest():
    assert profile.SCIENTIFIC_COMMIT == 'cb63305f4115526417a830829a0a30c36b85b4e8'
    assert profile.TEST_COMMIT == '8eee149cb8cb246ce6335d18d0f07b04498ff40e'
    assert profile.RECEIPT_SHA256['installed_qualification'] == (
        '6257c034070f01c8908c9eb898d5f4a5a4a67d4395f398b3f8a97e2a733b6b11')


def test_well_formed_receipts_only_verify_prior_receipts_not_current_execution(monkeypatch):
    result = profile.verify_receipts(bind_synthetic(monkeypatch, fixture()))
    assert result['status'] == 'VERIFIED_INSTALLED_PRODUCER_RECEIPTS_ONLY'
    assert 'no current-runtime' in result['qualification']


def test_receipt_roles_bytes_and_original_hashes_are_not_optional(monkeypatch):
    blobs = bind_synthetic(monkeypatch, fixture())
    for changed in ({k: v for k, v in blobs.items() if k != 'junit'},
                    blobs | {'extra': b'not permitted'},
                    blobs | {'junit': bytearray(blobs['junit'])},
                    blobs | {'junit': blobs['junit']+b' '}):
        with pytest.raises(ValueError):
            profile.verify_receipts(changed)


@pytest.mark.parametrize('change', [
    dict(status='started'), dict(runtime_unchanged=False),
    dict(build_commit='a'*40), dict(build_tree='b'*40),
    dict(test_source_commit='c'*40), dict(harness_commit='d'*40),
    dict(expected_test_count=197), dict(test_summary=dict(tests=430, failures=1, errors=0, skipped=0)),
    dict(test_summary=dict(tests=430, failures=0, errors=0, skipped=1)),
    dict(test_only_revision_changes=sorted(profile.TEST_ONLY_CHANGES)*2),
])
def test_cross_wired_or_incomplete_qualification_is_refused(monkeypatch, change):
    data = fixture()
    data['installed_qualification'].update(change)
    with pytest.raises(ValueError, match='source/test/runtime'):
        profile.verify_receipts(bind_synthetic(monkeypatch, data))


@pytest.mark.parametrize('target,change', [
    ('build_identity', dict(clean_source=False)),
    ('build_identity', dict(fresh_build=False)),
    ('build_identity', dict(commit='a'*40)),
    ('imported_identity', dict(version='foreign')),
    ('imported_identity', dict(package='/foreign/__init__.py')),
    ('imported_identity', dict(implementation=dict(sha256=dict(core='0'*64, isapol_python='1'*64)))),
])
def test_other_build_or_implementation_is_not_the_qualified_producer(monkeypatch, target, change):
    data = fixture()
    data[target].update(change)
    with pytest.raises(ValueError):
        profile.verify_receipts(bind_synthetic(monkeypatch, data))


def test_junit_contents_cannot_be_replaced_by_successful_report_totals(monkeypatch):
    data = fixture()
    data['junit'] = b'<testsuites><testsuite tests="429" failures="0" errors="0" skipped="0"/></testsuites>'
    with pytest.raises(ValueError, match='exact430'):
        profile.verify_receipts(bind_synthetic(monkeypatch, data))
