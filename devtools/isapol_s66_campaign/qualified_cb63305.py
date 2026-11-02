"""Closed receipt profile for one installed producer; no execution or admission.

Inputs are receipt bytes, never paths, implementation overrides, credentials or
scientific state. Live runtime, descriptor/resource and molecular checks remain
separate. The previous acf54 producer and its receipts are not reinterpreted.
"""
import hashlib
import json
import xml.etree.ElementTree as ET

SCIENTIFIC_COMMIT = 'cb63305f4115526417a830829a0a30c36b85b4e8'
SCIENTIFIC_TREE = 'a9fc4b5d4d0227825ef57f6aec95d32e574eff74'
TEST_COMMIT = '8eee149cb8cb246ce6335d18d0f07b04498ff40e'
VERSION = '1.12a1.dev152'
IMPLEMENTATION = {
    'core': '84e709f408eece45a226065ce01ebb7cabd46dbef87be0d6c1459f8352485bf0',
    'isapol_python': 'ae3feec24ae43a826af02f8e7d6cec5603e4283fe567ce4970f539d1af7c69aa',
}
PFIT_SHA256 = 'd77c1bab49acb3c01185f8e1a974c1145d6ce11e101cb9c62b2b274d5c297841'
RECEIPT_SHA256 = {
    'build_identity': '0497ac89d6c58d3056a07c00847099606812203e446f30e95979f359df6cd511',
    'installed_qualification': '6257c034070f01c8908c9eb898d5f4a5a4a67d4395f398b3f8a97e2a733b6b11',
    'imported_identity': 'da51423f525f099ea7c2381294b80ec21b71a9e14022b7fc075ce773f3d07650',
    'junit': 'aab8ba03ab8c5b5f058fbf15232acf8bc6bc81c245683ea8ee466a14e2ae4230',
}
TEST_ONLY_CHANGES = {
    'devtools/isapol_s66_campaign/phoenix_installed_checks.py',
    'devtools/isapol_s66_campaign/test_installed_checks.py',
    'tests/pytests/test_isapol_molecular_declaration.py',
}
TEST_SUMMARY = dict(tests=430, failures=0, errors=0, skipped=0)


def verify_receipts(blobs):
    """Verify this exact producer's immutable installed-qualification receipts.

    A caller cannot select another profile, relax the case count or reuse an
    old receipt. This proves receipt identity only, not current runtime bytes.
    """
    if type(blobs) is not dict or set(blobs) != set(RECEIPT_SHA256):
        raise ValueError('exact cb63305 receipt roles required')
    observed = {}
    for role, expected in RECEIPT_SHA256.items():
        if type(blobs[role]) is not bytes:
            raise ValueError('immutable receipt bytes required')
        observed[role] = hashlib.sha256(blobs[role]).hexdigest()
        if observed[role] != expected:
            raise ValueError('qualified cb63305 receipt hash differs: '+role)
    manifest = json.loads(blobs['build_identity'])
    qualification = json.loads(blobs['installed_qualification'])
    identity = json.loads(blobs['imported_identity'])
    if (manifest.get('schema') != 'fresh-native-build-identity/1'
            or manifest.get('commit') != SCIENTIFIC_COMMIT
            or manifest.get('tree') != SCIENTIFIC_TREE
            or manifest.get('clean_source') is not True
            or manifest.get('fresh_build') is not True
            or qualification.get('status') != 'passed'
            or qualification.get('runtime_unchanged') is not True
            or qualification.get('build_commit') != SCIENTIFIC_COMMIT
            or qualification.get('build_tree') != SCIENTIFIC_TREE
            or qualification.get('harness_commit') != TEST_COMMIT
            or qualification.get('test_source_commit') != TEST_COMMIT
            or qualification.get('expected_test_count') != 430
            or qualification.get('test_summary') != TEST_SUMMARY
            or qualification.get('build_identity_sha256') != observed['build_identity']
            or qualification.get('imported_identity_sha256') != observed['imported_identity']
            or qualification.get('junit_sha256') != observed['junit']
            or set(qualification.get('test_only_revision_changes', ())) != TEST_ONLY_CHANGES
            or len(qualification['test_only_revision_changes']) != len(TEST_ONLY_CHANGES)):
        raise ValueError('qualified cb63305 source/test/runtime receipt differs')
    if (identity.get('version') != VERSION
            or identity.get('implementation', {}).get('sha256') != IMPLEMENTATION
            or identity.get('core') != manifest.get('installed_core')
            or identity.get('package') != manifest['installed_package']+'/__init__.py'
            or identity['implementation'].get('python_files', {}).get(
                'procrouting/isapol_pfit_plan.py') != PFIT_SHA256
            or manifest.get('installed_files', {}).get(
                'driver/procrouting/isapol_pfit_plan.py') != PFIT_SHA256):
        raise ValueError('qualified cb63305 installed implementation differs')
    root = ET.fromstring(blobs['junit'])
    suites = [root] if root.tag == 'testsuite' else root.findall('testsuite')
    summary = {key: sum(int(suite.get(key, '0')) for suite in suites) for key in TEST_SUMMARY}
    if summary != TEST_SUMMARY:
        raise ValueError('exact430 successful unskipped installed cases required')
    return dict(
        schema='native-cb63305-qualified-producer-pins/1',
        status='VERIFIED_INSTALLED_PRODUCER_RECEIPTS_ONLY',
        scientific_commit=SCIENTIFIC_COMMIT, scientific_tree=SCIENTIFIC_TREE,
        installed_test_commit=TEST_COMMIT, implementation_sha256=dict(IMPLEMENTATION),
        receipt_sha256=observed, pfit_plan_sha256=PFIT_SHA256, version=VERSION,
        qualification='Pinned prior installed qualification only; no current-runtime, '
                      'descriptor/resource, molecular/QCArchive, accuracy or parity acceptance.')
