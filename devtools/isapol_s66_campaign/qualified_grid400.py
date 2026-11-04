"""Measured grid400 installed receipts, not descriptor/model admission.

Fresh build15078722 and installed qualification15080503 supplied these
identities. All461 cases passed unskipped, including source/installed dense
forwarding. Historical producers and receipts retain their own identities.
"""
import importlib.util
from pathlib import Path

SCIENTIFIC_COMMIT = '1aaa9e0a306373b42a3decc25c7dfa106dbc8ea2'
SCIENTIFIC_TREE = '8f4705983f81bc5acf42625192ece9b0379c8209'
TEST_COMMIT = SCIENTIFIC_COMMIT
VERSION = '1.12a1.dev180'
IMPLEMENTATION = {
    'core': 'ef69cddb2dd7b675688a733e4c5dd68b78fc78ac35b77b363114d638ac9dae44',
    'isapol_python': 'e1476e2d0b8e395c40b48219ca8bc8def9e69035cab3ab970f6b88940bebc74b',
}
PFIT_SHA256 = 'd77c1bab49acb3c01185f8e1a974c1145d6ce11e101cb9c62b2b274d5c297841'
RECEIPT_SHA256 = {
    'build_identity': 'd2b4049c2e5818e462ad3c6ac36f93d6a9623a6a205c701b885887ec2a9f3bfd',
    'installed_qualification': 'a9f7179ec4721a79b462a9516639d474f9ca952b7adbe89b4069438b4ed8e8dc',
    'imported_identity': 'b8f5f5afcc3ffb372cc9524406159327c9bdb6894e8bb5deabf275781d264e78',
    'junit': '06ca24994e7fa11a0c5b4ab9d463068872f4b02b3554e5ab94965d84a62fd188',
}
TEST_ONLY_CHANGES = set()
TEST_SUMMARY = dict(tests=461, failures=0, errors=0, skipped=0)
PROFILE_ID = 'grid400'
_PIN_NAMES = (
    'SCIENTIFIC_COMMIT', 'SCIENTIFIC_TREE', 'TEST_COMMIT', 'VERSION',
    'IMPLEMENTATION', 'PFIT_SHA256', 'RECEIPT_SHA256', 'TEST_ONLY_CHANGES',
    'TEST_SUMMARY', 'PROFILE_ID')


def verify_receipts(blobs):
    """Verify only the four measured installed receipts; no caller overrides."""
    path = Path(__file__).with_name('qualified_cb63305.py')
    spec = importlib.util.spec_from_file_location('_grid400_receipt_engine', path)
    shared = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec'), shared.__dict__)
    return shared._verify_receipts(blobs, {key: globals()[key] for key in _PIN_NAMES})
