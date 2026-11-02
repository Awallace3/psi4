"""Closed measured cap32 installed receipts; never descriptor/model acceptance.

All identities came from fresh build14990587 and installed tests14991005.
The old cb63305 profile/records retain their exact original identities.
"""
import importlib.util
from pathlib import Path

SCIENTIFIC_COMMIT = 'ed561666fcac212fffa774429fc3bd8f8cfc903d'
SCIENTIFIC_TREE = '5f4c9a1603d0761503d124fd3bdaba4b9b383192'
TEST_COMMIT = SCIENTIFIC_COMMIT
VERSION = '1.12a1.dev165'
IMPLEMENTATION = {
    'core': 'de2de4f3d237035f1f05906176fd29c58a86aab002e060840de582a870e21ef4',
    'isapol_python': 'b95733bf489152bc4a08b459b9ffda1056e42a87ff8954663e31abf1b00cbcb0',
}
PFIT_SHA256 = 'd77c1bab49acb3c01185f8e1a974c1145d6ce11e101cb9c62b2b274d5c297841'
RECEIPT_SHA256 = {
    'build_identity': '00cecb3108ce8b7c6c54edc5828aedf609f9a73e44715dd634c447244a91a7a2',
    'installed_qualification': '5d69a1fa17e519da9ec02e45fe0a65e8f6cbdfd806573f2a73ba24e7f38b357f',
    'imported_identity': 'a7443b7b18448a005ebfd1d860ee2a61c6cdc7c1fdc11b6154dd903c4c86f9ce',
    'junit': 'a517dcc0eb743ebd9502a8232fdad43828934d651eb7845a86d2b5561c70450e',
}
TEST_ONLY_CHANGES = set()
TEST_SUMMARY = dict(tests=459, failures=0, errors=0, skipped=0)
PROFILE_ID = 'ed561666'
_PIN_NAMES = (
    'SCIENTIFIC_COMMIT', 'SCIENTIFIC_TREE', 'TEST_COMMIT', 'VERSION',
    'IMPLEMENTATION', 'PFIT_SHA256', 'RECEIPT_SHA256', 'TEST_ONLY_CHANGES',
    'TEST_SUMMARY', 'PROFILE_ID')


def verify_receipts(blobs):
    """Accept only this exact installed producer's four immutable receipts."""
    path = Path(__file__).with_name('qualified_cb63305.py')
    spec = importlib.util.spec_from_file_location('_drho32_receipt_engine', path)
    shared = importlib.util.module_from_spec(spec)
    # Execute the selected operational source bytes, never a cached .pyc.
    exec(compile(path.read_bytes(), str(path), 'exec'), shared.__dict__)
    return shared._verify_receipts(blobs, {key: globals()[key] for key in _PIN_NAMES})
