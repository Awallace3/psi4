"""Closed installed qomp producer. NOT model/cohort/production acceptance.

The build allocation FAILED after successful compilation. This DISTINCT
verification/collection/qualification chain never becomes a normal READY
build receipt. Runtime checks use a separate pinned finite view.
"""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

SCIENTIFIC_COMMIT = 'fb11b78d1f44658d3a9d62365b77f3d217f53129'
SCIENTIFIC_TREE = '3a1922c885498131b3b51b7214dcab429cac073e'
TEST_COMMIT = '311d17bf9177a972f3212c048e2f2177408b95d7'
TEST_TREE = '57ff961824c0da6d76cf48e4da63fc83979f4f91'
QUALIFYING_OPS = '970f6ad23efbe01d2a620c49e7ea4894423c2ef2'
QUALIFYING_OPS_TREE = '06c9718ed3109ffcc6899650b986b046eff4da2a'
VERSION = '1.12a1.dev212'
PROFILE_ID = 'qomp'
IMPLEMENTATION = dict(
    core='f4a1b302d56a1904ddd807460082c1f21b2d8b418539e7c808e617b1515e5469',
    isapol_python='331bfcedbdfd0cb66656c6de5bdb84297d8645ff21d4ad6675832e3592dbbf8b')
PFIT_SHA256 = 'd77c1bab49acb3c01185f8e1a974c1145d6ce11e101cb9c62b2b274d5c297841'
VIEW_SHA256 = '29d688df80fc3eb55a516042d86f4913100a8e845a3c8152acaf1a1eccf2ad6e'
INVENTORY_SHA256 = '46146a2e5b706d9ed657a1d29a55bd058d64fd2cd540db7889e1e71abcc3b9b5'
# The reviewed ops ancestry already differs from SCI in these two files.
# Permit their EXACT frozen970 bytes, never an arbitrary build-engine delta.
INHERITED_OPS = {
    'devtools/isapol_native_acceptance/phoenix_build.py':
        'c9b9dd75f01fc9933ee914dc92b4b267367c1f0e4117611fec4466b20a6ff2b6',
    'devtools/isapol_native_acceptance/qomp.sbatch':
        '17d7aa10afa02a262fb7e42e2176368cd2564e13f151b57fe87f8acf283e3a22'}
TEST_SUMMARY = dict(tests=556, failures=0, errors=0, skipped=0)
RECEIPT_PATHS = {
    'qualification': 'qualification/installed-validation.json',
    'qualification_result': 'qualification/result.json',
    'verification': 'collection/verification.json',
    'collection': 'collection/collection.json', 'collection_result': 'collection/result.json',
    'identity': 'qualification/imported-identity.json', 'junit': 'qualification/junit.xml',
    'executed': 'qualification/executed-nodes.json',
    'command': 'qualification/installed-tests-command.json',
    'command_result': 'qualification/installed-tests-result.json',
    'historical_build_result': 'build002/result.json',
    'configure_result': 'build002/configure-result.json',
    'install_result': 'build002/build-install-result.json',
    'original_configure': 'original/build-record.json',
}
RECEIPT_SHA256 = dict(zip(RECEIPT_PATHS, (
    '6e60da7759f3e1ae53bc86b93303bd321fa17bab5daac9905cb440cef5bd3f48',
    '38a98a426297086cfd3aabc76188fc0c27d8b4d48dcc398faf0dd79852fd7d54',
    '5c02cd8a2bc54ae601ea98eadd735eb4115005e1d16f3f2e53b65271370d77ec',
    'cccb0c295731be0f7eb78cf37289629505652c9a527e3aaac5e11b049f199cc4',
    '40cceac66bd2c68ed44de78425f810c79a023ef3b504ba348d192f10e7eb7112',
    'c92d6576d06894acb23e6a3e225fbde28cdce1c6e15e219b51ff234c69a1c084',
    '2bc42fcab3a353342eb6ba35110cb3bcf23410d4bbd540cef236a1680cc0d04f',
    'f0475926561de76cb281905f4a1a40c12c00bc22d1a7048943964e6bbe4703ee',
    '5ff195419420ed1ff2f88ede9b144efe38350b0c1cdc629d39c231291b8a842c',
    'db2655fc896e931b937439da6c14c6135dafaf2a2243a5081f2067d25b0f4b18',
    'a67fe225027d06d2cb805b3c606b2f5f9a753670551c772ce5dd111eea5a0400',
    '9a0955b57550a45b8cc8a4cccf8270d309df4238e738e4a7b049f08af84cada5',
    '113cb17b13aeec638c872cdf8436d5c70b64665de3ebc114a32583eb03b4cdf2',
    '896949860e3eb012e07b2ec45556ab4580d87e36de094ca5567d800149803380')))


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    def pairs(values):
        out = {}
        for key, value in values:
            need(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    def invalid(value):
        raise ValueError('nonfinite JSON')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def runtime():
    path = Path(__file__).with_name('runtime_qomp.py')
    spec = importlib.util.spec_from_file_location('_closed_qomp_runtime', path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def read_receipts(root):
    """Deployment input may be relocated; bytes/roles never selectable."""
    return {role: (Path(root)/name).read_bytes() for role, name in RECEIPT_PATHS.items()}


def verify_descriptor_operations(source, changes):
    """Explicit qomp ancestry exception, not a new generic source allowlist."""
    for name in changes:
        if name.startswith('devtools/isapol_s66_campaign/'):
            continue
        need(name in INHERITED_OPS and digest((Path(source)/name).read_bytes()) == INHERITED_OPS[name],
             'descriptor operations change scientific/build code: '+name)


def verify_receipts(blobs):
    need(type(blobs) is dict and set(blobs) == set(RECEIPT_SHA256), 'exact qomp receipt roles required')
    for role, expected in RECEIPT_SHA256.items():
        need(type(blobs[role]) is bytes and digest(blobs[role]) == expected,
             'measured qomp receipt digest differs: '+role)
    docs = {k: decode(v) for k,v in blobs.items() if k != 'junit'}
    q, r, c, e = (docs[k] for k in ('qualification','qualification_result','collection','executed'))
    roles = {'compiled': (SCIENTIFIC_COMMIT,SCIENTIFIC_TREE),
             'science': (SCIENTIFIC_COMMIT,SCIENTIFIC_TREE),
             'tests': (TEST_COMMIT,TEST_TREE), 'ops': (QUALIFYING_OPS,QUALIFYING_OPS_TREE)}
    need(q['schema'] == 'existing-install-qualification/1' and q['tests'] == TEST_SUMMARY
         and q['historical_job'] == '15320675' and q['historical_build_status'] == 'FAILED'
         and q['acceptance'] is False and q['production_adoption'] is False
         and r['status'] == 'PASSED_EXISTING_INSTALL_INTERFACE_NOT_PRODUCTION_ACCEPTANCE'
         and r['preserved_current_build_and_old_base'] is True
         and r['source'] == q['source']
         and all((q['source'][k]['commit'],q['source'][k]['tree']) == v for k,v in roles.items())
         and q['allocation']['JobId'] == r['allocation']['JobId'] == '15326636',
         'qomp qualification/source/preservation differs')
    identity = docs['identity']
    need(identity == q['imported_identity'] == r['identity']
         and identity['implementation'] == IMPLEMENTATION and identity['version'] == VERSION,
         'qomp installed identity differs')
    rt = runtime()
    need(identity['package'] == rt.PREFIX+'/lib/python3.13/site-packages/psi4/__init__.py'
         and str(Path(identity['core']).parent) == rt.PREFIX+'/lib/python3.13/site-packages/psi4',
         'foreign installed qomp prefix')
    nodes = c['nodes']
    need(len(nodes) == len(set(nodes)) == 556 and e['nodes'] == e['call_nodes'] == nodes
         and all(type(e[k]) is int and e[k] == 0 for k in (
             'returncode','collection_errors','collection_skips','failed_reports','skipped_reports')),
         'exact ordered successful556 required')
    cases = ET.fromstring(blobs['junit']).findall('.//testcase')
    need(len(cases) == 556 and not any(x.find(t) is not None for x in cases
         for t in ('failure','error','skipped')), 'qomp JUnit failure/skip/count')
    actual, files = [], {n.split('::')[0] for n in nodes}
    for case in cases:
        parts = case.get('classname','').split('.')
        match = [f for f in files if Path(f).stem in parts]
        need(len(match) == 1, 'foreign JUnit module')
        f = match[0]
        actual.append('::'.join([f]+parts[parts.index(Path(f).stem)+1:]+[case.get('name','')]))
    need(Counter(actual) == Counter(nodes), 'JUnit node multiplicity differs')
    cmd = docs['command_result']
    need(cmd['argv'] == docs['command'] and type(cmd['returncode']) is int
         and cmd['returncode'] == 0 and cmd['interrupted'] is False,
         'qualification command failed/interrupted')
    need(docs['historical_build_result']['status'] == 'FAILED'
         and all(type(docs[k]['returncode']) is int and docs[k]['returncode'] == 0
                 for k in ('configure_result','install_result')), 'historical build chain differs')
    need(q['verification_sha256'] == RECEIPT_SHA256['verification']
         and q['collection_sha256'] == RECEIPT_SHA256['collection']
         and q['collection_result_sha256'] == RECEIPT_SHA256['collection_result']
         and q['junit_sha256'] == RECEIPT_SHA256['junit'], 'qualification chain differs')
    return dict(schema='native-qomp-qualified-producer-pins/1',
        status='VERIFIED_INSTALLED_PRODUCER_RECEIPTS_ONLY',
        scientific_commit=SCIENTIFIC_COMMIT, scientific_tree=SCIENTIFIC_TREE,
        installed_test_commit=TEST_COMMIT, qualifying_ops_commit=QUALIFYING_OPS,
        implementation_sha256=dict(IMPLEMENTATION), version=VERSION,
        pfit_plan_sha256=PFIT_SHA256, receipt_sha256=dict(RECEIPT_SHA256),
        runtime_view_sha256=VIEW_SHA256, production_adoption=False,
        qualification='Installed interface only; not model/solver/cohort/production acceptance.')
