"""Read-only initial baselines and explicitly audited owned355 advancement.

No entries, specifications, records, managers or jobs are created or changed.
Initial capture cannot bless extra records. Advancement requires the observed
new audit hash, actual proof bytes and unchanged live state; it emits immutable
metadata, not a server lock or execution attestation.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

URI = 'http://128.61.254.96:7777'
OWNER = 'awallace43'
SERVER = dict(name='CDS QCFractal Server', version='0.70')


def helper(name):
    if name not in ('owned_dataset_cb63305', 'phoenix_pilot', 'dataset_fresh_cb63305',
                    'dataset_owned354_cb63305'):
        raise ValueError('closed read-only owned354 helpers required')
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_owned354_baseline_'+name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def prepare_baseline(client, *, blobs, authenticated_owner, specification, producer='cb63305'):
    if type(producer) is not str or producer not in ('cb63305', 'ed561666'):
        raise ValueError('closed baseline producer required')
    selection = {} if producer == 'cb63305' else dict(producer=producer)
    info = client.get_server_information()
    if {key: info.get(key) for key in SERVER} != SERVER:
        raise ValueError('approved owned deployment identity changed')
    guard = helper('owned_dataset_cb63305')
    accepted = guard.verify_first_acceptance(blobs, **selection)
    before = guard.snapshot_dataset(client, **selection)
    checkpoint = guard.prepare_initial_checkpoint(before,
        authenticated_owner=authenticated_owner, specification=specification, accepted=accepted,
        **selection)
    after = guard.snapshot_dataset(client, **selection)
    if before != after:
        raise RuntimeError('owned dataset changed during read-only baseline preparation')
    return dict(checkpoint, server=SERVER,
        qualification='Read-only accepted-first-case baseline, not mutation authorization alone '
                      'or an atomic server lock. Prepared cases require pinned-input reconstruction; '
                      'the live before-state must match exactly before any future append.')


def advance_baseline(client, *, blobs, authenticated_owner, specification,
                     previous_raw, previous_sha256, receipts, expected_audit_sha256,
                     captured, tensor_raw, proof_blobs=None):
    """Read-only live confirmation around the pure one-step audited reducer."""
    info = client.get_server_information()
    if {key: info.get(key) for key in SERVER} != SERVER:
        raise ValueError('approved owned deployment identity changed')
    guard = helper('owned_dataset_cb63305')
    accepted = guard.verify_first_acceptance(blobs, producer='ed561666')
    before = guard.snapshot_dataset(client, producer='ed561666')
    guard.validate_owned_dataset(before, authenticated_owner=authenticated_owner,
        specification=specification, accepted=accepted, producer='ed561666')
    if before != captured['delivered']:
        raise ValueError('live owned355 state differs from the captured audited delivery')
    result = guard.advance_checkpoint(previous_raw, previous_sha256, accepted=accepted,
        receipts=receipts, expected_audit_sha256=expected_audit_sha256,
        captured=captured, tensor_raw=tensor_raw, proof_blobs=proof_blobs)
    after = guard.snapshot_dataset(client, producer='ed561666')
    if after != before:
        raise RuntimeError('owned355 changed during audited checkpoint advancement')
    return dict(result, server=SERVER)


def approved_client(*, producer='cb63305'):
    """Keep credentials private and fix the only authorized owner/deployment."""
    helper('owned_dataset_cb63305').context(producer)
    import yaml
    server = yaml.safe_load(Path(
        '/storage/project/r-cs207-0/awallace43/qca/resources_losii_ac.yml').read_text())['server']
    if server.get('fractal_uri') != URI or server.get('username') != OWNER:
        raise ValueError('approved owner and URI required')
    selected = {} if producer == 'cb63305' else dict(producer=producer)
    transport = helper('dataset_fresh_cb63305')
    binding = transport.transport_binding(**selected)
    client = transport.client_class(**selected)(URI, username=OWNER, password=server['password'],
        verify=server['verify'], show_motd=False)
    return client, binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ('audit', 'outcome', 'pilot', 'envelope', 'output'):
        parser.add_argument('--'+role, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--producer', choices=('cb63305', 'ed561666'), default='cb63305')
    parser.add_argument('--advance', action='store_true')
    for name in ('previous-checkpoint', 'checkpoint-proofs', 'case-data', 'case', 'case-audit'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--previous-sha256')
    parser.add_argument('--case-audit-sha256')
    args = parser.parse_args()
    advance_fields = (args.previous_checkpoint, args.previous_sha256, args.case_data, args.case,
                      args.case_audit, args.case_audit_sha256)
    if args.advance:
        if args.producer != 'ed561666' or not all(advance_fields):
            raise ValueError('explicit cap32 advance requires parent, new audit/hash and case payload')
    elif any(advance_fields) or args.checkpoint_proofs is not None:
        raise ValueError('advancement inputs cannot be ignored in initial-baseline mode')
    root = Path(__file__).resolve().parents[2]
    git = lambda *argv: subprocess.check_output(['git', '-C', str(root), *argv], text=True).strip()
    if git('rev-parse', 'HEAD') != args.commit or git('status', '--porcelain'):
        raise ValueError('exact clean baseline-preparation checkout required')
    guard = helper('owned_dataset_cb63305')
    selection = {} if args.producer == 'cb63305' else dict(producer=args.producer)
    selected = guard.context(args.producer)
    blobs = {role: getattr(args, role).read_bytes() for role in selected['accepted_sha256']}
    guard.verify_first_acceptance(blobs, **selection)
    from qcportal.singlepoint import QCSpecification
    client, binding = approved_client(producer=args.producer)
    pilot = json.loads(blobs['pilot'])
    specification = QCSpecification(**helper('phoenix_pilot').execution_specification(pilot['specification']))
    receipts = None
    if args.advance:
        append = helper('dataset_owned354_cb63305')
        previous_raw = args.previous_checkpoint.read_bytes()
        proofs = append.read_checkpoint_proofs(previous_raw, args.previous_sha256, args.checkpoint_proofs)
        data = args.case_data
        receipts = dict(audit=args.case_audit.read_bytes(), case=args.case.read_bytes(),
            outcome=(data/'pilot-outcome.json').read_bytes(),
            envelope=(data/'server-native-envelope.json').read_bytes())
        captured = {role: json.loads((data/name).read_bytes()) for role, name in dict(
            before='owned-before-entry.json', entry='owned-after-entry.json',
            record='owned-after-record.json', delivered='owned-after-delivery.json').items()}
        report = advance_baseline(client, blobs=blobs, authenticated_owner=OWNER,
            specification=specification, previous_raw=previous_raw,
            previous_sha256=args.previous_sha256, receipts=receipts,
            expected_audit_sha256=args.case_audit_sha256, captured=captured,
            tensor_raw=(data/'server-tensors.npz').read_bytes(), proof_blobs=proofs)
    else:
        report = prepare_baseline(client, blobs=blobs, authenticated_owner=OWNER,
                                  specification=specification, **selection)
    report.update(operational_commit=args.commit, transport_binding=binding,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if receipts is not None:
        links = helper('dataset_owned354_cb63305').store_checkpoint_proofs(receipts)
        if links != report['ledger'][-1]['receipts_sha256']:
            raise ValueError('stored proof commitments differ from the audited transition')
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], dataset_id=selected['dataset_id'],
        entries=len(report['snapshot']['entries']), records=len(report['snapshot']['records']),
        generation=report.get('generation'), sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
