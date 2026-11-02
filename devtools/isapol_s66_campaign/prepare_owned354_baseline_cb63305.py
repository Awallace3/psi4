"""Read-only accepted-methylamine baseline for the remaining native bank.

No entries, specifications, records, managers or jobs are created or changed.
This snapshot is not an atomic lock or authorization to dispatch prepared cases.
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
    if name not in ('owned_dataset_cb63305', 'phoenix_pilot', 'dataset_fresh_cb63305'):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ('audit', 'outcome', 'pilot', 'envelope', 'output'):
        parser.add_argument('--'+role, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--producer', choices=('cb63305', 'ed561666'), default='cb63305')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    git = lambda *argv: subprocess.check_output(['git', '-C', str(root), *argv], text=True).strip()
    if git('rev-parse', 'HEAD') != args.commit or git('status', '--porcelain'):
        raise ValueError('exact clean baseline-preparation checkout required')
    guard = helper('owned_dataset_cb63305')
    selection = {} if args.producer == 'cb63305' else dict(producer=args.producer)
    selected = guard.context(args.producer)
    blobs = {role: getattr(args, role).read_bytes() for role in selected['accepted_sha256']}
    guard.verify_first_acceptance(blobs, **selection)
    # Credentials remain in the approved remote config and never enter output.
    import yaml
    from qcportal.singlepoint import QCSpecification
    server = yaml.safe_load(Path(
        '/storage/project/r-cs207-0/awallace43/qca/resources_losii_ac.yml').read_text())['server']
    if server.get('fractal_uri') != URI or server.get('username') != OWNER:
        raise ValueError('approved owner and URI required')
    transport = helper('dataset_fresh_cb63305')
    binding = transport.transport_binding(**selection)
    client = transport.client_class(**selection)(URI, username=OWNER, password=server['password'],
                                      verify=server['verify'], show_motd=False)
    pilot = json.loads(blobs['pilot'])
    specification = QCSpecification(**helper('phoenix_pilot').execution_specification(pilot['specification']))
    report = prepare_baseline(client, blobs=blobs, authenticated_owner=OWNER,
                              specification=specification, **selection)
    report.update(operational_commit=args.commit, transport_binding=binding,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], dataset_id=selected['dataset_id'],
        entries=len(report['snapshot']['entries']), records=len(report['snapshot']['records']),
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
