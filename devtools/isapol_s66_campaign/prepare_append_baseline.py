"""Read existing owned dataset only; write a new first-methanol baseline file."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def helper(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-audit', type=Path, required=True)
    parser.add_argument('--previous-envelope', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipts = helper('append_receipts')
    guard = helper('dataset_guard')
    pilot = helper('phoenix_pilot')
    audit, audit_sha = receipts.load(args.previous_audit, receipts.WATER_AUDIT_SHA256)
    envelope, envelope_sha = receipts.load(args.previous_envelope, receipts.WATER_ENVELOPE_SHA256)
    draft, _ = receipts.load(args.pilot, receipts.METHANOL_PILOT_SHA256)
    if audit['status'] != 'PASS_SINGLE_PILOT_EVIDENCE' or audit['record_id'] != guard.WATER_RECORD:
        raise RuntimeError('accepted prior water audit required')
    import yaml
    from qcportal import PortalClient
    from qcportal.singlepoint import QCSpecification
    server = yaml.safe_load(
        Path('/storage/project/r-cs207-0/awallace43/qca/resources_losii_ac.yml').read_text())['server']
    if server['fractal_uri'] != 'http://128.61.254.96:7777' or server['username'] != 'awallace43':
        raise RuntimeError('approved owned deployment required')
    client = PortalClient(server['fractal_uri'], username=server['username'],
                          password=server['password'], verify=server['verify'], show_motd=False)
    info = client.get_server_information()
    if info['name'] != 'CDS QCFractal Server' or info['version'] != '0.70':
        raise RuntimeError('approved server identity changed')
    before = guard.snapshot_dataset(client)
    guard.validate_owned(before, authenticated_owner=server['username'],
                         implementation=envelope['implementation']['sha256'],
                         specification=QCSpecification(**pilot.execution_specification(draft['specification'])),
                         water_envelope_sha256=guard.digest(envelope))
    guard.validate_new_entry(before, draft)
    if (set(before['entries']) != {guard.WATER_ENTRY}
            or set(before['records']) != {str(guard.WATER_RECORD)}):
        raise RuntimeError('first methanol requires the exact audited water-only dataset')
    after = guard.snapshot_dataset(client)
    if before != after:
        raise RuntimeError('dataset changed during read-only baseline preparation')
    report = dict(schema='native-owned-append-baseline/1',
                  status='READ_ONLY_BASELINE_NOT_MOLECULAR_QUALIFICATION',
                  next_monomer_id='monomer-0004', snapshot=before,
                  previous_audit_sha256=audit_sha, previous_envelope_sha256=envelope_sha,
                  water_envelope_canonical_sha256=guard.digest(envelope),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  qualification='Read-only normal-server snapshot, not an atomic server lock.')
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], dataset_id=353,
                          output=str(args.output),
                          sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
