"""Propose finite, immutable resource declarations from source-width forecasts.

This runs before descriptors, never as a retry/promotion after refusal. Closed
resource classes select explicit declarations only; actual installed shell
descriptors and effective job admission are still mandatory. No SCF or Psi4
package is imported, and no molecule or result is accepted by this proposal.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

GIB = 1024**3
MEMORY_CLASSES = (
    dict(numeric_bytes=12*GIB, psi4_bytes=20*GIB, job_bytes=24*GIB, cpus=4),
    dict(numeric_bytes=32*GIB, psi4_bytes=40*GIB, job_bytes=56*GIB, cpus=8),
    dict(numeric_bytes=64*GIB, psi4_bytes=80*GIB, job_bytes=96*GIB, cpus=16),
)
WORK_CLASSES = tuple(t*10**12 for t in (6, 12, 15, 25, 50, 75, 100, 128))


def helper(name):
    if name not in ('resource_plan_cb63305', 'prepare_descriptor_bank_cb63305'):
        raise ValueError('closed proposal helpers required')
    path = Path(__file__).with_name(name+'.py')
    raw = path.read_bytes()
    spec = importlib.util.spec_from_file_location('_resource_proposal_'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def propose(inputs):
    prepare, planner = helper('prepare_descriptor_bank_cb63305'), helper('resource_plan_cb63305')
    if type(inputs) is not dict or set(inputs) != {'inventory', 'entries', 'declarations'}:
        raise ValueError('exact closed raw proposal inputs required')
    for role, raw in inputs.items():
        if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != prepare.INPUT_SHA256[role]:
            raise ValueError('raw source-width forecast input hash differs: '+role)
    inventory, entries, declarations = (json.loads(inputs[k])
                                        for k in ('inventory', 'entries', 'declarations'))
    ids = {f'monomer-{i:04d}' for i in range(1, 132)}
    monomers = {m['id']: m for m in inventory['monomers']}
    drafted = {e['attributes']['monomer_id']: e for e in entries['entries']}
    declared = {d['monomer_id']: d for d in declarations['monomers']}
    if (set(monomers) != ids or set(drafted) != ids or set(declared) != ids
            or any(len(rows) != 131 for rows in (
                inventory['monomers'], entries['entries'], declarations['monomers']))):
        raise ValueError('complete unique131-monomer forecast input set required')
    if any(doc['inventory_sha256'] != prepare.INPUT_SHA256['inventory']
           for doc in (entries, declarations)):
        raise ValueError('proposal inputs are cross-wired')
    contracts, forecasts = {}, []
    for monomer_id in sorted(ids):
        raw, entry, declaration = monomers[monomer_id], drafted[monomer_id], declared[monomer_id]
        symbols = raw['symbols']
        if (any(s not in ('C', 'H', 'N', 'O') for s in symbols)
                or raw['charge'] != 0 or raw['multiplicity'] != 1):
            raise ValueError('profile neutral CHNO singlet descriptors required')
        heavy, hydrogen = len(symbols)-symbols.count('H'), symbols.count('H')
        electrons = sum({'C': 6, 'H': 1, 'N': 7, 'O': 8}[s] for s in symbols)
        if electrons % 2:
            raise ValueError('closed-shell forecast requires even electrons')
        geometry = prepare.json_digest(entry['molecule'])
        if geometry != declaration['geometry_sha256'] or geometry != entry['attributes']['geometry_sha256']:
            raise ValueError('forecast geometry/declaration differs')
        dimensions = planner.Dimensions(
            sites=len(symbols), bonds=len(declaration['declaration']['bonds']),
            main_nbf=46*heavy+23*hydrogen, main_nshell=14*heavy+9*hydrogen,
            nocc=electrons//2, response_aux_width=136*heavy+55*hydrogen,
            density_aux_width=136*heavy+55*hydrogen,
            atomic_widths=tuple(49 if s == 'H' else 109 for s in symbols),
            shape_widths=tuple(8 if s == 'H' else 12 for s in symbols))
        # Explicit declaration policy, not a hidden solver cap increase.
        cap = 256*1024**2 if len(symbols) <= 7 else 2*GIB
        envelope = planner.plan(dimensions, dict(max_bytes=64*GIB, max_work=128*10**12,
            max_io_bytes=16*GIB, capacity='extended'),
            max_replay_work=10**12, pfit_maximum_work_bytes=cap)
        if envelope['status'] != 'WITHIN_PREDICTED_LEDGER_LIMITS':
            raise ValueError('forecast exceeds closed proposal envelope; explicit new review required')
        memory = next((m for m in MEMORY_CLASSES
                       if envelope['admission_numeric_bytes'] <= m['numeric_bytes']), None)
        work = next((w for w in WORK_CLASSES if envelope['charged_work'] <= w), None)
        if memory is None or work is None:
            raise ValueError('forecast exceeds closed finite resource classes')
        budget = dict(max_bytes=memory['numeric_bytes'], max_work=work, max_io_bytes=16*GIB,
                      capacity='extended' if work > 100*10**12 else 'large')
        confirmed = planner.plan(dimensions, budget, max_replay_work=10**12,
                                 pfit_maximum_work_bytes=cap)
        if confirmed['status'] != 'WITHIN_PREDICTED_LEDGER_LIMITS':
            raise ValueError('selected declaration is refused; no automatic promotion')
        contracts[monomer_id] = dict(geometry_sha256=geometry, resources=budget,
            max_replay_work=10**12, pfit_maximum_work_bytes=cap,
            intended_memory_bytes=memory['psi4_bytes'])
        forecasts.append(dict(monomer_id=monomer_id, charged_work=envelope['charged_work'],
            numeric_bytes=envelope['admission_numeric_bytes'],
            checkpoint_io_bytes=envelope['checkpoint_io_bytes'], memory_class=dict(memory),
            dimensions=envelope['dimensions'], native_base_bytes=envelope['native_pfit']['base_bytes']))
    manifest = dict(schema='native-s66x8-resource-declarations/2', status='DECLARED_NOT_ADMITTED',
        inventory_sha256=prepare.INPUT_SHA256['inventory'], policy_id=planner.POLICY, monomers=contracts)
    summary = dict(schema='native-s66x8-resource-proposal-forecast/2',
        status='FORECAST_DECLARATIONS_ONLY', policy_id=planner.POLICY,
        scientific_commit=planner.SCIENTIFIC_COMMIT,
        input_sha256={k: hashlib.sha256(v).hexdigest() for k, v in inputs.items()},
        monomers=forecasts,
        qualification='Source-derived widths, not actual installed shell descriptors, total RSS, '
                      'wall time, SCF scratch, admission or molecular results. Closed declarations '
                      'chosen before descriptors; no retry, model reduction or promotion after refusal.')
    return manifest, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inventory', 'entries', 'declarations', 'output', 'forecast'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    manifest, summary = propose({k: getattr(args, k).read_bytes()
                                 for k in ('inventory', 'entries', 'declarations')})
    summary['proposal_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for path, data in ((args.output, manifest), (args.forecast, summary)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=summary['status'], monomers=len(manifest['monomers']),
        resources_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        forecasts_sha256=hashlib.sha256(args.forecast.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
