"""Pure descriptor-allocation guards; receipts/live runtime are caller gates.

Passing any function here alone is not bound allocation admission. The
supervisor must first verify immutable raw bank/producer/code receipts and
then verify the full runtime and inputs again after child execution.

MAIN shell contents independently determine MAIN width here. Auxiliary,
atomic, shape and recipe fingerprints are opaque measurements from that
trusted pinned child, not independently reconstructed shell/order proofs.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys


def planner():
    path = Path(__file__).with_name('resource_plan_cb63305.py')
    raw = path.read_bytes()
    spec = importlib.util.spec_from_file_location('_descriptor_admission_plan', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def slurm_memory_bytes(value):
    if type(value) is not str or re.fullmatch(r'[1-9]\d*[KMGT]?', value) is None:
        raise ValueError('explicit finite effective SLURM memory required')
    unit = value[-1] if value[-1].isalpha() else 'M'
    number = int(value[:-1] if value[-1].isalpha() else value)
    return number*1024**({'K': 1, 'M': 2, 'G': 3, 'T': 4}[unit])


def admit_memory(bank, allocation, *, nonnumeric_headroom_bytes):
    if (type(nonnumeric_headroom_bytes) is not int or nonnumeric_headroom_bytes <= 0
            or allocation.get('NumNodes') != '1'):
        raise ValueError('one-node explicit nonnumeric memory headroom required')
    cpus = allocation.get('NumCPUs', '')
    if type(cpus) is not str or not cpus.isdigit() or int(cpus) < 1:
        raise ValueError('explicit effective CPU allocation required')
    if allocation.get('MinMemoryNode') not in (None, '0'):
        effective = slurm_memory_bytes(allocation['MinMemoryNode'])
    elif allocation.get('MinMemoryCPU') not in (None, '0'):
        effective = slurm_memory_bytes(allocation['MinMemoryCPU'])*int(cpus)
    else:
        raise ValueError('explicit effective SLURM memory required')
    maximum = numeric = 0
    if not bank.get('monomers'):
        raise ValueError('nonempty complete descriptor bank required')
    for row in bank['monomers']:
        request = row['runtime_input']['specification']['keywords']['function_kwargs']['native_monomer']
        memory, budget = row['intended_memory_bytes'], request['resources']['max_bytes']
        if type(memory) is not int or type(budget) is not int or not 0 < budget <= memory:
            raise ValueError('declared numeric budget exceeds intended Psi4 memory')
        maximum, numeric = max(maximum, memory), max(numeric, budget)
    if maximum+nonnumeric_headroom_bytes > effective:
        raise ValueError('intended Psi4 memory plus headroom exceeds effective job memory')
    return dict(effective_job_memory_bytes=effective, maximum_numeric_budget=numeric,
        maximum_intended_psi4_memory=maximum, nonnumeric_headroom_bytes=nonnumeric_headroom_bytes,
        concurrency=1, qualification='Declared accounting only; not total measured process RSS.')


def verify_descriptor(report, row, identity, *, policy, operational_sources_sha256, runtime_sha256):
    """Recompute arithmetic from measured shell dimensions, not forecast widths."""
    module = planner()
    request = row['runtime_input']['specification']['keywords']['function_kwargs']['native_monomer']
    if (report.get('schema') != 'native-campaign-descriptor-probe/2'
            or report.get('policy_id') != module.POLICY
            or policy != module.FILES or report.get('source_policy_sha256') != policy
            or report.get('monomer_id') != row['monomer_id']
            or report.get('runtime_input_sha256') != runtime_sha256
            or runtime_sha256 != row['runtime_input_sha256']
            or runtime_sha256 != json_digest(row['runtime_input'])
            or report.get('operational_sources_sha256') != operational_sources_sha256
            or report.get('implementation', {}).get('sha256') != identity['implementation']['sha256']
            or Path(report.get('package', '')).resolve() != Path(identity['package']).resolve()
            or Path(report.get('core', '')).resolve() != Path(identity['core']).resolve()
            or report.get('version') != identity['version']
            or report.get('method') != 'PBE0/aug-cc-pVTZ'
            or request.get('schema') != 'native-monomer-request/3'
            or report.get('request_schema') != 'native-monomer-request/3'
            or report.get('coordinate_change_bohr') != 0
            or report.get('intended_memory_bytes') != row['intended_memory_bytes']
            or report.get('resources') != request['resources']
            or report.get('max_replay_work') != request['max_replay_work']
            or report.get('pfit_maximum_work_bytes') != request['pfit_maximum_work_bytes']):
        raise ValueError('descriptor producer/input/policy/resource binding differs')
    symbols = row['runtime_input']['molecule']['symbols']
    raw_dimensions = report['plan']['dimensions']
    data = {key: raw_dimensions[key] for key in (
        'sites', 'bonds', 'main_nbf', 'main_nshell', 'nocc',
        'response_aux_width', 'density_aux_width', 'atomic_widths', 'shape_widths')}
    for key in ('atomic_widths', 'shape_widths'):
        data[key] = tuple(data[key])
    dimensions = module.Dimensions(**data)
    if (dimensions.sites != len(symbols)
            or dimensions.nocc != sum({'C': 6, 'H': 1, 'N': 7, 'O': 8}[s] for s in symbols)//2
            or dimensions.bonds != len(request['declaration']['bonds'])):
        raise ValueError('descriptor molecular dimensions differ from immutable input')
    shells = report['main_shells']
    if (type(shells) is not list or len(shells) != dimensions.main_nshell
            or json_digest(shells) != report['main_descriptor_sha256']):
        raise ValueError('MAIN shell inventory/hash differs')
    for shell in shells:
        if (type(shell) is not dict or set(shell) != {'centre', 'am', 'exponents', 'coefficients'}
                or type(shell['centre']) is not int or not 0 <= shell['centre'] < len(symbols)
                or type(shell['am']) is not int or not 0 <= shell['am'] <= 5
                or not shell['exponents'] or len(shell['exponents']) != len(shell['coefficients'])
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                       for v in shell['exponents'])
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       for v in shell['coefficients'])):
            raise ValueError('invalid measured MAIN shell descriptor')
    if sum(2*s['am']+1 for s in shells) != dimensions.main_nbf:
        raise ValueError('measured MAIN width differs from shell inventory')
    for name in ('atomic_descriptor_sha256', 'shape_descriptor_sha256'):
        values = report[name]
        if type(values) is not list or len(values) != len(symbols):
            raise ValueError('complete ordered per-site descriptors required')
    hashes = [report['auxiliary_descriptor_sha256'], report['recipe_sha256'],
              *report['atomic_descriptor_sha256'], *report['shape_descriptor_sha256']]
    if any(type(v) is not str or re.fullmatch('[0-9a-f]{64}', v) is None for v in hashes):
        raise ValueError('exact descriptor hashes required')
    expected = module.plan(dimensions, request['resources'],
        max_replay_work=request['max_replay_work'],
        pfit_maximum_work_bytes=request['pfit_maximum_work_bytes'])
    if json_digest(expected) != json_digest(report['plan']):
        raise ValueError('descriptor arithmetic does not reproduce the pinned ledger')
    status = 'REFUSED_DESCRIPTOR_PLAN' if expected['refusals'] else 'PASS_DESCRIPTOR_PLAN_ONLY'
    if report.get('status') != status:
        raise ValueError('descriptor status/refusal consistency differs')
    return dict(monomer_id=row['monomer_id'], status=status, dimensions=raw_dimensions,
        charged_work=expected['charged_work'], numeric_bytes=expected['admission_numeric_bytes'],
        checkpoint_io_bytes=expected['checkpoint_io_bytes'],
        descriptor_assurance='MAIN width recomputed from shell inventory; AUX/atomic/shape/recipe '
                             'fingerprints and widths recorded from the trusted pinned child, '
                             'not independent shell-content/order verification.')


def verify_coverage(reports, expected_ids):
    ids = [r['monomer_id'] for r in reports]
    if (len(expected_ids) != 131 or len(reports) != 131 or len(set(ids)) != 131
            or set(ids) != set(expected_ids)):
        raise ValueError('complete unique exact131-descriptor identity set required')
    statuses = {r['status'] for r in reports}
    if not statuses <= {'PASS_DESCRIPTOR_PLAN_ONLY', 'REFUSED_DESCRIPTOR_PLAN'}:
        raise ValueError('unknown descriptor status')
    return ('PASS_ALL_131_DESCRIPTOR_PLANS_ONLY' if statuses == {'PASS_DESCRIPTOR_PLAN_ONLY'}
            else 'REFUSED_DESCRIPTOR_PLANS')
