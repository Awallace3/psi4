"""Pure guards on synthetic measurements; not bound allocation acceptance."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'descriptor_admission_cb63305', Path(__file__).with_name('descriptor_admission_cb63305.py'))
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def fixture(producer='cb63305'):
    planner = guard.planner(producer)
    budget = dict(max_bytes=12*1024**3, max_work=6*10**12, max_io_bytes=16*1024**3, capacity='large')
    request = dict(schema='native-monomer-request/3', resources=budget,
        max_replay_work=10**12, pfit_maximum_work_bytes=256*1024**2,
        declaration=dict(bonds=[[0, 1], [0, 2]]))
    runtime = dict(molecule=dict(symbols=['O', 'H', 'H']),
        specification=dict(keywords=dict(function_kwargs=dict(native_monomer=request))))
    row = dict(monomer_id='monomer-0001', intended_memory_bytes=20*1024**3,
               runtime_input=runtime, runtime_input_sha256=guard.json_digest(runtime))
    identity = dict(package='/synthetic/private/psi4/__init__.py',
                    core='/synthetic/private/psi4/core.so', version='synthetic-only',
                    implementation=dict(sha256=dict(core='0'*64, isapol_python='1'*64)))
    dimensions = planner.Dimensions(3, 2, 12, 4, 5, 10, 10, (9, 9, 9), (2, 2, 2))
    plan = planner.plan(dimensions, budget, max_replay_work=10**12,
                        pfit_maximum_work_bytes=256*1024**2)
    shells = [dict(centre=0, am=1, exponents=[1.], coefficients=[1.]) for _ in range(4)]
    operations = {'descriptor_probe_cb63305.py': '2'*64, 'descriptor_probe.py': '3'*64,
                  'resource_plan_cb63305.py': '4'*64}
    if producer == 'ed561666':
        operations['resource_plan_ed561666.py'] = '5'*64
    report = dict(schema='native-campaign-descriptor-probe/2', policy_id=planner.POLICY,
        source_policy_sha256=planner.FILES, monomer_id=row['monomer_id'],
        runtime_input_sha256=row['runtime_input_sha256'], operational_sources_sha256=operations,
        implementation=identity['implementation'], package=identity['package'], core=identity['core'],
        version=identity['version'], method='PBE0/aug-cc-pVTZ',
        request_schema='native-monomer-request/3', coordinate_change_bohr=0.,
        intended_memory_bytes=row['intended_memory_bytes'], resources=budget,
        max_replay_work=10**12, pfit_maximum_work_bytes=256*1024**2,
        plan=plan, main_shells=shells, main_descriptor_sha256=guard.json_digest(shells),
        auxiliary_descriptor_sha256='a'*64, recipe_sha256='b'*64,
        atomic_descriptor_sha256=['c'*64]*3, shape_descriptor_sha256=['d'*64]*3,
        status='PASS_DESCRIPTOR_PLAN_ONLY')
    kwargs = dict(policy=planner.FILES, operational_sources_sha256=operations,
                  runtime_sha256=row['runtime_input_sha256'])
    return report, row, identity, kwargs


def test_new_cap32_child_cannot_be_swapped_with_historical_cap10_arithmetic():
    report, row, identity, kwargs = fixture('ed561666')
    result = guard.verify_descriptor(report, row, identity, **kwargs, producer='ed561666')
    assert result['dimensions']['drho_refinement_iterations'] == 32
    with pytest.raises(ValueError, match='binding differs'):
        guard.verify_descriptor(report, row, identity, **kwargs)
    historical, _, _, _ = fixture()
    report['plan'] = historical['plan']
    with pytest.raises(ValueError, match='arithmetic'):
        guard.verify_descriptor(report, row, identity, **kwargs, producer='ed561666')


def test_measured_shell_width_and_ledger_are_recomputed_without_formula_lookup():
    result = guard.verify_descriptor(*fixture()[:3], **fixture()[3])
    assert result['dimensions']['main_nbf'] == 12  # toy, not element-derived92
    assert result['status'] == 'PASS_DESCRIPTOR_PLAN_ONLY'


def test_opaque_fingerprints_are_recorded_not_independently_reconstructed():
    report, row, identity, kwargs = fixture()
    # Deliberately valid but changed opaque hashes document this trust boundary.
    # The surrounding supervisor seals the child code and full runtime.
    report['auxiliary_descriptor_sha256'] = 'e'*64
    report['recipe_sha256'] = 'f'*64
    report['atomic_descriptor_sha256'] = ['1'*64, '2'*64, '3'*64]
    report['shape_descriptor_sha256'] = ['4'*64, '5'*64, '6'*64]
    result = guard.verify_descriptor(report, row, identity, **kwargs)
    assert 'trusted pinned child' in result['descriptor_assurance']
    assert 'not independent shell-content/order' in result['descriptor_assurance']


def test_genuine_recomputed_refusal_is_verified_without_promotion():
    report, row, identity, kwargs = fixture()
    request = row['runtime_input']['specification']['keywords']['function_kwargs']['native_monomer']
    request['resources']['max_work'] = 1
    row['runtime_input_sha256'] = guard.json_digest(row['runtime_input'])
    kwargs['runtime_sha256'] = report['runtime_input_sha256'] = row['runtime_input_sha256']
    report['resources'] = request['resources']
    planner = guard.planner()
    dimensions = planner.Dimensions(**{
        key: tuple(value) if key in ('atomic_widths', 'shape_widths') else value
        for key, value in report['plan']['dimensions'].items()
        if key in ('sites', 'bonds', 'main_nbf', 'main_nshell', 'nocc',
                   'response_aux_width', 'density_aux_width', 'atomic_widths', 'shape_widths')})
    report['plan'] = planner.plan(dimensions, request['resources'],
        max_replay_work=request['max_replay_work'],
        pfit_maximum_work_bytes=request['pfit_maximum_work_bytes'])
    assert report['plan']['refusals']
    report['status'] = 'REFUSED_DESCRIPTOR_PLAN'
    assert guard.verify_descriptor(report, row, identity, **kwargs)['status'] == report['status']


@pytest.mark.parametrize('change', [
    dict(schema='native-campaign-descriptor-probe/1'), dict(monomer_id='monomer-0002'),
    dict(runtime_input_sha256='f'*64), dict(coordinate_change_bohr=1e-12),
    dict(pfit_maximum_work_bytes=2*1024**3), dict(max_replay_work=2*10**12),
    dict(package='/foreign/__init__.py'), dict(status='REFUSED_DESCRIPTOR_PLAN'),
    dict(request_schema='native-monomer-request/2'), dict(method='PBE0/unknown'),
    dict(operational_sources_sha256={}),
])
def test_cross_wired_descriptor_or_status_cannot_pass(change):
    report, row, identity, kwargs = fixture()
    report.update(change)
    with pytest.raises(ValueError):
        guard.verify_descriptor(report, row, identity, **kwargs)


@pytest.mark.parametrize('change', ['ledger_work', 'shell_width', 'site_inventory', 'request_body'])
def test_self_consistent_labels_do_not_replace_independent_arithmetic_or_input_binding(change):
    report, row, identity, kwargs = fixture()
    if change == 'ledger_work':
        report['plan']['charged_work'] += 1
    elif change == 'shell_width':
        report['main_shells'][0]['am'] = 2
        report['main_descriptor_sha256'] = guard.json_digest(report['main_shells'])
    elif change == 'site_inventory':
        report['atomic_descriptor_sha256'].pop()
    else:
        row['runtime_input']['specification']['keywords']['function_kwargs']['native_monomer']['max_replay_work'] += 1
    with pytest.raises(ValueError):
        guard.verify_descriptor(report, row, identity, **kwargs)


def test_effective_memory_includes_explicit_nonnumeric_headroom_and_shared_numeric_budget():
    _, row, _, _ = fixture()
    row['intended_memory_bytes'] = 80*1024**3
    row['runtime_input']['specification']['keywords']['function_kwargs']['native_monomer']['resources']['max_bytes'] = 64*1024**3
    allocation = dict(NumNodes='1', NumCPUs='16', MinMemoryNode='96G')
    result = guard.admit_memory({'monomers': [row]}, allocation, nonnumeric_headroom_bytes=16*1024**3)
    assert result['effective_job_memory_bytes'] == 96*1024**3
    with pytest.raises(ValueError, match='headroom exceeds'):
        guard.admit_memory({'monomers': [row]}, allocation,
                           nonnumeric_headroom_bytes=16*1024**3+1)
    per_cpu = dict(NumNodes='1', NumCPUs='16', MinMemoryCPU='6G')
    assert guard.admit_memory({'monomers': [row]}, per_cpu,
        nonnumeric_headroom_bytes=16*1024**3)['effective_job_memory_bytes'] == 96*1024**3


@pytest.mark.parametrize('value', [None, True, 0, '0', '96.5G', 'unlimited', '-1G'])
def test_implicit_or_unbounded_effective_memory_is_not_admitted(value):
    with pytest.raises(ValueError):
        guard.slurm_memory_bytes(value)


def test_full_identity_set_required_and_refusals_are_not_dropped():
    ids = {f'monomer-{i:04d}' for i in range(1, 132)}
    reports = [dict(monomer_id=mid, status='PASS_DESCRIPTOR_PLAN_ONLY') for mid in sorted(ids)]
    assert guard.verify_coverage(reports, ids) == 'PASS_ALL_131_DESCRIPTOR_PLANS_ONLY'
    for partial in (reports[:-1], reports+[copy.deepcopy(reports[0])],
                    [reports[0]]+reports[0:130]):
        with pytest.raises(ValueError, match='exact131'):
            guard.verify_coverage(partial, ids)
    reports[0]['status'] = 'REFUSED_DESCRIPTOR_PLAN'
    assert guard.verify_coverage(reports, ids) == 'REFUSED_DESCRIPTOR_PLANS'
