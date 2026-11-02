"""Pure preflight/launch bindings; live singleton transport has separate tests."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    'expanded_bootstrap_fixture', Path(__file__).with_name('phoenix_pilot_cb63305.py'))
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def fixture():
    draft = dict(monomer_id='monomer-0006', runtime_input_sha256='runtime-only-fixture',
        producer=dict(scientific_commit='cb63305-unit-fixture',
                      implementation_sha256={'synthetic': 'not scientific evidence'}))
    kwargs = dict(manifest_sha='manifest-fixture', qualification_sha='qualification-fixture',
        pilot_sha='pilot-fixture', worker_sources={'source.py': 'synthetic'},
        transport={'synthetic': 'unit-only'})
    preflight = dict(status='PASS_EXPANDED_WORKER_PREFLIGHT_ONLY', runtime_unchanged=True,
        monomer_id=draft['monomer_id'], scientific_build_commit=draft['producer']['scientific_commit'],
        build_identity_sha256=kwargs['manifest_sha'], qualification_sha256=kwargs['qualification_sha'],
        pilot_sha256=kwargs['pilot_sha'], runtime_input_sha256=draft['runtime_input_sha256'],
        operational_sources_sha256=kwargs['worker_sources'], transport_binding=kwargs['transport'],
        science=dict(implementation={'sha256': draft['producer']['implementation_sha256']}),
        descriptor_verification=dict(status='PASS_DESCRIPTOR_PLAN_ONLY'))
    return preflight, draft, kwargs


def test_worker_discovery_does_not_mean_molecular_acceptance():
    preflight, draft, kwargs = fixture()
    pilot.validate_preflight(preflight, draft, **kwargs)
    assert preflight['status'] != 'PASS_SINGLE_PILOT_EVIDENCE'


@pytest.mark.parametrize('field,value', [
    ('status', 'PASS_WORKER_PREFLIGHT_ONLY'), ('runtime_unchanged', False),
    ('monomer_id', 'monomer-0004'), ('scientific_build_commit', 'acf54'),
    ('build_identity_sha256', 'foreign'), ('qualification_sha256', '197-case-fixture'),
    ('pilot_sha256', 'foreign'), ('runtime_input_sha256', 'changed-request'),
    ('operational_sources_sha256', {}), ('transport_binding', {}),
    ('science', {'implementation': {'sha256': {'foreign': 'runtime'}}}),
    ('descriptor_verification', {'status': 'REFUSED_DESCRIPTOR_PLAN'})])
def test_old_foreign_or_changed_worker_preflight_cannot_authorize_a_live_run(field, value):
    preflight, draft, kwargs = fixture()
    preflight[field] = copy.deepcopy(value)
    with pytest.raises(ValueError, match='exact expanded'):
        pilot.validate_preflight(preflight, draft, **kwargs)


def test_only_exact_approved_embers_molecular_shape_is_admitted():
    allocation = dict(Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small',
        NumCPUs='4', NumNodes='1', MinMemoryNode='24G', TimeLimit='02:00:00')
    pilot.admit_allocation(allocation)
    for key, value in [('QOS', 'inferno'), ('Account', 'foreign'), ('Partition', 'cpu-large'),
                       ('NumCPUs', '8'), ('NumNodes', '2'), ('MinMemoryNode', '20G'),
                       ('TimeLimit', '08:00:00')]:
        with pytest.raises(ValueError, match='exact effective'):
            pilot.admit_allocation(allocation | {key: value})


def test_help_is_pure_and_wrapper_is_term_safe_with_no_false_acceptance_marker():
    result = subprocess.run([sys.executable, '-I', '-S', '-B', pilot.__file__, '--help'],
                            text=True, capture_output=True)
    assert result.returncode == 0 and '--worker-preflight' in result.stdout
    text = Path(pilot.__file__).with_suffix('.sbatch').read_text()
    assert '#SBATCH --qos=embers' in text and '#SBATCH --signal=TERM@180' in text
    assert 'ccasp_wait_child "$child" 20' in text
    assert text.index('ccasp_wait_child') < text.index('EXPANDED_PILOT_DELIVERED_NOT_QUALIFIED')
    assert 'COMPLETED' not in text
