"""Synthetic algebra/controller checks only; no local quantum computation."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('drho_diagnostic_fixture',
    Path(__file__).with_name('drho_diagnostic_cb63305.py'))
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def test_input_is_exact_stripped_geometry_runtime_not_a_reference_or_saved_density(monkeypatch):
    raw = json.dumps(dict(molecule={'synthetic_geometry': True},
        specification={'model': {'method': 'pbe0', 'basis': 'aug-cc-pvtz'}})).encode()
    monkeypatch.setattr(diag, 'RUNTIME_SHA256', diag.sha(raw))
    assert diag.load_runtime(raw)['molecule'] == {'synthetic_geometry': True}
    with pytest.raises(ValueError, match='exact geometry-only'):
        diag.load_runtime(raw+b' ')
    value = json.loads(raw)
    value['saved_density'] = 'not allowed'
    changed = json.dumps(value).encode()
    monkeypatch.setattr(diag, 'RUNTIME_SHA256', diag.sha(changed))
    with pytest.raises(ValueError, match='one native'):
        diag.load_runtime(changed)


def test_only_exact_embers_diagnostic_allocation_is_accepted():
    allocation = dict(Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small',
                      NumCPUs='12', NumNodes='1', MinMemoryNode='64G', TimeLimit='01:00:00')
    diag.admit_allocation(allocation)
    for key, value in [('QOS', 'inferno'), ('NumCPUs', '4'), ('MinMemoryNode', '144G'),
                       ('TimeLimit', '08:00:00'), ('Account', 'foreign')]:
        with pytest.raises(ValueError):
            diag.admit_allocation(allocation | {key: value})


@pytest.mark.parametrize('change', ['none','manifest_file','worker_manifest_hash',
                                   'worker_policy','worker_commit','worker_input'])
def test_build_manifest_bytes_and_worker_provenance_must_agree(tmp_path,change):
    path = tmp_path/'synthetic-build-identity.json'
    raw = b'{"synthetic":"not a scientific build"}'
    path.write_bytes(raw)
    expected = diag.sha(raw)
    report = dict(status='COMPLETE_DIAGNOSTIC_NOT_A_MODEL',
        scientific_commit=diag.SCIENTIFIC_COMMIT, operational_commit='SYNTHETIC_OPS',
        runtime_input_sha256=diag.RUNTIME_SHA256, production_cap_unchanged=10,
        build_identity_sha256=expected)
    if change == 'manifest_file':
        path.write_bytes(raw+b' ')
    elif change == 'worker_manifest_hash':
        report['build_identity_sha256'] = 'foreign'
    elif change == 'worker_policy':
        report['production_cap_unchanged'] = 32
    elif change == 'worker_commit':
        report['operational_commit'] = 'foreign'
    elif change == 'worker_input':
        report['runtime_input_sha256'] = 'foreign'
    if change == 'none':
        diag.validate_report(path,expected,report,'SYNTHETIC_OPS')
    else:
        with pytest.raises(ValueError,match='manifest'):
            diag.validate_report(path,expected,report,'SYNTHETIC_OPS')


def test_capture_uses_native_assembly_and_discards_plain_lu_coefficients_unread():
    effects = []
    class Assembled:
        metric = NS(np=np.eye(1146))
        coulomb_metric = NS(np=np.eye(1146))
        rhs = np.ones(1146)
        raw_rhs = np.ones(1146)
        charges = np.ones(1146)
        @property
        def coefficients(self):
            raise AssertionError('plain-LU coefficients must not be read or exported')
    class Coulomb:
        def __init__(self, auxiliary):
            assert auxiliary == 'SYNTHETIC_EXPLICIT_AUX'
        def charges(self):
            return np.ones(1146)
        def fit_drho_c(self, basis, occupied, penalty, *, max_refinement_iterations):
            assert effects == ['fp_environment_probe']
            effects.append('native_assembly_cap0_only')
            assert basis == 'SYNTHETIC_EXPLICIT_MAIN'
            assert occupied.shape == (414,21) and penalty == 1000.
            assert max_refinement_iterations == 0
            return Assembled()
    def probe(a,b,cap):
        effects.append('fp_environment_probe')
        assert a.shape == (1,1) and b.shape == (1,) and cap == 1
        return ([1.],1,0.)
    core = NS(IsaAuxCoulomb=Coulomb, Matrix=NS(from_array=lambda a:a),
              _isa_refined_lu_solve=probe, _isa_exact_residual=lambda *args: None)
    arrays = diag.capture_operands(core,
        NS(basis='SYNTHETIC_EXPLICIT_MAIN', occupied=np.zeros((414,21))),
        'SYNTHETIC_EXPLICIT_AUX', np.zeros((12,3)))
    assert effects == ['fp_environment_probe','native_assembly_cap0_only']
    assert set(arrays) == {'A','b','J','q','raw_rhs','occupied','geometry'}
    assert all(a.dtype.str == '<f8' and a.flags.c_contiguous and not a.flags.writeable
               for a in arrays.values())


def test_capture_refusal_is_not_replaced_by_another_solver():
    core = NS(_isa_refined_lu_solve=lambda *args: (_ for _ in ()).throw(ValueError('FP refusal')),
              _isa_exact_residual=lambda *args: None)
    with pytest.raises(ValueError, match='FP refusal'):
        diag.capture_operands(core, NS(occupied=np.zeros((414,21))), None, np.zeros((12,3)))


def example_arrays():
    return {k:diag.owned_array(v) for k,v in dict(A=np.eye(2),b=np.ones(2),q=np.ones(2)).items()}


def test_cap_refusal_is_preserved_and_32_is_only_a_diagnostic_candidate():
    arrays = example_arrays()
    calls = []
    def solve(a,b,cap):
        calls.append(cap)
        if cap == 10:
            raise ValueError('Drho-C refinement did not converge within the iteration cap')
        return ([1.,1.],12,.125)
    core = NS(_isa_refined_lu_solve=solve, _isa_exact_residual=lambda *args:[0.,0.])
    refused, x = diag.replay(core,arrays,10)
    assert refused['status'] == 'refused' and x is None
    assert refused['error_message'].endswith('iteration cap')
    converged, x = diag.replay(core,arrays,32)
    assert calls == [10,32] and converged['iterations'] == 12
    assert converged['status'] == 'converged_diagnostic_only'
    assert converged['fitted_electrons'] == 2. and converged['electron_error'] == -40.
    assert np.array_equal(x,[1.,1.])  # never rescale to the expected42 electrons


@pytest.mark.parametrize('cap', [0,1,11,33,True,None])
def test_replay_cap_is_not_a_production_policy_or_unbounded_scan(cap):
    with pytest.raises(ValueError, match='only explicit'):
        diag.replay(None,example_arrays(),cap)


def test_operand_mutation_and_unexpected_errors_remain_failures():
    arrays = example_arrays()
    def mutate(a,b,cap):
        a.flags.writeable = True
        a[0,0] = 2.
        return ([1.,1.],1,0.)
    core = NS(_isa_refined_lu_solve=mutate, _isa_exact_residual=lambda *args:[0.,0.])
    with pytest.raises(ValueError, match='operands changed'):
        diag.replay(core,arrays,32)
    core._isa_refined_lu_solve = lambda *args: (_ for _ in ()).throw(RuntimeError('unexpected'))
    with pytest.raises(RuntimeError, match='unexpected'):
        diag.replay(core,example_arrays(),32)


def test_checkpoint_preserves_unsymmetrized_bits_and_is_exclusive(tmp_path,monkeypatch):
    arrays = example_arrays()
    arrays['A'] = diag.owned_array(np.array([[1.,np.nextafter(.5,1.)],[.5,2.]]))
    path = tmp_path/'synthetic-operands.npz'
    report = diag.checkpoint(path,arrays)
    with np.load(path,allow_pickle=False) as saved:
        assert saved['A'][0,1] != saved['A'][1,0]
        assert diag.array_identity(saved['A']) == report['arrays']['A']
    with pytest.raises(FileExistsError):
        diag.checkpoint(path,arrays)
    monkeypatch.setattr(diag,'MAX_CHECKPOINT_BYTES',1)
    with pytest.raises(ValueError, match='before write'):
        diag.checkpoint(tmp_path/'too-large.npz',arrays)
    assert not (tmp_path/'too-large.npz').exists()


def test_checkpoint_refuses_nonnative_array_types(tmp_path):
    with pytest.raises(ValueError, match='binary64'):
        diag.checkpoint(tmp_path/'bad.npz',{'A':np.eye(2,dtype=np.float32)})


def test_help_and_source_cannot_launch_a_manager_or_isa_response_pipeline():
    result = subprocess.run([sys.executable,'-I','-S','-B',diag.__file__,'--help'],
                            capture_output=True,text=True)
    assert result.returncode == 0 and '--runtime' in result.stdout
    tree = ast.parse(Path(diag.__file__).read_text())
    calls = {ast.unparse(node.func).split('.')[-1] for node in ast.walk(tree)
             if isinstance(node,ast.Call)}
    assert not calls.intersection({'native_partition','run_molecular_properties',
        'IsaFixedDensity','IsaAController','bounded_properties','PortalClient',
        'ComputeManager','reset_records','submit','oeprop'})
    batch = Path(diag.__file__).with_suffix('.sbatch').read_text()
    assert '#SBATCH --qos=embers' in batch and '#SBATCH --signal=TERM@180' in batch
    assert 'ccasp_wait_child "$child" 20' in batch
    assert batch.index('ccasp_wait_child') < batch.index('DRHO_DIAGNOSTIC_CAPTURE_COMPLETE_NOT_MODEL')
