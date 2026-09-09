"""Pure selector/reporting tests with explicit doubles, NOT native acceptance."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest


def load():
    path = Path(__file__).parent/'data_isapol/oracle/compare_native_ov.py'
    spec = importlib.util.spec_from_file_location('ov_producer_test',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_numpy_default_preserves_left_association():
    r = load()
    rng = np.random.default_rng(21)
    occ, vir = rng.normal(size=(5,2)), rng.normal(size=(5,3))
    b = rng.normal(size=(4,5,5))
    j, q = np.diag([1.,2.,3.,4.]), np.array([.1,-.2,.3,.4])
    p = NS(charges=lambda:q, metric=lambda:NS(np=j), three_center=lambda main:NS(np=b.reshape(4,25)))
    got = r._fit(p,None,occ,vir,1.,None,'numpy','control')
    t = np.asarray([(occ.T@block@vir).T.reshape(-1) for block in b]).T
    A = j+q[:,None]*q[None,:]
    for actual, expected in zip(got[:5],(q,j,A,t,np.linalg.solve(A,t.T).T)):
        np.testing.assert_array_equal(actual,expected)
    assert got[5] is None


def test_cpp_calls_binding_and_uses_returned_operands_without_numpy_solve(monkeypatch):
    r = load()
    occ, vir = np.ones((4,2)), np.arange(8.).reshape(4,2)
    # Deliberately distinct sentinels, not a NumPy-generated fit; spy verifies plumbing.
    arrays = [np.arange(2.), np.eye(2)*3, np.eye(2)*7, np.ones((4,2))*11, np.ones((4,2))*13]
    result = NS(charges=arrays[0], coulomb_metric=NS(np=arrays[1]), metric=NS(np=arrays[2]),
                rhs=NS(np=arrays[3]), coefficients=NS(np=arrays[4]),
                representation='fitted_density_coefficients', lapack_info=0,
                provenance='exact input', charge_penalty=1., nmain=4,naux=2,noccupied=2,nvirtual=2,ntransition=4,
                order='p=a+noccupied*r; occupied-fast', solver='C_DGESV',relative_backward_residual=1e-17)
    calls = []
    def fit(main, cm, vm, provenance, penalty):
        calls.append((main,provenance,penalty))
        np.testing.assert_array_equal(cm.np,occ)
        np.testing.assert_array_equal(vm.np,vir)
        return result
    p = NS(fit_ov=fit)
    core = NS(Matrix=NS(from_array=lambda x:NS(np=x.copy())))
    monkeypatch.setattr(np.linalg,'solve',lambda *a:pytest.fail('CPP must not solve in NumPy'))
    got = r._fit(p,'MAIN',occ,vir,1.,NS(core=core),'cpp','exact input')
    assert calls == [('MAIN','exact input',1.)]
    for actual, expected in zip(got[:5],arrays):
        np.testing.assert_array_equal(actual,expected)
        actual[:] = -1
        assert not np.all(expected == -1)
    assert got[5]['solver'] == 'C_DGESV'
    result.provenance = 'wrong'
    with pytest.raises(ValueError,match='identity mismatch'):
        r._fit(p,'MAIN',occ,vir,1.,NS(core=core),'cpp','exact input')


def test_cpp_measurement_measures_binding_arrays(monkeypatch, tmp_path):
    r = load()
    A = np.array([[4.,1.],[2.,3.]])  # no assumed symmetry in reporting
    D = np.arange(8.).reshape(4,2)/8
    T = (A@D.T).T
    J, q = np.eye(2)*9, np.array([.2,-.3])
    descriptor = dict(representation='C',nfunction=2,centres=np.zeros((1,3)))
    orbitals = dict(kind='ORBITALS',header=(0,0,4,2,2,2,0),c=np.arange(16.).reshape(4,4))
    state = dict(kind='DIAGONAL_ENERGIES',bases={'AUX':descriptor,'MAIN':dict(descriptor,representation='S',nfunction=4)})
    snapshots = [dict(fit=((1,), (1.,0.,0.,0.)),d={'matrix':D.copy()})]
    phi = np.array([[1.,.5],[.2,1.]])
    aux = NS(evaluate_screened=lambda points,sites:NS(np=phi))
    checkpoint = dict(descriptors={'density_basis':descriptor,'density_neighbours':[1]},points=np.zeros((2,3)),weights=np.ones(2))
    base = NS(explicit_basis=lambda desc,role:aux if role=='aux' else 'main',read_checkpoint=lambda p:checkpoint)
    replay = NS(base=base,associate=lambda directory:([orbitals,state,dict(kind='J',matrix=J)],snapshots,None,None,None))
    calls = []
    def fit(main, cm, vm, provenance, penalty):
        calls.append(json.loads(provenance))
        return NS(charges=q,coulomb_metric=NS(np=J),metric=NS(np=A),rhs=NS(np=T),coefficients=NS(np=D),
                  representation='fitted_density_coefficients',lapack_info=0,provenance=provenance,
                  charge_penalty=penalty,nmain=4,naux=2,noccupied=2,nvirtual=2,ntransition=4,
                  order='p=a+noccupied*r; occupied-fast',solver='C_DGESV',relative_backward_residual=0.)
    core = NS(IsaBasisRole=NS(MolecularAux='aux',Orbital='main'),IsaAuxCoulomb=lambda aux:NS(fit_ov=fit),
              Matrix=NS(from_array=lambda a:NS(np=a.copy())))
    monkeypatch.setattr(np.linalg,'solve',lambda *args:pytest.fail('Measurement must not re-solve CPP result'))
    report, coefficients = r._measure(tmp_path,tmp_path/'checkpoint',NS(core=core),replay,producer='cpp')
    assert report['passed'] and report['relative_backward_residual']==0.
    assert report['native_fit_charge_max_absolute']==float(np.max(np.abs(D@q)))
    assert report['condition_A']==float(np.linalg.cond(A))
    assert report['compared_input_lineage']==calls[0]
    assert report['operand_identity']['A']==r._array_identity(A)
    assert report['operand_identity']['T']==r._array_identity(T)
    assert report['producer_api']=='psi4.core.IsaAuxCoulomb.fit_ov'
    assert 'NumPy diagnostic' not in report['limitations'][0]
    np.testing.assert_array_equal(coefficients,D)


@pytest.mark.parametrize('producer', ['numpy','cpp'])
def test_reporting_preserves_gates_lineage_outputs_and_missing_scope(producer,tmp_path,monkeypatch):
    r = load()
    directory = tmp_path/'input'; directory.mkdir()
    (directory/'event').write_text('immutable input')
    checkpoint, extension, replay = (tmp_path/name for name in ('checkpoint','core.so','replay.json'))
    checkpoint.write_text('checkpoint'); extension.write_bytes(b'explicit test double'); replay.write_text('{"passed":true}')
    monkeypatch.setattr(r.profiles,'production_psi4',lambda:NS(core=NS(__file__=str(extension))))
    monkeypatch.setattr(r.profiles,'load',lambda name:NS(replay=lambda d:dict(passed=True)))
    seen = []
    def fake_measure(*args, **kwargs):
        seen.append(kwargs)
        report = dict(passed=False, errors={key:dict(max_absolute=value,max_scaled=value)
                      for key,value in [('metric',0.),('coefficients',2e-5),('sampled_transition_density',1e-6)]},
                      pointwise_scaled_density_error=1e-6,relative_abs_weighted_density_l2=1e-6,
                      relative_backward_residual=1e-17)
        if producer == 'cpp':
            report.update(r._producer_identity('cpp'),cpp_diagnostics={'lapack_info':0})
        return report,np.array([[42.]])
    monkeypatch.setattr(r,'_measure',fake_measure)
    report_path, coeff_path = tmp_path/'new.json',tmp_path/'new.npy'
    report = r.run(directory,checkpoint,report_path,coeff_path,reference_replay=replay,
                   profile='provisional-1e-3',producer=producer)
    assert seen == ([{}] if producer=='numpy' else [{'producer':'cpp'}])
    assert report['producer'] == producer
    assert not report['passed'] and report['selected_profile_passed']
    assert report['checks']['metric']['provisional_threshold']==1e-9
    assert report['checks']['coefficients']['provisional_threshold']==1e-3
    assert report['provenance']['before']==report['provenance']['after']
    missing = report['broader_pipeline']['missing_implementations']
    assert ('production C++ OV API' in missing) == (producer=='numpy')
    assert all(x in missing for x in ('native SCF','native response operator producers','published localization'))
    assert report['broader_pipeline']['passed'] is False
    assert json.loads(report_path.read_text())['producer'] == producer
    np.testing.assert_array_equal(np.load(coeff_path),[[42.]])
    with pytest.raises(FileExistsError):
        r.run(directory,checkpoint,report_path,coeff_path,reference_replay=replay,producer=producer)
    assert len(seen)==1


def test_unknown_producer_and_basis_lineage():
    r = load()
    with pytest.raises(ValueError,match='Unknown OV producer'):
        r._producer_identity('reference-replay')
    d = dict(representation='S', nfunction=4, shells=np.array([[1,0,1,1]]),centres=np.zeros((1,3)))
    identity = r._basis_identity(d)
    assert identity['representation']=='S' and identity['nfunction']==4
    assert identity['shells']['shape']==[1,4]
    assert identity['shells']['sha256'] != r._array_identity(d['shells']+1)['sha256']


@pytest.mark.parametrize('producer', ['numpy','cpp'])
def test_cli_producer(producer,monkeypatch):
    r = load()
    argv = ['ov','capture','--checkpoint','cp','--reference-replay','replay',
            '--report','fresh.json','--coefficients','fresh.npy']
    if producer=='cpp':
        argv += ['--producer','cpp']
    monkeypatch.setattr(r.profiles.sys,'argv',argv)
    def run(*args,**kwargs):
        assert kwargs['producer']==producer
        return dict(selected_profile_passed=True)
    monkeypatch.setattr(r,'run',run)
    r.main()
