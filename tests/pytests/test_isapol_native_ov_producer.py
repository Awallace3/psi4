"""NumPy OV comparison fit and reporting, with explicit doubles; NOT native acceptance."""
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
    got = r._fit(p,None,occ,vir,1.)
    t = np.asarray([(occ.T@block@vir).T.reshape(-1) for block in b]).T
    A = j+q[:,None]*q[None,:]
    for actual, expected in zip(got,(q,j,A,t,np.linalg.solve(A,t.T).T)):
        np.testing.assert_array_equal(actual,expected)


def test_reporting_preserves_gates_lineage_outputs_and_missing_scope(tmp_path,monkeypatch):
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
        return report,np.array([[42.]])
    monkeypatch.setattr(r,'_measure',fake_measure)
    report_path, coeff_path = tmp_path/'new.json',tmp_path/'new.npy'
    report = r.run(directory,checkpoint,report_path,coeff_path,reference_replay=replay,
                   profile='provisional-1e-3')
    assert seen == [{}]
    assert report['producer'] == 'numpy'
    assert not report['passed'] and report['selected_profile_passed']
    assert report['checks']['metric']['provisional_threshold']==1e-9
    assert report['checks']['coefficients']['provisional_threshold']==1e-3
    assert report['provenance']['before']==report['provenance']['after']
    missing = report['broader_pipeline']['missing_implementations']
    assert all(x in missing for x in ('production C++ OV API','native SCF','native response operator producers','published localization'))
    assert report['broader_pipeline']['passed'] is False
    assert json.loads(report_path.read_text())['producer'] == 'numpy'
    np.testing.assert_array_equal(np.load(coeff_path),[[42.]])
    with pytest.raises(FileExistsError):
        r.run(directory,checkpoint,report_path,coeff_path,reference_replay=replay)
    assert len(seen)==1


def test_basis_lineage():
    r = load()
    d = dict(representation='S', nfunction=4, shells=np.array([[1,0,1,1]]),centres=np.zeros((1,3)))
    identity = r._basis_identity(d)
    assert identity['representation']=='S' and identity['nfunction']==4
    assert identity['shells']['shape']==[1,4]
    assert identity['shells']['sha256'] != r._array_identity(d['shells']+1)['sha256']
