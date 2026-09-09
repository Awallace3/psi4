"""Native API validation and direct-OV independent sampled-polynomial checks."""
import numpy as np
import pytest
import psi4
from psi4.driver.procrouting import isapol_oeprop as api
from test_isapol_partitioned_response import basis, site, poly2


def test_direct_ov_polynomial_order_ownership_and_representation():
    b = basis(((0.,0.,0.), (.8,.1,-.3)), psi4.core.IsaBasisRole.Orbital)
    s = site(neighbours=(0,1))
    c = np.array([[.8,.2,.4],[-.1,.7,-.2]])
    out = psi4.core.IsaPartitionedMultipoles(b,[s],'synthetic direct OV',1e-36,
                                            psi4.core.Matrix.from_array(c),1)
    pts = np.array(s.samples.points)
    centers = np.array([[0.,0.,0.],[.8,.1,-.3]])
    mo = (1.2*np.exp(-.7*np.sum((pts[:,None,:]-centers)**2,axis=2))) @ c
    expected = poly2(pts) @ (mo[:,:1]*mo[:,1:])
    np.testing.assert_allclose(out.values.np, expected, atol=2e-15, rtol=2e-15)
    assert out.representation == 'direct_ov'
    val = out.values; val.np[:] = 0
    np.testing.assert_allclose(out.values.np, expected, atol=2e-15)
    with pytest.raises(ValueError, match='representation'):
        psi4.core.IsaDistributedResponse(out,[0.],[psi4.core.Matrix.from_array(-np.eye(2))],
                                         'fitted_density_coefficients','wrong coordinates')
    r = psi4.core.IsaDistributedResponse(out,[0.],[psi4.core.Matrix.from_array(-np.eye(2))],
                                         'direct_ov','direct response')
    np.testing.assert_allclose(r.at_index(0).np,expected@expected.T,atol=2e-15)


@pytest.mark.parametrize('nocc',[0,3,-1])
def test_direct_ov_bad_occupation(nocc):
    with pytest.raises(ValueError):
        psi4.core.IsaPartitionedMultipoles(basis(role=psi4.core.IsaBasisRole.Orbital),
            [site()], 'bad occupancy',1e-36,psi4.core.Matrix.from_array(np.ones((1,3))),nocc)


@pytest.mark.parametrize('tasks',[(),('NOT_A_TASK',),('ATOMIC_PARTITION','ATOMIC_PARTITION')])
def test_request_validation(tasks):
    with pytest.raises(ValueError, match='request'):
        api.validate_request(None,tasks)


@pytest.mark.parametrize('option,value,message',[
    ('PARTITION_SCHEME','MBIS','MBIS'),('ATOMIC_RESPONSE_LOCALIZATION','LS','LW')])
def test_unsupported_policy(option,value,message):
    old = psi4.core.get_global_option(option)
    try:
        psi4.set_options({option:value})
        with pytest.raises(ValueError,match=message):
            api.validate_request(None,('ATOMIC_PARTITION',))
    finally:
        psi4.core.set_global_option(option,old)


def test_missing_result_and_unconverged_wavefunction():
    w = psi4.core.Wavefunction.build(psi4.geometry('O\nH 1 1\nH 1 1 2 100\nsymmetry c1'),'cc-pvdz')
    with pytest.raises(ValueError):
        psi4.atomic_property_result(w)
    with pytest.raises(ValueError):
        psi4.oeprop(w,'ATOMIC_PARTITION')


def test_generated_recipe_is_self_contained():
    w = psi4.core.Wavefunction.build(psi4.geometry('O\nH 1 1\nH 1 1 2 100\nsymmetry c1'),'cc-pvdz')
    r = api.generated_recipe(w)
    assert 'NOT modern CamCASP' in r.origin
    assert r.auxiliary.representation == 'Cartesian'
    assert [len(s.shape.shells) for s in r.sites] == [17,11,11]
    assert min(s.exponents[0] for s in r.sites[0].shape.shells) == .1
    assert r.controller.convergence == 1e-9


@pytest.mark.parametrize('tasks', [
    ('ATOMIC_POLARIZABILITY',),
    ('ATOMIC_PARTITION', 'ATOMIC_POLARIZABILITY'),
    ('DIPOLE', 'ATOMIC_POLARIZABILITY'),
    ('ATOMIC_PARTITION', 'ATOMIC_PARTITION'),
    ('ATOMIC_PARTITION', 42),
])
def test_public_rejection_invalidates_result_before_any_compute(tasks, monkeypatch):
    w = psi4.core.Wavefunction.build(psi4.geometry('H\nH 1 0.74\nsymmetry c1'), 'sto-3g')
    old = api.AtomicPropertyResult(('ATOMIC_PARTITION',), (), object(), None, 0.)
    w._native_atomic_property_result = old
    monkeypatch.setattr(psi4.core, 'OEProp', lambda *a: pytest.fail('ordinary compute entered'))
    monkeypatch.setattr(api, 'generated_recipe', lambda *a: pytest.fail('native compute entered'))
    # Psi4 ValidationError is not necessarily a ValueError.
    from psi4.driver.p4util.exceptions import ValidationError
    with pytest.raises((ValueError, ValidationError)):
        psi4.oeprop(w, *tasks)
    with pytest.raises(ValueError, match='No native'):
        psi4.atomic_property_result(w)
    assert old.tasks == ('ATOMIC_PARTITION',) and old.partition is not None


@pytest.mark.parametrize('entry', ['public', 'direct'])
def test_policy_rejection_invalidates_previous_result(entry):
    from psi4.driver.p4util import OptionsStateCM
    w = psi4.core.Wavefunction.build(psi4.geometry('H\nH 1 0.74\nsymmetry c1'), 'sto-3g')
    old = api.AtomicPropertyResult(('ATOMIC_PARTITION',), (), object(), None, 0.)
    w._native_atomic_property_result = old
    with OptionsStateCM(['PARTITION_SCHEME']):
        psi4.core.set_global_option('PARTITION_SCHEME', 'MBIS')
        with pytest.raises(ValueError, match='MBIS'):
            if entry == 'public':
                psi4.oeprop(w, 'ATOMIC_PARTITION')
            else:
                api.run(w, ('ATOMIC_PARTITION',))
    with pytest.raises(ValueError, match='No native'):
        psi4.atomic_property_result(w)
    assert old.tasks == ('ATOMIC_PARTITION',)


def test_ordinary_property_behavior_unchanged(monkeypatch):
    calls = []
    class Ordinary:
        def __init__(self, wfn):
            calls.append('init')
        def add(self, prop):
            calls.append(prop)
        def compute(self):
            calls.append('compute')
    monkeypatch.setattr(psi4.core, 'OEProp', Ordinary)
    class Wfn:
        pass
    w = Wfn()
    old = object()
    w._native_atomic_property_result = old
    assert psi4.oeprop(w, 'dipole') is None
    assert calls == ['init', 'DIPOLE', 'compute']
    assert w._native_atomic_property_result is old


def test_static_request_no_quadrature_or_pfit(monkeypatch):
    # Real request dispatch; forbid expensive prerequisites not needed for static.
    monkeypatch.setattr(api,'validate_request',lambda *a: 0.)
    monkeypatch.setattr(api.n.Quadrature,'from_casimir',lambda *a: pytest.fail('static quadrature'))
    class Stop(Exception): pass
    def capture(*a,**kw):
        assert kw['frequencies'] == (0.,) and kw['quadrature'] is None and not kw['pair_self']
        assert kw['response_basis'] == 'direct_ov'
        assert a[1].grid.radial_points == 160
        # Response has its own quadrature, rather than reusing the denser ISA grid.
        # Euler–MacLaurin n_r yields n_r-1 finite radial shells.
        assert kw['response_grid'].shape == (3 * (99 - 1) * 590, 4)
        raise Stop
    monkeypatch.setattr(api.n,'native_properties',capture)
    w = psi4.core.Wavefunction.build(psi4.geometry('O\nH 1 1\nH 1 1 2 100\nsymmetry c1'),'cc-pvdz')
    with pytest.raises(Stop):
        api.run(w,('ATOMIC_POLARIZABILITIES',))


@pytest.mark.parametrize('option,expected',[(None,'ordered_pairwise'),
                                            ('ORDERED_PAIRWISE','ordered_pairwise'),
                                            ('SHARED_SWEEP','shared_sweep')])
def test_named_response_algorithm_reaches_preflight_and_factory(option,expected,monkeypatch):
    # The public option only names the arrangement; it is recorded in the
    # request options and forwarded to both gates, never inferred.
    monkeypatch.setattr(api,'validate_request',lambda *a: 0.)
    seen = []
    original = api.estimate_response_work
    def observe(*a,**kw):
        seen.append(kw['algorithm'])
        return original(*a,**kw)
    monkeypatch.setattr(api,'estimate_response_work',observe)
    class Stop(Exception): pass
    def capture(*a,**kw):
        assert kw['response_algorithm'] == expected
        raise Stop
    monkeypatch.setattr(api.n,'native_properties',capture)
    old = psi4.core.get_global_option('ATOMIC_RESPONSE_ALGORITHM')
    try:
        if option is not None:
            psi4.set_options({'ATOMIC_RESPONSE_ALGORITHM':option})
        w = psi4.core.Wavefunction.build(psi4.geometry('O\nH 1 1\nH 1 1 2 100\nsymmetry c1'),'cc-pvdz')
        with pytest.raises(Stop):
            api.run(w,('ATOMIC_POLARIZABILITIES',))
    finally:
        psi4.core.set_global_option('ATOMIC_RESPONSE_ALGORITHM',old)
    assert seen == [expected]
    with pytest.raises(RuntimeError,match='not a valid choice'):
        psi4.core.set_global_option('ATOMIC_RESPONSE_ALGORITHM','BLAS3')
    assert psi4.core.get_global_option('ATOMIC_RESPONSE_ALGORITHM') == old
