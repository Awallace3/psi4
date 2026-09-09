"""Strict algorithm tests; require a parent-built extension with fit_ov.

These are not reference-forward acceptance tests (no provisional tolerances).
"""
import gc
import math
import numpy as np
import pytest
import psi4


def matrix(a):
    return psi4.core.Matrix.from_array(np.asarray(a, dtype=float))


def basis(exponents, coefficients=None, orbital=False):
    core = psi4.core
    shells = []
    if coefficients is None:
        coefficients = np.ones(len(exponents))
    for exponent, coefficient in zip(exponents, coefficients):
        s = core.IsaGaussianShell()
        s.centre, s.l = 0, 0
        s.exponents, s.coefficients = [float(exponent)], [float(coefficient)]
        shells.append(s)
    return core.IsaExplicitBasis(core.IsaBasisRole.Orbital if orbital else core.IsaBasisRole.MolecularAux,
                                 core.IsaBasisRepresentation.Spherical if orbital else core.IsaBasisRepresentation.Cartesian,
                                 [[0., 0., 0.]], shells)


def problem(aux_exp=(.3, 1.1, 3.2), aux_coef=None):
    main_exp = np.array([.25, .8, 1.7, 4.])
    aux_exp = np.asarray(aux_exp)
    aux_coef = np.ones(len(aux_exp)) if aux_coef is None else np.asarray(aux_coef)
    aux, main = basis(aux_exp, aux_coef), basis(main_exp, orbital=True)
    occ = np.array([[.7, -.2], [.1, .8], [-.4, .3], [.6, -.1]])
    vir = np.array([[.2, .9], [-.7, .1], [.5, -.3], [.4, .8]])
    # Independent unnormalized co-centred s-Gaussian Coulomb formula.
    def coulomb(a, b):
        return 2*math.pi**2.5/(a*b*math.sqrt(a+b))
    q = np.array([c*(math.pi/a)**1.5 for a,c in zip(aux_exp,aux_coef)])
    j = np.array([[ca*cb*coulomb(a,b) for b,cb in zip(aux_exp,aux_coef)]
                  for a,ca in zip(aux_exp,aux_coef)])
    b = np.array([[[ca*coulomb(a,mu+nu) for nu in main_exp] for mu in main_exp]
                  for a,ca in zip(aux_exp,aux_coef)])
    return psi4.core.IsaAuxCoulomb(aux), main, occ, vir, q, j, b


def explicit_t(occ, vir, b):
    o, v = occ.shape[1], vir.shape[1]
    t = np.zeros((o*v, len(b)))
    for k in range(len(b)):
        left = np.zeros((o, len(occ)))
        for a in range(o):
            for nu in range(len(occ)):
                for mu in range(len(occ)):
                    left[a,nu] += occ[mu,a]*b[k,mu,nu]
        for r in range(v):
            for a in range(o):
                for nu in range(len(occ)):
                    t[a+o*r,k] += left[a,nu]*vir[nu,r]
    return t


@pytest.mark.parametrize('penalty', [0., .25, 1., 7.])
@pytest.mark.parametrize('permuted', [False, True])
def test_analytic_objective_order_and_ownership(penalty, permuted):
    p, main, occ, vir, q, j, b = problem()
    if permuted:
        occ, vir = occ[:, ::-1].copy(), vir[:, ::-1].copy()
    cm, vm = matrix(occ), matrix(vir)
    result = p.fit_ov(main, cm, vm, 'analytic explicit MAIN; not SCF', penalty)
    t = explicit_t(occ, vir, b)
    a = j+(penalty*q[:,None])*q[None,:]
    np.testing.assert_allclose(result.charges, q, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(result.coulomb_metric.np, j, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(result.metric.np, a, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(result.rhs.np, t, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(result.coefficients.np, np.linalg.solve(a,t.T).T, rtol=2e-10, atol=2e-10)
    # Same native operands / same general-LU objective, strict, no conditioning waiver.
    A, T, D = (getattr(result, key).np.copy() for key in ('metric','rhs','coefficients'))
    native_q = np.asarray(result.charges)
    np.testing.assert_array_equal(A, result.coulomb_metric.np+(penalty*native_q[:,None])*native_q[None,:])
    native_b = p.three_center(main).np.copy().reshape(3,4,4)
    native_t = np.asarray([((occ.T@block)@vir).T.reshape(-1) for block in native_b]).T
    np.testing.assert_allclose(T,native_t,rtol=3e-14,atol=3e-14)
    np.testing.assert_allclose(D, np.linalg.solve(A,T.T).T, rtol=2e-12, atol=2e-12)
    residual = np.linalg.norm(A@D.T-T.T)/(np.linalg.norm(A)*np.linalg.norm(D)+np.linalg.norm(T))
    assert residual < 2e-15 and result.relative_backward_residual < 2e-15
    assert abs(residual-result.relative_backward_residual) < 3e-16
    assert (result.nmain,result.naux,result.noccupied,result.nvirtual,result.ntransition)==(4,3,2,2,4)
    assert result.order == 'p=a+noccupied*r; occupied-fast'
    assert result.representation == 'fitted_density_coefficients'
    assert result.charge_penalty == penalty and result.lapack_info == 0
    assert 'C_DGESV' in result.solver
    cm.np[:] = 91.; vm.np[:] = -83.
    for key in ('metric','coulomb_metric','rhs','coefficients'):
        copy = getattr(result,key)
        expected = copy.np.copy()
        copy.np[:] = np.nan
        np.testing.assert_array_equal(getattr(result,key).np,expected)
    charges = result.charges
    original_charges = charges[:]
    charges[0] = 999.
    np.testing.assert_array_equal(result.charges,original_charges)
    survivor = result.coefficients
    del p, main, cm, vm
    gc.collect()
    np.testing.assert_array_equal(result.coefficients.np,D)
    assert result.provenance == 'analytic explicit MAIN; not SCF'
    del result
    gc.collect()
    np.testing.assert_array_equal(survivor.np,D)


def test_auxiliary_permutation_and_zero_transition_rhs():
    p, main, occ, vir, *_ = problem()
    original = p.fit_ov(main,matrix(occ),matrix(vir),'original AUX')
    permutation = [2,0,1]
    permuted = psi4.core.IsaAuxCoulomb(basis(np.array([.3,1.1,3.2])[permutation]))
    result = permuted.fit_ov(main,matrix(occ),matrix(vir),'permuted AUX')
    np.testing.assert_allclose(result.coefficients.np,original.coefficients.np[:,permutation],rtol=2e-11,atol=2e-11)
    np.testing.assert_allclose(result.rhs.np,original.rhs.np[:,permutation],rtol=3e-14,atol=3e-14)
    zero = p.fit_ov(main,matrix(occ*0),matrix(vir),'zero supplied Cocc',1.)
    np.testing.assert_array_equal(zero.rhs.np,np.zeros((4,3)))
    np.testing.assert_array_equal(zero.coefficients.np,np.zeros((4,3)))
    assert zero.relative_backward_residual == 0.


def test_ill_conditioned_is_not_rejected_or_regularized():
    p, main, occ, vir, _, _, _ = problem((.3, 1.1), (1., 1.e-6))
    result = p.fit_ov(main, matrix(occ), matrix(vir), 'intentionally scaled AUX')
    q = np.asarray(result.charges)
    J, A, T, D = (getattr(result,k).np.copy() for k in ('coulomb_metric','metric','rhs','coefficients'))
    assert np.linalg.cond(A) > 1e12
    np.testing.assert_array_equal(A,J+q[:,None]*q[None,:])
    np.testing.assert_allclose(D,np.linalg.solve(A,T.T).T,rtol=2e-12,atol=2e-12)
    assert result.relative_backward_residual < 2e-15
    ridge = np.linalg.solve(A+1e-10*np.eye(2),T.T).T
    assert np.linalg.norm(D-ridge)/np.linalg.norm(D) > .01


@pytest.mark.parametrize('penalty', [-1., np.inf, np.nan])
def test_bad_penalty(penalty):
    p, main, occ, vir, *_ = problem()
    with pytest.raises(Exception, match='penalty'):
        p.fit_ov(main,matrix(occ),matrix(vir),'test',penalty)


@pytest.mark.parametrize('which', ['occupied','virtual'])
@pytest.mark.parametrize('value', [np.inf, np.nan, 1e308])
def test_nonfinite_inputs_or_products(which, value):
    p, main, occ, vir, *_ = problem()
    (occ if which=='occupied' else vir)[:] = value
    with pytest.raises(Exception, match='nonfinite'):
        p.fit_ov(main,matrix(occ),matrix(vir),'test')


@pytest.mark.parametrize('shape', [(3,2), (4,0), (4,3)])
def test_bad_dimensions(shape):
    p, main, occ, vir, *_ = problem()
    with pytest.raises(Exception, match='dimensions'):
        p.fit_ov(main,matrix(np.zeros(shape)),matrix(vir),'test')


def test_roles_provenance_symmetry_singular():
    p, main, occ, vir, *_ = problem()
    with pytest.raises(Exception, match='Orbital'):
        p.fit_ov(basis([1.,2.,3.,4.]),matrix(occ),matrix(vir),'test')
    with pytest.raises(Exception, match='provenance'):
        p.fit_ov(main,matrix(occ),matrix(vir),'  ')
    dim = psi4.core.Dimension.from_list([2,2])
    symmetric = psi4.core.Matrix('two blocks',dim,dim)
    with pytest.raises(Exception, match='single symmetry block'):
        p.fit_ov(main,symmetric,matrix(vir),'test')
    singular = psi4.core.IsaAuxCoulomb(basis([1.,1.], [0.,0.]))
    with pytest.raises(Exception, match='C_DGESV.*INFO=.*singular'):
        singular.fit_ov(main,matrix(occ),matrix(vir),'test')


@pytest.mark.parametrize('n, message', [(5000,'resource limit'), (47000,'overflow')])
def test_preallocation_guards(n, message):
    # Small descriptors and thin C; must fail BEFORE allocating n*n B.
    main = basis(np.ones(n), orbital=True)
    p = psi4.core.IsaAuxCoulomb(basis([1.,2.,3.,4.,5.,6.]))
    with pytest.raises(Exception, match=message):
        p.fit_ov(main,matrix(np.ones((n,1))),matrix(np.ones((n,1))),'guard')


def test_penalty_product_overflow():
    p, main, occ, vir, *_ = problem()
    with pytest.raises(Exception, match='nonfinite'):
        p.fit_ov(main,matrix(occ),matrix(vir),'test',1e308)
