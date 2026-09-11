# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""The declared eta of the constrained OV fit, at the C++ and driver levels.

``eta`` is the reference protocol's ``Eta`` under ``ConstraintType = 1``: every
Coulomb-metric element whose two AUX functions sit on DIFFERENT centres is scaled
by ``1-eta`` before the charge penalty is added.  It is a MODEL DECLARATION, not
a tolerance, a preconditioner or a conditioning repair.  These tests therefore
pin the exact algebraic form and the refusals; they never assert that two
different eta agree, because a chain run at a different eta is a different model.
"""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import psi4
from psi4 import core

PATH = Path(__file__).resolve().parents[2]/'psi4/driver/procrouting/isapol_native.py'
SPEC = importlib.util.spec_from_file_location('psi4.driver.procrouting._test_ov_eta', PATH)
n = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = n; SPEC.loader.exec_module(n)

pipeline = importlib.util.spec_from_file_location(
    'psi4.driver.procrouting._test_ov_eta_pipeline',
    Path(__file__).resolve().parent/'test_isapol_native_pipeline.py')
_pipe = importlib.util.module_from_spec(pipeline); pipeline.loader.exec_module(_pipe)
helium, run = _pipe.helium, _pipe.run

CENTRES = ((0., 0., 0.), (0., 0., 1.9))
#: (centre index, exponent) of every AUX s primitive; two centres, so the damping
#: is exercised on a genuinely off-centre block rather than a co-centred no-op.
AUX = ((0, .35), (0, 1.4), (1, .5), (1, 2.1))
MAIN = ((0, .3), (0, 1.1), (1, .45), (1, 1.7), (1, 5.))


def _shell(centre, exponent):
    s = core.IsaGaussianShell()
    s.centre, s.l = centre, 0
    s.exponents, s.coefficients = [float(exponent)], [1.]
    return s


def _basis(spec, *, orbital):
    return core.IsaExplicitBasis(
        core.IsaBasisRole.Orbital if orbital else core.IsaBasisRole.MolecularAux,
        core.IsaBasisRepresentation.Spherical if orbital else core.IsaBasisRepresentation.Cartesian,
        [list(c) for c in CENTRES], [_shell(*s) for s in spec])


@pytest.fixture(scope='module')
def two_centre():
    aux, main = _basis(AUX, orbital=False), _basis(MAIN, orbital=True)
    rng = np.random.default_rng(20260910)
    c = rng.standard_normal((len(MAIN), 4))
    occ, vir = core.Matrix.from_array(c[:, :2].copy()), core.Matrix.from_array(c[:, 2:].copy())
    return core.IsaAuxCoulomb(aux), main, occ, vir


def _fit(two_centre, eta, penalty=7.5):
    aux, main, occ, vir = two_centre
    return aux.fit_ov(main, occ, vir, 'analytic explicit MAIN; not SCF', penalty, eta)


@pytest.mark.parametrize('eta', [0., 1.e-12, 5.e-4, .25, .9999])
def test_damped_metric_is_exactly_the_declared_form(two_centre, eta):
    """A[k,l] = (1 - eta*[centre(k) != centre(l)])*J[k,l] + (lambda*q[k])*q[l], exactly."""
    result = _fit(two_centre, eta)
    centre = np.array([c for c, _ in AUX])
    scale = np.where(centre[:, None] == centre[None, :], 1., 1.-eta)
    j, q = result.coulomb_metric.np.copy(), np.asarray(result.charges)
    # Bitwise, so the association must be the implementation's own (lambda*q_k)*q_l.
    np.testing.assert_array_equal(result.metric.np, scale*j + (7.5*q)[:, None]*q[None, :])
    assert result.offsite_metric_damping == eta and result.charge_penalty == 7.5
    assert result.lapack_info == 0
    # The Coulomb metric itself is never damped; only the fit's A is.
    undamped = _fit(two_centre, 0.)
    np.testing.assert_array_equal(result.coulomb_metric.np, undamped.coulomb_metric.np)
    np.testing.assert_array_equal(result.rhs.np, undamped.rhs.np)


def test_default_is_the_undamped_fit_bitwise(two_centre):
    aux, main, occ, vir = two_centre
    default = aux.fit_ov(main, occ, vir, 'analytic explicit MAIN; not SCF', 7.5)
    explicit = _fit(two_centre, 0.)
    assert default.offsite_metric_damping == 0.
    for key in ('metric', 'coulomb_metric', 'rhs', 'coefficients'):
        np.testing.assert_array_equal(getattr(default, key).np, getattr(explicit, key).np)


def test_damping_actually_changes_the_fitted_density(two_centre):
    """Declared, not incidental: eta must move the coefficients it declares to move."""
    base = _fit(two_centre, 0.).coefficients.np.copy()
    damped = _fit(two_centre, .25).coefficients.np.copy()
    assert np.max(np.abs(damped-base)) > 1.e-6*max(1., np.max(np.abs(base)))


def test_damping_is_inert_on_a_single_centre_aux():
    """Every pair is on-centre, so the declared form reduces to the undamped fit."""
    aux = core.IsaExplicitBasis(core.IsaBasisRole.MolecularAux,
                                core.IsaBasisRepresentation.Cartesian, [[0., 0., 0.]],
                                [_shell(0, a) for a in (.3, 1.1, 3.2)])
    main = core.IsaExplicitBasis(core.IsaBasisRole.Orbital,
                                 core.IsaBasisRepresentation.Spherical, [[0., 0., 0.]],
                                 [_shell(0, a) for a in (.25, .8, 1.7, 4.)])
    rng = np.random.default_rng(7)
    c = rng.standard_normal((4, 4))
    occ, vir = core.Matrix.from_array(c[:, :2].copy()), core.Matrix.from_array(c[:, 2:].copy())
    p = core.IsaAuxCoulomb(aux)
    a = p.fit_ov(main, occ, vir, 'analytic explicit MAIN; not SCF', 3., 0.)
    for eta in (1.e-9, .4, .9999):
        b = p.fit_ov(main, occ, vir, 'analytic explicit MAIN; not SCF', 3., eta)
        np.testing.assert_array_equal(b.metric.np, a.metric.np)
        np.testing.assert_array_equal(b.coefficients.np, a.coefficients.np)
        assert b.offsite_metric_damping == eta


@pytest.mark.parametrize('eta', [1., 1.5, -1.e-16, -1., float('nan'), float('inf'), float('-inf')])
def test_undeclarable_damping_is_refused_outright(two_centre, eta):
    """eta must leave the damped metric a metric; nothing is silently clamped."""
    with pytest.raises(Exception, match='offsite metric damping must be finite'):
        _fit(two_centre, eta)


def test_driver_records_and_validates_the_declared_eta(helium):
    """He is one centre, so eta is inert here by construction, not by tolerance."""
    zero = run(helium, ov_metric_damping=0.)
    assert zero.ov_fit.offsite_metric_damping == 0.
    assert 'lambda=1.0; eta=0.0' in zero.model
    assert 'eta=0.0' in zero.ov_fit.provenance
    declared = run(helium, ov_metric_damping=.125, response_context=zero.context)
    assert declared.ov_fit.offsite_metric_damping == .125
    assert 'lambda=1.0; eta=0.125' in declared.model
    # Single-centre AUX: the declared form is the identity here, and the fit proves it.
    np.testing.assert_array_equal(declared.ov_fit.metric.np, zero.ov_fit.metric.np)


@pytest.mark.parametrize('eta', [1., -0.5, 0, '0.', None, np.float64(.5), float('nan')])
def test_bad_ov_metric_damping_rejected_before_any_fit(helium, eta):
    with pytest.raises(ValueError, match=r'ov_metric_damping must be an explicit finite float in \[0,1\)'):
        run(helium, ov_metric_damping=eta)


def test_direct_ov_forms_no_fit_and_refuses_a_declared_eta(helium):
    with pytest.raises(ValueError, match='direct_ov forms no transition fit; no metric damping applies'):
        run(helium, response_basis='direct_ov', ov_metric_damping=5.e-4)
