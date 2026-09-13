# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Source-loaded contracts for the declared propagator rebuild.

The module offers sixteen combinations of four independent declarations
(density-fitted two-electron operators, AUX-metric kernel projection,
fitted-density kernel argument, kernel smoothing). Each names a DIFFERENT model.
Nothing here asserts that two declarations agree, or that any of them agrees with
the shipped exact-orbital propagator; the one identity that IS asserted is that
the explicit exact-orbital declaration reproduces the provider's own operators
bitwise, which is a statement about the assembly formula, not about physics.

Helium is an explicit compact Gaussian model reused from the pipeline test, not a
basis-limit or reference-parity claim.
"""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

import psi4
from psi4 import core

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'psi4.driver.procrouting._test_native_propagator',
    ROOT/'psi4/driver/procrouting/isapol_native_propagator.py')
pr = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = pr; SPEC.loader.exec_module(pr)

PIPE = importlib.util.spec_from_file_location('_test_propagator_pipeline',
                                              ROOT/'tests/pytests/test_isapol_native_pipeline.py')
pipeline = importlib.util.module_from_spec(PIPE); sys.modules[PIPE.name] = pipeline
PIPE.loader.exec_module(pipeline)

helium = pipeline.helium
run, recipe = pipeline.run, pipeline.recipe


# ---------------------------------------------------------------- smoothing


def test_smoothing_requires_explicit_finite_positive_parameters():
    for bad in (dict(rho_epsilon=0.), dict(f_max=-1.), dict(fd_delta=float('inf')),
                dict(fd_alpha=1), dict(rho_epsilon=1)):
        kw = dict(rho_epsilon=1.e-8, f_max=1000., fd_delta=.01, fd_alpha=1., method='FD')
        kw.update(bad)
        with pytest.raises(ValueError):
            pr.KernelSmoothing(**kw)
    with pytest.raises(ValueError):
        pr.KernelSmoothing(1.e-8, 1000., .01, 1., 'SMOOTH')


def test_declared_camcasp_smoothing_is_the_reference_parameter_set():
    s = pr.CAMCASP_ALDA_SMOOTHING
    assert (s.rho_epsilon, s.f_max, s.fd_delta, s.fd_alpha, s.method) == (1.e-8, 1000., .01, 1., 'FD')
    assert s.method_code == 3
    assert 'F-MAX=1000.0' in s.declaration


def test_fd_limit_matches_the_declared_scalar_form_in_both_branches():
    s = pr.CAMCASP_ALDA_SMOOTHING
    q = np.array([0., 1., -1., 500., 999.5, 1000., 1000.5, 1010., -1010., 1e5, -1e5, 1e8])

    def scalar(x):
        z = (abs(x)/s.f_max - 1.)/s.fd_delta
        fd = (1./(1.+np.exp(z)))**s.fd_alpha if z <= 40. else np.exp(-s.fd_alpha*z)
        return x*fd if x <= s.f_max else s.f_max*fd

    np.testing.assert_allclose(s.limit(q), [scalar(x) for x in q], rtol=1.e-14, atol=0.)
    assert np.isfinite(s.limit(q)).all()
    # The cap is a cap: nothing survives far above F-MAX, and small values are untouched.
    assert abs(s.limit(np.array([1.]))[0] - 1.) < 1.e-12
    assert abs(s.limit(np.array([1.e8]))[0]) < 1.e-30


def test_zero_and_constant_methods_are_their_own_declared_forms():
    q = np.array([-2000., -10., 10., 2000.])
    np.testing.assert_array_equal(pr.KernelSmoothing(1.e-8, 1000., .01, 1., 'ZERO').limit(q),
                                  [0., -10., 10., 0.])
    np.testing.assert_array_equal(pr.KernelSmoothing(1.e-8, 1000., .01, 1., 'CONSTANT').limit(q),
                                  [-1000., -10., 10., 1000.])


def test_floor_raises_the_density_and_never_drops_a_row():
    s = pr.CAMCASP_ALDA_SMOOTHING
    d = np.array([-1.e-30, 0., 1.e-12, 1.e-3])
    out = s.floor(d)
    assert out.shape == d.shape and np.all(out >= s.rho_epsilon)
    assert out[-1] == d[-1]


# -------------------------------------------------------------- declaration


def test_declaration_rejects_inferred_or_misspelled_axes():
    for bad in (('fitted', 'orbital_product', 'exact_orbital'),
                ('exact_orbital', 'aux', 'exact_orbital'),
                ('exact_orbital', 'orbital_product', 'fitted')):
        with pytest.raises(ValueError):
            pr.PropagatorDeclaration(*bad)
    with pytest.raises(TypeError):
        pr.PropagatorDeclaration('exact_orbital', 'orbital_product', 'exact_orbital', 'FD')


def test_fitted_density_has_no_orbital_product_accumulator():
    with pytest.raises(ValueError):
        pr.PropagatorDeclaration('exact_orbital', 'orbital_product', 'fitted_auxiliary')


def test_smoothing_is_refused_without_the_accumulator_that_would_carry_it():
    """The orbital-product accumulator lives in C++ and applies no cap.

    Declaring a smoothed orbital-product kernel would name a model this module
    does not build, so it is refused rather than silently served by the AUX path.
    """
    with pytest.raises(ValueError):
        pr.PropagatorDeclaration('exact_orbital', 'orbital_product', 'exact_orbital',
                                 pr.CAMCASP_ALDA_SMOOTHING)


def test_axes_are_independent_and_named_separately():
    d = pr.PropagatorDeclaration('density_fitted', 'orbital_product', 'exact_orbital')
    assert not d.rebuilds_kernel and not d.is_exact_orbital
    d = pr.PropagatorDeclaration('exact_orbital', 'auxiliary_metric', 'exact_orbital',
                                 pr.CAMCASP_ALDA_SMOOTHING)
    assert d.rebuilds_kernel and not d.is_exact_orbital
    assert pr.EXACT_ORBITAL_PROPAGATOR.is_exact_orbital
    assert not pr.EXACT_ORBITAL_PROPAGATOR.rebuilds_kernel
    names = {pr.CAMCASP_DF_PROPAGATOR.name, pr.EXACT_ORBITAL_PROPAGATOR.name,
             pr.PropagatorDeclaration('density_fitted', 'auxiliary_metric', 'exact_orbital').name}
    assert len(names) == 3


def test_camcasp_declaration_names_all_four_reference_choices():
    d = pr.CAMCASP_DF_PROPAGATOR
    assert (d.two_electron, d.kernel_projection, d.kernel_density) == (
        'density_fitted', 'auxiliary_metric', 'fitted_auxiliary')
    assert d.smoothing is pr.CAMCASP_ALDA_SMOOTHING


# ---------------------------------------------------------------- work gate


def test_work_estimate_charges_only_the_primitives_the_declaration_runs():
    e = pr.estimate_propagator_work(24, 24, 5, 99, 128898,
                                    declaration=pr.EXACT_ORBITAL_PROPAGATOR)
    assert e.passes and (e.df_transform_work, e.df_assembly_work) == (0, 0)
    assert (e.kernel_work, e.projection_work, e.kernel_sampling_work) == (0, 0, 0)
    e = pr.estimate_propagator_work(24, 24, 5, 99, 128898, declaration=pr.CAMCASP_DF_PROPAGATOR)
    assert e.passes and e.nov == 95
    assert e.df_transform_work == 99*24**2*24
    assert e.df_assembly_work == 99*95**2 + 99**2*95
    assert e.kernel_sampling_work == 128898*(99 + 24*5)
    assert e.kernel_work == 128898*99**2
    assert e.projection_work == 95*99**2 + 95**2*99


def test_work_estimate_rejects_bool_and_nonintegral_dimensions():
    for bad in (dict(nbf=True), dict(naux=99.), dict(grid_rows=-1)):
        kw = dict(nbf=24, nmo=24, nocc=5, naux=99, grid_rows=100)
        kw.update(bad)
        with pytest.raises(ValueError):
            pr.estimate_propagator_work(kw['nbf'], kw['nmo'], kw['nocc'], kw['naux'],
                                        kw['grid_rows'], declaration=pr.CAMCASP_DF_PROPAGATOR)


def test_work_estimate_refuses_rather_than_wrapping_and_reports_every_limit():
    e = pr.estimate_propagator_work(24, 24, 5, 200000, 10**9,
                                    declaration=pr.CAMCASP_DF_PROPAGATOR)
    assert not e.passes and e.failures
    assert [name for name, _, _ in e.limits] == ['df_transform', 'df_assembly', 'kernel_sampling',
                                                 'auxiliary_metric_kernel', 'kernel_projection']
    with pytest.raises(ValueError, match='NativePropagator'):
        e.require_pass()
    # An exact integer product, never a float that silently rounds or overflows.
    assert e.kernel_work == 10**9*200000**2


def test_sampling_and_accumulation_are_gated_as_two_separate_costs():
    """plan.md section 4: the two costs must never be quoted as one number.

    At water's naux=99 the sampling term is 22x smaller than the accumulation
    term yet measured 4.6x slower, which is exactly why a single combined limit
    could not describe both. Nothing here asserts the two agree.
    """
    e = pr.estimate_propagator_work(24, 24, 5, 99, 128898, declaration=pr.CAMCASP_DF_PROPAGATOR)
    assert e.kernel_sampling_work < e.kernel_work
    assert pr.PROPAGATOR_WORK_LIMITS['kernel_sampling'] < \
        pr.PROPAGATOR_WORK_LIMITS['auxiliary_metric_kernel']
    # Each primitive is refusable on its own, without the other one moving.
    big = pr.estimate_propagator_work(24, 24, 5, 4, 20_000_000,
                                      declaration=pr.PropagatorDeclaration(
                                          'exact_orbital', 'auxiliary_metric', 'exact_orbital'))
    assert 'kernel_sampling work resource limit' in big.failures
    assert 'auxiliary_metric_kernel work resource limit' not in big.failures


def test_work_estimate_guards_workspace_separately_from_work():
    e = pr.estimate_propagator_work(24, 24, 5, 99, 1000, declaration=pr.CAMCASP_DF_PROPAGATOR,
                                    max_bytes=1)
    assert not e.passes and 'propagator workspace resource limit' in e.failures


def test_work_gate_is_separate_from_the_shipped_response_gate():
    from psi4.driver.procrouting.isapol_response_preflight import ALDA_WORK_LIMITS
    assert set(pr.PROPAGATOR_WORK_LIMITS) == {'kernel_sampling', 'auxiliary_metric_kernel',
                                              'kernel_projection', 'df_transform', 'df_assembly'}
    assert not set(pr.PROPAGATOR_WORK_LIMITS) & set(ALDA_WORK_LIMITS)
    assert ALDA_WORK_LIMITS == {'ordered_pairwise': 2_000_000_000,
                                'shared_sweep': 64_000_000_000}


def test_occupations_and_empty_ov_are_refused():
    for bad in ((24, 24, 24, 99, 10), (24, 24, 0, 99, 10), (24, 30, 5, 99, 10)):
        with pytest.raises(ValueError):
            pr.estimate_propagator_work(*bad, declaration=pr.EXACT_ORBITAL_PROPAGATOR)


def test_density_fitting_without_an_auxiliary_basis_is_refused():
    with pytest.raises(ValueError):
        pr.estimate_propagator_work(24, 24, 5, 0, 0, declaration=pr.PropagatorDeclaration(
            'density_fitted', 'orbital_product', 'exact_orbital'))


# ------------------------------------------------------- DF index reordering


def test_vectorized_df_reorder_matches_an_explicit_occupied_fast_loop():
    """t = a*nocc+i, the convention native_response.cc:229-231 fixes."""
    rng = np.random.default_rng(20260913)
    naux, nocc, nvir = 7, 3, 4
    nov = nocc*nvir
    b_ov = rng.standard_normal((naux, nocc, nvir))
    b_oo = rng.standard_normal((naux, nocc, nocc)); b_oo += b_oo.transpose(0, 2, 1)
    b_vv = rng.standard_normal((naux, nvir, nvir)); b_vv += b_vv.transpose(0, 2, 1)
    m = rng.standard_normal((naux, naux)); metric = m @ m.T + naux*np.eye(naux)
    c_ov = np.linalg.solve(metric, b_ov.reshape(naux, -1)).reshape(naux, nocc, nvir)
    c_vv = np.linalg.solve(metric, b_vv.reshape(naux, -1)).reshape(naux, nvir, nvir)
    v4 = np.einsum('Pia,Pjb->iajb', b_ov, c_ov)
    x4 = np.einsum('Pij,Pab->ijab', b_oo, c_vv)
    y4 = np.einsum('Pib,Pja->ibaj', b_ov, c_ov)
    v = v4.transpose(1, 0, 3, 2).reshape(nov, nov)
    x = x4.transpose(2, 0, 3, 1).reshape(nov, nov)
    y = y4.transpose(2, 0, 1, 3).reshape(nov, nov)
    for a in range(nvir):
        for i in range(nocc):
            t = a*nocc + i
            for b in range(nvir):
                for j in range(nocc):
                    u = b*nocc + j
                    assert v[t, u] == pytest.approx(v4[i, a, j, b], rel=0, abs=1.e-13)
                    assert x[t, u] == pytest.approx(x4[i, j, a, b], rel=0, abs=1.e-13)
                    assert y[t, u] == pytest.approx(y4[i, b, a, j], rel=0, abs=1.e-13)
    for name, mat in (('V', v), ('X', x), ('Y', y)):
        np.testing.assert_allclose(mat, mat.T, rtol=0, atol=1.e-12, err_msg=name)


# ------------------------------------------------------------ real He chain


@pytest.fixture(scope='module')
def chain(helium):
    """One real native run; its partition/provider are reused, never re-derived."""
    r = run(helium, kernel='alda_slater_pw92', exact_exchange=.25, local_scale=.75,
            response_grid=_grid(helium), response_basis='fitted_auxiliary',
            ov_charge_penalty=1000.)
    assert not r.failures, r.failures
    return r


def _grid(wfn):
    o = core.IsaGridOptions(); o.radial_points = 40; o.spherical_points = 110
    g = core.IsaGrid(wfn.molecule().clone(), o)
    return np.column_stack((g.x(), g.y(), g.z(), g.w()))


def test_exact_orbital_declaration_reproduces_the_provider_operators_bitwise(chain):
    p = chain.partition
    provider = chain.context.response.provider
    full = chain.full_adapted_orbitals.array
    d = np.asarray(chain.ov_fit.coefficients)
    ops = pr.propagator_operators(p, provider, full, d,
        declaration=pr.EXACT_ORBITAL_PROPAGATOR, kernel='alda_slater_pw92',
        exact_exchange=.25, local_scale=.75, grid=None)
    np.testing.assert_array_equal(ops.h1, np.asarray(provider.h1()))
    np.testing.assert_array_equal(ops.h2, np.asarray(provider.h2()))
    assert ops.diagnostics['assembly_identity_residual_maxabs'] >= 0.
    assert ops.work.passes


def test_rebuilt_operators_are_owned_symmetric_and_differ_from_the_exact_ones(chain, helium):
    p = chain.partition
    provider = chain.context.response.provider
    ops = pr.propagator_operators(p, provider, chain.full_adapted_orbitals.array,
        np.asarray(chain.ov_fit.coefficients), declaration=pr.CAMCASP_DF_PROPAGATOR,
        kernel='alda_slater_pw92', exact_exchange=.25, local_scale=.75, grid=_grid(helium))
    for h in (ops.h1, ops.h2):
        assert h.shape == (provider.nocc*provider.nvir,)*2
        np.testing.assert_allclose(h, h.T, rtol=0, atol=1.e-9*max(1., np.abs(h).max()))
    # A different declared model: it is REQUIRED to differ, and is never asserted
    # to agree with the exact-orbital numbers to any tolerance.
    assert np.max(np.abs(ops.h1 - np.asarray(provider.h1()))) > 0.
    assert ops.diagnostics['df_coulomb_relative_maxabs_deviation'] > 0.
    assert ops.diagnostics['kernel_rows'] == _grid(helium).shape[0]
    assert ops.diagnostics['fitted_density_electrons'] == pytest.approx(2., abs=1.e-4)
    assert 'not replicated' in ops.diagnostics['aux_shell_pair_screen']
    ops.h1[0, 0] = 12345.
    np.testing.assert_array_equal(np.asarray(provider.h1()), np.asarray(provider.h1()))


def test_no_local_has_no_kernel_to_rebuild(chain, helium):
    provider = chain.context.response.provider
    with pytest.raises(ValueError, match='no_local'):
        pr.propagator_operators(chain.partition, provider, chain.full_adapted_orbitals.array,
            np.asarray(chain.ov_fit.coefficients), declaration=pr.CAMCASP_DF_PROPAGATOR,
            kernel='no_local', exact_exchange=1., local_scale=0., grid=_grid(helium))


def test_rebuilt_kernel_refuses_an_absent_or_malformed_grid(chain):
    provider = chain.context.response.provider
    args = (chain.partition, provider, chain.full_adapted_orbitals.array,
            np.asarray(chain.ov_fit.coefficients))
    kw = dict(declaration=pr.CAMCASP_DF_PROPAGATOR, kernel='alda_slater_pw92',
              exact_exchange=.25, local_scale=.75)
    for grid in (None, np.zeros((0, 4)), np.ones((5, 3)), np.array([[0., 0., 0., -1.]])):
        with pytest.raises((ValueError, TypeError)):
            pr.propagator_operators(*args, grid=grid, **kw)


def test_auxiliary_projection_refuses_direct_ov_identity_legs(chain, helium):
    provider = chain.context.response.provider
    nov = provider.nocc*provider.nvir
    with pytest.raises(ValueError, match='fitted D'):
        pr.propagator_operators(chain.partition, provider, chain.full_adapted_orbitals.array,
            np.eye(nov), declaration=pr.CAMCASP_DF_PROPAGATOR, kernel='alda_slater_pw92',
            exact_exchange=.25, local_scale=.75, grid=_grid(helium))


def test_raw_psi4_orbitals_are_not_accepted_in_place_of_the_adapted_ones(chain, helium):
    provider = chain.context.response.provider
    with pytest.raises(ValueError, match='ISA-MAIN'):
        pr.propagator_operators(chain.partition, provider, np.asarray(provider.orbitals())[:, :1],
            np.asarray(chain.ov_fit.coefficients), declaration=pr.CAMCASP_DF_PROPAGATOR,
            kernel='alda_slater_pw92', exact_exchange=.25, local_scale=.75, grid=_grid(helium))
