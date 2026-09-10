# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""What actually blocked the traced constrained-NN route at PBE0/aug-cc-pVTZ.

The generated demo recipe declares a Cartesian molecular AUX, and that AUX also
carries the Drho-C/ISA-A density fit.  Its default, cc-pVDZ-JKFIT, is far smaller
than MAIN=aug-cc-pVTZ, and there the traced penalty lambda=1000 constrained-NN
chain supplies a charge-flow defect that strict production LW rejects at most of
the Casimir nodes.  The obvious reading -- "the declared penalty is not
converged" -- is wrong, and that is what is measured here: raising lambda by a
decade leaves the defect against the fit-free route identical to five digits,
while matching the declared AUX to MAIN removes it at the traced lambda itself.

The fix is therefore a declared model choice (which AUX), never a tolerance, a
grid or a penalty: every LW call below is the same strict production policy and
`residual_policy` is untouched.  The two AUX choices are two different declared
models -- different partition, different dispersion -- so nothing is compared
across them except to report how far apart they are.  ``aux_basis`` is an
argument for exactly that reason; it is never inferred from MAIN.
"""
import os
import numpy as np
import pytest
import psi4
from psi4 import core
from psi4.driver.procrouting import isapol_native as n
from psi4.driver.procrouting import isapol_oeprop as o
from psi4.driver.procrouting.isapol_response_preflight import estimate_response_work

PRODUCTION_GATE = 1.e-6
#: The reference input's own value: I.P. 12.62063 eV minus HOMO -0.3989 Eh.
SHIFT = .06490004527520865
DEFAULT_AUX = 'cc-pVDZ-JKFIT'
MATCHED_AUX = 'aug-cc-pVTZ-JKFIT'
TRACED = 1.e3  # the traced constrained-NN penalty; see test_isapol_native_charge_penalty
LAMBDAS = (TRACED, 1.e4)


@pytest.fixture(scope='module')
def chains():
    """One PBE0/aug-cc-pVTZ fixed-GRAC water state; two declared AUX models.

    Threads are raised only to keep this affordable: the ordered sweep and the
    ALDA quadrature are thread-count independent to 2e-12 (asserted in
    test_isapol_native_response), which is far below everything compared here.
    """
    core.be_quiet()
    threads = core.get_num_threads()
    core.set_num_threads(min(8, os.cpu_count() or 1))
    water = psi4.geometry('0 1\nO 0. 0. 0.\nH -1.45365196 0. -1.12168732\n'
                          'H 1.45365196 0. -1.12168732\nunits bohr\nsymmetry c1\nno_com\nno_reorient\n')
    psi4.set_options({'basis': 'aug-cc-pvtz', 'reference': 'rks', 'scf_type': 'pk',
                      'e_convergence': 1e-10, 'd_convergence': 1e-10,
                      'dft_radial_points': 99, 'dft_spherical_points': 590, 'dft_alpha': .25,
                      'dft_grac_shift': SHIFT, 'dft_grac_alpha': .5, 'dft_grac_beta': 40.,
                      'dft_grac_x_func': 'XC_GGA_X_LB', 'dft_grac_c_func': 'XC_LDA_C_VWN'})
    try:
        _, wfn = psi4.energy('pbe0', molecule=water, return_wfn=True)
        options = core.IsaGridOptions()
        options.radial_points, options.spherical_points = 99, 590
        grid = core.IsaGrid(wfn.molecule().clone(), options)
        response_grid = np.column_stack((grid.x(), grid.y(), grid.z(), grid.w()))
        quadrature = n.Quadrature.from_casimir(core.CasimirGrid(10, .5))
        estimate_response_work(wfn.basisset().nbf(), wfn.nmo(), wfn.nalpha(),
                               response_grid.shape[0], algorithm='shared_sweep').require_pass()
        shared = dict(bonds=((1, 0), (2, 0)), frames=None, caller_converged=True,
                      kernel='alda_slater_pw92', exact_exchange=.25, local_scale=.75,
                      response_grid=response_grid, frequencies=quadrature.frequencies,
                      quadrature=quadrature, pair_self=True, response_algorithm='shared_sweep',
                      scf_correction='FIXED_GRAC', expected_grac_shift=SHIFT)
        out = {'wfn': wfn, 'recipes': {}}
        for aux in (DEFAULT_AUX, MATCHED_AUX):
            # Each AUX owns its own partition, so the fit-free route is recomputed
            # inside it rather than borrowed from the other model.
            recipe = o.generated_recipe(wfn, aux_basis=aux)
            out['recipes'][aux] = recipe
            context = None
            out[aux, 'direct_ov'] = n.native_properties(wfn, recipe, response_basis='direct_ov',
                                                        **shared)
            context = out[aux, 'direct_ov'].context
            for lam in LAMBDAS:
                out[aux, lam] = n.native_properties(wfn, recipe, response_basis='fitted_auxiliary',
                    response_context=context, ov_charge_penalty=lam, **shared)
                assert out[aux, lam].context is context
        return out
    finally:
        core.set_num_threads(threads)


def _c6_total(properties):
    return sum(float(c.value) for pair in properties.dispersion.pairs
               for c in pair.coefficients if c.order == 6)


def _order_total(properties, order):
    return sum(float(c.value) for pair in properties.dispersion.pairs
               for c in pair.coefficients if c.order == order)


def _reported(failure, key):
    return float(dict(part.split('=') for part in
                      failure.message.split('tolerance (')[1].split(')')[0].split(', '))[key])


@pytest.mark.long
def test_declared_aux_selects_a_partition_and_is_never_inferred_from_main(chains):
    """The AUX is an argument. Only the AUX moves; sites, grid and controller do not."""
    default, matched = (chains['recipes'][aux] for aux in (DEFAULT_AUX, MATCHED_AUX))
    assert default.auxiliary.name == f'{DEFAULT_AUX} Cartesian molecular AUX'
    assert matched.auxiliary.name == f'{MATCHED_AUX} Cartesian molecular AUX'
    assert MATCHED_AUX in matched.origin and 'NOT modern CamCASP preset' in matched.origin
    assert len(matched.auxiliary.shells) > len(default.auxiliary.shells)
    assert default.name == matched.name == 'GENERATED_JKFIT_ISA_A'
    assert default.grid == matched.grid and default.controller == matched.controller
    assert [s.label for s in default.sites] == [s.label for s in matched.sites]
    assert [[sh.exponents for sh in s.shape.shells] for s in default.sites] == \
           [[sh.exponents for sh in s.shape.shells] for s in matched.sites]
    # Both partitions are real converged Drho-C/ISA-A solves, and the declared
    # identity of each reaches the properties-level model string, which is what
    # forbids quoting one model's numbers against the other's.
    for aux in (DEFAULT_AUX, MATCHED_AUX):
        properties = chains[aux, 'direct_ov']
        assert properties.partition.converged and properties.partition.drho.charge_penalty == 1000.
        assert f'Drho-C ISA-A[{aux} Cartesian molecular AUX]' in properties.model
        assert aux in properties.partition.provenance


@pytest.mark.long
def test_molecular_isotropic_c6_is_partition_invariant_but_higher_orders_are_not(chains):
    """Why the end-to-end comparison is anchored on C6 and not on C8/C10.

    The fit-free route's molecular isotropic C6 total is the same number under
    both declared partitions -- to well below any tolerance asserted anywhere in
    this file -- because the site sum reconstructs a molecular observable.  The
    higher orders depend on where the distributed origins put the multipoles, so
    they move, and they must never be quoted as if they were equally invariant.
    """
    c6 = [_c6_total(chains[aux, 'direct_ov']) for aux in (DEFAULT_AUX, MATCHED_AUX)]
    assert abs(c6[0]-c6[1]) < 1e-9 and 46. < c6[0] < 48.
    for order in (8, 10):
        totals = [_order_total(chains[aux, 'direct_ov'], order) for aux in (DEFAULT_AUX, MATCHED_AUX)]
        assert abs(totals[0]-totals[1])/totals[0] > 1e-4


@pytest.mark.long
def test_default_aux_rejects_the_traced_penalty_at_aug_cc_pvtz(chains):
    """The blocker, stated as what LW actually reports: a supplied input defect."""
    rejected = chains[DEFAULT_AUX, TRACED]
    nodes = len(rejected.frequencies)
    assert 0 < len(rejected.failures) < nodes  # most nodes, not all of them
    for failure in rejected.failures:
        assert failure.stage == 'LW' and 'input-sum-rule' in failure.message
        assert _reported(failure, 'input-sum-rule') > PRODUCTION_GATE
        # Only the supplied-input residual exceeds the gate: the
        # algorithm-controlled ones stay inside it, because LW transports the
        # defect exactly instead of repairing it.
        assert _reported(failure, 'off-site') < PRODUCTION_GATE
        assert _reported(failure, 'reciprocity') < 1e-10
    assert max(_reported(f, 'input-sum-rule') for f in rejected.failures) > 5.*PRODUCTION_GATE
    with pytest.raises(RuntimeError, match='input-sum-rule'):
        _ = rejected.atomic_scalars


@pytest.mark.long
def test_matched_aux_accepts_the_traced_penalty_under_the_same_strict_gate(chains):
    """Same declared penalty, same strict policy, MAIN-matched declared AUX."""
    accepted = chains[MATCHED_AUX, TRACED]
    assert not accepted.failures, [f.message for f in accepted.failures]
    assert accepted.local.metadata.residual_policy == 'production'
    assert accepted.local.metadata.residual_tolerance == PRODUCTION_GATE
    assert all(f.production_postcondition_passed for f in accepted.local.frequency_diagnostics)
    residual = max(f.residuals.input_sum_rule for f in accepted.local.frequency_diagnostics)
    assert residual < PRODUCTION_GATE/3.
    # It is the same penalty that failed above, by more than an order of magnitude.
    worst = max(_reported(f, 'input-sum-rule') for f in chains[DEFAULT_AUX, TRACED].failures)
    assert worst > 20.*residual
    assert accepted.ov_fit.charge_penalty == TRACED
    assert accepted.dispersion is not None and len(accepted.dispersion.pairs) == 9


@pytest.mark.long
def test_the_defect_against_the_fit_free_route_is_the_aux_and_not_the_penalty(chains):
    """The measurement that rules the penalty out as the cause.

    A decade of lambda moves the fitted pair tensors by nothing at five digits
    under either AUX; changing the declared AUX moves them by almost an order of
    magnitude.  A quantity that is flat in the knob being blamed is not
    controlled by it.
    """
    defect = {}
    for aux in (DEFAULT_AUX, MATCHED_AUX):
        reference = chains[aux, 'direct_ov'].pair_tensors.array
        scale = max(1., float(np.max(np.abs(reference))))
        for lam in LAMBDAS:
            defect[aux, lam] = float(np.max(np.abs(
                chains[aux, lam].pair_tensors.array-reference)))/scale
        assert defect[aux, TRACED] == pytest.approx(defect[aux, 1.e4], rel=1e-5)
    assert defect[DEFAULT_AUX, TRACED] > .5
    assert defect[MATCHED_AUX, TRACED] < .1
    assert defect[DEFAULT_AUX, TRACED] > 8.*defect[MATCHED_AUX, TRACED]


@pytest.mark.long
def test_accepted_matched_chain_is_still_its_own_declared_model(chains):
    """Passing the gate does not make the fitted chain the fit-free chain.

    The residual difference is small but real and is reported, not absorbed: the
    fitted-auxiliary molecular C6 is a constrained-NN number and the direct-OV
    one is not, so the two are quoted separately even here.
    """
    fitted, unfitted = chains[MATCHED_AUX, TRACED], chains[MATCHED_AUX, 'direct_ov']
    assert 'fitted_auxiliary lambda=1000.0;' in fitted.model
    assert 'direct_ov' in unfitted.model and 'lambda' not in unfitted.model
    assert 'no PFIT' in fitted.model and 'no PFIT' in unfitted.model
    moved = abs(_c6_total(fitted)-_c6_total(unfitted))/_c6_total(unfitted)
    assert 1e-4 < moved < 1e-2
    # The largest remaining site defect is the H rank-3 scalar, a component the
    # rank-limited reference model does not even carry, so it bounds nothing
    # about the reference comparison and is recorded rather than asserted tight.
    scalars = {k: np.asarray(chains[MATCHED_AUX, k].atomic_scalars.array[0])
               for k in ('direct_ov', TRACED)}
    assert scalars['direct_ov'].shape == scalars[TRACED].shape == (3, 3)
    moved_scalar = np.abs(scalars[TRACED]-scalars['direct_ov'])
    assert moved_scalar.argmax() % 3 == 2  # the rank-3 column, on O and on H
    assert moved_scalar[1:, 2].min() > 5.*moved_scalar[1:, :2].max()
