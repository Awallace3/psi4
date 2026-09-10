# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Why strict production LW rejects the fitted-auxiliary water chain, and what fixes it.

The rejection is an *input* sum-rule defect: LW transports the supplied
charge-flow defect exactly and cannot repair it, so the sanctioned place to
address one is the producer.  The producer here is the OV transition fit, whose
charge penalty ``A = J + lambda q q^T`` enforces something the exact answer
already satisfies (an OV transition density has exactly zero charge by MO
orthonormality), so its residual fitted charge falls as 1/lambda.  Converging
that declared constraint is not relaxing a gate: every LW call below is the same
strict production policy, and `residual_policy` is never touched.

What is measured, not assumed: that the archived lambda1 defect really is the
penalty artifact, that the same strict gate is met once the constraint is
converged, and that the converged chain is nonetheless a *differently declared
model* whose numbers must never be quoted against a recorded lambda1 reference.
"""
import numpy as np
import pytest
import psi4
from psi4 import core
from psi4.driver.procrouting import isapol_native as n
from psi4.driver.procrouting import isapol_oeprop as o

PRODUCTION_GATE = 1.e-6
#: 1e3 is the *traced* constrained-NN value: the reference route's own exported
#: penalty is ``lambda=1000`` (`data_isapol/oracle/replay_native_df.py` refuses
#: anything else), so it is measured here rather than only the round decades.
TRACED = 1.e3
LAMBDAS = (1., 1.e2, TRACED, 1.e4)


@pytest.fixture(scope='module')
def chains():
    """One PBE0/cc-pVDZ water state, one native context, several declared lambdas.

    The context is reused across lambda deliberately: H1/H2 come from the
    orbitals and are independent of the penalty, so the penalty is outside the
    response policy hash and no hidden re-solve happens between rows.
    """
    core.be_quiet()
    water = psi4.geometry('0 1\nO 0. 0. 0.\nH -1.45365196 0. -1.12168732\n'
                          'H 1.45365196 0. -1.12168732\nunits bohr\nsymmetry c1\nno_com\nno_reorient\n')
    psi4.set_options({'basis': 'cc-pvdz', 'reference': 'rks', 'scf_type': 'pk',
                      'e_convergence': 1e-10, 'd_convergence': 1e-10,
                      'dft_radial_points': 99, 'dft_spherical_points': 590, 'dft_alpha': .25})
    _, wfn = psi4.energy('pbe0', molecule=water, return_wfn=True)
    recipe = o.generated_recipe(wfn)
    options = core.IsaGridOptions()
    options.radial_points, options.spherical_points = 99, 590
    grid = core.IsaGrid(wfn.molecule().clone(), options)
    response_grid = np.column_stack((grid.x(), grid.y(), grid.z(), grid.w()))
    quadrature = n.Quadrature.from_casimir(core.CasimirGrid(4, .5))
    shared = dict(bonds=((1, 0), (2, 0)), frames=None, caller_converged=True,
                  kernel='alda_slater_pw92', exact_exchange=.25, local_scale=.75,
                  response_grid=response_grid, frequencies=quadrature.frequencies,
                  quadrature=quadrature, pair_self=True, response_algorithm='shared_sweep')
    out, context = {}, None
    out['direct_ov'] = n.native_properties(wfn, recipe, response_basis='direct_ov', **shared)
    for lam in LAMBDAS:
        out[lam] = n.native_properties(wfn, recipe, response_basis='fitted_auxiliary',
                                       response_context=context, ov_charge_penalty=lam, **shared)
        context = out[lam].context
    assert all(out[lam].context is context for lam in LAMBDAS)
    return out


@pytest.mark.long
def test_water_fitted_rejection_is_the_declared_lambda1_penalty_artifact(chains):
    """The 1/lambda law, and the frequency-by-frequency rejection it explains."""
    charge = {k: p.diagnostics['fitted_transition_charge_maxabs'] for k, p in chains.items()}
    assert charge['direct_ov'] == 0.  # no fit at all; zero analytically and exactly
    # Asymptotic in lambda: lambda1 is not yet in the 1/lambda regime, the decades above it are.
    assert 1.e2*charge[1.e2] == pytest.approx(1.e4*charge[1.e4], rel=5e-3)
    assert charge[1.] > 50.*charge[1.e2] and charge[1.e2] > 50.*charge[1.e4] > 0.

    # LW rejects every node at lambda1 and only some at 1e2: it is the SUPPLIED
    # charge-flow sum rule, transported exactly, that fails -- never the
    # algorithm-controlled residuals, which stay at machine level throughout.
    nodes = len(chains[1.].frequencies)
    assert len(chains[1.].failures) == nodes and 0 < len(chains[1.e2].failures) < nodes
    for failure in chains[1.].failures + chains[1.e2].failures:
        assert failure.stage == 'LW' and 'input-sum-rule' in failure.message
        reported = dict(part.split('=') for part in
                        failure.message.split('tolerance (')[1].split(')')[0].split(', '))
        assert float(reported['input-sum-rule']) > PRODUCTION_GATE
        assert float(reported['off-site']) < 1e-13 and float(reported['reciprocity']) < 1e-12
    with pytest.raises(RuntimeError, match='input-sum-rule'):
        _ = chains[1.].atomic_scalars


@pytest.mark.long
def test_water_converged_constraint_meets_the_same_strict_gate(chains):
    """The first accepted multi-atom fitted-auxiliary chain: declared lambda1e4.

    The residual left at lambda1e4 is no longer penalty-limited -- it sits at the
    same floor the fit-free direct_ov route reports on the identical grid, so it
    is a property of the response quadrature and not of the constraint.
    """
    accepted = chains[1.e4]
    assert not accepted.failures, [f.message for f in accepted.failures]
    assert accepted.local.metadata.residual_policy == 'production'
    assert accepted.local.metadata.residual_tolerance == PRODUCTION_GATE
    assert all(f.production_postcondition_passed for f in accepted.local.frequency_diagnostics)
    fitted = max(f.residuals.input_sum_rule for f in accepted.local.frequency_diagnostics)
    unfitted = max(f.residuals.input_sum_rule for f in chains['direct_ov'].local.frequency_diagnostics)
    assert fitted < PRODUCTION_GATE/10. and unfitted < PRODUCTION_GATE/10.
    assert .1 < fitted/unfitted < 10.
    assert accepted.dispersion is not None and len(accepted.dispersion.pairs) == 9


@pytest.mark.long
def test_water_accepted_penalty_is_a_differently_declared_model(chains):
    """Converging the constraint changes the fitted D, hence every downstream number.

    So the accepted chain is NOT the archived lambda1 model measured better: the
    two are different declared models and a lambda1 recorded reference error must
    never be quoted against this one.  The movement is bounded here, not ignored.
    """
    raw = {k: p.pair_tensors.array for k, p in chains.items()}
    scale = max(1., float(np.max(np.abs(raw[1.]))))
    moved = float(np.max(np.abs(raw[1.e4]-raw[1.])))/scale
    assert 1e-5 < moved < 1e-3
    # The largest defect against the fit-free route is a y-odd component of the
    # totally symmetric geometry, on which the rank-1 penalty direction has an
    # identically zero projection; it is EXACTLY lambda-independent, so the
    # penalty movement has to be read at its own argument, not at that one.
    index = np.unravel_index(np.argmax(np.abs(raw[1.]-raw['direct_ov'])), raw[1.].shape)
    assert abs(raw[1.e4][index]-raw[1.][index]) < 1e-14  # bit-identical in this build
    assert float(np.max(np.abs(raw[1.]-raw['direct_ov'])))/scale > 100.*moved
    for lam in LAMBDAS:
        assert f'fitted_auxiliary lambda={lam!r};' in chains[lam].model
        assert chains[lam].ov_fit.charge_penalty == lam
    assert 'lambda' not in chains['direct_ov'].model


@pytest.mark.long
def test_traced_constrained_nn_penalty_is_itself_accepted(chains):
    """The reference route declares lambda=1000, and that value passes the gate.

    It passes with only a small margin and its residual is still
    penalty-dominated -- an order above the quadrature floor lambda1e4 reaches --
    so the two are reported separately rather than as one "converged" claim.
    They agree closely enough in the tensors that the margin, not the physics,
    is what distinguishes them.
    """
    traced = chains[TRACED]
    assert not traced.failures, [f.message for f in traced.failures]
    assert all(f.production_postcondition_passed for f in traced.local.frequency_diagnostics)
    residual = {lam: max(f.residuals.input_sum_rule for f in chains[lam].local.frequency_diagnostics)
                for lam in (TRACED, 1.e4)}
    assert PRODUCTION_GATE/20. < residual[TRACED] < PRODUCTION_GATE
    assert residual[TRACED] > 5.*residual[1.e4]
    assert TRACED*chains[TRACED].diagnostics['fitted_transition_charge_maxabs'] == pytest.approx(
        1.e4*chains[1.e4].diagnostics['fitted_transition_charge_maxabs'], rel=5e-3)
    raw = {lam: chains[lam].pair_tensors.array for lam in (1., TRACED, 1.e4)}
    scale = max(1., float(np.max(np.abs(raw[1.]))))
    assert 0. < float(np.max(np.abs(raw[TRACED]-raw[1.e4])))/scale < 1e-6
