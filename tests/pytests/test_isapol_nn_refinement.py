# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Refinement driven from the accepted constrained-NN chain's own LW tensors.

The trace's path is constrained NN -> distributed response -> LW -> PFIT.  The
committed refinement coverage so far anchored on values *declared by the test*
(`test_isapol_native_point_response.py`) or on the ISA-A/oeprop chain
(`.pi/audit/avtz-grac-refinement-demo.py`); neither is the traced path.  Here
the anchors are the LW local tensors of a strict-production-accepted
`fitted_auxiliary` chain at the traced penalty `lambda=1000`, and the fit-free
`direct_ov` row is refined alongside it off the *same* native context.

What that does and does not establish, stated exactly.
`native_point_charge_response` reuses the caller's orbital-derived H1/H2 with
direct-OV point-charge legs, so the refinement **targets are identical** for the
two rows -- that is asserted below rather than assumed.  What differs is only
the anchor and penalty centre, which is precisely the piece that was missing.
So this certifies the refinement stage consuming the constrained-NN chain's own
localized model; it does **not** turn those targets into the reference's
`SuppliedFittedPropagatorPointResponse` quantity, and no reference number is
quoted anywhere in this file.

Points and model stay caller-declared.  The historical lattice and `.pdef` are
missing artifacts, so nothing here is read from or inferred from a reference
`Cn` potential and no parameter count is borrowed from the dispersion track.

Frames are declared, and they have to be.  A `COPY` equivalence is expressed in
each site's *own local axes*, so the two hydrogens can share one variable set
only if their local tensors coincide -- and under global-identity frames
(`frames=None`) they do not: the in-plane `10,11c` dipole coupling has opposite
signs on the two, because the sites are mirror images.  The frames used here are
the reference case's `H2O.axes` convention, which is an **input** file already
committed verbatim as the `axes` field of
`data_isapol/camcasp_cn_pot_h2o_l2h1.json`; it is not read from the reference
`Cn` output, and it is applied below by reconstructing the two declared axes
from this molecule's own geometry rather than by importing any rotation.  What
goes wrong without it is tested exactly, and without an SCF, by
`test_isapol_refine.py::test_a_copy_equivalence_reports_how_far_its_sites_disagree`.
"""
import numpy as np
import pytest
import psi4
from psi4 import core
from psi4.driver.procrouting import isapol_native as n
from psi4.driver.procrouting import isapol_oeprop as o
from psi4.driver.procrouting import isapol_refine as R
from psi4.driver.procrouting import isapol_native_point_response as npr

PRODUCTION_GATE = 1.e-6
#: The traced constrained-NN penalty; `test_isapol_native_charge_penalty.py`
#: measures that this is the value the reference route exports and that strict
#: production LW accepts it on this state.
TRACED = 1.e3
#: Caller-declared, in the shape of the reference model but not read from it.
RANKS = {'O': 2, 'H': 1}


def _shell(radius, count):
    """Deterministic golden-angle shell: integer construction, no RNG."""
    k = np.arange(count)
    z = 1. - (2. * k + 1.) / count
    rho = np.sqrt(np.maximum(0., 1. - z * z))
    phi = np.pi * (3. - np.sqrt(5.)) * k
    return radius * np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1)


LATTICE = np.concatenate([_shell(4.5, 16), _shell(6.0, 16)], axis=0)


def axes_frame(z_direction, x_direction):
    """A proper local-to-global frame from a `z` axis and an in-plane `x` hint.

    Columns are the local axes in global Cartesian components, LW's and
    `RefinementSite`'s convention.  `x_direction` is projected onto the plane
    normal to `z_direction`; `y` completes the right-handed set.
    """
    zhat = np.asarray(z_direction, dtype=float)
    zhat = zhat/np.linalg.norm(zhat)
    xhat = np.asarray(x_direction, dtype=float)
    xhat = xhat - zhat*float(xhat @ zhat)
    assert np.linalg.norm(xhat) > 1.e-8, 'x hint is parallel to z'
    xhat = xhat/np.linalg.norm(xhat)
    return np.column_stack([xhat, np.cross(zhat, xhat), zhat])


def declared_frames(geometry_bohr):
    """The committed `H2O.axes` *input* declaration, rebuilt for this geometry.

        Axes
          H1  z global Z x from H2 to H1
          H2  z global Z x from H1 to H2
        End

    The oxygen is not declared there and therefore keeps global axes.  For a
    water in the `xz` plane with `z` bisecting it this is `diag(-1,-1,1)` on the
    first hydrogen and the identity on the other two sites -- the local `x` flip
    that makes the two hydrogens' local tensors coincide, which is what the
    `H2 H2 COPY H1 H1` declaration needs.
    """
    positions = np.asarray(geometry_bohr, dtype=float)
    global_z = (0., 0., 1.)
    return np.array([np.eye(3),
                     axes_frame(global_z, positions[1] - positions[2]),
                     axes_frame(global_z, positions[2] - positions[1])])


@pytest.fixture(scope='module')
def chains():
    """One PBE0/cc-pVDZ water state; the traced-lambda NN row and the fit-free row.

    cc-pVDZ MAIN with the recipe's default cc-pVDZ-JKFIT AUX is already
    MAIN-matched, which is why the constrained-NN row is accepted here at a
    fraction of the aVTZ cost (`test_isapol_matched_auxiliary.py` is where the
    AUX dependence itself is measured).  The native context is shared between
    the rows on purpose: H1/H2 come from the orbitals and are outside the
    penalty, so the two rows differ in the fit alone.
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
    assert quadrature.frequencies[0] == 0.  # the static node the anchors are taken at
    frames = declared_frames(wfn.molecule().geometry().to_array())
    shared = dict(bonds=((1, 0), (2, 0)), frames=frames, caller_converged=True,
                  kernel='alda_slater_pw92', exact_exchange=.25, local_scale=.75,
                  response_grid=response_grid, frequencies=quadrature.frequencies,
                  quadrature=quadrature, pair_self=True, response_algorithm='shared_sweep')
    out = {'direct_ov': n.native_properties(wfn, recipe, response_basis='direct_ov', **shared)}
    out['nn'] = n.native_properties(wfn, recipe, response_basis='fitted_auxiliary',
                                    response_context=out['direct_ov'].context,
                                    ov_charge_penalty=TRACED, **shared)
    for name, properties in out.items():
        assert not properties.failures, (name, [f.message for f in properties.failures])
    return wfn, out


def caller_model(local, provenance, *, weight_coefficient=1.e-3):
    """One variable set per site type, rank 2 on O and rank 1 on H, in LW local axes.

    `raw_local` carries ranks 1..3, i.e. channels 1..15 of the refinement's
    `(rank+1)**2` layout; channel 0 is the monopole, which is identically zero
    for a polarizability and is dropped by the cutoff rather than by a special
    case.
    """
    frames = np.asarray(local.frames.array, dtype=float)
    origins = np.asarray(local.origins.array, dtype=float)
    static = np.asarray(local.raw_local.array, dtype=float)[0]
    sites, anchors = [], []
    for index, label in enumerate(local.labels):
        kind = 'O' if label.upper().startswith('O') else 'H'
        rank = RANKS[kind]
        sites.append(R.RefinementSite(label=label, site_type=kind,
                                      origin_bohr=tuple(origins[index]),
                                      frame=tuple(map(tuple, frames[index])),
                                      rank_limit=rank))
        ncomp = (rank + 1)**2
        anchor = np.zeros((ncomp, ncomp))
        anchor[1:, 1:] = static[index][:ncomp - 1, :ncomp - 1]
        anchors.append(anchor)
    model = R.refinement_model(sites, anchors, frequency_au=0., cutoff=1.e-4, weight_type=3,
                               weight_coefficient=weight_coefficient, provenance=provenance)
    return model, anchors


@pytest.fixture(scope='module')
def refinements(chains):
    """Refine both rows' localized models against ONE shared target set."""
    wfn, properties = chains
    targets = npr.native_point_charge_response(properties['direct_ov'].context.response,
                                              wfn, LATTICE, frequencies=(0.,))
    out = {}
    for name, chain in properties.items():
        model, anchors = caller_model(
            chain.require_local(),
            f'caller-declared L2H1-shaped model on the {name} chain LW local tensors')
        result = R.refine(
            model, LATTICE, np.asarray(targets.packed_targets[0]), damping=0.,
            target_origin=core.IsaPfitTargetOrigin.NativeDirectActualPointResponse,
            response_representation=targets.representation,
            source_id=f'native PBE0/cc-pVDZ direct-OV point response; {name} anchors',
            generation_record=targets.generation_record)
        out[name] = (model, anchors, result)
    return targets, out


@pytest.mark.long
def test_the_traced_nn_chain_supplies_the_anchors_and_they_are_its_own(chains, refinements):
    """The anchors are the accepted constrained-NN chain's LW tensors, not the other row's."""
    _, properties = chains
    _, results = refinements
    assert f'fitted_auxiliary lambda={TRACED!r};' in properties['nn'].model
    assert properties['nn'].ov_fit.charge_penalty == TRACED
    assert 'lambda' not in properties['direct_ov'].model
    local = properties['nn'].require_local()
    assert local.metadata.residual_policy == 'production'
    assert local.metadata.residual_tolerance == PRODUCTION_GATE
    assert all(f.production_postcondition_passed for f in local.frequency_diagnostics)

    nn_model, nn_anchors, _ = results['nn']
    ov_model, ov_anchors, _ = results['direct_ov']
    # Same declared model shape, different numbers: two declared models, and the
    # anchor digest is what separates them downstream.
    assert nn_model.parameter_labels == ov_model.parameter_labels
    assert nn_model.channel_count == ov_model.channel_count
    assert nn_model.anchor_sha256 != ov_model.anchor_sha256

    # And *where* they differ, which is the measurement worth keeping.  Sorting
    # the moved anchors by the ranks the variable couples: the pure dipole
    # variables move by at most ~2e-3 of the largest anchor -- the two rows carry
    # essentially the same site dipole polarizability -- while the variables
    # touching rank 2 move by ~9e-2 of it, forty times as much (the worst is O's
    # 21s,21s, 5.02 against 4.12).  The same ordering holds on the aVTZ protocol
    # with a smaller ratio, 8.1 (`.pi/audit/avtz-nn-refinement.json`).  So the
    # constrained-NN fit's site-level footprint is a quadrupole effect, and a
    # dipole-level agreement between the two rows must not be quoted as
    # agreement of the localized model.  Measured; no reference value involved.
    ranks = [tuple(sorted((R.component_rank(e[0][1]), R.component_rank(e[0][2]))))
             for e in nn_model.parameter_entries]
    moved = [abs(a - b) for a, b in zip(nn_model.anchors, ov_model.anchors)]
    scale = max(1., max(abs(a) for a in ov_model.anchors))
    dipole = max(m for m, r in zip(moved, ranks) if r == (1, 1))
    beyond = max(m for m, r in zip(moved, ranks) if r != (1, 1))
    assert 0. < dipole/scale < 5.e-3
    assert beyond/scale > 2.e-2
    assert beyond > 5. * dipole
    # The penalty strengths follow the anchors, so they move too and are not shared.
    assert nn_model.strengths != ov_model.strengths
    # Under the declared frames the two hydrogens' local tensors coincide, so the
    # one shared variable set represents both of them.  The model measures that
    # itself, and here it is grid noise rather than the mirror-image sign flip a
    # global-identity frame would leave behind.
    for model, anchors in ((nn_model, nn_anchors), (ov_model, ov_anchors)):
        assert model.copy_anchor_discrepancy < 1.e-6
        assert np.max(np.abs(anchors[1] - anchors[2])) < 1.e-6
        for anchor in anchors:
            assert np.array_equal(anchor[0], np.zeros(anchor.shape[0]))


@pytest.mark.long
def test_both_rows_refine_against_one_shared_target_set(chains, refinements):
    """The targets do not depend on the response basis, and that is asserted, not assumed.

    `native_point_charge_response` reuses the shared H1/H2 with direct-OV legs,
    so feeding it the constrained-NN chain's context cannot produce a fitted
    target.  The declared origin therefore stays
    `NativeDirectActualPointResponse` for both rows: the difference between the
    refinements is the anchor centre alone.
    """
    wfn, properties = chains
    targets, results = refinements
    assert targets.representation == 'native_point_charge_ov_operators'
    other = npr.native_point_charge_response(properties['nn'].context.response, wfn,
                                             LATTICE, frequencies=(0.,))
    assert properties['nn'].context is properties['direct_ov'].context
    assert np.array_equal(np.asarray(other.packed_targets[0]),
                          np.asarray(targets.packed_targets[0]))
    for _, _, result in results.values():
        assert result.status == core.IsaPfitStatus.Solved
        assert result.refinement_status == 'PFIT_refined_against_point_to_point_response'


@pytest.mark.long
def test_each_refinement_beats_its_own_anchors_and_keeps_the_declared_symmetry(refinements):
    """A refinement is judged against the model it started from, not the other row's."""
    targets, results = refinements
    packed = np.asarray(targets.packed_targets[0])
    for name, (model, anchors, result) in results.items():
        fields = R.channel_fields(LATTICE, model)

        def residual_rms(tensors):
            predicted = R.pack_lower_triangle(R.point_to_point_response(fields, model, tensors))
            return float(np.sqrt(np.mean((predicted - packed)**2)))

        assert result.diagnostics.numerical_rank == model.parameter_count, name
        assert residual_rms(result.refined_tensors) < residual_rms(anchors), name
        assert residual_rms(result.refined_tensors) == pytest.approx(
            result.diagnostics.data_rms, rel=1.e-10)
        assert np.array_equal(result.refined_tensors[1], result.refined_tensors[2]), name
        for tensor in result.refined_tensors:
            assert np.array_equal(tensor, tensor.T)
            assert np.array_equal(tensor[0], np.zeros(tensor.shape[0]))


@pytest.mark.long
def test_the_refinement_moves_the_two_rows_much_more_than_the_penalty_separates_them(refinements):
    """Where the remaining site-level disagreement lives: the lattice, not the fit.

    Refining against one shared target set on this lattice shifts each row's
    dipole block by far more than the constrained-NN fit shifts the anchors, so
    the refinement stage -- not the choice of response basis -- is the dominant
    unmodelled effect at the site level.  Reported as a measured ordering; no
    reference value is involved.
    """
    _, results = refinements

    def isotropic(tensors):
        return np.array([np.trace(np.asarray(t)[1:4, 1:4])/3. for t in tensors])

    anchor_alpha = {k: isotropic(v[1]) for k, v in results.items()}
    refined_alpha = {k: isotropic(v[2].refined_tensors) for k, v in results.items()}
    between_rows = float(np.max(np.abs(anchor_alpha['nn'] - anchor_alpha['direct_ov'])))
    within_row = min(float(np.max(np.abs(refined_alpha[k] - anchor_alpha[k]))) for k in results)
    assert 0. < between_rows < within_row
    # And the refinement does not wash the rows together either: the two refined
    # models stay distinct, because the penalty holds each near its own anchors.
    assert float(np.max(np.abs(refined_alpha['nn'] - refined_alpha['direct_ov']))) > 0.


@pytest.mark.long
def test_a_strong_penalty_holds_the_constrained_nn_anchors(chains, refinements):
    """The penalty centre really is the NN chain's tensors, not a shared default."""
    _, properties = chains
    targets, _ = refinements
    local = properties['nn'].require_local()
    pinned, anchors = caller_model(local, 'pinned constrained-NN anchors',
                                   weight_coefficient=1.e12)
    result = R.refine(
        pinned, LATTICE, np.asarray(targets.packed_targets[0]), damping=0.,
        target_origin=core.IsaPfitTargetOrigin.NativeDirectActualPointResponse,
        response_representation=targets.representation,
        source_id='native PBE0/cc-pVDZ direct-OV point response; pinned nn anchors',
        generation_record=targets.generation_record)
    assert result.anchor_shift_maxabs < 1.e-6
    # The pinned parameters are the reference hydrogen's, written to both, so a
    # pinned fit reproduces the *equivalent* site's own anchors only to within
    # how far the COPY declaration stands from them to begin with.
    tolerance = 1.e-6 + pinned.copy_anchor_discrepancy
    for tensor, anchor in zip(result.refined_tensors, anchors):
        assert np.max(np.abs(np.asarray(tensor) - anchor)) < tolerance
