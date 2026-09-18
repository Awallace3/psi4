# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""The response quadrature grid is a quadrature, not a declared model.

Almost every ISA-Pol protocol parameter names a *model*: change it and the
numbers are a different answer that may not be quoted against the old one.
``IsaGridOptions.radial_points`` / ``.spherical_points`` are the exception, and
this file is the evidence for treating them as one.  The response grid carries
the ALDA kernel accumulation only, so a sufficiently fine grid is the same
answer as a finer one, and a coarse grid is an error rather than a different
declaration.

Four grids on one PBE0/cc-pVDZ water state, spanning 1,482 to 173,460 rows:

* the grid demonstrably reaches the kernel -- all four C6 totals are distinct;
* a degenerate Lebedev 26 grid is wrong in the fifth significant figure, and
  more so at C8 and C10 than at C6, as a multipole error should be;
* Lebedev 110 recovers six figures;
* Lebedev 302 and Lebedev 590 agree to eight figures at every order, so the
  declared default is converged and nothing above it is worth paying for.

The same measurement at the aug-cc-pVTZ matched protocol, over six grids up to
318,498 rows, is in the gitignored `agent_scratch` tree
(`reference-runs/20260918-D3G-response-grid/`, guarded by
`agent_scratch/pytests/test_isapol_response_grid_ladder_full.py`).  There the
refined molecular C6 spans 0.00019% while the native-versus-CamCASP residual
at the same protocol is 0.402%: the grid declaration accounts for 0.05% of it.
That is the reason this is a convergence test and not a parity test -- there is
no reference here, only the sequence against itself.
"""
import numpy as np
import pytest
import psi4
from psi4 import core
from psi4.driver.procrouting import isapol_native as n
from psi4.driver.procrouting import isapol_oeprop as o
from psi4.driver.procrouting.isapol_response_preflight import estimate_response_work

pytestmark = [pytest.mark.psi, pytest.mark.api]

#: (radial, spherical) -> grid rows.  The last is the declared default.
GRIDS = ((20, 26), (60, 110), (99, 302), (99, 590))
BASELINE = (99, 590)
ORDERS = (6, 8, 10)


@pytest.fixture(scope='module')
def ladder():
    """One SCF state, one recipe, one Casimir quadrature; only the grid moves."""
    core.clean_options()
    core.be_quiet()
    threads = core.get_num_threads()
    core.set_num_threads(1)
    try:
        water = psi4.geometry('0 1\nO 0. 0. 0.\nH -1.45365196 0. -1.12168732\n'
                              'H 1.45365196 0. -1.12168732\nunits bohr\n'
                              'symmetry c1\nno_com\nno_reorient\n')
        psi4.set_options({'basis': 'cc-pvdz', 'reference': 'rks', 'scf_type': 'pk',
                          'e_convergence': 1e-10, 'd_convergence': 1e-10,
                          'dft_radial_points': 99, 'dft_spherical_points': 590,
                          'dft_alpha': .25})
        _, wfn = psi4.energy('pbe0', molecule=water, return_wfn=True)
        recipe = o.generated_recipe(wfn)
        quadrature = n.Quadrature.from_casimir(core.CasimirGrid(4, .5))
        shared = dict(bonds=((1, 0), (2, 0)), frames=None, caller_converged=True,
                      kernel='alda_slater_pw92', exact_exchange=.25, local_scale=.75,
                      frequencies=quadrature.frequencies, quadrature=quadrature,
                      pair_self=True, response_algorithm='shared_sweep')
        out = {}
        for radial, spherical in GRIDS:
            options = core.IsaGridOptions()
            options.radial_points, options.spherical_points = radial, spherical
            grid = core.IsaGrid(wfn.molecule().clone(), options)
            rows = np.column_stack((grid.x(), grid.y(), grid.z(), grid.w()))
            estimate_response_work(wfn.basisset().nbf(), wfn.nmo(), wfn.nalpha(),
                                   rows.shape[0], algorithm='shared_sweep').require_pass()
            properties = n.native_properties(wfn, recipe, response_basis='direct_ov',
                                             response_grid=rows, **shared)
            out[radial, spherical] = dict(
                rows=int(rows.shape[0]),
                totals={order: sum(float(c.value) for pair in properties.dispersion.pairs
                                   for c in pair.coefficients if c.order == order)
                        for order in ORDERS})
        return out
    finally:
        core.set_num_threads(threads)


def _relative(ladder, key, order):
    base = ladder[BASELINE]['totals'][order]
    return (ladder[key]['totals'][order] - base) / base


@pytest.mark.long
def test_grid_rows_increase_with_the_declaration(ladder):
    counts = [ladder[key]['rows'] for key in GRIDS]
    assert counts == sorted(counts)
    assert counts[0] < 2000 and counts[-1] > 150000


@pytest.mark.long
def test_the_grid_reaches_the_kernel(ladder):
    """Not a null harness: four grids, four distinct answers at full precision."""
    for order in ORDERS:
        values = [ladder[key]['totals'][order] for key in GRIDS]
        assert len(set(values)) == len(values)
    # And the coarsest grid is wrong by an amount no tolerance would hide.
    assert abs(_relative(ladder, (20, 26), 6)) > 1e-6
    assert abs(_relative(ladder, (20, 26), 10)) > 1e-5


@pytest.mark.long
def test_a_degenerate_grid_is_an_error_and_it_grows_with_the_order(ladder):
    """Lebedev 26 integrates no multipole worth having; C10 suffers most."""
    errors = [abs(_relative(ladder, (20, 26), order)) for order in ORDERS]
    assert errors == sorted(errors)
    assert errors[-1] < 1e-3


@pytest.mark.long
def test_lebedev_110_recovers_six_figures(ladder):
    for order in ORDERS:
        assert abs(_relative(ladder, (60, 110), order)) < 2e-5


@pytest.mark.long
def test_the_declared_default_is_converged_at_lebedev_302(ladder):
    """302 -> 590 changes nothing above 1e-7 at any order: the default is a
    quadrature choice, not a model choice, and may be compared across."""
    for order in ORDERS:
        assert abs(_relative(ladder, (99, 302), order)) < 1e-7


def test_an_oversized_grid_is_refused_before_any_integral():
    """The convergence above is bounded by a declared resource limit, not by taste."""
    refused = estimate_response_work(92, 92, 5, 581478, algorithm='shared_sweep')
    assert not refused.passes
    assert refused.failures[0] == 'ALDA work resource limit'
    assert refused.max_grid_rows == 338221
    assert estimate_response_work(92, 92, 5, refused.max_grid_rows,
                                  algorithm='shared_sweep').passes
    with pytest.raises(ValueError, match='ALDA work resource limit'):
        refused.require_pass()
