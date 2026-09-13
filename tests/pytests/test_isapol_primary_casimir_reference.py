"""The PRIMARY reference's printed C_n table, closed against Psi4's own assembly.

The acceptance target of this chain is the shipped `tests/H2O_props/psi4`
declaration (`SCFcode psi4`, cc-pVDZ, PBE0, `AC NONE`, ALDA+CHF, constrained-NN
DF, 100x400 grid).  Its CASIMIR stage prints one dispersion block per site TYPE
pair, whose `00 00 0` row is the isotropic site-site C_n.

Two reference quantities are involved and they stay separate here:

  PRINTED     the `00 00 0` row of `water_L3_casimir.out`;
  INTEGRATED  the Casimir-Polder integral of `water_L3_0f10.pol`, the localized
              file that same stage read, under Psi4's isotropic assembly.

Their agreement is MEASURED below, not assumed -- and the measurement is the
point of the module.  An earlier claim in this branch's history (commit
eb49a236ea) held that the printed C6/C8/C10 came from some other reference path
than the integral, on the strength of three constants, 8.400508 / 81.842910 /
700.674100, that exist in no CamCASP file anywhere.  They were a transcription
error.  `test_withdrawn_constants_are_not_reference_data` keeps them from coming
back; the rest of the module records what the reference actually prints.

Nothing here runs a native chain: the native model's recorded coefficients are
read from committed evidence, because reproducing them needs the full 128898-
point response.  Recomputing them is the job of the long-marked chains in
test_isapol_reference_dispersion.py.
"""
import json
import pathlib

import numpy as np
import pytest

import psi4

c = psi4.core
DATA = pathlib.Path(__file__).parent / "data_isapol"
REFERENCE = json.loads((DATA / "camcasp_casimir_h2o_vdz_l3.json").read_text())
EVIDENCE = json.loads((DATA / "psi4_primary_casimir_residual_evidence.json").read_text())

#: CASIMIR prints one block per site TYPE.  With two equivalent hydrogens that
#: leaves which site the block stands for undetermined by the file; the answer,
#: measured in test_printed_blocks_name_the_first_site_of_each_type, is the
#: FIRST site of each type in declaration order.
FIRST_SITE_OF_TYPE = {"O O": ("O1", "O1"), "H O": ("O1", "H2"), "H H": ("H2", "H2")}

#: Orders the isotropic row prints nonzero for this rank-3 localized model.
EVEN_ORDERS = (6, 8, 10, 12)


def _weights():
    """CP weights from Psi4's own quadrature at the reference's declared Beta/N."""
    grid = c.CasimirGrid(REFERENCE["grid"]["frequency_count"], REFERENCE["grid"]["beta"])
    return [grid.cp_weight(k) for k in range(grid.n_freq() + 1)]


def _frequencies():
    """The `.pol` file stores FREQSQ on the imaginary axis, i.e. -omega**2."""
    return [np.sqrt(-f) for f in REFERENCE["grid"]["frequencies_squared"]]


def _model():
    """CamCASP's own localized rank isotropics, as a Psi4 isotropic model."""
    local = REFERENCE["localized"]
    ranks = local["ranks"]
    sites = []
    for label in local["site_labels"]:
        by_rank = local["isotropic_by_rank"][label]
        site = c.IsaIsotropicSite()
        site.label, site.origin, site.ranks = label, [0.0, 0.0, 0.0], list(ranks)
        site.polarizabilities = c.Matrix.from_array(
            np.array([by_rank[str(l)] for l in ranks], dtype=float).T)
        sites.append(site)
    return c.IsaIsotropicModel(_frequencies(), sites, "camcasp_reference_local_l3")


@pytest.fixture(scope="module")
def integrated():
    """`{(site_a, site_b): {order: C_n}}` from the reference's own localized file."""
    model = _model()
    result = c.isa_isotropic_dispersion(model, model, _weights())
    return {(result.labels_a[p.site_a], result.labels_b[p.site_b]):
            {x.order: x.value for x in p.coefficients} for p in result.pairs}


def _printed(block):
    return {n: REFERENCE["dispersion"]["blocks"][block]["isotropic"][str(n)] for n in EVEN_ORDERS}


def test_frequencies_match_the_declared_quadrature():
    """The localized file's FREQSQ column is the declared Beta 0.5, 10-point grid.

    FREQSQ is printed to seven significant figures, truncated at seven decimals,
    so the comparison is made in FREQSQ space against that printing precision
    (5e-7 relative or 5e-8 absolute, whichever is looser).  The first dynamic
    node prints as -0.0000437 -- three significant figures; taking its square
    root first and comparing omega would compare against a rounding artefact and
    fail at 3e-4 for reasons that have nothing to do with the quadrature.
    """
    grid = c.CasimirGrid(REFERENCE["grid"]["frequency_count"], REFERENCE["grid"]["beta"])
    printed = REFERENCE["grid"]["frequencies_squared"]
    assert len(printed) == grid.n_freq() + 1
    for k, freqsq in enumerate(printed):
        assert -freqsq == pytest.approx(grid.omega(k) ** 2, rel=5e-7, abs=5e-8), k
    assert REFERENCE["dispersion"]["declared"]["Frequencies"] == "0.5 10"
    assert REFERENCE["dispersion"]["declared"]["Skip"] == "0"


def test_isotropic_row_is_zero_at_every_odd_order():
    """Rank 1..3 isotropics can contribute to even orders only."""
    for block in FIRST_SITE_OF_TYPE:
        row = REFERENCE["dispersion"]["blocks"][block]["isotropic"]
        assert [row[str(n)] for n in (7, 9, 11)] == [0.0, 0.0, 0.0]
        assert all(row[str(n)] > 0.0 for n in EVEN_ORDERS)


@pytest.mark.parametrize("block", sorted(FIRST_SITE_OF_TYPE))
def test_printed_row_is_the_casimir_integral_of_its_own_localized_file(block, integrated):
    """PRINTED == INTEGRATED, on every even order, for the primary reference.

    This is the measurement that dissolves the supposed C6/C8/C10 discrepancy:
    the reference has ONE dispersion path here, not two.  The two columns are
    still distinct quantities -- one is CamCASP's published number, the other is
    what CamCASP's own last localized intermediate implies under Psi4's assembly
    -- and 1e-6 is the printed table's own 7-digit resolution, not a tolerance
    absorbing a model difference.
    """
    computed = integrated[FIRST_SITE_OF_TYPE[block]]
    for order, printed in _printed(block).items():
        assert computed[order] == pytest.approx(printed, rel=1e-6), order


def test_printed_blocks_name_the_first_site_of_each_type(integrated):
    """Measured, not assumed: the H blocks are H2's, not H3's.

    The reference's two hydrogens are equivalent by symmetry but its localized
    file is not symmetric in them to better than 1.8e-4, so the choice is
    resolvable from the numbers alone.  That residual asymmetry is also the
    scale against which this reference's own H-containing rows should be read.
    """
    worst = {}
    for block, first in FIRST_SITE_OF_TYPE.items():
        printed = _printed(block)
        for pair, computed in integrated.items():
            # Unordered: the assembly is symmetric under exchange, so ('H2','O1')
            # is the same candidate as ('O1','H2') and is not a rival assignment.
            if sorted(l[0] for l in pair) != sorted(l[0] for l in first):
                continue
            worst.setdefault(block, {})[frozenset(pair)] = max(
                abs(computed[n] / printed[n] - 1.0) for n in EVEN_ORDERS)
    for block, first in FIRST_SITE_OF_TYPE.items():
        best = worst[block][frozenset(first)]
        assert best < 1e-6
        for pair, value in worst[block].items():
            if pair != frozenset(first):
                assert value > 10 * best, (block, sorted(pair))
    assert worst["H H"][frozenset(("H3", "H3"))] > 1e-4


def test_recorded_native_residual_against_the_printed_reference():
    """The declared native model reproduces the printed row; record by how much.

    Bounds are the measured residuals rounded up, so a regression moves them.
    O-O is closed to 3.1e-5 at every even order; the H-containing rows close to
    2.1e-4, which is the same order as the reference's OWN H2/H3 asymmetry
    measured above.  No claim is made that the two are the same effect.
    """
    native = EVIDENCE["isotropic_cn"]
    assert EVIDENCE["model"].endswith("DFPROP_ISA_A")
    worst = {}
    for block, (a, b) in FIRST_SITE_OF_TYPE.items():
        printed = _printed(block)
        row = native["%s-%s" % (a, b)]
        worst[block] = max(abs(row[str(n)] / printed[n] - 1.0) for n in EVEN_ORDERS)
    assert worst["O O"] < 3.2e-5
    assert worst["H O"] < 1.0e-4
    assert worst["H H"] < 2.1e-4


def test_withdrawn_constants_are_not_reference_data():
    """8.400508 / 81.842910 / 700.674100 are in no reference file; keep it that way."""
    withdrawn = EVIDENCE["withdrawn_constants"]
    printed = _printed("O O")
    for order, bad in ((6, withdrawn["C6"]), (8, withdrawn["C8"]), (10, withdrawn["C10"])):
        assert printed[order] != pytest.approx(bad, rel=1e-4), order
    assert "8.400508" not in (DATA / "camcasp_casimir_h2o_vdz_l3.json").read_text()
