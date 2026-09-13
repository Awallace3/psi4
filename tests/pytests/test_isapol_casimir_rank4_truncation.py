"""Where the isotropic C12 ``INCOMPLETE`` mark comes from, measured not inferred.

Psi4 marks the isotropic C12 of a rank-3 localized model incomplete, reporting
``missing_rank_pairs = [(1, 4), (4, 1)]``.  At order ``n`` the isotropic sum runs
over site ranks with ``la + lb == n/2 - 1``, so C12 needs (1,4), (2,3), (3,2) and
(4,1); a rank-3 model supplies only the middle two.  The open question was whether
that mark named a genuine gap against CamCASP -- something the reference has and
Psi4 throws away -- or a truncation the two share.

It is shared, and this module is the measurement.  A rank-4 localization of the
SAME primary reference response was produced (``localize.py water --limit 4
--hlimit 4 --norefine`` over the run's own ``water_ISA-GRID_f11_NL4_fmtB.pol``)
and decoded into ``camcasp_casimir_h2o_vdz_l4.json`` beside the rank-3 fixture.
Its ``Recoupled polarizabilities`` section shows CASIMIR builds the single-site
table only where ``la + lb <= 6`` at ``Dispersion 12``: (1,4) and (4,1) exist,
(3,4), (4,3) and (4,4) do not.  The isotropic ``L = 0`` components present are
``00(11)``, ``00(22)``, ``00(33)`` and nothing else -- there is no ``00(44)``.
The isotropic C12 term (1,4) needs ``00(44)`` on the partner site, so it is
structurally absent from the reference's own printed coefficient, and the printed
rows do not move between the rank-3 and rank-4 runs.

CASIMIR refuses to lift the truncation.  Rewriting ``Dispersion 12 water`` to 14
in the decoded deck and rerunning the shipped executable exits ``STOP 9`` with
``Dispersion coefficients only up to C12``, so no CamCASP-produced ``00(44)``,
and hence no reference (1,4)/(4,1) contribution, exists to compare against.

The rank-3 and rank-4 localizations are DIFFERENT declared models.  Their printed
isotropic rows are bit-identical, and that identity is asserted below as a fact
about the two files -- it is never a statement that the models agree, and no
number from one is quoted as the other's.  The ``INCOMPLETE`` marking convention
stays exactly as it is: it reports the unrestricted rank sum honestly, and this
measurement does not license absorbing it into a tolerance.
"""
import json
import pathlib

import numpy as np
import pytest

import psi4

c = psi4.core
DATA = pathlib.Path(__file__).parent / "data_isapol"
L3 = json.loads((DATA / "camcasp_casimir_h2o_vdz_l3.json").read_text())
L4 = json.loads((DATA / "camcasp_casimir_h2o_vdz_l4.json").read_text())

#: `Dispersion 12` declared by both runs; CASIMIR's single-site recoupled table
#: is built only for `la + lb <= max_order / 2`.
MAX_ORDER = 12
PAIR_SUM_LIMIT = MAX_ORDER // 2

#: What CASIMIR refuses when the declared order is raised past its compiled cap.
CASIMIR_ORDER_REFUSAL = "Dispersion coefficients only up to C12"

FIRST_SITE_OF_TYPE = {"O O": ("O1", "O1"), "H O": ("O1", "H2"), "H H": ("H2", "H2")}


def _weights():
    grid = c.CasimirGrid(L3["grid"]["frequency_count"], L3["grid"]["beta"])
    return [grid.cp_weight(k) for k in range(grid.n_freq() + 1)]


def _model(reference, ranks, name):
    """A Psi4 isotropic model over `ranks` from one decoded localized file."""
    local = reference["localized"]
    frequencies = [np.sqrt(-f) for f in reference["grid"]["frequencies_squared"]]
    sites = []
    for label in local["site_labels"]:
        by_rank = local["isotropic_by_rank"][label]
        site = c.IsaIsotropicSite()
        site.label, site.origin, site.ranks = label, [0.0, 0.0, 0.0], list(ranks)
        site.polarizabilities = c.Matrix.from_array(
            np.array([by_rank[str(l)] for l in ranks], dtype=float).T)
        sites.append(site)
    return c.IsaIsotropicModel(frequencies, sites, name)


def _c12(reference, ranks, name):
    """`{(a, b): coefficient}` at order 12 for the named declared model."""
    model = _model(reference, ranks, name)
    result = c.isa_isotropic_dispersion(model, model, _weights(), MAX_ORDER)
    return {(result.labels_a[p.site_a], result.labels_b[p.site_b]): x
            for p in result.pairs for x in p.coefficients if x.order == MAX_ORDER}


def test_the_two_fixtures_are_different_declared_models():
    """Rank 3 and rank 4 are separate files, separate prefixes, separate models."""
    assert L3["localized"]["ranks"] == [1, 2, 3]
    assert L4["localized"]["ranks"] == [1, 2, 3, 4]
    assert (L3["prefix"], L4["prefix"]) == ("water_L3", "water_L4")
    assert L3["localized"]["sha256"] != L4["localized"]["sha256"]
    assert L3["dispersion"]["sha256"] != L4["dispersion"]["sha256"]


def test_rank4_run_builds_single_site_pairs_only_up_to_the_order_limit():
    """CASIMIR's recoupled table stops at `la + lb == 6`, with rank 4 available."""
    for label, site in L4["recoupled"]["sites"].items():
        pairs = {tuple(p) for p in site["pairs"]}
        assert pairs == {(la, lb) for la in range(1, 5) for lb in range(1, 5)
                         if la + lb <= PAIR_SUM_LIMIT}, label
        # Rank 4 IS localized and IS recoupled -- just not against ranks 3 or 4.
        assert {(1, 4), (4, 1), (2, 4), (4, 2)} <= pairs, label
        assert pairs.isdisjoint({(3, 4), (4, 3), (4, 4)}), label


def test_no_isotropic_rank4_component_exists_in_either_run():
    """`00(44)` is printed by neither run, so the C12 (1,4) term has no partner.

    The isotropic dispersion at order `n` contracts `00(la la)` on one site with
    `00(lb lb)` on the other.  Only `la == lb` has an `L = 0` component at all,
    and the rank-4 one is past the `la + lb <= 6` cut in both runs.
    """
    for reference in (L3, L4):
        for label, site in reference["recoupled"]["sites"].items():
            assert [tuple(p) for p in site["isotropic_pairs"]] == [(1, 1), (2, 2), (3, 3)], label


def test_printed_isotropic_rows_do_not_move_between_the_two_models():
    """A fact about the two files, NOT a claim that the two models agree.

    Raising the localization limit to 4 changes the recoupled row census in every
    block, so CASIMIR plainly did read the extra rank; it changes no printed
    isotropic coefficient, because nothing rank 4 supplies survives the
    `la + lb <= 6` cut at order 12.
    """
    for block in FIRST_SITE_OF_TYPE:
        l3, l4 = (r["dispersion"]["blocks"][block] for r in (L3, L4))
        assert l3["isotropic"] == l4["isotropic"], block
        assert l4["census"]["rows"] > l3["census"]["rows"], block


def test_the_incomplete_mark_names_pairs_the_reference_also_omits():
    """Psi4's rank-3 C12 gap is exactly the pairs CamCASP never builds either.

    With ranks 1..3 Psi4 reproduces the reference's printed C12 on all three type
    pairs and reports the gap rather than hiding it.  The reported pairs are
    (1,4) and (4,1) -- both of which need the `00(44)` component that neither
    reference run prints.  So the mark is a shared protocol truncation, not a
    native-versus-CamCASP discrepancy, and it is still correctly a mark: the
    unrestricted rank sum genuinely does include a term this model lacks.
    """
    coefficients = _c12(L3, [1, 2, 3], "camcasp_reference_local_l3")
    for block, key in FIRST_SITE_OF_TYPE.items():
        coefficient = coefficients[key]
        assert coefficient.complete is False, block
        assert sorted(map(tuple, coefficient.missing_rank_pairs)) == [(1, 4), (4, 1)], block
        printed = L3["dispersion"]["blocks"][block]["isotropic"][str(MAX_ORDER)]
        assert coefficient.value == pytest.approx(printed, rel=1e-6), block
        # Every missing pair needs `00(l l)` at a rank the reference never builds.
        for la, lb in map(tuple, coefficient.missing_rank_pairs):
            assert la + lb == MAX_ORDER // 2 - 1
            assert max(la, lb) == 4


def test_supplying_rank_four_completes_the_sum_but_declares_another_model():
    """Feeding the rank-4 file's own rank 4 closes the mark and moves C12.

    This is the proof the truncation is not vacuous -- the missing term is large.
    It is also, by the same token, a DIFFERENT declared model from anything
    CamCASP printed: these numbers correspond to no reference coefficient, and
    the H-H value is negative, because an unrefined rank-4 LW site tensor is
    indefinite.  They are recorded here to bound the truncation, never quoted as
    a reference comparison.
    """
    truncated = _c12(L4, [1, 2, 3], "camcasp_reference_local_l4_ranks123")
    completed = _c12(L4, [1, 2, 3, 4], "camcasp_reference_local_l4_ranks1234")
    for key in FIRST_SITE_OF_TYPE.values():
        assert truncated[key].complete is False
        assert completed[key].complete is True
        assert sorted(map(tuple, completed[key].missing_rank_pairs)) == []
    assert completed[("O1", "O1")].value == pytest.approx(4427.26876306, rel=1e-9)
    assert completed[("O1", "H2")].value == pytest.approx(85.44334043, rel=1e-9)
    assert completed[("H2", "H2")].value == pytest.approx(-49.03028880, rel=1e-9)
    # Ranks 1..3 of the rank-4 file reproduce the rank-3 file exactly, which is
    # what makes the pair above a controlled comparison rather than two runs.
    for key in FIRST_SITE_OF_TYPE.values():
        assert truncated[key].value == pytest.approx(
            _c12(L3, [1, 2, 3], "camcasp_reference_local_l3")[key].value, rel=1e-12)


def test_the_reference_executable_caps_the_table_at_c12():
    """The closure route is shut: CASIMIR will not compute past C12.

    Recorded as a constant rather than re-run, because running it needs the
    external CamCASP build.  Raising `Dispersion 12 water` to 14, 16 or 18 in the
    decoded deck exits `STOP 9` with this diagnostic, which is why no reference
    `00(44)` -- and so no reference (1,4)/(4,1) term -- can be obtained at all.
    """
    assert CASIMIR_ORDER_REFUSAL == "Dispersion coefficients only up to C12"
    assert max(L3["dispersion"]["blocks"]["O O"]["census"]["orders"]) == MAX_ORDER
    assert max(L4["dispersion"]["blocks"]["O O"]["census"]["orders"]) == MAX_ORDER
