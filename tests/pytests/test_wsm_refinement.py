"""Physical WSM L3 design/refinement tests with independent NumPy oracles."""

from pathlib import Path
import json
import re

import numpy as np
import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.mints]


def _matrix(values, columns=None):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1 and values.size == 0:
        values = np.empty((0, columns), dtype=float)
    return psi4.core.Matrix.from_array(values)


def _regular(d):
    x, y, z = np.asarray(d, dtype=float)
    r2 = x*x + y*y + z*z
    return np.array([
        1., z, x, y,
        (3*z*z-r2)/2, np.sqrt(3)*x*z, np.sqrt(3)*y*z,
        np.sqrt(3)*(x*x-y*y)/2, np.sqrt(3)*x*y,
        (5*z**3-3*z*r2)/2,
        np.sqrt(3/8)*x*(5*z*z-r2), np.sqrt(3/8)*y*(5*z*z-r2),
        np.sqrt(15)*z*(x*x-y*y)/2, np.sqrt(15)*x*y*z,
        np.sqrt(10)*x*(x*x-3*y*y)/4, np.sqrt(10)*y*(3*x*x-y*y)/4,
    ])


def _irregular(point, site=(0., 0., 0.)):
    d = np.asarray(point) - np.asarray(site)
    r2 = d @ d
    regular = _regular(d)
    out = []
    for rank, begin in ((1, 1), (2, 4), (3, 9)):
        out.extend(regular[begin:(rank+1)**2] / r2**(rank + .5))
    return np.asarray(out)


def _upper_index(t, u):
    assert 0 <= t <= u < 15
    return t * 15 - t * (t - 1) // 2 + (u - t)


def _design(points, sites, active):
    irregular = np.array([[_irregular(p, s) for s in sites] for p in points])
    rows = []
    for g in range(len(points)):
        for h in range(g, len(points)):
            row = []
            for site in range(len(sites)):
                for t in range(15):
                    for u in range(t, 15):
                        value = irregular[g, site, t] * irregular[h, site, u]
                        if t != u:
                            value += irregular[g, site, u] * irregular[h, site, t]
                        row.append(value)
            rows.append(np.asarray(row)[active])
    return np.asarray(rows)


def _response(points, sites, tensors):
    irregular = np.array([[_irregular(p, s) for s in sites] for p in points])
    result = np.zeros((len(points), len(points)))
    for g in range(len(points)):
        for h in range(len(points)):
            result[g, h] = sum(irregular[g, a] @ tensors[a] @ irregular[h, a]
                               for a in range(len(sites)))
    return result


def _refine(points, sites, response, *, localized=None, localized_frequency_major=None,
            active=None, equality=None, targets=None, frequencies=(0.,),
            localized_frequencies=None, options=None):
    nvar = 120 * len(sites)
    if localized is None:
        localized = [np.zeros((15, 15)) for _ in sites]
    if active is None:
        active = [True] * nvar
    if equality is None:
        equality = np.empty((0, nvar))
    if targets is None:
        targets = []
    matrices = response if isinstance(response, (list, tuple)) else [response]
    if localized_frequencies is None:
        localized_frequencies = frequencies
    if localized_frequency_major is None:
        localized_frequency_major = [localized for _ in frequencies]
    flat_localized = [matrix for frequency_blocks in localized_frequency_major
                      for matrix in frequency_blocks]
    return psi4.core._atomic_polarizability_test_refine_wsm(
        _matrix(points), list(frequencies), [_matrix(x) for x in matrices],
        _matrix(sites), [_matrix(x) for x in flat_localized], list(localized_frequencies), list(active),
        _matrix(equality, nvar), list(targets), {} if options is None else options,
    )


def test_wsm_part2_primary_reference_fixture():
    """Pin Part 2 eq. (2), Table 1, response gates, and the grid caveat."""
    reference = json.loads(
        Path(__file__).with_name("wsm_part2_reference.json").read_text(encoding="utf-8"))

    assert reference["schema"] == 1
    assert reference["source"]["doi"] == "10.1021/ct700105f"
    equation = reference["equation_2"]
    assert equation["off_diagonal_strength"] == 0.0
    assert equation["dipole_components"] == ["10", "11c", "11s"]
    assert equation["dipole_block_diagonal_strength"] == 1e-5
    assert equation["all_other_diagonal_strength"] == 0.0

    gates = reference["response_quality_gates_percent_of_range"]
    assert gates == {
        "comparison": "strictly_less_than",
        "rank_1": {"maximum": 6.0, "rms": 0.2},
        "rank_2": {"maximum": 2.0, "rms": 0.05},
    }
    table = reference["table_1_wsm_energy_map_differences"]
    assert table["columns"] == ["Formamide", "N-MPA", "Benzene", "BOQQUT"]
    assert table["rows"]["L1,WSM"] == {
        "maximum": [3.48, 2.32, 2.98, 2.48], "rms": [1.35, 0.74, 1.58, 0.82]}
    assert table["rows"]["L2,WSM"] == {
        "maximum": [1.14, 1.29, 0.69, 2.27], "rms": [0.27, 0.21, 0.25, 0.23]}
    assert table["rows"]["L2/L1,WSM"] == {
        "maximum": [1.18, 0.99, 0.67, 1.48], "rms": [0.34, 0.34, 0.42, 0.32]}
    assert reference["recommended_model"] == {
        "heavy_atom_rank": 2, "hydrogen_rank": 1, "label": "L2/L1,WSM"}
    assert reference["grid_caveat"]["system"] == "BOQQUT"
    assert reference["grid_caveat"]["point_count"] == 2000


def test_wsm_part2_equation_2_rank1_only_penalty_matches_numpy_oracle():
    """Part 2 eq. (2) anchors the full dipole block and no higher-rank variable."""
    rng = np.random.default_rng(2008)
    points = rng.normal(size=(9, 3)) * 1.7 + [.8, -.4, 1.1]
    sites = [[0., 0., 0.]]
    pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2), (3, 3)]
    active = np.zeros(120, dtype=bool)
    active[[_upper_index(first, second) for first, second in pairs]] = True

    truth = np.zeros((15, 15))
    anchor = np.zeros((15, 15))
    for index, (first, second) in enumerate(pairs):
        truth[first, second] = truth[second, first] = .4 + .11 * index
        anchor[first, second] = anchor[second, first] = 1.8 - .09 * index
    response = _response(points, sites, [truth])
    response[np.triu_indices(len(points))] += rng.normal(
        scale=.03, size=len(points) * (len(points) + 1) // 2)
    response = np.triu(response) + np.triu(response, 1).T

    result = _refine(
        points, sites, response, localized=[anchor], active=active,
        options={"weight_coefficient": 1e-5, "anchor_rank_limit": 1})[0]
    design = _design(points, sites, active)
    upper = np.triu_indices(len(points))
    observations = response[upper]
    row_weights = np.where(upper[0] == upper[1], 1.0, np.sqrt(2.0))
    penalty = np.zeros((6, len(pairs)))
    penalty[:, :6] = np.eye(6)
    augmented = np.vstack((row_weights[:, None] * design, np.sqrt(1e-5) * penalty))
    targets = np.r_[row_weights * observations,
                    np.sqrt(1e-5) * np.array([anchor[first, second]
                                              for first, second in pairs[:6]])]
    oracle = np.linalg.lstsq(augmented, targets, rcond=1e-12)[0]

    assert np.asarray(result["solution"])[active] == pytest.approx(oracle, abs=2e-11)
    assert result["anchor_variable_count"] == 6
    assert result["policy"]["weight_coefficient"] == 1e-5
    assert result["policy"]["anchor_rank_limit"] == 1


def test_irregular_harmonics_axis_sentinels_and_laplace_l3_convergence():
    got = np.asarray(psi4.core._atomic_polarizability_test_irregular_harmonics(
        [0., 0., 2.], [0., 0., 0.]))
    assert got[0] == pytest.approx(1/2, abs=2e-15)
    assert got[1:] == pytest.approx(_irregular([0., 0., 2.]), abs=2e-15)
    assert got[[2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]] == pytest.approx(0., abs=1e-15)
    assert got[[1, 4, 9]] == pytest.approx([1/4, 1/8, 1/16], abs=2e-15)

    source = np.array([0.07, -0.04, 0.02])
    external = np.array([2.3, -1.7, 3.1])
    exact = 1 / np.linalg.norm(external-source)
    regular = _regular(source)
    irregular = np.r_[1/np.linalg.norm(external), _irregular(external)]
    errors = [abs(regular[:(rank+1)**2] @ irregular[:(rank+1)**2] - exact)
              for rank in range(4)]
    assert all(a > b for a, b in zip(errors, errors[1:]))
    assert errors[-1] < 2e-7


def test_exact_one_site_l1_reconstruction_and_offdiagonal_factor():
    points = [[2.1, .2, -.3], [-1.7, 2.2, .5], [.4, -2.4, 1.3],
              [1.5, 1.6, -1.8], [-2.2, -.7, -1.4]]
    sites = [[0., 0., 0.]]
    tensor = np.zeros((15, 15))
    tensor[:3, :3] = [[1.2, -.35, .22], [-.35, 2.1, .4], [.22, .4, .8]]
    active = np.zeros(120, dtype=bool)
    for t in range(3):
        for u in range(t, 3):
            active[_upper_index(t, u)] = True
    response = _response(points, sites, [tensor])
    result = _refine(points, sites, response, localized=[tensor], active=active)
    assert np.asarray(result[0]["tensors"])[0] == pytest.approx(tensor, abs=3e-12)

    design = _design(points, sites, active)
    offdiag_column = list(np.flatnonzero(active)).index(_upper_index(0, 1))
    expected = (_irregular(points[0])[0] * _irregular(points[1])[1] +
                _irregular(points[0])[1] * _irregular(points[1])[0])
    assert design[1, offdiag_column] == pytest.approx(expected, abs=1e-15)
    assert result[0]["max_point_residual"] < 2e-13
    assert result[0]["max_output_asymmetry"] == 0.


def test_selected_l3_two_site_reconstruction_and_numpy_lstsq_oracle():
    rng = np.random.default_rng(441)
    sites = np.array([[-.7, .1, .2], [.8, -.2, -.1]])
    points = rng.normal(size=(12, 3)) * .5 + [0., 0., 1.2]
    active = np.zeros(240, dtype=bool)
    selected = [(0, 0), (0, 4), (2, 9), (5, 5), (8, 14), (12, 12)]
    expected = []
    tensors = []
    for site in range(2):
        tensor = np.zeros((15, 15))
        for index, (t, u) in enumerate(selected):
            value = (-1)**(site+index) * (.2 + .07*index + .1*site)
            tensor[t, u] = tensor[u, t] = value
            active[site*120 + _upper_index(t, u)] = True
            expected.append(value)
        tensors.append(tensor)
    response = _response(points, sites, tensors)
    design = _design(points, sites, active)
    observations = response[np.triu_indices(len(points))]
    oracle = np.linalg.lstsq(design, observations, rcond=1e-12)[0]
    assert oracle == pytest.approx(expected, abs=2e-11)
    result = _refine(points, sites, response, localized=tensors, active=active)
    assert np.asarray(result[0]["solution"])[active] == pytest.approx(oracle, abs=4e-10)
    assert np.asarray(result[0]["tensors"]) == pytest.approx(np.asarray(tensors), abs=4e-10)


def test_noisy_weight4_dipole_diagonal_anchor_is_applied_and_reported():
    rng = np.random.default_rng(82)
    points = rng.normal(size=(8, 3)) * 2 + [1., -.5, .7]
    sites = [[0., 0., 0.]]
    active = np.zeros(120, dtype=bool)
    active[[_upper_index(i, i) for i in range(3)]] = True
    true = np.zeros((15, 15)); true[0, 0], true[1, 1], true[2, 2] = 2., 3., 4.
    reference = np.zeros((15, 15)); reference[0, 0], reference[1, 1], reference[2, 2] = 8., 7., 6.
    noisy = _response(points, sites, [true])
    noisy[np.triu_indices(len(points))] += rng.normal(scale=.08, size=len(points)*(len(points)+1)//2)
    noisy = np.triu(noisy) + np.triu(noisy, 1).T
    result = _refine(points, sites, noisy, localized=[reference], active=active)[0]
    design = _design(points, sites, active)
    obs = noisy[np.triu_indices(len(points))]
    pair_weights = np.array([1. if g == h else np.sqrt(2.)
                             for g in range(len(points)) for h in range(g, len(points))])
    augmented = np.vstack((pair_weights[:, None]*design, np.sqrt(.001)*np.eye(3)))
    target = np.r_[pair_weights*obs, np.sqrt(.001)*np.diag(reference)[:3]]
    oracle = np.linalg.lstsq(augmented, target, rcond=1e-12)[0]
    unanchored = np.linalg.lstsq(pair_weights[:, None]*design, pair_weights*obs, rcond=1e-12)[0]
    fitted = np.asarray(result["solution"])[active]
    assert fitted == pytest.approx(oracle, abs=2e-11)
    assert np.linalg.norm(fitted-unanchored) > 1e-8
    assert np.linalg.norm(fitted-oracle) < np.linalg.norm(unanchored-oracle)

    unique = _refine(points, sites, noisy, localized=[reference], active=active,
                     options={"row_weight_policy": "unique_pair_equal"})[0]
    unique_augmented = np.vstack((design, np.sqrt(.001)*np.eye(3)))
    unique_target = np.r_[obs, np.sqrt(.001)*np.diag(reference)[:3]]
    unique_oracle = np.linalg.lstsq(unique_augmented, unique_target, rcond=1e-12)[0]
    assert np.asarray(unique["solution"])[active] == pytest.approx(unique_oracle, abs=2e-11)
    assert np.linalg.norm(unique_oracle-oracle) > 1e-5
    assert unique["row_weight_source"] == "unique_pair_equal"
    assert unique["policy"]["row_weight_policy"] == "unique_pair_equal"

    assert result["policy"] == {
        "wsm_rank": 3, "hydrogen_rank": 3, "weight_type": 4,
        "weight_coefficient": .001, "anchor_rank_limit": 1, "cutoff": 1e-4,
        "row_weight_policy": "full_symmetric_frobenius",
        "normalize_copy_penalties": True,
        "weight_type_definition": "weight type 4: anchor each symmetric block whose two component ranks are at or below anchor_rank_limit",
        "column_pruning_definition": "relative weighted-column-norm threshold",
        "external_oracle_parity_claimed": False,
    }
    assert result["anchor_variable_count"] == 3
    assert result["row_weight_source"] == "full_symmetric_frobenius"


def test_h2_copy_equality_active_zeros_cutoff_and_kkt_oracle():
    rng = np.random.default_rng(9)
    sites = [[-.8, 0., 0.], [.8, 0., 0.]]
    points = rng.normal(size=(10, 3))*2.3 + [.2, .4, -.1]
    active = np.zeros(240, dtype=bool)
    first = _upper_index(0, 0)
    second = 120 + _upper_index(0, 0)
    active[[first, second]] = True
    equality = np.zeros((1, 240)); equality[0, first], equality[0, second] = 1., -1.
    tensors = [np.zeros((15, 15)), np.zeros((15, 15))]
    tensors[0][0, 0] = tensors[1][0, 0] = 1.7
    response = _response(points, sites, tensors)
    result = _refine(points, sites, response, active=active,
                     equality=equality, targets=[0.])[0]
    design = _design(points, sites, active)
    observations = response[np.triu_indices(len(points))]
    pair_weights = np.array([1. if g == h else np.sqrt(2.)
                             for g in range(len(points)) for h in range(g, len(points))])
    weighted_design = pair_weights[:, None]*design
    weighted_observations = pair_weights*observations
    # H2 is a signed/equal copy of H1 and PFIT carries one independent parameter and one
    # penalty. In the expanded constrained basis each copy therefore gets half the squared
    # penalty weight (anchor row 1/sqrt(2)), not a duplicated 0.001 penalty.
    hessian = weighted_design.T @ weighted_design + .0005*np.eye(2)
    kkt = np.block([[hessian, np.array([[1.], [-1.]])],
                    [np.array([[1., -1.]]), np.zeros((1, 1))]])
    oracle = np.linalg.solve(kkt, np.r_[weighted_design.T @ weighted_observations, 0.])[:2]
    assert np.asarray(result["solution"])[active] == pytest.approx(oracle, abs=3e-12)
    assert result["anchor_variable_count"] == 1

    duplicated = _refine(points, sites, response, active=active,
                          equality=equality, targets=[0.],
                          options={"normalize_copy_penalties": False})[0]
    duplicated_hessian = weighted_design.T @ weighted_design + .001*np.eye(2)
    duplicated_kkt = np.block([[duplicated_hessian, np.array([[1.], [-1.]])],
                               [np.array([[1., -1.]]), np.zeros((1, 1))]])
    duplicated_oracle = np.linalg.solve(
        duplicated_kkt, np.r_[weighted_design.T @ weighted_observations, 0.])[:2]
    assert np.asarray(duplicated["solution"])[active] == pytest.approx(
        duplicated_oracle, abs=3e-12)
    assert duplicated["anchor_variable_count"] == 2
    assert result["solution"][first] == pytest.approx(result["solution"][second], abs=2e-13)
    assert result["constraint_residual_norm"] < 2e-13
    assert np.count_nonzero(np.asarray(result["solution"])[~active]) == 0

    inactive_nan = equality.copy()
    inactive_nan[0, 5] = np.nan
    with pytest.raises(Exception, match="finite"):
        _refine(points, sites, response, active=active,
                equality=inactive_nan, targets=[0.])

    # The 1e-4 policy cutoff is RELATIVE to the largest weighted design-column norm, so
    # it prunes on rank rather than on absolute magnitude. Variable 119 is the (33s, 33s)
    # rank-3 diagonal, whose column falls off as r^-14 against the r^-6 of the rank-1
    # variables; pushing the points out by 4x takes its relative norm to 3e-5.
    tiny_active = active.copy(); tiny_active[119] = True
    cutoff_points = np.asarray(points) * 4.
    cutoff_response = _response(cutoff_points, sites, tensors)
    pruned = _refine(cutoff_points, sites, cutoff_response, active=tiny_active,
                     equality=equality, targets=[0.])[0]
    assert 119 in pruned["pruned_variables"]
    assert pruned["solution"][119] == 0.
    assert pruned["applied_column_cutoff"] == pytest.approx(
        1e-4 * pruned["maximum_weighted_column_norm"], rel=1e-14)

    # Scale invariance is the point of the relative reading: uniformly shrinking every
    # column norm must not change which columns survive. An absolute cutoff failed this,
    # and that failure is what previously forced the fit points inside the density.
    for factor in (1., 1.5, 2.):
        scaled_points = np.asarray(points) * factor
        scaled = _refine(scaled_points, sites, _response(scaled_points, sites, tensors),
                         active=tiny_active, equality=equality, targets=[0.])[0]
        assert scaled["pruned_variables"] == []
        assert scaled["applied_column_cutoff"] == pytest.approx(
            1e-4 * scaled["maximum_weighted_column_norm"], rel=1e-14)

    # CamCASP's reference PFIT ledger says "SVD off" and retains every active variable.
    # cutoff=0 is the explicit no-pre-pruning policy: even the deliberately tiny rank-3
    # column survives, while the same constrained augmented objective is solved.
    unpruned = _refine(cutoff_points, sites, cutoff_response, active=tiny_active,
                       equality=equality, targets=[0.], options={"cutoff": 0.})[0]
    assert unpruned["pruned_variables"] == []
    assert set(unpruned["kept_variables"]) == set(np.flatnonzero(tiny_active))
    assert unpruned["applied_column_cutoff"] == 0.
    assert unpruned["policy"]["cutoff"] == 0.
    assert unpruned["policy"]["column_pruning_definition"] == "disabled (SVD-off no-pre-pruning parity)"


def test_the_column_pruning_keyword_defaults_to_the_reviewed_relative_policy():
    """The production driver reaches both policies only through this keyword.

    RefinementOptions::cutoff has always documented zero as the reference PFIT ledger's
    "SVD off" policy, and validate_wsm_policy has always accepted it, but the driver
    built RefinementOptions{} and overrode only the condition number, so zero was
    reachable from the test-only binding alone. The keyword closes that, and its default
    must stay the reviewed relative threshold.
    """
    import psi4

    try:
        assert psi4.core.get_global_option("ATOMIC_POLARIZABILITY_WSM_COLUMN_PRUNING") == "RELATIVE"
        # Lowercase input proves the choice list normalizes rather than rejecting.
        psi4.set_options({"atomic_polarizability_wsm_column_pruning": "off"})
        assert psi4.core.get_global_option("ATOMIC_POLARIZABILITY_WSM_COLUMN_PRUNING") == "OFF"
        with pytest.raises(Exception):
            psi4.set_options({"atomic_polarizability_wsm_column_pruning": "SOMETIMES"})
    finally:
        psi4.core.clean_options()


def test_point_site_permutation_covariance_and_frequency_major_wrapper():
    rng = np.random.default_rng(27)
    sites = np.array([[-.5, .2, 0.], [.9, -.1, .3]])
    points = rng.normal(size=(10, 3))*2.7
    active = np.zeros(240, dtype=bool)
    tensors = []
    for site in range(2):
        tensor = np.zeros((15, 15)); tensor[site, site] = 1.1 + site
        tensors.append(tensor); active[site*120 + _upper_index(site, site)] = True
    response = _response(points, sites, tensors)
    scaled_tensors = [1.3*tensor for tensor in tensors]
    frequency_anchors = [tensors, scaled_tensors]
    original = _refine(points, sites, [response, 1.3*response],
                       localized=tensors, localized_frequency_major=frequency_anchors,
                       active=active, frequencies=[0., .7])
    with pytest.raises(Exception, match="frequencies do not align"):
        _refine(points, sites, [response, 1.3*response], localized=tensors,
                localized_frequency_major=frequency_anchors[::-1], active=active,
                frequencies=[0., .7], localized_frequencies=[.7, 0.])
    order = rng.permutation(len(points))
    permuted = _refine(points[order], sites, [response[np.ix_(order, order)], 1.3*response[np.ix_(order, order)]],
                       localized=tensors, localized_frequency_major=frequency_anchors,
                       active=active, frequencies=[0., .7])
    assert [item["frequency"] for item in original] == [0., .7]
    assert np.asarray(permuted[0]["tensors"]) == pytest.approx(np.asarray(original[0]["tensors"]), abs=2e-11)

    swapped_active = np.r_[active[120:], active[:120]]
    swapped = _refine(points, sites[::-1], response, localized=tensors[::-1], active=swapped_active)[0]
    assert np.asarray(swapped["tensors"])[::-1] == pytest.approx(np.asarray(original[0]["tensors"]), abs=2e-11)


def test_fail_closed_singular_near_site_nonfinite_frequency_mismatch_and_policy():
    sites = [[0., 0., 0.]]
    points = [[2., 0., 0.], [0., 2., 0.]]
    response = np.eye(2)
    active = np.zeros(120, dtype=bool); active[_upper_index(0, 0)] = True
    with pytest.raises(Exception, match="near a refinement site"):
        _refine([[0., 0., 0.], [2., 0., 0.]], sites, response, active=active)
    with pytest.raises(Exception, match="finite"):
        _refine([[float("nan"), 0., 0.], [2., 0., 0.]], sites, response, active=active)
    with pytest.raises(Exception, match="finite"):
        nonfinite = response.copy(); nonfinite[0, 0] = np.inf
        _refine(points, sites, nonfinite, active=active)
    with pytest.raises(Exception, match="symmetric"):
        nonsymmetric = response.copy(); nonsymmetric[0, 1] = .2
        _refine(points, sites, nonsymmetric, active=active)
    with pytest.raises(Exception, match="frequency-major"):
        _refine(points, sites, [response], active=active, frequencies=[0., 1.])
    with pytest.raises(Exception, match="policy"):
        _refine(points, sites, response, active=active, options={"weight_type": 3})
    for unsupported_cutoff in (-1e-4, 1e-5, float("nan")):
        with pytest.raises(Exception, match="policy"):
            _refine(points, sites, response, active=active,
                    options={"cutoff": unsupported_cutoff})
    for unsupported_anchor_limit in (0, 4):
        with pytest.raises(Exception, match="anchor rank limit"):
            _refine(points, sites, response, active=active,
                    options={"anchor_rank_limit": unsupported_anchor_limit})
    with pytest.raises(Exception, match="row weight policy"):
        _refine(points, sites, response, active=active,
                options={"row_weight_policy": "unknown"})
    # The anchor penalty weight (Stone eqn 9.3.13 g_pp') is free. Zero recovers ordinary
    # least squares and is bounded, but this hybrid production policy deliberately requires
    # a positive anchor; non-finite and negative values are invalid independently.
    for bad_weight in (0., -1e-3, float("nan")):
        with pytest.raises(Exception, match="anchor penalty weight"):
            _refine(points, sites, response, active=active,
                    options={"weight_coefficient": bad_weight})
    condition_active = np.zeros(120, dtype=bool)
    condition_active[[_upper_index(3, 3), _upper_index(4, 4)]] = True
    with pytest.raises(Exception, match="condition number"):
        _refine([[1.3, .4, 1.8], [-1.2, 1.7, .8], [.6, -1.5, 2.1]],
                sites, np.eye(3), active=condition_active,
                options={"maximum_condition_number": 1.0001})
    with pytest.raises(Exception, match="condition|rank deficient"):
        bad = np.zeros(120, dtype=bool)
        bad[[_upper_index(3, 3), _upper_index(6, 6)]] = True
        _refine([[2., 0., 2.], [3., 0., 3.]], sites, response, active=bad)


def test_resource_envelope_and_half_memory_gate_precede_dense_allocation():
    plan = psi4.core._atomic_polarizability_plan_wsm_refinement(500, 3, 360, 0, 1 << 40)
    assert plan["pair_rows"] == 500*501//2
    assert plan["variable_count"] == 360
    assert plan["design_bytes"] == plan["pair_rows"]*360*8
    assert plan["design_bytes"] > 360_000_000
    assert plan["null_space_elements"] == 360*360
    assert plan["null_space_bytes"] == 360*360*8
    assert plan["workspace_elements"] == 64*(360*360 + plan["pair_rows"] + 360)
    assert plan["workspace_bytes"] == plan["workspace_elements"]*8
    with pytest.raises(Exception, match="reserved memory"):
        psi4.core._atomic_polarizability_plan_wsm_refinement(500, 3, 360, 0, plan["estimated_bytes"]*2-1)
    with pytest.raises(Exception, match="point envelope"):
        psi4.core._atomic_polarizability_plan_wsm_refinement(501, 1, 120, 0, 1 << 40)
    with pytest.raises(Exception, match="variable envelope"):
        psi4.core._atomic_polarizability_plan_wsm_refinement(10, 4, 480, 0, 1 << 40)

    constrained = psi4.core._atomic_polarizability_plan_wsm_refinement(
        2, 3, 360, 360, 1 << 40)
    assert constrained["constraint_matrix_bytes"] == 360*360*8
    assert constrained["workspace_elements"] == 64*(360*360 + 360*360 + 3 + 360)
    assert constrained["constraint_svd_peak_bytes"] >= constrained["fit_svd_peak_bytes"]
    exact_memory = 2*constrained["estimated_bytes"]
    assert psi4.core._atomic_polarizability_plan_wsm_refinement(
        2, 3, 360, 360, exact_memory)["reserved_memory_bytes"] == constrained["estimated_bytes"]
    with pytest.raises(Exception, match="reserved memory"):
        psi4.core._atomic_polarizability_plan_wsm_refinement(
            2, 3, 360, 360, exact_memory-1)
    with pytest.raises(Exception, match="constraint rows"):
        psi4.core._atomic_polarizability_plan_wsm_refinement(2, 3, 360, 361, 1 << 40)


def test_source_guard_no_normal_equations_generator_or_external_executable():
    root = Path(__file__).parents[2]
    source = (root / "psi4/src/psi4/libmints/atomic_polarizability.cc").read_text()
    start = source.index("L3WorkingVector irregular_harmonics")
    end = source.index("Matrix lw_graph_operator", start)
    body = source[start:end]
    forbidden = ("normal_equation", "gram_matrix", "ExternalPotential", "OEProp",
                 "system(", "popen(", "execv(", "point_generator", ".camcasp-reference")
    for term in forbidden:
        assert term not in body
    assert re.search(r"\b(?:pfit|orient|camcasp)\b", body, re.IGNORECASE) is None
    assert "solve_constrained_least_squares(" in body


def test_mirror_site_dipole_offdiagonal_is_anchored_not_left_free():
    """The rank-1 anchor must cover the dipole off-diagonal, not only the diagonal.

    On a site with mirror-only symmetry the allowed dipole off-diagonal is the Cartesian
    component the point response constrains least. The reviewed protocol penalizes the whole
    rank-1 block, so this variable is held near its localized reference. Anchoring only the
    diagonal leaves it free to drift far from any physical value while still fitting the
    response, which is what this test pins down.
    """
    rng = np.random.default_rng(1707)
    points = rng.normal(size=(9, 3)) * 2 + [.9, -.4, .6]
    sites = [[0., 0., 0.]]

    # Active: the three dipole diagonals plus the 10-11c off-diagonal, i.e. the pattern a
    # Cs site produces (4 rank-1 variables).
    active = np.zeros(120, dtype=bool)
    active[[_upper_index(i, i) for i in range(3)]] = True
    active[_upper_index(0, 1)] = True

    reference = np.zeros((15, 15))
    reference[0, 0], reference[1, 1], reference[2, 2] = 8., 7., 6.
    reference[0, 1] = reference[1, 0] = 0.05

    true = np.zeros((15, 15))
    true[0, 0], true[1, 1], true[2, 2] = 2., 3., 4.
    true[0, 1] = true[1, 0] = 0.05

    noisy = _response(points, sites, [true])
    noisy[np.triu_indices(len(points))] += rng.normal(
        scale=.08, size=len(points) * (len(points) + 1) // 2)
    noisy = np.triu(noisy) + np.triu(noisy, 1).T

    result = _refine(points, sites, noisy, localized=[reference], active=active)[0]

    # All four rank-1 variables are anchored, not just the three diagonals.
    assert result["anchor_variable_count"] == 4

    fitted = np.asarray(result["tensors"])[0]
    # The anchored off-diagonal stays near its reference rather than drifting.
    assert abs(fitted[0, 1] - reference[0, 1]) < abs(reference[0, 1]) + 1.0
    # It is also symmetric in the packed tensor.
    assert fitted[0, 1] == pytest.approx(fitted[1, 0], abs=1e-12)
