"""Production-runtime tests; parent stages Python only and runs these explicitly.

No missing-API skips. Synthetic Lorentz oracles are independent of the parser;
portable external tensors test imported arithmetic, not wavefunction prediction.
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from psi4 import core
from psi4.driver.procrouting import isapol_supplied as b

DATA = Path(__file__).parent/"data_isapol/orient_local"


def imported():
    return b.read_orient_local_response(DATA, DATA/"manifest.json")


def lorentz(ranks=(1, 2, 3), matrix=None):
    # Independent 80-node rule for analytic tests, not the archive frequency rule.
    t, w = np.polynomial.legendre.leggauss(80)
    freq = np.r_[0, (1+t)/(1-t)]
    weights = np.r_[0, w/(np.pi*(1-t)**2)]
    if matrix is None:
        matrix = np.diag(np.repeat([2., 3., 5.][:len(ranks)], [2*l+1 for l in ranks]))
    s = b.Site("A", (0, 0, 0), np.eye(3), ranks)
    tensors = tuple(b.RawTensor("test", "A", ranks[0], ranks[-1], k, str(-f*f), k,
                                "handwritten synthetic Lorentz", 0, "synthetic, not archive", matrix/(1+f*f))
                    for k, f in enumerate(freq))
    return b.LocalResponse("test", "synthetic mathematical oracle, not molecular provenance", b.MODES[0],
                           (s,), tuple(freq), tuple(weights), tensors, (), "synthetic", "{}", "synthetic",
                           ("Synthetic Lorentz input; no external producer/native acceptance",), ())


def test_actual_core_grid_exact_and_cp_once():
    m = imported(); g = core.CasimirGrid(10, .5)
    assert m.frequencies == tuple(g.omega(k) for k in range(11))
    assert m.cp_weights == tuple(g.cp_weight(k) for k in range(11))
    assert m.cp_weights[0] == 0
    t, w = np.polynomial.legendre.leggauss(10)
    # Independent mapped Gauss rule; tabulated core root rounding is not bitwise NumPy.
    np.testing.assert_allclose(m.frequencies[1:], .5*(1+t)/(1-t), rtol=2e-11)
    np.testing.assert_allclose(m.cp_weights[1:], w/(1-t)**2/(2*np.pi), rtol=2e-11)
    p = b.supplied_local_properties(m)
    a = m.atomic_scalars[0][:, 0]
    assert p.isotropic[0].coefficients[0].value == pytest.approx(6*np.dot(m.cp_weights, a*a), rel=3e-14)


def test_lorentz_c6_analytic_and_higher_rank_independent_scalar_quadrature():
    m = lorentz(); p = b.supplied_local_properties(m)
    c6, c8, c10, c12 = p.isotropic[0].coefficients
    assert c6.value == pytest.approx(3., rel=3e-13)  # 3/4 * alpha_1(0)^2
    # Independent quadrature on xi=tan(theta), using a different integration map.
    t, w = np.polynomial.legendre.leggauss(120)
    theta = (t+1)*np.pi/4
    integral = np.dot(w*np.pi/4/np.cos(theta)**2, 1/(1+np.tan(theta)**2)**2)/(2*np.pi)
    np.testing.assert_allclose([c8.value, c10.value, c12.value],
                               [15*(2*3+3*2)*integral,
                                (28*2*5+70*3*3+28*5*2)*integral,
                                210*(3*5+5*3)*integral], rtol=4e-13)
    assert c10.unrestricted_complete and not c12.unrestricted_complete
    assert set(c12.missing) == {(1, 4), (4, 1)}
    assert set(c12.included) == {(2, 3), (3, 2)}


def test_dipole_cartesian_contraction_and_explicit_geometry():
    aa = np.array([[2., .2, -.1], [.2, 3., .4], [-.1, .4, 4.]])
    bb = np.array([[1., -.3, .2], [-.3, 4., -.1], [.2, -.1, 2.]])
    a, other = lorentz((1,), aa), lorentz((1,), bb)
    rot = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    placement = b.Placement([1., 2., 10.], rot)
    result = b.supplied_local_properties(a, other, placement_b=placement, anisotropic=True)
    r = np.array(placement.translation); u = r/np.linalg.norm(r)
    tau = np.eye(3)-3*np.outer(u, u)
    ac, bc = a.global_dipoles[0], b.place_model(other, placement).global_dipoles[0]
    expected = sum(w*np.trace(x@tau@y@tau.T) for w, x, y in zip(a.cp_weights, ac, bc))
    pair = result.anisotropic[0]
    assert pair.coefficients[0].value == pytest.approx(expected, rel=3e-13)
    assert pair.coefficients[0].energy == pytest.approx(-expected/np.linalg.norm(r)**6, rel=3e-13)
    assert [c.order for c in pair.coefficients] == list(range(6, 13))
    with pytest.raises(ValueError, match="explicit"):
        b.supplied_local_properties(a, anisotropic=True)
    with pytest.raises(ValueError, match="coincident"):
        b.supplied_local_properties(a, other, placement_b=b.Placement([0, 0, 0], np.eye(3)), anisotropic=True)


def test_tiny_asymmetry_keeps_isotropic_output_and_strict_policy():
    a = np.eye(3); a[0, 1] = 1e-12
    m = lorentz((1,), a)
    result = b.supplied_local_properties(m, m, placement_b=b.Placement([0, 0, 10], np.eye(3)), anisotropic=True)
    assert result.computed_status == "available" and result.isotropic
    assert result.anisotropic_status == "rejected_nonreciprocal_raw_input" and result.anisotropic == ()
    assert result.model_a.raw_tensors[0][0, 0, 1] == 1e-12
    assert result.model_a.raw_tensors[0][0, 1, 0] == 0
    assert any("No projection" in w for w in result.warnings)
    with pytest.raises(ValueError, match="exact reciprocity"):
        b._core_model(m, core, True)


def test_indefinite_is_legal_and_raw_preserved():
    m = lorentz((1,), np.diag([-1., 2., 3.]))
    r = b.supplied_local_properties(m, m, placement_b=b.Placement([0, 0, 10], np.eye(3)), anisotropic=True)
    assert r.anisotropic_status == "available"
    assert r.model_a.raw_tensors[0][0, 0, 0] == -1
    assert any("indefinite" in w for w in r.warnings)


def test_grids_must_match():
    a = lorentz()
    f = list(a.frequencies); f[1] *= 1.01
    with pytest.raises(ValueError, match="grids"):
        b.supplied_local_properties(a, replace(a, frequencies=f))


def test_actual_refined_outputs_status_serialization_and_comparison_failure():
    a = imported()
    r = b.supplied_local_properties(a, a, placement_b=b.Placement([0, 0, 10], np.eye(3)), anisotropic=True)
    assert len(r.isotropic) == len(r.anisotropic) == 9
    assert r.anisotropic_status == "available"
    comparisons = b.compare_casimir_data(a)
    r = replace(r, comparisons=comparisons)
    assert r.numerical_agreement
    failed = replace(r, comparisons=comparisons+(b.Comparison("deliberately failing reference", 1., 1e-9, False, "test"),))
    assert not failed.numerical_agreement and failed.computed_status == "available"
    assert failed.isotropic == r.isotropic and failed.anisotropic == r.anisotropic
    payload = json.loads(failed.to_json())
    assert payload["mode"] == "supplied_external_local_response"
    assert not payload["native_verified"] and not payload["wavefunction_first"]
    assert payload["model_a"]["tensors"][3]["section"] == 1
    assert len(payload["model_a"]["atomic_scalars"]) == 3
    assert payload["core_sha256"] and payload["driver_sha256"]
    assert not r.isotropic[0].coefficients[-1].unrestricted_complete
    frequency_checks = b.compare_frequency_headers(a)
    assert len(frequency_checks) == 22
    assert all(c.agreement for c in frequency_checks if "NL4" in c.name)
    assert {c.name for c in frequency_checks if not c.agreement} == {
        f"frequency_header:H2O_L3_{k:03}.pol" for k in range(7, 11)}
    with pytest.raises(ValueError):
        replace(r, comparisons=(b.Comparison("nonfinite", float("nan"), 0., False, "invalid"),)).to_json()


def test_cli_keeps_perturbed_hydrogen_without_type_oracle(tmp_path):
    import hashlib
    import shutil
    import subprocess
    import sys
    fixture = tmp_path/'fixture'
    shutil.copytree(DATA, fixture)
    manifest_path = fixture/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    response_path = fixture/manifest['files'][0]['name']
    lines = response_path.read_text().splitlines()
    index = next(i for i, line in enumerate(lines)
                 if line.split()[:5] == ['ALPHA', 'H2O', 'SITE-NAMES', 'H1', 'H1'])+1
    row = lines[index].split()
    expected = float(row[0])+.01
    row[0] = repr(expected)
    lines[index] = ' '.join(row)
    response_path.write_text('\n'.join(lines)+'\n')
    manifest['files'][0]['sha256'] = hashlib.sha256(response_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    (fixture/'manifest.json.sha256').write_text(hashlib.sha256(manifest_path.read_bytes()).hexdigest()+'  manifest.json\n')
    output = tmp_path/'perturbed-properties.json'
    cli = DATA.parent/'oracle/run_supplied_local_properties.py'
    subprocess.run([sys.executable, '-P', str(cli), '--manifest', str(manifest_path),
                    '--model-a', str(fixture), '--placement-b', str(fixture/'placement_b_example.json'),
                    '--output', str(output)], check=True, capture_output=True, text=True)
    payload = json.loads(output.read_text())
    assert payload['computed_status'] == payload['anisotropic_status'] == 'available'
    assert len(payload['isotropic']) == len(payload['anisotropic']) == 9
    assert payload['model_a']['global_cartesian_dipoles'][1][0][2][2] == expected
    comparison, = [c for c in payload['comparisons'] if c['name'] == 'pot:type-H-mapping']
    assert comparison['availability'] == 'unavailable'
    assert comparison['agreement'] is None and comparison['max_absolute_error'] is None
    assert comparison['checked_points'] == 0 and 'no representative selected' in comparison['reason']
    assert not any(c['name'].startswith('pot:H-') for c in payload['comparisons'])
    assert payload['numerical_agreement'] is False


def test_cli_emits_values_despite_reference_disagreement_and_exclusive_output(tmp_path):
    import subprocess
    import sys
    cli = DATA.parent/"oracle/run_supplied_local_properties.py"
    output = tmp_path/"properties.json"
    command = [sys.executable, "-P", str(cli), "--manifest", str(DATA/"manifest.json"),
               "--model-a", str(DATA), "--placement-b", str(DATA/"placement_b_example.json"),
               "--output", str(output)]
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    payload = json.loads(output.read_text())
    assert payload["computed_status"] == "available" and payload["structural_parse_success"]
    assert payload["numerical_agreement"] is False  # literal unrefined headers remain failed checks
    assert len(payload["isotropic"]) == len(payload["anisotropic"]) == 9
    assert payload["placement_source"]["sha256"]
    assert "computed=available" in completed.stdout
    original = output.read_bytes()
    second = subprocess.run(command, text=True, capture_output=True)
    assert second.returncode != 0 and "output exists" in second.stderr
    assert output.read_bytes() == original
