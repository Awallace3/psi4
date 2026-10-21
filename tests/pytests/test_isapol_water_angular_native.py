# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Explicit native WATER smoke, ported from accepted bounded-recoupled tests.

UNRUN during source assembly. No frozen arrays, goldens or tolerance changes.
"""
import numpy as np
import pytest
import psi4
from psi4.driver.procrouting import isapol_bounded_oeprop as api
from psi4.driver.procrouting.isapol_logging import StageLog


def test_native_water_public_rank2_chain(tmp_path, monkeypatch):
    psi4.set_memory(4*1024**3)
    psi4.set_num_threads(1)
    molecule = psi4.geometry(
        '0 1\nO 0 0 0\nH .757 0 .586\nH -.757 0 .586\n'
        'units angstrom\nsymmetry c1\nno_com\nno_reorient')
    psi4.set_options({'basis': 'sto-3g', 'reference': 'rks', 'scf_type': 'pk',
        'e_convergence': 1e-10, 'd_convergence': 1e-10,
        'dft_radial_points': 50, 'dft_spherical_points': 110})
    energy, wfn = psi4.energy('pbe0', molecule=molecule, return_wfn=True)
    calls = []
    real = api.bounded_properties
    def counted(*args, **kwargs):
        calls.append(True)
        return real(*args, **kwargs)
    monkeypatch.setattr(api, 'bounded_properties', counted)
    assert psi4.oeprop(wfn, 'ATOMIC_REFINED_POLARIZABILITIES', 'ATOMIC_REFINED_DISPERSION',
        'ATOMIC_REFINED_RECOUPLED_DISPERSION', atomic_backend='BOUNDED_DF', preset='water',
        npoints=32, radial_points=20, spherical_points=50, scratch_directory=tmp_path,
        log=StageLog(0)) is None
    assert len(calls) == 1 and not list(tmp_path.iterdir())
    result = psi4.atomic_property_result(wfn)
    out = result.recoupled_dispersion
    assert out is not None and len(result.frequencies) == 11
    assert out.frequencies == list(result.frequencies)
    assert out.cp_weights[0] == 0 and all(w > 0 for w in out.cp_weights[1:])
    assert out.cp_weights == list(result.provenance['cp_weights'])
    assert [s.ranks for s in out.model.source.sites] == [[1, 2]]*3
    assert len(out.pairs) == 9 and out.max_order == 10
    assert len(result.refined_tensors) == 11
    for node, tensors in enumerate(result.refined_tensors):
        for i, a in enumerate(tensors):
            assert a.shape == (9, 9) and np.isfinite(a).all()
            np.testing.assert_array_equal(a, a.T)
            np.testing.assert_array_equal(a[0], np.zeros(9))
            np.testing.assert_array_equal(out.model.source.sites[i].responses[node].np, a[1:, 1:])
    assert any(np.ptp(np.diag(a)[1:4]) > 1e-5 for a in result.refined_tensors[0])
    assert any(np.max(np.abs(a[1:4, 4:9])) > 1e-5 for a in result.refined_tensors[0])
    assert any((c.t != 1 or c.u != 1) and abs(c.value) > 1e-7
               for p in out.pairs for c in p.coefficients)
    for p, iso in zip(out.pairs, result.dispersion.pairs):
        assert all(np.isfinite(c.value) for c in p.coefficients)
        for c in iso.coefficients:
            assert p.coefficient(c.order, 1, 1, 0) == pytest.approx(c.value, rel=3e-13, abs=1e-13)
        assert [c.order for c in p.coverage] == [6, 7, 8, 9, 10]
        assert p.coverage[0].unrestricted_complete
        assert not p.coverage[2].unrestricted_complete
        assert all(max(q) <= 2 for c in p.coverage for q in c.included_rank_quadruples)
    assert all(c.declared_model_complete for c in out.declared_model_coverage)
    assert max(d['response_residual'] for d in result.diagnostics) < 1e-10
    assert sum(s['stage'] == 'native recoupled dispersion' for s in result.resources['stages']) == 1
