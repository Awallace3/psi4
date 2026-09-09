# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Hermetic supplied-LW driver tests. Only staged core and portable fixture IO.

Run with PYTHONPATH=$PWD/build_camcasp_psi4_joint/stage/lib python -P -m pytest ...
The new module is source-loaded, registered for dataclasses, without staging it.
"""
from dataclasses import FrozenInstanceError
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api]
ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'psi4/driver/procrouting/isapol_lw.py'
spec = importlib.util.spec_from_file_location('psi4.driver.procrouting._lw_driver_test_source', MODULE)
lw = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lw
spec.loader.exec_module(lw)
PROV = lw.Provenance('synthetic', hashlib.sha256(b'synthetic').hexdigest(), 'analytic test', 'Supplied synthetic, not native.')


def request(n=1, frequencies=(0.,), rank=3):
    return dict(labels=[f'S{i}' for i in range(n)], origins=np.zeros((n,3)), bonds=[],
                frequencies=list(frequencies), tensors=np.zeros((len(frequencies),n,n,(rank+1)**2,(rank+1)**2)),
                input_rank=rank, provenance=PROV,
                truncation=lw.TRUNCATE_RANK4 if rank == 4 else None)


@pytest.fixture(scope='module')
def water():
    path = ROOT / 'tests/pytests/data_isapol/orient_local/lw-hermetic-water.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == lw.HISTORICAL_FIXTURE_SHA256
    f = json.loads(path.read_text())
    labels = [s['label'] for s in f['sites']]
    assert [s['labels'] for s in f['distributed']['sections']] == [[a,b] for a in labels for b in labels]
    return f


def water_request(f):
    return dict(labels=[s['label'] for s in f['sites']], origins=[s['origin'][:] for s in f['sites']],
                frames=np.array([s['frame'] for s in f['sites']], dtype=float), bonds=[p[:] for p in f['bonds_zero_based']],
                frequencies=[f['frequency']], input_rank=4, truncation=lw.TRUNCATE_RANK4,
                tensors=np.array([s['values'] for s in f['distributed']['sections']], dtype=float).reshape(1,3,3,25,25),
                provenance=lw.Provenance(f['distributed']['filename'], f['distributed']['sha256'],
                                         'archived external numeric input', 'portable historical unrefined water fixture'))


def test_static_zero_and_metadata():
    r = lw.supplied_nonlocal_properties(**request(2))
    assert r.raw_local.shape == r.raw_global.shape == (1,2,15,15)
    assert r.atomic_scalars.shape == (1,2,3)
    assert r.global_dipoles.shape == (1,2,3,3)
    assert not np.any(r.raw_local.array)
    assert len(r.frequency_diagnostics) == 1
    assert r.frequency_diagnostics[0].residuals.maximum == 0
    # Four algorithm-controlled residuals, plus the supplied input's sum-rule defect
    # and the two names that reproduce it after localization.
    assert tuple(r.frequency_diagnostics[0].residuals.__dataclass_fields__) == (
        'off_site', 'charge_sum', 'reciprocity', 'molecular_sum', 'local_charge',
        'input_sum_rule', 'charge_sum_transport')
    assert r.frequency_diagnostics[0].residuals.algorithm_maximum == 0
    m = r.metadata
    assert (m.mode,m.tensor_origin,m.wavefunction_status,m.refinement_status) == ('supplied_nonlocal','Psi4_LW','no_native_wavefunction','no_PFIT')
    assert m.production_postcondition_passed and m.residual_tolerance == 1e-6
    assert m.numerical_agreement is None and not m.native_verified
    assert m.discarded_rank4_entry_count == 0
    assert m.anisotropic_status == 'separate_explicit_placement_adapter_available_not_computed'


def test_analytic_charge_flow():
    q = request(2)
    q['origins'][1] = [.2,-.3,.4]
    q['bonds'] = [[0,1]]
    q['tensors'][0,:,:,0,0] = [[-2,2],[2,-2]]
    r = lw.supplied_nonlocal_properties(**q)
    a = r.raw_global.array[0]
    np.testing.assert_allclose(a[0,0,:4], [-.16,-.08,.12,-.038], atol=1e-14,rtol=0)
    assert a[1,0,3] == pytest.approx(.038, abs=1e-14)
    np.testing.assert_allclose(a[:,14,14], -.000050625, atol=1e-14,rtol=0)
    assert r.frequency_diagnostics[0].residuals.maximum < 1e-10
    assert any('indefinite' in w for w in r.warnings)
    np.testing.assert_allclose(r.global_dipoles.array[0,0], -np.outer([.2,-.3,.4],[.2,-.3,.4]), atol=1e-14)


def test_frames_scalars_asymmetry_and_ownership():
    q = request()
    q['frames'] = [[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]]
    q['tensors'][0,0,0,1:4,1:4] = [[3,1e-8,0],[0,6,0],[0,0,9]]
    q['tensors'] = q['tensors'].tolist()  # deeply nested caller ownership
    before = np.array(q['tensors'])
    r = lw.supplied_nonlocal_properties(**q)
    np.testing.assert_array_equal(r.raw_input.array, before)
    np.testing.assert_array_equal(r.raw_global.array[0,0], before[0,0,0,1:,1:])
    assert r.raw_global.array[0,0,0,1] == 1e-8 and r.raw_global.array[0,0,1,0] == 0
    d = np.asarray(psi4.core.isa_multipole_rotation(3,q['frames'][0]))[1:,1:]
    np.testing.assert_allclose(r.raw_local.array[0,0], d.T@before[0,0,0,1:,1:]@d, atol=1e-14)
    np.testing.assert_array_equal(r.atomic_scalars.array[0,0], [6,0,0])
    np.testing.assert_array_equal(r.global_dipoles.array[0,0], [[6,0,0],[0,9,0],[1e-8,0,3]])
    q['labels'][0] = 'changed'; q['frames'][0][0][0] = 999
    q['origins'][0,0] = 999; q['frequencies'][0] = 9; q['tensors'][0][0][0][1][1] = 999
    for field in ('raw_input','raw_global','raw_local','atomic_scalars','global_dipoles','origins','frames'):
        s = getattr(r,field)
        copy = s.array; copy.fill(100)
        assert not np.all(s.array == 100)
    np.testing.assert_array_equal(r.raw_input.array,before)
    assert r.labels == ('S0',) and r.frequencies == (0.,)
    assert any('asymmetric' in w for w in r.warnings)
    with pytest.raises(FrozenInstanceError): r.labels = ('bad',)
    with pytest.raises(FrozenInstanceError): r.provenance.producer = 'bad'
    with pytest.raises(TypeError): r.raw_input.data[0] = 1
    with pytest.raises(FrozenInstanceError): r.frequency_diagnostics[0].residuals.off_site = 10


@pytest.mark.parametrize('fault', ['empty_labels','duplicate_labels','bad_label','label_string','rank2','rank5','rank_float','rank_bool',
    'missing_truncation','extra_truncation','wrong_truncation','empty_freq','negative_freq','nan_freq','inf_freq','unordered_freq','duplicate_freq',
    'freq_shape','tensor_shape','tensor_nan','tensor_inf','tensor_complex','tensor_bool','tensor_strings','ragged_tensor',
    'origin_shape','origin_nan','frame_shape','frame_nan','frame_reflection','frame_scale','frame_shear',
    'self_edge','duplicate_edge','reverse_edge','range_edge','negative_edge','bool_edge','float_edge','edge_shape',
    'no_provenance','bad_policy','too_many_sites','too_many_frequencies','input_budget','native_budget','generator'])
def test_invalid_boundaries(fault):
    q = request(2)
    if fault == 'empty_labels': q['labels'] = []
    elif fault == 'duplicate_labels': q['labels'] = ['S','S']
    elif fault == 'bad_label': q['labels'][0] = ' '
    elif fault == 'label_string': q['labels'] = 'SS'
    elif fault.startswith('rank'): q['input_rank'] = {'rank2':2,'rank5':5,'rank_float':3.,'rank_bool':True}[fault]
    elif fault == 'missing_truncation': q = request(rank=4); q['truncation'] = None
    elif fault == 'extra_truncation': q['truncation'] = lw.TRUNCATE_RANK4
    elif fault == 'wrong_truncation': q = request(rank=4); q['truncation'] = 'truncate'
    elif fault == 'empty_freq': q['frequencies'] = []
    elif fault == 'negative_freq': q['frequencies'] = [-1.]
    elif fault == 'nan_freq': q['frequencies'] = [np.nan]
    elif fault == 'inf_freq': q['frequencies'] = [np.inf]
    elif fault == 'unordered_freq': q['frequencies'] = [1.,0.]
    elif fault == 'duplicate_freq': q['frequencies'] = [0.,0.]
    elif fault == 'freq_shape': q['frequencies'] = [[0.]]
    elif fault == 'tensor_shape': q['tensors'] = np.zeros((1,4,16,16))
    elif fault in ('tensor_nan','tensor_inf'): q['tensors'][0,0,0,0,0] = np.nan if fault.endswith('nan') else np.inf
    elif fault == 'tensor_complex': q['tensors'] = q['tensors'].astype(complex)
    elif fault == 'tensor_bool': q['tensors'] = q['tensors'].astype(bool)
    elif fault == 'tensor_strings': q['tensors'] = q['tensors'].astype(str).tolist()
    elif fault == 'ragged_tensor': q['tensors'] = q['tensors'].tolist(); q['tensors'][0][0][0].pop()
    elif fault == 'origin_shape': q['origins'] = [[0,0,0]]
    elif fault == 'origin_nan': q['origins'][0,0] = np.nan
    elif fault.startswith('frame'):
        q['frames'] = np.tile(np.eye(3),(2,1,1))
        if fault == 'frame_shape': q['frames'] = np.eye(3)
        elif fault == 'frame_nan': q['frames'][0,0,0] = np.nan
        elif fault == 'frame_reflection': q['frames'][0,0,0] = -1
        elif fault == 'frame_scale': q['frames'][0,0,0] = 2
        elif fault == 'frame_shear': q['frames'][0,0,1] = .01
    elif fault.endswith('edge') or fault == 'edge_shape':
        q = request(3)
        q['bonds'] = {'self_edge':[[0,0]],'duplicate_edge':[[0,1],[0,1]],'reverse_edge':[[0,1],[1,0]],
                      'range_edge':[[0,3]],'negative_edge':[[-1,0]],'bool_edge':[[False,1]],
                      'float_edge':[[0.,1]],'edge_shape':[[0,1,2]]}[fault]
    elif fault == 'no_provenance': q['provenance'] = {'producer':'fake'}
    elif fault == 'bad_policy': q['residual_policy'] = 'arbitrary_1e-3'
    elif fault == 'too_many_sites': q['labels'] = ['x']*257
    elif fault == 'too_many_frequencies': q['frequencies'] = [0]*4097
    elif fault == 'input_budget': q['labels'] = [str(i) for i in range(200)]; q['frequencies'] = [0,1]
    elif fault == 'native_budget':
        q['labels'] = [str(i) for i in range(256)]; q['bonds'] = [[i,j] for i in range(256) for j in range(i+1,256)]
    elif fault == 'generator': q['frequencies'] = iter([0.])
    with pytest.raises(ValueError): lw.supplied_nonlocal_properties(**q)


@pytest.mark.parametrize('field,value', [('source_name',''),('source_sha256','bad'),('producer',' '),('description',[])])
def test_provenance_required(field,value):
    q = dict(source_name='s',source_sha256='0'*64,producer='p',description='d'); q[field] = value
    with pytest.raises(ValueError): lw.Provenance(**q)


def test_resource_guard_before_materialization():
    class Exploding(list):
        def __iter__(self): raise AssertionError('should not materialize tensors')
    q = request(); q['labels'] = [str(i) for i in range(200)]; q['frequencies'] = [0,1]
    q['tensors'] = Exploding()
    with pytest.raises(ValueError,match='resource'): lw.supplied_nonlocal_properties(**q)


def test_water_production_rejects(water):
    with pytest.raises(RuntimeError,match='postcondition.*charge-sum=.*local-charge='):
        lw.supplied_nonlocal_properties(**water_request(water))


def test_water_historical_all675(water):
    q = water_request(water)
    r = lw.supplied_nonlocal_properties(**q,residual_policy='historical_water_diagnostic')
    expected = np.array([s['values'] for s in water['expected_local']['sections']],float)
    assert expected.size == 675
    np.testing.assert_allclose(r.raw_local.array[0], expected, atol=1e-11,rtol=0)
    assert np.max(np.abs(r.raw_global.array[0,1]-expected[1])) > 15
    np.testing.assert_array_equal(r.raw_input.array,q['tensors'])
    assert r.metadata.discarded_rank4_entry_count == 3321
    assert r.metadata.canonical_input_array_sha256 != r.provenance.source_sha256
    assert r.metadata.residual_tolerance == 1e-3
    assert not r.metadata.production_postcondition_passed
    assert not r.frequency_diagnostics[0].production_postcondition_passed
    assert r.metadata.numerical_agreement is None and not r.metadata.native_verified
    assert r.frequency_diagnostics[0].residuals.charge_sum == pytest.approx(.0007011,abs=1e-14)


@pytest.mark.parametrize('fault',['tensor','discarded_tensor','signed_zero','origin','frame','frequency','graph','reverse_graph','reorder_graph','label','source','rank'])
def test_historical_exact_identity(water,fault):
    q = water_request(water)
    if fault == 'tensor': q['tensors'][0,0,0,1,1] += 1e-12
    elif fault == 'discarded_tensor': q['tensors'][0,0,0,24,24] += 1e-12
    elif fault == 'signed_zero': q['origins'][0][0] = -0.
    elif fault == 'origin': q['origins'][1][0] += 1e-12
    elif fault == 'frame': q['frames'][1] = np.eye(3)
    elif fault == 'frequency': q['frequencies'] = [1e-12]
    elif fault == 'graph': q['bonds'] = [[0,1]]
    elif fault == 'reverse_graph': q['bonds'][0] = [1,0]
    elif fault == 'reorder_graph': q['bonds'].reverse()
    elif fault == 'label': q['labels'][0] = 'O2'
    elif fault == 'source': q['provenance'] = PROV
    elif fault == 'rank': q['input_rank'] = 3; q['truncation'] = None; q['tensors'] = q['tensors'][...,:16,:16]
    with pytest.raises(ValueError,match='exact approved water identity'):
        lw.supplied_nonlocal_properties(**q,residual_policy='historical_water_diagnostic')


def test_rank4_truncation_is_exact_and_finite_required():
    q = request(rank=4)
    q['tensors'][0,0,0,1:4,1:4] = np.eye(3)*2
    q['tensors'][...,16:,:] = 321
    q['tensors'][...,:,16:] = 654
    r = lw.supplied_nonlocal_properties(**q)
    np.testing.assert_array_equal(r.raw_global.array[0,0],q['tensors'][0,0,0,1:16,1:16])
    assert r.metadata.discarded_rank4_entry_count == 369
    q['tensors'][...,24,24] = np.nan
    with pytest.raises(ValueError,match='finite'): lw.supplied_nonlocal_properties(**q)


def lorentz(n, frequencies, alpha, omega):
    q = request(n,frequencies)
    for k,xi in enumerate(frequencies):
        for s in range(n):
            q['tensors'][k,s,s,1:4,1:4] = np.eye(3)*alpha/(1+(xi/omega)**2)
    return lw.supplied_nonlocal_properties(**q)


def coefficient(r,order):
    return next(c for c in r.pairs[0].coefficients if c.order == order)


def test_dynamic_two_frequency_different_site_sets():
    a = lorentz(1,[0,2],3,2); b = lorentz(2,[0,2],5,4)
    weights = [0,.17]
    r = lw.isotropic_dispersion(a,b,cp_weights=weights,quadrature_provenance=PROV)
    assert len(r.pairs) == 2 and r.pairs[1].label_b == 'S1'
    assert coefficient(r,6).value == pytest.approx(6*.17*(3/2)*(5/1.25),abs=1e-14)
    c12 = coefficient(r,12)
    assert c12.value == 0 and not c12.unrestricted_complete
    assert c12.missing_rank_pairs == ((1,4),(4,1))
    assert c12.included_rank_pairs == ((2,3),(3,2))
    assert r.model_a.provenance == PROV
    weights[1] = 99
    assert r.cp_weights == (0.,.17)


def test_literal_lorentz_integral_independent_formula():
    grid = psi4.core.CasimirGrid(10,.5)
    frequencies = [grid.omega(i) for i in range(11)]
    weights = [grid.cp_weight(i) for i in range(11)]
    a = lorentz(1,frequencies,3,.5); b = lorentz(1,frequencies,5,.5)
    r = lw.isotropic_dispersion(a,b,cp_weights=weights,quadrature_provenance=PROV)
    # Integral of two identical Lorentz denominators: C6=3/4*alphaA*alphaB*omega.
    assert coefficient(r,6).value == pytest.approx(.75*3*5*.5,rel=2e-7)
    sa = psi4.core.IsaIsotropicSite(); sa.label='A'; sa.origin=[0,0,0]; sa.ranks=[1]
    sa.polarizabilities=psi4.core.Matrix.from_array(a.atomic_scalars.array[:,0,:1])
    sb = psi4.core.IsaIsotropicSite(); sb.label='B'; sb.origin=[0,0,0]; sb.ranks=[1]
    sb.polarizabilities=psi4.core.Matrix.from_array(b.atomic_scalars.array[:,0,:1])
    direct = psi4.core.isa_isotropic_dispersion(psi4.core.IsaIsotropicModel(frequencies,[sa],'synthetic'),
              psi4.core.IsaIsotropicModel(frequencies,[sb],'synthetic'),weights,12)
    assert coefficient(r,6).value == direct.pairs[0].coefficients[0].value


@pytest.mark.parametrize('weights',[[0],[1],[-1],[np.nan],[0,1]])
def test_static_cannot_supply_dispersion(weights):
    a = lw.supplied_nonlocal_properties(**request())
    with pytest.raises(ValueError): lw.isotropic_dispersion(a,a,cp_weights=weights,quadrature_provenance=PROV)


def test_dispersion_requires_explicit_grid_weights_and_provenance():
    a = lorentz(1,[0,1],2,1); b = lorentz(1,[0,2],2,1)
    with pytest.raises(ValueError,match='grids'): lw.isotropic_dispersion(a,b,cp_weights=[0,1],quadrature_provenance=PROV)
    with pytest.raises(TypeError): lw.isotropic_dispersion(a,a,quadrature_provenance=PROV)
    with pytest.raises(ValueError): lw.isotropic_dispersion(a,a,cp_weights=[0,1],quadrature_provenance=None)
    with pytest.raises(ValueError): lw.isotropic_dispersion(a,a,cp_weights=[0,1],quadrature_provenance=PROV,max_order=True)
    with pytest.raises(ValueError): lw.isotropic_dispersion(a,a,cp_weights=[1,1],quadrature_provenance=PROV)


def test_normal_package_import_without_staging():
    # Exercise normal Python package resolution with source added to the installed
    # package search path in memory only; no __init__, install, or staging writes.
    import importlib
    package = importlib.import_module('psi4.driver.procrouting')
    name = 'psi4.driver.procrouting.isapol_lw'
    old_path = package.__path__
    previous = sys.modules.pop(name, None)
    prior_attribute = getattr(package, 'isapol_lw', None)
    try:
        package.__path__ = [str(MODULE.parent)] + list(old_path)
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve() == MODULE
        q = request()
        q['provenance'] = module.Provenance('s','0'*64,'test','synthetic')
        assert module.supplied_nonlocal_properties(**q).metadata.tensor_origin == 'Psi4_LW'
    finally:
        package.__path__ = old_path
        sys.modules.pop(name,None)
        if previous is not None: sys.modules[name] = previous
        if prior_attribute is None: delattr(package, 'isapol_lw')
        else: package.isapol_lw = prior_attribute


def test_runtime_has_no_fixture_io_or_executables(water,monkeypatch):
    import builtins
    import subprocess
    q = water_request(water)  # bounded fixture loaded before IO guard
    def forbidden(*args,**kwargs): raise AssertionError('runtime IO/executable forbidden')
    monkeypatch.setattr(builtins,'open',forbidden)
    monkeypatch.setattr(Path,'open',forbidden)
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    assert lw.supplied_nonlocal_properties(**request()).metadata.production_postcondition_passed
    assert not lw.supplied_nonlocal_properties(**q,residual_policy='historical_water_diagnostic').metadata.production_postcondition_passed


def test_no_generic_tolerance_and_no_asymmetry_repair():
    with pytest.raises(TypeError): lw.supplied_nonlocal_properties(**request(),residual_tolerance=1e-3)
    q = request(); q['tensors'][0,0,0,1,2] = 2e-6
    with pytest.raises(RuntimeError): lw.supplied_nonlocal_properties(**q)


def test_nonzero_only_dynamic_grid_is_explicitly_supported():
    a = lorentz(1,[1,2],3,2)
    r = lw.isotropic_dispersion(a,a,cp_weights=[.2,.1],quadrature_provenance=PROV)
    assert coefficient(r,6).value == pytest.approx(6*(.2*(3/1.25)**2+.1*1.5**2))


def test_nested_record_constructor_snapshots():
    from dataclasses import replace
    r = lw.supplied_nonlocal_properties(**request())
    bonds = [[0,1]]; warnings = ['test']
    changed = replace(r, bonds=bonds, warnings=warnings)
    bonds[0][0] = 5; warnings[0] = 'changed'
    assert changed.bonds == ((0,1),) and changed.warnings == ('test',)
    pairs = [[1,4]]
    c = lw.Coefficient(12,0,[],pairs,False)
    pairs[0][0] = 9
    assert c.missing_rank_pairs == ((1,4),)


def test_disconnected_inconsistent_flow_fails_no_fallback():
    q = request(2); q['tensors'][0,:,:,0,0] = [[-2,2],[2,-2]]
    with pytest.raises(RuntimeError): lw.supplied_nonlocal_properties(**q)
