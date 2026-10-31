"""Native request/transport seams only; no campaign SCF is run here."""
import json
import io
import base64
import hashlib
import zipfile

import numpy as np
import pytest

from psi4.driver.procrouting.isapol_molecular import MolecularDeclaration, PROFILE
from psi4.driver.procrouting.isapol_schema import NativeMonomerRequest


def request_data():
    declaration = MolecularDeclaration(PROFILE, (np.eye(3),)*3, ((0, 1), (0, 2)))
    return dict(
        schema='native-monomer-request/1', declaration=declaration.as_dict(),
        resources=dict(max_bytes=1024**3, max_work=100_000_000_000_000,
                       max_io_bytes=64*1024**3, capacity='large'),
        max_replay_work=1_000_000_000_000,
        implementation_sha256=dict(core='0'*64, isapol_python='1'*64))


def test_native_request_is_strict_json_and_retains_finite_model_inputs():
    raw = request_data()
    request = NativeMonomerRequest.from_dict(json.loads(json.dumps(raw)))
    assert request.as_dict() == raw
    assert request.declaration.profile == PROFILE
    assert request.resources.capacity == 'large'
    with pytest.raises(ValueError, match='versioned native request'):
        NativeMonomerRequest.from_dict(dict(raw, reference_dispersion=-2.))
    with pytest.raises(ValueError, match='versioned native request'):
        NativeMonomerRequest.from_dict(dict(raw, schema='native-monomer-request/2'))


@pytest.mark.parametrize('change', [
    dict(max_replay_work=True),
    dict(resources=dict(max_bytes=1024**3, max_work=100_000_000_000_000,
                        max_io_bytes=64*1024**3, capacity='standard')),
    dict(implementation_sha256=dict(core='unbound', isapol_python='1'*64)),
    dict(implementation_sha256=dict(core='0'*64, isapol_python='1'*64, oracle='2'*64)),
])
def test_unbounded_or_unpinned_native_requests_are_refused(change):
    with pytest.raises((TypeError, ValueError)):
        NativeMonomerRequest.from_dict(dict(request_data(), **change))


def input_molecule():
    import psi4
    return psi4.geometry(
        '0 1\nO 0 0 0\nH 1.4 0 1\nH -1.4 0 1\n'
        'symmetry c1\nno_com\nno_reorient\nunits bohr')


def test_native_input_preflight_requires_bound_code_and_exact_model(monkeypatch):
    from psi4 import core
    from psi4.driver.procrouting import isapol_schema as api
    pins = request_data()['implementation_sha256']
    # Installed-code identity is a platform boundary, not scientific data.
    monkeypatch.setattr(api, 'implementation_identity', lambda: dict(sha256=pins))
    monkeypatch.setattr(core, 'get_memory', lambda: 2*1024**3)
    molecule = input_molecule()
    request, identity = api.validate_native_request(
        request_data(), method='pbe0', basis='aug-cc-pVTZ', molecule=molecule)
    assert request.as_dict() == request_data() and identity['sha256'] == pins
    for change in (dict(method='hf'), dict(basis='aug-cc-pVDZ')):
        kwargs = dict(method='pbe0', basis='aug-cc-pVTZ', molecule=molecule)
        kwargs.update(change)
        with pytest.raises(ValueError, match='PBE0/aug-cc-pVTZ'):
            api.validate_native_request(request_data(), **kwargs)
    wrong_pin = dict(request_data(), implementation_sha256=dict(core='2'*64, isapol_python='1'*64))
    with pytest.raises(ValueError, match='implementation identity'):
        api.validate_native_request(wrong_pin, method='pbe0', basis='aug-cc-pVTZ', molecule=molecule)
    molecule.fix_orientation(False)
    with pytest.raises(ValueError, match='fixed'):
        api.validate_native_request(request_data(), method='pbe0', basis='aug-cc-pVTZ', molecule=molecule)


def test_numeric_tensor_artifact_roundtrips_through_json_without_pickle():
    from psi4.driver.procrouting import isapol_schema as api
    local = np.arange(11*3*15*15, dtype=np.float64).reshape(11, 3, 15, 15)
    refined = np.arange(11*3*9*9, dtype=np.float64).reshape(11, 3, 9, 9) / 10
    buffer = io.BytesIO()
    np.savez(buffer, local=local, refined=refined)
    artifact = api.tensor_artifact(buffer.getvalue())
    saved = json.loads(json.dumps(artifact, allow_nan=False))
    decoded = api.decode_tensor_artifact(saved)
    np.testing.assert_array_equal(decoded['local'], local)
    np.testing.assert_array_equal(decoded['refined'], refined)
    assert artifact['encoding'] == 'base64'
    assert artifact['size'] == len(buffer.getvalue())


def envelope(data):
    return dict(encoding='base64', media_type='application/x-npz',
                sha256=hashlib.sha256(data).hexdigest(), size=len(data),
                data=base64.b64encode(data).decode('ascii'))


@pytest.mark.parametrize('bad', ['digest', 'size', 'object', 'nonfinite', 'shape', 'extra'])
def test_corrupt_or_unsupported_tensor_artifacts_are_refused(bad):
    from psi4.driver.procrouting import isapol_schema as api
    local = np.zeros((11, 3, 15, 15))
    refined = np.zeros((11, 3, 9, 9))
    if bad == 'object':
        local = local.astype(object)
    elif bad == 'nonfinite':
        refined[0, 0, 0, 0] = float('nan')
    elif bad == 'shape':
        local = np.zeros((11, 3, 16, 16))
    buffer = io.BytesIO()
    np.savez(buffer, local=local, refined=refined, **({'extra': np.zeros(1)} if bad == 'extra' else {}))
    value = envelope(buffer.getvalue())
    if bad == 'digest':
        value['sha256'] = '0'*64
    elif bad == 'size':
        value['size'] -= 1
    with pytest.raises(ValueError):
        api.decode_tensor_artifact(value)


def test_forged_huge_npy_shape_is_refused_without_array_allocation(monkeypatch):
    from psi4.driver.procrouting import isapol_schema as api
    raw = io.BytesIO()
    np.lib.format.write_array_header_1_0(raw, dict(
        descr='<f8', fortran_order=False, shape=(11, 10**12, 15, 15)))
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as saved:
        saved.writestr('local.npy', raw.getvalue())
        saved.writestr('refined.npy', raw.getvalue())
    def no_allocation(*args, **kwargs):
        raise AssertionError('forged shape must be refused before allocation')
    monkeypatch.setattr(np, 'empty', no_allocation)
    with pytest.raises(ValueError, match='layouts'):
        api.decode_tensor_artifact(envelope(archive.getvalue()))


def test_failed_native_execution_request_clears_owned_results_before_any_export():
    import psi4
    from psi4 import core
    from psi4.driver.procrouting import isapol_schema as api
    wfn = core.Wavefunction.build(input_molecule(), 'sto-3g')
    wfn._native_atomic_property_result = object()
    request = NativeMonomerRequest.from_dict(request_data())
    with pytest.raises(ValueError, match='PBE0/aug-cc-pVTZ'):
        api.execute_native_request(wfn, request)
    with pytest.raises(ValueError, match='No native atomic property'):
        psi4.atomic_property_result(wfn)


@pytest.mark.parametrize('option,value', [('PUREAM', False), ('REFERENCE', 'UHF')])
def test_unsupported_orbital_options_are_refused_in_pre_scf_admission(option, value):
    from psi4 import core
    from psi4.driver.procrouting import isapol_schema as api
    core.clean_options()
    molecule = input_molecule()
    try:
        core.set_global_option(option, value)
        with pytest.raises(ValueError, match='spherical restricted'):
            api.validate_native_request(
                request_data(), method='pbe0', basis='aug-cc-pVTZ', molecule=molecule)
    finally:
        core.clean_options()


def schema_spec():
    return dict(
        driver='properties', model=dict(method='pbe0', basis='aug-cc-pvtz'),
        keywords=dict(
            reference='rhf', puream=True, scf_type='pk', guess='sad',
            e_convergence=1e-12, d_convergence=1e-12,
            function_kwargs=dict(native_monomer=request_data())),
        extras={})


def molecule_schema():
    return dict(symbols=['O', 'H', 'H'], geometry=[0., 0., 0., 1.4, 0., 1., -1.4, 0., 1.],
                molecular_charge=0., molecular_multiplicity=1,
                fix_com=True, fix_orientation=True, fix_symmetry='c1')


def test_schema_admission_binds_geometry_without_mutating_inputs():
    from psi4.driver.procrouting import isapol_schema as api
    spec, molecule = schema_spec(), molecule_schema()
    before = json.dumps([spec, molecule], sort_keys=True)
    request, snapshot = api.admit_native_schema(spec, molecule)
    assert request.as_dict() == request_data()
    api.check_native_geometry(input_molecule(), snapshot)
    moved = input_molecule()
    from psi4 import core
    moved.set_geometry(core.Matrix.from_array(moved.geometry().np + [1., 0., 0.]))
    with pytest.raises(ValueError, match='geometry'):
        api.check_native_geometry(moved, snapshot)
    assert json.dumps([spec, molecule], sort_keys=True) == before


@pytest.mark.parametrize('name,value', [
    ('dft_functional', 'b3lyp'), ('scf__reference', 'uhf'),
    ('basis', 'sto-3g'), ('perturb_h', True), ('grac_shift', .1),
])
def test_schema_refuses_scientific_overrides_before_driver(name, value):
    from psi4.driver.procrouting import isapol_schema as api
    spec = schema_spec()
    spec['keywords'][name] = value
    with pytest.raises(ValueError, match='scientific keywords'):
        api.admit_native_schema(spec, molecule_schema())


@pytest.mark.parametrize('value', [None, {}, False])
def test_present_but_malformed_schema_request_does_not_become_ordinary_properties(value):
    from psi4.driver.procrouting import isapol_schema as api
    spec = schema_spec()
    spec['keywords']['function_kwargs']['native_monomer'] = value
    with pytest.raises(ValueError, match='versioned native request'):
        api.admit_native_schema(spec, molecule_schema())


@pytest.mark.parametrize('change', ['files', 'kwargs', 'driver', 'frames', 'fragments'])
def test_schema_refuses_external_inputs_and_unsupported_geometry_contracts(change):
    from psi4.driver.procrouting import isapol_schema as api
    spec, molecule = schema_spec(), molecule_schema()
    if change == 'files':
        spec['extras']['extra_infiles'] = {'saved-orbitals': 'not a permitted input'}
    elif change == 'kwargs':
        spec['keywords']['function_kwargs']['external_potentials'] = {}
    elif change == 'driver':
        spec['driver'] = 'energy'
    elif change == 'frames':
        molecule['fix_orientation'] = False
    else:
        molecule['fragments'] = [[0], [1, 2]]
    with pytest.raises(ValueError):
        api.admit_native_schema(spec, molecule)


@pytest.mark.parametrize('value', [None, {}, False])
def test_schema_runner_refuses_malformed_native_presence_before_driver(monkeypatch, value):
    from psi4 import schema_wrapper, p4util
    data = dict(schema_name='qcschema_atomic_input', schema_version=2,
                molecule=dict(molecule_schema(), schema_name='qcschema_molecule', schema_version=2),
                specification=schema_spec())
    data['specification']['keywords']['function_kwargs']['native_monomer'] = value
    called = []
    def no_driver(*args, **kwargs):
        called.append(True)
        raise AssertionError('invalid native request must not reach any quantum driver')
    monkeypatch.setitem(schema_wrapper.methods_dict_, 'properties', no_driver)
    with pytest.raises(p4util.ValidationError, match='native'):
        schema_wrapper.run_json_qcschema(data, clean=True, json_serialization=False)
    assert not called


@pytest.mark.parametrize('version', [1, 2])
def test_synthetic_qcschema_route_preserves_extras_and_standard_dipole(monkeypatch, tmp_path, version):
    """Stub the quantum boundary; this is NOT a molecular acceptance run."""
    from psi4 import core, schema_wrapper
    from psi4.driver.procrouting import isapol_schema as api
    pins = request_data()['implementation_sha256']
    monkeypatch.setattr(api, 'implementation_identity', lambda: dict(sha256=pins))
    monkeypatch.setattr(core, 'get_memory', lambda: 2*1024**3)
    seen = []
    def descriptor_driver(method, **kwargs):
        assert method == 'pbe0'
        assert 'native_monomer' not in kwargs
        assert kwargs['properties'] == ['dipole'] and kwargs['return_wfn']
        wfn = core.Wavefunction.build(kwargs['molecule'], 'sto-3g')
        dipole = core.Matrix.from_array(np.array([[.1, .2, .3]]))
        core.set_variable('PBE0 DIPOLE', dipole)
        wfn.set_variable('PBE0 DIPOLE', dipole)
        seen.append(wfn)
        return 0., wfn
    payload = dict(schema='native-monomer-artifacts/1',
                   qualification='SYNTHETIC TRANSPORT FIXTURE, NOT MOLECULAR DATA')
    def synthetic_native(wfn, request):
        assert wfn is seen[-1] and request.as_dict() == request_data()
        return payload
    monkeypatch.setitem(schema_wrapper.methods_dict_, 'properties', descriptor_driver)
    monkeypatch.setattr(api, 'execute_native_request', synthetic_native)
    specification = schema_spec()
    molecule = dict(molecule_schema(), schema_name='qcschema_molecule',
                    schema_version=2 if version == 1 else 3)
    if version == 1:
        data = dict(schema_name='qcschema_input', schema_version=1, molecule=molecule,
                    **specification)
    else:
        data = dict(schema_name='qcschema_atomic_input', schema_version=2,
                    molecule=molecule, specification=specification)
    manager = core.IOManager.shared_object()
    previous_scratch = manager.get_default_path()
    manager.set_default_path(str(tmp_path))
    try:
        output = schema_wrapper.run_qcschema(data, return_dict=True)
    finally:
        manager.set_default_path(previous_scratch)
    assert output['success'], output.get('error')
    assert output['extras']['psi4:native_monomer'] == payload
    np.testing.assert_array_equal(np.asarray(output['return_result']['dipole']).ravel(), [.1, .2, .3])
    stored_input = output['input_data'] if version == 2 else output
    if version == 2:
        stored_input = stored_input['specification']
    assert stored_input['keywords']['function_kwargs']['native_monomer'] == request_data()
