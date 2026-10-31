"""Geometry-only input declaration tests; no molecular quantum calculation."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    's66_declarations', Path(__file__).with_name('declarations.py'))
declarations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(declarations)


def test_water_has_explicit_bonds_and_proper_local_to_global_frames():
    xyz_angstrom = np.array([[0., 0., 0.], [.7586, 0., .5043], [-.7586, 0., .5043]])
    declaration, evidence = declarations.declare_geometry(
        ['O', 'H', 'H'], xyz_angstrom / .529177210903)
    assert declaration['schema'] == 'native-monomer-declaration/1'
    assert declaration['bonds'] == [[0, 1], [0, 2]]
    assert len(evidence['frames']) == 3
    assert not evidence['lab_axis_fallback_sites']
    for index, frame in enumerate(declaration['frames']):
        frame = np.asarray(frame)
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(np.linalg.det(frame), 1., atol=1e-14)
        neighbor = 1 if index == 0 else 0
        z = xyz_angstrom[neighbor] - xyz_angstrom[index]
        np.testing.assert_allclose(frame[:, 2], z / np.linalg.norm(z), atol=1e-14)


def test_nondegenerate_frames_follow_proper_rotation_and_translation():
    xyz = np.array([[0., 0., 0.], [.7586, 0., .5043], [-.7586, 0., .5043]])
    rotation = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    before, _ = declarations.declare_geometry(['O', 'H', 'H'], xyz / .529177210903)
    after, _ = declarations.declare_geometry(
        ['O', 'H', 'H'], (xyz @ rotation.T + [3., -2., 4.]) / .529177210903)
    assert before['bonds'] == after['bonds']
    for a, b in zip(before['frames'], after['frames']):
        np.testing.assert_allclose(b, rotation @ a, atol=1e-14)


def test_near_linear_ethyne_discloses_lab_axis_azimuth():
    xyz = np.array([[0., 0., 0.], [1.20, 0., 0.], [-1.06, 1e-8, 0.],
                    [2.26, 0., 1e-8]])
    declaration, evidence = declarations.declare_geometry(
        ['C', 'C', 'H', 'H'], xyz / .529177210903)
    assert declaration['bonds'] == [[0, 1], [0, 2], [1, 3]]
    assert evidence['lab_axis_fallback_sites'] == [0, 1, 2, 3]
    assert all(row['x_atom'] is None for row in evidence['frames'])


@pytest.mark.parametrize('symbols,xyz,match', [
    (['O', 'H', 'H'], [[0, 0, 0], [10, 0, 0], [0, 10, 0]], 'connected'),
    (['O', 'H', 'H'], [[0, 0, 0], [0, 0, 0], [0, 1, 0]], 'overlapping'),
    (['O', 'H', 'H'], [[0, 0, 0], [1.164, 0, 0], [0, 1, 0]], 'ambiguous'),
    (['O', 'H', 'He'], [[0, 0, 0], [1, 0, 0], [0, 1, 0]], 'CHNO'),
    (['O', 'H', 'H'], [[0, 0, 0], [float('nan'), 0, 0], [0, 1, 0]], 'finite'),
    (['C', 'H', 'H'], [[0, 0, 0], [.8, 0, 0], [.8, .6, 0]], 'degree'),
])
def test_ambiguous_or_unsupported_inputs_are_refused(symbols, xyz, match):
    with pytest.raises(ValueError, match=match):
        declarations.declare_geometry(symbols, np.asarray(xyz) / .529177210903)


def test_prepared_declaration_is_bound_to_atom_order_and_geometry(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    module_spec = importlib.util.spec_from_file_location(
        'prepare_declarations', Path(__file__).with_name('prepare_declarations.py'))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    water = dict(id='monomer-test', formula='H2O', symbols=['O', 'H', 'H'],
                 geometry_bohr=[[0., 0., 0.], [1.4, 0., 1.], [-1.4, 0., 1.]],
                 charge=0., multiplicity=1)
    prepared = module.prepare(dict(monomers=[water]))[0]
    assert prepared['monomer_id'] == 'monomer-test'
    assert prepared['policy']['independent_parameters'] == 108
    assert prepared['geometry_sha256'] != module.geometry_digest(
        dict(water, symbols=['H', 'O', 'H']))
    assert prepared['geometry_sha256'] != module.geometry_digest(
        dict(water, geometry_bohr=[[0., 0., 0.], [1.4, 0., 1.], [-1.5, 0., 1.]]))
    with pytest.raises(ValueError, match='neutral singlets'):
        module.prepare(dict(monomers=[dict(water, charge=1.)]))
    with pytest.raises(ValueError, match='duplicate'):
        module.prepare(dict(monomers=[water, water]))
