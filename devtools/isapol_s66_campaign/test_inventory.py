import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('s66_inventory', Path(__file__).with_name('inventory.py'))
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


def occurrence(xyz, name, symbols=None):
    return dict(symbols=symbols or ['C', 'H', 'N', 'O'], charge=0., multiplicity=1,
                geometry_bohr=np.asarray(xyz).tolist(), dimer_id=name, system='synthetic', R=1.,
                role='A', fragment_indices=list(range(len(xyz))), source_fragment_sha256='0'*64)


def test_proper_rotation_translation_and_deduplication():
    xyz = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [.1, .2, 3.]])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    moved = xyz@rotation.T+[3., -4., 2.]
    groups = inventory.group_fragments([occurrence(xyz, 'first'), occurrence(moved, 'second')], 1e-10)
    assert len(groups) == 1
    placement = groups[0]['members'][1]
    np.testing.assert_allclose(placement['rotation'], rotation, atol=1e-14)
    assert placement['max_abs_error_bohr'] < 1e-13


def test_reflections_and_different_conformations_are_not_merged():
    xyz = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [.1, .2, 3.]])
    mirror = xyz.copy()
    mirror[:, 0] *= -1
    changed = xyz.copy()
    changed[-1, -1] += 1e-3
    groups = inventory.group_fragments(
        [occurrence(xyz, 'first'), occurrence(mirror, 'mirror'), occurrence(changed, 'changed')], 1e-7)
    assert len(groups) == 3


def test_atom_order_charge_and_spin_are_identity_inputs():
    xyz = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [.1, .2, 3.]])
    first = occurrence(xyz, 'first')
    groups = inventory.group_fragments([
        first, dict(first, dimer_id='charge', charge=1.),
        dict(first, dimer_id='spin', multiplicity=3),
        occurrence(xyz, 'order', ['H', 'C', 'N', 'O']),
    ], 1e-7)
    assert len(groups) == 4


def test_linear_placement_is_disclosed_not_repaired():
    xyz = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    placement = inventory.proper_placement(xyz, xyz+[1., 2., 3.])
    assert placement['coordinate_rank'] == 1
    assert placement['max_abs_error_bohr'] < 1e-13
