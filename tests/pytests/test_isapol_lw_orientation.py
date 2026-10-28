# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Known limitation: LW localization is not rotationally covariant.

This records the reference behaviour; it is not an acceptance of covariance.
Rotating a whole two-site system (origins and every block alpha(a,b) -> D alpha
D^T) does not rotate the LW output when the reciprocal off-site block is not
symmetric. ORIENT 5.0.11 gives the same localized tensors on these inputs, so
the deviation is inherited from the reference model. Symmetric blocks and
rotations with a diagonal D are covariant controls. The rank-1 isotropic value
is unchanged here; that is a sampled observation, not a guarantee for C6.
"""
import numpy as np
import pytest
import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api]

POSITIONS = np.array([[0.0, 0.0, 0.0], [0.3, -0.2, 1.1]])
SYMMETRIC_OFF_SITE = np.array([[0.7, 0.2, -0.1], [0.2, 0.5, 0.3], [-0.1, 0.3, 0.9]])
NONSYMMETRIC_OFF_SITE = np.array([[0.7, 0.35, -0.1], [0.05, 0.5, 0.3], [-0.25, 0.1, 0.9]])
RANK = {1: slice(0, 3), 2: slice(3, 8), 3: slice(8, 15)}


def _rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


C2X = _rotation([1, 0, 0], np.pi)
C4Z = _rotation([0, 0, 1], np.pi / 2)
GENERAL = _rotation([1, -1, 2], 0.83)


def _tensors(off_site):
    # Dipole-only, reciprocal: alpha(1,0) = alpha(0,1)^T.
    t = np.zeros((2, 2, 16, 16))
    t[0, 0, 1:4, 1:4] = t[1, 1, 1:4, 1:4] = 3.0 * np.eye(3)
    t[0, 1, 1:4, 1:4] = off_site
    t[1, 0, 1:4, 1:4] = off_site.T
    return t


def _localize(tensors, positions):
    result = psi4.core.isa_localize_lw(
        psi4.core.Matrix.from_array(np.asarray(positions, float)),
        [psi4.core.Matrix.from_array(np.ascontiguousarray(tensors[a, b])) for a in range(2) for b in range(2)],
        0.0, [[0, 1]], 1e-6, -1.0, 3)
    return np.array([np.asarray(m)[:15, :15] for m in result.local])


def _rotated_run(off_site, rotation):
    """Return (deviation from D g D^T, tensor scale, rank-1/2/3 isotropic changes)."""
    t = _tensors(off_site)
    d = np.asarray(psi4.core.isa_multipole_rotation(3, rotation.tolist()))
    g = _localize(t, POSITIONS)
    g_rot = _localize(np.einsum('ij,abjk,lk->abil', d, t, d), POSITIONS @ rotation.T)
    expected = np.einsum('ij,sjk,lk->sil', d[1:, 1:], g, d[1:, 1:])
    iso = lambda x: np.array([[np.trace(s[RANK[l], RANK[l]]) / (2 * l + 1) for l in (1, 2, 3)] for s in x])
    return np.abs(g_rot - expected).max(), np.abs(g).max(), np.abs(iso(g_rot) - iso(g)).max(axis=0)


@pytest.mark.parametrize('rotation', [C2X, C4Z, GENERAL], ids=['C2x', 'C4z', 'general'])
def test_symmetric_off_site_block_is_covariant(rotation):
    deviation, scale, _ = _rotated_run(SYMMETRIC_OFF_SITE, rotation)
    assert deviation <= 1e-12 * scale


def test_nonsymmetric_off_site_block_is_covariant_under_diagonal_rotation():
    deviation, scale, _ = _rotated_run(NONSYMMETRIC_OFF_SITE, C2X)
    assert deviation <= 1e-12 * scale


# ORIENT 5.0.11 "Localise LW" on the same inputs deviates by 2.835e-01 (C4z) and
# 2.201e-01 (general); the native output matched ORIENT to 1e-15 relative.
@pytest.mark.parametrize('rotation,reference', [(C4Z, 2.835e-01), (GENERAL, 2.201e-01)], ids=['C4z', 'general'])
def test_nonsymmetric_off_site_block_is_not_covariant(rotation, reference):
    deviation, scale, iso_change = _rotated_run(NONSYMMETRIC_OFF_SITE, rotation)
    assert deviation == pytest.approx(reference, rel=1e-3)
    assert deviation > 1e-2 * scale
    assert iso_change[0] <= 1e-12 * scale
    assert iso_change[1] > 1e-4
