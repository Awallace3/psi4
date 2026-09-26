# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Casimir-Polder quadrature and the exact wavefunction context fingerprint."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from psi4 import core
from .isapol_native_correction import functional_definition
from . import isapol_lw as lw


@dataclass(frozen=True)
class Quadrature:
    """Authoritative complete node list; CP weights ALREADY include 1/(2*pi)."""
    frequencies: tuple
    cp_weights: tuple
    provenance: lw.Provenance

    def __post_init__(self):
        f = _frequencies(self.frequencies)
        w = np.asarray(self.cp_weights)
        if (w.dtype.kind not in 'fiu' or w.shape != (len(f),) or not np.isfinite(w).all()
                or np.any(w < 0) or not np.any(w > 0)
                or any(x == 0 and y != 0 for x, y in zip(f, w))
                or any(x > 0 and y <= 0 for x, y in zip(f, w))):
            raise ValueError('complete quadrature requires positive weight at every dynamic node and zero static weight')
        if not isinstance(self.provenance, lw.Provenance):
            raise TypeError('explicit quadrature provenance required')
        object.__setattr__(self, 'frequencies', f)
        object.__setattr__(self, 'cp_weights', tuple(map(float, w)))

    @classmethod
    def from_casimir(cls, grid):
        if not isinstance(grid, core.CasimirGrid):
            raise TypeError('actual native CasimirGrid required')
        f = tuple(grid.omega(i) for i in range(grid.n_freq()+1))
        w = tuple(grid.cp_weight(i) for i in range(grid.n_freq()+1))
        # Actual core nodes, not rounded headers or a reimplemented rule.
        order = np.argsort(f)
        f, w = tuple(f[i] for i in order), tuple(w[i] for i in order)
        digest = hashlib.sha256(np.asarray([f, w], dtype='<f8').tobytes()).hexdigest()
        return cls(f, w, lw.Provenance('native CasimirGrid nodes/CP weights', digest,
                   'Psi4 core.CasimirGrid', f'n={grid.n_freq()}, beta={grid.omega0()}; CP prefactor included once'))


def _frequencies(values):
    a = np.asarray(values)
    if (a.dtype.kind not in 'fiu' or a.ndim != 1 or not 0 < len(a) <= 4096
            or not np.isfinite(a).all() or np.any(a < 0) or np.any(np.diff(a) <= 0)):
        raise ValueError('frequencies require finite strictly increasing nonnegative nodes')
    return tuple(map(float, a))


def _context(wfn):
    """Exact context fingerprint, including effective basis and actual full state."""
    if not isinstance(wfn, core.Wavefunction) or wfn.nirrep() != 1:
        raise ValueError('actual C1 wavefunction required')
    b, m = wfn.basisset(), wfn.molecule()
    shells = []
    for i in range(b.nshell()):
        s = b.shell(i)
        shells.append((b.shell_to_center(i), b.shell_to_basis_function(i), s.am, s.is_pure(),
                       tuple(s.exp(k) for k in range(s.nprimitive)),
                       tuple(s.coef(k) for k in range(s.nprimitive))))
    meta = (b.nbf(), wfn.nalpha(), wfn.nbeta(), tuple(wfn.doccpi()), tuple(wfn.soccpi()),
            tuple((m.Z(i), m.x(i), m.y(i), m.z(i)) for i in range(m.natom())), shells)
    h = hashlib.sha256(json.dumps(meta).encode())
    h.update(repr(functional_definition(wfn.functional(), include_cutoffs=True)).encode())
    for obj in (wfn.Ca(), wfn.Cb(), wfn.epsilon_a(), wfn.epsilon_b(), wfn.Da(), wfn.Db()):
        a = np.asarray(obj, dtype='<f8')
        if not np.isfinite(a).all():
            raise ValueError('nonfinite wavefunction context')
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()
