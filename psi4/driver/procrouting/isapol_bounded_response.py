# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Disk-staged exact DF response primitives for explicitly budgeted campaigns.

This is a distinct backend, not a relaxation of the legacy response gates.
Only arrays produced in the current calculation enter its private checkpoints;
no checkpoint resume, external response import, or convergence certification is
provided. Budgets cover explicit numerical buffers and cumulative arithmetic/I/O,
not process RSS, caller-owned wavefunctions, or vendor-library workspaces.
"""
from dataclasses import dataclass
import hashlib
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, eigvalsh, lu_factor, lu_solve


@dataclass(frozen=True)
class BoundedResources:
    """Explicit opt-in ceilings; no individual stage can reset this allowance.

    ``max_bytes`` is sized by the caller (the oeprop preset uses the Psi4
    memory setting); work and checkpoint I/O keep fixed upper bounds.
    """
    max_bytes: int
    max_work: int
    max_io_bytes: int

    def __post_init__(self):
        for name, ceiling in (("max_bytes", None),
                              ("max_work", 6_000_000_000_000),
                              ("max_io_bytes", 64*1024**3)):
            value = getattr(self, name)
            if type(value) is not int or value <= 0 or (ceiling is not None and value > ceiling):
                bound = '' if ceiling is None else f' <= {ceiling}'
                raise ValueError(f'{name} must be a positive integer{bound}')


class _Ledger:
    def __init__(self, resources, reserved=0):
        if not isinstance(resources, BoundedResources):
            raise TypeError('explicit BoundedResources required')
        self.resources, self.reserved = resources, reserved
        self.work = self.io = self.peak = 0
        self.stages = []

    def admit(self, stage, numeric, work=0):
        total = numeric+self.reserved
        if total > self.resources.max_bytes:
            raise ValueError(f'{stage}: shared numeric byte resource limit')
        if self.work+work > self.resources.max_work:
            raise ValueError(f'{stage}: cumulative work resource limit')
        self.work += work  # charge before dispatch, never refund failures
        self.peak = max(self.peak, total)
        self.stages.append(dict(stage=stage, numeric_bytes=total, work=work))
        from psi4 import core
        core.print_out(f'\n  Bounded properties: {stage}; numeric plan {total} bytes; '
                       f'cumulative work {self.work}\n')

    def charge_io(self, count):
        if self.io+count > self.resources.max_io_bytes:
            raise ValueError('cumulative checkpoint I/O resource limit')
        self.io += count


class _Store:
    """Current-call private checkpoints with shape/type and content identity."""
    def __init__(self, directory, ledger):
        self.directory, self.ledger, self.records = directory, ledger, {}

    def save(self, name, array):
        a = np.asarray(array)
        if a.dtype != np.float64 or not np.isfinite(a).all():
            raise ValueError('finite float64 checkpoint required')
        if name in self.records or not name.replace('_', '').isalnum():
            raise ValueError('invalid or duplicate checkpoint name')
        self.ledger.charge_io(a.nbytes+256)
        path = self.directory/(name+'.npy')
        with path.open('xb') as stream:
            np.save(stream, a, allow_pickle=False)
        self.records[name] = (a.shape, self.digest(a))

    @staticmethod
    def digest(a):
        value = hashlib.sha256()
        # No full-array bytes duplicate while large operator arrays are live.
        for row in a:
            value.update(np.ascontiguousarray(row).view(np.uint8))
        return value.digest()

    def load(self, name):
        shape, expected = self.records[name]
        self.ledger.charge_io(8*int(np.prod(shape))+256)
        a = np.load(self.directory/(name+'.npy'), allow_pickle=False)
        if (a.dtype != np.float64 or a.shape != shape
                or self.digest(a) != expected):
            raise ValueError('checkpoint identity changed')
        return a


def _gram_terms(ov, dual):
    p, no, nv = ov.shape
    v = ov.transpose(0, 2, 1).reshape(p, no*nv).T @ dual.transpose(0, 2, 1).reshape(p, no*nv)
    y = v.reshape(nv, no, nv, no).transpose(2, 1, 0, 3).reshape(no*nv, no*nv).copy()
    return v, y


def _exchange_x(oo, vv_tiles, nv):
    p, no, _ = oo.shape
    x = np.empty((no, nv, no, nv))
    start = 0
    for tile in vv_tiles:
        if tile.ndim != 2 or tile.shape[0] != p or start+tile.shape[1] > nv*nv:
            raise ValueError('invalid VV tile coverage')
        block = (oo.reshape(p, no*no).T @ tile).reshape(no, no, -1)
        for k in range(tile.shape[1]):
            a, b = divmod(start+k, nv)
            x[:, a, :, b] = block[:, :, k]
        start += tile.shape[1]
    if start != nv*nv:
        raise ValueError('incomplete VV coverage')
    return x.transpose(1, 0, 3, 2).reshape(no*nv, no*nv).copy()


def _assemble(store, dimensions, tiles, exact_exchange, local_scale):
    p, no, nv = dimensions
    n = no*nv
    plan = 8*(3*n*n+4*n*p+p*p)+16*1024**2
    store.ledger.admit('dense operator assembly', plan, 8*p*n*n+2*n*p*p)
    oo = store.load('oo')
    x = _exchange_x(oo, (store.load(f'dualvv{i}') for i in range(tiles[1])), nv)
    store.save('x', x)
    del x, oo
    dual = np.empty((p, no*nv))
    start = 0
    for i in range(tiles[0]):
        tile = store.load(f'dualov{i}')
        dual[:, start:start+tile.shape[1]] = tile
        start += tile.shape[1]
    if start != n:
        raise ValueError('incomplete OV coverage')
    del tile
    ov = store.load('ov')
    v, y = _gram_terms(ov, dual.reshape(p, no, nv))
    store.save('v', v)
    store.save('y', y)
    del ov, dual, v, y
    h = store.load('x')
    h *= -exact_exchange
    y, v = store.load('y'), store.load('v')
    for a in range(0, n, 32):
        h[a:a+32] += 4*v[a:a+32]-exact_exchange*y[a:a+32]
    del y, v
    d, k = store.load('target'), store.load('kernel')
    for a in range(0, n, 32):
        h[a:a+32] += (4*local_scale)*(d[a:a+32] @ k) @ d.T
    h.flat[::n+1] += store.load('gaps')
    store.save('h1', h)
    del h, d, k
    h, y = store.load('x'), store.load('y')
    h *= -exact_exchange
    h += exact_exchange*y
    del y
    h.flat[::n+1] += store.load('gaps')
    store.save('h2', h)


def _stability(store, n):
    store.ledger.admit('operator stability', 32*n*n+1024**2, 4*n**3)
    result = {}
    for name in ('h1', 'h2'):
        h = store.load(name)
        asym = max(float(np.max(np.abs(h[i]-h[:, i]))) for i in range(n))
        scale = float(np.max(np.abs(h)))
        diagnostic = (h+h.T)*.5
        minimum = float(eigvalsh(diagnostic, subset_by_index=[0, 0], overwrite_a=True)[0])
        if asym > 1e-10*max(1., scale) or minimum <= 0 or not np.isfinite(minimum):
            raise ValueError(f'{name} reciprocity/stability gate')
        result[name] = dict(max_asymmetry=asym, minimum_symmetric_eigenvalue=minimum)
        del h, diagnostic
    return result


def _frequency(store, dimensions, q, omega):
    p, no, nv = dimensions
    n = no*nv
    planned = 8*(3*n*n+4*n*(p+q)+p*p+q*q+16*n)+8*1024**2
    store.ledger.admit('frequency response', planned, int(2*n**3+2*n**3/3+8*n*n*(p+q)))
    h1, h2 = store.load('h1'), store.load('h2')
    matrix = h2 @ h1
    del h1
    matrix.flat[::n+1] += omega*omega
    d, anchor = store.load('target'), store.load('anchor')
    legs = np.column_stack((d, anchor))
    del d, anchor
    rhs = -4*(h2 @ legs)
    del h2
    with warnings.catch_warnings():
        warnings.simplefilter('error', LinAlgWarning)
        lu, piv = lu_factor(matrix, check_finite=True)
    solved = lu_solve((lu, piv), rhs)
    del lu, piv
    worst = 0.
    for begin in range(0, p+q, 32):
        residual = matrix @ solved[:, begin:begin+32]-rhs[:, begin:begin+32]
        relative = np.linalg.norm(residual, axis=0)/np.maximum(
            np.linalg.norm(rhs[:, begin:begin+32], axis=0), np.finfo(float).tiny)
        worst = max(worst, float(np.max(relative)))
    if not np.isfinite(solved).all() or worst > 1e-10:
        raise ValueError('original H2H1 response residual gate')
    return legs[:, :p].T @ solved[:, :p], -legs[:, p:].T @ solved[:, p:], worst


def _kernel(auxiliary, density, grid, smoothing, cutoff, ledger):
    """Complete weighted Gram integration, then the exact fixed shell mask.

    The mask is independent of grid row. Applying it after integration is the
    same screened kernel, without allocating a full grid-by-AUX matrix.
    """
    from psi4 import core
    from .isapol_native_propagator import _superfunctional
    # Blocks of at least 256 points let the C++ basis sampling thread, and
    # the Gram GEMM sees a deep contraction.
    n, rows, block = auxiliary.nfunction, len(grid), 2048
    ledger.admit('full-grid AUX kernel',
                 8*(4*n*n+8*block*n+64*block)+2*grid.nbytes, 2*rows*n*n)
    layout = auxiliary.shell_layout()
    mask = np.abs(np.asarray(auxiliary.screening_s_overlap())) >= cutoff
    functional = _superfunctional('alda_slater_pw92', smoothing.rho_epsilon, block)
    result = np.zeros((n, n))
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for begin in range(0, rows, block):
            part = grid[begin:begin+block]
            chi = np.asarray(auxiliary.evaluate(part[:, :3].tolist()))
            rho = chi @ density
            values = functional.compute_functional(
                {'RHO_A': core.Vector.from_array(np.maximum(rho, smoothing.rho_epsilon))}, len(part), True)
            fxc = np.asarray(values['V_RHO_A_RHO_A'])[:len(part)]
            weights = part[:, 3]*smoothing.limit(fxc)
            result += chi.T @ (weights[:, None]*chi)
    for i, (offset, count, _, _) in enumerate(layout):
        for j, (other, width, _, _) in enumerate(layout):
            if not mask[i, j]:
                result[offset:offset+count, other:other+width] = 0.
    if not np.isfinite(result).all():
        raise ValueError('nonfinite AUX kernel')
    return result
