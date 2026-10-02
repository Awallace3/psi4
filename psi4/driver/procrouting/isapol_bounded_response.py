# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Disk-staged exact DF response primitives for explicitly budgeted campaigns.

Only arrays produced in the current calculation enter its private checkpoints;
there is no checkpoint resume, external response import, or convergence
certification. Budgets cover explicit numerical buffers and cumulative arithmetic/I/O,
not process RSS, caller-owned wavefunctions, or vendor-library workspaces.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import tempfile
import warnings

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import LinAlgWarning, cho_factor, cho_solve, eigvalsh, lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, eigsh

from .isapol_distribution import DistributedMoments, analytic_df_moments
from .isapol_logging import StageLog
from .isapol_native import Quadrature, _context
from .isapol_native_correction import validate_correction, require_scf_seal
from .isapol_native_factors import native_plain_df_operators, native_constrained_ov, native_plain_density
from .isapol_basis import BasisRecipe, adapt_main
from .isapol_native_propagator import KernelSmoothing
from .isapol_native_response import native_restricted_state_from_wavefunction


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


#: Canonical response selector spellings; the reference is the default.
RESPONSE_MODELS = ('reference_h2h1', 'native_fdds')

#: Adapter-only refusal of an explicit native FDDS instance whose declared
#: metric LU 1-norm reciprocal condition estimate is below this value. A
#: heuristic separator of the sampled epsilon-level metrics (<= 1.13e-15) from
#: the declared recipes (>= 5.77e-14); not an accuracy or forward-error bound.
FDDS_METRIC_RCOND_CUTOFF = 1e-14


@dataclass(frozen=True)
class NativeFDDSOptions:
    """Explicit native FDDS resources: OpenMP threads and peak scratch bytes.

    ``disk_bytes`` is the native instance's peak private scratch size, enforced
    by native. ``max_io_bytes`` is charged a source-derived estimate of the
    native tensor-file traffic at admission (see ``_native_fdds_estimates``),
    not a measured count.
    """
    nthread: int
    disk_bytes: int
    subalgo: str = 'OUT_OF_CORE'

    def __post_init__(self):
        for name in ('nthread', 'disk_bytes'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.subalgo not in ('OUT_OF_CORE', 'INCORE'):
            raise ValueError('subalgo must be OUT_OF_CORE or INCORE')


def _native_fdds_estimates(nbf, no, nv, pr, pd, requirement):
    """Source-derived hybrid native FDDS estimates: (bytes I/O, flops) per phase.

    Read from ``fdds_disp.cc`` (declared hybrid path): every native file is
    counted as written once and read once per pass, with the worst-case outer
    blocking of one occupied index per block for the exchange_X/exchange_Y
    re-reads. They are estimates, not proofs or bounds: native exposes no I/O
    or arithmetic counter. DFHelper raw-file traffic is estimated as one write
    and one read of ``disk:raw_peak`` (its per-file traffic is not modeled),
    and only tensor payload is counted, not file headers, logging, or
    stdio buffering (DFHelper flushes the tail of a file it wrote at its next
    file access, which can be in a later stage). Arithmetic covers the modeled
    dense and streamed operations only; AO integral generation (Libint2) has
    no arithmetic estimate at all.
    """
    n = no*nv
    npd, npr = n*pd, n*pr
    raw = requirement['disk:raw_peak']//8
    declared = 2*npd+(7*npd+no*no*pd+nv*nv*pd)
    qr = 3*npd
    exchange_x = 2*(3*npd+3*no*npd)
    exchange_y = 2*(no*no*pd+no*(nv*nv*pd+2*npd)+2*npd)
    construction_io = 8*(2*raw+(2*npr+no*no*pr+nv*nv*pr)+declared+qr+exchange_x+exchange_y)
    node_io = 8*(5*npd+8*npd)  # form_aux_matrices streams, then the eight S2 streams
    rows = 2*n+no*no+nv*nv
    construction_work = (2*pd*pr*(pr+pd)+40*pd**3                      # T J T^T, eigen/power/LU
                         + 4*pr*nbf*nbf*(no+nv)+2*pr*nbf*(no+nv)**2      # DFHelper transforms
                         + 2*rows*pd*(pr+3*pd)                           # declared pass
                         + 4*n*pd*pd+24*pd**3                            # QR, pinv(R)
                         + 24*no*no*nv*nv*pd)                            # two exchange_X, two exchange_Y
    node_work = 12*n*pd*pd+14*n*pd*pd+80*pd**3  # aux matrices, S2 streams, products/SVDs/LU
    return dict(construction_io_bytes=construction_io, node_io_bytes=node_io,
                construction_work=construction_work, node_work=node_work)


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
    del oo
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
    del ov, dual
    # x, v and y are combined in place (three n x n buffers, no checkpoints):
    # h1 = -cx x + (4v - cx y) + K in v's buffer, h2 = -cx x + cx y in x's.
    x *= -exact_exchange
    for a in range(0, n, 512):
        v[a:a+512] = x[a:a+512]+(4*v[a:a+512]-exact_exchange*y[a:a+512])
        x[a:a+512] += exact_exchange*y[a:a+512]
    h = v
    del y, v
    d, k = store.load('target'), store.load('kernel')
    dk = d @ k
    del k
    for a in range(0, n, 512):
        h[a:a+512] += (4*local_scale)*dk[a:a+512] @ d.T
    del d, dk
    gaps = store.load('gaps')
    h.flat[::n+1] += gaps
    store.save('h1', h)
    del h
    x.flat[::n+1] += gaps
    store.save('h2', x)


def _minimum_eigenvalue(a):
    """Lowest eigenvalue of a symmetric matrix; nonpositive whenever it is not SPD.

    A successful Cholesky factorization is the positive-definiteness
    certificate; shift-invert Lanczos on that factor then converges the
    reported minimum to machine precision in a few O(n^2) solves, instead of
    a dense O(n^3) tridiagonal reduction.
    """
    n = len(a)
    if n <= 64:
        return float(eigvalsh(a, subset_by_index=[0, 0])[0])
    try:
        factor = cho_factor(a, lower=True, check_finite=True)
    except LinAlgError:
        return 0.
    inverse = LinearOperator((n, n), dtype=float,
                             matvec=lambda v: cho_solve(factor, v, check_finite=False))
    # A seeded generic start: a symmetric one (e.g. all ones) can be exactly
    # orthogonal to the lowest mode of a symmetric molecule.
    start = np.random.default_rng(0).standard_normal(n)
    largest = eigsh(inverse, k=1, which='LA', tol=0, v0=start, return_eigenvectors=False)[0]
    return float(1/largest) if largest > 0 else 0.


def _stability(store, n):
    store.ledger.admit('operator stability', 32*n*n+1024**2, 4*n**3)
    result = {}
    for name in ('h1', 'h2'):
        h = store.load(name)
        asym = max(float(np.max(np.abs(h[i]-h[:, i]))) for i in range(n))
        scale = float(np.max(np.abs(h)))
        diagnostic = (h+h.T)*.5
        minimum = _minimum_eigenvalue(diagnostic)
        if asym > 1e-10*max(1., scale) or minimum <= 0 or not np.isfinite(minimum):
            raise ValueError(f'{name} reciprocity/stability gate')
        result[name] = dict(max_asymmetry=asym, minimum_symmetric_eigenvalue=minimum)
        del h, diagnostic
    return result


class _H2H1Response:
    """The original (unsymmetrized) H2 H1 product and H2-applied legs, formed once.

    Neither depends on the frequency, so the loop retains them instead of
    reloading and re-multiplying H1/H2 per node. Each node factors its own
    shifted copy and gates the residual against the same shifted operator.
    """
    def __init__(self, store, dimensions, q):
        p, no, nv = dimensions
        n = no*nv
        self.ledger, self.p, self.q, self.n = store.ledger, p, q, n
        self.retained = 8*(n*n+2*n*(p+q))
        self.ledger.admit('H2H1 response operator', self.retained+8*(2*n*n+n*(p+q))+8*1024**2,
                          int(2*n**3+2*n*n*(p+q)))
        h1, h2 = store.load('h1'), store.load('h2')
        self.matrix = h2 @ h1
        del h1
        d, anchor = store.load('target'), store.load('anchor')
        self.legs = np.column_stack((d, anchor))
        del d, anchor
        self.rhs = -4*(h2 @ self.legs)
        del h2
        self.ledger.reserved += self.retained

    def close(self):
        self.ledger.reserved -= self.retained
        self.matrix = self.legs = self.rhs = None

    def solve(self, omega):
        p, q, n, block = self.p, self.q, self.n, 512
        self.ledger.admit('frequency response', 8*(n*n+2*n*(p+q)+3*n*block+p*p+q*q+16*n)+8*1024**2,
                          int(2*n**3/3+6*n*n*(p+q)))
        shifted = self.matrix.copy()
        shifted.flat[::n+1] += omega*omega
        with warnings.catch_warnings():
            warnings.simplefilter('error', LinAlgWarning)
            lu, piv = lu_factor(shifted, overwrite_a=True, check_finite=True)
        del shifted
        solved = lu_solve((lu, piv), self.rhs)
        del lu, piv
        worst = 0.
        for begin in range(0, p+q, block):
            x, rhs = solved[:, begin:begin+block], self.rhs[:, begin:begin+block]
            residual = self.matrix @ x+omega*omega*x-rhs
            relative = np.linalg.norm(residual, axis=0)/np.maximum(
                np.linalg.norm(rhs, axis=0), np.finfo(float).tiny)
            worst = max(worst, float(np.max(relative)))
        if not np.isfinite(solved).all() or worst > 1e-10:
            raise ValueError('original H2H1 response residual gate')
        legs = self.legs
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
    # The kernel is symmetric: accumulate only its block upper triangle, in
    # column panels wide enough to keep the GEMMs threaded, then mirror.
    edges = np.linspace(0, n, 5).astype(int)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for begin in range(0, rows, block):
            part = grid[begin:begin+block]
            chi = np.asarray(auxiliary.evaluate(part[:, :3].tolist()))
            rho = chi @ density
            values = functional.compute_functional(
                {'RHO_A': core.Vector.from_array(np.maximum(rho, smoothing.rho_epsilon))}, len(part), True)
            fxc = np.asarray(values['V_RHO_A_RHO_A'])[:len(part)]
            weighted = (part[:, 3]*smoothing.limit(fxc))[:, None]*chi
            for start, end in zip(edges[:-1], edges[1:]):
                result[:end, start:end] += chi[:, :end].T @ weighted[:, start:end]
    result = np.triu(result)
    result += np.triu(result, 1).T
    for i, (offset, count, _, _) in enumerate(layout):
        for j, (other, width, _, _) in enumerate(layout):
            if not mask[i, j]:
                result[offset:offset+count, other:other+width] = 0.
    if not np.isfinite(result).all():
        raise ValueError('nonfinite AUX kernel')
    return result


def _partition_record(moments):
    return dict(model=moments.model, provenance=moments.provenance,
                labels=moments.labels, origins_bohr=moments.origins_bohr, rank=moments.rank,
                q_shape=moments.values.shape, q_sha256=hashlib.sha256(moments.values.tobytes()).hexdigest(),
                auxiliary_sha256=moments.auxiliary_sha256, state_sha256=moments.state_sha256,
                converged=moments.converged, convention=moments.convention,
                diagnostics=dict(moments.diagnostics))


def _sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@dataclass(frozen=True)
class FrequencyResponse:
    """One declared quadrature node of the selected response (atomic units).

    Reference H2H1: ``target_response`` is the p x p response of the eta=0
    target legs in the fitted response-AUX coefficient space; PFIT turns it into
    point responses. ``nonlocal_response`` is the (25*nsite)^2 site-major
    real-Racah rank-4 nonlocal polarizability ``-anchor.T @ X_anchor`` of the
    declared distributed moments. ``residual`` is the relative H2H1 residual.

    Native FDDS: ``target_response`` is the native declared coefficient
    response chi (p x p) and ``nonlocal_response`` is ``-Q chi Q.T`` with the
    same Q and order; both are symmetrized, and negative/positive
    semidefiniteness is NOT enforced. ``residual`` is native ``solve_residual``
    (the RMS residual of (I - A) chi = chi0 relative to chi0), not an accuracy
    certificate. ``diagnostics`` holds owned (name, value) scalar pairs.

    Arrays are fresh, owned, and never views of private checkpoints or native state.
    """
    frequency: float
    target_response: np.ndarray
    nonlocal_response: np.ndarray
    residual: float
    model: str
    diagnostics: tuple = ()


class BoundedResponse:
    """Plain-DF PBE0 response of one live wavefunction: reference H2H1 or explicit native FDDS.

    The call order is the resource-admission order of the complete chain::

        with BoundedResponse(...) as response:  # SCF seal, state, ledger, private scratch
            response.prepare(provider)          # factors, OV fits, Q legs, kernel, H1/H2
            ...                                 # caller admissions on response.ledger
            response.solve(omega)               # per node; operator formed at first solve
            response.release()                  # return the operator allowance

    ``provider(ledger)`` returns the declared DistributedMoments; None selects
    the analytic DF-centre producer, built only at the anchor contraction (its
    work charge lives there). The density-provider choice stays outside this
    runner, which holds the only reference and frees Q after the contraction.
    ``input_bytes`` reserves caller inputs retained beside the response (a
    partition grid or supplied Q). Leaving the context removes the private
    scratch directory on success and on failure.

    ``response='reference_h2h1'`` (the default) is the reference model above.
    ``response='native_fdds'`` (explicit; requires ``fdds=NativeFDDSOptions``)
    solves each node with the native declared-AUX FDDS instance (fused S2,
    native T1/T2/QR, full J_d^-1 coefficient conversion) on the Psi4 AO-order
    canonical orbitals, with W = J_d + 0.75 K from the same fitted-density
    Slater/PW92 kernel K. Q is retained and charged for every node; the native
    instance's memory is reserved for its lifetime and its scratch lives in the
    private directory. An instance whose ``metric_lu_rcond`` is below
    FDDS_METRIC_RCOND_CUTOFF is refused before any response; native Dyson
    admission refusals propagate unchanged. Neither admission is an accuracy
    guarantee, and there is no molecular accuracy proof for this route.
    """
    def __init__(self, wfn, auxiliary_recipe, sites, *, caller_converged, quadrature, response_grid,
                 smoothing, shell_cutoff, charge_penalty, anchor_metric_damping, resources,
                 scf_correction, expected_grac_shift=None, scratch_directory=None, log=None,
                 label='df_centre_analytic', input_bytes=0, response='reference_h2h1', fdds=None):
        from psi4 import core
        if response not in RESPONSE_MODELS:
            raise ValueError(f'response must be one of {RESPONSE_MODELS}')
        if response == 'reference_h2h1' and fdds is not None:
            raise ValueError('fdds options apply only to response=native_fdds')
        if response == 'native_fdds' and not isinstance(fdds, NativeFDDSOptions):
            raise TypeError('response=native_fdds requires explicit NativeFDDSOptions')
        if log is None:
            log = StageLog(1)
        if not isinstance(log, StageLog):
            raise TypeError('log must be a StageLog')
        if not isinstance(resources, BoundedResources):
            raise TypeError('explicit BoundedResources required')
        if not isinstance(auxiliary_recipe, BasisRecipe):
            raise ValueError('explicit AUX recipe required')
        if not isinstance(quadrature, Quadrature) or not isinstance(smoothing, KernelSmoothing):
            raise TypeError('explicit Quadrature and KernelSmoothing required')
        if scf_correction not in ('NONE', 'FIXED_GRAC'):
            raise ValueError('bounded route supports NONE or FIXED_GRAC only')
        for name, value, upper in (('shell_cutoff', shell_cutoff, float('inf')),
                                   ('charge_penalty', charge_penalty, float('inf')),
                                   ('anchor_metric_damping', anchor_metric_damping, 1.)):
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or not 0 <= value < upper:
                raise ValueError(f'invalid {name}')
        sites = tuple(sites)
        geometry = np.asarray(wfn.molecule().geometry())
        if (not 1 <= len(sites) <= 64 or any(not isinstance(s, core.IsaMultipoleSite) or s.rank != 4 for s in sites)
                or np.asarray([list(s.origin) for s in sites]).shape != geometry.shape
                or not np.array_equal([list(s.origin) for s in sites], geometry)
                or not np.array_equal(auxiliary_recipe.centres, geometry)):
            raise ValueError('rank-4 multipole sites, AUX and wavefunction centres must agree')
        if (not isinstance(response_grid, np.ndarray) or response_grid.dtype != np.float64
                or response_grid.ndim != 2 or response_grid.shape[1] != 4
                or not len(response_grid) or not np.isfinite(response_grid).all()):
            raise ValueError('finite float64 full response grid (rows,4) required')
        if type(input_bytes) is not int or input_bytes < 0:
            raise ValueError('input_bytes must be a nonnegative integer')
        self.wfn, self.auxiliary_recipe, self.sites = wfn, auxiliary_recipe, sites
        self.caller_converged, self.quadrature, self.response_grid = caller_converged, quadrature, response_grid
        self.smoothing, self.shell_cutoff = smoothing, shell_cutoff
        self.charge_penalty, self.anchor_metric_damping = charge_penalty, anchor_metric_damping
        self.resources, self.scf_correction, self.expected_grac_shift = resources, scf_correction, expected_grac_shift
        self.scratch_directory, self.log, self.label, self.input_bytes = scratch_directory, log, label, input_bytes
        self.response, self.fdds = response, fdds
        if response == 'reference_h2h1':
            self.model = (f'plain DF PBE0; fitted-density ALDA Slater/PW92; lambda={charge_penalty}; '
                          f'target/kernel eta=0; anchor eta={anchor_metric_damping}; original H2H1; -B.T C0 B')
            # Downstream (stage07) LW policy: strict, and local charge enters the sum rule.
            self.lw_residual_policy, self.lw_disclosure = 'production', None
        else:
            self.model = ('native declared-AUX FDDS (fused S2, T1/T2/QR, full J_d^-1 coefficient conversion); '
                          'PBE0 x_alpha=0.25; W = J_d + 0.75 K, K fitted-density ALDA Slater/PW92 kernel eta=0; '
                          'charge_penalty and anchor_metric_damping inapplicable; residual = native solve_residual; '
                          '-Q chi Q.T; symmetrized, NSD not enforced')
            self.lw_residual_policy = 'reported_input_sum_rule'
            self.lw_disclosure = ('local_charge is report-only; the rank-0 remainder is not carried into '
                                  'local tensors, PFIT or C6')
        self.provenance = None
        self._fdds = self._fdds_kernel = self._moments = self._fdds_inputs = self._fdds_plan = None
        self._held = {}
        self._temporary = self._operator = self.partition = self.stability = None
        self._prepared = self._released = False

    def __enter__(self):
        if self._temporary is not None or self._released:
            raise RuntimeError('BoundedResponse is single-use')
        wfn, log, resources = self.wfn, self.log, self.resources
        self.correction = validate_correction(wfn, scf_correction=self.scf_correction,
                                              expected_grac_shift=self.expected_grac_shift, require_canonical=True)
        require_scf_seal(wfn)
        state = native_restricted_state_from_wavefunction(wfn, caller_converged=self.caller_converged,
                                                          max_bytes=resources.max_bytes)
        grid = np.array(self.response_grid, copy=True)
        # Reserve caller input/copy overlap, state and MAIN transformations together.
        self.ledger = _Ledger(resources, 2*grid.nbytes+state.planned_bytes+32*state.nbf**2+self.input_bytes)
        self.ledger.admit('state and grid', 0)
        main = adapt_main(wfn, caller_converged=True)
        coefficients = main.transform @ np.asarray(state.orbitals())
        energies = np.asarray(state.energies()).copy()
        self.auxiliary = self.auxiliary_recipe.build('MolecularAux')
        p, no, nv = self.auxiliary.nfunction, state.nocc, state.nvir
        n, q, nf = no*nv, 25*len(self.sites), len(self.quadrature.frequencies)
        self.dimensions = (p, no, nv)
        log.items((('AUX / occupied / virtual / OV', (p, no, nv, n)),
                   ('sites / frequencies / grid points', (len(self.sites), nf, len(grid))),
                   ('distribution', self.label), ('correction', repr(self.correction)),
                   ('resource ceilings', repr(resources))))
        # Frequency outputs (local tensors, PFIT models) are retained across the sweep.
        self._retained = self.response_grid.nbytes+self.input_bytes+nf*(len(self.sites)*25*25*8*4+4*64*64*8)
        if self.response == 'native_fdds':
            self._fdds_preflight(state, grid, main)
        else:
            frequency_plan = 8*(3*n*n+4*n*(p+q)+p*p+q*q+16*n)+8*1024**2
            if frequency_plan+self._retained > resources.max_bytes:
                raise ValueError('complete frequency plan exceeds shared numeric byte resource limit')
            frequency_work = nf*int(2*n**3+2*n**3/3+8*n*n*(p+q))
            if frequency_work > resources.max_work:
                raise ValueError('complete quadrature exceeds cumulative work resource limit')
        self.state_sha256 = _context(wfn)
        self.auxiliary_sha256 = hashlib.sha256(repr(self.auxiliary_recipe).encode()).hexdigest()
        self.grid_sha256 = hashlib.sha256(grid.tobytes()).hexdigest()
        self._state, self._main, self._coefficients, self._energies, self._grid = (
            state, main, coefficients, energies, grid)
        self._temporary = tempfile.TemporaryDirectory(prefix='psi4-bounded-', dir=self.scratch_directory)
        self.store = _Store(Path(self._temporary.name), self.ledger)
        return self

    def __exit__(self, *exception):
        try:
            self.release()
        finally:
            self._state = self._main = self._coefficients = self._energies = self._grid = None
            # The native destructor removes its files before the directory is cleaned.
            self._fdds = self._fdds_kernel = self._moments = self._fdds_inputs = None
            self._temporary.cleanup()
        return False

    def prepare(self, provider=None):
        """Factors, both constrained OV fits, anchor legs, kernel, H1/H2 and gates."""
        if self._temporary is None or self._prepared or self._released:
            raise RuntimeError('prepare once, inside the BoundedResponse context')
        self._prepared = True
        if self.response == 'native_fdds':
            return self._prepare_fdds(provider)
        ledger, store, log, resources = self.ledger, self.store, self.log, self.resources
        moments = None if provider is None else provider(ledger)
        if provider is not None and not isinstance(moments, DistributedMoments):
            raise TypeError('provider must return DistributedMoments')
        auxiliary, recipe, sites = self.auxiliary, self.auxiliary_recipe, self.sites
        p, no, nv = self.dimensions
        n, q = no*nv, 25*len(sites)
        supplied = moments is not None
        if supplied:
            moments.validate_for(recipe, sites, 4, self.state_sha256)
            ledger.reserved += moments.values.nbytes
            ledger.admit('retained distributed moments', 0)
        log.stage('Native plain-DF factors', (('exact exchange', .25), ('tile columns', 512)))
        ledger.admit('native factor work reservation', 0,
                     8*p*(no+nv)**3+4*p*p*(no+nv)**2+2*p**3)
        ops = native_plain_df_operators(auxiliary, self._main.basis, self._coefficients, self._energies,
            nocc=no, shell_count=len(recipe.shells), exact_exchange=.25,
            tile_columns=512,
            max_bytes=resources.max_bytes-ledger.reserved)
        ledger.admit('native factors', ops.construction_planned_bytes)
        for name, value in (('gaps', ops._gaps), ('oo', ops._oo), ('ov', ops._ov),
                            ('density', ops.plain_density_coefficients)):
            store.save(name, value)
        tiles = (len(ops._dual_ov.tiles), len(ops._dual_vv.tiles))
        for kind in ('ov', 'vv'):
            for i, tile in enumerate(getattr(ops, '_dual_'+kind).tiles):
                store.save(f'dual{kind}{i}', tile)
        del tile, value
        for name, eta in (('anchor', self.anchor_metric_damping), ('target', 0.)):
            log.stage(name + ' constrained OV fit', (('metric damping', eta), ('charge penalty', self.charge_penalty)))
            ledger.admit(name+' fit work reservation', 0, 2*p**3+4*p*p*n)
            fit = native_constrained_ov(ops, charge_penalty=self.charge_penalty,
                offsite_metric_damping=eta, tile_columns=32,
                max_bytes=resources.max_bytes-ledger.reserved-32*p*q-16*n*q)
            ledger.admit(name+' fit', fit.planned_bytes+32*p*q+16*n*q)
            if name == 'anchor':
                ledger.admit('distributed moment contraction', fit.planned_bytes+32*p*q+16*n*q,
                             2*n*p*q + (2000*p*q if moments is None else 0))
                if moments is None:
                    moments = analytic_df_moments(recipe, sites, 4)
                moments.validate_for(recipe, sites, 4, self.state_sha256)
                self.partition = _partition_record(moments)
                store.save(name, moments.anchor_legs(fit.coefficients))
                if supplied:
                    ledger.reserved -= moments.values.nbytes
                del moments
            else:
                store.save(name, fit.coefficients)
            del fit
        del ops
        self._state = self._main = self._coefficients = self._energies = None
        grid, self._grid = self._grid, None
        ledger.reserved = 2*grid.nbytes+self.input_bytes
        log.stage('Full-grid ALDA kernel', (('smoothing', repr(self.smoothing)), ('shell cutoff', self.shell_cutoff)))
        kernel = _kernel(auxiliary, store.load('density'), grid, self.smoothing, self.shell_cutoff, ledger)
        store.save('kernel', kernel)
        del kernel, grid
        # Input response_grid remains caller-owned; retained output allowance is
        # charged across the entire frequency sweep rather than node by node.
        ledger.reserved = self._retained
        log.stage('H1/H2 assembly')
        _assemble(store, (p, no, nv), tiles, .25, .75)
        log.stage('Response stability checks')
        self.stability = _stability(store, n)
        log.items(self.stability.items())
        return self

    def solve(self, omega):
        """Solve one declared quadrature node; the H2H1 operator is formed on first use."""
        frequencies = self.quadrature.frequencies
        if not self._prepared or self._released:
            raise RuntimeError('solve requires a prepared, unreleased BoundedResponse')
        if omega not in frequencies:
            raise ValueError('solve accepts only declared quadrature nodes')
        p = self.dimensions[0]
        q = 25*len(self.sites)
        node = frequencies.index(omega)+1
        tag = f'node {node}/{len(frequencies)}, omega={omega:.12g} au'
        if self.response == 'native_fdds':
            return self._solve_fdds(omega, tag)
        if self._operator is None:
            self._operator = _H2H1Response(self.store, self.dimensions, q)
        self.log.stage('Original H2H1 response: ' + tag, (('RHS', p+q),))
        target, nonlocal_response, residual = self._operator.solve(omega)
        self.log.items((('relative response residual', residual),))
        return FrequencyResponse(omega, target, nonlocal_response, residual, self.model)

    def _hold(self, **reservations):
        """Named live reservations; the ledger's reserved total is always their sum."""
        self._held.update(reservations)
        self.ledger.reserved = sum(self._held.values())

    def _fdds_preflight(self, state, grid, main):
        """Native inputs and the complete explicit-FDDS plan, before any scratch or integral."""
        from psi4 import core
        resources, fdds, wfn = self.resources, self.fdds, self.wfn
        p, no, nv = self.dimensions
        q, nf = 25*len(self.sites), len(self.quadrature.frequencies)
        nbf, nmo = state.nbf, state.nmo
        # Psi4 AO-order canonical orbitals, never the DALTON MAIN transform.
        orbitals, energies = state.orbitals().to_array(), state.energies().to_array()
        if not (np.array_equal(orbitals, np.asarray(wfn.Ca()))
                and np.array_equal(energies, np.asarray(wfn.epsilon_a()))):
            raise ValueError('native FDDS state must equal the sealed wavefunction Ca/epsilon_a')
        primary = state.basis_snapshot()
        raw, transform = core.IsaAuxCoulomb(self.auxiliary).native_auxiliary()
        if transform.rows() != p or transform.cols() != raw.nbf():
            raise ValueError('declared AUX map does not match the explicit AUX basis')
        pr = raw.nbf()
        descriptors = (16*1024*(primary.nshell()+raw.nshell())
                       + 4*1024*(primary.molecule().natom()+raw.molecule().natom()))
        inputs = 8*(p*pr+2*(nbf*nmo+nmo))+descriptors  # T, orbital/energy copies, basis descriptors
        self._hold(grid=2*grid.nbytes, state=state.planned_bytes+32*nbf**2, input=self.input_bytes,
                   native_inputs=inputs)
        self.ledger.admit('native FDDS inputs', 0)
        arrays = (orbitals[:, :no], orbitals[:, no:], energies[:no], energies[no:])
        hashes = dict(zip(('Cocc', 'Cvir', 'eps_occ', 'eps_vir'), map(_sha256, arrays)), T=_sha256(transform.np))
        cocc, cvir = core.Matrix.from_array(arrays[0]), core.Matrix.from_array(arrays[1])
        eocc, evir = core.Vector.from_array(arrays[2]), core.Vector.from_array(arrays[3])
        del orbitals, energies, arrays
        requirement = core.FDDS_Monomer.requirement(primary, raw, no, nv, p, True, fdds.subalgo, fdds.nthread)
        memory, disk = requirement['memory_bytes'], requirement['disk_bytes']
        if disk > fdds.disk_bytes:
            raise ValueError(f'native FDDS peak scratch {disk} bytes exceeds NativeFDDSOptions.disk_bytes '
                             f'{fdds.disk_bytes}')
        estimates = _native_fdds_estimates(nbf, no, nv, pr, p, requirement)
        node_bytes = 8*(2*p*p+p*q+2*q*q)+8*1024**2
        node_work = estimates['node_work']+2*q*p*p+2*q*q*p
        persistent = memory+8*q*p+8*p*p
        # Native admission, W formation (after construction) and every node, checked before any scratch.
        if (persistent+inputs+self.input_bytes > resources.max_bytes
                or persistent+32*p*p+self.input_bytes > resources.max_bytes
                or persistent+node_bytes+self._retained > resources.max_bytes):
            raise ValueError('complete native FDDS plan exceeds shared numeric byte resource limit')
        nao = main.basis.nfunction
        work = (2000*p*q+2*p*nao*nmo*(nao+nmo)+p**3+2*len(grid)*p*p
                + estimates['construction_work']+nf*node_work)
        if work > resources.max_work:
            raise ValueError('complete quadrature exceeds cumulative work resource limit')
        io = estimates['construction_io_bytes']+nf*estimates['node_io_bytes']
        if io > resources.max_io_bytes:
            raise ValueError('native FDDS cumulative I/O exceeds checkpoint I/O resource limit')
        self._fdds_inputs = (primary, raw, transform, cocc, cvir, eocc, evir)
        self._fdds_plan = dict(requirement=dict(requirement), estimates=estimates, node_bytes=node_bytes,
                               node_work=node_work, planned_work=work, planned_io_bytes=io)
        self.provenance = dict(response=self.response, options=self.fdds, input_sha256=hashes,
                               ints_tolerance=core.get_global_option('INTS_TOLERANCE'),
                               screening=core.get_global_option('SCREENING'),
                               naux_raw=pr, plan=self._fdds_plan,
                               lw_residual_policy=self.lw_residual_policy, lw_disclosure=self.lw_disclosure)

    def _prepare_fdds(self, provider):
        """Q, plain density, kernel, native instance and its metric guard, then W."""
        from psi4 import core
        ledger, log, resources, fdds = self.ledger, self.log, self.resources, self.fdds
        auxiliary, recipe, sites = self.auxiliary, self.auxiliary_recipe, self.sites
        p, no, nv = self.dimensions
        q = 25*len(sites)
        moments = None if provider is None else provider(ledger)
        if provider is not None and not isinstance(moments, DistributedMoments):
            raise TypeError('provider must return DistributedMoments')
        if moments is None:
            ledger.admit('distributed moments', 0, 2000*p*q)
            moments = analytic_df_moments(recipe, sites, 4)
            supplied = 0
        else:
            supplied = moments.values.nbytes
        moments.validate_for(recipe, sites, 4, self.state_sha256)
        self.partition = _partition_record(moments)
        # FDDS contracts Q at every node: an owned copy is retained and charged.
        ledger.admit('retained distributed moments', 8*q*p+supplied)
        values = np.array(moments.values, dtype=np.float64, order='C', copy=True)
        del moments
        if values.shape != (q, p) or not np.isfinite(values).all():
            raise ValueError('distributed moments must be finite (25*nsite, naux)')
        self._moments = values
        self._hold(moments=values.nbytes)
        del values
        log.stage('Native plain density', (('AUX shells', len(recipe.shells)),))
        main, coefficients = self._main, self._coefficients
        nao, nmo = main.basis.nfunction, coefficients.shape[1]
        ledger.admit('native plain density work reservation', 0, 2*p*nao*nmo*(nao+nmo)+p**3)
        density, planned = native_plain_density(auxiliary, main.basis, coefficients, nocc=no,
                                                shell_count=len(recipe.shells),
                                                max_bytes=resources.max_bytes-ledger.reserved)
        ledger.admit('native plain density', planned)
        del main, coefficients
        self._state = self._main = self._coefficients = self._energies = None
        self._hold(state=0)
        grid, self._grid = self._grid, None
        log.stage('Full-grid ALDA kernel', (('smoothing', repr(self.smoothing)), ('shell cutoff', self.shell_cutoff)))
        kernel = _kernel(auxiliary, density, grid, self.smoothing, self.shell_cutoff, ledger)
        del grid, density
        self._hold(grid=0, kernel=kernel.nbytes)

        plan = self._fdds_plan
        memory = plan['requirement']['memory_bytes']
        primary, raw, transform, cocc, cvir, eocc, evir = self._fdds_inputs
        log.stage('Native FDDS instance', (('memory bytes', memory),
                                           ('peak scratch bytes', plan['requirement']['disk_bytes']),
                                           ('scratch allowance', fdds.disk_bytes), ('threads', fdds.nthread),
                                           ('subalgo', fdds.subalgo)))
        ledger.admit('native FDDS instance', memory, plan['estimates']['construction_work'])
        ledger.charge_io(plan['estimates']['construction_io_bytes'])
        directory = Path(self._temporary.name)/'fdds'
        directory.mkdir()
        self._fdds = core.FDDS_Monomer(primary, raw, cocc, cvir, eocc, evir, True, memory_bytes=memory,
                                       disk_bytes=fdds.disk_bytes, scratch_dir=str(directory),
                                       nthread=fdds.nthread, subalgo=fdds.subalgo, aux_transform=transform)
        self._hold(native=memory)
        self._fdds_inputs = None
        del primary, raw, transform, cocc, cvir, eocc, evir
        self._hold(native_inputs=0)
        model = dict(self._fdds.model())
        rcond = float(model['metric_lu_rcond'])
        self.provenance.update(native_model=model, metric_guard=dict(
            metric_lu_rcond=rcond, cutoff=FDDS_METRIC_RCOND_CUTOFF,
            meaning='LAPACK 1-norm reciprocal condition estimate of J_d; heuristic separator, not an accuracy bound'))
        log.items((('metric_lu_rcond', rcond), ('metric guard cutoff', FDDS_METRIC_RCOND_CUTOFF),
                   ('metric dropped / max dropped / min kept',
                    (model['metric_dropped'], model['metric_max_dropped'], model['metric_min_kept'])),
                   ('QR rank', model['qr_rank'])))
        if not rcond >= FDDS_METRIC_RCOND_CUTOFF:
            self._fdds = None
            self._hold(native=0)
            raise ValueError(f'native FDDS metric resolution guard: metric_lu_rcond = {rcond:.6e} is below '
                             f'the cutoff {FDDS_METRIC_RCOND_CUTOFF:.0e}; the reference H2H1 response is unaffected')

        log.stage('FDDS kernel W = J_d + 0.75 K')
        ledger.admit('FDDS kernel W', 8*4*p*p)
        metric = self._fdds.metric().to_array()
        reference = np.asarray(core.IsaAuxCoulomb(auxiliary).metric())
        self.provenance['metric_minus_isa_aux_coulomb_max_abs'] = float(np.max(np.abs(metric-reference)))
        del reference
        kernel *= .75
        kernel += metric
        del metric
        if not np.isfinite(kernel).all():
            raise ValueError('nonfinite FDDS kernel W')
        self._fdds_kernel = core.Matrix.from_array(kernel)
        del kernel
        # The sweep holds the caller allowances, Q, the native instance and W.
        self._held = dict(retained=self._retained, moments=self._moments.nbytes, native=memory, W=8*p*p)
        ledger.reserved = sum(self._held.values())
        log.items((('metric - IsaAuxCoulomb.metric() max abs', self.provenance['metric_minus_isa_aux_coulomb_max_abs']),
                   ('reserved bytes', ledger.reserved)))
        return self

    def _solve_fdds(self, omega, tag):
        p, q = self.dimensions[0], 25*len(self.sites)
        plan = self._fdds_plan
        self.log.stage('Native FDDS response: ' + tag, (('x_alpha', .25), ('kernel', 'J_d + 0.75 K')))
        self.ledger.admit('FDDS frequency response', plan['node_bytes'], plan['node_work'])
        self.ledger.charge_io(plan['estimates']['node_io_bytes'])
        result = self._fdds.form_coefficient_response(omega, .25, self._fdds_kernel)
        chi = result['response'].to_array()
        diagnostics = (('native_dyson_ratio', float(result['native_dyson_ratio'])),
                       ('s2_dyson_ratio', float(result['s2_dyson_ratio'])),
                       ('solve_residual', float(result['solve_residual'])),
                       ('masked_transitions', int(result['masked_transitions'])),
                       ('metric_lu_rcond', self.provenance['metric_guard']['metric_lu_rcond']),
                       ('metric_lu_rcond_cutoff', FDDS_METRIC_RCOND_CUTOFF))
        del result
        if chi.shape != (p, p) or not np.isfinite(chi).all() or not all(np.isfinite(v) for _, v in diagnostics):
            raise ValueError('native FDDS response or diagnostics are not finite')
        half = self._moments @ chi
        product = half @ self._moments.T
        del half
        nonlocal_response = np.add(product, product.T)
        del product
        nonlocal_response *= -.5
        if not np.isfinite(nonlocal_response).all():
            raise ValueError('nonfinite FDDS nonlocal response')
        residual = diagnostics[2][1]
        self.log.items(diagnostics)
        return FrequencyResponse(omega, chi, nonlocal_response, residual, self.model, diagnostics)

    def release(self):
        """Return the retained operator (or native instance) allowance; later solves are refused."""
        if self._operator is not None:
            self._operator.close()
            self._operator = None
        if self._fdds is not None:
            self._fdds = self._fdds_kernel = self._moments = None
            self._held = dict(retained=self._retained)
            self.ledger.reserved = self._retained
        self._released = True
