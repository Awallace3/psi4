# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Bounded complete-cloud PFIT: packed design rows and the streamed solve."""

import numpy as np

from psi4 import core

from . import isapol_refine as _refine
from .isapol_refine import RefinementModel


class PackedDesignRows:
    """Replay deterministic per-point design contractions for one complete cloud.

    Supplied fields have shape (npoint, model.channel_count) and already
    include the caller's frames and damping. Parameters retain model order,
    COPY entries, and tensor off-diagonal symmetry. No off-diagonal *point*
    weight or half-factor is introduced.

    Blocks are computational only: (start, rows) follows global packed order
    i*(i+1)//2+j, j<=i, including cross-block point pairs. Used field channels
    and their contraction are snapshotted, so each replay pass is bitwise
    identical. Returned blocks are independent and mutable; retaining them is
    the consumer's budget.

    Admission covers input/snapshot overlap, validation, the contraction and
    bounded row workspaces, not target generation, solver state or runtime
    overhead; arithmetic is bounded across all declared passes. Nonfinite rows
    raise ``FloatingPointError``. Requesting a pass charges it immediately and
    there is no reset. Target provenance and the solve are separate contracts.
    """

    MAX_BYTES = 512*1024**2
    MAX_WORK = 64_000_000_000
    MAX_CHUNK = 64

    def __init__(self, model, fields, *, block_rows=256, max_passes=2,
                 max_bytes=MAX_BYTES):
        if not isinstance(model, RefinementModel):
            raise ValueError("explicit RefinementModel required")
        if type(block_rows) is not int or not 1 <= block_rows <= 4096:
            raise ValueError("block_rows must be an integer in [1,4096]")
        if type(max_passes) is not int or not 1 <= max_passes <= 2:
            raise ValueError("max_passes must be 1 or 2")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if (not isinstance(fields, np.ndarray) or fields.dtype != np.float64
                or fields.ndim != 2 or fields.shape[0] < 1
                or fields.shape[1] != model.channel_count or model.parameter_count < 1):
            raise ValueError("nonempty float64 fields and at least one model parameter required")
        npoint, channels = fields.shape
        nparam = model.parameter_count
        if len(model.parameter_entries) != nparam or any(not e for e in model.parameter_entries):
            raise ValueError("parameter entry table must match nonempty model columns")
        if len(model.channel_offsets) != len(model.sites):
            raise ValueError("channel offsets must match model sites")
        expected_offset = 0
        for site, offset in zip(model.sites, model.channel_offsets):
            if type(offset) is not int or offset != expected_offset:
                raise ValueError("model channel offsets must be contiguous integer site offsets")
            expected_offset += site.component_count
        if expected_offset != channels:
            raise ValueError("model site widths must match channel count")
        row_count = npoint*(npoint+1)//2
        block = min(block_rows, row_count)
        indices = []
        for entries in model.parameter_entries:
            pairs = []
            for entry in entries:
                if not isinstance(entry, (tuple, list)) or len(entry) != 3:
                    raise ValueError("invalid model parameter entry")
                site, row, col = entry
                if (any(type(v) is not int for v in (site, row, col))
                        or not 0 <= site < len(model.sites)
                        or not 0 <= row <= col < model.sites[site].component_count):
                    raise ValueError("invalid model parameter entry")
                offset = model.channel_offsets[site]
                if not 0 <= offset <= channels-model.sites[site].component_count:
                    raise ValueError("invalid model channel offset")
                pairs.append((offset+row, offset+col))
            indices.append(tuple(pairs))
        # Expanded (left, right) channel terms per parameter. Tensor
        # off-diagonal entries contribute both orderings.
        terms = [(parameter, row, col) for parameter, pairs in enumerate(indices)
                 for r, c in pairs for row, col in (((r, c), (c, r)) if r != c else ((r, c),))]
        used = sorted({col for _, _, col in terms})
        position = {channel: k for k, channel in enumerate(used)}
        # A pass evaluates `chunk` points' runs as one BLAS product; the chunk
        # is tied to the block size so small blocks keep a small workspace.
        chunk = max(1, min(self.MAX_CHUNK, 16*block//npoint))
        # Caller input and its isfinite mask, the owned used-channel fields,
        # the point contraction H, one chunk product per declared pass, and
        # one shared block plus each yielded copy.
        planned = (fields.nbytes + fields.size + 8*npoint*(len(used)*(nparam+1)+1)
                   + max_passes*8*(npoint*chunk*nparam+block*(2*nparam+12)) + 256)
        if planned > max_bytes:
            raise ValueError("packed design byte resource limit exceeded")
        # Dense (used channels x parameters) products, including the unused
        # j > i corner of the widest possible chunk, plus writes. H is formed
        # once. Charge the entire replay contract.
        work = ((row_count+self.MAX_CHUNK*npoint)*2*len(used)*nparam
                + row_count*(4*nparam+8))
        if max_passes*work > self.MAX_WORK:
            raise ValueError("packed design replay work resource limit exceeded")
        if not np.isfinite(fields).all():
            raise ValueError("fields must be finite")
        # Row (i, j) of parameter p is F[j] @ H[:, i, p] with
        # H[b, i, p] = sum of F[i, a] over the terms (p, a, b). The operands are
        # owned and shared, and each pass's product buffer is 64-byte aligned,
        # so replay passes repeat the BLAS result bit for bit.
        self._fields = _aligned((npoint, len(used)))
        self._fields[:] = fields[:, used]
        self._contraction = _aligned((len(used), npoint, nparam))
        self._contraction[:] = 0.
        with np.errstate(over="raise", invalid="raise"):
            for parameter, row, col in terms:
                self._contraction[position[col], :, parameter] += fields[:, row]
        self._products = [_aligned((npoint*chunk*nparam,)) for _ in range(max_passes)]
        self._chunk = chunk
        self._block = np.empty((block, nparam))
        self.row_count, self.parameter_count = row_count, nparam
        self.planned_bytes, self.planned_work = planned, max_passes*work
        self._block_rows, self._max_passes = block, max_passes
        self._passes_used = 0

    @property
    def passes_used(self):
        return self._passes_used

    def blocks(self):
        """Start one charged pass; yield bounded (global_start, design) blocks."""
        if self._passes_used >= self._max_passes:
            raise RuntimeError("packed design replay pass budget exhausted")
        self._passes_used += 1
        return self._iterate(self._products[self._passes_used-1])

    def _iterate(self, products):
        # Point i owns the contiguous packed run j = 0..i; a block may split a
        # run or span several. Each block is filled before it is yielded, so
        # interleaved passes can share the block buffer.
        fields, contraction, buffer = self._fields, self._contraction, self._block
        npoint, nparam = len(fields), self.parameter_count
        i = j = 0
        first = last = 0
        for start in range(0, self.row_count, self._block_rows):
            count = min(self._block_rows, self.row_count-start)
            k = 0
            while k < count:
                if i >= last:
                    first, last = i, min(i+self._chunk, npoint)
                    width = (last-first)*nparam
                    product = products[:last*width].reshape(last, width)
                    with np.errstate(over="raise", invalid="raise"):
                        np.matmul(fields[:last], contraction[:, first:last].reshape(-1, width),
                                  out=product)
                n = min(i+1-j, count-k)
                column = (i-first)*nparam
                buffer[k:k+n] = product[j:j+n, column:column+nparam]
                k, j = k+n, j+n
                if j > i:
                    i, j = i+1, 0
            result = buffer[:count].copy()
            if not np.isfinite(result).all():
                raise FloatingPointError("overflow in packed design row")
            yield start, result


def _aligned(shape):
    """Uninitialized float64 array whose data starts on a 64-byte boundary."""
    size = int(np.prod(shape))
    raw = np.empty(size+8)
    offset = (-raw.ctypes.data % 64)//8
    return raw[offset:offset+size].reshape(shape)


def fitted_point_targets(auxiliary, points_bohr, coefficient_responses, *,
                         max_bytes=PackedDesignRows.MAX_BYTES):
    """Packed point-response targets ``v = -P^T C P`` of fitted AUX responses.

    ``auxiliary`` is the declared response AUX (``core.IsaExplicitBasis``) and
    each ``C`` a p x p coefficient response in it, e.g. the ``target_response``
    of a bounded response node.  ``P[k, i]`` is AUX function ``k``'s exact
    positive Coulomb potential at point ``i`` (``IsaAuxCoulomb.point_potentials``,
    32 points per call).  Column ``t`` of the returned float64
    ``(npoint*(npoint+1)//2, len(coefficient_responses))`` array is the lower
    triangle of ``-P^T C[t] P``, packed for :func:`refine_streamed` with origin
    ``NativeFittedPointResponse`` and representation
    ``fitted_density_coefficients``.  Nothing is symmetrized or rescaled.

    ``max_bytes`` must cover the point copies, P, one product and its dense
    target, one potential block and the packed result; it is checked before
    any of them is allocated.  Libint workspace is not charged.
    """
    if not isinstance(auxiliary, core.IsaExplicitBasis):
        raise ValueError('auxiliary must be a core.IsaExplicitBasis')
    points = np.array(points_bohr, dtype=float, copy=True)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
        raise ValueError('points_bohr must be a nonempty npoint x 3 array')
    responses = tuple(coefficient_responses)
    naux, npoint, nt = auxiliary.nfunction, len(points), len(responses)
    if nt < 1:
        raise ValueError('at least one coefficient response is required')
    for response in responses:
        if (not isinstance(response, np.ndarray) or response.dtype != np.float64
                or response.shape != (naux, naux) or not np.all(np.isfinite(response))):
            raise ValueError(f'each coefficient response must be a finite float64 '
                             f'{naux} x {naux} array')
    row_count = npoint*(npoint+1)//2
    block = min(32, npoint)
    # Points and one Matrix block of them, P, P^T C, the dense target, one
    # potential block and the packed result.
    planned = 8*(3*npoint+3*block+2*naux*npoint+npoint*npoint+naux*block+row_count*nt)
    if type(max_bytes) is not int or planned > max_bytes:
        raise ValueError(f'fitted point targets byte resource limit exceeded ({planned} > {max_bytes})')
    provider = core.IsaAuxCoulomb(auxiliary)
    potentials = np.empty((naux, npoint))
    for start in range(0, npoint, block):
        potentials[:, start:start+block] = provider.point_potentials(
            core.Matrix.from_array(points[start:start+block])).np
    packed = np.empty((row_count, nt))
    for t, response in enumerate(responses):
        target = -(potentials.T @ response) @ potentials
        start = 0
        for i in range(npoint):
            packed[start:start+i+1, t] = target[i, :i+1]
            start += i+1
    return packed


def refine_streamed(models, points_bohr, packed_targets, *, target_origin=None, source_id,
                    generation_record, response_representation='', auxiliary_basis_id='',
                    fields=None, damping=0.0, block_rows=4096,
                    max_bytes=PackedDesignRows.MAX_BYTES, options=None):
    """Refine several targets over one complete cloud and one shared design.

    ``models`` (typically one per frequency) must share sites, channels and
    variables; only anchors, strengths and frequency differ.  Column ``t`` of
    ``packed_targets`` (``npoint*(npoint+1)//2`` rows in the packing of
    :func:`isapol_refine.pack_lower_triangle`) belongs to ``models[t]``, and all
    columns share one declared target provenance.  The design is replayed twice
    by :class:`PackedDesignRows` and solved by ``core.isa_pfit_solve_rows_multi``
    with ``NormalEquationsDSYSV`` unless ``options`` says otherwise, so each
    result is the dense :func:`isapol_refine.refine` of its column up to
    rounding, without that function's point cap.

    Resources: ``max_bytes`` bounds this function's planned Python buffers (the
    row producer's plan, the point copy, computed fields and block-sized target
    copies) and ``options.maximum_work_bytes`` the native kernel's; caller
    ``fields`` are charged by the row producer, ``packed_targets`` stay the
    caller's.  Targets are copied block by block and the replay digest refuses
    any change between the two passes.  Any fit that is not ``Solved`` raises
    ``ValueError``.
    """
    models = tuple(models)
    if not models or any(not isinstance(m, RefinementModel) for m in models):
        raise ValueError('a nonempty sequence of RefinementModel is required')
    model = models[0]
    for other in models[1:]:
        if (other.sites != model.sites or other.channel_labels != model.channel_labels
                or other.parameter_labels != model.parameter_labels
                or other.parameter_entries != model.parameter_entries):
            raise ValueError('models sharing one design must share sites, channels and variables')
    provenance = _refine.target_provenance(target_origin, source_id=source_id,
                                           generation_record=generation_record,
                                           response_representation=response_representation,
                                           auxiliary_basis_id=auxiliary_basis_id)
    points = np.array(points_bohr, dtype=float, copy=True)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
        raise ValueError('points_bohr must be a nonempty npoint x 3 array')
    if not np.all(np.isfinite(points)):
        raise ValueError('points_bohr must be finite')
    npoint, nt = len(points), len(models)
    row_count = npoint*(npoint+1)//2
    targets = np.asarray(packed_targets)
    if targets.dtype != np.float64 or targets.shape != (row_count, nt):
        raise ValueError(f'packed_targets must be float64 of shape ({row_count}, {nt})')
    if type(block_rows) is not int or block_rows < 1:
        raise ValueError('block_rows must be a positive integer')
    # Beyond the row producer's own plan: the point copy, the computed fields,
    # one block-sized finiteness mask, and two live target-block copies.
    block = min(block_rows, row_count)
    overhead = 24*npoint+(8*npoint*model.channel_count if fields is None else 0)+17*block*nt
    if type(max_bytes) is not int or overhead >= max_bytes:
        raise ValueError('streamed refinement byte resource limit exceeded')
    for start in range(0, row_count, block):
        if not np.isfinite(targets[start:start+block]).all():
            raise ValueError('packed_targets must be finite')
    if fields is None:
        # channel_fields caps one call at MAX_POINTS; rows are independent.
        fields = np.empty((npoint, model.channel_count))
        for start in range(0, npoint, 256):
            fields[start:start+256] = _refine.channel_fields(points[start:start+256], model,
                                                             damping=damping)
    rows = PackedDesignRows(model, fields, block_rows=block_rows, max_passes=2,
                            max_bytes=max_bytes-overhead)

    declaration = core.IsaPfitRowModel()
    declaration.channel_labels = list(model.channel_labels)
    declaration.parameter_labels = list(model.parameter_labels)
    declaration.parameter_units = _refine.parameter_units(model)
    declaration.fixed = [False]*model.parameter_count
    declaration.fixed_values = [0.]*model.parameter_count
    declaration.provenance = model.provenance
    cloud = core.IsaPfitCloudRows()
    cloud.label, cloud.points, cloud.full_row_count = 'complete cloud', npoint, row_count
    cloud.maximum_block_rows = block_rows
    problems = []
    for m in models:
        problem = core.IsaPfitRowProblem()
        problem.frequency_au = m.frequency_au
        problem.model, problem.cloud, problem.target_provenance = declaration, cloud, provenance
        problem.penalty = _refine.matrix_penalty(m)
        problems.append(problem)

    def blocks():
        for start, design in rows.blocks():
            yield start, design, np.array(targets[start:start+len(design)], order='C')

    if options is None:
        options = core.IsaPfitOptions()
        options.solver = core.IsaPfitSolver.NormalEquationsDSYSV
    results = core.isa_pfit_solve_rows_multi(problems, blocks, options)
    for m, result in zip(models, results):
        if result.status != core.IsaPfitStatus.Solved:
            raise ValueError(f'PFIT did not solve at frequency {m.frequency_au!r}: {result.status}')
    return tuple(_refine.refinement_result(m, r) for m, r in zip(models, results))
