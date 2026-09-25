# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Bounded complete-cloud design rows, not yet a streaming PFIT solver."""

import numpy as np

from .isapol_refine import RefinementModel


class PackedDesignRows:
    """Replay deterministic per-point design contractions for one complete cloud.

    Supplied fields have shape (npoint, model.channel_count) and already
    include the caller's frames and damping. Parameters retain model order,
    COPY entries, and tensor off-diagonal symmetry. No off-diagonal *point*
    weight or half-factor is introduced.

    Blocks are computational only: (start, rows) follows global packed order
    i*(i+1)//2+j, j<=i, including cross-block point pairs. Used field channels
    and their per-point entry contraction are snapshotted, and each replay
    pass reproduces the previous one bit for bit. Returned blocks are independent and mutable.
    Retaining old blocks is the consumer's budget responsibility.

    Admission covers input/snapshot overlap, validation, the contraction and
    bounded row workspaces; it does not cover target generation, solver state, or runtime
    overhead. Arithmetic is conservatively bounded across all declared passes.
    Nonfinite rows fail as ``FloatingPointError``. Requesting a pass charges
    it immediately, even if iteration is abandoned;
    the object has no reset. This supplies design rows only: replayable target
    provenance and one global PFIT solve remain separate contracts. Existing
    dense point/batch gates are not changed.
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
