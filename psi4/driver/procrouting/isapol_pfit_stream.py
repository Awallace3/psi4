# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Bounded complete-cloud design rows, not yet a streaming PFIT solver."""

import numpy as np

from .isapol_refine import RefinementModel


class PackedDesignRows:
    """Replay exact sparse-entry contractions for one complete point cloud.

    Supplied fields have shape (npoint, model.channel_count) and already
    include the caller's frames and damping. Parameters retain model order,
    COPY entries, and tensor off-diagonal symmetry. No off-diagonal *point*
    weight or half-factor is introduced.

    Blocks are computational only: (start, rows) follows global packed order
    i*(i+1)//2+j, j<=i, including cross-block point pairs. Fields and entry
    indices are snapshotted. Returned blocks are independent and mutable.
    Retaining old blocks is the consumer's budget responsibility.

    Admission covers input/snapshot overlap, validation and bounded row
    workspaces; it does not cover target generation, solver state, or runtime
    overhead. Arithmetic is conservatively bounded across all declared passes.
    Requesting a pass charges it immediately, even if iteration is abandoned;
    the object has no reset. This supplies design rows only: replayable target
    provenance and one global PFIT solve remain separate contracts. Existing
    dense point/batch gates are not changed.
    """

    MAX_BYTES = 512*1024**2
    MAX_WORK = 64_000_000_000

    def __init__(self, model, fields, *, block_rows=256, max_passes=2,
                 max_bytes=MAX_BYTES):
        if not isinstance(model, RefinementModel):
            raise ValueError("explicit RefinementModel required")
        if type(block_rows) is not int or not 1 <= block_rows <= 4096:
            raise ValueError("block_rows must be an integer in [1,4096]")
        if type(max_passes) is not int or not 1 <= max_passes <= 2:
            raise ValueError("max_passes must be 1 or 2")
        if type(max_bytes) is not int or not 0 < max_bytes <= self.MAX_BYTES:
            raise ValueError("max_bytes must be a positive integer at most 512 MiB")
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
        # Reserve every declared iterator's block workspace, even if callers
        # interleave replay passes. Retained output histories remain external.
        planned = 3*fields.nbytes + max_passes*8*block*(2*nparam+12)
        if planned > max_bytes:
            raise ValueError("packed design byte resource limit exceeded")
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
        # Two products for tensor off-diagonal entries, accumulations and
        # conservative indexing overhead. Charge the entire replay contract.
        work = row_count*(6*sum(map(len, indices))+4*nparam+8)
        if max_passes*work > self.MAX_WORK:
            raise ValueError("packed design replay work resource limit exceeded")
        if not np.isfinite(fields).all():
            raise ValueError("fields must be finite")
        self._fields = np.frombuffer(fields.tobytes(), dtype=np.float64).reshape(fields.shape)
        self._indices = tuple(indices)
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
        return self._iterate()

    def _iterate(self):
        i = j = 0
        fields = self._fields
        for start in range(0, self.row_count, self._block_rows):
            count = min(self._block_rows, self.row_count-start)
            left = np.empty(count, dtype=np.int64)
            right = np.empty(count, dtype=np.int64)
            for k in range(count):
                left[k], right[k] = i, j
                j += 1
                if j > i:
                    i, j = i+1, 0
            result = np.zeros((count, self.parameter_count))
            with np.errstate(over="raise", invalid="raise"):
                for parameter, entries in enumerate(self._indices):
                    for row, col in entries:
                        result[:, parameter] += fields[left, row]*fields[right, col]
                        if row != col:
                            result[:, parameter] += fields[left, col]*fields[right, row]
            if not np.isfinite(result).all():
                raise ValueError("nonfinite packed design row")
            yield start, result
