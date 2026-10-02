# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Stage-record reporting; never reruns or mutates a scientific stage.

StageLog verbosity: 0 silent, 1 banners/results, 2 diagnostics, 3 node detail.
Long tables are elided and wide rows refused.

QCVariable arrays use core.Matrix to avoid name-driven ndarray reshaping.
QCVariables are not part of the wavefunction's SCF/state seals.
"""
import dataclasses
import time

import numpy as np

from psi4 import core

#: Longest table body printed in full; longer tables are elided in the middle
#: (both ends are kept) with an explicit count of the omitted rows.
MAX_TABLE_ROWS = 60
#: A row wider than this is bulk data, not a property table. Refused outright:
#: reaching it means a caller handed a raw intermediate to the formatter.
MAX_ROW_CELLS = 24


def _fmt(value):
    """Render one cell. Wide sequences are summarized, never expanded."""
    if value is None:
        return 'none'
    if isinstance(value, (bool, np.bool_)):
        return 'true' if value else 'false'
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if not np.isfinite(v):
            return repr(v)
        if v == 0.:
            return '0'
        return ('%.6f' % v) if 1.e-4 <= abs(v) < 1.e6 else ('%.6e' % v)
    if isinstance(value, np.ndarray):
        return f'<array{tuple(value.shape)} not printed>'
    if isinstance(value, (tuple, list)):
        if len(value) > MAX_ROW_CELLS:
            return f'<{len(value)} entries not printed>'
        return '(' + ', '.join(_fmt(v) for v in value) + ')'
    return str(value)


class StageLog:
    """Verbosity-gated writer with stage banners, wall clocks and small tables.

    Holds no wavefunction and no scientific record. ``writer`` exists so tests
    can capture the text without touching the Psi4 output file; the production
    writer is ``core.print_out``.
    """

    def __init__(self, verbosity=0, writer=None, indent=2):
        if type(verbosity) is not int or isinstance(verbosity, bool) or verbosity < 0:
            raise ValueError('logging verbosity must be an explicit nonnegative integer')
        if type(indent) is not int or indent < 0:
            raise ValueError('indent must be a nonnegative integer')
        self._level = int(verbosity)
        self._writer = core.print_out if writer is None else writer
        if not callable(self._writer):
            raise TypeError('writer must be callable')
        self._pad = ' ' * int(indent)
        self._open = None
        self._started = None
        self._stages = []

    @property
    def verbosity(self):
        return self._level

    @property
    def stages(self):
        """``(name, seconds)`` for every stage closed so far; timings, not results."""
        return tuple(self._stages)

    def enabled(self, level=1):
        return self._level >= int(level)

    def line(self, text='', level=1, indent=0):
        if self.enabled(level):
            self._writer((self._pad + ' ' * int(indent) + str(text) + '\n') if text != '' else '\n')

    def banner(self, title, level=1):
        if self.enabled(level):
            self._writer('\n' + self._pad + '==> ' + str(title) + ' <==\n\n')

    def items(self, pairs, level=1, indent=4):
        """Aligned ``name  value`` block; the stage-parameter rendering."""
        if not self.enabled(level):
            return
        rendered = [(str(k), _fmt(v)) for k, v in pairs]
        if not rendered:
            return
        width = max(len(k) for k, _ in rendered)
        for k, v in rendered:
            self._writer(self._pad + ' ' * int(indent) + k.ljust(width) + '   ' + v + '\n')

    def table(self, title, headers, rows, level=1, indent=4, note=None):
        """Print a small aligned table, eliding the middle of a long body.

        Truncation is announced with the omitted row count; a row wider than
        :data:`MAX_ROW_CELLS` is refused, because that is a raw intermediate.
        """
        headers = [str(h) for h in headers]
        if len(headers) > MAX_ROW_CELLS:
            raise ValueError('refusing to print a wide intermediate as a property table')
        rows = list(rows)
        for row in rows:
            if len(row) != len(headers):
                raise ValueError('table rows must match the declared header count')
        if not self.enabled(level):
            return
        omitted = 0
        if len(rows) > MAX_TABLE_ROWS:
            head = MAX_TABLE_ROWS // 2
            omitted = len(rows) - MAX_TABLE_ROWS
            rows = rows[:head] + rows[len(rows) - (MAX_TABLE_ROWS - head):]
        body = [[_fmt(c) for c in row] for row in rows]
        widths = [max([len(h)] + [len(r[i]) for r in body]) for i, h in enumerate(headers)]
        pad = self._pad + ' ' * int(indent)
        if title:
            self._writer(pad + str(title) + '\n')
        rule = '  '.join('-' * w for w in widths)
        self._writer(pad + '  '.join(h.rjust(w) for h, w in zip(headers, widths)) + '\n')
        self._writer(pad + rule + '\n')
        for i, row in enumerate(body):
            if omitted and i == MAX_TABLE_ROWS // 2:
                self._writer(pad + f'... {omitted} intermediate rows not printed; the full record is '
                             'available through atomic_property_result(wfn)\n')
            self._writer(pad + '  '.join(c.rjust(w) for c, w in zip(row, widths)) + '\n')
        self._writer(pad + rule + '\n')
        if note:
            self._writer(pad + str(note) + '\n')

    def stage(self, name, parameters=(), level=1):
        """Close any open stage, then announce this one with all its parameters."""
        self.stage_end(level=level)
        self._open, self._started = str(name), time.time()
        self.banner('Stage: ' + str(name), level=level)
        if parameters:
            self.line('Parameters (every tweakable input of this stage):', level=level)
            self.items(parameters, level=level)
            self.line(level=level)

    def stage_end(self, level=1):
        """Close the open stage, recording and printing its wall clock."""
        if self._open is None:
            return
        name, seconds = self._open, time.time() - self._started
        self._stages.append((name, float(seconds)))
        self._open, self._started = None, None
        self.line('Stage complete: %s (%.2f s)' % (name, seconds), level=level)


def silent():
    """The default log: accepts every call, writes nothing, records timings."""
    return StageLog(0, writer=lambda text: None)


def dataclass_parameters(obj, *, prefix='', skip=()):
    """Every declared field of a recipe dataclass, in declaration order.

    Enumerating the dataclass rather than a hand-written list is the point: a
    stage banner that claims to list all tweakable parameters cannot silently
    omit one that was added later. Bulk fields are reported by size.
    """
    out = []
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        if isinstance(value, (tuple, list)) and len(value) > MAX_ROW_CELLS:
            value = f'<{len(value)} entries>'
        if field.name in skip:
            continue
        out.append((prefix + field.name, value))
    return tuple(out)


def _matrix(array):
    """Wrap for ``set_variable``: bypasses name-based ndarray reshaping."""
    return core.Matrix.from_array(np.ascontiguousarray(np.asarray(array, dtype=float)))


def _set(wfn, key, value):
    if wfn is None:
        return
    wfn.set_variable(key, value)
