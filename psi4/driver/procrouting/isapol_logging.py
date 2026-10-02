# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Stage-record reporting; never reruns or mutates a scientific stage.

StageLog verbosity: 0 silent, 1 banners/results, 2 diagnostics, 3 node detail.
"""
import time

import numpy as np

from psi4 import core

#: A sequence longer than this is bulk data and is summarized, never printed.
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
    """Verbosity-gated writer with stage banners, wall clocks and aligned items.

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

