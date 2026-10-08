"""Contract tests for the stage log: verbosity gating, banners and bulk-data summaries."""

import numpy as np
import pytest

from psi4.driver.procrouting import isapol_logging as lg

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


class Capture:
    """Writer that keeps the text instead of touching the Psi4 output file."""

    def __init__(self):
        self.chunks = []

    def __call__(self, text):
        self.chunks.append(text)

    @property
    def text(self):
        return ''.join(self.chunks)


def _log(verbosity):
    cap = Capture()
    return lg.StageLog(verbosity, writer=cap), cap


# ------------------------------------------------------------------ _fmt ----

@pytest.mark.parametrize('value,want', [
    (None, 'none'),
    (True, 'true'),
    (False, 'false'),
    (3, '3'),
    (np.int64(-7), '-7'),
    (0., '0'),
    (1.5, '1.500000'),
    (1.e-9, '1.000000e-09'),
    (1.e7, '1.000000e+07'),
    (float('inf'), 'inf'),
    (float('nan'), 'nan'),  # reported, not hidden
    ((1, 2.5, None), '(1, 2.500000, none)'),
    ('ISA_A', 'ISA_A'),
])
def test_fmt_renders_each_cell_kind(value, want):
    assert lg._fmt(value) == want


def test_fmt_never_expands_bulk_data():
    """An array or a long sequence is summarized by size, never printed."""
    assert lg._fmt(np.zeros((512, 3))) == '<array(512, 3) not printed>'
    assert lg._fmt(tuple(range(lg.MAX_ROW_CELLS + 1))) == f'<{lg.MAX_ROW_CELLS + 1} entries not printed>'
    assert lg._fmt(tuple(range(lg.MAX_ROW_CELLS))).startswith('(0, 1,')


# -------------------------------------------------------------- StageLog ----

def test_silent_log_still_accepts_every_call():
    log, cap = _log(0)
    assert log.verbosity == 0
    assert not log.enabled(1)
    log.banner('x')
    log.line('y')
    log.items((('a', 1),))
    log.stage('s', (('p', 1),))
    log.stage_end()
    assert cap.text == ''
    assert [name for name, _ in log.stages] == ['s']


def test_verbosity_gates_each_level():
    for level in (0, 1, 2, 3):
        log, cap = _log(level)
        for requested in (1, 2, 3):
            log.line(f'L{requested}', level=requested)
        for requested in (1, 2, 3):
            assert (f'L{requested}' in cap.text) == (level >= requested)


def test_verbosity_must_be_an_explicit_nonnegative_integer():
    with pytest.raises(ValueError):
        lg.StageLog(-1)
    with pytest.raises(ValueError):
        lg.StageLog(True)
    with pytest.raises(ValueError):
        lg.StageLog(1.0)
    with pytest.raises(TypeError):
        lg.StageLog(1, writer='not callable')


def test_items_are_name_aligned():
    log, cap = _log(1)
    log.items((('short', 1), ('a much longer name', 2)))
    columns = {line.index('   ' + line.strip().split()[-1]) for line in cap.text.splitlines()}
    assert len(columns) == 1


def test_stage_banner_lists_parameters_and_closes_the_previous_stage():
    log, cap = _log(1)
    log.stage('first', (('alpha', .25),))
    log.stage('second')
    log.stage_end()
    assert '==> Stage: first <==' in cap.text
    assert 'Parameters (every tweakable input of this stage):' in cap.text
    assert 'alpha' in cap.text and '0.250000' in cap.text
    assert 'Stage complete: first' in cap.text
    assert 'Stage complete: second' in cap.text
    assert [name for name, _ in log.stages] == ['first', 'second']
    assert all(seconds >= 0. for _, seconds in log.stages)


def test_stage_end_without_an_open_stage_is_a_noop():
    log, cap = _log(1)
    log.stage_end()
    assert cap.text == ''
