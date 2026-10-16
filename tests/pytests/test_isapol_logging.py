"""Contract tests for the stage log: verbosity gating, banners, tables, bulk-data summaries and QCVariable wrapping."""

import dataclasses

import numpy as np
import pytest

import psi4
from psi4 import core
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


def test_default_log_is_silent_and_accepts_tables():
    log = lg.silent()
    assert log.verbosity == 0 and not log.enabled(1)
    log.table('t', ('h',), [(1,)])
    log.stage('s', (('p', 1),))
    log.stage_end()
    assert [name for name, _ in log.stages] == ['s']


# ------------------------------------------------- no large intermediates ----

def test_table_refuses_a_wide_row_even_when_silent():
    """A row wider than MAX_ROW_CELLS is a raw intermediate, not a property."""
    headers = tuple('c%d' % i for i in range(lg.MAX_ROW_CELLS + 1))
    for verbosity in (0, 3):
        log, _ = _log(verbosity)
        with pytest.raises(ValueError, match='wide intermediate'):
            log.table('t', headers, [tuple(range(len(headers)))])


def test_table_refuses_rows_that_do_not_match_the_headers():
    log, _ = _log(1)
    with pytest.raises(ValueError, match='header count'):
        log.table('t', ('a', 'b'), [(1, 2), (3,)])


def test_table_elides_the_middle_of_a_long_body_and_says_so():
    log, cap = _log(1)
    n = lg.MAX_TABLE_ROWS + 17
    log.table('t', ('i',), [(i,) for i in range(n)])
    assert '... 17 intermediate rows not printed' in cap.text
    printed = {int(line.strip()) for line in cap.text.splitlines()
               if line.strip().isdigit()}
    assert len(printed) == lg.MAX_TABLE_ROWS
    assert 0 in printed and n - 1 in printed


def test_short_table_is_printed_in_full_with_no_elision_note():
    log, cap = _log(1)
    log.table('t', ('i',), [(i,) for i in range(lg.MAX_TABLE_ROWS)])
    assert 'not printed' not in cap.text
    assert len([l for l in cap.text.splitlines() if l.strip().isdigit()]) == lg.MAX_TABLE_ROWS


# --------------------------------------------- dataclass_parameters -------

@dataclasses.dataclass(frozen=True)
class _Knobs:
    convergence: float = 1.e-9
    max_iterations: int = 120
    bulk: tuple = ()


def test_dataclass_parameters_enumerates_every_declared_field():
    got = lg.dataclass_parameters(_Knobs(), prefix='c.')
    assert [name for name, _ in got] == ['c.convergence', 'c.max_iterations', 'c.bulk']
    assert dict(got)['c.max_iterations'] == 120


def test_dataclass_parameters_reports_bulk_fields_by_size():
    got = dict(lg.dataclass_parameters(_Knobs(bulk=tuple(range(lg.MAX_ROW_CELLS + 5)))))
    assert got['bulk'] == f'<{lg.MAX_ROW_CELLS + 5} entries>'


def test_dataclass_parameters_skip_is_explicit():
    got = lg.dataclass_parameters(_Knobs(), skip=('bulk',))
    assert [name for name, _ in got] == ['convergence', 'max_iterations']


# ------------------------------------------------------------ QCVariables ----

def test_set_is_a_noop_without_a_wavefunction():
    lg._set(None, 'ATOMIC ANYTHING', 1.)


def test_matrix_wrap_defeats_the_name_driven_reshaper():
    """The reason arrays are wrapped: p4util reshapes a bare ndarray by NAME.

    A 3x3 table stored under a name p4util reads as a multipole is forced to
    ``(1, 3)`` and simply fails; wrapped as a ``core.Matrix`` it is stored
    verbatim, which is what every array published here relies on.
    """
    mol = psi4.geometry('units bohr\nsymmetry c1\nno_com\nno_reorient\n'
                        'O 0 0 0\nH -1.45365196 0 -1.12168732\nH 1.45365196 0 -1.12168732\n')
    wfn = core.Wavefunction.build(mol, 'sto-3g')
    tensor = np.arange(9, dtype=float).reshape(3, 3)
    with pytest.raises(ValueError):
        wfn.set_variable('SOMETHING DIPOLE', tensor)
    lg._set(wfn, 'SOMETHING DIPOLE', lg._matrix(tensor))
    got = np.asarray(wfn.array_variable('SOMETHING DIPOLE'))
    assert got.shape == (3, 3)
    assert np.array_equal(got, tensor)
