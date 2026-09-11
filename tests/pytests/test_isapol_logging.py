"""Contract tests for the native atomic-property reporting surface.

``isapol_logging`` narrates the pipeline and publishes QCVariables; it computes
no science.  The properties pinned here are exactly the ones that make that
claim checkable:

* the default log is silent and accepts every call, so an expert caller that
  passes no log behaves as it did before the module existed;
* a stage banner enumerates the dataclass, not a hand-written list, so a knob
  added later cannot silently vanish from "every tweakable parameter";
* large intermediates are unprintable -- a wide row is refused outright and a
  long body is elided in the middle with its omitted count;
* array QCVariables keep the shape the stage gave them, because they are wrapped
  as ``core.Matrix`` instead of being handed to p4util's name-driven reshaper;
* a record that marks a number incomparable (an incomplete dispersion order)
  reaches the variable map only under an ``INCOMPLETE``-marked name.
"""

import dataclasses

import numpy as np
import pytest

import psi4
from psi4 import core
from psi4.driver.procrouting import isapol_logging as lg
from psi4.driver.procrouting import isapol_lw as lw

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
    ((1, 2.5, None), '(1, 2.500000, none)'),
    ('ISA_A', 'ISA_A'),
])
def test_fmt_renders_each_cell_kind(value, want):
    assert lg._fmt(value) == want


def test_fmt_nan_is_reported_not_hidden():
    assert lg._fmt(float('nan')) == 'nan'


def test_fmt_never_expands_bulk_data():
    """An array or a long sequence is summarized by size, never printed."""
    assert lg._fmt(np.zeros((512, 3))) == '<array(512, 3) not printed>'
    assert lg._fmt(tuple(range(lg.MAX_ROW_CELLS + 1))) == f'<{lg.MAX_ROW_CELLS + 1} entries not printed>'
    assert lg._fmt(tuple(range(lg.MAX_ROW_CELLS))).startswith('(0, 1,')


# -------------------------------------------------------------- StageLog ----

def test_default_log_is_silent_and_still_accepts_every_call():
    log = lg.silent()
    assert log.verbosity == 0
    assert not log.enabled(1)
    log.banner('x')
    log.line('y')
    log.items((('a', 1),))
    log.table('t', ('h',), [(1,)])
    log.stage('s', (('p', 1),))
    log.stage_end()
    log.failures(())
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


def test_failures_are_printed_verbatim():
    from psi4.driver.procrouting.isapol_native import StageFailure
    log, cap = _log(1)
    log.failures((StageFailure('LW', .5, 'RuntimeError', 'postcondition exceeds tolerance'),))
    assert 'postcondition exceeds tolerance' in cap.text
    assert 'no stage was retried or relaxed' in cap.text


def test_no_failures_prints_nothing():
    log, cap = _log(1)
    log.failures(())
    assert cap.text == ''


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
    assert f'... 17 intermediate rows not printed' in cap.text
    assert 'atomic_property_result(wfn)' in cap.text
    printed = {int(line.strip()) for line in cap.text.splitlines()
               if line.strip().isdigit()}
    assert len(printed) == lg.MAX_TABLE_ROWS
    assert 0 in printed and n - 1 in printed


def test_short_table_is_printed_in_full_with_no_elision_note():
    log, cap = _log(1)
    log.table('t', ('i',), [(i,) for i in range(lg.MAX_TABLE_ROWS)])
    assert 'not printed' not in cap.text
    assert len([l for l in cap.text.splitlines() if l.strip().isdigit()]) == lg.MAX_TABLE_ROWS


@pytest.fixture
def wfn():
    mol = psi4.geometry('units bohr\nsymmetry c1\nno_com\nno_reorient\n'
                        'O 0 0 0\nH -1.45365196 0 -1.12168732\nH 1.45365196 0 -1.12168732\n')
    psi4.set_options({'basis': 'cc-pvdz'})
    return core.Wavefunction.build(mol, 'cc-pvdz')


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


def test_partition_parameters_cover_the_whole_controller_and_grid(wfn):
    """The recipe's own dataclasses, so a new knob appears without a code change."""
    from psi4.driver.procrouting.isapol_oeprop import generated_recipe
    recipe = generated_recipe(wfn, 50, 110, 'cc-pVDZ-JKFIT')
    names = [name for name, _ in lg.partition_parameters(recipe)]
    for field in dataclasses.fields(recipe.controller):
        assert 'controller.' + field.name in names
    for field in dataclasses.fields(recipe.grid):
        assert 'grid.' + field.name in names
    assert 'drho_profile' in names
    assert 'site tail cutoffs [bohr]' in names


# ------------------------------------------------------------ QCVariables ----

def test_set_is_a_noop_without_a_wavefunction():
    lg._set(None, 'ATOMIC ANYTHING', 1.)


def test_partition_narration_requires_the_owned_record(wfn):
    """A look-alike publishes nothing rather than a name without its companions.

    Orchestration doubles legitimately carry only the handful of fields the
    caller touches. Narrating one would publish, say, an iteration count with no
    convergence flag beside it, so the reporter declares the record unnarratable
    and sets no variable at all.
    """
    from types import SimpleNamespace
    log, cap = _log(3)
    imitation = SimpleNamespace(trajectory=SimpleNamespace(state=SimpleNamespace(iteration=1)),
                                drho=SimpleNamespace(fitted_electrons=10.))
    lg.report_partition(log, wfn, imitation)
    assert 'not an owned NativePartitionResult' in cap.text
    assert not wfn.has_variable('ISA ITERATIONS')
    assert not wfn.has_variable('ISA DRHO FITTED ELECTRONS')


def test_matrix_wrap_defeats_the_name_driven_reshaper(wfn):
    """The reason arrays are wrapped: p4util reshapes a bare ndarray by NAME.

    A 3x3 table stored under a name p4util reads as a multipole is forced to
    ``(1, 3)`` and simply fails; wrapped as a ``core.Matrix`` it is stored
    verbatim, which is what every array published here relies on.
    """
    tensor = np.arange(9, dtype=float).reshape(3, 3)
    with pytest.raises(ValueError):
        wfn.set_variable('SOMETHING DIPOLE', tensor)
    lg._set(wfn, 'SOMETHING DIPOLE', lg._matrix(tensor))
    got = np.asarray(wfn.array_variable('SOMETHING DIPOLE'))
    assert got.shape == (3, 3)
    assert np.array_equal(got, tensor)


def _local_properties(n_freq=1):
    """A small hand-built LW record carrying the fields the reporters read.

    Hand-built on purpose: this exercises the formatter, not the localization.
    It is not a certified factory result and no scientific claim is made about
    the numbers in it -- only about how they are rendered and published.
    """
    labels = ('O1', 'H2', 'H3')
    freq = tuple(float(k) for k in range(n_freq))
    means = (3.5, .87, .87)
    scalars = np.array([[[3.5, 6.0, 7.7], [.87, .18, -1.97], [.87, .18, -1.97]]] * n_freq)
    dipoles = np.array([[np.diag([m, m, m]) for m in means]] * n_freq, dtype=float)
    residuals = lw.Residuals(off_site=1.e-15, charge_sum=2.5e-8, reciprocity=1.e-15,
                             molecular_sum=6.4e-14, local_charge=2.5e-8,
                             input_sum_rule=2.5e-8, charge_sum_transport=5.e-16)
    diagnostics = tuple(lw.FrequencyDiagnostics(
        frequency=x, residuals=residuals, transfer_count=456, omitted_component_pairs=(),
        omitted_transfer_count=0, input_max_reciprocity_error=1.7e-15,
        production_postcondition_passed=True) for x in freq)
    tensors = tuple(lw.TensorDiagnostic(k, label, axes, 1.e-15, -2.6)
                    for k in range(n_freq) for label in labels
                    for axes in ('global', 'site_local'))
    metadata = lw.Metadata(input_rank=3, truncation=None, discarded_rank4_entry_count=0,
                           canonical_input_array_sha256='0' * 64, residual_policy='production',
                           residual_tolerance=1.e-6, production_postcondition_passed=True,
                           historical_fixture_sha256=None)
    snap = lw.ArraySnapshot.of
    return lw.LocalProperties(
        labels=labels, origins=snap(np.zeros((3, 3))), frames=snap(np.zeros((3, 3, 3))),
        frequencies=freq, bonds=((1, 0), (2, 0)),
        provenance=lw.Provenance('test', 'a' * 64, 'pytest', 'reporting contract fixture'),
        raw_input=snap(np.zeros((n_freq, 3, 3, 16, 16))),
        raw_global=snap(np.zeros((n_freq, 3, 15, 15))),
        raw_local=snap(np.zeros((n_freq, 3, 15, 15))),
        atomic_scalars=snap(scalars), global_dipoles=snap(dipoles),
        frequency_diagnostics=diagnostics, tensor_diagnostics=tensors,
        warnings=('O1[0] global: indefinite symmetric part (-2.6); no clipping',),
        metadata=metadata)


def test_atomic_polarizability_variables_keep_their_declared_shapes(wfn):
    local = _local_properties()
    lg.report_atomic_polarizabilities(lg.silent(), wfn, local)
    assert wfn.variable('ATOM O1 DIPOLE POLARIZABILITY') == pytest.approx(3.5)
    for rank, want in ((1, 3.5), (2, 6.0), (3, 7.7)):
        assert wfn.variable(f'ATOM O1 ISOTROPIC POLARIZABILITY RANK {rank}') == pytest.approx(want)
        assert np.asarray(wfn.variable(f'ATOMIC ISOTROPIC POLARIZABILITIES RANK {rank}')).shape == (1, 3)
    # The name ends in TENSOR, but the wrap is what guarantees this stays (3, 3).
    assert np.asarray(wfn.variable('ATOM O1 DIPOLE POLARIZABILITY TENSOR')).shape == (3, 3)
    assert np.asarray(wfn.variable('ATOMIC POLARIZABILITY SITE SUM TENSOR')).shape == (3, 3)
    assert np.asarray(wfn.variable('ATOMIC POLARIZABILITY FREQUENCIES')).shape == (1, 1)
    assert wfn.variable('ATOMIC POLARIZABILITY SITE SUM ISOTROPIC') == pytest.approx(3.5 + .87 + .87)


def test_atomic_polarizability_tables_are_formatted_and_bounded():
    log, cap = _log(3)
    lg.report_atomic_polarizabilities(log, None, _local_properties())
    assert 'Static atomic isotropic polarizabilities' in cap.text
    assert 'alpha_1 [bohr^3]' in cap.text and 'alpha_3 [bohr^7]' in cap.text
    assert 'Static atomic dipole polarizability tensors' in cap.text
    assert 'site sum' in cap.text
    assert 'not printed' not in cap.text
    assert 'O1' in cap.text and 'H3' in cap.text


def test_localization_variables_are_published(wfn):
    lg.report_localization(lg.silent(), wfn, _local_properties())
    assert wfn.variable('ATOMIC RESPONSE LOCALIZATION RANK LIMIT') == pytest.approx(3.)
    assert wfn.variable('ATOMIC RESPONSE LW PRODUCTION POSTCONDITION') == pytest.approx(1.)
    assert wfn.variable('ATOMIC RESPONSE LW MAX RESIDUAL') == pytest.approx(2.5e-8)


def test_localization_residual_components_stay_separate():
    """Each named residual gets its own column; none is folded into a maximum."""
    log, cap = _log(2)
    lg.report_localization(log, None, _local_properties())
    assert 'LW localization residuals per node' in cap.text
    for name in dataclasses.fields(lw.Residuals):
        assert name.name in cap.text
    assert 'production postcondition passed' in cap.text
    assert 'WARNING: O1[0] global: indefinite symmetric part' in cap.text


def test_tensor_diagnostics_are_level_three_only():
    log2, cap2 = _log(2)
    lg.report_localization(log2, None, _local_properties())
    assert 'Localized tensor diagnostics' not in cap2.text
    log3, cap3 = _log(3)
    lg.report_localization(log3, None, _local_properties())
    assert 'Localized tensor diagnostics (reported, never repaired)' in cap3.text


def test_frequency_dependent_table_appears_only_with_more_than_one_node():
    log1, cap1 = _log(3)
    lg.report_atomic_polarizabilities(log1, None, _local_properties(n_freq=1))
    assert 'Frequency-dependent atomic isotropic polarizabilities' not in cap1.text
    log3, cap3 = _log(3)
    lg.report_atomic_polarizabilities(log3, None, _local_properties(n_freq=4))
    assert 'Frequency-dependent atomic isotropic polarizabilities' in cap3.text
    log2, cap2 = _log(2)
    lg.report_atomic_polarizabilities(log2, None, _local_properties(n_freq=4))
    assert 'Frequency-dependent atomic isotropic polarizabilities' not in cap2.text


def _dispersion(complete_c12=False):
    """Orders 6 and 12, with 12 deliberately missing a rank pair."""
    def pair(a, b, c6, c12):
        return lw.DispersionPair(a, b, ('O1', 'H2', 'H3')[a], ('O1', 'H2', 'H3')[b], (
            lw.Coefficient(6, c6, ((1, 1),), (), True),
            lw.Coefficient(12, c12, ((1, 3),), () if complete_c12 else ((1, 4), (4, 1)),
                           complete_c12)))
    labels = _local_properties()
    pairs = tuple(pair(a, b, 1. + a + b, 100. + a + b)
                  for a in range(3) for b in range(3))
    return lw.IsotropicDispersion(model_a=labels, model_b=labels, cp_weights=(1.,),
                                  quadrature_provenance=None, pairs=pairs)


def test_incomplete_dispersion_order_is_published_only_as_incomplete(wfn):
    lg.report_dispersion(lg.silent(), wfn, _dispersion(complete_c12=False))
    assert wfn.variable('ATOMIC DISPERSION C6 O1 O1') == pytest.approx(1.)
    assert wfn.variable('ATOMIC DISPERSION C6 TOTAL') == pytest.approx(
        sum(1. + a + b for a in range(3) for b in range(3)))
    assert wfn.has_variable('ATOMIC DISPERSION C12 O1 O1 INCOMPLETE')
    assert not wfn.has_variable('ATOMIC DISPERSION C12 O1 O1')
    assert wfn.has_variable('ATOMIC DISPERSION C12 TOTAL INCOMPLETE')
    assert not wfn.has_variable('ATOMIC DISPERSION C12 TOTAL')
    assert wfn.has_variable('ATOM O1 C6 DISPERSION COEFFICIENT')
    assert wfn.has_variable('ATOM O1 C12 DISPERSION COEFFICIENT INCOMPLETE')
    assert not wfn.has_variable('ATOM O1 C12 DISPERSION COEFFICIENT')


def test_complete_dispersion_order_drops_the_incomplete_mark(wfn):
    lg.report_dispersion(lg.silent(), wfn, _dispersion(complete_c12=True))
    assert wfn.has_variable('ATOMIC DISPERSION C12 TOTAL')
    assert not wfn.has_variable('ATOMIC DISPERSION C12 TOTAL INCOMPLETE')


def test_dispersion_tables_show_atomic_pairwise_and_missing_ranks():
    log, cap = _log(1)
    lg.report_dispersion(log, None, _dispersion())
    assert 'Atomic (same-site) isotropic dispersion coefficients' in cap.text
    assert 'Pairwise isotropic dispersion coefficients (ordered A x B pairs)' in cap.text
    assert 'Rank pairs absent from each order' in cap.text
    assert '((1, 4), (4, 1))' in cap.text
    assert 'INCOMPLETE-marked variable name' in cap.text
    # nine ordered pairs, three of them same-site
    assert cap.text.count('O1  H2') >= 1
    assert 'false' in cap.text


def test_reporting_at_verbosity_zero_still_publishes_every_variable(wfn):
    """The narrative is a print switch; the machine-readable surface is not."""
    lg.report_dispersion(lg.StageLog(0, writer=lambda text: None), wfn, _dispersion())
    lg.report_atomic_polarizabilities(lg.StageLog(0, writer=lambda text: None), wfn,
                                      _local_properties())
    assert wfn.has_variable('ATOMIC DISPERSION C6 TOTAL')
    assert wfn.has_variable('ATOM O1 DIPOLE POLARIZABILITY')


def test_print_option_is_a_declared_global_defaulting_to_one():
    psi4.core.clean_options()
    assert psi4.core.get_global_option('ATOMIC_PROPERTY_PRINT') == 1
