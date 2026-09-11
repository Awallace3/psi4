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
import inspect

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


# ------------------------------------------------- dispersion parameters ----

_SYNTHETIC = lw.Provenance('synthetic', '0' * 64, 'pytest reporting fixture',
                           'Not molecular or native acceptance')


def _factory(n=2, frequencies=(0., 1.)):
    """A real factory LW model, so the producer-owned banners run the real stage.

    Synthetic tensors: this pins the reporting surface, never a C_n value.
    """
    raw = np.zeros((len(frequencies), n, n, 16, 16))
    block = np.diag(np.arange(1., 16.))
    for f, xi in enumerate(frequencies):
        for s in range(n):
            raw[f, s, s, 1:, 1:] = block * (s + 1) / (1. + xi * xi)
    return lw.supplied_nonlocal_properties(
        labels=[f'S{s}' for s in range(n)], origins=[[0., 0., 3. * s] for s in range(n)],
        bonds=[], frequencies=list(frequencies), tensors=raw, input_rank=3,
        provenance=_SYNTHETIC)


def _placement(t=(0., 0., 0.)):
    return lw.Placement(np.eye(3), t)


def _placed(n=3):
    snap = lw.ArraySnapshot.of
    return lw.PlacedGeometry(origins=snap(np.zeros((n, 3))),
                             source_frames=snap(np.zeros((n, 3, 3))),
                             component_frames=snap(np.zeros((n, 3, 3))),
                             core_provenance='pytest placement')


def _anisotropic(*, unrestricted_c7=False):
    """Orders 6 and 7, with 7 complete in the declared model but not unrestricted.

    That combination is the one the two flags exist to distinguish, so the
    fixture carries it rather than making both flags agree.
    """
    model = _local_properties(n_freq=2)

    def coefficients(scale):
        return (lw.AnisotropicCoefficient(6, 2. * scale, -2. * scale / 64.,
                                          ((1, 1, 1, 1),), (), True, True),
                lw.AnisotropicCoefficient(7, .5 * scale, -.5 * scale / 128.,
                                          ((1, 1, 1, 2), (1, 2, 1, 1)),
                                          () if unrestricted_c7 else ((1, 1, 1, 5),),
                                          True, unrestricted_c7))
    pairs = []
    for a, b in ((0, 1), (1, 0)):
        coeffs = coefficients(1. + a + b)
        pairs.append(lw.AnisotropicPair(a, b, model.labels[a], model.labels[b], 2. + a + b,
                                        (0., 0., 2. + a + b), (0., 0., 1.), coeffs,
                                        sum(c.energy for c in coeffs)))
    return lw.AnisotropicDispersion(
        model_a=model, model_b=model, placement_a=_placement(),
        placement_b=_placement((0., 0., 2.)), placed_a=_placed(), placed_b=_placed(),
        frequencies=model.frequencies, cp_weights=(0., .25), quadrature_provenance=None,
        max_order=7, pairs=tuple(pairs),
        truncated_energy=sum(p.truncated_energy for p in pairs))


def test_dispersion_record_fields_name_only_declared_fields():
    """The skip list is a partition of real fields, not a bag of stale names."""
    declared = {f.name for f in dataclasses.fields(lw.IsotropicDispersion)}
    declared |= {f.name for f in dataclasses.fields(lw.AnisotropicDispersion)}
    assert set(lg.DISPERSION_RECORD_FIELDS) <= declared


@pytest.mark.parametrize('record', ['isotropic', 'anisotropic'])
def test_every_dispersion_record_field_is_skipped_or_narrated(record):
    obj = _dispersion() if record == 'isotropic' else _anisotropic()
    narrated = {name for name, _ in
                lg.dataclass_parameters(obj, skip=lg.DISPERSION_RECORD_FIELDS)}
    for field in dataclasses.fields(obj):
        assert field.name in lg.DISPERSION_RECORD_FIELDS or field.name in narrated


#: Each keyword-only knob of a producer, and a narrated name that carries it.
#: Comparing the key set with the live signature is the point: a knob added to
#: either producer later fails this test until its banner names it.
_ISOTROPIC_KNOBS = {'max_order': 'max_order',
                    'cp_weights': 'quadrature nodes',
                    'quadrature_provenance': 'quadrature.description',
                    'site_ranks_a': 'A.site_ranks',
                    'site_ranks_b': 'B.site_ranks'}
_ANISOTROPIC_KNOBS = {'max_order': 'max_order',
                      'cp_weights': 'quadrature nodes',
                      'quadrature_provenance': 'quadrature.description',
                      'placement_a': 'A.placement.translation [bohr]',
                      'placement_b': 'B.placement.translation [bohr]'}


def _keyword_knobs(producer):
    return {name for name, p in inspect.signature(producer).parameters.items()
            if p.kind is p.KEYWORD_ONLY and name not in ('log', 'wfn')}


def test_dispersion_parameters_narrate_every_producer_knob():
    assert _keyword_knobs(lw.isotropic_dispersion) == set(_ISOTROPIC_KNOBS)
    model = _local_properties()
    names = {name for name, _ in lg.dispersion_parameters(
        model_a=model, model_b=model, max_order=12, cp_weights=(1.,),
        quadrature_provenance=model.provenance)}
    for narrated in _ISOTROPIC_KNOBS.values():
        assert narrated in names


def test_anisotropic_parameters_narrate_every_producer_knob():
    assert _keyword_knobs(lw.anisotropic_dispersion) == set(_ANISOTROPIC_KNOBS)
    model = _local_properties()
    names = {name for name, _ in lg.anisotropic_dispersion_parameters(
        model_a=model, model_b=model, placement_a=_placement(),
        placement_b=_placement((0., 0., 2.)), max_order=12, cp_weights=(1.,),
        quadrature_provenance=model.provenance)}
    for narrated in _ANISOTROPIC_KNOBS.values():
        assert narrated in names
    assert 'not declarable: ranks 1..3 of raw_global on every site' in dict(
        lg.anisotropic_dispersion_parameters(
            model_a=model, model_b=model, placement_a=_placement(),
            placement_b=_placement(), max_order=12, cp_weights=(1.,),
            quadrature_provenance=None))['site ranks']


def test_dispersion_parameters_report_both_model_identities():
    """A C_n is only as declared as the two models it contracts."""
    model = _local_properties()
    got = dict(lg.dispersion_parameters(model_a=model, model_b=model, max_order=12,
                                        cp_weights=(1.,), quadrature_provenance=None))
    for side in ('A.', 'B.'):
        assert got[side + 'localization_rank_limit'] == 3
        assert got[side + 'residual_policy'] == 'production'
        assert got[side + 'canonical_input_array_sha256'] == '0' * 64
    assert got['model B'].startswith('this same model')
    assert got['CP weight sum'] == pytest.approx(1.)
    assert got['quadrature.provenance'] == 'none declared with this record'


def test_declared_site_ranks_resolve_from_the_model_when_undeclared():
    """``None`` means every rank the model was localized at, not a fixed (1,2,3)."""
    model = _local_properties()
    default = dict(lg.declared_site_ranks(model, None, prefix='A.'))
    assert 'read off' in default['A.site_ranks']
    assert default['A.resolved site ranks'] == ((1, 2, 3),) * 3
    declared = dict(lg.declared_site_ranks(model, ((1,), (1, 2), (1, 2, 3)), prefix='A.'))
    assert declared['A.site_ranks'] == 'declared explicitly, per site'
    assert declared['A.resolved site ranks'] == ((1,), (1, 2), (1, 2, 3))


def test_absent_quadrature_provenance_is_reported_not_omitted():
    log, cap = _log(2)
    lg.report_quadrature(log, (0., 1.), (0., .25))
    assert 'no quadrature provenance is declared on this record' in cap.text
    assert lg.provenance_parameters(None, prefix='q.') == (
        ('q.provenance', 'none declared with this record'),)


def test_report_quadrature_needs_both_sequences():
    log, cap = _log(3)
    lg.report_quadrature(log, None, (0., .25))
    lg.report_quadrature(log, (0., 1.), None)
    assert cap.text == ''


def test_comparability_table_precedes_the_rank_pair_inventory():
    """The level-1 warning comes first; the inventory below it is the evidence."""
    log, cap = _log(2)
    lg.report_dispersion(log, None, _dispersion())
    assert cap.text.index('Rank pairs absent from each order') < cap.text.index(
        'Rank pairs entering each order')


def test_rank_pair_inventory_is_level_two_only():
    log1, cap1 = _log(1)
    lg.report_dispersion(log1, None, _dispersion())
    assert 'Rank pairs entering each order' not in cap1.text
    log2, cap2 = _log(2)
    lg.report_dispersion(log2, None, _dispersion())
    assert 'Rank pairs entering each order (union over the ordered site pairs)' in cap2.text
    assert 'not because its contribution was found small' in cap2.text


def test_dispersion_narration_reports_the_quadrature_it_contracted():
    log, cap = _log(2)
    lg.report_dispersion(log, None, _dispersion())
    assert 'Casimir-Polder quadrature' in cap.text
    assert 'quadrature nodes' in cap.text


def test_dispersion_publishes_the_quadrature_alongside_the_coefficients(wfn):
    lg.report_dispersion(lg.silent(), wfn, _dispersion())
    assert wfn.variable('ATOMIC DISPERSION SITE PAIRS') == pytest.approx(9.)
    assert wfn.variable('ATOMIC DISPERSION MAX ORDER') == pytest.approx(12.)
    assert wfn.variable('ATOMIC DISPERSION QUADRATURE NODES') == pytest.approx(1.)
    assert np.asarray(wfn.variable('ATOMIC DISPERSION CP WEIGHTS')).shape == (1, 1)
    assert np.asarray(wfn.variable('ATOMIC DISPERSION QUADRATURE FREQUENCIES')).shape == (1, 1)


# --------------------------------------- oriented (anisotropic) dispersion ----

def test_placement_parameters_identify_the_rotation_by_hash():
    got = dict(lg.placement_parameters(_placement((0., 0., 2.)), prefix='B.placement.'))
    assert got['B.placement.translation [bohr]'] == (0., 0., 2.)
    assert got['B.placement.rotation'] == 'explicit identity'
    assert got['B.placement.rotation trace'] == pytest.approx(3.)
    assert len(got['B.placement.rotation_sha256']) == 64
    assert got['B.placement.translation_sha256'] != got['B.placement.rotation_sha256']


def test_anisotropic_tables_print_both_completeness_flags():
    log, cap = _log(1)
    lg.report_anisotropic_dispersion(log, None, _anisotropic())
    assert 'Orientation-resolved dispersion coefficients (ordered A x B pairs)' in cap.text
    assert 'declared complete' in cap.text and 'unrestricted complete' in cap.text
    assert 'Placed site-pair geometry' in cap.text
    assert 'truncated interaction energy [Eh]' in cap.text
    assert 'no damping, no retardation' in cap.text
    # An oriented scalar must never be readable as an isotropic C_n.
    assert 'orientation_resolved_scalars_not_recoupled_components' in cap.text


def test_anisotropic_energy_table_and_quadruple_counts_are_level_two():
    log1, cap1 = _log(1)
    lg.report_anisotropic_dispersion(log1, None, _anisotropic())
    assert 'Orientation-resolved -C_n/R^n contributions' not in cap1.text
    assert 'Rank quadruples entering each order' not in cap1.text
    log2, cap2 = _log(2)
    lg.report_anisotropic_dispersion(log2, None, _anisotropic())
    assert 'Orientation-resolved -C_n/R^n contributions [Eh]' in cap2.text
    assert 'Rank quadruples entering each order' in cap2.text
    assert 'theoretical ranks 5..7 no rank-3 model can carry' in cap2.text
    # Counts, never the quadruple lists themselves.
    assert '(1, 1, 1, 2)' not in cap2.text


def test_incomplete_anisotropic_order_is_published_only_as_incomplete(wfn):
    """The name mark follows ``unrestricted_complete``, as the isotropic stage does."""
    lg.report_anisotropic_dispersion(lg.silent(), wfn, _anisotropic())
    assert wfn.has_variable('ATOMIC ANISOTROPIC DISPERSION C6 O1 H2')
    assert wfn.has_variable('ATOMIC ANISOTROPIC DISPERSION C6 TOTAL')
    assert wfn.has_variable('ATOMIC ANISOTROPIC DISPERSION C7 O1 H2 INCOMPLETE')
    assert not wfn.has_variable('ATOMIC ANISOTROPIC DISPERSION C7 O1 H2')
    assert wfn.has_variable('ATOMIC ANISOTROPIC DISPERSION C7 ENERGY O1 H2 INCOMPLETE')
    assert wfn.has_variable('ATOMIC ANISOTROPIC DISPERSION C7 TOTAL INCOMPLETE')
    assert wfn.has_variable('ATOMIC ANISOTROPIC DISPERSION C7 TOTAL ENERGY INCOMPLETE')
    assert not wfn.has_variable('ATOMIC ANISOTROPIC DISPERSION C7 TOTAL')


def test_complete_anisotropic_order_drops_the_incomplete_mark(wfn):
    lg.report_anisotropic_dispersion(lg.silent(), wfn, _anisotropic(unrestricted_c7=True))
    assert wfn.has_variable('ATOMIC ANISOTROPIC DISPERSION C7 TOTAL')
    assert not wfn.has_variable('ATOMIC ANISOTROPIC DISPERSION C7 TOTAL INCOMPLETE')


def test_anisotropic_variables_keep_their_declared_shapes(wfn):
    record = _anisotropic()
    lg.report_anisotropic_dispersion(lg.silent(), wfn, record)
    assert wfn.variable('ATOMIC ANISOTROPIC DISPERSION TRUNCATED ENERGY') == pytest.approx(
        record.truncated_energy)
    assert wfn.variable('ATOMIC ANISOTROPIC DISPERSION MAX ORDER') == pytest.approx(7.)
    assert wfn.variable('ATOMIC ANISOTROPIC DISPERSION SITE PAIRS') == pytest.approx(2.)
    assert wfn.variable('ATOMIC ANISOTROPIC DISPERSION QUADRATURE NODES') == pytest.approx(2.)
    for name in ('QUADRATURE FREQUENCIES', 'CP WEIGHTS', 'PAIR DISTANCES'):
        assert np.asarray(wfn.variable(
            'ATOMIC ANISOTROPIC DISPERSION ' + name)).shape == (1, 2)
    assert wfn.variable('ATOMIC ANISOTROPIC DISPERSION PAIR ENERGY O1 H2') == pytest.approx(
        record.pairs[0].truncated_energy)


def test_anisotropic_narration_needs_no_wavefunction():
    log, cap = _log(3)
    lg.report_anisotropic_dispersion(log, None, _anisotropic())
    assert 'Sum over all ordered site pairs' in cap.text
    assert 'not printed' not in cap.text


def test_empty_anisotropic_record_reports_nothing_it_does_not_have():
    record = dataclasses.replace(_anisotropic(), pairs=(), truncated_energy=0.)
    log, cap = _log(3)
    lg.report_anisotropic_dispersion(log, None, record)
    assert 'Orientation-resolved dispersion coefficients' not in cap.text
    assert 'Rank quadruples entering each order' not in cap.text


# ------------------------------------------------ producer-owned banners ----

def test_isotropic_producer_owns_exactly_one_stage_banner():
    """The producer, not the caller, narrates: only it knows the resolved ranks."""
    log, cap = _log(2)
    model = _factory()
    record = lw.isotropic_dispersion(model, model, cp_weights=[0., .25],
                                     quadrature_provenance=_SYNTHETIC, max_order=6, log=log)
    assert cap.text.count('==> Stage: isotropic dispersion coefficients (Casimir-Polder)') == 1
    assert 'A.resolved site ranks' in cap.text
    assert 'Pairwise isotropic dispersion coefficients' in cap.text
    assert [c.order for c in record.pairs[0].coefficients] == [6]


def test_anisotropic_producer_owns_exactly_one_stage_banner():
    log, cap = _log(2)
    model = _factory()
    lw.anisotropic_dispersion(model, model, placement_a=_placement(),
                              placement_b=_placement((0., 0., 8.)), cp_weights=[0., .25],
                              quadrature_provenance=_SYNTHETIC, max_order=7, log=log)
    assert cap.text.count('==> Stage: oriented (anisotropic) dispersion coefficients') == 1
    assert 'not declarable: ranks 1..3 of raw_global on every site' in cap.text
    assert 'Orientation-resolved dispersion coefficients' in cap.text
    assert 'A.placement.rotation_sha256' in cap.text


def test_producers_are_silent_and_publish_nothing_by_default(wfn):
    model = _factory()
    lw.isotropic_dispersion(model, model, cp_weights=[0., .25],
                            quadrature_provenance=_SYNTHETIC, max_order=6)
    assert not wfn.has_variable('ATOMIC DISPERSION C6 TOTAL')
    lw.isotropic_dispersion(model, model, cp_weights=[0., .25],
                            quadrature_provenance=_SYNTHETIC, max_order=6, wfn=wfn)
    assert wfn.has_variable('ATOMIC DISPERSION C6 TOTAL')
    lw.anisotropic_dispersion(model, model, placement_a=_placement(),
                              placement_b=_placement((0., 0., 8.)), cp_weights=[0., .25],
                              quadrature_provenance=_SYNTHETIC, max_order=6, wfn=wfn)
    assert wfn.has_variable('ATOMIC ANISOTROPIC DISPERSION TRUNCATED ENERGY')


# ----------------------------------------------------------------- PFIT ----

def _refinement(*, masked=False):
    """A small solved refinement: two site types, rank 1, eight points.

    Built through ``isapol_refine`` rather than faked, because the reporters
    read the solver's own diagnostics, batch records and parameter labels; a
    double would pin the double's field names instead of the binding's.
    """
    from psi4.driver.procrouting import isapol_refine as R
    identity = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    sites = (R.RefinementSite('O', 'O', (0., 0., -.13), identity, 1),
             R.RefinementSite('H1', 'H', (-1.45, 0., 1.02), identity, 1),
             R.RefinementSite('H2', 'H', (1.45, 0., 1.02), identity, 1))

    def block(n, offset):
        b = np.array([[(((3*i + 5*k + offset) % 19) - 9)/8. for k in range(n)]
                      for i in range(n)])
        return b @ b.T/8. + np.eye(n)*(n/4.)

    anchor_o, anchor_h = block(4, 7), block(4, 11)
    if masked:
        anchor_o = np.diag(np.diag(anchor_o))
    model = R.refinement_model(sites, [anchor_o, anchor_h, anchor_h.copy()],
                               provenance='test_isapol_logging refinement')
    rng = np.random.default_rng(20260911)
    points = np.asarray(4. + 2.*rng.random((8, 3)))
    fields = R.channel_fields(points, model)
    source = [anchor_o + block(4, 3)/8., anchor_h + block(4, 5)/8.]
    response = R.point_to_point_response(fields, model, [source[0], source[1], source[1]])
    return R.refine(model, points, R.pack_lower_triangle(response), fields=fields,
                    target_origin=core.IsaPfitTargetOrigin.SyntheticAnalyticTest,
                    source_id='test_isapol_logging',
                    generation_record='forward map of a perturbed local model')


def test_pfit_option_fields_match_the_binding():
    """Options are a pybind object, so the list is written out; it must be whole."""
    declared = {name for name, value in vars(core.IsaPfitOptions).items()
                if isinstance(value, property)}
    assert set(lg.PFIT_OPTION_FIELDS) == declared


def test_pfit_diagnostic_keys_cover_every_bound_diagnostic():
    """Narrated metrics plus the two bulk records account for all of them."""
    declared = {name for name, value in vars(core.IsaPfitDiagnostics).items()
                if isinstance(value, property)}
    assert set(lg.PFIT_DIAGNOSTIC_KEYS) | set(lg.PFIT_DIAGNOSTIC_BULK_KEYS) == declared
    assert not set(lg.PFIT_DIAGNOSTIC_KEYS) & set(lg.PFIT_DIAGNOSTIC_BULK_KEYS)


def test_refine_parameters_cover_every_model_knob_and_solver_control():
    from psi4.driver.procrouting import isapol_refine as R
    refinement = _refinement()
    options = core.IsaPfitOptions()
    names = dict(lg.refine_parameters(model=refinement.model, points=((0., 0., 0.),),
                                      fields=None, damping=0., options=options,
                                      source_id='id'))
    structure = set(lg.REFINEMENT_STRUCTURE_FIELDS)
    for field in dataclasses.fields(R.RefinementModel):
        if field.name in structure:
            # narrated as a count or a table, never as the variable list itself
            assert names.get(field.name) != getattr(refinement.model, field.name)
            continue
        assert field.name in names, field.name
        assert names[field.name] == getattr(refinement.model, field.name)
    assert names['sites'] == len(refinement.model.sites)
    assert names['site types'] == refinement.model.site_types
    for name in lg.PFIT_OPTION_FIELDS:
        assert 'options.' + name in names
    assert names['fit points'] == 1 and names['packed target rows'] == 1
    assert names['T-function fields'] == 'generated from the fit points'


def test_refine_parameters_report_the_lattice_by_count_not_by_value():
    refinement = _refinement()
    text = '\n'.join('%s %s' % (k, lg._fmt(v)) for k, v in lg.refine_parameters(
        model=refinement.model, points=np.zeros((8, 3)), fields=object(), damping=1.5,
        options=core.IsaPfitOptions(), source_id='id'))
    assert 'fit points 8' in text and 'packed target rows 36' in text
    assert 'caller-supplied' in text
    assert 'not printed' not in text


def test_refinement_model_tables_show_sites_and_copy_equivalence():
    log, cap = _log(1)
    refinement = _refinement()
    lg.report_refinement_model(log, refinement.model)
    assert 'Refinement sites' in cap.text
    assert 'COPY equivalence' in cap.text
    # the two hydrogens share one variable set, read off the first of them
    assert '(H1, H2)' in cap.text
    assert 'Declared site frames' not in cap.text


def test_declared_site_frames_are_level_three_only():
    log, cap = _log(3)
    lg.report_refinement_model(log, _refinement().model)
    assert 'Declared site frames' in cap.text


def test_refinement_batch_table_is_level_two_and_names_its_batch():
    refinement = _refinement()
    log, cap = _log(1)
    lg.report_refinement_batches(log, refinement.result)
    assert 'Fit residuals per data batch' not in cap.text
    log, cap = _log(2)
    lg.report_refinement_batches(log, refinement.result)
    assert 'Fit residuals per data batch' in cap.text
    assert 'refinement' in cap.text


def test_refinement_narration_shows_anchors_shifts_and_refined_isotropics():
    log, cap = _log(1)
    refinement = _refinement()
    lg.report_refinement(log, None, refinement, frequency=0.)
    assert 'Refined variables against their anchors' in cap.text
    assert 'penalty strength' in cap.text
    assert 'Refined atomic isotropic polarizabilities' in cap.text
    assert 'rank 0 is a refinement variable' in cap.text
    assert 'free parameters' in cap.text
    # the bulk of the stage never reaches the narrative
    assert 'not printed' not in cap.text.replace('higher virtual orbitals not printed', '')


def test_refinement_isotropics_come_from_the_refinement_module(wfn):
    """The Racah reduction has one home; reporting reads it, never restates it."""
    from psi4.driver.procrouting import isapol_refine as R
    refinement = _refinement()
    ranks, scalars = R.isotropic_scalars(refinement)
    lg.report_refinement(lg.silent(), wfn, refinement)
    for site, site_ranks, site_scalars in zip(refinement.model.sites, ranks, scalars):
        for rank, value in zip(site_ranks, site_scalars):
            key = 'ATOM %s REFINED ISOTROPIC POLARIZABILITY RANK %d' % (site.label, rank)
            assert wfn.variable(key) == pytest.approx(float(value), rel=0, abs=0)


def test_refinement_variables_are_published_with_declared_shapes(wfn):
    refinement = _refinement()
    lg.report_refinement(lg.silent(), wfn, refinement, frequency=.5)
    n = refinement.model.parameter_count
    assert np.asarray(wfn.variable('ATOMIC REFINEMENT PARAMETERS XI 0.50000000')).shape == (1, n)
    assert np.asarray(wfn.variable('ATOMIC REFINEMENT ANCHORS XI 0.50000000')).shape == (1, n)
    assert wfn.variable('ATOMIC REFINEMENT STATUS XI 0.50000000') == 1.
    assert wfn.variable('ATOMIC REFINEMENT NUMERICAL RANK XI 0.50000000') == float(n)
    assert wfn.has_variable('ATOMIC REFINEMENT DATA MAX RESIDUAL XI 0.50000000')
    assert wfn.has_variable('ATOMIC REFINEMENT COPY ANCHOR DISCREPANCY XI 0.50000000')
    assert not wfn.has_variable('ATOMIC REFINEMENT PARAMETERS')


def test_unsolved_refinement_marks_the_properties_but_not_the_diagnostics(wfn):
    """A rank-deficient fit still returns numbers; its properties say so."""
    refinement = _refinement()
    unsolved = dataclasses.replace(refinement, status=core.IsaPfitStatus.RankDeficient)
    lg.report_refinement(lg.silent(), wfn, unsolved)
    assert wfn.variable('ATOMIC REFINEMENT STATUS') == 0.
    assert wfn.has_variable('ATOMIC REFINEMENT DATA RMS')
    assert wfn.has_variable('ATOMIC REFINEMENT PARAMETERS RANKDEFICIENT')
    assert not wfn.has_variable('ATOMIC REFINEMENT PARAMETERS')
    assert wfn.has_variable('ATOM O REFINED ISOTROPIC POLARIZABILITY RANK 1 RANKDEFICIENT')
    assert not wfn.has_variable('ATOM O REFINED ISOTROPIC POLARIZABILITY RANK 1')


def test_refine_publishes_only_when_handed_a_wavefunction(wfn):
    """``refine`` takes no wavefunction by default, so expert callers are unchanged."""
    import inspect
    from psi4.driver.procrouting import isapol_refine as R
    signature = inspect.signature(R.refine)
    assert signature.parameters['wfn'].default is None
    _refinement()  # solved with no wavefunction at all
    assert not wfn.variables()


# ------------------------------------------------- declared AC iterations ----

def _declaration():
    from psi4.driver.procrouting import isapol_native_ac as ac
    return ac.AcDeclaration(ionization_potential=.46380)


def _convergence(**kwargs):
    from psi4.driver.procrouting import isapol_native_ac as ac
    fields = dict(iterations=17, delta_energy=1.e-11, orbital_gradient=2.e-9,
                  energy_threshold=1.e-10, gradient_threshold=1.e-8, shift=.164933,
                  shift_clamped=0, homo=-.298867, lumo=.075810, energy=-76.338427,
                  reference_energy=-76.338456, grid_points=66202)
    fields.update(kwargs)
    return ac.AcConvergence(**fields)


def test_ac_parameters_cover_every_declaration_field_and_its_limits():
    from psi4.driver.procrouting import isapol_native_ac as ac
    names = dict(lg.ac_parameters(_declaration(), maxiter=200, energy_threshold=1.e-10,
                                  gradient_threshold=1.e-8, diis_subspace=10,
                                  shift_damping=.5,
                                  max_energy_threshold=ac.MAX_ENERGY_THRESHOLD,
                                  max_gradient_threshold=ac.MAX_GRADIENT_THRESHOLD))
    for field in dataclasses.fields(ac.AcDeclaration):
        assert field.name in names
    assert names['label'] == _declaration().label()
    # the declared thresholds are shown against the loosest ones admission takes
    assert names['loosest admitted energy_threshold'] == ac.MAX_ENERGY_THRESHOLD
    assert names['loosest admitted gradient_threshold'] == ac.MAX_GRADIENT_THRESHOLD
    assert names['energy_threshold'] <= names['loosest admitted energy_threshold']


def test_ac_splice_reports_the_grid_it_was_given():
    log, cap = _log(1)
    lg.report_ac_splice(log, exact_exchange=.25, fermi_amaldi_scale=.075, electrons=10,
                        occupied=5, basis_functions=24, origin_bohr=(0., 0., .1),
                        bragg_radii=(1.134, .661, .661), grid_points=66202,
                        grid_blocks=553, active_blocks=288, active_points=15598)
    assert 'the SCF exchange-correlation grid, unchanged' in cap.text
    assert 'points with f > 0' in cap.text
    assert '0.235612' in cap.text  # 15598/66202, the corrected fraction


def test_ac_iteration_rows_are_level_two_and_keep_no_history():
    log, cap = _log(1)
    rows = lg.AcIterationLog(log)
    rows.row(iteration=0, energy=-76.3, delta=None, gradient=1.e-3, shift=.16,
             homo=-.3, lumo=.07, clamped=0)
    assert cap.text == ''
    log, cap = _log(2)
    rows = lg.AcIterationLog(log)
    for i in range(3):
        rows.row(iteration=i, energy=-76.3, delta=None if not i else 1.e-5,
                 gradient=1.e-3, shift=.16, homo=-.3, lumo=.07, clamped=0)
    assert 'Tozer-Handy shift; DIIS' in cap.text
    assert cap.text.count('max|[F,D]|') == 1  # one header for the whole loop
    assert len([line for line in cap.text.splitlines() if line.strip()]) == 5
    assert not hasattr(rows, 'history')


def test_ac_spectrum_prints_the_occupied_set_and_publishes_the_whole_vector(wfn):
    log, cap = _log(1)
    energies = np.arange(-19., 5., 1.)
    lg.report_ac_spectrum(log, wfn, energies, 5)
    assert 'Corrected orbital energies' in cap.text
    assert 'gap' in cap.text
    assert '%d higher virtual orbitals not printed' % (energies.size - 15) in cap.text
    published = np.asarray(wfn.variable('ATOMIC DECLARED AC ORBITAL ENERGIES'))
    assert published.shape == (1, energies.size)
    assert np.array_equal(published.ravel(), energies)


def test_ac_spectrum_refuses_a_spectrum_without_a_virtual():
    with pytest.raises(ValueError, match='occupied set and one virtual'):
        lg.report_ac_spectrum(lg.silent(), None, np.arange(5.), 5)


def test_ac_convergence_publishes_the_iteration_metrics(wfn):
    lg.report_ac_convergence(lg.silent(), wfn, _convergence(), _declaration(),
                             converged=True)
    assert wfn.variable('ATOMIC DECLARED AC CONVERGED') == 1.
    assert wfn.variable('ATOMIC DECLARED AC ITERATIONS') == 17.
    assert wfn.variable('ATOMIC DECLARED AC GAP') == pytest.approx(.075810 + .298867)
    assert wfn.variable('ATOMIC DECLARED AC DECLARED IP') == .46380
    for key in ('SHIFT', 'SHIFT CLAMP HITS', 'HOMO', 'LUMO', 'DELTA ENERGY',
                'ORBITAL GRADIENT', 'ENERGY NOT VARIATIONAL', 'REFERENCE SCF ENERGY',
                'GRID POINTS'):
        assert wfn.has_variable('ATOMIC DECLARED AC ' + key)


def test_unconverged_ac_is_narrated_but_never_published_unmarked(wfn):
    """The producer reports before it refuses; the refusal is not reformatted."""
    log, cap = _log(1)
    lg.report_ac_convergence(log, wfn, _convergence(orbital_gradient=1.e-3),
                             _declaration(), converged=False)
    assert 'converged' in cap.text and 'false' in cap.text
    assert wfn.variable('ATOMIC DECLARED AC CONVERGED') == 0.
    assert wfn.has_variable('ATOMIC DECLARED AC SHIFT UNCONVERGED')
    assert not wfn.has_variable('ATOMIC DECLARED AC SHIFT')


def test_ac_convergence_narration_shows_the_shift_fixed_point():
    log, cap = _log(1)
    lg.report_ac_convergence(log, None, _convergence(), _declaration(), converged=True)
    assert 'I + eps_HOMO' in cap.text
    assert 'energy (not variational)' in cap.text
    assert 'plain SCF reference energy' in cap.text
    assert 'energy above plain SCF' in cap.text


def test_ac_application_parameters_name_the_mutation_and_its_cost():
    from psi4.driver.procrouting import isapol_native_ac as ac
    record = ac.DeclaredAcOrbitals(_declaration(), np.zeros((24, 24)), np.zeros(24),
                                   np.zeros((24, 24)), np.zeros((24, 24)),
                                   _convergence(), 5)
    names = dict(lg.ac_application_parameters(record))
    assert names['basis functions'] == 24 and names['occupied orbitals'] == 5
    assert 'epsilon_a' in names['replaced state']
    assert 'not a variational minimum' in names['energy replaced by']
    assert 'invalidated' in names['SCF seal']
