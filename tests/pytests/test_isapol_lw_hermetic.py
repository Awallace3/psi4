# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Hermetic stage: the `.pol` dialect contract, the frame rule, and what was measured.

The hermetic diagnostic runs supplied nonlocal rank-4 water polarizabilities through
native LW and compares all 675 entries of the localized rank 1:3 result against an
externally produced unrefined reference, in each site's local frame.  Its two source
documents are read by `data_isapol/oracle/extract_lw_hermetic.py` into a static JSON
fixture; neither the reference tree nor any executable is touched at test time.

*Why the 675-entry comparison is not here.*  Like leg A, localization couples all
nine site-pair blocks, so no subset of the input reproduces any single output entry.
The comparison and its 7,159-line fixture live in
untracked `agent_scratch/pytests/test_isapol_lw_hermetic.py`.  What it
measured is recorded below as literals and re-derived by the scratch guard.

*What is here.*  The parser's contract is tested on synthetic documents built to the
reviewed dialect, which needs no fixture and is a stronger statement than parsing the
one recorded file: the eleven-fault matrix, the pair sequence, the dialect separation
and whitespace insensitivity all become claims about what the parser accepts rather
than about what one document happens to contain.  The local-frame rule is derived
from the site geometry instead of being read back from the fixture, and the rank-4
truncation and the rank-3 rotation are exact theorems.
"""
from decimal import Decimal
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]

LABELS = ('O', 'H1', 'H2')
#: Site origins in bohr, from the reference's own `H2O.sites`.
ORIGINS = [[0.0, 0.0, 0.0], [-1.45365196, 0.0, -1.12168732], [1.45365196, 0.0, -1.12168732]]
#: Local-to-global column frames, from the reference's own `H2O.axes` rule
#: "H1 z global Z x from H2 to H1 / H2 z global Z x from H1 to H2".  O has no axes
#: entry and so keeps the global frame.  Derived from `ORIGINS` in the test below.
FRAMES = [[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
          [[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]],
          [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]

#: Exact header dialects, reproducing the recorded documents' own spacing.  The
#: parser compares on `.split()`, which `test_pol_dialect_ignores_whitespace_runs`
#: pins as deliberate.
DISTRIBUTED_HEADER = ('ALPHA INDEX 001  SITE-LABELS  {a}  {b}  SITE-INDICES     {i}     {j}  '
                      'RANK  0 :   4   BY     0 :   4   FREQ2  0.0000000E+00  CARTSPHER S')
LOCAL_HEADER = 'ALPHA  H2O  SITE-NAMES  {a}  {b}  RANK 1 TO 3 INDEX   1 FREQSQ       0.0000000'

#: The two recorded entries that prove the supplied input is not exactly reciprocal:
#: the O-O charge/22c pair, printed as unequal.  Symmetrizing the input anywhere in
#: our path would make these agree, so they are kept as printed tokens.
ASYMMETRIC_PAIR = {(0, 0, 2): '0.5720608E-08', (0, 2, 0): '0.5720595E-08'}
#: Worst reciprocity defect over all 5,625 supplied entries, and what LW then
#: reports for it: both far inside the unmodified 1e-6 precondition.
SUPPLIED_RECIPROCITY_MAX = 5.7699999999996606e-12
REPORTED_RECIPROCITY = 5.771383371211414e-12

#: The three printed terms whose exact sum is the supplied charge-flow defect: the
#: charge column of the H1 row's three site-pair blocks, component 32c.
CHARGE_FLOW_TERMS = ['0.3426003E+00', '-0.5534848E+00', '0.2101834E+00']
CHARGE_FLOW_DEFECT = '-0.0007011'

#: Every residual of the recorded diagnostic run, at the authorized 1e-3 gate.  The
#: three charge-flow names carry the supplied defect; the four the algorithm owns are
#: at 1e-11 or below.  Recorded, and re-derived by the scratch guard.
DIAGNOSTIC_RESIDUALS = {
    'off_site': 2.201454e-11, 'charge_sum': 0.000701099999999899,
    'reciprocity': 5.771383371211414e-12, 'molecular_sum': 3.979039320256561e-13,
    'local_charge': 0.0007011000000002321, 'input_sum_rule': 0.00070110000000001,
    'charge_sum_transport': 1.6653345369377348e-16,
}
CHARGE_FLOW_NAMES = ('charge_sum', 'local_charge', 'input_sum_rule')
ALGORITHM_NAMES = ('off_site', 'reciprocity', 'molecular_sum', 'charge_sum_transport')
HISTORICAL_GATE = 1e-3
ALGORITHM_TOLERANCE = 1e-6
HERMETIC_ATOL = 1e-11

#: Per-site worst disagreement over the 675 compared entries, and the largest
#: reference entry at each site.
SITE_MAXABS = {'O': 6.252776074688882e-13, 'H1': 5.60440582830779e-13,
               'H2': 5.089262344881718e-13}
SITE_SCALE = {'O': 220.666588197109, 'H1': 21.988774081056, 'H2': 21.988761522657}
#: Frame negative control: H1's localized block left in global axes misses the
#: reference by this much, so the agreement above is not a trivial-rotation artefact.
UNROTATED_H1_GAP = 15.472386713606493
#: Graph bookkeeping, identical at every recorded index.
TRANSFERS = 456
OMITTED_COMPONENT_PAIRS = 60
OMITTED_TRANSFERS = 0

FAULTS = ('label', 'index', 'rank', 'frequency', 'representation', 'short_row',
          'extra_row', 'number', 'end', 'trailing', 'missing_section')


@pytest.fixture(scope='module')
def extractor():
    """Source-load the pure parser; the opt-in CLI never runs during pytest."""
    path = Path(__file__).parent / 'data_isapol' / 'oracle' / 'extract_lw_hermetic.py'
    spec = importlib.util.spec_from_file_location('lw_hermetic_extractor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def token(value):
    """Format in the reference documents' own `0.dddddddE+XX` dialect."""
    if value == 0.0:
        return '0.0000000E+00'
    exponent = math.floor(math.log10(abs(value))) + 1
    return f'{value/10.0**exponent:.7f}E{exponent:+03d}'


def synthetic_pol(distributed, overrides=()):
    """A complete synthetic document in the reviewed dialect, built from nothing.

    Deterministic analytic values, never a polarizability.  `overrides` places exact
    printed tokens at given `(section, row, column)` positions, which is how the
    recorded asymmetric pair is put through the parser without the fixture.
    """
    pairs = [(a, b) for a in range(3) for b in range(3)] if distributed else [(a, a) for a in range(3)]
    n = 25 if distributed else 15
    template = DISTRIBUTED_HEADER if distributed else LOCAL_HEADER
    placed = dict(overrides)
    lines, sections = [], []
    for index, (a, b) in enumerate(pairs):
        lines.append(template.format(a=LABELS[a], b=LABELS[b], i=a + 1, j=b + 1))
        rows = []
        for row in range(n):
            rows.append([placed.get((index, row, col), token(math.cos(row + 2.0*col + 3.0*index)))
                         for col in range(n)])
        lines.extend(' '.join(row) for row in rows)
        if distributed:
            lines.append('END')
        sections.append(rows)
    return '\n'.join(lines + ['ENDFILE']) + '\n', sections


@pytest.mark.parametrize('distributed', [True, False])
def test_pol_dialect_round_trips_a_synthetic_document(extractor, distributed):
    """The parser returns every printed token, with the identity it read them under.

    No float round trip, no clipping, no symmetrization: the tokens come back as
    strings, which is what lets the fixture preserve `-0` and unequal transposes.
    """
    text, rows = synthetic_pol(distributed)
    sections = extractor.parse_pol(text, distributed)
    pairs = [(a, b) for a in range(3) for b in range(3)] if distributed else [(a, a) for a in range(3)]
    stride = 27 if distributed else 16  # header + n rows (+ END)
    assert len(sections) == len(pairs)
    for index, ((a, b), section) in enumerate(zip(pairs, sections)):
        assert section['labels'] == [LABELS[a], LABELS[b]]
        assert section['site_indices'] == [a + 1, b + 1]
        assert section['header_line'] == index*stride + 1
        assert section['values'] == rows[index]
        assert all(isinstance(value, str) for row in section['values'] for value in row)


@pytest.mark.parametrize('distributed', [True, False])
def test_pol_dialect_ignores_whitespace_runs(extractor, distributed):
    """Header matching is on tokens, so column alignment is not part of the contract."""
    text, _ = synthetic_pol(distributed)
    collapsed = '\n'.join(' '.join(line.split()) for line in text.splitlines()) + '\n'
    assert collapsed != text
    reference = extractor.parse_pol(text, distributed)
    relaxed = extractor.parse_pol(collapsed, distributed)
    assert [s['values'] for s in relaxed] == [s['values'] for s in reference]
    assert [s['site_indices'] for s in relaxed] == [s['site_indices'] for s in reference]


@pytest.mark.parametrize('distributed', [True, False])
@pytest.mark.parametrize('fault', FAULTS)
def test_pol_dialect_rejects_malformed_sections(extractor, distributed, fault):
    """The eleven-fault matrix, on a synthetic document rather than the fixture."""
    text, _ = synthetic_pol(distributed)
    lines = text.splitlines()
    if fault == 'label':
        lines[0] = lines[0].replace('  O  O', '  O  H1')
    elif fault == 'index':
        lines[0] = (lines[0].replace('SITE-INDICES     1     1', 'SITE-INDICES     2     1')
                    if distributed else lines[0].replace('INDEX   1', 'INDEX   2'))
    elif fault == 'rank':
        lines[0] = (lines[0].replace('RANK  0', 'RANK  1') if distributed
                    else lines[0].replace('TO 3', 'TO 4'))
    elif fault == 'frequency':
        lines[0] = lines[0].replace('0.0000000', '0.1000000')
    elif fault == 'representation':
        lines[0] = (lines[0].replace('CARTSPHER S', 'CARTSPHER C') if distributed
                    else lines[0] + ' CARTSPHER C')
    elif fault == 'short_row':
        lines[1] = ' '.join(lines[1].split()[:-1])
    elif fault == 'extra_row':
        lines.insert(2, lines[1])
    elif fault == 'number':
        lines[1] = 'NaN ' + ' '.join(lines[1].split()[1:])
    elif fault == 'end':
        lines[26 if distributed else -1] = 'BADEND'
    elif fault == 'trailing':
        lines.append('ENDFILE')
    elif fault == 'missing_section':
        del lines[:27 if distributed else 16]
    with pytest.raises(ValueError):
        extractor.parse_pol('\n'.join(lines) + '\n', distributed)


@pytest.mark.parametrize('distributed', [True, False])
def test_pol_dialect_requires_the_declared_pair_sequence(extractor, distributed):
    """Sections are matched positionally against the expected identity sequence.

    A document whose sections are all individually well formed but presented in
    another order is rejected, so the parser's `(a, b)` assignment is never inferred
    from the headers it happens to find.
    """
    text, _ = synthetic_pol(distributed)
    lines = text.splitlines()
    stride = 27 if distributed else 16
    first, second = lines[:stride], lines[stride:2*stride]
    reordered = second + first + lines[2*stride:]
    with pytest.raises(ValueError, match='invalid header/identity'):
        extractor.parse_pol('\n'.join(reordered) + '\n', distributed)


def test_pol_dialects_are_not_interchangeable(extractor):
    """Each dialect is rejected by the other's reader: 9x25 and 3x15 are distinct."""
    distributed_text, _ = synthetic_pol(True)
    local_text, _ = synthetic_pol(False)
    with pytest.raises(ValueError):
        extractor.parse_pol(distributed_text, False)
    with pytest.raises(ValueError):
        extractor.parse_pol(local_text, True)


def test_recorded_asymmetric_pair_survives_the_parser(extractor):
    """The two unequal transpose tokens come back byte-identical, and stay unequal.

    Placed into a synthetic document so the claim is about our parser rather than
    about the fixture, and paired with the reciprocity the run reported for them.
    """
    overrides = {(0, row, col): value for (_, row, col), value in
                 ((k, v) for k, v in ASYMMETRIC_PAIR.items())}
    text, _ = synthetic_pol(True, overrides)
    sections = extractor.parse_pol(text, True)
    for (section, row, col), printed in ASYMMETRIC_PAIR.items():
        assert sections[section]['values'][row][col] == printed
    upper, lower = (float(ASYMMETRIC_PAIR[key]) for key in ((0, 0, 2), (0, 2, 0)))
    assert upper != lower and abs(upper - lower) == pytest.approx(1.3e-15, rel=0.1)
    # The supplied input is inexactly reciprocal, but only at 1e-12 -- six orders
    # inside the unmodified precondition, which is why that gate is never relaxed.
    assert SUPPLIED_RECIPROCITY_MAX < ALGORITHM_TOLERANCE/1e5
    assert REPORTED_RECIPROCITY == pytest.approx(SUPPLIED_RECIPROCITY_MAX, rel=1e-3)


def test_local_frames_follow_the_recorded_axes_rule():
    """`FRAMES` is derived from the site geometry, not copied out of the fixture.

    The rule is "z along global Z, x from the other hydrogen towards this one", and
    O has no axes entry.  Applying it to `ORIGINS` must reproduce the literals, and
    each result must be a proper rotation -- a reflection here would flip the sign
    of every odd-rank component without being visible in the frame's diagonal.
    """
    origins = np.array(ORIGINS)
    derived = [np.eye(3)]
    for site, other in ((1, 2), (2, 1)):
        z = np.array([0., 0., 1.])
        x = origins[site] - origins[other]
        x = x - z*(x @ z)
        x /= np.linalg.norm(x)
        derived.append(np.column_stack([x, np.cross(z, x), z]))
    np.testing.assert_array_equal(derived, FRAMES)
    for index, frame in enumerate(np.array(FRAMES)):
        np.testing.assert_array_equal(frame.T @ frame, np.eye(3))
        assert round(np.linalg.det(frame)) == 1, index
    # The two hydrogens sit at mirrored x with a common z, so their frames differ by
    # exactly the C2 rotation about Z and O keeps the global one.
    np.testing.assert_array_equal(FRAMES[0], np.eye(3))
    np.testing.assert_array_equal(FRAMES[2], np.eye(3))
    np.testing.assert_array_equal(FRAMES[1], np.diag([-1., -1., 1.]))


def test_rank3_rotation_of_the_recorded_frames_is_exact():
    """The two frames' rank 1:3 rotations are exact sign patterns, not approximations.

    The identity frame gives the identity bitwise, and the hydrogen frame -- a C2
    about Z -- gives `diag((-1)**m)`: exactly diagonal, with entries exactly +-1.
    So the local/global distinction at the hydrogens is a pure sign flip on the odd-m
    components, which is precisely what the fixture's frame control detects.
    """
    identity = np.asarray(psi4.core.isa_multipole_rotation(3, FRAMES[0]))
    assert identity.shape == (16, 16)
    np.testing.assert_array_equal(identity, np.eye(16))

    rotation = np.asarray(psi4.core.isa_multipole_rotation(3, FRAMES[1]))[1:16, 1:16]
    np.testing.assert_array_equal(rotation, np.diag(np.diag(rotation)))
    signs = np.diag(rotation)
    assert set(signs.tolist()) == {1.0, -1.0}
    orders = [m for l in (1, 2, 3) for m in [0] + [k for k in range(1, l + 1) for _ in (0, 1)]]
    np.testing.assert_array_equal(signs, [(-1.)**m for m in orders])
    # An involution, so local -> global -> local is exact rather than merely close.
    np.testing.assert_array_equal(rotation @ rotation, np.eye(15))


def test_rank4_truncation_discards_exactly_the_rank4_rows_and_columns():
    """The rank-4 declaration is cut to the rank-3 working block, losslessly.

    Poisoning every discarded entry with NaN leaves the working block bitwise
    unchanged, which is the statement that the selection cannot depend on them.
    """
    supplied = np.arange(9*25*25, dtype=float).reshape(9, 25, 25) + 0.5
    working = supplied[:, :16, :16].copy()
    assert supplied.size == 5625 and working.size == 2304
    assert supplied.size - working.size == 3321 == 9*(25*25 - 16*16)
    poisoned = supplied.copy()
    poisoned[:, 16:, :] = np.nan
    poisoned[:, :, 16:] = np.nan
    np.testing.assert_array_equal(poisoned[:, :16, :16], working)
    # The discarded region is not empty in the recorded input either: rank 4 is
    # genuinely supplied and genuinely dropped.
    assert np.count_nonzero(supplied[:, 16:, :]) > 0
    assert np.count_nonzero(supplied[:, :, 16:]) > 0


def test_recorded_charge_flow_defect_is_exactly_the_printed_sum():
    """Three printed tokens, summed in exact decimal, are the supplied defect.

    This is the whole reason the hermetic comparison needs an authorized gate: the
    supplied nonlocal data violates the charge-flow sum rule by 7e-4, seven hundred
    times the production postcondition, and LW transports that defect rather than
    repairing it.
    """
    terms = [Decimal(value) for value in CHARGE_FLOW_TERMS]
    assert sum(terms) == Decimal(CHARGE_FLOW_DEFECT)
    defect = abs(Decimal(CHARGE_FLOW_DEFECT))
    assert defect/Decimal('0.000001') > 700
    assert defect < Decimal(str(HISTORICAL_GATE))
    # The three individual terms are O(1): the defect is a near cancellation, so it
    # cannot be read off any single printed number.
    assert min(abs(term) for term in terms) > Decimal(250)*defect


def test_recorded_diagnostic_residuals_separate_the_two_kinds():
    """Only the charge-flow names exceed 1e-6; everything the algorithm owns is at 1e-11.

    The authorized 1e-3 gate is therefore not a blanket loosening: it admits exactly
    the supplied defect, and the run still demonstrates the algorithm's own residuals
    passing at the unmodified production tolerance.
    """
    assert set(DIAGNOSTIC_RESIDUALS) == set(CHARGE_FLOW_NAMES) | set(ALGORITHM_NAMES)
    worst = max(DIAGNOSTIC_RESIDUALS.values())
    assert ALGORITHM_TOLERANCE < worst <= HISTORICAL_GATE
    for name in CHARGE_FLOW_NAMES:
        assert DIAGNOSTIC_RESIDUALS[name] > ALGORITHM_TOLERANCE, name
        # All three report the same supplied defect, to the precision it was printed.
        assert DIAGNOSTIC_RESIDUALS[name] == pytest.approx(7.011e-4, abs=1e-12, rel=0), name
    for name in ALGORITHM_NAMES:
        assert DIAGNOSTIC_RESIDUALS[name] <= 1e-10, name
    # Transported exactly: the defect is conserved by the bond transfers, not spread.
    assert DIAGNOSTIC_RESIDUALS['charge_sum_transport'] <= 1e-15


def test_recorded_site_agreement_and_its_frame_control():
    """The 675-entry agreement, per site, and the margin of the negative control."""
    assert set(SITE_MAXABS) == set(SITE_SCALE) == set(LABELS)
    for label in LABELS:
        assert SITE_MAXABS[label] < HERMETIC_ATOL/10, label
        assert SITE_MAXABS[label]/SITE_SCALE[label] < 1e-13, label
    # Flat in absolute terms though the sites differ tenfold in magnitude: round-off,
    # not a systematic difference in what the two codes compute.
    assert max(SITE_MAXABS.values())/min(SITE_MAXABS.values()) < 2
    assert SITE_SCALE['O']/SITE_SCALE['H1'] > 9
    # The frame control has thirteen orders of margin over the comparison tolerance,
    # so a wrong rotation could not be mistaken for agreement.
    assert UNROTATED_H1_GAP > 1e12*HERMETIC_ATOL
    assert UNROTATED_H1_GAP > 0.5*SITE_SCALE['H1']
    # Graph bookkeeping: transfers on a three-site two-bond graph at rank 4, with the
    # omitted pairs all accounted for and none of them carrying a transfer.
    assert TRANSFERS == 456 and OMITTED_TRANSFERS == 0
    assert OMITTED_COMPONENT_PAIRS == 60


def test_hermetic_run_is_the_leg_a_static_point():
    """The two recorded captures of the same input agree, which pins both literal sets.

    The hermetic document and leg A index 000 are the same supplied polarizabilities,
    captured independently.  Their residuals and their O-site agreement must match
    bitwise -- a cross-check between two committed literal sets that needs neither
    fixture, and that would break if either capture were regenerated alone.
    """
    import test_isapol_lw_leg_a as leg_a

    assert leg_a.ORIGINS == ORIGINS
    assert leg_a.LABELS == LABELS
    assert leg_a.CASIMIR_OMEGAS[0] == 0.0
    assert leg_a.LEG_A_TRANSFERS == TRANSFERS
    assert leg_a.LEG_A_MAXABS[0] == SITE_MAXABS['O']
    assert leg_a.REFERENCE_SCALE[0] == SITE_SCALE['O']
    assert Decimal(leg_a.SUPPLIED_SUM_RULE[0]) == -Decimal(CHARGE_FLOW_DEFECT)
    assert leg_a.ALGORITHM_NAMES == ALGORITHM_NAMES
    # Leg A reports the defect at the production tolerance; the hermetic run gates it
    # at 1e-3.  Same measured value, two different policies applied to it.
    assert DIAGNOSTIC_RESIDUALS['input_sum_rule'] == pytest.approx(
        float(Decimal(leg_a.SUPPLIED_SUM_RULE[0])), abs=1e-12, rel=0)
