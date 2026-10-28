# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Portable `.pol` parser contracts and native rank-3 rotation checks.

Historical numerical tokens preserve raw asymmetry. Full molecular reference
comparison is external development evidence, not asserted by this module.
"""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]

LABELS = ('O', 'H1', 'H2')
#: Local-to-global column frames, from the reference's own `H2O.axes` rule
#: "H1 z global Z x from H2 to H1 / H2 z global Z x from H1 to H2".  O has no axes
#: entry and so keeps the global frame.
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
    """The fault matrix, on a synthetic document rather than the fixture."""
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
    assert upper != lower and abs(upper - lower) == pytest.approx(1.3e-14, rel=0.1, abs=0)


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
