# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Portable dynamic literal import and truthful strict-rejection regressions.

No external paths are opened, no reference executables or extractor CLI invoked.
Passing these tests is NOT molecular localization/dispersion acceptance.
"""
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / 'data_isapol/orient_local'
EXTRACTOR = DATA.parent / 'oracle/extract_lw_dynamic.py'
SPEC = importlib.util.spec_from_file_location('_lw_dynamic_literal_parser', EXTRACTOR)
parser = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parser)

#: The two frequency nodes retained under version control out of the reference's
#: ten: the lowest (index 1) and a mid node (index 5).  The full ten-node import
#: lives in `agent_scratch/pytests/test_isapol_lw_dynamic_full.py` together with
#: the 70,681-line fixture these numbers were selected from.
#:
#: These are reference *inputs*, so the numerical claim below is narrow: Psi4's
#: own quadrature and multipole-rotation machinery reproduce the grid and the
#: frame algebra the reference used.  It is NOT an end-to-end localization
#: check -- `isa_localize_lw` only accepts whole 16x16/25x25 site blocks, which
#: cannot be hand-listed, so that comparison stays in `agent_scratch/`.
CASIMIR_NODES = [
    (1, 0.006609601596087073, 0.002723367038256463),
    (5, 0.3704172128053662, 0.03563429408419695),
]

#: `(index, distributed FREQ2 token, expected-local FREQSQ token, tokens agree)`
#: for all ten nodes.  Short strings, so the whole finding is affordable here:
#: the reference's *distributed* headers round-trip at every node, while its
#: *local* headers lose the frequency for nodes 7-10.
NODE_HEADERS = [
    (1, '-0.4368683E-04', '-0.0000437', True),
    (2, '-0.1308617E-02', '-0.0013086', True),
    (3, '-0.9110199E-02', '-0.0091102', True),
    (4, '-0.3906323E-01', '-0.0390632', True),
    (5, '-0.1372089E+00', '-0.1372089', True),
    (6, '-0.4555098E+00', '-0.4555098', True),
    (7, '-0.1599970E+01', '-1.5999700', False),
    (8, '-0.6860443E+01', '-6.8604430', False),
    (9, '-0.4776034E+02', '-47.7603400', False),
    (10, '-0.1430637E+04', '-1430.6370000', False),
]

#: H1 is the only site of the reference whose local frame is not the identity:
#: a C2 rotation about z.  For that frame the rank 0:3 multipole rotation is
#: diagonal with this signature -- every odd-m component changes sign.
H1_FRAME = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
H1_C2_SIGNATURE = [1,            # 00
                   1, -1, -1,    # 10 11c 11s
                   1, -1, -1, 1, 1,          # 20 21c 21s 22c 22s
                   1, -1, -1, 1, 1, -1, -1]  # 30 31c 31s 32c 32s 33c 33s

#: `(node index, local rank 1:3 diagonal, (element, value))` for H1 at the two
#: retained nodes.  The diagonal is the per-component polarizability, the
#: natural physical unit of this tensor; the extra element is where the C2
#: rotation moves the tensor furthest, i.e. the reference's own worst case for
#: `d @ L @ d.T != L`.  Element (10, 12) is the 31s-32s coupling: 31s flips
#: sign under C2 and 32s does not.
H1_LOCAL = [
    (1, [2.008360927604, 1.557254268788, 1.620448, 4.783445627734, 1.169667685575,
         2.848845165781, 3.094272422379, 1.261242099739, 16.783192447694,
         8.353290824146, 14.502487160621, 5.370672081708, -16.901485259017,
         9.931837998097, 21.990586204665], ((10, 12), -7.736619107569)),
    (5, [1.485446758536, 1.238186307925, 1.216268, 3.61086805941, 1.38508810844,
         2.546825301442, 1.987455572502, 1.338680156389, 8.705821553639,
         6.465684323996, 9.734393575542, 4.743913604895, -14.564689040883,
         8.098535577568, 19.711642557296], ((10, 12), -7.879989327304)),
]


def test_retained_casimir_nodes_are_the_reference_quadrature():
    """Psi4's `Quad 10, Beta 0.5` grid at the two retained reference nodes."""
    from psi4 import core
    grid = core.CasimirGrid(10, 0.5)
    for index, omega, weight in CASIMIR_NODES:
        assert grid.omega(index) == omega > 0
        assert grid.cp_weight(index) == weight > 0
    assert grid.cp_weight(0) == 0


def test_reference_headers_round_trip_against_psi4_frequencies():
    """All ten printed headers, reproduced from Psi4's own node frequencies.

    Keeps the finding the full import made: the reference's local pol files
    print too few digits to identify nodes 7-10.
    """
    from psi4 import core
    grid = core.CasimirGrid(10, 0.5)
    failures = []
    for index, distributed, local, agrees in NODE_HEADERS:
        omega = grid.omega(index)
        assert parser.printed_matches(omega, distributed)
        assert parser.printed_matches(omega, local) == agrees
        if not agrees:
            failures.append(index)
    assert failures == [7, 8, 9, 10]


def test_h1_local_frame_is_a_sign_flip_of_the_odd_m_components():
    """The rank 0:3 multipole rotation of the reference's one nontrivial frame."""
    from psi4 import core
    d = np.asarray(core.isa_multipole_rotation(3, H1_FRAME))
    assert d.shape == (16, 16)
    np.testing.assert_array_equal(d, np.diag(np.diag(d)))
    np.testing.assert_allclose(np.diag(d), H1_C2_SIGNATURE, atol=1e-15, rtol=0)


@pytest.mark.parametrize('index,diagonal,element', [(i, d, e) for i, d, e in H1_LOCAL])
def test_h1_local_tensor_selected_elements_under_its_own_frame(index, diagonal, element):
    """Frame algebra on the retained elements, not on the full 15x15 tensor.

    `d.T @ (d @ L @ d.T) @ d == L` must hold elementwise, and the C2 rotation
    must genuinely move the tensor: the retained off-diagonal element flips
    sign, while the diagonal -- being a product of a component with itself --
    cannot and is therefore checked for invariance instead.
    """
    from psi4 import core
    d = np.asarray(core.isa_multipole_rotation(3, H1_FRAME))[1:, 1:]
    (row, col), value = element
    # Reassemble only the retained elements of the reference tensor and push
    # them through Psi4's actual rank 1:3 rotation.
    local = np.diag(np.array(diagonal, float))
    local[row, col] = local[col, row] = value
    rotated = d @ local @ d.T
    np.testing.assert_allclose(np.diag(rotated), diagonal, atol=1e-13, rtol=0)
    assert rotated[row, col] == -value       # the C2 does move the tensor
    np.testing.assert_allclose(d.T @ rotated @ d, local, atol=1e-13, rtol=0)


def synthetic(distributed):
    size = 25 if distributed else 15
    lines = []
    pairs = [(a, b) for a in range(3) for b in range(3)] if distributed else [(a, a) for a in range(3)]
    for a, b in pairs:
        if distributed:
            lines.append(f'ALPHA INDEX 002 SITE-LABELS {parser.LABELS[a]} {parser.LABELS[b]} SITE-INDICES {a+1} {b+1} RANK 0 : 4 BY 0 : 4 FREQ2 -0.1000000E+00 CARTSPHER S')
        else:
            lines.append(f'ALPHA H2O SITE-NAMES {parser.LABELS[a]} {parser.LABELS[b]} RANK 1 TO 3 INDEX 2 FREQSQ -0.1000000E+00')
        lines.extend([' '.join(['-0.0000000E+00']*size)]*size)
        if distributed:
            lines.append('END')
    return lines + ['ENDFILE']


@pytest.mark.parametrize('distributed', [False, True])
@pytest.mark.parametrize('mutation', ['label', 'index', 'rank', 'frequency', 'extra_header', 'short_row',
                                     'long_row', 'nan', 'inf', 'integer', 'bad_exponent', 'comma',
                                     'endfile', 'trailing', 'missing_block', 'reordered', 'duplicate',
                                     'short_file', 'extra_end', 'token_suffix'])
def test_bounded_synthetic_malformed_inputs(distributed, mutation):
    lines = synthetic(distributed)
    size = 25 if distributed else 15
    stride = size + (2 if distributed else 1)
    if mutation == 'label': lines[0] = lines[0].replace('O O', 'O H1')
    elif mutation == 'index': lines[0] = lines[0].replace('INDEX 002' if distributed else 'INDEX 2', 'INDEX 003' if distributed else 'INDEX 3')
    elif mutation == 'rank': lines[0] = lines[0].replace('0 : 4' if distributed else '1 TO 3', '0 : 3' if distributed else '1 TO 4')
    elif mutation == 'frequency': lines[0] = lines[0].replace('-0.1000000', '-0.2000000')
    elif mutation == 'extra_header': lines[0] += ' EXTRA'
    elif mutation == 'short_row': lines[1] = ' '.join(lines[1].split()[:-1])
    elif mutation == 'long_row': lines[1] += ' 0.0'
    elif mutation in ('nan', 'inf', 'integer', 'bad_exponent', 'comma', 'token_suffix'):
        token = {'nan': 'NaN', 'inf': '-Inf', 'integer': '0', 'bad_exponent': '0.0E+', 'comma': '0,0', 'token_suffix': '0.0junk'}[mutation]
        lines[1] = ' '.join([token] + lines[1].split()[1:])
    elif mutation == 'endfile': lines[-1] = 'END'
    elif mutation == 'trailing': lines.append('ignored?')
    elif mutation == 'missing_block': del lines[stride:2*stride]
    elif mutation == 'reordered': lines[:2*stride] = lines[stride:2*stride] + lines[:stride]
    elif mutation == 'duplicate': lines[stride:2*stride] = lines[:stride]
    elif mutation == 'short_file': lines = lines[:3]
    elif mutation == 'extra_end': lines.insert(1+size, 'END')
    with pytest.raises(ValueError):
        parser.parse_pol('\n'.join(lines)+'\n', distributed, 1, '-0.1000000E+00')


@pytest.mark.parametrize('distributed', [False, True])
def test_synthetic_signed_zero_and_termination(distributed):
    lines = synthetic(distributed)
    records = parser.parse_pol('\n'.join(lines)+'\n', distributed, 1, '-0.1000000E+00')
    assert all(t == '-0.0000000E+00' for s in records for row in s['values'] for t in row)
    if distributed:
        lines[26] = 'WRONG'
        with pytest.raises(ValueError, match='missing END'):
            parser.parse_pol('\n'.join(lines), True, 1, '-0.1000000E+00')
        lines = synthetic(True)
        lines[0] = lines[0].replace('SITE-INDICES 1 1', 'SITE-INDICES 1 2')
        with pytest.raises(ValueError, match='header'):
            parser.parse_pol('\n'.join(lines), True, 1, '-0.1000000E+00')
        lines[0] = synthetic(True)[0].replace('CARTSPHER S', 'CARTSPHER C')
        with pytest.raises(ValueError, match='header'):
            parser.parse_pol('\n'.join(lines), True, 1, '-0.1000000E+00')


@pytest.mark.parametrize('index,token', [(0, '-0.1'), (11, '-0.1'), (True, '-0.1'), (1, '0.0'), (1, 'NaN'), (1, '0.1')])
def test_invalid_parser_contract(index, token):
    with pytest.raises(ValueError):
        parser.parse_pol('\n'.join(synthetic(True)), True, index, token)


def test_size_bound():
    with pytest.raises(ValueError, match='bounded'):
        parser.parse_pol(' '*100001, True, 1, '-0.1')
