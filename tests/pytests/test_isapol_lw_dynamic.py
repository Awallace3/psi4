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
FIXTURE_SHA256 = '132283408a5906e523231df9f99b1dcec2b88a29eb773b0c867e541a7eeced20'
WORST_SUMS = ['-0.0007011', '-0.0007001', '-0.0006940', '-0.0006727', '-0.0006185',
              '-0.00051040', '-0.00034365', '0.00017116', '-0.00006008', '-0.000002990']


@pytest.fixture(scope='module')
def fixture():
    raw = (DATA / 'lw-dynamic-water.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256
    return json.loads(raw)


def serialize(record, distributed):
    lines = []
    for section in record['sections']:
        lines.append(section['header'])
        lines.extend(' '.join(row) for row in section['values'])
        if distributed:
            lines.append('END')
    return '\n'.join(lines + ['ENDFILE']) + '\n'


def test_provenance_structure_and_complete_literals(fixture):
    f = fixture
    assert f['schema_version'] == 1
    assert f['units'] == 'atomic' and f['geometry_units'] == 'bohr'
    assert f['input_frame'] == 'global' and f['expected_frame'] == 'site_local'
    assert f['frame_convention'] == 'local_to_global_columns'
    assert f['bonds_zero_based'] == [[0, 1], [0, 2]]
    assert f['sites'] == json.loads((DATA / 'manifest.json').read_text())['sites']
    assert f['provenance']['extractor_sha256'] == hashlib.sha256(EXTRACTOR.read_bytes()).hexdigest()
    for name in ('manifest.json', 'frequency_header_excerpt.json', 'H2O.sites', 'H2O.axes', 'H2O.ornt'):
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == f['authority_sha256'][name]
    # Inventory paths are strings only, never followed by portable tests.
    assert f['authority_sha256']['orient-bridge-input-hashes.json'] == parser.INVENTORY_SHA256
    excerpt = json.loads((DATA / 'frequency_header_excerpt.json').read_text())['rows']
    assert [n['index'] for n in f['nodes']] == list(range(1, 11))
    total = 0
    for node in f['nodes']:
        for key, distributed, size in [('distributed', True, 25), ('expected_local', False, 15)]:
            r = node[key]
            e, = [e for e in excerpt if Path(e['source']).name == r['filename']]
            assert r['sha256'] == r['adjacent_sha256'] == e['sha256']
            assert r['frequency_squared'] == e['freqsq_token']
            assert r['sections'][0]['header'] == e['header']
            assert r['raw_frequency_index'] == node['index'] + 1
            assert r['representation'] == 'real_Racah_spherical'
            assert (r['rank_min'], r['rank_max']) == ((0, 4) if distributed else (1, 3))
            assert len(r['sections']) == (9 if distributed else 3)
            parsed = parser.parse_pol(serialize(r, distributed), distributed, node['index'], r['frequency_squared'])
            assert parsed == r['sections']  # all 63000 tokens, signed zeros, labels, raw headers, line indices
            for s in r['sections']:
                assert len(s['values']) == size
                assert all(len(row) == size for row in s['values'])
                assert all(isinstance(t, str) and parser.NUMBER.fullmatch(t) for row in s['values'] for t in row)
                total += size * size
    assert total == 63000
    assert f['rank_policy']['retained_per_node'] == 2304
    assert f['rank_policy']['discarded_rank4_per_node'] == 3321
    assert f['component_order'].split(',')[12] == '32c'
    assert len(f['component_order'].split(',')) == 25
    assert f['quadrature']['exact_original_producer_quadrature_verified'] is False
    assert f['provenance']['generator_commands'] is None  # absent authority, not invented history


def test_chosen_grid_and_four_literal_header_failures(fixture):
    from psi4 import core
    grid = core.CasimirGrid(10, 0.5)
    failures = []
    for node in fixture['nodes']:
        i = node['index']
        assert node['omega'] == grid.omega(i) > 0
        assert node['cp_weight'] == grid.cp_weight(i) > 0
        assert parser.printed_matches(node['omega'], node['distributed']['frequency_squared'])
        assert node['distributed']['chosen_node_header_agreement'] is True
        agrees = parser.printed_matches(node['omega'], node['expected_local']['frequency_squared'])
        assert agrees == node['expected_local']['chosen_node_header_agreement']
        if not agrees:
            failures.append(i)
    assert failures == [7, 8, 9, 10]
    assert grid.cp_weight(0) == 0


@pytest.mark.parametrize('i', range(10))
def test_strict_rejection_no_dynamic_waiver(fixture, i):
    from psi4 import core
    node = fixture['nodes'][i]
    values = [s['values'] for s in node['distributed']['sections']]
    sums = [sum(Decimal(values[3*a+b][k][0]) for b in range(3)) for a in range(3) for k in range(16)]
    assert max(sums, key=abs) == Decimal(WORST_SUMS[i])
    assert max(map(abs, sums)) > Decimal('0.000001')
    raw = np.array(values, float)
    working = raw[:, :16, :16].copy()
    assert working.size == 2304 and raw.size-working.size == 3321
    poisoned = raw.copy()
    poisoned[:, 16:, :] = np.nan
    poisoned[:, :, 16:] = np.nan
    np.testing.assert_array_equal(poisoned[:, :16, :16], working)
    matrix = lambda x: core.Matrix.from_array(np.asarray(x, dtype=float))
    # Default production tolerance, no exception catch/retry or relaxed output comparison.
    with pytest.raises(RuntimeError, match=r'postcondition exceeds residual tolerance .*charge-sum=.*local-charge='):
        core.isa_localize_lw(matrix([s['origin'] for s in fixture['sites']]),
                            [matrix(b) for b in working], node['omega'], fixture['bonds_zero_based'])


@pytest.mark.parametrize('i', range(10))
def test_reference_only_frames_and_raw_asymmetry(fixture, i):
    from psi4 import core
    node = fixture['nodes'][i]
    raw = np.array([s['values'] for s in node['distributed']['sections']], float).reshape(3, 3, 25, 25)
    assert np.max(np.abs(raw-raw.transpose(1, 0, 3, 2))) > 0  # no tensor repair
    for a, site in enumerate(fixture['sites']):
        expected = np.array(node['expected_local']['sections'][a]['values'], float)
        d = np.asarray(core.isa_multipole_rotation(3, site['frame']))[1:, 1:]
        np.testing.assert_allclose(d.T@(d@expected@d.T)@d, expected, atol=1e-11, rtol=0)
        if a == 1:
            assert np.max(np.abs(d@expected@d.T-expected)) > 1e-11
        else:
            np.testing.assert_array_equal(d, np.eye(15))


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
