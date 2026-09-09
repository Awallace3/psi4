#!/usr/bin/env python3
# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Opt-in import of exactly twenty historical numerical files; no reference code.

Parser is independent of Psi4 and consumes the complete bounded dense dialect.
CLI uses installed/staged core only to record a CHOSEN integration rule, not to
infer the original producer's Newton-root response quadrature from rounded headers.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import re

LABELS = ('O', 'H1', 'H2')
BASE = Path('/home/awallace43/gits/camcasp_psi4/.camcasp-reference/work/H2O')
INVENTORY_SHA256 = '236a0d86949033150e1459c1d9e95ec399c50ee242fc0256240351126c455176'
EXCERPT_SHA256 = 'e562125d901f8bd3b54dcfc34fdc3287a36e6290b71e2e9f9ac885cf769858b3'
MANIFEST_SHA256 = 'dea5868b6f9bb9ec2ed77ace9d3dd0ce7cf6238c7394af1de4fa8778d5f275f0'
NUMBER = re.compile(r'[+-]?\d+\.\d+(?:E[+-]\d+)?')


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def printed_matches(omega, token):
    """Compare -omega**2 to the half-unit interval of the printed decimal."""
    with localcontext() as ctx:
        ctx.prec = 50
        value = Decimal(token)
        square = -Decimal.from_float(float(omega)) ** 2
        return abs(square - value) <= Decimal(5).scaleb(value.as_tuple().exponent - 1)


def parse_pol(text, distributed, index, frequency_squared):
    """Preserve raw headers and every decimal token; reject any extra content.

    index is the positive-node filename index (1..10); raw INDEX is index+1.
    L3 has no numeric site indices or CARTSPHER header: these are declared
    metadata, not invented source fields. Repeated frequency tokens must agree
    literally with the independently pinned header excerpt supplied by caller.
    """
    if type(index) is not int or not 1 <= index <= 10:
        raise ValueError('positive node index must be 1..10')
    if not isinstance(frequency_squared, str) or not NUMBER.fullmatch(frequency_squared):
        raise ValueError('invalid frequency token')
    if Decimal(frequency_squared) >= 0:
        raise ValueError('dynamic imaginary frequency requires negative FREQSQ')
    if len(text) > 100000:
        raise ValueError('input exceeds bounded file size')
    lines = text.splitlines()
    cursor = 0
    sections = []

    def take():
        nonlocal cursor
        if cursor >= len(lines):
            raise ValueError(f'premature EOF at line {cursor + 1}')
        line = lines[cursor]
        cursor += 1
        return line

    pairs = [(a, b) for a in range(3) for b in range(3)] if distributed else [(a, a) for a in range(3)]
    size = 25 if distributed else 15
    for a, b in pairs:
        line_number = cursor + 1
        header = take()
        if distributed:
            expected = (f'ALPHA INDEX {index+1:03} SITE-LABELS {LABELS[a]} {LABELS[b]} '
                        f'SITE-INDICES {a+1} {b+1} RANK 0 : 4 BY 0 : 4 '
                        f'FREQ2 {frequency_squared} CARTSPHER S')
        else:
            expected = (f'ALPHA H2O SITE-NAMES {LABELS[a]} {LABELS[b]} RANK 1 TO 3 '
                        f'INDEX {index+1} FREQSQ {frequency_squared}')
        if header.split() != expected.split():
            raise ValueError(f'invalid header/order/identity at line {line_number}: {header!r}')
        values = []
        for _ in range(size):
            tokens = take().split()
            if len(tokens) != size or any(len(t) > 40 or not NUMBER.fullmatch(t) for t in tokens):
                raise ValueError(f'invalid dense numerical row at line {cursor}')
            values.append(tokens)
        if distributed and take() != 'END':
            raise ValueError(f'missing END at line {cursor}')
        sections.append({'labels': [LABELS[a], LABELS[b]], 'site_indices': [a+1, b+1],
                         'site_indices_authority': 'raw_header' if distributed else 'ordered_SITE-NAMES',
                         'header_line': line_number, 'header': header, 'values': values})
    if take() != 'ENDFILE' or cursor != len(lines):
        raise ValueError('missing ENDFILE or trailing content')
    return sections


def pinned_json(path, expected):
    raw = path.read_bytes()
    if sha256(raw) != expected:
        raise ValueError(f'BLOCKER authority hash: {path}: {sha256(raw)} != {expected}')
    return json.loads(raw)


def extract(root, output):
    # Never enumerate any external directory or follow paths supplied by its files.
    data = root / 'tests/pytests/data_isapol/orient_local'
    inventory = pinned_json(root / '.pi/audit/orient-bridge-input-hashes.json', INVENTORY_SHA256)
    excerpt = pinned_json(data / 'frequency_header_excerpt.json', EXCERPT_SHA256)
    manifest = pinned_json(data / 'manifest.json', MANIFEST_SHA256)
    from psi4 import core
    grid = core.CasimirGrid(10, 0.5)
    nodes = []
    for i in range(1, 11):
        node = {'index': i, 'omega': grid.omega(i), 'cp_weight': grid.cp_weight(i)}
        for kind, key in [('NL4', 'distributed'), ('L3', 'expected_local')]:
            path = BASE / f'H2O_{kind}_{i:03}.pol'
            inv = [r for r in inventory if r['path'] == str(path)]
            hdr = [r for r in excerpt['rows'] if r['source'] == str(path) and r['canonical_index'] == i]
            if len(inv) != 1 or len(hdr) != 1 or inv[0]['sha256'] != hdr[0]['sha256']:
                raise ValueError(f'BLOCKER inventory/header identity: {path}')
            raw = path.read_bytes()
            actual = sha256(raw)
            if actual != inv[0]['sha256'] or len(raw) != inv[0]['bytes']:
                raise ValueError(f'BLOCKER {path}: SHA256={actual}, bytes={len(raw)}; expected {inv[0]}')
            adjacent = path.with_suffix('.pol.sha256')
            adjacent_token = adjacent.read_text().split()[0] if adjacent.exists() else None
            if adjacent_token is not None and (adjacent_token != actual or adjacent_token != inv[0]['adjacent_sha256']):
                raise ValueError(f'BLOCKER adjacent hash: {adjacent}: {adjacent_token} != {actual}')
            token = hdr[0]['freqsq_token']
            sections = parse_pol(raw.decode('ascii'), kind == 'NL4', i, token)
            if sections[0]['header'] != hdr[0]['header']:
                raise ValueError(f'BLOCKER literal header excerpt: {path}')
            agrees = printed_matches(node['omega'], token)
            if kind == 'NL4' and not agrees:
                raise ValueError(f'BLOCKER chosen node mapping fails NONLOCAL header: {path}')
            node[key] = {'filename': path.name, 'sha256': actual, 'bytes': len(raw),
                         'adjacent_sha256': adjacent_token, 'raw_frequency_index': i+1,
                         'frequency_squared': token, 'chosen_node_header_agreement': agrees,
                         'rank_min': 0 if kind == 'NL4' else 1, 'rank_max': 4 if kind == 'NL4' else 3,
                         'representation': 'real_Racah_spherical', 'sections': sections}
        nodes.append(node)
    failures = [n['index'] for n in nodes if not n['expected_local']['chosen_node_header_agreement']]
    if failures != [7, 8, 9, 10]:
        raise ValueError(f'BLOCKER changed known L3 header failures: {failures}')
    result = {
        'schema_version': 1, 'molecule': 'H2O', 'units': 'atomic', 'geometry_units': 'bohr',
        'input_frame': 'global', 'expected_frame': 'site_local',
        'frame_convention': 'local_to_global_columns', 'sites': manifest['sites'],
        'bonds_zero_based': [[0, 1], [0, 2]],
        'component_order': '00,10,11c,11s,20,21c,21s,22c,22s,30,31c,31s,32c,32s,33c,33s,40,41c,41s,42c,42s,43c,43s,44c,44s',
        'rank_policy': {'working': 'exact top-left 16x16 of each ordered 25x25 block; rank0 retained',
                        'expected': 'three independent unrefined L3 15x15 outputs',
                        'retained_per_node': 2304, 'discarded_rank4_per_node': 3321,
                        'discarded_policy': 'No rank4 contribution restored; no clipping, symmetrization, fitting or static extrapolation'},
        'quadrature': {'rule': 'chosen core.CasimirGrid(10,0.5)', 'n': 10, 'beta': 0.5,
                       'authority': 'manifest.json grid declaration + explicit dynamic-import task choice; mapping separately validated against every NONLOCAL header',
                       'exact_original_producer_quadrature_verified': False,
                       'limitation': 'Manifest originally maps refined sections; no exact Newton-root response nodes/weights available in permitted authority. Printed agreement does not certify exact producer quadrature.',
                       'cp_weight_convention': 'includes 1/(2*pi) exactly once; positive nodes only, no static contribution',
                       'header_check': '|-omega^2 - Decimal(token)| <= half printed decimal unit; no square root of rounded headers'},
        'authority_sha256': {'manifest.json': MANIFEST_SHA256, 'frequency_header_excerpt.json': EXCERPT_SHA256,
                             'orient-bridge-input-hashes.json': INVENTORY_SHA256,
                             **{r['name']: r['sha256'] for r in manifest['provenance_sources'] if r['name'] in ('H2O.sites', 'H2O.axes', 'H2O.ornt')}},
        'provenance': {'producer': 'external ORIENT 5.0.10 (d8d8610), historical H2O numerical output',
                       'source_directory': str(BASE), 'extractor': 'tests/pytests/data_isapol/oracle/extract_lw_dynamic.py',
                       'extractor_sha256': sha256(Path(__file__).read_bytes()),
                       'generator_commands': None,
                       'generator_commands_limitation': 'Exact historical shell invocation is not established by the permitted local inventory. Local H2O.ornt pins the template recipe; no producer executed or source inspected.',
                       'extraction_command': 'python -P tests/pytests/data_isapol/oracle/extract_lw_dynamic.py --output tests/pytests/data_isapol/orient_local/lw-dynamic-water.json',
                       'geometry_authority': 'existing manifest bohr origins and column frames; O/H2 identity, H1 diag(-1,-1,1); graph explicitly O-H1/O-H2',
                       'l3_component_authority': 'NEW-format real Racah spherical convention declared in existing README; not a raw CARTSPHER field',
                       'limitations': 'Identity hashes do not certify producer correctness. Supplied NONLOCAL only, not native wavefunction response or PFIT. Static fixture unchanged.'},
        'nodes': nodes}
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(f'{output}: {sha256(output.read_bytes())}; 56250 NONLOCAL + 6750 independent L3 literals')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    extract(Path(__file__).resolve().parents[4], args.output)
