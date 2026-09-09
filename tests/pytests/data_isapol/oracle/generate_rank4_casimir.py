#!/usr/bin/env python3
"""DEVELOPMENT ONLY: rank-4 recoupling/dispersion oracle capture from built casimir.

CamCASP numerical artifacts, Alston J. Misquitta and Anthony J. Stone.
Copyright (c) 2019 Anthony Stone; MIT, see RECOUPLED_CAMCASP_LICENSE.

Writes a hash-pinned compressed archive of the LITERAL casimir print tokens for
a synthetic mixed-rank 1..4 deck.  No expected recoupled tensor or dispersion
coefficient is computed here: the deck is generated from an explicit seed and
the reference values are read back verbatim from casimir's own output.  Only
this opt-in tool opens authorized external CamCASP paths; production and pytest
never invoke or import it.  No ORIENT source is consulted.
"""
import argparse
import gzip
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path

SEED = 20260909
NFREQ = 10
OMEGA0 = 0.5
RANKS = (1, 2, 3, 4)
DECK_RECIPE = (
    'labels 10..44s in casimir index order; upper-triangular t<=u entries only; '
    f'value = random.Random({SEED}).uniform(-1,1) written "{{:15.8f}}", column-major '
    'per entry over NFREQ nodes; reciprocal lower triangle supplied by casimir '
    '"if (t/=u) alpha_u(i,u,t)=alpha_u(i,t,u)"; no 00 monopole entry'
)
HEADER = re.compile(r'^([0-9]{1,2}[cs]?)\((\d)(\d)\)( all zero)?\s*$')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def labels():
    out = []
    for l in RANKS:
        out.append(f'{l}0')
        out += [f'{l}{m}{x}' for m in range(1, l+1) for x in ('c', 's')]
    return out


def tindex(label):
    """One-based global component index; matches component_first/component_label."""
    l = int(label[0])
    if len(label) == 2 and label[1] == '0':
        return l*l + 1
    return l*l + 2*int(label[1]) + (1 if label[2] == 's' else 0)


def deck(cgdir):
    rng = random.Random(SEED)
    names, entries = labels(), []
    for i, t in enumerate(names):
        for u in names[i:]:
            values = ''.join(f'{rng.uniform(-1,1):15.8f}' for _ in range(NFREQ))
            entries.append(f'  {t:<5s}{u:<5s}{values}')
    return ('TITLE "rank4 recoupling parity oracle"\nPRINT ALL\n'
            f'FREQUENCIES {OMEGA0} {NFREQ}\nSKIP 0\nCGDIR {cgdir}\n'
            'MOLECULE A\nSITE S1 TYPE T1\n' + '\n'.join(entries) +
            '\nEND\nRECOUPLE ALL\nDISPERSION 12 A A\nFINISH\n')


def parse_deck(text):
    """Literal deck tokens with one-based source line numbers."""
    out = []
    for line, raw in enumerate(text.splitlines(), 1):
        field = raw.split()
        if len(field) == 2 + NFREQ and field[0][0].isdigit():
            out.append([tindex(field[0]), tindex(field[1]), line, field[2:]])
    return out


def parse_alpha(lines):
    """Literal g14.6 recoupled tokens, 15-char (value, imaginary flag) records."""
    out, i = [], 0
    while i < len(lines):
        match = HEADER.match(lines[i])
        if not match:
            i += 1
            continue
        label, la, lap = match.group(1), int(match.group(2)), int(match.group(3))
        if match.group(4):                       # "<label>(<la><lap>) all zero"
            out.append([la, lap, label, tindex(label), i+1, None])
            i += 1
            continue
        tokens, j = [], i+1
        while len(tokens) < NFREQ:
            row = lines[j]
            for k in range(0, len(row)-1, 15):
                field = row[k:k+15]
                if field.strip():
                    tokens.append([field[:14].strip(), field[14:15] == 'i'])
            j += 1
        assert len(tokens) == NFREQ, (label, tokens)
        out.append([la, lap, label, tindex(label), i+1, tokens])
        i = j
    return out


def parse_cn(lines, first):
    """Literal fixed-width Cn fields; trailing below-threshold fields are absent."""
    out = []
    for line, raw in enumerate(lines, first):
        if raw.strip() in ('', 'End'):
            continue
        t, u, J = raw[2:8].strip(), raw[8:13].strip(), int(raw[13:16])
        n = (len(raw.rstrip()) - 16 + 14)//15
        fields = [raw[16+15*k:16+15*(k+1)].strip() for k in range(n)]
        assert all(fields) and 1 <= n <= 7, (raw, fields)
        out.append([tindex(t), tindex(u), J, t, u, line, fields])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camcasp', type=Path, required=True,
                        help='CamCASP source/data tree (realcg + casimir.f90 hashes)')
    parser.add_argument('--casimir', type=Path, required=True,
                        help='built casimir executable; NOT rebuilt or modified here')
    parser.add_argument('--work', type=Path, required=True, help='scratch directory')
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[4])
    args = parser.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    deck_path, out_path = args.work/'rank4_casimir.data', args.work/'rank4_casimir.out'
    cgdir = (args.camcasp/'data/realcg').resolve()
    deck_path.write_text(deck(cgdir))
    with deck_path.open('rb') as stdin, out_path.open('wb') as stdout:
        subprocess.run([str(args.casimir)], stdin=stdin, stdout=stdout, check=True)

    text = out_path.read_text().splitlines()
    head = next(k for k, x in enumerate(text) if x.startswith('Dispersion coefficients'))
    alpha = parse_alpha(text[:head])
    cn = parse_cn(text[head+2:], head+3)
    pairs = sorted({(x[0], x[1]) for x in alpha})
    assert pairs == [(l, p) for l in RANKS for p in RANKS if l+p <= 6], pairs

    lib = args.repo/'psi4/src/psi4/libisapol'
    license_text = (lib/'RECOUPLED_CAMCASP_LICENSE').read_text()
    pinned = json.loads((lib/'realcg_manifest.txt').read_text())
    assert hashlib.sha256(license_text.encode()).hexdigest() == pinned['license_sha256']

    dest = args.repo/'tests/pytests/data_isapol/recoupled_rank4_casimir'
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'LICENSE').write_text(license_text)
    version = (args.camcasp/'VERSION').read_text().split() if (args.camcasp/'VERSION').exists() else []
    payload = json.dumps({
        'track': 'synthetic mixed-rank 1..4 casimir recouple+dispersion',
        'generator': 'oracle/generate_rank4_casimir.py',
        'deck_recipe': DECK_RECIPE,
        'ranks': list(RANKS),
        'frequencies': {'kind': 'CasimirGrid', 'n': NFREQ, 'omega0': OMEGA0},
        'ordered_pairs': [list(p) for p in pairs],
        'boundary': 'casimir.f90 read_cg/recouple "if (j1+j2>6) cycle"; realcg_3_4, '
                    'realcg_4_3 and realcg_4_4 are never read and alpha_c is never '
                    'initialized for (3,4), (4,3), (4,4)',
        'camcasp_version': version,
        'casimir_sha256': sha(args.casimir),
        'casimir_source_sha256': sha(args.camcasp/'src/casimir/casimir.f90'),
        'source_sha256': {deck_path.name: sha(deck_path), out_path.name: sha(out_path)},
        'deck_entries': parse_deck(deck_path.read_text()),
        'alpha': alpha,
        'cn_rows': cn,
    }, separators=(',', ':')).encode()
    blob = gzip.compress(payload, 9, mtime=0)
    (dest/'literal_numeric.json.gz').write_bytes(blob)
    (dest/'manifest.txt').write_text(json.dumps({
        'fixture_sha256': hashlib.sha256(blob).hexdigest(),
        'compressed_bytes': len(blob),
        'uncompressed_bytes': len(payload),
        'source_sha256': {deck_path.name: sha(deck_path), out_path.name: sha(out_path)},
        'casimir_sha256': sha(args.casimir),
        'casimir_source_sha256': sha(args.camcasp/'src/casimir/casimir.f90'),
        'license_sha256': pinned['license_sha256'],
        'license_git_object': pinned['license_git_object'],
        'authors': pinned['authors'],
        'alpha_write_precision': 'g14.6, six significant figures, "i" suffix marks '
                                 'a pure imaginary recoupled component',
        'cn_write_precision': 'fixed-width seven significant figures; J=0..8 only; '
                              'trailing below-threshold fields omitted',
        'counts': {'deck_entries': len(parse_deck(deck_path.read_text())),
                   'alpha_components': len(alpha),
                   'alpha_declared_zero': sum(x[5] is None for x in alpha),
                   'cn_rows': len(cn)},
    }, indent=2)+'\n')
    print(f'{len(alpha)} alpha components over {len(pairs)} ordered pairs, '
          f'{len(cn)} Cn rows, {len(blob)} compressed bytes')


if __name__ == '__main__':
    main()
