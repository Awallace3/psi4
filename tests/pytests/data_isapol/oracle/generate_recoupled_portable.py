#!/usr/bin/env python3
"""DEVELOPMENT ONLY: numerical data extraction, never imported by production/tests.

CamCASP numerical artifacts, Alston J. Misquitta and Anthony J. Stone.
Copyright (c) 2019 Anthony Stone; MIT, see RECOUPLED_CAMCASP_LICENSE.
No executable Fortran extraction and no recoupling/evaluation of expected values.
Only this opt-in tool opens authorized external source paths.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def labels():
    return [s for l in range(9) for s in
            ([f"{l}0"] + [f"{l}{m}{x}" for m in range(1, l+1) for x in ('c', 's')])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camcasp', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True,
                        help='exact work/H2O-isagrid directory, NOT work/H2O')
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[4])
    args = parser.parse_args()
    lib = args.repo/'psi4/src/psi4/libisapol'
    # Parse all seven exact stage-two tables with the existing development parser,
    # compare every integer record and block key/offset; never rewrite those files.
    spec = importlib.util.spec_from_file_location('cn_development_parser', Path(__file__).with_name('parse_cncode.py'))
    cn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cn)
    cn.SRC = str(args.camcasp/'src/casimir')
    expected_terms, expected_blocks = [], []
    for n in range(6,13):
        for key, terms in sorted(cn.parse(n).items()):
            expected_blocks.append((n,*key,len(expected_terms),len(terms)))
            for p,r,la,lap,lb,lbp,ip in terms:
                expected_terms.append((p.numerator,p.denominator,r.numerator,r.denominator,la,lap,lb,lbp,ip))
    inc = (lib/'recoupling_data.inc').read_text()
    actual = [tuple(map(int, row.split(','))) for row in re.findall(r'\{([\d, -]+)\}',inc)]
    assert len(expected_blocks)==393 and len(expected_terms)==4673
    assert actual==expected_terms+expected_blocks
    dest = args.repo/'tests/pytests/data_isapol/recoupled_h2o_isagrid_l3'
    dest.mkdir(exist_ok=True)
    license_text = subprocess.check_output(
        ['git', '-C', str(args.camcasp), 'show', 'b40ae4f^:LICENSE'], text=True)
    assert 'Copyright (c) 2019 Anthony Stone' in license_text
    (lib/'RECOUPLED_CAMCASP_LICENSE').write_text(license_text)
    (dest/'LICENSE').write_text(license_text)
    hashes, records = {}, []
    for l in range(1, 4):
        for p in range(1, 4):
            path = args.camcasp/f'data/realcg/realcg_{l}_{p}'
            hashes[str(path.relative_to(args.camcasp))] = sha(path)
            seen = set()
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                row = list(map(int, line.split()))
                assert len(row) == 7
                k, q, v, num, den, r, s = row
                assert 0 <= k <= 2*l and 0 <= q <= 2*p
                assert (l-p)**2 <= v < (l+p+1)**2 and den > 0 and s > 0 and r != 0
                assert (k,q,v) not in seen
                seen.add((k,q,v))
                records.append((l,p,*row))
    records.sort(key=lambda x: (x[0],x[1],x[4],x[2],x[3]))
    header = ('// Generated numerical records only by generate_recoupled_portable.py.\n'
              '// CamCASP: Alston J. Misquitta and Anthony J. Stone.\n'
              '// Copyright (c) 2019 Anthony Stone; MIT: RECOUPLED_CAMCASP_LICENSE.\n'
              '// Exact source SHA256: realcg_manifest.txt.\n')
    (lib/'realcg_data.inc').write_text(header + ''.join(
        '{' + ','.join(map(str,row)) + '},\n' for row in records))
    for name in ['casimir.f90'] + [f'c{n}code.f90' for n in range(6,13)]:
        path = args.camcasp/'src/casimir'/name
        hashes[str(path.relative_to(args.camcasp))] = sha(path)
    manifest_path = lib/'realcg_manifest.txt'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())['source_sha256'] == hashes, 'Pinned numerical sources changed'
    manifest_path.write_text(json.dumps({
        'source_sha256': hashes, 'records': len(records),
        'license_sha256': hashlib.sha256(license_text.encode()).hexdigest(),
        'license_git_object': 'b40ae4f^:LICENSE',
        'authors': ['Alston J. Misquitta', 'Anthony J. Stone'],
        'ordering': 'la,lap,v,k,q; source line recoverable by unique k,q,v'}, indent=2)+'\n')
    deck = args.reference/'H2O_ref_wt4_L3_casimir.data'
    pot = args.reference/'H2O_ref_wt4_L3_C12.pot'
    assert args.reference.name == 'H2O-isagrid'
    assert sha(deck) == '1d230aebab7a59c809028e28f98e2ecff47b806d771a125f1d223144adc2ea8b'
    assert sha(pot) == '04feceb378fc2e6865224c45e7cba00bdcef294246a8f2289e5c2d792c408393'
    index = {x:i+1 for i,x in enumerate(labels())}
    lines = deck.read_text().splitlines()
    sites, i = [], 0
    while i < len(lines):
        f = lines[i].split(); i += 1
        if not f or f[0].startswith('!'): continue
        if f[0].lower() == 'frequencies':
            assert f[1:] == ['0.5', '10']
        if f[0].lower() == 'skip': assert f[1:] == ['0']
        if f[0].lower() != 'site': continue
        site = {'label': f[1], 'type': f[3], 'entries': []}
        sites.append(site)
        while lines[i].strip().lower() != 'end':
            start = i+1
            f = lines[i].split(); i += 1
            t,u = index[f[0]],index[f[1]]
            assert 2 <= t <= 16 and 2 <= u <= 16
            tokens = f[2:]
            while len(tokens) < 10:
                tokens += lines[i].split(); i += 1
            assert len(tokens) == 10
            [float(x) for x in tokens]  # strict numeric validation; store literal tokens
            site['entries'].append([t,u,start,i,tokens])
        i += 1
    pairs, current = [], None
    for lineno, line in enumerate(pot.read_text().splitlines(),1):
        f = line.split()
        if not f or line.startswith('!'): continue
        if f[0].lower() == 'end': current = None; continue
        if current is None:
            assert f[2] == 'C6'
            current = {'types': f[:2], 'rows': []}; pairs.append(current)
            continue
        t,u,J = index[f[0]],index[f[1]],int(f[2])
        fields = [line[15+15*k:30+15*k].strip() or None for k in range(7)]
        [float(x) for x in fields if x is not None]
        current['rows'].append([t,u,J,lineno,fields])
    rows = [row for p in pairs for row in p['rows']]
    values = [float(v) for row in rows for v in row[4] if v is not None]
    assert len(rows)==6285 and sum(v!=0 for v in values)==10457 and sum(v==0 for v in values)==30791
    assert pairs[0]['rows'][0][4][0] == '26.48177'
    artifact = {'track': 'work/H2O-isagrid', 'omega0': .5, 'n_freq': 10,
        'ranks': [1,2,3], 'sites': sites, 'pairs': pairs,
        'provenance': 'Literal input tokens and pot fixed-width numerical fields. Missing deck entries are exact zero; reciprocal entries copied as specified by deck reader. No geometry/frame in deck: identity frames and zero origins are explicit representation bookkeeping, not molecular geometry.',
        'source_sha256': {deck.name: sha(deck), pot.name: sha(pot)},
        'format': 'deck entry: t,u,first_line,last_line,10 literal tokens; pot row: t,u,J,line,7 literal 15-column fields or null; all line numbers one-based',
        'limits': {'rows':6285,'nonzero_values':10457,'zero_placeholders':30791,
                   'extra_high_J_rows':411,'rtol':1e-6,'placeholder_atol':1e-6}}
    payload = json.dumps(artifact, separators=(',',':')).encode()
    compressed = gzip.compress(payload, mtime=0)
    name = 'literal_numeric.json.gz'
    (dest/name).write_bytes(compressed)
    (dest/'manifest.txt').write_text(json.dumps({
        'track':artifact['track'], 'source_sha256':artifact['source_sha256'],
        'fixture_sha256':hashlib.sha256(compressed).hexdigest(),
        'uncompressed_bytes':len(payload), 'compressed_bytes':len(compressed),
        'limits':artifact['limits'], 'license':'LICENSE',
        'not_validated': 'J9,10 numeric parity; native SCF/PFIT/GRAC protocol'},indent=2)+'\n')
    print(f'{len(records)} CG records; {len(rows)} literal pot rows; {len(compressed)} compressed bytes')

if __name__ == '__main__':
    main()
