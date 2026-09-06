#!/usr/bin/env python3
"""Prepare an explicitly ADAPTED archived-water Drho-C/LU production run.

This is not the modern ISA-Pol preset: retain archived OBS, Cartesian molecular
AUX, spherical aVTZ AtomAux and 100/200 grid. Change DF/ISA settings explicitly,
remove response/GDMA work, and retain a fully expanded input plus source hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(archive, camcasp, destination):
    archive, camcasp, destination = map(lambda p: Path(p).resolve(), (archive, camcasp, destination))
    if destination.exists() or archive in destination.parents or camcasp in destination.parents:
        raise ValueError('Run directory must be new and outside reference trees')
    original = (archive / 'H2O.cks').read_text()
    # Fail closed if the archived protocol changes rather than silently adapting it.
    for anchor in ('DF = Doo-C', 'Solver BVLS', 'Angular 200', 'Radial  100', 'ISA-Algorithm A'):
        if original.count(anchor) != 1:
            raise ValueError(f'Unexpected archived input anchor: {anchor}')
    expected_geometry = {'O': [0., 0., 0.], 'H1': [-1.45365196, 0., -1.12168732],
                         'H2': [1.45365196, 0., -1.12168732]}
    sites = re.findall(r'^\s*(O|H1|H2)\s+\S+\s+(\S+)\s+(\S+)\s+(\S+)\s+TYPE\s+\S+\s*$', original, re.M)
    if len(sites) != 6 or [s[0] for s in sites] != ['O', 'H1', 'H2'] * 2:
        raise ValueError('Unexpected archived atom order')
    for label, *xyz in sites:
        if [float(x) for x in xyz] != expected_geometry[label]:
            raise ValueError('Unexpected archived geometry')
    prefix = original[:original.index('SET Lattice')]
    # This source revision predates the archive's SET Num-Int-Pars parser.
    # Omit that block explicitly; record the use of source-default integration controls.
    isa = original[original.index('Begin ISA'):original.index('BEGIN Polarizability', original.index('Begin ISA'))]
    isa = isa.replace('DF = Doo-C', 'DF = Drho-C').replace('Solver BVLS', 'Solver LU')
    # Legacy parser enables tails by default but does not accept modern FIX = ON.
    isa = re.sub(r'^\s*FIX\s*=\s*ON\s*$', '', isa, flags=re.M)
    density_fit = '''
SET DF-INTEGRALS
  DF-TYPE-MONOMER NN
END
BEGIN DF
  Molecule H2O
  Type NN
  Eta = 0.0
  Lambda = 1000.0
  Gamma = 0.0
  Solver LU
END
BEGIN DF
  Molecule H2O
  Type NN
  Eta = 0.0
  Lambda = 0.0
  Gamma = 0.0
  Solver LU
END
Edit
  NEIGHBOURS TYPE = OVR EPS = 0.0001 PRINT
End
'''
    adapted = prefix + density_fit + isa + '\nFINISH\n'
    sources = {str(archive / 'H2O.cks'): sha(archive / 'H2O.cks')}

    def expand(text, base, stack=()):
        out = []
        for line in text.splitlines():
            match = re.fullmatch(r'\s*#include(-camcasp)?\s+(\S+)\s*', line, re.I)
            if not match:
                if '#include' in line.lower():
                    raise ValueError(f'Unsupported include directive: {line}')
                out.append(line)
                continue
            path = ((camcasp if match[1] else base) / match[2]).resolve()
            if path in stack:
                raise ValueError(f'Include cycle: {path}')
            sources[str(path)] = sha(path)
            out.append(expand(path.read_text(), path.parent, stack + (path,)))
        return '\n'.join(out) + '\n'

    expanded = expand(adapted, archive)
    destination.mkdir(parents=True)
    for name in ('H2O-A-asc.movecs', 'H2O-A.basis'):
        shutil.copy2(archive / name, destination / name)
        sources[str(archive / name)] = sha(archive / name)
    (destination / 'H2O.cks').write_text(adapted)
    (destination / 'H2O-expanded.cks').write_text(expanded)
    metadata = dict(schema_version=1, protocol='adapted-archive-Drho-C-LU',
                    units=dict(geometry='bohr', density='electron/bohr^3', population='electron'),
                    atoms=['O', 'H1', 'H2'], geometry_bohr=[[0., 0., 0.], [-1.45365196, 0., -1.12168732],
                                                        [1.45365196, 0., -1.12168732]],
                    source_sha256=sources, expanded_input_sha256=sha(destination / 'H2O-expanded.cks'),
                    settings={'density': 'Drho-C', 'solver': 'LU', 'algorithm': 'A',
                              'molecular_aux': 'archived Cartesian aVTZ', 'atom_aux': 'spherical aVTZ + ISA set2',
                              'radial': 100, 'angular_requested': 200, 'df_lambda_sequence': [1000., 0.],
                              'num_int_pars': 'source defaults; archive SET Num-Int-Pars unsupported and omitted',
                              'tail_fix': 'source default ON; unsupported explicit FIX = ON omitted'},
                    limitations=['Not the modern ISA-Pol preset; archived orbital provenance is inherited, not revalidated.',
                                 'Fixed exported orbitals, no independent SCF or native Psi4 DF density.',
                                 'Capture records effective sampled basis normalization and shell coefficients; no native basis equivalence claim.'])
    (destination / 'run-provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--camcasp', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.camcasp, args.destination), indent=2))
