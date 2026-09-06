#!/usr/bin/env python3
"""Strict whole-sweep state reader and explicit-input C++ controller replay.

Requires the sidecar and three v2 atom streams emitted by capture_isa_sweep.py.
This is one matched exported-input transition, not native or end-to-end parity.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('atom_replay', Path(__file__).with_name('replay_isa_checkpoint.py'))
atom_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(atom_tool)


def read_phase(r, label, n):
    r.expect(label)
    active = r.numbers(4)
    if min(active[:2]) < 0 or any(x not in (0, 1) for x in active[2:]):
        raise ValueError('Invalid active controls')
    atoms = []
    for index in range(n):
        r.expect('ATOM')
        atom, z = r.numbers(2, integer=True)
        if atom != index+1 or z < 0:
            raise ValueError('Invalid sweep atom identity/order')
        name, centre = r.line(), r.numbers(3)
        if not name:
            raise ValueError('Empty sweep atom name')
        vectors = {key: r.vector(key) for key in ('D', 'D0', 'W', 'W0')}
        if len(vectors['D']) != len(vectors['D0']) or len(vectors['W']) != len(vectors['W0']):
            raise ValueError('Sweep state vector dimensions disagree')
        r.expect('CHARGES')
        words = r.line().split()
        if len(words) != 3:
            raise ValueError('Expected three charge fields')
        charge_values = np.asarray([float(w.replace('D', 'E')) for w in words])
        if not np.isfinite(charge_values[:2]).all():
            raise ValueError('Nonfinite saved shape charge')
        # Legacy rescaled ISAcharge is 0/0 before the first fitted sweep. It is
        # diagnostic only, NOT a controller input; preserve its token explicitly.
        charges = charge_values[:2]
        legacy_isacharge = float(charge_values[2]) if np.isfinite(charge_values[2]) else None
        r.expect('FLAGS')
        flags = r.numbers(3, integer=True)
        if any(f not in (0, 1) for f in flags[:2]) or flags[2] not in (-1, 0, 1):
            raise ValueError('Invalid sweep flags or unsupported tail function')
        r.expect('TAIL')
        tail = r.numbers(4)
        if flags[1] and (flags[2] != 1 or min(tail[:2]) < 0 or tail[3] <= 0):
            raise ValueError('Invalid defined tail')
        neighbours = r.vector('SHAPE_NEIGHBOURS', integer=True)
        if (len(set(neighbours)) != len(neighbours) or np.any(neighbours < 1)
                or np.any(neighbours > n) or atom not in neighbours):
            raise ValueError('Invalid shape neighbours')
        atoms.append(dict(atom=int(atom), z=int(z), name=name, centre=centre, vectors=vectors,
                          charges=charges, legacy_isacharge=legacy_isacharge,
                          legacy_isacharge_token=words[2], flags=flags, tail=tail, neighbours=neighbours))
    return dict(active=active, atoms=atoms)


def read_state(path):
    with Path(path).open() as stream:
        r = atom_tool.Reader(stream)
        r.expect('ISAPOL_SWEEP_STATE 1')
        iteration, n = map(int, r.numbers(2, integer=True))
        if iteration < 1 or n != 3:
            raise ValueError('Expected one ordinary three-site sweep')
        r.expect('CONFIG_FLOATS')
        floats = r.numbers(9)
        if min(floats) < 0 or floats[4] <= 0 or floats[8] > 1:
            raise ValueError('Invalid controller settings')
        r.expect('CONFIG_INTS')
        ints = r.numbers(8, integer=True)
        if any(ints[i] not in (0, 1) for i in (0, 1, 5, 6, 7)) or ints[2] < iteration or min(ints[3:5]) < 0:
            raise ValueError('Invalid controller flags/limits')
        pre = read_phase(r, 'PRE', n)
        r.expect('DELTAS')
        deltas = r.numbers(n)
        r.expect('MAX_DELTA')
        max_delta, = r.numbers(1)
        if np.any(deltas < 0) or max_delta < 0:
            raise ValueError('Invalid convergence deltas')
        r.expect('CALLS')
        calls = r.numbers(n, integer=True)
        if np.any(calls < 1) or np.any(np.diff(calls) != 1):
            raise ValueError('Invalid atom call associations')
        post = read_phase(r, 'POST', n)
        r.expect('END_SWEEP')
        if stream.read().strip():
            raise ValueError('Trailing sweep data')
    for a, b in zip(pre['atoms'], post['atoms']):
        if (a['name'] != b['name'] or a['z'] != b['z'] or not np.array_equal(a['centre'], b['centre'])
                or not np.array_equal(a['neighbours'], b['neighbours'])):
            raise ValueError('Sweep site identity/neighbours changed')
        if (not np.array_equal(b['vectors']['D'], b['vectors']['D0'])
                or not np.array_equal(b['vectors']['W'], b['vectors']['W0'])):
            raise ValueError('Post state is not committed')
    return dict(iteration=iteration, floats=floats, ints=ints, pre=pre, post=post,
                deltas=deltas, max_delta=float(max_delta), calls=calls)


def prepare_replay(directory):
    """Build immutable providers and an explicit entry cursor, without a sweep."""
    import psi4
    core = psi4.core
    directory = Path(directory)
    sidecar = directory/'isapol-sweep-state.dat'
    s = read_state(sidecar)
    checkpoints = [atom_tool.read_checkpoint(directory/f'isapol-atom-{a+1}.dat') for a in range(3)]
    atomic, shape, maps, grids = [], [], [], []
    for a, c in enumerate(checkpoints):
        pre, post = s['pre']['atoms'][a], s['post']['atoms'][a]
        if c['schema_version'] != 2 or c['call'] != s['calls'][a] or c['atom'] != a+1:
            raise ValueError('Atom stream identity mismatch')
        d = c['descriptors']
        for x, y in [(c['centre'], pre['centre']), (c['previous'], pre['vectors']['D']),
                     (d['shape_old'], pre['vectors']['W0']), (c['coefficients'], post['vectors']['D'])]:
            if not np.array_equal(x, y):
                raise ValueError('Atom stream disagrees with sweep state')
        if (c['options']['w_eps'] != s['pre']['active'][0]
                or c['options']['positive_lambda'] != s['pre']['active'][1]
                or c['tail_flags'][0] != s['pre']['active'][2]):
            raise ValueError('Atom active controls disagree with sidecar')
        atomic.append(atom_tool.explicit_basis(d['atomic_basis'], core.IsaBasisRole.AtomAux))
        shape.append(atom_tool.explicit_basis(d['shape_basis'], core.IsaBasisRole.Shape))
        maps.append((d['shape_map']-1).tolist())
        grid = core.IsaNoTailGrid()
        grid.points, grid.weights = c['points'].tolist(), c['weights'].tolist()
        grid.density_sites = [int(x)-1 for x in d['density_neighbours'] if x > 0]
        grid.shape_sites = (pre['neighbours']-1).tolist()
        grids.append(grid)
    first = checkpoints[0]['descriptors']
    for c in checkpoints[1:]:
        d = c['descriptors']
        if not np.array_equal(d['density_coefficients'], first['density_coefficients']):
            raise ValueError('Density coefficients differ between atom streams')
        for key in ('centres', 'shells', 'exponents', 'contractions'):
            if not np.array_equal(d['density_basis'][key], first['density_basis'][key]):
                raise ValueError('Molecular AUX descriptors differ between streams')
        if d['density_basis']['representation'] != first['density_basis']['representation']:
            raise ValueError('Molecular AUX convention mismatch')
    density = core.IsaFixedDensity(atom_tool.explicit_basis(first['density_basis'], core.IsaBasisRole.MolecularAux),
                                   first['density_coefficients'].tolist())
    f, i = s['floats'], s['ints']
    options = core.IsaAControllerOptions()
    fit = core.IsaAFitOptions()
    fit.w_eps, fit.positive_lambda, fit.damping, fit.positive_max_alpha = map(float, f[:4])
    fit.s_block_only, fit.positive_auto = bool(i[0]), bool(i[1])
    fit.density_cutoff = checkpoints[0]['options']['density_cutoff']
    options.fit = fit
    options.convergence, options.w_eps_activation, options.positive_activation, options.tail_activation, options.mixing = map(float, f[4:])
    options.max_iterations, options.mixing_skip, options.tail_iteration_limit = map(int, i[2:5])
    options.fix_tails = bool(i[5])
    options.tail_cutoffs = [float(a['tail'][0]) for a in s['post']['atoms']]
    options.tail_allowed = [a['z'] > 0 or bool(i[6]) for a in s['pre']['atoms']]
    options.convergence_included = [a['z'] > 0 or not bool(i[7]) for a in s['pre']['atoms']]
    controller = core.IsaAController(atomic, shape, maps, density, grids, options)
    coefficients = core.IsaSweepState()
    coefficients.atomic_coefficients = [a['vectors']['D'].tolist() for a in s['pre']['atoms']]
    coefficients.shape_coefficients = [a['vectors']['W0'].tolist() for a in s['pre']['atoms']]
    old = controller.initialize(coefficients)
    old.iteration = s['iteration']-1
    old.active_w_eps, old.active_positive_lambda = map(float, s['pre']['active'][:2])
    old.apply_tails, old.converged = map(bool, s['pre']['active'][2:])
    old.saved_shape_charges = [float(a['charges'][1]) for a in s['pre']['atoms']]
    tails = []
    for a in s['pre']['atoms']:
        tail = core.IsaExponentialTail()
        tail.defined = bool(a['flags'][1])
        tail.cutoff, tail.amplitude, tail.exponent = map(float, a['tail'][[0, 2, 3]])
        tails.append(tail)
    old.tails = tails
    return controller, old, checkpoints, s


def replay(directory):
    import psi4
    core = psi4.core
    directory = Path(directory)
    sidecar = directory/'isapol-sweep-state.dat'
    controller, old, checkpoints, s = prepare_replay(directory)
    result = controller.step(old)
    errors = {}
    def compare(name, actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError(f'Invalid comparison for {name}')
        error = float(np.max(np.abs(actual-expected)))
        errors[name] = dict(max_absolute=error, max_scaled=error/max(1., float(np.max(np.abs(expected)))))
    compare('deltas', result.deltas, s['deltas'])
    compare('max_delta', result.next.max_delta, s['max_delta'])
    flags_match = result.next.converged == bool(s['post']['active'][3])
    flags_match &= result.atom_converged == [bool(a['flags'][0]) for a in s['post']['atoms']]
    compare('next_controls', [result.next.active_w_eps, result.next.active_positive_lambda, int(result.next.apply_tails)], s['post']['active'][:3])
    for a, c in enumerate(checkpoints):
        post = s['post']['atoms'][a]
        compare(f'atom{a+1}_D', result.next.coefficients.atomic_coefficients[a], post['vectors']['D'])
        compare(f'atom{a+1}_W', result.next.coefficients.shape_coefficients[a], post['vectors']['W0'])
        compare(f'atom{a+1}_charge', result.next.saved_shape_charges[a], post['charges'][1])
        compare(f'atom{a+1}_rhs', result.raw_sweep.fits[a].rhs.np[:, 0], c['rhs'])
        compare(f'atom{a+1}_population', result.raw_sweep.fits[a].population, c['population'])
        flags_match &= result.next.tails[a].defined == bool(post['flags'][1])
        if post['flags'][1] and result.next.tails[a].defined:
            compare(f'atom{a+1}_tail', [result.next.tails[a].amplitude, result.next.tails[a].exponent], post['tail'][2:])
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [sidecar]+[directory/f'isapol-atom-{a+1}.dat' for a in range(3)]}
    return dict(schema_version=1, evidence_class='one exported-input controller transition; not native/end-to-end parity',
                iteration=s['iteration'], points=[len(c['points']) for c in checkpoints], errors=errors,
                flags_match=bool(flags_match), residuals=[r.relative_residual for r in result.raw_sweep.fits],
                source_sha256=hashes, psi4_extension=core.__file__,
                limitations='Reference tail fit uses a stale saved-A sign gate; C++ is deterministic. Undefined reference IP is excluded.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--scaled-tolerance', type=float, default=1e-9)
    args = parser.parse_args()
    if not np.isfinite(args.scaled_tolerance) or args.scaled_tolerance <= 0:
        parser.error('scaled tolerance must be finite and positive')
    report = replay(args.directory)
    report['scaled_tolerance'] = args.scaled_tolerance
    report['passed'] = (report['flags_match'] and all(e['max_scaled'] <= args.scaled_tolerance for e in report['errors'].values())
                        and all(np.isfinite(r) and r <= args.scaled_tolerance for r in report['residuals']))
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit('Controller transition replay exceeds tolerance')
