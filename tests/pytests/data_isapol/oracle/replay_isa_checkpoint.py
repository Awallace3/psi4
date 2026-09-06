#!/usr/bin/env python3
"""Strict reader and staged-Psi4 replay for capture_isa_checkpoint.py output.

The text stream is an internal versioned diagnostic format, not a public restart
API. An incomplete capture, contracted basis or nonfinite data is an error. The
report establishes same-production-sample arithmetic only, not native basis/DF
construction or converged properties. Source/run provenance must accompany it.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


class Reader:
    def __init__(self, stream):
        self.stream = stream
        self.line_number = 0

    def line(self):
        self.line_number += 1
        value = self.stream.readline()
        if not value:
            raise ValueError(f'Truncated checkpoint at line {self.line_number}')
        return value.strip()

    def expect(self, label):
        actual = self.line()
        if actual != label:
            raise ValueError(f'Expected {label!r}, got {actual!r} at line {self.line_number}')

    def numbers(self, count, integer=False):
        words = self.line().split()
        if len(words) != count:
            raise ValueError(f'Expected {count} numbers at line {self.line_number}, got {len(words)}')
        values = np.asarray([int(w) if integer else float(w.replace('D', 'E')) for w in words])
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite checkpoint values at line {self.line_number}')
        return values

    def vector(self, label, count=None, integer=False):
        self.expect(label)
        n, = self.numbers(1, integer=True)
        if n < 1 or (count is not None and n != count):
            raise ValueError(f'Incorrect {label} size')
        return self.numbers(int(n), integer=integer)

    def basis(self, label):
        self.expect(label)
        np_, nshell, nsite, nf, max_l = map(int, self.numbers(5, integer=True))
        if min(np_, nshell, nsite, nf, max_l) < 1:
            raise ValueError(f'Invalid {label} counts')
        representation = self.line()
        if representation not in ('C', 'S'):
            raise ValueError(f'Invalid {label} representation')
        labels, sites = [], []
        for _ in range(nsite):
            labels.append(self.line())
            sites.append(self.numbers(4))
        if not all(labels):
            raise ValueError(f'Empty {label} site label')
        primitives = np.asarray([self.numbers(1 + max_l) for _ in range(np_)])
        shells = np.asarray([self.numbers(4, integer=True) for _ in range(nshell)])
        if np.any(primitives[:, 0] <= 0):
            raise ValueError(f'Nonpositive {label} exponent')
        for site, l, first, last in shells:
            if not (1 <= site <= nsite and 0 <= l < max_l and 1 <= first <= last <= np_):
                raise ValueError(f'Invalid {label} shell index')
        components = [2*l+1 if representation == 'S' else (l+1)*(l+2)//2 for l in shells[:, 1]]
        if sum(components) != nf:
            raise ValueError(f'Incorrect {label} component count')
        sites = np.asarray(sites)
        return dict(representation=representation, labels=labels, charges=sites[:, 0],
                    centres=sites[:, 1:], exponents=primitives[:, 0],
                    contractions=primitives[:, 1:], shells=shells, nfunction=nf)

    def matrix(self, label, rows, cols):
        self.expect(label)
        if tuple(self.numbers(2, integer=True)) != (rows, cols):
            raise ValueError(f'Incorrect {label} dimensions')
        return np.asarray([self.numbers(cols) for _ in range(rows)])


def read_checkpoint(path):
    with Path(path).open() as stream:
        r = Reader(stream)
        header = r.line()
        if header not in ('ISAPOL_CHECKPOINT 1', 'ISAPOL_CHECKPOINT 2'):
            raise ValueError(f'Expected supported ISAPOL_CHECKPOINT header, got {header!r}')
        version = int(header.split()[1])
        call, atom, nf, nshells = map(int, r.numbers(4, integer=True))
        if min(call, atom, nf, nshells) < 1 or nshells > nf:
            raise ValueError('Invalid checkpoint counts')
        label, representation = r.line(), r.line()
        if not label or representation not in ('S', 'C'):
            raise ValueError('Invalid atom label or harmonic representation')
        centre = r.numbers(3)
        eps, damping, ridge, max_alpha = r.numbers(4)
        s_only, auto = r.numbers(2, integer=True)
        if s_only not in (0, 1) or auto not in (0, 1):
            raise ValueError('Invalid option flags')
        if min(eps, damping, ridge, max_alpha) < 0:
            raise ValueError('Negative fitting option')
        previous = r.numbers(nf)
        angular, exponents, shells = [], [], []
        for _ in range(nshells):
            l, components, first, last = map(int, r.numbers(4, integer=True))
            expected = 2*l+1 if representation == 'S' else (l+1)*(l+2)//2
            if l < 0 or components != expected or first < 1 or first != last:
                raise ValueError('Invalid or contracted atomic shell')
            exponent, coefficient = r.numbers(2)
            if exponent <= 0:
                raise ValueError('Nonpositive primitive exponent')
            angular.extend([l] * components)
            exponents.extend([exponent] * components)
            shells.append(dict(l=l, components=components, first=first, last=last,
                               exponent=float(exponent), coefficient=float(coefficient)))
        if len(angular) != nf:
            raise ValueError('Shell component count does not match nfunction')
        overlap = r.matrix('OVERLAP', nf, nf)
        metric = r.matrix('METRIC', nf, nf)
        r.expect('DENSITY')
        density_name = r.line()
        cutoff, = r.numbers(1)
        tails = r.numbers(2, integer=True)
        if cutoff < 0 or not all(t in (0, 1) for t in tails):
            raise ValueError('Invalid cutoff or tail flags')
        descriptors = None
        if version == 2:
            r.expect('DESCRIPTORS')
            atomic_basis = r.basis('ATOMIC_BASIS')
            density_basis = r.basis('DENSITY_BASIS')
            density_coefficients = r.vector('DENSITY_COEFFICIENTS', density_basis['nfunction'])
            neighbours = r.vector('DENSITY_NEIGHBOURS', integer=True)
            # The live reference exports an allocated array (typically length 400)
            # with a positive active prefix and zero padding. Its ANY(site == list)
            # comparison ignores those zeros; retain the raw array for provenance.
            active_count = np.count_nonzero(neighbours)
            if (active_count == 0 or np.any(neighbours < 0)
                    or np.any(neighbours > len(density_basis['labels']))
                    or np.any(neighbours[:active_count] == 0)):
                raise ValueError('Invalid density neighbour index or padding')
            shape_basis = r.basis('SHAPE_BASIS')
            shape_map_storage = r.vector('SHAPE_MAP', integer=True)
            nshape = shape_basis['nfunction']
            if len(shape_map_storage) < nshape or np.any(shape_map_storage[nshape:] != 0):
                raise ValueError('Incorrect SHAPE_MAP size or padding')
            shape_map = shape_map_storage[:nshape].copy()
            shape_old = r.vector('SHAPE_OLD', nshape)
            if (atomic_basis['nfunction'] != nf or len(atomic_basis['shells']) != nshells
                    or atomic_basis['representation'] != representation):
                raise ValueError('Atomic descriptor dimension/convention mismatch')
            for legacy, (site, l, first, last) in zip(shells, atomic_basis['shells']):
                if (l != legacy['l'] or first != legacy['first'] or last != legacy['last']
                        or atomic_basis['exponents'][first-1] != legacy['exponent']
                        or atomic_basis['contractions'][first-1, l] != legacy['coefficient']
                        or not np.array_equal(atomic_basis['centres'][site-1], centre)):
                    raise ValueError('Atomic descriptor disagrees with frozen-fit metadata')
            if np.any(shape_basis['shells'][:, 1] != 0):
                raise ValueError('Shape basis must be s-only')
            if (len(np.unique(shape_map)) != len(shape_map) or np.any(shape_map < 1)
                    or np.any(shape_map > len(atomic_basis['shells']))):
                raise ValueError('Invalid shape shell map')
            if np.any(atomic_basis['shells'][shape_map-1, 1] != 0):
                raise ValueError('Shape map targets non-s shells')
            descriptors = dict(atomic_basis=atomic_basis, density_basis=density_basis,
                               density_coefficients=density_coefficients, density_neighbours=neighbours,
                               shape_basis=shape_basis, shape_map=shape_map,
                               shape_map_storage=shape_map_storage, shape_old=shape_old)
        batches, rows = [], []
        while True:
            section = r.line()
            if section == 'RHS':
                break
            if section != 'BATCH':
                raise ValueError(f'Expected BATCH or RHS, got {section!r}')
            site, start, count = map(int, r.numbers(3, integer=True))
            if min(site, start, count) < 1:
                raise ValueError('Invalid batch bounds')
            batches.append(dict(site=site, start=start, count=count))
            rows.extend(r.numbers(7 + nf) for _ in range(count))
        if not rows:
            raise ValueError('Checkpoint contains no samples')
        if tuple(r.numbers(2, integer=True)) != (nf, 1):
            raise ValueError('Incorrect RHS dimensions')
        rhs = np.asarray([r.numbers(1)[0] for _ in range(nf)])
        r.expect('POPULATION')
        population, = r.numbers(1)
        coefficients = r.matrix('COEFFICIENTS', nf, 1)[:, 0]
        if version == 2:
            descriptors['shape_new_raw'] = r.vector('SHAPE_NEW_RAW', len(descriptors['shape_old']))
        r.expect('END')
        if stream.read().strip():
            raise ValueError('Unexpected trailing checkpoint data')
    samples = np.asarray(rows)
    return dict(schema_version=version, descriptors=descriptors,
                call=call, atom=atom, atom_label=label, representation=representation,
                centre=centre, shells=shells, density_name=density_name, batches=batches,
                tail_flags=tails.tolist(), previous=previous, angular_momenta=np.asarray(angular),
                exponents=np.asarray(exponents), overlap=overlap, metric=metric,
                points=samples[:, :3], weights=samples[:, 3], density=samples[:, 4],
                shape=samples[:, 5], shape_sum=samples[:, 6], basis_values=samples[:, 7:],
                rhs=rhs, coefficients=coefficients, population=float(population),
                options=dict(w_eps=float(eps), damping=float(damping), positive_lambda=float(ridge),
                             positive_max_alpha=float(max_alpha), s_block_only=bool(s_only),
                             positive_auto=bool(auto), density_cutoff=float(cutoff)))


def explicit_basis(descriptor, role):
    """Index/ownership adapter only; numerical work stays in the C++ provider."""
    import psi4
    shells = []
    for site, l, first, last in descriptor['shells']:
        s = psi4.core.IsaGaussianShell()
        s.centre, s.l = int(site)-1, int(l)
        s.exponents = descriptor['exponents'][first-1:last].tolist()
        s.coefficients = descriptor['contractions'][first-1:last, l].tolist()
        shells.append(s)
    rep = (psi4.core.IsaBasisRepresentation.Cartesian if descriptor['representation'] == 'C'
           else psi4.core.IsaBasisRepresentation.Spherical)
    basis = psi4.core.IsaExplicitBasis(role, rep, descriptor['centres'].tolist(), shells)
    if basis.nfunction != descriptor['nfunction']:
        raise ValueError('Descriptor function count mismatch')
    return basis


def provider_assembly(c, options):
    """Reconstruct all points/metric, retaining captured shape and quadrature policy."""
    import psi4
    d = c.get('descriptors')
    if d is None:
        raise ValueError('Provider reconstruction requires v2 descriptors')
    atomic = explicit_basis(d['atomic_basis'], psi4.core.IsaBasisRole.AtomAux)
    molecular = explicit_basis(d['density_basis'], psi4.core.IsaBasisRole.MolecularAux)
    density = psi4.core.IsaFixedDensity(molecular, d['density_coefficients'].tolist())
    samples = psi4.core.IsaAFitSamples()
    for key in ('points', 'weights', 'shape', 'shape_sum', 'previous'):
        setattr(samples, key, c[key].tolist())
    # Reader has already checked positive-prefix/zero-padding format. The C++ API
    # rejects duplicate active sites rather than silently changing the protocol.
    samples.density_sites = [int(s)-1 for s in d['density_neighbours'] if s > 0]
    data = psi4.core.IsaAFitProvider(atomic, density).assemble(samples, options)
    shape = explicit_basis(d['shape_basis'], psi4.core.IsaBasisRole.Shape)
    mapping = psi4.core.IsaShapeMap(atomic, shape, (d['shape_map']-1).tolist())
    return data, mapping


def replay(checkpoint, reconstruct_providers=False):
    import psi4  # optional until replay; parsing/validation needs only NumPy
    c = checkpoint
    data, options = psi4.core.IsaAFitData(), psi4.core.IsaAFitOptions()
    for key in ('weights', 'density', 'shape', 'shape_sum', 'previous', 'angular_momenta', 'exponents'):
        setattr(data, key, c[key].tolist())
    data.radius_squared = np.sum((c['points'] - c['centre'])**2, axis=1).tolist()
    for key in ('basis_values', 'overlap'):
        setattr(data, key, psi4.core.Matrix.from_array(c[key]))
    for key, value in c['options'].items():
        setattr(options, key, value)
    errors = {}
    mapping = None
    def compare(key, actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError(f'Invalid replay comparison for {key}')
        error = float(np.max(np.abs(actual - expected)))
        errors[key] = dict(max_absolute=error, max_scaled=error / max(1., float(np.max(np.abs(expected)))))
    if reconstruct_providers:
        data, mapping = provider_assembly(c, options)
        for key in ('density', 'exponents', 'angular_momenta'):
            compare('provider_' + key, getattr(data, key), c[key])
        for key in ('basis_values', 'overlap'):
            compare('provider_' + key, getattr(data, key).np, c[key])
        compare('provider_radius_squared', data.radius_squared, np.sum((c['points']-c['centre'])**2, axis=1))
    result = psi4.core.isa_a_fit_step(data, options)
    for key, actual in [('metric', result.metric.np), ('rhs', result.rhs.np[:, 0]),
                        ('coefficients', result.coefficients.np[:, 0]), ('population', result.population)]:
        compare(key, actual, c[key])
    if mapping is not None:
        compare('provider_raw_shape', mapping.project(result.coefficients.np[:, 0].tolist()),
                c['descriptors']['shape_new_raw'])
    return dict(schema_version=1, evidence_class=('full exported-input provider reconstruction with supplied shape samples'
                                                 if reconstruct_providers else 'same-production-sample arithmetic'),
                limitations=('Not native basis/DF generation; captured shapes retain screening and active tails; no controller'
                             if reconstruct_providers else 'Supplied density/basis/shape/metric samples; no native generation'),
                psi4_extension=psi4.core.__file__, call=c['call'], atom=c['atom'],
                density_name=c['density_name'], npoint=len(c['weights']), nfunction=len(c['previous']),
                options=c['options'], errors=errors, relative_residual=result.relative_residual,
                excluded_points=result.excluded_points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--scaled-tolerance', type=float, default=1e-9)
    parser.add_argument('--reconstruct-providers', action='store_true',
                        help='v2 only: C++ reconstructs all density/basis/metric inputs; shapes remain supplied')
    args = parser.parse_args()
    if not np.isfinite(args.scaled_tolerance) or args.scaled_tolerance <= 0:
        parser.error('scaled tolerance must be finite and positive')
    report = replay(read_checkpoint(args.checkpoint), args.reconstruct_providers)
    report['checkpoint_sha256'] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    report['scaled_tolerance'] = args.scaled_tolerance
    report['passed'] = (all(e['max_scaled'] <= args.scaled_tolerance for e in report['errors'].values())
                        and np.isfinite(report['relative_residual'])
                        and report['relative_residual'] <= args.scaled_tolerance)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit('Production checkpoint replay exceeds tolerance; see report')


if __name__ == '__main__':
    main()
