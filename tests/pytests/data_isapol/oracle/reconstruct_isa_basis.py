#!/usr/bin/env python3
"""Independent polynomial reconstruction of exported CamCASP basis descriptors.

Development audit, not a production Psi4 basis provider. Coefficients are the
already-normalized effective coefficients exported by CamCASP: do NOT normalize
them again. Real regular harmonics are derived from differentiated Legendre
polynomials, not copied from the reference's component formulas. No SciPy needed.
Supported: Cartesian GAMINT and real spherical DALTON component orders, l <= 4.
"""
import argparse
from fractions import Fraction
from functools import lru_cache
import importlib.util
import hashlib
import json
from math import comb, factorial, gamma, sqrt
from pathlib import Path

import numpy as np


CARTESIAN = (
    ((0, 0, 0),),
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)),
    ((3, 0, 0), (0, 3, 0), (0, 0, 3), (2, 1, 0), (2, 0, 1), (1, 2, 0),
     (0, 2, 1), (1, 0, 2), (0, 1, 2), (1, 1, 1)),
    ((4, 0, 0), (0, 4, 0), (0, 0, 4), (3, 1, 0), (3, 0, 1), (1, 3, 0),
     (0, 3, 1), (1, 0, 3), (0, 1, 3), (2, 2, 0), (2, 0, 2), (0, 2, 2),
     (2, 1, 1), (1, 2, 1), (1, 1, 2)),
)


def double_factorial(n):
    result = 1
    for k in range(n, 0, -2):
        result *= k
    return result


def regular_harmonic(l, m, sine=False):
    """Polynomial for real Racah-normalized r^l C_lm, without Condon-Shortley phase."""
    if not (0 <= m <= l <= 4) or (m == 0 and sine):
        raise ValueError('Unsupported real harmonic')
    coefficients = {}
    # Differentiate the finite Legendre polynomial m times, then multiply by
    # (x+iy)^m and r^(2k). Accumulate rational terms exactly before normalization.
    for k in range((l-m)//2 + 1):
        factor = Fraction((-1)**k * factorial(2*l-2*k),
                          2**l * factorial(k) * factorial(l-k) * factorial(l-m-2*k))
        for a in range(k+1):
            for b in range(k-a+1):
                c = k-a-b
                radial = factorial(k) // (factorial(a)*factorial(b)*factorial(c))
                for j in range(m+1):
                    if j % 2 != int(sine):
                        continue
                    power = (2*a+m-j, 2*b+j, 2*c+l-m-2*k)
                    sign = (-1)**(j//2)
                    coefficients[power] = coefficients.get(power, Fraction(0)) + factor*radial*comb(m, j)*sign
    norm = sqrt(factorial(l-m)/factorial(l+m)) * (sqrt(2.) if m else 1.)
    return tuple((p, float(c)*norm) for p, c in sorted(coefficients.items()) if c)


@lru_cache(maxsize=None)
def angular_polynomials(l, representation):
    if l not in range(5) or representation not in ('C', 'S'):
        raise ValueError('Only Cartesian/spherical ranks 0 through 4 are supported')
    if representation == 'C':
        return tuple(((p, sqrt(double_factorial(2*l-1) /
                              np.prod([double_factorial(2*n-1) for n in p]))),) for p in CARTESIAN[l])
    if l == 1:  # DALTON's anomalous p order is x,y,z, not y,z,x.
        return (regular_harmonic(1, 1), regular_harmonic(1, 1, True), regular_harmonic(1, 0))
    return (tuple(regular_harmonic(l, m, True) for m in range(l, 0, -1))
            + (regular_harmonic(l, 0),)
            + tuple(regular_harmonic(l, m) for m in range(1, l+1)))


def polynomial_values(poly, xyz):
    values = np.zeros(len(xyz))
    for powers, coefficient in poly:
        term = np.full(len(xyz), coefficient)
        for axis, power in enumerate(powers):
            if power:
                term *= xyz[:, axis]**power
        values += term
    return values


def evaluate_basis(basis, points, neighbours=None):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Points must be finite (npoint,3)')
    allowed = None if neighbours is None else set(map(int, neighbours))
    columns = []
    for site, l, first, last in basis['shells']:
        xyz = points - basis['centres'][site-1]
        r2 = np.sum(xyz*xyz, axis=1)
        radial = np.zeros(len(points))
        if allowed is None or site in allowed:
            for p in range(first-1, last):
                radial += basis['contractions'][p, l] * np.exp(-basis['exponents'][p] * r2)
        columns.extend(radial * polynomial_values(poly, xyz)
                       for poly in angular_polynomials(int(l), basis['representation']))
    result = np.column_stack(columns)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite reconstructed basis')
    return result


@lru_cache(maxsize=None)
def polynomial_product_integral(poly_a, poly_b, beta):
    if beta <= 0:
        raise ValueError('Nonintegrable weighted primitive overlap')
    total = 0.
    for pa, ca in poly_a:
        for pb, cb in poly_b:
            powers = tuple(a+b for a, b in zip(pa, pb))
            if any(p % 2 for p in powers):
                continue
            total += ca*cb*np.prod([gamma((p+1)/2) / beta**((p+1)/2) for p in powers])
    return total


def atomic_overlap(basis, w_eps=0., s_block_only=True):
    """Analytic same-centre Gaussian overlap, including explicit contracted shells."""
    if not np.isfinite(w_eps) or w_eps < 0:
        raise ValueError('W-Eps must be finite and nonnegative')
    centres = basis['centres'][basis['shells'][:, 0]-1]
    if not np.all(centres == centres[0]):
        raise ValueError('Analytic audit only supports co-centred atomic functions')
    functions = []
    for _, l, first, last in basis['shells']:
        for poly in angular_polynomials(int(l), basis['representation']):
            functions.append((l, range(first-1, last), poly))
    metric = np.zeros((len(functions), len(functions)))
    for i, (li, pi, poly_i) in enumerate(functions):
        for j in range(i+1):
            lj, pj, poly_j = functions[j]
            shift = w_eps if not s_block_only or (li == 0 and lj == 0) else 0.
            for a in pi:
                for b in pj:
                    beta = float(basis['exponents'][a] + basis['exponents'][b] - shift)
                    metric[i, j] += (basis['contractions'][a, li] * basis['contractions'][b, lj]
                                     * polynomial_product_integral(poly_i, poly_j, beta))
            metric[j, i] = metric[i, j]
    if not np.isfinite(metric).all():
        raise ValueError('Nonfinite reconstructed overlap')
    return metric


def contraction_norms(basis):
    """Independent radial normalization identity for exported effective coefficients."""
    norms = []
    for _, l, first, last in basis['shells']:
        total = 0.
        for a in range(first-1, last):
            for b in range(first-1, last):
                total += (basis['contractions'][a, l] * basis['contractions'][b, l]
                          / (basis['exponents'][a] + basis['exponents'][b])**(l+1.5))
        norms.append(total * double_factorial(2*int(l)-1) * np.pi**1.5 / 2**int(l))
    return np.asarray(norms)


def errors(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Invalid arrays in reconstruction comparison')
    absolute = float(np.max(np.abs(actual-expected)))
    return dict(max_absolute=absolute, max_scaled=absolute/max(1., float(np.max(np.abs(expected)))))


def audit(c, sample_count=257):
    d = c.get('descriptors')
    if d is None:
        raise ValueError('Basis audit requires a v2 descriptor checkpoint')
    if sample_count < 1:
        raise ValueError('Sample count must be positive')
    indices = np.unique(np.linspace(0, len(c['points'])-1, min(sample_count, len(c['points'])), dtype=int))
    points = c['points'][indices]
    basis_values = evaluate_basis(d['atomic_basis'], points)
    density = evaluate_basis(d['density_basis'], points, d['density_neighbours']) @ d['density_coefficients']
    overlap = atomic_overlap(d['atomic_basis'], c['options']['w_eps'], c['options']['s_block_only'])
    counts = [len(angular_polynomials(int(l), d['atomic_basis']['representation']))
              for l in d['atomic_basis']['shells'][:, 1]]
    starts = np.cumsum([0] + counts[:-1])
    mapped_columns = starts[d['shape_map']-1]
    shape_new = c['coefficients'][mapped_columns]
    shape_basis_values = evaluate_basis(d['shape_basis'], points)
    result = dict(schema_version=1, checkpoint_schema_version=c['schema_version'],
                  sample_indices=indices.tolist(), sampled_point_count=len(indices),
                  evidence_class='independent descriptor reconstruction; not native Psi4 basis construction',
                  errors=dict(atomic_samples=errors(basis_values, c['basis_values'][indices]),
                              density_samples=errors(density, c['density'][indices]),
                              atomic_overlap=errors(overlap, c['overlap']),
                              shape_projection=errors(shape_new, d['shape_new_raw']),
                              shape_basis_map=errors(shape_basis_values, basis_values[:, mapped_columns])))
    for label in ('atomic_basis', 'density_basis', 'shape_basis'):
        norms = contraction_norms(d[label])
        result['errors'][label + '_normalization'] = errors(norms, np.ones_like(norms))
    # Only the no-tail branch has a directly reproducible clipped bare shape.
    # Active-tail samples require independent tail metadata/implementation.
    if not c['tail_flags'][0]:
        shape = shape_basis_values @ d['shape_old']
        result['errors']['untailored_shape'] = errors(np.maximum(shape, 0), c['shape'][indices])
    else:
        result['shape_sample_limitation'] = 'Active tail samples not reconstructed from bare Gaussian coefficients'
    return result


def load_reader():
    path = Path(__file__).with_name('replay_isa_checkpoint.py')
    spec = importlib.util.spec_from_file_location('isapol_checkpoint_reader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_fixture(path, c, indices, checkpoint_path):
    """Retain bounded same-source samples, not a full frozen-fit replay fixture."""
    if len(indices) > 513:
        raise ValueError('Portable basis fixture is limited to 513 sampled points')
    selected = {key: c[key] for key in ('schema_version', 'descriptors', 'options', 'tail_flags',
                                       'overlap', 'coefficients', 'atom', 'atom_label')}
    for key in ('points', 'basis_values', 'density', 'shape'):
        selected[key] = c[key][indices]
    payload = dict(schema_version=1, kind='production-basis-descriptor-audit',
                   source_checkpoint_sha256=hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
                   sample_indices=list(map(int, indices)), source_point_count=len(c['points']),
                   units=dict(coordinates='bohr', exponents='bohr^-2', density='electron/bohr^3'),
                   normalization='Exported effective contraction coefficients; no renormalization',
                   component_conventions=dict(C='GAMINT', S='DALTON real regular harmonics, p=x,y,z'),
                   provenance='camcasp_isa_basis_evidence.json',
                   limitation='Selected descriptor reconstruction samples; not a complete frozen-fit replay',
                   checkpoint=selected)
    Path(path).write_text(json.dumps(payload, indent=2, default=lambda v: v.tolist()) + '\n')


def load_fixture(path):
    payload = json.loads(Path(path).read_text())
    if payload.get('schema_version') != 1 or payload.get('kind') != 'production-basis-descriptor-audit':
        raise ValueError('Unsupported basis fixture schema')
    c = payload['checkpoint']
    for key in ('points', 'basis_values', 'density', 'shape', 'overlap', 'coefficients'):
        c[key] = np.asarray(c[key], dtype=float)
    d = c['descriptors']
    for key in ('density_coefficients', 'density_neighbours', 'shape_map', 'shape_old', 'shape_new_raw'):
        d[key] = np.asarray(d[key], dtype=int if key in ('density_neighbours', 'shape_map') else float)
    for key in ('atomic_basis', 'density_basis', 'shape_basis'):
        for field in ('centres', 'exponents', 'contractions', 'shells', 'charges'):
            d[key][field] = np.asarray(d[key][field], dtype=int if field == 'shells' else float)
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--sample-count', type=int, default=257)
    parser.add_argument('--fixture-output', type=Path, help='Optional bounded portable basis audit fixture')
    parser.add_argument('--scaled-tolerance', type=float, default=1e-10)
    args = parser.parse_args()
    if not np.isfinite(args.scaled_tolerance) or args.scaled_tolerance <= 0:
        parser.error('scaled tolerance must be finite and positive')
    c = load_reader().read_checkpoint(args.checkpoint)
    result = audit(c, args.sample_count)
    result['source_checkpoint_sha256'] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    result['reconstruction_tool_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['scaled_tolerance'] = args.scaled_tolerance
    result['passed'] = all(e['max_scaled'] <= args.scaled_tolerance for e in result['errors'].values())
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'sample_indices'}, indent=2))
    if not result['passed']:
        raise SystemExit('Basis reconstruction exceeds tolerance')
    if args.fixture_output is not None:
        write_fixture(args.fixture_output, c, result['sample_indices'], args.checkpoint)


if __name__ == '__main__':
    main()
