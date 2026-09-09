#!/usr/bin/env python3
"""Native fit -> shared response error propagation with SUPPLIED operators only.

No SCF, kernel construction, quadrature, partition or molecular property acceptance.
Imports do not open reference data or import Psi4. Direct solves below are independent
unsymmetrized diagnostic controls, never a fallback response implementation.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

REPRESENTATION = 'fitted_density_coefficients'
ORDER = 'p=a+noccupied*r; occupied-fast'
TOLERANCE = 1.e-9
RCOND = 1.e-13
EXPERIMENTS = ('frozen_hessian_transition_legs', 'transition_legs_and_kernel_projection')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(name):
    spec = importlib.util.spec_from_file_location('native_legs_' + name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def numeric_identity(value):
    """Common numeric layout: real little-endian binary64, C order, not NPY bytes."""
    a = np.asarray(value)
    require(a.dtype.kind in 'fiu' and np.isfinite(a).all(), 'Invalid numeric identity')
    a = np.ascontiguousarray(a, dtype='<f8')
    return dict(shape=list(a.shape), dtype='<f8', order='C', sha256=hashlib.sha256(a.tobytes()).hexdigest())


def plain(value):
    if isinstance(value, (dict, MappingProxyType)):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def frozen(value):
    """Recursively immutable owned results, including non-writeable byte-backed arrays."""
    if isinstance(value, (dict, MappingProxyType)):
        return MappingProxyType({k: frozen(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(frozen(v) for v in value)
    if isinstance(value, np.ndarray):
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    return value


def array(value, shape, name):
    a = np.asarray(value)
    require(a.dtype.kind in 'fiu' and a.shape == shape and np.isfinite(a).all(), 'Invalid ' + name)
    return np.array(a, dtype=float, copy=True)


def error(actual, reference):
    absolute = float(np.max(np.abs(actual - reference)))
    return dict(max_absolute=absolute, max_scaled=absolute / max(1., float(np.max(np.abs(reference)))))


def backward(a, x, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(x) + np.linalg.norm(b)
    return float(np.linalg.norm(a @ x - b) / denominator) if denominator else 0.


def spectrum(a):
    s = np.linalg.svd(a, compute_uv=False)
    return dict(retained_rank=int(np.sum(s > RCOND * s[0])), dimension=len(s),
                singular_values=s, condition=float(s[0] / s[-1]) if s[-1] else None,
                singular=bool(s[-1] == 0), finite=bool(np.isfinite(s).all()))


def validate_identity(identity):
    require(identity['representation'] == REPRESENTATION, 'Wrong representation')
    require(identity['order'] == ORDER and identity['spin_factor'] == 1, 'Wrong packing/spin factor')
    require(identity['fit'] == dict(lambda_=1., eta=0., gamma=0., fourth=0.), 'Unsupported fit')
    n, m, o, v = identity['dimensions']
    require(all(type(i) is int and i > 0 for i in (n, m, o, v)) and o + v == n, 'Invalid full C dimensions')
    for key, shape in [('full_C', [n, n]), ('diagonal_energies', [n])]:
        require(identity[key]['shape'] == shape and len(identity[key]['sha256']) == 64, 'Invalid ' + key + ' identity')
    for role, size, representation in [('MAIN', n, 'S'), ('AUX', m, 'C')]:
        basis = identity[role]
        require(basis['nfunction'] == size and basis['representation'] == representation, 'Invalid basis identity')
        require(bool(basis['descriptor_sha256']) and bool(basis['descriptor']), 'Missing exact basis descriptors')
        require(plain(basis) == descriptor_identity(basis['descriptor']), 'Basis descriptor/hash mismatch')
        desc = basis['descriptor']
        require(set(desc) == {'nfunction', 'representation', 'labels', 'charges', 'centres', 'exponents',
                              'contractions', 'shells'}, 'Incomplete basis descriptor')
        centres, shells, exponents, contractions = [np.asarray(desc[k]) for k in
                                                     ('centres', 'shells', 'exponents', 'contractions')]
        require(centres.shape == (len(desc['labels']), 3) and len(desc['charges']) == len(centres) and
                shells.ndim == 2 and shells.shape[1] == 4 and exponents.ndim == 1 and
                contractions.ndim == 2 and len(contractions) == len(exponents), 'Basis descriptor dimensions')
        require(np.all(exponents > 0) and all(np.isfinite(a).all() for a in
                (centres, shells, exponents, contractions, np.asarray(desc['charges']))), 'Nonfinite basis descriptor')
        count = 0
        for site, angular, first, last in shells:
            require(all(float(i).is_integer() for i in (site, angular, first, last)) and
                    1 <= site <= len(centres) and 0 <= angular < contractions.shape[1] and
                    1 <= first <= last <= len(exponents), 'Invalid shell/order')
            count += 2 * angular + 1 if representation == 'S' else (angular + 1) * (angular + 2) // 2
        require(count == size, 'Basis function count')
    return n, m, o, v


def compare(*, h1, h10, h2, kernel, exchange, reference_d, native_d,
            omega2, reference_cdf, identity, native_identity):
    """Callable supplied-array diagnostic. Identity/shape errors fail closed.

    Numerical failures are per-frequency records, not exceptions that suppress
    remaining experiments. Inputs/results are never mutated. No tolerance knob.
    """
    from psi4.driver.procrouting.sapt.fdds_response import FDDSFullOVResponse
    n, m, o, v = validate_identity(identity)
    require(plain(identity) == plain(native_identity), 'Native/reference input identity mismatch')
    p = o * v
    h1, h10, h2 = [array(x, (p, p), name) for x, name in [(h1, 'H1'), (h10, 'H10'), (h2, 'H2')]]
    k = array(kernel, (m, m), 'Kaux')
    dr = array(reference_d, (p, m), 'reference D')
    dn = array(native_d, (p, m), 'native D')
    w2 = np.asarray(omega2)
    require(w2.ndim == 1 and w2.size > 0 and w2.dtype.kind in 'fiu' and
            np.isfinite(w2).all() and np.all(w2 <= 0), 'Invalid omega2')
    references = array(reference_cdf, (len(w2), m, m), 'reference CDF')
    require(np.asarray(exchange).ndim == 0 and np.asarray(exchange).dtype.kind in 'fiu' and
            np.isfinite(exchange) and 0 <= exchange <= 1, 'Invalid exchange')
    experiments = {}
    for name in EXPERIMENTS:
        baseline = h1 if name == EXPERIMENTS[0] else h10
        coupling = np.zeros_like(k) if name == EXPERIMENTS[0] else (1 - exchange) * k
        rows = []
        for frequency_index, om2 in enumerate(w2):
            row = dict(omega2=float(om2), xi=float(np.sqrt(-om2)), controls={})
            for label, d in [('captured_D', dr), ('native_D', dn)]:
                item = dict(passed=False, finite=False)
                row['controls'][label] = item
                try:
                    with np.errstate(over='raise', invalid='raise', divide='raise'):
                        a0 = h2 @ baseline - om2 * np.eye(p)
                        rhs = -4 * (h2 @ d)
                        afull = h2 @ (baseline + 4 * (d @ (coupling @ d.T))) - om2 * np.eye(p)
                        item.update(baseline_operator=spectrum(a0), full_operator=spectrum(afull))
                        # Provider exclusively owns the response path.
                        out = FDDSFullOVResponse(h1_baseline=baseline, h2=h2, transition_legs=d,
                            coupling=coupling, representation=REPRESENTATION).at_frequency(row['xi'])
                        item.update(raw_cdf=out.raw_coupled, raw_baseline=out.raw_baseline,
                                    representation=out.representation, method=out.method)
                        # Independent full-OV solve; no inverse of H2, symmetry, or Woodbury.
                        denominator = np.eye(m) - out.raw_baseline @ coupling
                        item['coupling_denominator'] = spectrum(denominator)
                        z0 = np.linalg.solve(a0, rhs)
                        item['baseline_backward_residual'] = backward(a0, z0, rhs)
                        item['baseline_direct_error'] = error(out.raw_baseline, d.T @ z0)
                        item['coupling_backward_residual'] = backward(denominator, out.raw_coupled, out.raw_baseline)
                        item['reciprocity'] = error(out.raw_coupled, out.raw_coupled.T)
                        item['reference_error'] = error(out.raw_coupled, references[frequency_index])
                        z = np.linalg.solve(afull, rhs)
                        direct = d.T @ z
                        item.update(direct_cdf=direct, direct_error=error(out.raw_coupled, direct),
                                    direct_backward_residual=backward(afull, z, rhs),
                                    direct_reference_error=error(direct, references[frequency_index]),
                                    direct_reciprocity=error(direct, direct.T), finite=True)
                        metrics = [item[key]['max_scaled'] for key in
                                   ('baseline_direct_error', 'direct_error', 'reference_error', 'direct_reference_error')]
                        metrics += [item[key] for key in ('baseline_backward_residual', 'coupling_backward_residual', 'direct_backward_residual')]
                        item['equivalence_passed'] = bool(item['coupling_denominator']['retained_rank'] == m and
                            item['direct_error']['max_scaled'] <= TOLERANCE and item['baseline_direct_error']['max_scaled'] <= TOLERANCE and
                            max(item[key] for key in ('baseline_backward_residual', 'coupling_backward_residual', 'direct_backward_residual')) <= TOLERANCE)
                        item['forward_passed'] = bool(max(item['reference_error']['max_scaled'], item['direct_reference_error']['max_scaled']) <= TOLERANCE)
                        item['passed'] = bool(all(np.isfinite(x) and x <= TOLERANCE for x in metrics) and item['equivalence_passed'])
                except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
                    item['failure'] = type(exc).__name__ + ': ' + str(exc)
                if 'equivalence_passed' not in item:
                    item['equivalence_passed'] = False
                    item['forward_passed'] = False
            control, native = row['controls']['captured_D'], row['controls']['native_D']
            if 'raw_cdf' in control and 'raw_cdf' in native:
                native['captured_control_error'] = error(native['raw_cdf'], control['raw_cdf'])
                native['forward_passed'] = bool(native['forward_passed'] and
                    native['captured_control_error']['max_scaled'] <= TOLERANCE)
                native['passed'] = bool(native['passed'] and native['forward_passed'])
            row['passed'] = all(x['passed'] for x in row['controls'].values())
            rows.append(row)
        experiments[name] = dict(frequencies=rows, passed=all(x['passed'] for x in rows))
    return frozen(dict(scope='native-fit-to-shared-response error propagation with SUPPLIED operators',
        representation=REPRESENTATION, tolerance=TOLERANCE, pseudoinverse_rcond=RCOND,
        actual_exchange=float(exchange), helper_extra_exchange=0., identity=identity,
        coefficient_error=error(dn, dr), experiments=experiments,
        passed=all(x['passed'] for x in experiments.values()),
        limitations=['No native kernel, partition, molecular property, SCF or end-to-end acceptance',
                     'No quadrature inferred; xi comes only from supplied omega2',
                     'Native OV 1e-3 and Drho 1e-2 profiles do not apply',
                     'Historical dynamic LW failures remain blocked unchanged']))


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def descriptor_identity(basis):
    descriptor = plain(basis)
    payload = json.dumps(descriptor, sort_keys=True, allow_nan=False).encode()
    return dict(nfunction=basis['nfunction'], representation=basis['representation'],
                descriptor=descriptor, descriptor_sha256=hashlib.sha256(payload).hexdigest(),
                numeric_layouts={k: numeric_identity(basis[k]) for k in
                                 ('centres', 'charges', 'shells', 'exponents', 'contractions')})


def capture_inputs(events, snapshots, projection, policy, replay):
    """Validate ONE reader-associated capture, preserving original producer identities."""
    require(replay['passed'] is True and len(replay['frequencies']) == 11 and
            bool(replay['raw_fits']) and replay['hessian_reconstruction'] is not None,
            'Strict raw-fit/Hessian/11-frequency replay gate failed')
    one = lambda kind: next(e for e in events if e['kind'] == kind)
    orb = one('ORBITALS')
    state = one('DIAGONAL_ENERGIES')
    n, m, o, v = orb['header'][2:6]
    require((n, m, o, v) == (92, 246, 5, 87), 'Expected water MAIN92 AUX246 nocc5 nvir87')
    require(np.array_equal(orb['energies'], state['energies']), 'Orbital/diagonal energies mismatch')
    fit = projection['event']['fit']
    require(fit[0][0] == 1 and fit[1] == (1., 0., 0., 0.), 'Unsupported projection fit')
    require(all(s['fit'] == fit and s['d']['start'] == projection['d']['start'] and
                s['d']['end'] == projection['d']['end'] and s['d']['parent'] == projection['d']['parent'] and
                np.array_equal(s['d']['matrix'], projection['d']['matrix']) for s in snapshots),
            'Kernel/response D generation or fit mismatch; no cache retag allowed')
    require(policy['controls'][4] == 0, 'Unsupported kernel policy')
    identity = dict(dimensions=[n, m, o, v], representation=REPRESENTATION, order=ORDER, spin_factor=1,
        fit=dict(lambda_=1., eta=0., gamma=0., fourth=0.), full_C=numeric_identity(orb['c']),
        diagonal_energies=numeric_identity(state['energies']),
        MAIN=descriptor_identity(state['bases']['MAIN']), AUX=descriptor_identity(state['bases']['AUX']))
    validate_identity(identity)
    return orb, state, identity


def match_raw_fit(events, raw_fits, projection):
    """Bind the original NN solve to its observed outgoing OV subset, not a new solve.

    NN(type1) -> OV(type3) is an explicit subset metadata transition. All other
    fields must be identical. Preserve both records; do not relabel the NN cache.
    """
    target = projection['event']['fit']
    matches = []
    for fit in raw_fits:
        original = fit['begin']['fit']
        same_parameters = (original == target or
            (original[0][2] == 1 and target[0][2] == 3 and
             original[0][:2] == target[0][:2] and original[0][3:] == target[0][3:] and
             original[1:] == target[1:]))
        outgoing = [e for e in events if e['kind'] == 'OV_ROW' and
                    e['parent'] == fit['begin']['x_identity'] and
                    e['header'][0] > fit['end']['header'][0] and e['ar'] == 1]
        first = min((e['header'][0] for e in outgoing), default=None)
        if (same_parameters and fit['begin']['x_identity'] == projection['d']['parent'] and
                first == projection['d']['start'] and np.array_equal(fit['ov'], projection['d']['matrix'])):
            matches.append(fit)
    require(len(matches) == 1, 'Missing/ambiguous matching original raw fit')
    return matches[0]


def measure(directory):
    """Fresh bound C++ fit only, after strict retained-capture gates. No output I/O."""
    import psi4
    from psi4.driver.procrouting.sapt.fdds_response import FDDSFullOVResponse
    r = load('replay_response')
    ov = load('compare_native_ov')
    directory = Path(directory).resolve(strict=True)
    paths = sorted(directory.glob('isapol-response-*.dat'))
    require(bool(paths), 'Missing retained response events: ' + str(directory))
    root = Path(__file__).resolve().parents[4]
    shared = Path(inspect.getfile(FDDSFullOVResponse)).resolve()
    core = Path(psi4.core.__file__).resolve()
    require(core.suffix == '.so' and 'stage/lib/psi4' in str(core), 'Expected staged native core, no source shadow')
    source = root / 'psi4/driver/procrouting/sapt/fdds_response.py'
    require(shared.read_bytes() == source.read_bytes(), 'Stale staged shared FDDS helper')
    sources = [Path(__file__), Path(r.__file__), Path(r.base.__file__), Path(r.cache.__file__),
               Path(ov.__file__), Path(ov.profiles.__file__), shared, source, core]
    sources += [root / 'psi4/src/psi4/libisapol' / f for f in
                ('ov_fit.cc', 'ov_fit.h', 'explicit_basis.cc', 'explicit_basis.h', 'aux_coulomb.cc', 'aux_coulomb.h',
                 'orbital_coulomb.cc')]
    sources += [root / 'psi4/src/export_isapol.cc', root / '.pi/audit/measure-shared-full-ov-provider-v2.py']
    before = {str(p.resolve()): hash_file(p) for p in paths + sources}
    print('Validating retained raw-fit/Hessian/11-frequency capture', flush=True)
    replay = r.replay(directory)
    events, snapshots, projection, policy, associated_paths = r.associate(directory)
    require(paths == associated_paths, 'Capture path association changed')
    orb, state, identity = capture_inputs(events, snapshots, projection, policy, replay)
    raw_fits = r.validate_fit_events(events)
    matched = match_raw_fit(events, raw_fits, projection)
    identity.update(capture_directory=str(directory),
        capture_event_manifest_sha256=hashlib.sha256(json.dumps({str(p): before[str(p.resolve())] for p in paths},
                                                              sort_keys=True).encode()).hexdigest(),
        original_raw_fit=plain(matched['begin']['fit']), consumed_projection_fit=plain(projection['event']['fit']),
        reference_generation=dict(start=projection['d']['start'], end=projection['d']['end'],
                                  parent=plain(projection['d']['parent'])),
        metadata_association='original NN parent -> observed OV subset; both identities preserved, no fresh reference fit inferred')
    aux = r.base.explicit_basis(state['bases']['AUX'], psi4.core.IsaBasisRole.MolecularAux)
    main = r.base.explicit_basis(state['bases']['MAIN'], psi4.core.IsaBasisRole.Orbital)
    provenance = json.dumps(identity, sort_keys=True, allow_nan=False)
    print('Generating fresh IsaAuxCoulomb.fit_ov from matched capture full C', flush=True)
    import time
    started = time.perf_counter()
    q, j, a, rhs, d, cpp = ov._fit(psi4.core.IsaAuxCoulomb(aux), main, orb['c'][:, :5],
        orb['c'][:, 5:], 1., psi4, 'cpp', provenance)
    elapsed = time.perf_counter() - started
    require(cpp['provenance'] == provenance, 'Native provenance mismatch')
    tensors = {(e['hessian'], e['tensor_type']): e['matrix'] for e in events if e['kind'] == 'HESSIAN_TENSOR'}
    vv = tensors['h1', 'OVOV']
    x, y = r.exchange_views(vv, tensors['h1', 'VVOO'], 5, 87)
    energy = state['energies']
    delta = np.diag((energy[5:, None] - energy[:5]).ravel())
    exchange = policy['exchange'][0]
    h10 = ((delta + 4 * vv) - exchange * x) - exchange * y
    one = lambda kind: next(e for e in events if e['kind'] == kind)
    operands = dict(h1=one('H1')['matrix'], h10=h10, h2=one('H2')['matrix'], kernel=projection['kernel']['matrix'],
        reference_d=projection['d']['matrix'], native_d=d,
        omega2=np.asarray([s['event']['extra'][0] for s in snapshots]),
        reference_cdf=np.asarray([s['event']['matrix'] for s in snapshots]))
    print('Propagating both experiments and captured-D controls at all 11 frequencies', flush=True)
    report = plain(compare(**operands, exchange=exchange, identity=identity, native_identity=json.loads(cpp['provenance'])))
    # Original values preserved, not A*D reconstructed RHS or old NPY caches.
    operands.update(full_C=orb['c'], diagonal_energies=energy, native_q=q, native_J=j, native_A=a, native_T=rhs,
        original_raw_A=matched['a'], original_raw_RHS=matched['rhs'],
        original_raw_OV_indices=matched['ov_indices'], V=vv, X=x, Y=y,
        VVOO=tensors['h1', 'VVOO'], H2_OVOV=tensors['h2', 'OVOV'], H2_VVOO=tensors['h2', 'VVOO'],
        raw_projected_kernel=one('KERNEL_OVOV_RAW')['matrix'], captured_J=one('J')['matrix'], Delta=delta)
    report.update(execution_kind='fresh psi4.core.IsaAuxCoulomb.fit_ov', fit_seconds=elapsed,
        cpp_diagnostics=cpp, native_fit_backward_residual=backward(a, d.T, rhs.T), native_fit_condition=spectrum(a),
        reference_validation=replay, capture_directory=str(directory),
        generation=dict(projection=plain(projection['event']), response=[dict(start=s['d']['start'], end=s['d']['end'],
            parent=s['d']['parent'], fit=s['fit']) for s in snapshots], raw_fit_begin=plain(matched['begin']),
            raw_fit_end=plain(matched['end'])), policy=plain(policy),
        numerical_kernel_policy=plain(one('NUMERICAL_KERNEL_POLICY')),
        integral_controls=plain(one('J')['extra']),
        operand_identity={k: numeric_identity(v) for k, v in operands.items()},
        core_path=str(core), shared_path=str(shared))
    after = {str(p.resolve()): hash_file(p) for p in paths + sources}
    require(before == after and paths == sorted(directory.glob('isapol-response-*.dat')), 'Inputs/source/core changed during measurement')
    report['custody'] = dict(before=before, after=after)
    report['passed'] = bool(report['passed'] and report['native_fit_backward_residual'] <= TOLERANCE)
    return frozen(report), frozen(operands)


def run(directory, report_path, operands_path):
    """Exclusive destinations outside capture. Numerical failed gates still emit diagnostics."""
    report_path, operands_path = Path(report_path).resolve(), Path(operands_path).resolve()
    directory = Path(directory).resolve()
    require(report_path != operands_path, 'Outputs must differ')
    for path in (report_path, operands_path):
        require(not path.exists() and path.parent.is_dir(), 'Output exists or missing parent: ' + str(path))
        require(not path.is_relative_to(directory), 'Output inside capture forbidden')
    report, operands = measure(directory)
    report = plain(report)
    with operands_path.open('xb') as stream:
        np.savez(stream, **operands)
    report['outputs'] = dict(operands=str(operands_path), operands_file_sha256=hash_file(operands_path),
        report=str(report_path), numeric_layout='all operand identities hash <f8 C-order values, not NPY/NPZ storage')
    with report_path.open('x') as stream:
        json.dump(plain(report), stream, indent=2, allow_nan=False)
        stream.write('\n')
    return frozen(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--operands', required=True, type=Path)
    args = parser.parse_args()
    report = run(args.directory, args.report, args.operands)
    print(json.dumps(dict(passed=report['passed'], report=str(args.report), operands=str(args.operands))))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
