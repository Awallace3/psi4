#!/usr/bin/env python3
"""Strict exported-input trajectory comparison; not native/end-to-end certification."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('trajectory_sweep_reader', Path(__file__).with_name('replay_isa_sweep.py'))
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def compare(actual, reference, initial, tolerance=1e-9):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Invalid tolerance')
    post = reference['post']
    atoms = post['atoms']
    final = actual['final']
    history = actual['history']
    n = len(atoms)
    if not history or any(len(final[k]) != n for k in
                          ['atomic_coefficients', 'shape_coefficients', 'saved_shape_charges', 'tails']):
        raise ValueError('Invalid trajectory dimensions/history')
    last = history[-1]
    errors, diagnostics = {}, {}

    def error(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.shape != y.shape or not x.size or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite or inconsistent comparison arrays')
        absolute = float(np.max(np.abs(x-y)))
        return dict(max_absolute=absolute, max_scaled=absolute/max(1., float(np.max(np.abs(y)))))

    def check(name, x, y):
        errors[name] = error(x, y)

    iterations_match = actual['iterations'] == reference['iteration']
    history_consistent = ([h['iteration'] for h in history] == list(range(1, actual['iterations']+1)))
    flags_match = (last['atom_converged'] == [bool(a['flags'][0]) for a in atoms])
    flags_match &= last['converged'] == actual['converged'] == bool(post['active'][3])
    flags_match &= last['apply_tails'] == final['apply_tails'] == bool(post['active'][2])
    check('deltas', last['deltas'], reference['deltas'])
    check('max_delta', final['max_delta'], reference['max_delta'])
    check('history_max_delta', last['max_delta'], reference['max_delta'])
    check('next_controls', [final['active_w_eps'], final['active_positive_lambda'], int(final['apply_tails'])], post['active'][:3])
    check('history_controls', [last['active_w_eps'], last['active_positive_lambda']], post['active'][:2])
    for i, expected in enumerate(atoms):
        check(f'atom{i+1}_D', final['atomic_coefficients'][i], expected['vectors']['D'])
        check(f'atom{i+1}_W', final['shape_coefficients'][i], expected['vectors']['W0'])
        check(f'atom{i+1}_charge', final['saved_shape_charges'][i], expected['charges'][1])
        check(f'atom{i+1}_history_charge', last['shape_charges'][i], expected['charges'][1])
        tail = final['tails'][i]
        if type(tail['defined']) is not bool or not np.isfinite([tail['amplitude'], tail['exponent'], tail['cutoff']]).all():
            raise ValueError('Invalid tail data')
        flags_match &= tail['defined'] == bool(expected['flags'][1])
        check(f'atom{i+1}_cutoff', tail['cutoff'], expected['tail'][0])
        if tail['defined'] and expected['flags'][1]:
            if tail['exponent'] <= 0 or tail['cutoff'] < 0:
                raise ValueError('Invalid defined tail parameters')
            # Preserve the historical joint denominator; scalar diagnostics do
            # not silently replace that raw-parameter acceptance contract.
            check(f'atom{i+1}_tail', [tail['amplitude'], tail['exponent']], expected['tail'][2:])
            diagnostics[f'atom{i+1}_amplitude'] = error(tail['amplitude'], expected['tail'][2])
            diagnostics[f'atom{i+1}_exponent'] = error(tail['exponent'], expected['tail'][3])
    # Cutoffs are supplied configuration taken from the first capture's POST,
    # not intermediate trajectory injection. Require fixed endpoint radii.
    cutoffs = [a['tail'][0] for a in initial['post']['atoms']]
    cutoff_invariance = (initial['iteration'] == 1 and len(cutoffs) == n and
                         cutoffs == [a['tail'][0] for a in reference['pre']['atoms']] and
                         cutoffs == [a['tail'][0] for a in atoms])
    config_match = (np.array_equal(initial['floats'], reference['floats']) and
                    np.array_equal(initial['ints'], reference['ints']))
    both_converged = actual['converged'] is True and bool(post['active'][3])
    passed = (both_converged and iterations_match and history_consistent and flags_match and
              cutoff_invariance and config_match and all(e['max_scaled'] <= tolerance for e in errors.values()))
    return dict(schema_version=2, evidence_class=__doc__, scaled_tolerance=tolerance,
                passed=bool(passed), both_converged=bool(both_converged),
                iteration_count_matches=bool(iterations_match), history_consistent=bool(history_consistent),
                flags_match=bool(flags_match), cutoff_endpoint_invariance=bool(cutoff_invariance),
                configuration_matches=bool(config_match), cpp_iterations=actual['iterations'],
                reference_iterations=reference['iteration'], errors=errors,
                parameter_diagnostics=diagnostics,
                limitations=['Endpoint cutoff checks do not certify every intermediate reference iteration',
                             'Raw parameter gate is not replaced by represented-function agreement',
                             'Exported basis/density/grid inputs are not native generation'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory', type=Path)
    parser.add_argument('--initial', type=Path, required=True, help='Initial capture directory')
    parser.add_argument('--reference', type=Path, required=True, help='Final capture directory')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f'Refusing to overwrite {args.report}')
    initial_path = args.initial/'isapol-sweep-state.dat'
    final_path = args.reference/'isapol-sweep-state.dat'
    report = compare(json.loads(args.trajectory.read_text()), sweep.read_state(final_path), sweep.read_state(initial_path))
    report['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.trajectory, initial_path, final_path, Path(__file__)]}
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit('Strict trajectory comparison failed; report retained')
