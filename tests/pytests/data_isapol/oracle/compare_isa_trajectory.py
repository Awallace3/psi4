#!/usr/bin/env python3
"""Strict exported-input trajectory comparison; not native/end-to-end certification."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('trajectory_sweep_reader', Path(__file__).with_name('replay_isa_sweep.py'))
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


spec = importlib.util.spec_from_file_location('trajectory_profiles', Path(__file__).with_name('provisional_acceptance.py'))
profiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiles)
ORIGINAL_SHA256 = '420bf0a375f622bea7ef00635993134986c6b788714a42a141c74b0552d26904'


def compare(actual, reference, initial, tolerance=1e-9, *, profile='strict'):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Invalid tolerance: expected a finite positive value')
    profiles.require(profile in profiles.PROFILES, 'Unknown profile')
    custom_tolerance = tolerance != profiles.STRICT
    profiles.require(not custom_tolerance or profile == 'strict',
                     'Custom tolerance cannot redefine a fixed provisional profile')
    post = reference['post']
    atoms = post['atoms']
    final = actual['final']
    history = actual['history']
    n = len(atoms)
    if not n or not history or any(len(final[k]) != n for k in
                          ['atomic_coefficients', 'shape_coefficients', 'saved_shape_charges', 'tails']):
        raise ValueError('Invalid trajectory dimensions/history')
    last = history[-1]
    profiles.require(type(actual['iterations']) is int and actual['iterations'] > 0 and
                     type(actual['converged']) is bool and type(final['apply_tails']) is bool,
                     'Invalid trajectory iteration/flags')
    for h in history:
        profiles.require(type(h['iteration']) is int and type(h['converged']) is bool and
                         type(h['apply_tails']) is bool and len(h['atom_converged']) == n and
                         all(type(v) is bool for v in h['atom_converged']), 'Invalid history flags')
        profiles.require(len(h['deltas']) == len(h['shape_charges']) == n and
                         all(np.isfinite(v).all() for v in (h['deltas'], h['shape_charges'],
                                     [h['max_delta'], h['active_w_eps'], h['active_positive_lambda']])),
                         'Invalid history values')
    for state in (initial, reference):
        profiles.require(np.isfinite(state['floats']).all() and np.isfinite(state['ints']).all(),
                         'Nonfinite configuration')
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
    report = dict(schema_version=2, evidence_class=__doc__, scaled_tolerance=tolerance,
                passed=bool(passed), both_converged=bool(both_converged),
                iteration_count_matches=bool(iterations_match), history_consistent=bool(history_consistent),
                flags_match=bool(flags_match), cutoff_endpoint_invariance=bool(cutoff_invariance),
                configuration_matches=bool(config_match), cpp_iterations=actual['iterations'],
                reference_iterations=reference['iteration'], errors=errors,
                parameter_diagnostics=diagnostics,
                limitations=['Endpoint cutoff checks do not certify every intermediate reference iteration',
                             'Raw parameter gate is not replaced by represented-function agreement',
                             'Exported basis/density/grid inputs are not native generation'])
    structural = {k: report[k] for k in profiles.contract('trajectory', n)[2]}
    if custom_tolerance:
        # Preserve the pre-profile callable API, but never turn a caller-selected
        # threshold into a named strict/provisional acceptance certificate.
        report.update(comparison_mode='legacy-custom-tolerance', profile_certified=False)
        for diagnostic in diagnostics.values():
            diagnostic.update(comparison_threshold=tolerance,
                              comparison_passed=diagnostic['max_scaled'] <= tolerance,
                              acceptance_role='diagnostic-only; never substituted for joint tail metric')
    else:
        gates = profiles.evaluate('trajectory', {k: e['max_scaled'] for k, e in errors.items()},
                                  structural, profile=profile, atom_count=n,
                                  tail_atoms=tuple(i+1 for i, a in enumerate(atoms) if a['flags'][1] and final['tails'][i]['defined']))
        profiles.require(gates['passed'] == report['passed'], 'Strict trajectory gate disagreement')
        report.update(gates)
        for diagnostic in diagnostics.values():
            diagnostic.update(strict_threshold=tolerance, strict_passed=diagnostic['max_scaled'] <= tolerance,
                              acceptance_role='diagnostic-only; never substituted for joint tail metric')
    report.update(scope='exported fixed-density trajectory endpoint comparison; not native generation',
                  execution_kind='recomparison of supplied trajectory; no controller execution',
                  broader_pipeline_status='missing-native-density-trajectory/native-response/localization; not certified',
                  provenance={'original_comparator_sha256': ORIGINAL_SHA256,
                              'comparison_sources': profiles.hashes([Path(__file__), Path(profiles.__file__)]),
                              'retained_execution': {k: actual.get(k) for k in ('entry_sha256', 'initial_checkpoint_sha256', 'psi4_extension')},
                              'retained_execution_status': 'reported metadata only; trajectory bytes are hashed by CLI, not a fresh execution attestation'})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory', type=Path)
    parser.add_argument('--initial', type=Path, required=True, help='Initial capture directory')
    parser.add_argument('--reference', type=Path, required=True, help='Final capture directory')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--profile', choices=profiles.PROFILES, default='strict')
    parser.add_argument('--core', type=Path, help='Optional current staged extension to hash only; never loaded/executed')
    args = parser.parse_args()
    profiles.preflight([args.report])
    initial_path = args.initial/'isapol-sweep-state.dat'
    final_path = args.reference/'isapol-sweep-state.dat'
    # Hash complete supplied capture directories, not merely the two state files.
    lineage = [args.trajectory, args.initial, args.reference, *profiles.source_paths()]
    if args.core is not None:
        lineage.append(args.core)
    profiles.require(not args.report.resolve().is_relative_to(args.initial.resolve()) and
                     not args.report.resolve().is_relative_to(args.reference.resolve()),
                     'Report must be outside reference input directories')
    before = profiles.hashes(lineage)
    report = compare(json.loads(args.trajectory.read_text()), sweep.read_state(final_path), sweep.read_state(initial_path), profile=args.profile)
    after = profiles.hashes(lineage)
    profiles.require(before == after, 'Source/input changed during comparison')
    report['source_sha256'] = after
    report['provenance'].update(before=before, after=after,
                                current_core_snapshot=str(args.core.resolve()) if args.core is not None else None,
                                current_core_status='hash only; not execution provenance' if args.core is not None else 'not requested; no core used in recomparison')
    with profiles.exclusive_outputs([args.report]) as streams:
        streams[0].write(profiles.json_bytes(report))
    print(json.dumps(report, indent=2))
    if not report['selected_profile_passed']:
        raise SystemExit('Selected trajectory comparison profile failed; report retained')


if __name__ == '__main__':
    main()
