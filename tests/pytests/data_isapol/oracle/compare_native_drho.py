#!/usr/bin/env python3
"""Fresh native C++ Drho-C comparison, not native end-to-end acceptance.

Faithfully refactored from .pi/audit/measure-native-drho.py; equations, multiplication
association, screening and original forward/backward metrics are unchanged.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

spec = importlib.util.spec_from_file_location('native_profiles_drho', Path(__file__).with_name('provisional_acceptance.py'))
profiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiles)
ORIGINAL_SOURCE = 'measure-native-drho.py'
ORIGINAL_SHA256 = 'eaf1852bf2667198754d34c1a83a05a94ee33882d7c0453170fce59f437e95e7'


def _measure(directory, checkpoint, psi4, r):
    s=r.read_state(directory/'isapol-native-df-state.dat')
    aux=r.base.explicit_basis(s['bases']['AUX'],psi4.core.IsaBasisRole.MolecularAux)
    main=r.base.explicit_basis(s['bases']['MAIN'],psi4.core.IsaBasisRole.Orbital)
    p=psi4.core.IsaAuxCoulomb(aux)
    started=time.perf_counter()
    fit=p.fit_drho_c(main,psi4.core.Matrix.from_array(s['c']))
    elapsed=time.perf_counter()-started
    reference_a,_=r.read_metric(directory/'isapol-native-df-A-000001.dat','A',s)
    def error(x,y):
        x,y=np.asarray(x),np.asarray(y)
        profiles.require(x.shape == y.shape and x.size and np.isfinite(x).all() and np.isfinite(y).all(), 'Invalid comparison arrays')
        absolute=float(np.max(np.abs(x-y)))
        return dict(max_absolute=absolute,max_scaled=absolute/max(1.,float(np.max(np.abs(y)))))
    errors=dict(metric=error(fit.metric.np,reference_a),rhs=error(fit.rhs,s['rhs']),
                coefficients=error(fit.coefficients,s['d']),
                fitted_electrons=error(fit.fitted_electrons,float(s['q']@s['d'])))
    c=r.base.read_checkpoint(checkpoint)
    desc=c['descriptors']
    profiles.require(np.array_equal(desc['density_coefficients'],s['d']), 'Checkpoint density coefficient identity mismatch')
    profiles.require(set(desc['density_basis']) == set(s['bases']['AUX']), 'Checkpoint basis schema mismatch')
    for key in s['bases']['AUX']:
        profiles.require(np.array_equal(desc['density_basis'][key],s['bases']['AUX'][key]), 'Checkpoint basis identity mismatch: '+key)
    sites=[int(x)-1 for x in desc['density_neighbours'] if x>0]
    native_density=psi4.core.IsaFixedDensity(aux,fit.coefficients)
    rho=np.asarray(native_density.evaluate(c['points'].tolist(),sites))
    errors['sampled_density']=error(rho,c['density'])
    delta=rho-c['density']
    pointwise=float(np.max(np.abs(delta)/np.maximum(1.,np.abs(c['density']))))
    l2=float(np.sqrt(np.dot(np.abs(c['weights']),delta*delta)/np.dot(np.abs(c['weights']),c['density']**2)))
    report=dict(schema_version=1,evidence_class='native explicit-input Libint2 Drho-C solve and sampled-density comparison; not native SCF or ISA trajectory',
        errors=errors,pointwise_scaled_density_error=pointwise,relative_abs_weighted_density_l2=l2,
        sampled_points=len(rho),density_sites=sites,relative_infinity_backward_residual=fit.relative_residual,
        native_condition_number_2=float(np.linalg.cond(fit.metric.np)),fitted_electrons=fit.fitted_electrons,
        elapsed_seconds=elapsed,scaled_tolerance=1e-9,
        coefficients_passed=errors['coefficients']['max_scaled']<=1e-9,
        sampled_density_passed=pointwise<=1e-9 and l2<=1e-9,
        limitations=['Orbitals and basis recipes supplied; not native SCF','No coefficient rescaling, regularization or symmetrization',
                     'No native-density ISA trajectory or downstream property acceptance','Ill-conditioned raw-coefficient and represented-density comparisons are distinct'])
    report['passed']=all(e['max_scaled']<=1e-9 for e in errors.values()) and report['sampled_density_passed'] and fit.relative_residual<=1e-9

    return report, dict(coefficients=list(fit.coefficients), charges=list(fit.charges))


def measure(directory, checkpoint, *, profile='strict'):
    """Execute a fresh measurement. No output side effects; returns report, coefficients.

    Reference replay remains strict and is independently recomputed. Production
    comparisons require an installed/staged Psi4 extension, never a source shim.
    """
    profiles.require(profile in profiles.PROFILES, 'Unknown profile')
    directory, checkpoint = Path(directory).resolve(strict=True), Path(checkpoint).resolve(strict=True)
    psi4 = profiles.production_psi4()
    lineage = [directory, checkpoint, *profiles.source_paths(ORIGINAL_SOURCE), Path(psi4.core.__file__)]
    before = profiles.hashes(lineage)
    r = profiles.load('replay_native_df')
    validation = r.replay(directory)
    profiles.require(validation['passed'] is True, 'Strict reference replay failed')
    report, coefficients = _measure(directory, checkpoint, psi4, r)
    # Fail closed on nonfinite diagnostics as well as acceptance metrics.
    profiles.json_bytes(report)
    metrics = {k: v['max_scaled'] for k, v in report['errors'].items()}
    for key in ('pointwise_scaled_density_error', 'relative_abs_weighted_density_l2', 'relative_infinity_backward_residual'):
        metrics[key] = float(report[key])
    gates = profiles.evaluate('native-drho', metrics,
                              dict(reference_validation=True, checkpoint_identity=True, finite_measurement=True), profile=profile)
    profiles.require(gates['passed'] == bool(report['passed']), 'Strict measurement gate disagreement')
    report.update(gates)
    report['psi4_extension_sha256'] = before[str(Path(psi4.core.__file__).resolve())]
    report['script_sha256'] = before[str(Path(__file__).resolve())]
    after = profiles.hashes(lineage)
    profiles.require(before == after, 'Input/source/extension changed during measurement')
    report.update(execution_kind='fresh native measurement execution',
                  scope="native C++ explicit-input finite-penalty Drho-C fit and density sampling; supplied orbitals, not native SCF",
                  reference_validation=validation,
                  provenance=dict(original_measurement_source=ORIGINAL_SOURCE, original_measurement_sha256=ORIGINAL_SHA256,
                                  before=before, after=after, psi4_extension=str(Path(psi4.core.__file__).resolve())),
                  broader_pipeline=dict(status='missing-implementation', passed=False,
                                        missing_implementations=['native SCF', "native-density ISA trajectory",  'native response operator producers', 'published localization'],
                                        response_transition_leg_diagnostic='deferred: dimensional/representation bridge through captured H1/H2 not proven here'))
    return report, coefficients


def run(directory, checkpoint, report_path, coefficients_path, *, profile='strict'):
    """Preflight both destinations before measuring; write exclusively, never overwrite."""
    profiles.preflight([report_path, coefficients_path])
    # Outputs inside input directories would change the snapshotted lineage.
    for path in (report_path, coefficients_path):
        profiles.require(not Path(path).resolve().is_relative_to(Path(directory).resolve()), 'Output must be outside reference input directory')
    report, coefficients = measure(directory, checkpoint, profile=profile)
    coefficient_payload = profiles.json_bytes(coefficients)
    report['outputs'] = dict(report=str(Path(report_path).resolve()), coefficients=str(Path(coefficients_path).resolve()),
                             coefficients_sha256=profiles.hashlib.sha256(coefficient_payload).hexdigest(), format='JSON coefficients and charges')
    payload = profiles.json_bytes(report)
    with profiles.exclusive_outputs([report_path, coefficients_path]) as streams:
        streams[0].write(payload)
        streams[1].write(coefficient_payload)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='Retained reference capture directory')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--coefficients', type=Path, required=True)
    parser.add_argument('--profile', choices=profiles.PROFILES, default='strict')
    args = parser.parse_args()
    report = run(args.directory, args.checkpoint, args.report, args.coefficients, profile=args.profile)
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report['selected_profile_passed']:
        raise SystemExit('Selected native comparison profile failed; fresh evidence retained')


if __name__ == '__main__':
    main()
