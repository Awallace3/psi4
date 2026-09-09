#!/usr/bin/env python3
"""Fresh native integral + supplied-C NumPy OV comparison, not native end-to-end acceptance.

Faithfully refactored from .pi/audit/measure-native-ov-fit.py; equations, multiplication
association, screening and original forward/backward metrics are unchanged.
Opt-in --producer cpp measures the bound native-integral/supplied-orbital C++ fit;
NumPy remains the default/control. Neither producer supplies native SCF/H1/H2.
"""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import time
import numpy as np

spec = importlib.util.spec_from_file_location('native_profiles_ov', Path(__file__).with_name('provisional_acceptance.py'))
profiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiles)
ORIGINAL_SOURCE = 'measure-native-ov-fit.py'
ORIGINAL_SHA256 = '79f16d0298db0475ce16022d97e2699306810f6f11be4878b7107de2c7bbbea0'


def _producer_identity(producer):
    profiles.require(producer in ('numpy', 'cpp'), 'Unknown OV producer')
    return dict(producer=producer,
                producer_api=('psi4.core.IsaAuxCoulomb.fit_ov' if producer == 'cpp' else 'numpy.linalg.solve'),
                representation='fitted_density_coefficients')


def _array_identity(array):
    array = np.asarray(array)
    profiles.require(array.dtype.kind in 'fiu', 'Unsupported lineage array dtype')
    array = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder('<'))
    return dict(shape=list(array.shape), dtype=array.dtype.str, sha256=profiles.hashlib.sha256(array.tobytes()).hexdigest())


def _basis_identity(descriptor):
    return {key: (_array_identity(value) if isinstance(value, np.ndarray) else value)
            for key, value in descriptor.items()}


def _fit(p, main, occ, vir, penalty, psi4, producer, provenance):
    """Select the actual producer; CPP operands are measured, never re-solved."""
    _producer_identity(producer)
    if producer == 'cpp':
        result = p.fit_ov(main, psi4.core.Matrix.from_array(occ),
                          psi4.core.Matrix.from_array(vir), provenance, penalty)
        profiles.require(result.representation == 'fitted_density_coefficients' and result.lapack_info == 0,
                         'Invalid C++ OV result representation/solve status')
        profiles.require(result.provenance == provenance and result.charge_penalty == penalty and
                         result.nmain == occ.shape[0] and result.noccupied == occ.shape[1] and
                         result.nvirtual == vir.shape[1] and result.ntransition == occ.shape[1]*vir.shape[1] and
                         result.order == 'p=a+noccupied*r; occupied-fast', 'C++ OV result identity mismatch')
        q = np.asarray(result.charges).copy()
        j, A, rhs, d = (getattr(result, key).np.copy() for key in ('coulomb_metric', 'metric', 'rhs', 'coefficients'))
        m, nov = len(q), occ.shape[1]*vir.shape[1]
        profiles.require(result.naux == m and j.shape == A.shape == (m,m) and
                         rhs.shape == d.shape == (nov,m), 'C++ OV result dimensions mismatch')
        profiles.require(all(np.isfinite(x).all() for x in (q,j,A,rhs,d)), 'Nonfinite C++ OV operands')
        return q, j, A, rhs, d, dict(lapack_info=result.lapack_info, solver=result.solver,
                                    relative_backward_residual=result.relative_backward_residual,
                                    provenance=result.provenance)
    q=np.asarray(p.charges());j=p.metric().np.copy()
    b=p.three_center(main).np.copy().reshape(len(q),occ.shape[0],occ.shape[0])
    # Fixed before measurement: left-associated Cocc.T @ B[k] @ Cvir, no spin factor.
    rhs=np.asarray([(occ.T@block@vir).T.reshape(-1) for block in b]).T
    A=j+(penalty*q[:,None])*q[None,:]
    d=np.linalg.solve(A,rhs.T).T
    return q, j, A, rhs, d, None


def _measure(directory, checkpoint, psi4, r, producer='numpy'):
    events,snapshots,projection,policy,paths=r.associate(directory)
    orb=next(x for x in events if x['kind']=='ORBITALS')
    state=next(x for x in events if x['kind']=='DIAGONAL_ENERGIES')
    _,_,n,m,o,v,_=orb['header']
    params=snapshots[0]['fit']; penalty=params[1][0]
    profiles.require(params[0][0]==1 and params[1][1:]==(0.,0.,0.) and penalty==1., 'Unsupported finite OV penalty/fit')
    profiles.require(all(s['fit']==params and np.array_equal(s['d']['matrix'],snapshots[0]['d']['matrix']) for s in snapshots), 'Response fit changed across frequencies')
    aux=r.base.explicit_basis(state['bases']['AUX'],psi4.core.IsaBasisRole.MolecularAux)
    main=r.base.explicit_basis(state['bases']['MAIN'],psi4.core.IsaBasisRole.Orbital)
    p=psi4.core.IsaAuxCoulomb(aux)
    start=time.perf_counter()
    c=orb['c'];occ=c[:,:o];vir=c[:,o:o+v]
    input_identity = dict(capture_directory=str(Path(directory).resolve()),
                          orbital_header=list(orb['header']), occupied_columns=[0,o], virtual_columns=[o,o+v],
                          full_C=_array_identity(c), occupied=_array_identity(occ), virtuals=_array_identity(vir),
                          MAIN=_basis_identity(state['bases']['MAIN']),
                          AUX=_basis_identity(state['bases']['AUX']),
                          reference_D=_array_identity(snapshots[0]['d']['matrix']))
    provenance = json.dumps(input_identity, sort_keys=True, allow_nan=False)
    q,j,A,rhs,d,cpp_diagnostics = _fit(p,main,occ,vir,penalty,psi4,producer,provenance)
    fit_time=time.perf_counter()-start
    reference=snapshots[0]['d']['matrix']
    def error(a,b):
        a,b=np.asarray(a),np.asarray(b)
        profiles.require(a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all(), 'Invalid comparison arrays')
        absolute=float(np.max(np.abs(a-b)))
        return dict(max_absolute=absolute,max_scaled=absolute/max(1.,float(np.max(np.abs(b)))))
    errors=dict(metric=error(j,next(e for e in events if e['kind']=='J')['matrix']),coefficients=error(d,reference))
    cp=r.base.read_checkpoint(checkpoint)
    for key in state['bases']['AUX']:
        profiles.require(np.array_equal(state['bases']['AUX'][key],cp['descriptors']['density_basis'][key]), 'Checkpoint basis identity mismatch: '+key)
    sites=[int(x)-1 for x in cp['descriptors']['density_neighbours'] if x>0]
    maximum=pointwise=reference_max=weighted_delta=weighted_reference=0.
    for offset in range(0,len(cp['points']),1024):
        points=cp['points'][offset:offset+1024]
        phi=aux.evaluate_screened(points.tolist(),sites).np.copy()
        ref=phi@reference.T
        delta=phi@(d-reference).T
        maximum=max(maximum,float(np.max(np.abs(delta))))
        reference_max=max(reference_max,float(np.max(np.abs(ref))))
        pointwise=max(pointwise,float(np.max(np.abs(delta)/np.maximum(1.,np.abs(ref)))))
        weights=np.abs(cp['weights'][offset:offset+1024])[:,None]
        weighted_delta+=float(np.sum(weights*delta*delta))
        weighted_reference+=float(np.sum(weights*ref*ref))
    errors['sampled_transition_density']=dict(max_absolute=maximum,max_scaled=maximum/max(1.,reference_max))
    residual=float(np.linalg.norm(A@d.T-rhs.T)/(np.linalg.norm(A)*np.linalg.norm(d)+np.linalg.norm(rhs)))
    report=dict(scope='native explicit-basis Libint2 q/J/B with supplied full C, NumPy finite-penalty OV solve; not native SCF/response',
        dimensions=dict(main=n,aux=m,occupied=o,virtual=v),penalty=penalty,
        contraction_order='left-associated Cocc.T @ B[k] @ Cvir; occupied-fast output; no factor2/4',
        errors=errors,pointwise_scaled_density_error=pointwise,
        relative_abs_weighted_density_l2=float(np.sqrt(weighted_delta/weighted_reference)),
        sampled_points=len(cp['points']),sampled_transitions=o*v,
        charge_difference_max_absolute=float(np.max(np.abs((d-reference)@q))),
        reference_fit_charge_max_absolute=float(np.max(np.abs(reference@q))),
        native_fit_charge_max_absolute=float(np.max(np.abs(d@q))),
        condition_A=float(np.linalg.cond(A)),relative_backward_residual=residual,fit_seconds=fit_time,
        tolerance=1e-9,limitations=['NumPy diagnostic fit, not native C++ transition-fit API',
           'No direct reference OV RHS capture','Sampled comparisons use a common independently validated native AUX sampler',
           'No rescaling, symmetrization, regularization or tolerance changes','No native response/end-to-end acceptance'])
    report['passed']=all(e['max_scaled']<=1e-9 for e in errors.values()) and pointwise<=1e-9 and residual<=1e-9 and report['relative_abs_weighted_density_l2']<=1e-9

    report.update(_producer_identity(producer))
    report['compared_input_lineage'] = input_identity
    report['operand_identity'] = {key:_array_identity(x) for key,x in zip(('q','J','A','T','D'), (q,j,A,rhs,d))}
    if producer == 'cpp':
        report['cpp_diagnostics'] = cpp_diagnostics
        report['scope'] = 'native-integral/supplied-orbital C++ OV fit; not native SCF/H1/H2/response'
        report['limitations'][0] = 'Supplied MAIN orbitals; no native SCF or reference-forward acceptance inferred from LU success'
    return report, d


def measure(directory, checkpoint, *, reference_replay, profile='strict', producer='numpy'):
    """Execute a fresh measurement. No output side effects; returns report, coefficients.

    Reference replay remains strict and is independently recomputed. Production
    comparisons require an installed/staged Psi4 extension, never a source shim.
    """
    identity = _producer_identity(producer)
    profiles.require(profile in profiles.PROFILES, 'Unknown profile')
    directory, checkpoint = Path(directory).resolve(strict=True), Path(checkpoint).resolve(strict=True)
    reference_replay = Path(reference_replay).resolve(strict=True)
    psi4 = profiles.production_psi4()
    lineage = [directory, checkpoint, reference_replay, *profiles.source_paths(ORIGINAL_SOURCE), Path(psi4.core.__file__)]
    before = profiles.hashes(lineage)
    retained_validation = json.loads(reference_replay.read_text())
    profiles.require(retained_validation['passed'] is True, 'Retained reference control failed; not a native acceptance gate')
    r = profiles.load('replay_response')
    validation = r.replay(directory)
    profiles.require(validation['passed'] is True, 'Strict reference replay failed')
    # Keep the legacy default call shape for existing test doubles.
    report, coefficients = (_measure(directory, checkpoint, psi4, r) if producer == 'numpy' else
                            _measure(directory, checkpoint, psi4, r, producer='cpp'))
    if producer == 'cpp':
        profiles.require(report.get('producer_api') == identity['producer_api'] and
                         'cpp_diagnostics' in report, 'C++ OV producer was not executed')
    # Fail closed on nonfinite diagnostics as well as acceptance metrics.
    profiles.json_bytes(report)
    metrics = {k: v['max_scaled'] for k, v in report['errors'].items()}
    for key in ('pointwise_scaled_density_error', 'relative_abs_weighted_density_l2', 'relative_backward_residual'):
        metrics[key] = float(report[key])
    gates = profiles.evaluate('native-ov', metrics,
                              dict(reference_validation=True, checkpoint_identity=True, finite_measurement=True), profile=profile)
    profiles.require(gates['passed'] == bool(report['passed']), 'Strict measurement gate disagreement')
    report.update(gates)
    report['response_replay_report_sha256'] = before[str(reference_replay)]
    report['sample_checkpoint_sha256'] = before[str(checkpoint)]
    report['extension_sha256'] = before[str(Path(psi4.core.__file__).resolve())]
    report['script_sha256'] = before[str(Path(__file__).resolve())]
    after = profiles.hashes(lineage)
    profiles.require(before == after, 'Input/source/extension changed during measurement')
    report.update(execution_kind='fresh native measurement execution',
                  scope="native-integral + supplied-C NumPy diagnostic; not production C++ OV API/native SCF",
                  reference_validation=validation,
                  provenance=dict(original_measurement_source=ORIGINAL_SOURCE, original_measurement_sha256=ORIGINAL_SHA256,
                                  before=before, after=after, psi4_extension=str(Path(psi4.core.__file__).resolve())),
                  broader_pipeline=dict(status='missing-implementation', passed=False,
                                        missing_implementations=['native SCF', "production C++ OV API",  'native response operator producers', 'published localization'],
                                        response_transition_leg_diagnostic='deferred: dimensional/representation bridge through captured H1/H2 not proven here'))
    report.update(identity)
    if producer == 'cpp':
        report['scope'] = 'native-integral/supplied-orbital C++ OV fit; not native SCF/H1/H2/response'
        report['broader_pipeline']['missing_implementations'].remove('production C++ OV API')
    return report, coefficients


def run(directory, checkpoint, report_path, coefficients_path, *, reference_replay, profile='strict', producer='numpy'):
    """Preflight both destinations before measuring; write exclusively, never overwrite."""
    profiles.preflight([report_path, coefficients_path])
    # Outputs inside input directories would change the snapshotted lineage.
    for path in (report_path, coefficients_path):
        profiles.require(not Path(path).resolve().is_relative_to(Path(directory).resolve()), 'Output must be outside reference input directory')
    _producer_identity(producer)
    kwargs = dict(reference_replay=reference_replay, profile=profile)
    if producer != 'numpy':
        kwargs['producer'] = producer
    report, coefficients = measure(directory, checkpoint, **kwargs)
    buffer = io.BytesIO()
    np.save(buffer, coefficients, allow_pickle=False)
    coefficient_payload = buffer.getvalue()
    report['outputs'] = dict(report=str(Path(report_path).resolve()), coefficients=str(Path(coefficients_path).resolve()),
                             coefficients_sha256=profiles.hashlib.sha256(coefficient_payload).hexdigest(), format='NPY occupied-fast transition-by-auxiliary coefficients')
    payload = profiles.json_bytes(report)
    with profiles.exclusive_outputs([report_path, coefficients_path]) as streams:
        streams[0].write(payload)
        streams[1].write(coefficient_payload)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='Retained reference capture directory')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference-replay', type=Path, required=True, help='Retained strict reference replay report; never native acceptance')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--coefficients', type=Path, required=True)
    parser.add_argument('--profile', choices=profiles.PROFILES, default='strict')
    parser.add_argument('--producer', choices=('numpy', 'cpp'), default='numpy',
                        help='NumPy control (default) or bound native-integral/supplied-orbital C++ fit')
    args = parser.parse_args()
    report = run(args.directory, args.checkpoint, args.report, args.coefficients, reference_replay=args.reference_replay,
                 profile=args.profile, producer=args.producer)
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report['selected_profile_passed']:
        raise SystemExit('Selected native comparison profile failed; fresh evidence retained')


if __name__ == '__main__':
    main()
