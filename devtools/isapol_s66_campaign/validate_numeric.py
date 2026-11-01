"""Independent rank2 numeric-export checks; no Psi4 imports or computation.

Recompute C6/C8/partial-C10 directly from exported tensor traces and exported
CP weights, using scalar binomial factors6/15/70 and compensated sums. This
checks transport/algebra, not quadrature convergence, model accuracy, source
origin, OS-level independence or independent solver certification.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor_arrays(raw, sites):
    require(type(raw) is bytes and 0 < len(raw) <= 16*1024**2, 'bounded tensor bytes required')
    arrays = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        require(sorted(archive.namelist()) == ['local.npy', 'refined.npy'], 'exact tensor members required')
        require(sum(f.file_size for f in archive.infolist()) <= 16*1024**2, 'expanded tensor bound exceeded')
        for name, width in (('local', 15), ('refined', 9)):
            data = archive.read(name+'.npy')
            stream = io.BytesIO(data)
            version = np.lib.format.read_magic(stream)
            require(version in ((1, 0), (2, 0)), 'unsupported NPY format')
            read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                           else np.lib.format.read_array_header_2_0)
            shape, fortran, dtype = read_header(stream)
            require(shape == (11, sites, width, width) and not fortran
                    and dtype == np.dtype('float64'), 'finite native float64 layout required')
            require(len(data) == stream.tell()+math.prod(shape)*8, 'NPY byte count differs')
            array = np.frombuffer(data, dtype=dtype, offset=stream.tell()).reshape(shape)
            require(np.isfinite(array).all(), 'nonfinite tensor values')
            arrays[name] = array
    return arrays


def validate_numeric(envelope, raw):
    require(envelope.get('schema') == 'native-monomer-artifacts/2', 'native envelope2 required')
    result, request = envelope['result'], envelope['request']
    require(result['schema'] == 'native-monomer-properties/1' and result['status'] == 'complete',
            'complete native numeric export required')
    require(result['method'] == 'PBE0/aug-cc-pVTZ' and math.isfinite(result['energy_Eh']),
            'declared finite molecular result required')
    sites = result['sites']
    labels = [s['label'] for s in sites]
    ns = len(sites)
    require(3 <= ns <= 17 and len(set(labels)) == ns, 'complete independent-site labels required')
    require(all(s['site_type'] == s['label'] and s['rank_limit'] == 2 for s in sites),
            'per-site independent rank2 inventory required')
    require(result['independent_parameters'] == 36*ns
            and result['declaration'] == request['declaration'], 'fit declaration/count differs')
    require(hashlib.sha256(raw).hexdigest() == result['tensors_sha256'], 'tensor checksum differs')
    arrays = tensor_arrays(raw, ns)
    layout = result['tensor_layout']
    require(all(tuple(layout[name+'_shape']) == array.shape for name, array in arrays.items())
            and layout['localization_rank_limit'] == 2, 'tensor layout metadata differs')
    local, refined = arrays['local'], arrays['refined']
    require(not np.any(local[:, :, 8:, :]) and not np.any(local[:, :, :, 8:]),
            'uncomputed rank3 packing is not structural zero')
    symmetry = {}
    for name, array in arrays.items():
        symmetry[name] = float(np.max(np.abs(array-array.swapaxes(2, 3))))
        require(symmetry[name] <= 1e-10+1e-12*float(np.max(np.abs(array))),
                'tensor reciprocity check failed')
    frequency, weights = (np.asarray(result[k], dtype=float) for k in ('frequencies', 'cp_weights'))
    require(frequency.shape == weights.shape == (11,) and np.isfinite(frequency).all()
            and np.isfinite(weights).all() and frequency[0] == 0.
            and np.all(np.diff(frequency) > 0.) and weights[0] == 0.
            and np.all(weights >= 0.) and np.any(weights > 0.), 'invalid CP node/weight inventory')
    diagnostics = result['diagnostics']
    require(len(diagnostics) == 11, 'incomplete frequency diagnostics')
    for f, diagnostic in zip(frequency, diagnostics):
        require(diagnostic['frequency'] == f, 'diagnostic frequency differs')
        for name, bound in (('response_residual', 1e-10), ('localization_residual', 1e-6)):
            value = diagnostic[name]
            require(math.isfinite(value) and 0 <= value < bound, 'response/localization residual failed')
    partition = result['provenance']['partition']
    require(partition['model'] == 'isa' and partition['converged'] is True,
            'native ISA convergence metadata absent')
    usage, limits = result['resources'], request['resources']
    for used, limit in (('maximum_numeric_plan', 'max_bytes'),
                        ('charged_work', 'max_work'), ('charged_io_bytes', 'max_io_bytes')):
        require(type(usage[used]) is int and 0 <= usage[used] <= limits[limit],
                'charged resource budget exceeded')
    alpha1 = np.trace(refined[:, :, 1:4, 1:4], axis1=2, axis2=3)/3.
    alpha2 = np.trace(refined[:, :, 4:9, 4:9], axis1=2, axis2=3)/5.
    pairs = result['isotropic_pairs']
    expected = {(a, b, n) for a in labels for b in labels for n in (6, 8, 10)}
    actual = [(p['site_a'], p['site_b'], p['order']) for p in pairs]
    require(len(actual) == len(expected) and set(actual) == expected, 'isotropic coverage differs')
    errors = []
    for pair in pairs:
        i, j, n = labels.index(pair['site_a']), labels.index(pair['site_b']), pair['order']
        included = {6: [[1, 1]], 8: [[1, 2], [2, 1]], 10: [[2, 2]]}[n]
        require(pair['included_rank_pairs'] == included
                and pair['missing_rank_pairs'] == ([] if n != 10 else [[1, 3], [3, 1]])
                and pair['unrestricted_complete'] is (n != 10), 'C_n rank coverage differs')
        if n == 6:
            integrand = 6*alpha1[:, i]*alpha1[:, j]
        elif n == 8:
            integrand = 15*(alpha1[:, i]*alpha2[:, j]+alpha2[:, i]*alpha1[:, j])
        else:
            integrand = 70*alpha2[:, i]*alpha2[:, j]
        computed = math.fsum((weights*integrand).tolist())
        error = abs(computed-pair['value'])
        require(math.isfinite(pair['value']) and error <= 1e-10+1e-12*abs(computed),
                'exported C_n does not match independent tensor reduction')
        errors.append(error)
    ineligible = [p for p in pairs if p['order'] in (6, 8) and p['value'] <= 0.]
    require(result['nonpositive_C6_C8_pairs'] == ineligible
            and result['bj_positive_eligible'] is (not ineligible), 'BJ eligibility metadata differs')
    verified = result['pfit_native_verified']
    require(len(verified) == 11 and all(type(v) is bool for v in verified), 'solver flags absent')
    return dict(
        status='PASS_NUMERIC_TRANSPORT_ONLY', tensor_sha256=hashlib.sha256(raw).hexdigest(),
        sites=ns, nodes=11, independent_parameters=36*ns, isotropic_entries=len(pairs),
        static_alpha1_sum_au=float(alpha1[0].sum()), static_alpha2_sum_au=float(alpha2[0].sum()),
        maximum_cn_absolute_difference=max(errors), tensor_reciprocity_maxabs=symmetry,
        maximum_response_residual=max(d['response_residual'] for d in diagnostics),
        maximum_localization_residual=max(d['localization_residual'] for d in diagnostics),
        bj_positive_eligible=not ineligible, nonpositive_C6_C8_entries=len(ineligible),
        reported_native_verified=verified, independent_solver_certified=False, full_C10=False,
        charged_work=usage['charged_work'], maximum_numeric_plan_bytes=usage['maximum_numeric_plan'],
        qualification='Numeric transport, coverage, residual and C_n algebra checks only; '
                      'no origin/runtime audit, convergence, accuracy/parity or independent solver certification.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    envelope = json.loads((args.evidence/'server-native-envelope.json').read_text())
    raw = (args.evidence/'server-tensors.npz').read_bytes()
    report = validate_numeric(envelope, raw)
    report['envelope_sha256'] = hashlib.sha256(
        (args.evidence/'server-native-envelope.json').read_bytes()).hexdigest()
    report['validator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
