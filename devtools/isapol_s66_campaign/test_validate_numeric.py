"""Independent export algebra on synthetic tensors, not molecular evidence."""
import hashlib
import importlib.util
import io
from pathlib import Path
import zipfile

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('validate_numeric', Path(__file__).with_name('validate_numeric.py'))
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def example():
    refined = np.zeros((11, 3, 9, 9))
    local = np.zeros((11, 3, 15, 15))
    for i in range(1, 4):
        refined[:, :, i, i] = 1.
        local[:, :, i-1, i-1] = 1.
    for i in range(4, 9):
        refined[:, :, i, i] = 2.
        local[:, :, i-1, i-1] = 2.
    stream = io.BytesIO()
    np.savez_compressed(stream, local=local, refined=refined)
    raw = stream.getvalue()
    labels = ['O1', 'H1', 'H2']
    pairs = [dict(site_a=a, site_b=b, order=n, value=value,
                  unrestricted_complete=n != 10,
                  included_rank_pairs={6: [[1, 1]], 8: [[1, 2], [2, 1]], 10: [[2, 2]]}[n],
                  missing_rank_pairs=[] if n != 10 else [[1, 3], [3, 1]])
             for a in labels for b in labels for n, value in [(6, 3.), (8, 30.), (10, 140.)]]
    request = dict(declaration={'frames': [np.eye(3).tolist()]*3},
                   resources=dict(max_bytes=10**9, max_work=10**9, max_io_bytes=10**9))
    result = dict(
        schema='native-monomer-properties/1', status='complete', method='PBE0/aug-cc-pVTZ',
        energy_Eh=-1., declaration=request['declaration'],
        independent_parameters=108, frequencies=list(map(float, range(11))),
        cp_weights=[0., .5]+[0.]*9,
        sites=[dict(label=s, site_type=s, rank_limit=2) for s in labels],
        tensor_layout=dict(local_shape=[11, 3, 15, 15], refined_shape=[11, 3, 9, 9],
                           localization_rank_limit=2),
        tensors_sha256=hashlib.sha256(raw).hexdigest(), isotropic_pairs=pairs,
        bj_positive_eligible=True, nonpositive_C6_C8_pairs=[],
        pfit_native_verified=[False]*11,
        diagnostics=[dict(frequency=float(i), response_residual=0., localization_residual=0.)
                     for i in range(11)],
        provenance={'partition': {'model': 'isa', 'converged': True}},
        resources=dict(maximum_numeric_plan=1000, charged_work=1000, charged_io_bytes=1000))
    return dict(schema='native-monomer-artifacts/2', request=request, result=result), raw


def test_reduction_matches_hand_worked_c6_c8_partial_c10_without_certifying_solver():
    envelope, raw = example()
    report = validator.validate_numeric(envelope, raw)
    assert report['status'] == 'PASS_NUMERIC_TRANSPORT_ONLY'
    assert report['isotropic_entries'] == 27
    assert report['maximum_cn_absolute_difference'] == 0.
    assert report['static_alpha1_sum_au'] == 3.
    assert report['static_alpha2_sum_au'] == 6.
    assert report['independent_solver_certified'] is False
    assert report['full_C10'] is False


@pytest.mark.parametrize('changed', ['coefficient', 'coverage', 'C10', 'residual', 'budget', 'checksum'])
def test_corrupted_or_incomplete_export_is_not_qualified(changed):
    envelope, raw = example()
    result = envelope['result']
    if changed == 'coefficient':
        result['isotropic_pairs'][0]['value'] += 1.
    elif changed == 'coverage':
        result['isotropic_pairs'].pop()
    elif changed == 'C10':
        result['isotropic_pairs'][2]['unrestricted_complete'] = True
    elif changed == 'residual':
        result['diagnostics'][0]['localization_residual'] = 1e-3
    elif changed == 'budget':
        result['resources']['charged_work'] = 10**9+1
    else:
        raw += b'changed'
    with pytest.raises(ValueError):
        validator.validate_numeric(envelope, raw)


def test_object_tensors_cannot_load_pickled_data():
    stream = io.BytesIO()
    np.savez_compressed(stream, local=np.zeros((11, 3, 15, 15), dtype=object),
                        refined=np.zeros((11, 3, 9, 9)))
    with pytest.raises(ValueError, match='float64'):
        validator.tensor_arrays(stream.getvalue(), 3)


def test_forged_huge_npy_shape_is_rejected_before_allocation():
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, dict(descr='<f8', fortran_order=False, shape=(11, 3, 2**60, 2**60)))
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for name in ('local.npy', 'refined.npy'):
            archive.writestr(name, header.getvalue())
    with pytest.raises(ValueError, match='float64'):
        validator.tensor_arrays(stream.getvalue(), 3)
