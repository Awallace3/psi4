# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Phoenix-only own-SCF benzene acceptance using the installed native chain.

Only a separately attributed geometry is an external scientific input. This
does not load prior orbitals, density, partition, tensor or fit products.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys

import numpy as np
import psi4
from psi4 import core
from psi4.driver.p4util import OptionsState

from .isapol_acceptance import (
    _angular_export, _isotropic_export, _plain, _require_angular_coverage,
    _sha, _verify_build, _write_tensor_artifact,
)
from .isapol_bounded_response import BoundedResources
from .isapol_logging import StageLog
from .isapol_native import Quadrature
from .isapol_native_propagator import KernelSmoothing
from .isapol_recipes import benzene_declared_variables, native_benzene_recipe


def _compute_allocation():
    host = socket.gethostname().split('.')[0]
    node = os.environ.get('SLURMD_NODENAME', '').split('.')[0]
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('USER') != 'awallace43'
            or not node or node != host or host.startswith('login')
            or re.fullmatch(r'atl\d+(?:-\d+){5}', host) is None
            or os.environ.get('SLURM_CLUSTER_NAME') != 'phoenix-slurm'
            or not os.environ.get('HOME', '').startswith('/storage/home/')):
        raise RuntimeError('benzene acceptance requires a Phoenix compute allocation')
    return dict(job_id=os.environ['SLURM_JOB_ID'], host=host,
                cluster=os.environ['SLURM_CLUSTER_NAME'],
                qualification='Accidental-local-execution guard; operational launcher must independently verify active allocation and authorized QoS.')


def _geometry(record):
    if (record.get('schema') != 'native-benzene-geometry/1'
            or record.get('units') != 'bohr'
            or record.get('symbols') != ['C']*6+['H']*6
            or record.get('charge') != 0 or record.get('multiplicity') != 1
            or not record.get('source') or not record.get('source_sha256')):
        raise ValueError('explicit attributed neutral benzene geometry in bohr required')
    xyz = np.asarray(record.get('geometry'), dtype=np.float64)
    if xyz.shape != (12, 3) or not np.isfinite(xyz).all():
        raise ValueError('finite12x3 benzene geometry required')
    return xyz


def _basis_identity(basis):
    record = dict(nbf=basis.nbf(), nao=basis.nao(), puream=basis.has_puream(),
                  centres=np.asarray(basis.molecule().geometry()).tolist(),
                  shells=[
                      dict(centre=int(basis.shell_to_center(i)), l=int(basis.shell(i).am),
                           pure=bool(basis.shell(i).is_pure()),
                           exponents=[basis.shell(i).exp(k) for k in range(basis.shell(i).nprimitive)],
                           coefficients=[basis.shell(i).coef(k) for k in range(basis.shell(i).nprimitive)])
                      for i in range(basis.nshell())])
    record['sha256'] = hashlib.sha256(
        json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return record


def _model_inventory(result, declared, *, expected_sites):
    """Certify actual72-column reference/COPY inventory, not a metadata count."""
    inventory = []
    if len(declared) != 72 or len(set(declared)) != 72:
        raise RuntimeError('complete72-variable declaration required')
    from .isapol_refine import COMPONENT_NAMES
    labels = tuple(f'{kind}{i+1}' for kind in ('C', 'H') for i in range(6))
    expected_entries = tuple(
        tuple((s, row, col) for s in range(offset, offset+6))
        for offset in (0, 6) for row in range(1, 9) for col in range(row, 9))
    expected_channels = tuple(f'{label}_{component}' for label in labels
                              for component in COMPONENT_NAMES[:9])
    frequencies = tuple(result.frequencies)
    if (len(frequencies) != 11 or frequencies[0] != 0
            or not np.isfinite(frequencies).all() or np.any(np.diff(frequencies) <= 0)
            or len(result.local_tensors) != 11 or len(result.refinements) != 11):
        raise RuntimeError('benzene inventory requires every static/dynamic node')
    tensors = []
    for snapshot in result.local_tensors:
        local = snapshot.array
        if local.shape != (1, 12, 15, 15) or not np.isfinite(local).all():
            raise RuntimeError('invalid native local anchor tensor inventory')
        full = np.zeros((12, 9, 9))
        full[:, 1:, 1:] = local[0, :, :8, :8]
        tensors.append(full)
    def anchor_hash(array):
        return hashlib.sha256(np.ascontiguousarray(array, dtype='<f8').tobytes()).hexdigest()
    static_hash = anchor_hash(tensors[0])
    selected = tuple(abs(float(tensors[0][entries[0]])) > .0001 for entries in expected_entries)
    free = tuple(label for label, penalized in zip(declared, selected) if not penalized)
    for index, refined in enumerate(result.refinements):
        model = refined.model
        if (model.sites != tuple(expected_sites)
                or tuple(s.label for s in model.sites) != labels
                or tuple(s.site_type for s in model.sites) != ('C',)*6+('H',)*6
                or any(s.rank_limit != 2 for s in model.sites)
                or model.reference_sites != (0, 6)
                or model.equivalent_sites != (tuple(range(6)), tuple(range(6, 12)))
                or model.channel_count != 108 or model.channel_labels != expected_channels
                or model.channel_offsets != tuple(range(0, 108, 9))):
            raise RuntimeError('incorrect benzene site/reference/COPY/channel inventory')
        if (model.parameter_count != 72 or tuple(model.parameter_labels) != declared
                or model.parameter_entries != expected_entries
                or model.declared_variables != declared
                or model.penalty_selection != 'static_reference'
                or model.static_reference_sha256 != static_hash
                or model.anchor_sha256 != anchor_hash(tensors[index])
                or model.selection_sha256 != '' or model.unpenalized_variables != free
                or model.frequency_au != frequencies[index]
                or model.weight_type != 3 or model.weight_coefficient != .001
                or model.cutoff != .0001):
            raise RuntimeError('inconsistent benzene reference/COPY/static penalty inventory')
        expected_anchors, expected_strengths = [], []
        discrepancy = 0.
        parameters = np.asarray(refined.parameters)
        fitted = np.asarray(refined.refined_tensors)
        if (parameters.shape != (72,) or fitted.shape != (12, 9, 9)
                or not np.isfinite(parameters).all() or not np.isfinite(fitted).all()
                or np.any(fitted[:, 0, :]) or np.any(fitted[:, :, 0])):
            raise RuntimeError('invalid fitted parameter/COPY tensor inventory')
        for k, entries in enumerate(expected_entries):
            alpha = float(tensors[index][entries[0]])
            expected_anchors.append(alpha if selected[k] else 0.)
            weight = .001/(alpha*alpha+1) if selected[k] else 0.
            if frequencies[index] != 0.:
                weight /= 1.+frequencies[index]*frequencies[index]
            expected_strengths.append(weight)
            for s, row, col in entries:
                if fitted[s, row, col] != parameters[k] or fitted[s, col, row] != parameters[k]:
                    raise RuntimeError('fitted COPY tensor does not reproduce declared parameters')
                if selected[k]:
                    discrepancy = max(discrepancy, abs(float(tensors[index][s, row, col])-alpha))
        if (model.anchors != tuple(expected_anchors) or model.strengths != tuple(expected_strengths)
                or model.copy_anchor_discrepancy != discrepancy):
            raise RuntimeError('benzene current-node anchors/strengths/COPY discrepancy mismatch')
        diagnostic_fields = (
            'data_rows', 'augmented_rows', 'numerical_rank', 'free_indices',
            'data_sse', 'data_rms', 'data_max_residual', 'matrix_objective',
            'lc_objective', 'total_objective', 'stationarity_inf', 'backward_residual',
            'normal_h_rcond', 'qr_r_rcond', 'condition_estimate_available',
            'lapack_info', 'rank_method', 'objective_available', 'native_verified')
        inventory.append(dict(
            frequency_au=model.frequency_au, labels=model.parameter_labels,
            entries=model.parameter_entries, declared=model.declared_variables,
            unpenalized_variables=model.unpenalized_variables,
            anchors=model.anchors, strengths=model.strengths,
            copy_anchor_discrepancy=model.copy_anchor_discrepancy,
            fit_status=str(refined.status), parameters=refined.parameters,
            anchor_shift_maxabs=refined.anchor_shift_maxabs,
            fit_diagnostics={name: getattr(refined.diagnostics, name) for name in diagnostic_fields},
            static_reference_sha256=model.static_reference_sha256,
            selection_sha256=model.selection_sha256, anchor_sha256=model.anchor_sha256))
    return _plain(inventory)


def run_benzene(output, manifest, geometry, *, geometry_sha256, forbidden_prefixes=()):
    allocation = _compute_allocation()  # before directories or native scientific work
    _verify_build(manifest)
    installed = Path(manifest['installed_package']).resolve()
    producer = Path(__file__).resolve()
    if (not producer.is_relative_to(installed)
            or manifest['installed_files'].get(producer.relative_to(installed).as_posix()) != _sha(producer)):
        raise ValueError('benzene acceptance producer differs from installed build manifest')
    xyz = _geometry(geometry)
    prefixes = tuple(os.path.abspath(p) for p in forbidden_prefixes)
    reads, processes = [], []

    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = str(Path(os.fsdecode(args[0])).resolve())
            if any(path == p or path.startswith(p+'/') for p in prefixes):
                raise PermissionError('forbidden research/source input: '+path)
            reads.append(path)
        if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn'):
            processes.append(event)
            raise PermissionError('benzene acceptance forbids child executables')

    sys.addaudithook(audit)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / 'checkpoints'
    scratch.mkdir()
    memory = 25*1024**3
    core.clean_options()
    psi4.set_num_threads(1)
    psi4.set_memory(memory)
    core.set_output_file(str(output/'psi4.out'), False)
    lines = ['0 1', 'units bohr', 'symmetry c1', 'no_com', 'no_reorient']
    lines += [symbol+' '+' '.join(format(float(v), '.17g') for v in point)
              for symbol, point in zip(geometry['symbols'], xyz)]
    molecule = psi4.geometry('\n'.join(lines))
    if not np.array_equal(np.asarray(molecule.geometry()), xyz):
        raise RuntimeError('geometry/frame changed during molecule construction')
    psi4.set_options(dict(
        basis='aug-cc-pvtz', puream=True, reference='rks', scf_type='pk',
        e_convergence=1e-11, d_convergence=1e-10, ints_tolerance=1e-10,
        s_tolerance=1e-7, soscf=False, dft_radial_points=75, dft_spherical_points=302))
    main = core.BasisSet.build(molecule, 'BASIS', 'aug-cc-pVTZ', puream=1)
    if main.nbf() != 414 or not main.has_puream():
        raise RuntimeError('actual414-function spherical MAIN required')
    admitted_main = _basis_identity(main)
    recipe = native_benzene_recipe(molecule)
    stash = OptionsState(['PUREAM'])
    try:
        core.set_global_option('PUREAM', False)
        auxiliary = core.BasisSet.build(molecule, 'DF_BASIS_SCF', 'aug-cc-pVTZ-RI', puream=0)
    finally:
        stash.restore()
    admitted_auxiliary = _basis_identity(auxiliary)
    if auxiliary.nbf() != 1146 or auxiliary.has_puream():
        raise RuntimeError('actual1146-function Cartesian AUX required')
    shells = recipe.auxiliary.shells
    if (len(shells) != auxiliary.nshell()
            or any((s.centre, s.l, s.exponents, s.coefficients) != (
                int(auxiliary.shell_to_center(i)), int(auxiliary.shell(i).am),
                tuple(auxiliary.shell(i).exp(k) for k in range(auxiliary.shell(i).nprimitive)),
                tuple(auxiliary.shell(i).coef(k) for k in range(auxiliary.shell(i).nprimitive)))
                   for i, s in enumerate(shells))):
        raise RuntimeError('recipe does not describe admitted current AUX basis')
    energy, wfn = psi4.energy('pbe0', molecule=molecule, return_wfn=True)
    if _basis_identity(wfn.basisset()) != admitted_main:
        raise RuntimeError('SCF MAIN differs from admitted basis')
    options = core.IsaGridOptions()
    options.radial_points, options.spherical_points = 200, 974
    grid = core.IsaGrid(molecule.clone(), options)
    integration = np.column_stack((grid.x(), grid.y(), grid.z(), grid.w()))
    quadrature = Quadrature.from_casimir(core.CasimirGrid(10, .5))
    declared = benzene_declared_variables()
    psi4.oeprop(
        wfn, 'ATOMIC_REFINED_DISPERSION', atomic_backend='BOUNDED_DF',
        preset='benzene', distribution='isa', partition_recipe=recipe,
        partition_grid=integration, auxiliary_recipe=recipe.auxiliary,
        response='reference_h2h1', quadrature=quadrature,
        radial_points=100, spherical_points=200,
        smoothing=KernelSmoothing(1e-8, 600., .01, 1., 'FD'),
        shell_cutoff=1e-8, charge_penalty=1000., anchor_metric_damping=.0005,
        npoints=2000, seed=1, lower_limit=2., upper_limit=4.,
        rank_limit=2, hydrogen_rank_limit=2, localization_rank_limit=2,
        weight_type=3, weight_coefficient=.001, cutoff=.0001,
        declared_variables=declared, max_refinement_parameters=72,
        max_order=10, scf_correction='NONE', recoupled_dispersion=True,
        resources=BoundedResources(memory, 64_000_000_000_000, 64*1024**3, capacity='large'),
        scratch_directory=scratch, log=StageLog(1))
    result = psi4.atomic_property_result(wfn)
    if (tuple(result.frequencies) != tuple(quadrature.frequencies)
            or len(result.refinements) != 11 or len(result.diagnostics) != 11
            or any(r.status != core.IsaPfitStatus.Solved for r in result.refinements)
            or list(scratch.iterdir())):
        raise RuntimeError('incomplete/unsolved benzene sweep or retained checkpoints')
    if (any(not np.isfinite(d['response_residual']) or d['response_residual'] >= 1e-10
            or not np.isfinite(d['localization_residual']) or d['localization_residual'] >= 1e-6
            for d in result.diagnostics)):
        raise RuntimeError('benzene response/localization residual acceptance failed')
    pairs = _isotropic_export(result.dispersion, max_order=10)
    if len(pairs) != 432 or any(not np.isfinite(p['value']) or p['value'] <= 0 for p in pairs):
        raise RuntimeError('invalid/incomplete benzene isotropic coefficients')
    _require_angular_coverage(result.recoupled_dispersion, nsite=12, max_order=10)
    from .isapol_bounded_oeprop import preset_sites
    expected_sites, _ = preset_sites(molecule, 'benzene', 2, 2)
    inventory = _model_inventory(result, declared, expected_sites=expected_sites)
    if any(n.endswith('isapol_supplied') or 'factorial_oracle' in n
           or n == 'isapol_water_recipe' for n in sys.modules):
        raise RuntimeError('test/oracle/supplied helper imported')
    layout = _write_tensor_artifact(output/'tensors.npz', result, localization_rank_limit=2)
    if layout['local_shape'] != (11, 12, 15, 15) or layout['refined_shape'] != (11, 12, 9, 9):
        raise RuntimeError('incorrect benzene tensor inventory')
    with gzip.open(output/'angular.json.gz', 'wt') as stream:
        json.dump(_plain(_angular_export(result.recoupled_dispersion)), stream, allow_nan=False)
    record = dict(
        schema='installed-native-isa-benzene/1', status='complete', profile='rank2',
        source_commit=manifest['commit'], source_tree=manifest['tree'], branch_qualified=True,
        own_scf=True, own_partition=True, own_response=True, own_pfit=True,
        method='PBE0/aug-cc-pVTZ', energy_Eh=energy, allocation=allocation,
        geometry_input=geometry, geometry_sha256=geometry_sha256,
        main_basis=admitted_main, auxiliary_basis=admitted_auxiliary,
        frequencies=result.frequencies, cp_weights=result.dispersion.cp_weights,
        sites=_plain(result.refinements[0].model.sites), isotropic_pairs=pairs,
        refinement_inventory=inventory, diagnostics=result.diagnostics,
        resources=result.resources, provenance=result.provenance,
        tensor_layout=layout, tensors_sha256=_sha(output/'tensors.npz'),
        angular_sha256=_sha(output/'angular.json.gz'),
        installed_runtime=_verify_build(manifest), child_executables=processes,
        forbidden_read_prefixes=prefixes, python_file_read_count=len(reads),
        model_qualification='Declared native benzene rank2 model;1.5x Bragg-Slater tails; property-grid convergence unestablished; C10 partial(2,2), not parity or an interaction energy.')
    (output/'file-access.json').write_text(json.dumps(reads, indent=2))
    (output/'result.json').write_text(json.dumps(_plain(record), indent=2, allow_nan=False))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--build-manifest', required=True)
    parser.add_argument('--geometry', required=True)
    parser.add_argument('--geometry-sha256', required=True)
    parser.add_argument('--forbid-read-prefix', action='append', default=[])
    args = parser.parse_args()
    _compute_allocation()
    geometry_bytes = Path(args.geometry).read_bytes()
    if hashlib.sha256(geometry_bytes).hexdigest() != args.geometry_sha256:
        raise ValueError('geometry input hash mismatch')
    result = run_benzene(
        args.output, json.loads(Path(args.build_manifest).read_text()),
        json.loads(geometry_bytes), geometry_sha256=args.geometry_sha256,
        forbidden_prefixes=args.forbid_read_prefix)
    print('NATIVE_INSTALLED_BENZENE_ACCEPTANCE', result['status'], result['source_commit'])


if __name__ == '__main__':
    main()
