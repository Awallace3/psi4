# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Installed-only, own-SCF water acceptance and attributable numerical export.

This computes a declared model; it imports no test/oracle/research helper and
never resumes molecular products. The build manifest is provenance, not
scientific input. A smoke profile is not C8/C10 or convergence evidence.
"""
import argparse
from dataclasses import asdict, is_dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import psi4
from psi4 import core

from .isapol_bounded_response import BoundedResources
from .isapol_logging import StageLog
from .isapol_native import Quadrature
from .isapol_recipes import native_water_recipe


def _plain(value):
    if is_dataclass(value):
        return {key: _plain(v) for key, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _plain(v) for key, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _isotropic_export(dispersion, *, max_order):
    labels = dispersion.labels
    if not labels or len(set(labels)) != len(labels):
        raise RuntimeError('nonempty unique isotropic site labels required')
    expected = {(a, b, n) for a in range(len(labels)) for b in range(len(labels))
                for n in range(6, max_order+1, 2)}
    actual = [(p.site_a, p.site_b, c.order)
              for p in dispersion.pairs for c in p.coefficients]
    if len(actual) != len(expected) or set(actual) != expected:
        raise RuntimeError('incomplete or duplicate isotropic site-pair/order coverage')
    return [
        dict(site_a=labels[p.site_a], site_b=labels[p.site_b],
             order=c.order, value=c.value,
             unrestricted_complete=c.unrestricted_complete,
             included_rank_pairs=c.included_rank_pairs,
             missing_rank_pairs=c.missing_rank_pairs)
        for p in dispersion.pairs for c in p.coefficients
    ]


def _write_tensor_artifact(path, result, *, localization_rank_limit):
    """Export numeric arrays, preserving the different LW/PFIT rank layouts."""
    local = np.concatenate([snapshot.array for snapshot in result.local_tensors], axis=0)
    refined = np.asarray(result.refined_tensors)
    nf, ns = len(result.frequencies), len(result.refinements[0].model.sites)
    if type(localization_rank_limit) is not int or localization_rank_limit not in (1, 2, 3, 4):
        raise RuntimeError('explicit native localization rank required')
    packing_rank = 4 if localization_rank_limit == 4 else 3
    width = (packing_rank+1)**2-1
    if (local.ndim != 4 or local.shape[:2] != (nf, ns)
            or local.shape[2:] != (width, width)
            or refined.ndim != 4 or refined.shape[:2] != (nf, ns)
            or refined.shape[2] != refined.shape[3]):
        raise RuntimeError(f'unexpected native tensor export dimensions: local={local.shape}, refined={refined.shape}, packed_width={width}')
    for name, array in (('local', local), ('refined', refined)):
        if array.dtype != np.float64 or not np.isfinite(array).all():
            raise RuntimeError(f'{name} tensor export requires finite float64 arrays')
    keep = (localization_rank_limit+1)**2-1
    if np.any(local[..., keep:, :]) or np.any(local[..., :, keep:]):
        raise RuntimeError('native local tensors contain response above declared localization rank')
    np.savez(path, local=local, refined=refined)
    with np.load(path, allow_pickle=False) as saved:
        for name, original in (('local', local), ('refined', refined)):
            if not np.array_equal(saved[name], original):
                raise RuntimeError(f'{name} tensor export roundtrip differs')
    return dict(
        axes='frequency,site,response_component,potential_component',
        local_components=f'packed positive ranks1..{packing_rank}; 10,11c,11s,...; no00',
        localization_rank_limit=localization_rank_limit,
        packing_rank=packing_rank,
        uncomputed_packing_components='Above localization rank: structural zeros, not computed response',
        refined_components='includes00; declared per-site rank; 00,10,11c,11s,...',
        frame='local-axis real Racah/no-CS',
        units='atomic units', dtype='float64',
        local_shape=local.shape, refined_shape=refined.shape,
    )


def _angular_export(angular):
    """Keep table, declared-model and unrestricted completeness distinct."""
    return dict(
        schema='native-recoupled-dispersion/1',
        convention=angular.convention, charge_policy=angular.charge_policy,
        component_indices='t/u: one-based real components00,10,11c,11s,...; J: angular momentum',
        units='atomic units', frequencies=angular.frequencies,
        cp_weights=angular.cp_weights, max_order=angular.max_order,
        cp_weight_convention='includes1/(2*pi); static node weight zero',
        pair_labels=angular.pair_labels,
        declared_model_coverage=_plain(angular.declared_model_coverage),
        quadrature_provenance=_plain(angular.quadrature_provenance),
        provenance=angular.provenance,
        pairs=[
            dict(site_a=p.site_a, site_b=p.site_b,
                 coefficients=[dict(order=c.order, t=c.t, u=c.u, J=c.J, value=c.value)
                               for c in p.coefficients],
                 coverage=[
                     dict(order=c.order, table_complete=c.table_complete,
                          unrestricted_complete=c.unrestricted_complete,
                          included_rank_quadruples=c.included_rank_quadruples,
                          missing_table_rank_quadruples=c.missing_table_rank_quadruples,
                          missing_unrestricted_rank_quadruples=c.missing_unrestricted_rank_quadruples)
                     for c in p.coverage])
            for p in angular.pairs],
    )


def _require_angular_coverage(angular, *, nsite, max_order):
    if angular is None:
        raise RuntimeError('rank2 acceptance requires native recoupled dispersion')
    expected = {(a, b, n) for a in range(nsite) for b in range(nsite)
                for n in range(6, max_order+1)}
    actual = [(p.site_a, p.site_b, c.order) for p in angular.pairs for c in p.coverage]
    declared = [(c.site_a, c.site_b, c.order) for c in angular.declared_model_coverage]
    if (len(actual) != len(expected) or set(actual) != expected
            or len(declared) != len(expected) or set(declared) != expected
            or any(c.table_complete != (not c.missing_table_rank_quadruples)
                   for p in angular.pairs for c in p.coverage)
            or any(not c.declared_model_complete or c.missing_declared_rank_quadruples
                   for c in angular.declared_model_coverage)
            or any(not p.coefficients for p in angular.pairs)):
        raise RuntimeError('incomplete native angular table/declared-model coverage')
    # Full-table and unrestricted completeness are NOT required: tables also
    # contain ranks outside this explicitly truncated rank2 model.


def _verify_build(manifest):
    """Require the installed module/core bytes recorded by the fresh build."""
    if not manifest.get('clean_source') or not manifest.get('fresh_build'):
        raise ValueError('clean-source fresh-build manifest required')
    installed = Path(manifest['installed_package']).resolve()
    if Path(psi4.__file__).resolve().parent != installed:
        raise ValueError('Psi4 import is not the declared installed package')
    producer = Path(__file__).resolve()
    if (not producer.is_relative_to(installed)
            or manifest['installed_files'].get(producer.relative_to(installed).as_posix()) != _sha(producer)):
        raise ValueError('acceptance producer differs from installed build manifest')
    actual = {}
    for name, module in tuple(sys.modules.items()):
        if name != 'psi4' and not name.startswith('psi4.'):
            continue
        filename = getattr(module, '__file__', None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(installed):
            raise ValueError(f'noninstalled Psi4 module: {name}: {path}')
        key = path.relative_to(installed).as_posix()
        digest = _sha(path)
        if manifest['installed_files'].get(key) != digest:
            raise ValueError(f'installed file differs from build manifest: {key}')
        actual[name] = dict(path=str(path), sha256=digest)
    return actual


def run_water(output, manifest, *, profile='smoke', forbidden_prefixes=()):
    """Run fresh PBE0/aug-cc-pVTZ SCF and the explicit native ISA chain."""
    if profile not in ('smoke', 'rank2'):
        raise ValueError('profile must be smoke or rank2')
    _verify_build(manifest)
    # The launcher supplies canonical prefixes. Normalizing this declaration
    # must not itself stat/read the forbidden research directories.
    prefixes = tuple(os.path.abspath(p) for p in forbidden_prefixes)
    reads, processes = [], []

    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = str(Path(os.fsdecode(args[0])).resolve())
            if any(path == p or path.startswith(p + '/') for p in prefixes):
                raise PermissionError(f'forbidden research/source input: {path}')
            reads.append(path)
        if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn'):
            processes.append(event)
            raise PermissionError('acceptance does not permit child executables')

    sys.addaudithook(audit)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / 'checkpoints'
    scratch.mkdir()
    core.clean_options()
    psi4.set_num_threads(1)
    psi4.set_memory('4 GiB')
    core.set_output_file(str(output / 'psi4.out'), False)
    molecule = psi4.geometry('''0 1
O 0 0 0
H -1.45365196 0 -1.12168732
H 1.45365196 0 -1.12168732
units bohr
symmetry c1
no_com
no_reorient
''')
    psi4.set_options(dict(basis='aug-cc-pvtz', puream=True, reference='rks',
                         scf_type='pk', e_convergence=1e-12, d_convergence=1e-12))
    energy, wfn = psi4.energy('pbe0', molecule=molecule, return_wfn=True)
    recipe = native_water_recipe(molecule)
    options = core.IsaGridOptions()
    options.radial_points, options.spherical_points = 200, 590
    grid = core.IsaGrid(molecule.clone(), options)
    integration = np.column_stack((grid.x(), grid.y(), grid.z(), grid.w()))
    rank = 1 if profile == 'smoke' else 2
    quadrature = Quadrature.from_casimir(core.CasimirGrid(2 if profile == 'smoke' else 10, .5))
    psi4.oeprop(
        wfn, 'ATOMIC_REFINED_DISPERSION', atomic_backend='BOUNDED_DF',
        preset='water', distribution='isa', partition_recipe=recipe,
        partition_grid=integration, auxiliary_recipe=recipe.auxiliary,
        quadrature=quadrature, npoints=64 if profile == 'smoke' else 2000,
        rank_limit=rank, hydrogen_rank_limit=rank, localization_rank_limit=2,
        weight_type=4 if profile == 'smoke' else 3,
        weight_coefficient=1e-5 if profile == 'smoke' else 1e-3,
        max_order=6 if profile == 'smoke' else 10, scf_correction='NONE',
        resources=BoundedResources(4*1024**3, 6_000_000_000_000, 64*1024**3),
        scratch_directory=scratch, recoupled_dispersion=profile == 'rank2',
        log=StageLog(1),
    )
    result = psi4.atomic_property_result(wfn)
    if (tuple(result.frequencies) != tuple(quadrature.frequencies)
            or len(result.refinements) != len(quadrature.frequencies)
            or len(result.diagnostics) != len(quadrature.frequencies)):
        raise RuntimeError('native result does not cover every declared node')
    if any(r.status != core.IsaPfitStatus.Solved for r in result.refinements):
        raise RuntimeError('unsolved native fit')
    if list(scratch.iterdir()):
        raise RuntimeError('private response checkpoints not cleaned')
    if max(d['response_residual'] for d in result.diagnostics) >= 1e-10:
        raise RuntimeError('native response residual acceptance failed')
    if max(d['localization_residual'] for d in result.diagnostics) >= 1e-6:
        raise RuntimeError('native localization residual acceptance failed')
    max_order = 6 if profile == 'smoke' else 10
    pairs = _isotropic_export(result.dispersion, max_order=max_order)
    if any(not np.isfinite(p['value']) or p['value'] <= 0 for p in pairs):
        raise RuntimeError('invalid isotropic coefficient')
    forbidden_imports = [name for name in sys.modules if
                         name.endswith('isapol_supplied') or 'isapol_factorial_oracle' in name
                         or name == 'isapol_water_recipe']
    if forbidden_imports:
        raise RuntimeError(f'test/oracle/supplied imports: {forbidden_imports}')
    if profile == 'rank2':
        _require_angular_coverage(result.recoupled_dispersion,
                                  nsite=len(result.dispersion.labels), max_order=max_order)
    installed = _verify_build(manifest)
    record = dict(
        schema='installed-native-isa-water/1', status='complete', source_commit=manifest['commit'],
        source_tree=manifest['tree'], profile=profile, branch_qualified=True,
        own_scf=True, own_partition=True, own_response=True, own_pfit=True,
        method='PBE0/aug-cc-pVTZ', energy_Eh=energy, frequencies=result.frequencies,
        cp_weights=result.dispersion.cp_weights, sites=_plain(result.refinements[0].model.sites),
        isotropic_pairs=pairs, diagnostics=result.diagnostics, resources=result.resources,
        provenance=result.provenance, installed_runtime=installed, child_executables=processes,
        forbidden_read_prefixes=prefixes, python_file_read_count=len(reads),
        file_access_qualification='Python audit hook; native-library file reads require independent OS tracing',
        model_qualification='Declared regression ISA recipe with absolute1.5-bohr tails; property-grid convergence not established',
        max_order=6 if profile == 'smoke' else 10,
    )
    record['tensor_layout'] = _write_tensor_artifact(output / 'tensors.npz', result,
                                                    localization_rank_limit=2)
    if result.recoupled_dispersion is not None:
        with gzip.open(output / 'angular.json.gz', 'wt') as f:
            json.dump(_plain(_angular_export(result.recoupled_dispersion)), f, allow_nan=False)
        record['angular_sha256'] = _sha(output / 'angular.json.gz')
    record['tensors_sha256'] = _sha(output / 'tensors.npz')
    (output / 'file-access.json').write_text(json.dumps(reads, indent=2))
    (output / 'result.json').write_text(json.dumps(_plain(record), indent=2, allow_nan=False))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--build-manifest', required=True)
    parser.add_argument('--profile', choices=('smoke', 'rank2'), default='smoke')
    parser.add_argument('--forbid-read-prefix', action='append', default=[])
    args = parser.parse_args()
    manifest = json.loads(Path(args.build_manifest).read_text())
    result = run_water(args.output, manifest, profile=args.profile,
                       forbidden_prefixes=args.forbid_read_prefix)
    print('NATIVE_INSTALLED_ACCEPTANCE', result['status'], result['source_commit'], result['profile'])


if __name__ == '__main__':
    main()
