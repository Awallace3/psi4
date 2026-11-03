"""Controlled AUX-charge quadrature study; frozen-response projections, not models."""
import argparse
from dataclasses import asdict
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import time

BLOCK_POINTS = 4096
MAX_POINTS = 10_000_000
MAX_AUX = 1065
GRIDS = ((200, 974), (300, 1454), (400, 2030))
CORE_MEMORY_BYTES = 40*1024**3
JOB_MEMORY_BYTES = 64*1024**3
STUDY_RUN_NAME = 'ccasp-native-s66-ed561666fc-grid-study002'
FROZEN_REPORT_SHA256 = 'de13107c3d96f824e0303648f2c81d81b0517a52aab9ddf5c413571a9f101af5'
FROZEN_RECEIPT_SHA256 = 'd46c1c683e6842668b1b2bfd3d3a4557098453b031348b3fc9f8e6a2d09b8a2d'
FROZEN_OPERATIONAL_COMMIT = '1a63292f319c9f21ff5ce385bce6c0ea81f4db99'


def operations():
    """Share existing reviewed operational guards, never overlay scientific code."""
    name = '_grid_charge_common_operations'
    if name not in sys.modules:
        path = Path(__file__).with_name('charge_flow_diagnostic_ed561666.py')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return sys.modules[name]


def frozen_receipts(payload):
    common = operations()
    report_raw = (payload/'charge-diagnostic.json').read_bytes()
    receipt_raw = (payload/'diagnostic-receipt.json').read_bytes()
    if (common.sha(report_raw) != FROZEN_REPORT_SHA256
            or common.sha(receipt_raw) != FROZEN_RECEIPT_SHA256):
        raise ValueError('exact independently verified original frozen diagnostic required')
    report, receipt = json.loads(report_raw), json.loads(receipt_raw)
    if (receipt['report_sha256'] != FROZEN_REPORT_SHA256
            or report['operational_commit'] != FROZEN_OPERATIONAL_COMMIT
            or report['runtime_input_sha256'] != common.RUNTIME_SHA256
            or report['production_tolerance'] != 1e-6
            or report['discriminator'] != 'FIRST_NODE_PRODUCTION_LW_REFUSAL_CAPTURED'):
        raise ValueError('frozen diagnostic source/input/refusal binding differs')
    return report, receipt


def load_frozen_context(payload):
    """Immutable owned NPY snapshots; mappings cannot change later projections."""
    report, receipt = frozen_receipts(payload)
    records = operand_records(report)
    arrays = {name: owned_frozen_array(payload/record['file'], record)
              for name, record in records.items()}
    if (arrays['q'].shape != (300, 1065)
            or arrays['anchor_coefficients'].shape != (7420, 1065)
            or arrays['solved_anchor'].shape != (7420, 300)
            or arrays['analytic_charge'].shape != (1065,)
            or report['observation']['localization_rank_limit'] != 2):
        raise ValueError('exact original monomer0008 frozen layouts required')
    return arrays, report, receipt


def operand_records(report):
    records = report['captured_arrays']
    expected = {'q', 'analytic_charge', 'anchor_coefficients', 'solved_anchor',
                'anchor_legs', 'nonlocal_response'}
    if (set(records) != expected or report['captured_numeric_bytes'] > 512*1024**2
            or sum(r['numeric_bytes'] for r in records.values()) != report['captured_numeric_bytes']
            or any(r['file'] != name+'.npy' or not 0 < r['numeric_bytes'] <= 256*1024**2
                   for name, r in records.items())):
        raise ValueError('bounded six-array original capture required')
    return records


def owned_frozen_array(path, record):
    import numpy as np
    maximum = record['numeric_bytes']+65536
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError('bounded regular frozen operand file required')
    with path.open('rb') as stream:
        raw = stream.read(maximum+1)
    # Hash the same immutable bytes decoded, not an earlier read of a mapping.
    if len(raw) > maximum or operations().sha(raw) != record['file_sha256']:
        raise ValueError('frozen operand hash changed')
    array = np.load(io.BytesIO(raw), allow_pickle=False)
    if (array.dtype.str != record['dtype'] or array.dtype.str != '<f8'
            or list(array.shape) != record['shape'] or array.nbytes != record['numeric_bytes']
            or not np.isfinite(array).all()):
        raise ValueError('frozen operand layout differs')
    return np.frombuffer(array.tobytes(order='C'), dtype='<f8').reshape(array.shape)


def verify_frozen_files(payload):
    """Standard-library-only final seal; parent never imports scientific code."""
    report, _ = frozen_receipts(payload)
    common = operations()
    for record in operand_records(report).values():
        path = payload/record['file']
        if (path.is_symlink() or not path.is_file()
                or path.stat().st_size > record['numeric_bytes']+65536
                or common.file_sha(path) != record['file_sha256']):
            raise ValueError('frozen operand changed before completion')
    return {name: record['file_sha256'] for name, record in operand_records(report).items()}


def integrate_grid(points, weights, evaluator, naux, *, block_points=BLOCK_POINTS, details=None):
    """Stream basis collocation; compare binary64 and extended signed reductions."""
    import numpy as np
    points, weights = np.asarray(points), np.asarray(weights)
    if (points.dtype != np.float64 or weights.dtype != np.float64
            or points.ndim != 2 or points.shape[1] != 3
            or weights.shape != (len(points),) or not 0 < len(points) <= MAX_POINTS
            or type(naux) is not int or not 0 < naux <= MAX_AUX
            or type(block_points) is not int or not 1 <= block_points <= BLOCK_POINTS):
        raise ValueError('bounded explicit binary64 grid/AUX layout required')
    if not np.isfinite(points).all() or not np.isfinite(weights).all():
        raise ValueError('finite grid coordinates and signed weights required')
    total64 = np.zeros(naux)
    extended = np.zeros(naux, dtype=np.longdouble)
    reversed_extended = np.zeros(naux, dtype=np.longdouble)
    absolute = np.zeros(naux, dtype=np.longdouble)
    for begin in range(0, len(points), block_points):
        end = min(len(points), begin+block_points)
        values = np.asarray(evaluator(points[begin:end]))
        if (values.dtype != np.float64 or values.shape != (end-begin, naux)
                or not np.isfinite(values).all()):
            raise ValueError('finite complete response-AUX collocation block required')
        total64 += np.sum(weights[begin:end, None]*values, axis=0)
        terms = weights[begin:end, None].astype(np.longdouble)*values.astype(np.longdouble)
        extended += np.sum(terms, axis=0)
        reversed_extended += np.sum(terms[::-1], axis=0)
        absolute += np.sum(np.abs(terms), axis=0)
    if not np.isfinite(total64).all() or not np.isfinite(extended).all():
        raise ValueError('nonfinite charge quadrature reduction')
    if details is not None:
        details.update(longdouble_mantissa_bits=int(np.finfo(np.longdouble).nmant),
            within_block_reversed_extended_difference=float(np.max(np.abs(extended-reversed_extended))),
            reduction_order='forward blocks; forward versus reversed points within each block',
            absolute_sum=np.asarray(absolute, dtype=np.float64))
    return total64, np.asarray(extended, dtype=np.float64), float(np.max(np.abs(extended-total64)))


def frozen_projection(q_charge, analytic_charge, coefficients, solved_anchor, rank_limit):
    """Apply the saved response operator only; do not solve or change a response."""
    import numpy as np
    q_charge, analytic_charge, coefficients, solved_anchor = map(
        np.asarray, (q_charge, analytic_charge, coefficients, solved_anchor))
    if (q_charge.dtype != np.float64 or analytic_charge.dtype != np.float64
            or coefficients.dtype != np.float64 or solved_anchor.dtype != np.float64
            or q_charge.ndim != 1 or analytic_charge.shape != q_charge.shape
            or coefficients.ndim != 2 or coefficients.shape[1] != len(q_charge)
            or solved_anchor.ndim != 2 or solved_anchor.shape[0] != coefficients.shape[0]
            or solved_anchor.shape[1] % 25 or type(rank_limit) is not int
            or not 1 <= rank_limit <= 4
            or not all(np.isfinite(a).all() for a in (
                q_charge, analytic_charge, coefficients, solved_anchor))):
        raise ValueError('finite matching frozen binary64 response layouts required')
    fit = -solved_anchor.T @ (coefficients @ analytic_charge)
    integration = -solved_anchor.T @ (coefficients @ (q_charge-analytic_charge))
    width = (rank_limit+1)**2
    retained = (fit+integration).reshape(-1, 25)[:, :width]
    site, component = np.unravel_index(np.argmax(np.abs(retained)), retained.shape)
    return dict(q_charge_row_error=float(np.max(np.abs(q_charge-analytic_charge))),
        frozen_transpose_charge_maximum=float(np.max(np.abs(retained))),
        at_maximum=dict(site_index=int(site), component_index=int(component),
            signed_sum=float(retained[site, component]),
            fit=float(fit.reshape(-1, 25)[site, component]),
            integration=float(integration.reshape(-1, 25)[site, component])),
        at_original_failing_component=dict(site_index=6, component_index=6,
            signed_sum=float((fit+integration)[156]) if len(fit) > 156 else None,
            fit=float(fit[156]) if len(fit) > 156 else None,
            integration=float(integration[156]) if len(fit) > 156 else None),
        qualification='Frozen C/X transpose-charge projection only. A changed Q also changes '
            'anchor legs and solved response; this is NOT a new physical response, LW pass '
            'or model acceptance, and does not certify grid convergence.')


def grid_plan(natom, radial, angular):
    if type(natom) is not int or natom != 12 or (radial, angular) not in GRIDS:
        raise ValueError('closed twelve-site three-grid study required')
    points = natom*(radial-1)*angular
    # Native + exported coordinate arrays, owned point matrix, batch copies,
    # long-double products and frozen context; not whole-process/vendor RSS.
    numeric_bytes = 96*points+128*BLOCK_POINTS*MAX_AUX+512*1024**2
    if points > MAX_POINTS or numeric_bytes > 4*1024**3:
        raise ValueError('grid-study preallocation byte/point ceiling exceeded')
    return dict(points=points, planned_numeric_bytes=numeric_bytes, block_points=BLOCK_POINTS)


def admit_study_memory(runtime, *, core_bytes=None, job_bytes=None):
    """Honor the original native admission budget even though no producer runs."""
    core_bytes = CORE_MEMORY_BYTES if core_bytes is None else core_bytes
    job_bytes = JOB_MEMORY_BYTES if job_bytes is None else job_bytes
    required = runtime['specification']['keywords']['function_kwargs']['native_monomer']['resources']['max_bytes']
    if (any(type(value) is not int or value <= 0 for value in (required, core_bytes, job_bytes))
            or required > core_bytes or core_bytes+4*1024**3 > job_bytes):
        raise ValueError('original native admission memory budget and job headroom required')
    return dict(original_request_numeric_bytes=required, core_bytes=core_bytes, job_bytes=job_bytes)


def save_vectors(path, arrays):
    import numpy as np
    if (set(arrays) != {'direct_binary64', 'direct_extended', 'absolute_sum'}
            or any(a.shape != (1065,) or a.dtype != np.float64 or not np.isfinite(a).all()
                   for a in arrays.values())):
        raise ValueError('complete bounded grid charge-vector roles required')
    with path.open('xb') as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    if path.stat().st_size > 65536:
        raise ValueError('grid-vector artifact exceeds declared I/O ceiling')
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(
                not np.array_equal(saved[name], array) for name, array in arrays.items()):
            raise ValueError('grid-vector round trip changed arrays')
    return operations().file_sha(path)


def sample_native_grid(core, molecule, auxiliary, radial, angular):
    """One complete declared grid; own each Matrix view before releasing it."""
    import numpy as np
    planned = grid_plan(molecule.natom(), radial, angular)
    started = time.perf_counter()
    options = core.IsaGridOptions()
    options.radial_points, options.spherical_points = radial, angular
    options.becke_smoothing, options.radius_scaling = 3, 1.
    grid = core.IsaGrid(molecule.clone(), options)
    if (grid.npoints() != planned['points'] or grid.natom() != 12
            or grid.radial_points() != radial or grid.spherical_points() != angular
            or grid.atom_start(12) != planned['points']):
        raise ValueError('actual complete native grid differs from declared point plan')
    points = np.column_stack((grid.x(), grid.y(), grid.z()))
    weights = np.asarray(grid.w())
    def evaluate(block):
        matrix = auxiliary.evaluate(block.tolist())
        return np.array(matrix.np, dtype=np.float64, order='C', copy=True)
    details = {}
    binary64, extended, sensitivity = integrate_grid(
        points, weights, evaluate, auxiliary.nfunction, details=details)
    vectors = dict(direct_binary64=binary64, direct_extended=extended,
                   absolute_sum=details.pop('absolute_sum'))
    return vectors, dict(plan=planned, elapsed_seconds=time.perf_counter()-started,
        binary64_vs_extended_maximum=sensitivity, reduction=details,
        maximum_absolute_weighted_sum=float(np.max(vectors['absolute_sum'])))


def worker(args):
    import numpy as np
    import psi4
    from psi4.driver.procrouting.isapol_schema import (
        admit_native_schema, native_molecule_schema, check_native_geometry,
        implementation_identity, validate_native_request)
    from psi4.driver.procrouting.isapol_recipes import native_monomer_recipe
    common = operations()
    environment = (common.BUILD/'env').resolve()
    qualified = common.qualified_receipts()
    if (not Path(psi4.__file__).resolve().is_relative_to(environment)
            or implementation_identity()['sha256'] != qualified['implementation_sha256']):
        raise ValueError('unchanged qualified private native installation required')
    raw = args.runtime.read_bytes()
    if common.sha(raw) != common.RUNTIME_SHA256:
        raise ValueError('exact original geometry-only runtime required')
    runtime = json.loads(raw)
    memory_admission = admit_study_memory(runtime)
    psi4.core.clean_options()
    psi4.set_num_threads(12, quiet=True)
    psi4.set_memory(memory_admission['core_bytes'], quiet=True)
    request, geometry = admit_native_schema(runtime['specification'], runtime['molecule'])
    molecule = psi4.core.Molecule.from_schema(
        native_molecule_schema(runtime['molecule'], request), nonphysical=True, verbose=0)
    check_native_geometry(molecule, geometry)
    validate_native_request(request.as_dict(), method='pbe0', basis='aug-cc-pvtz', molecule=molecule)
    recipe = native_monomer_recipe(molecule, profile=request.declaration.profile)
    auxiliary = recipe.auxiliary.build('MolecularAux')
    check_native_geometry(molecule, geometry)
    frozen_payload = common.RUNS/'ccasp-native-s66-ed561666fc-charge-diagnostic001/payload'
    arrays, original, _ = load_frozen_context(frozen_payload)
    analytic = np.asarray(psi4.core.IsaAuxCoulomb(auxiliary).charges())
    if (auxiliary.nfunction != 1065 or molecule.natom() != 12
            or not np.array_equal(analytic, arrays['analytic_charge'])):
        raise ValueError('reconstructed response AUX differs from frozen analytic charges/layout')
    fingerprint = common.sha(json.dumps(asdict(recipe.auxiliary),
        allow_nan=False, sort_keys=True, separators=(',', ':')).encode())
    payload = args.run/'payload'
    baseline_charge = arrays['q'][::25].sum(axis=0)
    previous = None
    records = []
    for radial, angular in GRIDS:
        vectors, diagnostics = sample_native_grid(
            psi4.core, molecule, auxiliary, radial, angular)
        extended = vectors['direct_extended']
        vector_path = payload/f'grid-{radial}-{angular}.npz'
        vector_sha = save_vectors(vector_path, vectors)
        projection = frozen_projection(extended, analytic, arrays['anchor_coefficients'],
                                       arrays['solved_anchor'], 2)
        difference = extended-baseline_charge
        baseline_difference = frozen_projection(
            difference, np.zeros_like(analytic), arrays['anchor_coefficients'],
            arrays['solved_anchor'], 2)
        record = dict(status='COMPLETE_GRID_QUADRATURE_NOT_A_RESPONSE', radial_points=radial,
            spherical_points=angular, npoints=diagnostics['plan']['points'], **diagnostics,
            vector_file=vector_path.name, vector_sha256=vector_sha,
            analytic_charge_error_column=int(np.argmax(np.abs(extended-analytic))),
            frozen_projection=projection,
            difference_from_original_partitioned_charge_maximum=float(np.max(np.abs(difference))),
            original_partitioned_difference_column=int(np.argmax(np.abs(difference))),
            difference_from_original_frozen_projection=baseline_difference,
            difference_from_previous_grid_maximum=(None if previous is None else
                                                  float(np.max(np.abs(extended-previous)))))
        common.write_json(payload/f'grid-{radial}-{angular}.json', record)
        records.append(record)
        previous = extended.copy()
        del extended, vectors
    # Numerical agreement screen only; never a changed production acceptance gate.
    baseline_distance = records[0]['difference_from_original_partitioned_charge_maximum']
    agreement_screen = max(1e-9, 32*records[0]['binary64_vs_extended_maximum'])
    report = dict(status='COMPLETE_GRID_STUDY_NOT_A_MODEL', operational_commit=args.commit,
        scientific_commit=common.SCIENTIFIC_COMMIT, runtime_sha256=common.RUNTIME_SHA256,
        build_identity_sha256=common.BUILD_SHA256,
        frozen_report_sha256=FROZEN_REPORT_SHA256, frozen_receipt_sha256=FROZEN_RECEIPT_SHA256,
        auxiliary_recipe_sha256=fingerprint, auxiliary_shell_layout=auxiliary.shell_layout(),
        analytic_charges_exactly_match_frozen=True, grids=records,
        baseline_charge_agreement_screen=agreement_screen,
        baseline_charge_sums_numerically_agree=baseline_distance <= agreement_screen,
        production_tolerance_unchanged=1e-6, core_sha256=common.file_sha(psi4.core.__file__),
        memory_admission=memory_admission,
        implementation=implementation_identity()['sha256'],
        execution_scope='Direct native grid/basis collocation and analytic charge integrals; '
            'no SCF, density fit, ISA iteration, new response solve, LW call or model fit/export.',
        qualification='Three non-nested coupled radial/angular grids do not establish mathematical '
            'convergence or separate radial/angular effects. Baseline charge agreement supports '
            'a quadrature explanation but cannot prove no exclusions occurred. A material baseline '
            'difference leaves exclusions/assembly/input identity unresolved. Frozen-C/X projections '
            'are NOT denser-grid physical responses, LW passes, accepted models or permission to restart.')
    check_native_geometry(molecule, geometry)
    if common.sha(args.runtime.read_bytes()) != common.RUNTIME_SHA256:
        raise ValueError('original runtime changed')
    report['frozen_operand_sha256'] = verify_frozen_files(frozen_payload)
    report['frozen_operands_owned_immutable'] = True
    common.write_json(payload/'grid-study.json', report)


def parent(args):
    common = operations()
    if Path(sys.executable).resolve() != (common.MANAGER/'bin/python').resolve():
        raise ValueError('manager interpreter required for study parent')
    source = Path(__file__).resolve().parents[2]
    sources = common.source_identities(source, args.commit)
    checks = common.helper('phoenix_installed_checks')
    allocation = checks.allocation()
    expected = dict(Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small',
        NumCPUs='12', NumNodes='1', MinMemoryNode='64G', TimeLimit='04:00:00')
    if any(str(allocation.get(k)) != v for k, v in expected.items()):
        raise ValueError('exact12CPU64GiB4h embers grid-study allocation required')
    run = args.run.resolve()
    if (args.run.is_symlink() or run.parent != common.RUNS
            or run.name != STUDY_RUN_NAME
            or not (run/'logs').is_dir()):
        raise ValueError('fresh isolated grid-study002 namespace required; preserve failed001')
    qualified = common.qualified_receipts()
    manifest_path = common.BUILD/'build-identity.json'
    manifest = json.loads(manifest_path.read_bytes())
    environment = common.BUILD/'env'
    checks.verify_runtime(environment, manifest)
    frozen_payload = common.RUNS/'ccasp-native-s66-ed561666fc-charge-diagnostic001/payload'
    frozen_receipts(frozen_payload)
    raw = common.bound_runtime(common.CASE.read_bytes())
    memory_admission = admit_study_memory(json.loads(raw))
    payload = run/'payload'
    payload.mkdir(exist_ok=False)
    scratch = common.SCRATCH/('ccasp-native-grid-study-'+os.environ['SLURM_JOB_ID'])
    scratch.mkdir(mode=0o700, exist_ok=False)
    for name in ('home', 'cache', 'cwd'):
        (scratch/name).mkdir()
    scratch_snapshot = common.helper('phoenix_pilot_cb63305').scratch_snapshot(
        scratch, payload, common.helper('descriptor_scratch_cb63305'),
        execution_profile=dict(scratch_bytes=2*1024**3, scratch_files=4096))
    runtime_path = scratch/'runtime-input.json'
    with runtime_path.open('xb') as stream:
        stream.write(raw)
    common.write_json(payload/'study-started.json', dict(
        allocation=allocation, operational_commit=args.commit, source_sha256=sources,
        runtime_sha256=common.RUNTIME_SHA256, scientific_commit=common.SCIENTIFIC_COMMIT,
        build_identity_sha256=common.BUILD_SHA256, frozen_report_sha256=FROZEN_REPORT_SHA256,
        memory_admission=memory_admission))
    env = common.helper('phoenix_descriptor_bank_cb63305').child_environment(environment, scratch)
    env.update(OMP_NUM_THREADS='12', MKL_NUM_THREADS='12')
    command = checks.installed_python_command(environment/'bin/python', scratch/'cache',
        str(Path(__file__).resolve()), '--worker', '--run', str(run),
        '--runtime', str(runtime_path), '--commit', args.commit)
    subprocess.run(command, env=env, cwd=scratch/'cwd', stdin=subprocess.DEVNULL, check=True)
    checks.verify_runtime(environment, manifest)
    if (common.source_identities(source, args.commit) != sources
            or common.qualified_receipts() != qualified
            or common.file_sha(manifest_path) != common.BUILD_SHA256
            or common.sha(common.CASE.read_bytes()) != common.CASE_SHA256
            or common.sha(runtime_path.read_bytes()) != common.RUNTIME_SHA256):
        raise ValueError('study source/input/qualified runtime changed')
    operand_hashes = verify_frozen_files(frozen_payload)
    report_path = payload/'grid-study.json'
    report = json.loads(report_path.read_bytes())
    if (report['status'] != 'COMPLETE_GRID_STUDY_NOT_A_MODEL'
            or report['operational_commit'] != args.commit
            or report['runtime_sha256'] != common.RUNTIME_SHA256
            or report['production_tolerance_unchanged'] != 1e-6
            or report['memory_admission'] != memory_admission
            or report['frozen_operands_owned_immutable'] is not True
            or report['frozen_operand_sha256'] != operand_hashes
            or [(r['radial_points'], r['spherical_points']) for r in report['grids']] != list(GRIDS)
            or any(r['status'] != 'COMPLETE_GRID_QUADRATURE_NOT_A_RESPONSE' for r in report['grids'])):
        raise ValueError('all three complete bound diagnostic grids required')
    common.write_json(payload/'study-receipt.json', dict(
        status='COMPLETE_GRID_STUDY_NOT_A_MODEL', allocation=allocation,
        operational_commit=args.commit, scientific_commit=common.SCIENTIFIC_COMMIT,
        source_sha256=sources, build_identity_sha256=common.BUILD_SHA256,
        runtime_unchanged=True, report_sha256=common.file_sha(report_path),
        scratch_snapshot=scratch_snapshot, frozen_report_sha256=FROZEN_REPORT_SHA256,
        frozen_operand_sha256=operand_hashes))
    print(json.dumps(dict(status=report['status'],
        baseline_charge_sums_numerically_agree=report['baseline_charge_sums_numerically_agree'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--runtime', type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.runtime is None:
            parser.error('--worker requires --runtime')
        worker(args)
    else:
        if args.runtime is not None:
            parser.error('parent derives its one exact geometry-only runtime')
        parent(args)


if __name__ == '__main__':
    main()
