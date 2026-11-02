"""Fresh standalone benzene density-fit diagnostic; never a native model.

K=0 is used solely to expose the native assembled operands. Its plain-LU
coefficients are discarded unread. Independent K=10/32 diagnostic replays do
not change production policy, resume a record, or run ISA/response/export.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

RUNTIME_SHA256 = 'c381842faf7f32bf279d9cf6f55983afc6d95c95e5a4ab16e5ed3fb78a300701'
SCIENTIFIC_COMMIT = 'cb63305f4115526417a830829a0a30c36b85b4e8'
RUNS = Path('/storage/project/r-cs207-0/awallace43/runs')
SCRATCH = Path('/storage/scratch1/3/awallace43')
MANAGER = Path('/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca')
MAX_CHECKPOINT_BYTES = 64*1024**2


def helper(name):
    if name not in ('phoenix_installed_checks', 'phoenix_descriptor_bank_cb63305',
                    'descriptor_scratch_cb63305', 'phoenix_pilot_cb63305'):
        raise ValueError('closed standalone diagnostic helpers required')
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_drho_diagnostic_'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def numeric_libraries():
    """Identity of mapped numeric libraries, not an inferred backend version."""
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/'):
            path = Path(fields[5])
            if any(key in path.name.lower() for key in (
                    'blas', 'lapack', 'mkl', 'libint', 'gfortran', 'libomp', 'libiomp')):
                if not path.is_file():
                    raise ValueError('mapped numeric library is no longer a readable file')
                paths.add(path.resolve())
    return [dict(path=str(path), sha256=file_sha(path)) for path in sorted(paths)]


def load_runtime(raw):
    if sha(raw) != RUNTIME_SHA256:
        raise ValueError('exact geometry-only benzene runtime input required')
    value = json.loads(raw)
    if (set(value) != {'molecule', 'specification'}
            or value['specification']['model'] != dict(method='pbe0', basis='aug-cc-pvtz')):
        raise ValueError('one native PBE0/aug-cc-pVTZ input required')
    return value


def admit_allocation(allocation):
    expected = dict(Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small',
                    NumCPUs='12', NumNodes='1', MinMemoryNode='64G', TimeLimit='01:00:00')
    if any(allocation.get(k) != v for k, v in expected.items()):
        raise ValueError('exact12CPU64GiB1h diagnostic embers allocation required')


def validate_report(manifest_path, original_sha, report, commit):
    if (file_sha(manifest_path) != original_sha
            or report.get('build_identity_sha256') != original_sha
            or report.get('status') != 'COMPLETE_DIAGNOSTIC_NOT_A_MODEL'
            or report.get('scientific_commit') != SCIENTIFIC_COMMIT
            or report.get('operational_commit') != commit
            or report.get('runtime_input_sha256') != RUNTIME_SHA256
            or report.get('production_cap_unchanged') != 10):
        raise ValueError('diagnostic manifest, source/input binding or unchanged policy differs')


def owned_array(value):
    import numpy as np
    value = np.asarray(value)
    if value.dtype.kind != 'f' or value.dtype.itemsize != 8 or not np.isfinite(value).all():
        raise ValueError('finite native binary64 operands required')
    result = np.array(value, dtype='<f8', order='C', copy=True)
    result.flags.writeable = False
    return result


def array_identity(value):
    return dict(shape=list(value.shape), dtype=value.dtype.str,
                bytes=value.nbytes, sha256=sha(value.tobytes(order='C')))


def require_hooks(core):
    if any(not callable(getattr(core, name, None)) for name in (
            '_isa_exact_residual', '_isa_refined_lu_solve')):
        raise ValueError('installed native solve/exact-residual diagnostic hooks required')


def check_fp_environment(core):
    import numpy as np
    require_hooks(core)
    probe = core._isa_refined_lu_solve(np.eye(1), np.ones(1), 1)
    if probe != ([1.0], 1, 0.0):
        raise ValueError('native round-to-nearest/gradual-underflow probe differs')


def capture_operands(core, main, auxiliary, geometry):
    """Identical native assembly; never inspect or retain the K=0 coefficients."""
    occupied = owned_array(main.occupied)
    if occupied.shape != (414, 21):
        raise ValueError('exact adapted benzene MAIN/occupied dimensions required')
    # Exercise production's FP check on this calling thread BEFORE assembly.
    check_fp_environment(core)
    coulomb = core.IsaAuxCoulomb(auxiliary)
    if len(coulomb.charges()) != 1146:
        raise ValueError('exact native Cartesian density-AUX width required')
    assembled = coulomb.fit_drho_c(main.basis, core.Matrix.from_array(occupied),
                                   1000., max_refinement_iterations=0)
    arrays = dict(A=owned_array(assembled.metric.np), b=owned_array(assembled.rhs),
        J=owned_array(assembled.coulomb_metric.np), q=owned_array(assembled.charges),
        raw_rhs=owned_array(assembled.raw_rhs), occupied=occupied,
        geometry=owned_array(geometry))
    del assembled  # plain-LU coefficients are neither read nor used
    if (arrays['A'].shape != (1146, 1146) or arrays['J'].shape != (1146, 1146)
            or any(arrays[k].shape != (1146,) for k in ('b', 'q', 'raw_rhs'))
            or arrays['geometry'].shape != (12, 3)):
        raise ValueError('unexpected native benzene operand shapes')
    return arrays


def replay(core, arrays, cap):
    """Refusal remains refusal; each call starts a fresh native factorization."""
    import numpy as np
    if type(cap) is not int or cap not in (10, 32):
        raise ValueError('only explicit diagnostic caps10 and32 are admitted')
    before = {k: array_identity(arrays[k]) for k in ('A', 'b')}
    try:
        x, iterations, displacement = core._isa_refined_lu_solve(arrays['A'], arrays['b'], cap)
    except ValueError as error:
        result, solution = dict(cap=cap, status='refused', error_type=type(error).__name__,
                                error_message=str(error)), None
    else:
        solution = owned_array(x)
        if (solution.shape != arrays['b'].shape or type(iterations) is not int
                or not 1 <= iterations <= cap or not np.isfinite(displacement)):
            raise ValueError('invalid diagnostic native solve return')
        residual = owned_array(core._isa_exact_residual(arrays['A'], solution, arrays['b']))
        anorm = np.max(np.sum(np.abs(arrays['A']), axis=1, dtype=np.longdouble))
        xnorm, bnorm = np.max(np.abs(solution)), np.max(np.abs(arrays['b']))
        rnorm = np.max(np.abs(residual))
        denominator = anorm*xnorm+bnorm
        electrons = np.dot(arrays['q'].astype(np.longdouble), solution.astype(np.longdouble))
        result = dict(cap=cap, status='converged_diagnostic_only', iterations=iterations,
            displacement=float(displacement), residual_infinity=float(rnorm),
            normwise_backward_residual=float(rnorm/denominator) if denominator else 0.,
            fitted_electrons=float(electrons), electron_error=float(electrons-42))
        if not all(np.isfinite(v) for v in (result['normwise_backward_residual'],
                                           result['fitted_electrons'], result['electron_error'])):
            raise ValueError('nonfinite diagnostic reduction')
    if before != {k: array_identity(arrays[k]) for k in ('A', 'b')}:
        raise ValueError('frozen native operands changed during diagnostic replay')
    return result, solution


def checkpoint(path, arrays):
    import numpy as np
    if any(a.dtype.str != '<f8' or not a.flags.c_contiguous or not np.isfinite(a).all()
           for a in arrays.values()):
        raise ValueError('finite little-endian C-order binary64 checkpoint arrays required')
    # Margin covers the bounded NPZ headers/ZIP directory, not just payload.
    if sum(a.nbytes for a in arrays.values())+65536 > MAX_CHECKPOINT_BYTES:
        raise ValueError('diagnostic checkpoint byte budget exceeded before write')
    with path.open('xb') as stream:
        np.savez(stream, **arrays)
    if path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ValueError('diagnostic checkpoint exceeds byte budget')
    identities = {k: array_identity(v) for k, v in arrays.items()}
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(
                array_identity(saved[k]) != identities[k] for k in arrays):
            raise ValueError('diagnostic checkpoint round-trip changed operand bytes')
    return dict(file_sha256=file_sha(path), arrays=identities)


def worker(args):
    import numpy as np
    import psi4
    from psi4.driver.procrouting.isapol_schema import (
        admit_native_schema, native_molecule_schema, check_native_geometry, validate_native_request)
    from psi4.driver.procrouting.isapol_native_correction import require_scf_seal, validate_correction
    from psi4.driver.procrouting.isapol_basis import adapt_main
    from psi4.driver.procrouting.isapol_recipes import native_monomer_recipe
    from psi4.driver.procrouting.isapol_distribution import isa_resource_plan
    from psi4.driver.procrouting.isapol_native_partition import DRHO_REFINEMENT_ITERATIONS
    environment = args.build.resolve()/'env'
    if (not Path(psi4.__file__).resolve().is_relative_to(environment)
            or DRHO_REFINEMENT_ITERATIONS != 10):
        raise ValueError('immutable installed cb producer and production cap10 required')
    runtime = load_runtime(args.runtime.read_bytes())
    spec = copy.deepcopy(runtime['specification'])
    request, original_geometry = admit_native_schema(spec, runtime['molecule'])
    out = args.run/'payload'
    psi4.core.set_output_file(str(out/'scf.log'), False)
    psi4.core.clean_options()
    psi4.set_memory(40*1024**3)
    psi4.set_num_threads(12)
    require_hooks(psi4.core)  # fail before expensive SCF if hooks are absent
    molecule = psi4.core.Molecule.from_schema(
        native_molecule_schema(runtime['molecule'], request), nonphysical=True)
    check_native_geometry(molecule, original_geometry)
    kwargs = spec['keywords'].pop('function_kwargs')
    kwargs.pop('native_monomer')
    kwargs.setdefault('properties', ['dipole'])
    psi4.set_options(spec['keywords'])
    psi4.core.set_global_option('BASIS', spec['model']['basis'])
    validate_native_request(request.as_dict(), method=spec['model']['method'],
                            basis=spec['model']['basis'], molecule=molecule)
    _, wfn = psi4.properties(spec['model']['method'], molecule=molecule, return_wfn=True, **kwargs)
    require_scf_seal(wfn)
    validate_correction(wfn, scf_correction='NONE', require_canonical=True)
    check_native_geometry(wfn.molecule(), original_geometry)
    recipe = native_monomer_recipe(wfn.molecule(), profile=request.declaration.profile)
    if recipe.auxiliary.representation != 'Cartesian':
        raise ValueError('same native Cartesian density-AUX recipe required')
    main = adapt_main(wfn, caller_converged=True)
    # Charge a conservative full ISA planning envelope even though no ISA/grid
    # is executed; add the extra replay factors, exact residuals and copies.
    numeric, work = isa_resource_plan(wfn, recipe, recipe.auxiliary, range(0), 4)
    d = 1146
    iteration = 256*d*(d+1)+16384*d+16*d*d+64*d
    numeric += 4*MAX_CHECKPOINT_BYTES
    # Two extra factorizations; 32 extra refinement passes and up to two
    # postsolve exact residuals; conservative hashes/copies and the tiny FP probe.
    work += 16*d**3+34*iteration+128*MAX_CHECKPOINT_BYTES+65536
    io = 3*MAX_CHECKPOINT_BYTES+8*1024**2
    if (numeric > request.resources.max_bytes or work > request.resources.max_work
            or io > request.resources.max_io_bytes):
        raise ValueError('separate diagnostic copies/replays/I/O exceed declared ceilings')
    arrays = capture_operands(psi4.core, main, recipe.auxiliary.build('MolecularAux'),
                             wfn.molecule().geometry().np)
    attempts = []
    for cap in (10, 32):
        result, solution = replay(psi4.core, arrays, cap)
        attempts.append(result)
        if solution is not None:
            arrays['diagnostic_solution_cap'+str(cap)] = solution
    frozen = checkpoint(out/'frozen-operands.npz', arrays)
    reproduced = (attempts[0]['status'] == 'refused' and
                  'did not converge within the iteration cap' in attempts[0]['error_message'])
    verdict = ('CAP10_REFUSAL_CAP32_CONVERGED_FOR_REGENERATED_SYSTEM' if reproduced and
        attempts[1]['status'] == 'converged_diagnostic_only' else
        'CAP32_REFUSED_FOR_REGENERATED_SYSTEM' if reproduced else 'HISTORICAL_CAP10_FAILURE_NOT_REPRODUCED')
    report = dict(status='COMPLETE_DIAGNOSTIC_NOT_A_MODEL', discriminator=verdict,
        scientific_commit=SCIENTIFIC_COMMIT, operational_commit=args.commit,
        runtime_input_sha256=RUNTIME_SHA256, core_sha256=file_sha(psi4.core.__file__),
        build_identity_sha256=file_sha(args.build/'build-identity.json'),
        mapped_numeric_libraries=numeric_libraries(),
        numpy_version=np.__version__, host=socket.gethostname(), production_cap_unchanged=10,
        fp_environment_verified_before_assembly=True,
        method='PBE0/aug-cc-pVTZ', point_group=wfn.molecule().schoenflies_symbol(),
        main_recipe=asdict(main.recipe), density_aux_recipe=asdict(recipe.auxiliary),
        penalty=1000., caps=attempts, checkpoint=frozen,
        planned_numeric_bytes=numeric, planned_work_units=work, planned_io_bytes=io,
        effective_threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
        core_threads=psi4.core.get_num_threads(),
        qualification='Fresh same-case reproduction, not the lost original operands. K=0 coefficients '
            'discarded unread; no plain-LU fallback, ISA/controller/density/response/Cn, QCArchive '
            'mutation, production-cap change, accepted model or forward-accuracy certificate. '
            'Planning covers native diagnostic numerical buffers/work and checkpoint I/O, '
            'not SCF, whole-process RSS, vendor workspace or system I/O.')
    helper('phoenix_descriptor_bank_cb63305').write_json_once(out/'drho-diagnostic.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'runtime', 'run'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    checks, common = helper('phoenix_installed_checks'), helper('phoenix_descriptor_bank_cb63305')
    if Path(sys.executable).resolve() != (MANAGER/'bin/python').resolve():
        raise ValueError('diagnostic controller must use the reviewed manager interpreter')
    allocation = checks.allocation()
    admit_allocation(allocation)
    source = Path(__file__).resolve().parents[2]
    if checks.git(source, 'rev-parse', 'HEAD') != args.commit or checks.git(source, 'status', '--porcelain'):
        raise ValueError('clean pinned diagnostic source required')
    build, run = args.build.resolve(), args.run.resolve()
    if (not build.is_relative_to(RUNS) or not run.is_relative_to(RUNS)
            or not run.name.startswith('ccasp-native-s66-') or build == run
            or build.is_relative_to(run) or run.is_relative_to(build)):
        raise ValueError('separate private diagnostic build and run required')
    raw = args.runtime.read_bytes()
    load_runtime(raw)
    manifest_path = build/'build-identity.json'
    manifest_raw = manifest_path.read_bytes()
    manifest, manifest_sha = json.loads(manifest_raw), sha(manifest_raw)
    if manifest['commit'] != SCIENTIFIC_COMMIT:
        raise ValueError('original immutable cb scientific producer required')
    changes = checks.git(source, 'diff', '--name-only', SCIENTIFIC_COMMIT, args.commit).splitlines()
    if any(not p.startswith('devtools/isapol_s66_campaign/')
           and p not in checks.TEST_ONLY_CHANGES for p in changes):
        raise ValueError('diagnostic operations changed scientific/build code')
    environment = build/'env'
    checks.verify_runtime(environment, manifest)
    operations = {n: checks.digest(Path(__file__).with_name(n)) for n in (
        'drho_diagnostic_cb63305.py', 'drho_diagnostic_cb63305.sbatch',
        'phoenix_installed_checks.py', 'phoenix_descriptor_bank_cb63305.py',
        'descriptor_scratch_cb63305.py', 'phoenix_pilot_cb63305.py', 'term_wait.sh')}
    payload = run/'payload'
    payload.mkdir(exist_ok=False)
    scratch = SCRATCH/('ccasp-native-drho-diagnostic-'+os.environ['SLURM_JOB_ID'])
    scratch.mkdir(mode=0o700, exist_ok=False)
    for name in ('home','cache','cwd'):
        (scratch/name).mkdir()
    disk = helper('phoenix_pilot_cb63305').scratch_snapshot(
        scratch, payload, helper('descriptor_scratch_cb63305'),
        execution_profile=dict(scratch_bytes=256*1024**3, scratch_files=131072))
    runtime_path = scratch/'runtime-input.json'
    with runtime_path.open('xb') as stream:
        stream.write(raw)
    env = common.child_environment(environment, scratch)
    env.update(OMP_NUM_THREADS='12', MKL_NUM_THREADS='12')
    command = checks.installed_python_command(environment/'bin/python', scratch/'cache',
        str(Path(__file__).resolve()), '--worker', '--build', str(build), '--runtime', str(runtime_path),
        '--run', str(run), '--commit', args.commit)
    subprocess.run(command, env=env, cwd=scratch/'cwd', stdin=subprocess.DEVNULL, check=True)
    checks.verify_runtime(environment, manifest)
    if (checks.git(source,'rev-parse','HEAD') != args.commit or checks.git(source,'status','--porcelain')
            or checks.digest(args.runtime) != RUNTIME_SHA256 or checks.digest(runtime_path) != RUNTIME_SHA256
            or any(checks.digest(Path(__file__).with_name(n)) != v for n,v in operations.items())):
        raise ValueError('diagnostic source, raw input or runtime changed')
    report = json.loads((payload/'drho-diagnostic.json').read_bytes())
    validate_report(manifest_path, manifest_sha, report, args.commit)
    common.write_json_once(payload/'diagnostic-receipt.json', dict(
        status='COMPLETE_DIAGNOSTIC_NOT_A_MODEL', allocation=allocation,
        operational_commit=args.commit, scientific_commit=SCIENTIFIC_COMMIT, runtime_unchanged=True,
        build_identity_sha256=manifest_sha,
        scratch_snapshot=disk,
        operational_sources_sha256=operations, report_sha256=checks.digest(payload/'drho-diagnostic.json'),
        discriminator=report['discriminator'], qualification='No model, record retry or convergence-policy change.'))
    print(json.dumps(dict(status=report['status'], discriminator=report['discriminator'])))


if __name__ == '__main__':
    main()
