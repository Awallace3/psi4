"""First-node charge-flow observation for one fresh native diagnostic, not a model.

The scientific installation and its numeric functions are never replaced. A
thread-local Python profile callback observes their actual return-frame arrays.
All scientific inputs and the original production LW gate remain unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

MAX_ARRAY_BYTES = 256 * 1024**2
MAX_CAPTURE_BYTES = 512 * 1024**2
SCIENTIFIC_COMMIT = 'ed561666fcac212fffa774429fc3bd8f8cfc903d'
RUNS = Path('/storage/project/r-cs207-0/awallace43/runs')
MANAGER = Path('/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca')
BUILD = RUNS/'ccasp-native-s66-ed561666fc-build001'
CASE = RUNS/'ccasp-native-s66-ed561666fc-remaining-preparation001/cases/monomer-0008.json'
CASE_SHA256 = '191d9c961f976777e09c6e69edf0188dcb6609c3734cebf822cdfb51124d09da'
RUNTIME_SHA256 = 'b400f552413e29f3eb9a4285257fabfaa45c4bb608f62bc021ee13a206b43dc7'
BUILD_SHA256 = '00cecb3108ce8b7c6c54edc5828aedf609f9a73e44715dd634c447244a91a7a2'
SCRATCH = Path('/storage/scratch1/3/awallace43')
QUALIFICATION = RUNS/'ccasp-native-s66-ed561666fc-checks001/payload'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    raw = json.dumps(value, allow_nan=False, indent=2).encode() + b'\n'
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha(raw)


def finite64(value):
    import numpy as np
    value = np.asarray(value)
    if value.dtype != np.dtype('float64') or not np.isfinite(value).all():
        raise ValueError('finite native float64 array required')
    return value


def charge_vectors(alpha, sites, rank_limit):
    """Both signed LW charge sums in its retained component space."""
    import numpy as np
    alpha = finite64(alpha)
    if (type(sites) is not int or sites < 1 or type(rank_limit) is not int
            or not 0 <= rank_limit <= 4 or alpha.shape != (25*sites, 25*sites)):
        raise ValueError('rank4 site-major matrix and explicit retained rank required')
    selector = np.zeros(25*sites)
    selector[::25] = 1.
    retained = np.array([25*a+t for a in range(sites)
                         for t in range((rank_limit+1)**2)])
    return (alpha @ selector)[retained], (alpha.T @ selector)[retained], retained


def analyze_charge_flow(q, analytic_charge, coefficients, solved, legs, alpha, rank_limit):
    """Attribute the observed transpose sum; never project or repair any array."""
    import numpy as np
    q, analytic_charge, coefficients, solved, legs, alpha = map(
        finite64, (q, analytic_charge, coefficients, solved, legs, alpha))
    if q.ndim != 2 or q.shape[0] % 25 or analytic_charge.shape != (q.shape[1],):
        raise ValueError('Q/analytic AUX charge dimensions differ')
    sites, p = q.shape[0]//25, q.shape[1]
    if (coefficients.ndim != 2 or coefficients.shape[1] != p
            or solved.shape != (coefficients.shape[0], 25*sites)
            or legs.shape != solved.shape or alpha.shape != (25*sites, 25*sites)):
        raise ValueError('anchor/solution/nonlocal dimensions differ')
    q_grid = q[::25].sum(axis=0)
    delta = q_grid - analytic_charge
    g_fit = coefficients @ analytic_charge
    g_grid = coefficients @ delta
    fit = -solved.T @ g_fit
    integration = -solved.T @ g_grid
    row, column, retained = charge_vectors(alpha, sites, rank_limit)
    reconstructed = (fit+integration)[retained]
    # Observational consistency, not an altered physical acceptance gate.
    scale = max(1., float(np.max(np.abs(alpha))))
    reconstruction_error = float(np.max(np.abs(column-reconstructed)))
    legs_error = float(np.max(np.abs(legs-coefficients @ q.T)))
    alpha_error = float(np.max(np.abs(alpha+legs.T @ solved)))
    if reconstruction_error > 1e-10*scale or alpha_error > 1e-10*scale or legs_error > 1e-10:
        raise ValueError('captured operands do not reconstruct the actual production response')
    maximum_index = int(np.argmax(np.abs(column)))
    component = int(retained[maximum_index])
    return dict(
        sites=sites, naux=p, nov=coefficients.shape[0], localization_rank_limit=rank_limit,
        input_sum_rule=max(float(np.max(np.abs(row))), float(np.max(np.abs(column)))),
        first_slot_maximum=float(np.max(np.abs(row))),
        transpose_sum_maximum=float(np.max(np.abs(column))),
        q_charge_row_error=float(np.max(np.abs(delta))),
        analytic_fit_charge_maximum=float(np.max(np.abs(g_fit))),
        integration_fit_charge_maximum=float(np.max(np.abs(g_grid))),
        fit_response_contribution_maximum=float(np.max(np.abs(fit[retained]))),
        integration_response_contribution_maximum=float(np.max(np.abs(integration[retained]))),
        transpose_maximum=dict(site_index=component//25, component_index=component % 25,
            signed_sum=float(column[maximum_index]), fit=float(fit[component]),
            integration=float(integration[component])),
        reconstruction_error=reconstruction_error, legs_error=legs_error, alpha_error=alpha_error,
        qualification='Observed first-node absolute charge sums and algebraic decomposition; '
            'no response repair, localization-policy change, grid-convergence or model acceptance.')


class Capture:
    """Exclusive bounded NPY files; only one array's contiguous copy is transient."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.records = {}
        self.total_bytes = 0

    def save(self, name, value):
        import numpy as np
        value = finite64(value)
        if (name not in {'q', 'analytic_charge', 'anchor_coefficients', 'solved_anchor',
                         'anchor_legs', 'nonlocal_response'} or name in self.records):
            raise ValueError('one exclusive file per declared capture role required')
        size = value.nbytes
        if size > MAX_ARRAY_BYTES or self.total_bytes+size > MAX_CAPTURE_BYTES:
            raise ValueError('diagnostic array byte ceiling exceeded before write')
        value = np.array(value, dtype='<f8', order='C', copy=True)
        path = self.directory/(name+'.npy')
        with path.open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        self.records[name] = dict(file=path.name, shape=list(value.shape),
            dtype=value.dtype.str, numeric_bytes=size, file_sha256=file_sha(path))
        self.total_bytes += size

    def load(self, name):
        import numpy as np
        record = self.records[name]
        path = self.directory/record['file']
        if path.is_symlink() or file_sha(path) != record['file_sha256']:
            raise ValueError('captured operand file changed')
        value = np.load(path, allow_pickle=False, mmap_mode='r')
        if list(value.shape) != record['shape'] or value.dtype.str != record['dtype']:
            raise ValueError('captured operand layout changed')
        return finite64(value)

    def analyze(self, rank_limit):
        return analyze_charge_flow(*(self.load(name) for name in (
            'q', 'analytic_charge', 'anchor_coefficients', 'solved_anchor',
            'anchor_legs', 'nonlocal_response')), rank_limit)


class DiagnosticStop(Exception):
    """Deliberate first-node stop, never a successful scientific result."""


class Observer:
    """Observe exact code objects; never modify a frame or a returned array."""
    def __init__(self, capture, *, q_function, fit_function, solve_function,
                 lw_function, refinement_function, prepare_function=None):
        self.capture = capture
        pairs = (
            (q_function, 'q'), (fit_function, 'fit'), (solve_function, 'solve'),
            (lw_function, 'lw'), (refinement_function, 'refinement'))
        self.functions = tuple(function for function, _ in pairs)
        self.codes = {function.__code__: role for function, role in pairs}
        if len(self.codes) != 5:
            raise ValueError('five distinct production observer seams required')
        self.prepare_code = None if prepare_function is None else prepare_function.__code__
        self.counts = dict(q=0, fit=0, solve=0, lw=0)
        self.details = {}

    def profile(self, frame, event, value):
        role = self.codes.get(frame.f_code)
        if role is None or event not in ('call', 'return'):
            return
        local = frame.f_locals
        if event == 'call':
            if role == 'refinement':
                raise ValueError('unexpected attempt to enter refinement/PFIT')
            self.counts[role] += 1
            if ((role in ('q', 'solve', 'lw') and self.counts[role] != 1)
                    or (role == 'fit' and self.counts[role] > 2)):
                raise ValueError('unexpected repeated stage; no second diagnostic attempt')
            if role == 'fit' and self.prepare_code is not None:
                caller = frame.f_back
                expected = 'anchor' if self.counts['fit'] == 1 else 'target'
                if (caller is None or caller.f_code is not self.prepare_code
                        or caller.f_locals.get('name') != expected
                        or local['charge_penalty'] != 1000.
                        or local['offsite_metric_damping'] != (.0005 if expected == 'anchor' else 0.)):
                    raise ValueError('original anchor-then-target production fit path required')
            if role == 'solve' and local['omega'] != 0.:
                raise ValueError('only the original first static node may execute')
            if role == 'lw':
                if local['residual_policy'] != 'production':
                    raise ValueError('unchanged production LW policy required')
                self.details['localization_rank_limit'] = local['localization_rank_limit']
                self.details['lw_policy'] = local['residual_policy']
            return
        # A Python exception also emits a return event with None. Do not mask it.
        if value is None:
            return
        if role == 'q':
            _, q, error = value
            self.capture.save('q', q)
            self.details['producer_q_charge_row_error'] = float(error)
        elif role == 'fit' and self.counts['fit'] == 1:
            self.capture.save('anchor_coefficients', value.coefficients)
            self.capture.save('analytic_charge', local['charges'])
            self.details['anchor_metric_damping'] = float(local['offsite_metric_damping'])
            self.details['charge_penalty'] = float(local['charge_penalty'])
        elif role == 'solve':
            solver = local['self']
            self.capture.save('solved_anchor', local['solved'][:, solver.p:])
            self.capture.save('anchor_legs', solver.legs[:, solver.p:])
            self.capture.save('nonlocal_response', value[1])
            self.details['frequency_au'] = float(local['omega'])
            self.details['response_residual'] = float(value[2])
        elif role == 'lw':
            self.details['lw_passed'] = True
            raise DiagnosticStop('first-node localization passed; stopped before refinement/PFIT')

    def validate(self):
        if self.counts != dict(q=1, fit=2, solve=1, lw=1):
            raise ValueError('one complete original Q/anchor/target/response/LW path required')
        if set(self.capture.records) != {
                'q', 'analytic_charge', 'anchor_coefficients', 'solved_anchor',
                'anchor_legs', 'nonlocal_response'}:
            raise ValueError('all six frozen charge-flow roles required')
        if self.details.get('lw_policy') != 'production':
            raise ValueError('production LW call was not observed')
        return self.capture.analyze(self.details['localization_rank_limit'])


def helper(name):
    if name not in {'phoenix_installed_checks', 'phoenix_descriptor_bank_cb63305',
                    'phoenix_pilot_cb63305', 'descriptor_scratch_cb63305',
                    'qualified_ed561666'}:
        raise ValueError('closed diagnostic helper required')
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_charge_diagnostic_'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def bound_runtime(case_raw):
    if sha(case_raw) != CASE_SHA256:
        raise ValueError('exact unchanged monomer0008 raw case required')
    case = json.loads(case_raw)
    entry, spec = case['entry'], case['specification']
    # Dataset-only reference attributes are deliberately absent.
    runtime = dict(molecule=entry['molecule'], specification=dict(driver=spec['driver'],
        model={k: spec[k] for k in ('method', 'basis')},
        keywords=dict(spec['keywords'], **entry['additional_keywords'])))
    raw = json.dumps(runtime, allow_nan=False).encode()
    if sha(raw) != RUNTIME_SHA256:
        raise ValueError('unchanged geometry-only runtime required')
    return raw


def source_identities(source, commit):
    """Verify every tracked campaign executable byte, including hidden index edits."""
    import re
    if re.fullmatch('[0-9a-f]{40}', commit) is None:
        raise ValueError('exact operational commit required')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args])
    if git('rev-parse', 'HEAD').decode().strip() != commit or git('status', '--porcelain').strip():
        raise ValueError('clean pinned operational source required')
    changes = git('diff', '--name-only', SCIENTIFIC_COMMIT, commit).decode().splitlines()
    if any(not name.startswith('devtools/isapol_s66_campaign/') for name in changes):
        raise ValueError('diagnostic operational revision changes scientific/build code')
    names = git('ls-tree', '-r', '--name-only', commit,
                'devtools/isapol_s66_campaign').decode().splitlines()
    identities = {}
    for name in names:
        if name.endswith(('.py', '.sh', '.sbatch')):
            raw = (source/name).read_bytes()
            if raw != git('show', commit+':'+name):
                raise ValueError('operational source bytes differ: '+name)
            identities[name] = sha(raw)
    if 'devtools/isapol_s66_campaign/'+Path(__file__).name not in identities:
        raise ValueError('diagnostic runner must be committed')
    return identities


def admit_allocation(allocation):
    expected = dict(Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small',
                    NumCPUs='12', NumNodes='1', MinMemoryNode='64G', TimeLimit='04:00:00')
    if any(str(allocation.get(k)) != v for k, v in expected.items()):
        raise ValueError('exact12CPU64GiB4h isolated diagnostic embers allocation required')


def qualified_receipts():
    paths = dict(build_identity=BUILD/'build-identity.json',
        installed_qualification=QUALIFICATION/'installed-validation.json',
        imported_identity=QUALIFICATION/'imported-identity.json', junit=QUALIFICATION/'junit.xml')
    return helper('qualified_ed561666').verify_receipts(
        {role: path.read_bytes() for role, path in paths.items()})


def observer_for_installation(capture, environment):
    import psi4
    from psi4.driver.procrouting import (
        isapol_distribution as distribution, isapol_native_factors as factors,
        isapol_bounded_response as response, isapol_bounded as bounded, isapol_lw as lw)
    from psi4.driver.procrouting.isapol_native_partition import DRHO_REFINEMENT_ITERATIONS
    from psi4.driver.procrouting.isapol_schema import implementation_identity
    qualified = qualified_receipts()
    if (not Path(psi4.__file__).resolve().is_relative_to(environment)
            or DRHO_REFINEMENT_ITERATIONS != 32 or lw.PRODUCTION_TOLERANCE != 1e-6
            or str(psi4.__version__) != qualified['version']
            or implementation_identity()['sha256'] != qualified['implementation_sha256']):
        raise ValueError('unchanged qualified cap32 installation and production tolerance required')
    functions = dict(q_function=distribution._stockholder_q,
        fit_function=factors.native_constrained_ov, solve_function=response._H2H1Response.solve,
        lw_function=lw.supplied_nonlocal_properties, refinement_function=bounded._refinement_node)
    # Names imported at call sites must still be the exact original objects.
    if (bounded.isa_moments is not distribution.isa_moments
            or response.native_constrained_ov is not factors.native_constrained_ov
            or bounded.lw is not lw):
        raise ValueError('production call-site function identities differ')
    bindings = {}
    for role, function in functions.items():
        path = Path(function.__code__.co_filename).resolve()
        if not path.is_relative_to(environment):
            raise ValueError('observer seam is not installed private scientific code')
        bindings[role] = dict(file=str(path), sha256=file_sha(path),
                              first_line=function.__code__.co_firstlineno)
    observer = Observer(capture, **functions, prepare_function=response.BoundedResponse._prepare_reference)
    return observer, bindings


def run_worker(args):
    import psi4
    from psi4.driver import schema_wrapper
    environment = (BUILD/'env').resolve()
    payload = args.run/'payload'
    raw = args.runtime.read_bytes()
    if sha(raw) != RUNTIME_SHA256:
        raise ValueError('exact frozen diagnostic runtime required before SCF')
    runtime = json.loads(raw)
    observer, bindings = observer_for_installation(Capture(payload), environment)
    if sys.getprofile() is not None:
        raise ValueError('unoccupied thread-local profiler required')
    if args.smoke:
        write_json(payload/'worker-smoke.json', dict(
            status='PASS_INSTALLED_OBSERVER_SEAMS_NO_SCF',
            runtime_sha256=sha(raw), bindings=bindings, production_tolerance=1e-6,
            production_cap=32, core_sha256=file_sha(psi4.core.__file__)))
        return
    psi4.set_memory(40*1024**3, quiet=True)
    psi4.set_num_threads(12, quiet=True)
    original_codes = tuple(function.__code__ for function in observer.functions)
    atomic_input = dict(schema_name='qcschema_input', schema_version=1,
                        molecule=runtime['molecule'], **runtime['specification'])
    result, failure, stopped = None, None, None
    try:
        sys.setprofile(observer.profile)
        result = schema_wrapper.run_qcschema(atomic_input, return_dict=True)
    except DiagnosticStop as error:
        stopped = str(error)
    finally:
        sys.setprofile(None)
    # The public schema wrapper preserves ordinary native exception cleanup,
    # then represents our deliberate stop as a FailedOperation.
    if (stopped is None and result.get('success') is False
            and (result.get('error') or {}).get('error_type') == 'DiagnosticStop'
            and observer.details.get('lw_passed') is True):
        stopped = 'first-node localization passed; stopped before refinement/PFIT'
    analysis = observer.validate()
    if (original_codes != tuple(function.__code__ for function in observer.functions)
            or sha(args.runtime.read_bytes()) != RUNTIME_SHA256):
        raise ValueError('observer identities or runtime changed')
    if stopped is not None:
        if observer.details.get('lw_passed') is not True:
            raise ValueError('diagnostic stopped before a completed first-node LW observation')
        discriminator = 'FIRST_NODE_LW_PASSED_NO_MODEL_FIT'
    else:
        failure = result.get('error') or {}
        if (result.get('success') is not False
                or 'localize_lw: postcondition exceeds residual tolerance' not in str(failure.get('error_message'))):
            raise ValueError('fresh run did not reach the expected production LW refusal')
        discriminator = 'FIRST_NODE_PRODUCTION_LW_REFUSAL_CAPTURED'
    stdout = (result or {}).get('input_data', {}).get('stdout', '')
    if not isinstance(stdout, str) or len(stdout.encode()) > 16*1024**2:
        raise ValueError('bounded first-node stdout required')
    stdout_path = payload/'scientific-output.log'
    with stdout_path.open('x') as stream:
        stream.write(stdout)
    for binding in bindings.values():
        if file_sha(binding['file']) != binding['sha256']:
            raise ValueError('installed observer source changed')
    report = dict(status='COMPLETE_DIAGNOSTIC_NOT_A_MODEL', discriminator=discriminator,
        scientific_commit=SCIENTIFIC_COMMIT, operational_commit=args.commit,
        runtime_input_sha256=RUNTIME_SHA256, build_identity_sha256=BUILD_SHA256,
        core_sha256=file_sha(psi4.core.__file__), production_tolerance=1e-6,
        production_cap_unchanged=32, bindings=bindings, stages=observer.counts,
        observation=observer.details, analysis=analysis, error=failure, stopped=stopped,
        stdout_sha256=file_sha(stdout_path), stdout_available=result is not None,
        captured_arrays=observer.capture.records, captured_numeric_bytes=observer.capture.total_bytes,
        diagnostic_capture_ceiling_bytes=MAX_CAPTURE_BYTES,
        qualification='Fresh regeneration, not the lost original operands. Original production '
            'functions observed without replacement; unchanged scientific inputs/tolerance. '
            'No second frequency, PFIT/export, model acceptance, QCArchive connection or record mutation. '
            'Diagnostic buffers/I/O are separate from production ledger and bounded to512MiB '
            'beside40GiB scientific memory within64GiB job; not a whole-process RSS certificate.')
    write_json(payload/'charge-diagnostic.json', report)


def run_parent(args):
    if Path(sys.executable).resolve() != (MANAGER/'bin/python').resolve():
        raise ValueError('manager interpreter required for diagnostic parent')
    source = Path(__file__).resolve().parents[2]
    operations = source_identities(source, args.commit)
    checks = helper('phoenix_installed_checks')
    common = helper('phoenix_descriptor_bank_cb63305')
    allocation = checks.allocation()
    admit_allocation(allocation)
    run = args.run.resolve()
    if (args.run.is_symlink() or run.parent != RUNS
            or run.name != 'ccasp-native-s66-ed561666fc-charge-diagnostic001'
            or not (run/'logs').is_dir()):
        raise ValueError('fresh isolated approved diagnostic001 run required')
    manifest_path = BUILD/'build-identity.json'
    if file_sha(manifest_path) != BUILD_SHA256:
        raise ValueError('pinned qualified build identity required')
    manifest = json.loads(manifest_path.read_bytes())
    if manifest['commit'] != SCIENTIFIC_COMMIT:
        raise ValueError('unchanged cap32 scientific producer required')
    environment = BUILD/'env'
    qualified = qualified_receipts()
    checks.verify_runtime(environment, manifest)
    runtime_raw = bound_runtime(CASE.read_bytes())
    payload = run/'payload'
    payload.mkdir(exist_ok=False)
    scratch = SCRATCH/('ccasp-native-charge-diagnostic-'+os.environ['SLURM_JOB_ID'])
    scratch.mkdir(mode=0o700, exist_ok=False)
    for name in ('home', 'cache', 'cwd'):
        (scratch/name).mkdir()
    snapshot = helper('phoenix_pilot_cb63305').scratch_snapshot(
        scratch, payload, helper('descriptor_scratch_cb63305'),
        execution_profile=dict(scratch_bytes=256*1024**3, scratch_files=131072))
    runtime_path = scratch/'runtime-input.json'
    with runtime_path.open('xb') as stream:
        stream.write(runtime_raw)
    env = common.child_environment(environment, scratch)
    env.update(OMP_NUM_THREADS='12', MKL_NUM_THREADS='12')
    write_json(payload/'diagnostic-started.json', dict(
        allocation=allocation, operations=operations, scientific_commit=SCIENTIFIC_COMMIT,
        build_identity_sha256=BUILD_SHA256, runtime_input_sha256=RUNTIME_SHA256,
        case_sha256=CASE_SHA256, scratch=str(scratch),
        policy='One isolated direct diagnostic, not a QCArchive campaign restart; no automatic retry.'))
    command = checks.installed_python_command(environment/'bin/python', scratch/'cache',
        str(Path(__file__).resolve()), '--worker', '--runtime', str(runtime_path),
        '--run', str(run), '--commit', args.commit)
    subprocess.run(command+['--smoke'], env=env, cwd=scratch/'cwd',
                   stdin=subprocess.DEVNULL, check=True)
    subprocess.run(command, env=env, cwd=scratch/'cwd', stdin=subprocess.DEVNULL, check=True)
    checks.verify_runtime(environment, manifest)
    if (source_identities(source, args.commit) != operations
            or qualified_receipts() != qualified
            or file_sha(manifest_path) != BUILD_SHA256
            or sha(CASE.read_bytes()) != CASE_SHA256
            or sha(runtime_path.read_bytes()) != RUNTIME_SHA256):
        raise ValueError('sealed source, input or installation changed')
    report_path = payload/'charge-diagnostic.json'
    report = json.loads(report_path.read_bytes())
    if (report['status'] != 'COMPLETE_DIAGNOSTIC_NOT_A_MODEL'
            or report['operational_commit'] != args.commit
            or report['scientific_commit'] != SCIENTIFIC_COMMIT
            or report['runtime_input_sha256'] != RUNTIME_SHA256
            or report['production_tolerance'] != 1e-6 or report['production_cap_unchanged'] != 32):
        raise ValueError('complete bound diagnostic report required')
    write_json(payload/'diagnostic-receipt.json', dict(
        status='COMPLETE_DIAGNOSTIC_NOT_A_MODEL', allocation=allocation,
        operational_commit=args.commit, scientific_commit=SCIENTIFIC_COMMIT,
        build_identity_sha256=BUILD_SHA256, operational_sources_sha256=operations,
        report_sha256=file_sha(report_path), scratch_snapshot=snapshot,
        runtime_unchanged=True, no_qcarchive_connection=True,
        discriminator=report['discriminator']))
    print(json.dumps(dict(status=report['status'], discriminator=report['discriminator'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--runtime', type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.runtime is None:
            parser.error('--worker requires --runtime')
        run_worker(args)
    else:
        if args.smoke or args.runtime:
            parser.error('parent derives its one pinned runtime; worker-only flags refused')
        run_parent(args)


if __name__ == '__main__':
    main()
