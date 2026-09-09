"""Fixed comparison-only profiles. No solver controls or end-to-end certification."""
import contextlib
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import math
import numbers
from pathlib import Path
import sys

DRHO_PROFILE = 'provisional-drho-1e-2'
PROFILES = ('strict', 'provisional-1e-3', DRHO_PROFILE)
STRICT = 1e-9
PROVISIONAL = 1e-3
DRHO_PROVISIONAL = 1e-2
ORACLE = Path(__file__).resolve().parent
ROOT = ORACLE.parents[3]


def load(name):
    spec = importlib.util.spec_from_file_location('provisional_' + name, ORACLE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise ValueError(message)


def contract(stage, atom_count=None, tail_atoms=()):
    density = {'pointwise_scaled_density_error', 'relative_abs_weighted_density_l2'}
    if stage == 'native-drho':
        relaxed = density | {'coefficients', 'sampled_density'}
        return relaxed | {'metric', 'rhs', 'fitted_electrons', 'relative_infinity_backward_residual'}, relaxed, {'reference_validation', 'checkpoint_identity', 'finite_measurement'}
    if stage == 'native-ov':
        relaxed = density | {'coefficients', 'sampled_transition_density'}
        return relaxed | {'metric', 'relative_backward_residual'}, relaxed, {'reference_validation', 'checkpoint_identity', 'finite_measurement'}
    if stage == 'trajectory':
        require(type(atom_count) is int and atom_count > 0, 'Invalid atom count')
        require(len(set(tail_atoms)) == len(tail_atoms) and all(type(i) is int and 1 <= i <= atom_count for i in tail_atoms), 'Invalid tail sites')
        relaxed = {f'atom{i}_tail' for i in tail_atoms}
        metrics = {'deltas', 'max_delta', 'history_max_delta', 'next_controls', 'history_controls'} | relaxed
        metrics |= {f'atom{i}_{key}' for i in range(1, atom_count + 1) for key in ('D', 'W', 'charge', 'history_charge', 'cutoff')}
        return metrics, relaxed, {'both_converged', 'iteration_count_matches', 'history_consistent', 'flags_match', 'cutoff_endpoint_invariance', 'configuration_matches'}
    raise ValueError('Unknown stage')


def evaluate(stage, metrics, checks, *, profile='strict', atom_count=None, tail_atoms=(), missing_implementations=()):
    """Reject incomplete/unknown checks. Missing algorithms block *both* statuses.

    Metrics are the original nonnegative scalar errors, not raw observables.
    Scope is the named comparison only; callers must not call an OV diagnostic
    a native API. Missing implementations required within a claimed scope must
    be supplied here, never converted into numerical checks.
    """
    require(profile in PROFILES, 'Unknown profile')
    require(profile != DRHO_PROFILE or stage == 'native-drho',
            'Drho-C-only profile cannot be used for another stage')
    forward_threshold = DRHO_PROVISIONAL if profile == DRHO_PROFILE else PROVISIONAL
    required, relaxed, structural = contract(stage, atom_count, tail_atoms)
    require(set(metrics) == required, 'Missing required checks or unknown metrics')
    require(set(checks) == structural, 'Missing required or unknown structural checks')
    require(all(type(v) is bool for v in checks.values()), 'Structural checks must be booleans')
    require(isinstance(missing_implementations, (tuple, list)) and all(isinstance(x, str) and x for x in missing_implementations), 'Invalid missing implementations')
    details = {}
    for name, value in metrics.items():
        require(isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value) and value >= 0, 'Invalid nonfinite or negative metric: ' + name)
        # Original measurement scalars are binary64. Never round a supplied
        # higher-precision value down onto the acceptance boundary.
        require(value == float(value), 'Metric is not exactly representable as an original binary64 error: ' + name)
        value = float(value)
        threshold = forward_threshold if name in relaxed else STRICT
        details[name] = dict(actual=value, metric=('original pointwise-scaled error' if name == 'pointwise_scaled_density_error' else 'original relative abs-weighted L2' if name == 'relative_abs_weighted_density_l2' else 'original backward residual' if 'residual' in name else 'max(abs(actual-reference))/max(1,max(abs(reference)))'), strict_threshold=STRICT, provisional_threshold=threshold, strict_passed=value <= STRICT, provisional_passed=value <= threshold, provisional_eligible=name in relaxed)
    for name, value in checks.items():
        details[name] = dict(actual=value, metric='strict structural requirement', strict_threshold=True, provisional_threshold=True, strict_passed=value, provisional_passed=value, provisional_eligible=False)
    strict_numeric = all(c['strict_passed'] for c in details.values())
    provisional_numeric = all(c['provisional_passed'] for c in details.values())
    strict = strict_numeric and not missing_implementations
    provisional = provisional_numeric and not missing_implementations
    selected = strict if profile == 'strict' else provisional
    todo = {'trajectory': 9, 'native-drho': 10, 'native-ov': 11}[stage]
    return dict(passed=bool(strict), strict_passed=bool(strict), provisional_passed=bool(provisional), selected_profile=profile, selected_profile_passed=bool(selected), checks=details,
                strict_numerical_passed=strict_numeric, provisional_numerical_passed=provisional_numeric,
                status='missing-implementation' if missing_implementations else 'strict-pass' if strict else 'provisional-pass' if provisional else 'failure',
                stage=stage, scope={'trajectory': 'exported fixed-density trajectory comparison', 'native-drho': 'native explicit-input C++ Drho-C comparison', 'native-ov': 'native-integral + supplied-C NumPy diagnostic; NOT production C++ OV API'}[stage], original_target=STRICT, tightening_todo=f'TODO{todo}/TODO12: tighten eligible forward comparisons to original scaled/pointwise/L2 1e-9 targets; measure downstream errors independently',
                implementation_status='missing' if missing_implementations else 'implemented-for-declared-comparison-scope', missing_implementations=list(missing_implementations))


def preflight(paths):
    paths = [Path(p).absolute() for p in paths]
    require(len({p.resolve() for p in paths}) == len(paths), 'Output destinations overlap')
    for p in paths:
        if p.exists() or p.is_symlink():
            raise FileExistsError(f'Refusing to overwrite {p}')
        if not p.parent.is_dir():
            raise FileNotFoundError(p.parent)
    return paths


@contextlib.contextmanager
def exclusive_outputs(paths):
    """Preflight all, then reserve all with O_EXCL before writing any bytes.

    A concurrent creator is never overwritten. Empty reservations made by us
    are removed if reservation fails; measurement/write failures retain evidence.
    """
    paths = preflight(paths)
    streams = []
    try:
        for p in paths:
            streams.append(p.open('xb'))
    except BaseException:
        for p, stream in zip(paths, streams):
            stream.close()
            p.unlink()
        raise
    try:
        yield streams
    finally:
        for stream in streams:
            stream.close()


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def hashes(paths):
    files = set()
    for item in paths:
        p = Path(item).resolve(strict=True)
        files.update(x for x in p.rglob('*') if x.is_file()) if p.is_dir() else files.add(p)
    result = {}
    for p in sorted(files):
        with p.open('rb') as stream:
            result[str(p)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result


def source_paths(original=None):
    # Include transitive Python helpers and the current native source, not only
    # the comparator. The loaded extension is added separately by native runs.
    paths = [*ORACLE.glob('*.py'), *(ROOT / 'psi4/src/psi4/libisapol').glob('*.[ch]*'),
             ROOT / 'psi4/src/export_isapol.cc']
    if original is not None:
        paths.append(ROOT / '.pi/audit' / original)
    return paths


def production_psi4():
    """Import installed/staged Psi4 without cwd or repository-package shadowing."""
    old = sys.path[:]
    source = ROOT / 'psi4'
    try:
        sys.path[:] = [p for p in old if p and Path(p).resolve() != ROOT and not Path(p).resolve().is_relative_to(source)]
        psi4 = importlib.import_module('psi4')
        core = Path(psi4.core.__file__).resolve(strict=True)
        require(not core.is_relative_to(source), 'Source-shadowed psi4 core')
        require(any(str(core).endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES), 'No production psi4 extension')
        return psi4
    finally:
        sys.path[:] = old
