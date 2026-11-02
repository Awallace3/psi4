"""Version2 arithmetic policy for the qualified cb63305 reference-H2H1 route.

This deliberately leaves the acf54 mirror/receipts unchanged. Whole-chain
formulas are retained; PFIT arithmetic comes from the exact shipped pure
planner. No Psi4 import, SCF, array, integral or grid is constructed here.
An unsigned calculator result is not descriptor/allocation admission.
"""
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys

SCIENTIFIC_COMMIT = 'cb63305f4115526417a830829a0a30c36b85b4e8'
POLICY = 'cb63305-reference-h2h1-ledger-v2'
FILES = {
    'psi4/driver/procrouting/isapol_bounded.py': '2258f8b44ff6d50f7c9f9fa91fa329e490bf241c258882454ad82ed7251eed11',
    'psi4/driver/procrouting/isapol_bounded_response.py': 'b8e9956e76abae3f9876b7b8cbe3a00bdc39c652cd4a4c57e2578e5d89b35a08',
    'psi4/driver/procrouting/isapol_distribution.py': '670d3b6a2cdfe3a2738a697405a7b769ac0291e5a998a9b8968e2571f41f87b9',
    'psi4/driver/procrouting/isapol_native_factors.py': '7d0187360751024d6774307d486beec7ab77c5d2c45f7bffa3fcf30b5f3bbc43',
    'psi4/driver/procrouting/isapol_pfit_stream.py': 'b02567f96dc9cfbc266ec3ac9ecc2b38b317dc7bd026ed090c0e3e618da3b020',
    'psi4/driver/procrouting/isapol_molecular.py': 'c1fbe8c9a67d230d991727afccc23f7b65e3ea84aa0e6dd5299e20b7907c6546',
    'psi4/driver/procrouting/isapol_recipes.py': '0d490627b7711a614e632080c373c7a94dd199dc8d5918b16422bd2913af277e',
    'psi4/driver/procrouting/isapol_native_partition.py': '321ea2530436277293f02843cc4c73b6d62892043dc6d77360500c4172d823fb',
    'psi4/src/psi4/libisapol/native_response.cc': 'd27e84107c4d975e7e02a97d5116e1f5d84b8767ebac2b789e91cd21ca753830',
    'psi4/src/psi4/libisapol/pfit.cc': '16354ebbe6212944fbc5018d3388bf2bfa2256239d0eb8a6ae5429564d948cbe',
    'psi4/src/psi4/libisapol/pfit.h': '7e0a092c5fd4ed887ef8c30ecd6a27f2a646a91398d292d0460cd90f8484cb8f',
    'psi4/src/psi4/libisapol/isa_grid.cc': 'd608c2850bef6d5bc4073c0e28f681a1de1549d20717af005f792c81e88958b6',
    'psi4/src/psi4/libfock/cubature.cc': '7458fddd6662582fe6a756314c456e03089706ea5518135f3fd0f012f929172f',
    'psi4/driver/procrouting/isapol_pfit_plan.py': 'd77c1bab49acb3c01185f8e1a974c1145d6ce11e101cb9c62b2b274d5c297841',
    'psi4/driver/procrouting/isapol_schema.py': '1ce6386fe5b25a13acb275ebad88e6f1f4a9bdc11dc35f6fea177696346a4795',
    'psi4/driver/procrouting/isapol_bounded_oeprop.py': '3cc990b17d23a9aebd0d3076d897faf9ee27050c2f0aca4a09caae2a8cea53cb',
}
MIB, GIB = 1024**2, 1024**3
PFIT_SOURCE = 'psi4/driver/procrouting/isapol_pfit_plan.py'


def source_binding(root):
    observed = {p: hashlib.sha256((Path(root)/p).read_bytes()).hexdigest() for p in FILES}
    if observed != FILES:
        raise ValueError('scientific planning sources differ from cb63305 policy')
    return dict(scientific_commit=SCIENTIFIC_COMMIT, policy_id=POLICY, files_sha256=observed)


def shared_pfit_plan():
    """Load only a fixed, hash-verified pure helper; never import Psi4 here.

    A descriptor child has already imported the installed native interface.
    An offline calculator instead uses this repository's fixed source path.
    There is no CLI/helper-path override, search-path fallback or cap retry.
    Complete installed-runtime/path qualification remains the supervisor's job.
    """
    installed = sys.modules.get('psi4.driver.procrouting.isapol_pfit_plan')
    path = (Path(installed.__file__) if installed is not None
            else Path(__file__).resolve().parents[2]/PFIT_SOURCE)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FILES[PFIT_SOURCE]:
        raise ValueError('shipped PFIT helper differs from cb63305 policy')
    spec = importlib.util.spec_from_file_location('_cb63305_verified_pfit_plan', path)
    module = importlib.util.module_from_spec(spec)
    # SourceFileLoader may read an accepted .pyc even with -B and would reopen
    # the file. Execute exactly the immutable bytes just hashed, never a cache
    # entry or a source replacement between verification and execution.
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


@dataclass(frozen=True)
class Dimensions:
    sites: int
    bonds: int
    main_nbf: int
    main_nshell: int
    nocc: int
    response_aux_width: int
    density_aux_width: int
    atomic_widths: tuple
    shape_widths: tuple

    def __post_init__(self):
        for key in ('sites', 'main_nbf', 'main_nshell', 'nocc',
                    'response_aux_width', 'density_aux_width'):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be a positive integer')
        if not 3 <= self.sites <= 17 or not 0 < self.nocc < self.main_nbf:
            raise ValueError('profile site/occupied dimensions')
        if type(self.bonds) is not int or not 0 <= self.bonds <= self.sites*(self.sites-1)//2:
            raise ValueError('invalid declared bond count')
        if self.main_nshell > self.main_nbf or self.response_aux_width != self.density_aux_width:
            raise ValueError('profile MAIN/AUX descriptors mismatch')
        for key in ('atomic_widths', 'shape_widths'):
            widths = getattr(self, key)
            if (type(widths) is not tuple or len(widths) != self.sites
                    or any(type(w) is not int or w <= 0 for w in widths)):
                raise ValueError('positive ordered per-site widths required')


def plan(dimensions, resources, *, max_replay_work, pfit_maximum_work_bytes):
    """Predict the unchanged full ledger, with E+native-cap in the fit stage.

    Native-base and packed subbudget refusals propagate unchanged from the
    shipped planner, before this calculator can issue even a prediction.
    Whole-chain work, byte, I/O and quadrature failures remain explicit
    REFUSED predictions. Neither kind is a molecular/solver certificate.
    """
    return _ledger(dimensions, resources, max_replay_work=max_replay_work,
        pfit_maximum_work_bytes=pfit_maximum_work_bytes, drho_refinement_iterations=10,
        scientific_commit=SCIENTIFIC_COMMIT, policy=POLICY)


def _ledger(dimensions, resources, *, max_replay_work, pfit_maximum_work_bytes,
            drho_refinement_iterations, scientific_commit, policy):
    """Shared arithmetic; public wrappers supply closed producer constants.

    Neither a wrapper nor this private calculator performs descriptor,
    installation or molecular admission. Historical plan() stays cap10.
    """
    if type(drho_refinement_iterations) is not int or drho_refinement_iterations not in (10, 32):
        raise ValueError('supported closed producer refinement budget required')
    if not isinstance(dimensions, Dimensions):
        raise ValueError('explicit Dimensions required')
    if type(resources) is not dict or set(resources) != {
            'max_bytes', 'max_work', 'max_io_bytes', 'capacity'}:
        raise ValueError('exact finite resource declaration required')
    ceilings = {'large': 100*10**12, 'extended': 128*10**12}
    if type(resources['capacity']) is not str or resources['capacity'] not in ceilings:
        raise ValueError('explicit large or extended resource capacity required')
    for key, ceiling in (('max_bytes', None), ('max_work', ceilings[resources['capacity']]),
                         ('max_io_bytes', 64*GIB)):
        value = resources[key]
        if type(value) is not int or value <= 0 or (ceiling is not None and value > ceiling):
            raise ValueError(f'invalid {key}')
    if pfit_maximum_work_bytes is None:
        raise ValueError('explicit finite native PFIT workspace budget required')
    pfit = shared_pfit_plan()
    cap = pfit.native_work_limit(pfit_maximum_work_bytes)
    pfit.check_replay_work(max_replay_work)
    s, e, b, z, o, p, d = (getattr(dimensions, k) for k in (
        'sites', 'bonds', 'main_nbf', 'main_nshell', 'nocc',
        'response_aux_width', 'density_aux_width'))
    m, v = b, b-o
    n, q, f, k, c, u, L = o*v, 25*s, 11, 36*s, 9*s, 8*s, 2000
    R = L*(L+1)//2
    fit = pfit.combined_fit_plan(L, c, k, f, maximum_work_bytes=cap)
    packed = pfit.packed_design_plan(npoint=L, channel_count=c, parameter_count=k,
        used_channels=u, block_rows=4096, max_bytes=resources['max_bytes'],
        max_work=max_replay_work)
    packed_overhead = pfit.streamed_overhead(L, c, f, 4096, computed_fields=True)
    # Unchanged whole-chain mirror of cb63305's inherited scientific policy.
    r, h, g = s*99*230, s*199*974, s*99*406
    aa, t = dimensions.atomic_widths, sum(dimensions.shape_widths)
    a2, a = sum(w*w for w in aa), sum(aa)
    gr, gh = 32*r, 32*h
    state = 512*b*b+16*MIB+16384*z+4096*s
    frequency = f*(20000*s+32*k*k)
    b0, base = 2*gr+gh+state+32*b*b, gr+gh+frequency
    targets, static = 8*(R*f+3*L), 8*s*81
    isa_bytes = 8*(g*(a+40*s+40)+(120+12)*(a2+16*a+16*t)
                   +h*(32*s+16)+8*p*q+16*d*d+8*d*b*b+64*b*b)+256*MIB+8*(4*d+134)
    isa_work = (120*(8*g*(a2+s*t)+8*sum(w**3 for w in aa))
                +16*d*b**3+8*d**3+4*h*q*p+16*d*d+64*d
                +drho_refinement_iterations*(256*d*(d+1)+16384*d+16*d*d+64*d))
    storage = 8*(p*(o*o+2*n+v*v+1)+n)
    factors = (storage+24*p*p+64*p*min(512, max(v*v, n))+8*p*max(o*o, n)
               +8*(24*b*b+18*m*m+4*b*m)+32*(n+m+p))
    constrained = storage+16*p*n+24*p*p+64*p*min(32, n)+64*p+32*n
    contraction = 32*p*q+16*n*q
    operator = 8*(n*n+2*n*(p+q))
    construction = 8*(2*n*n+n*(p+q))+8*MIB
    node = 8*(n*n+2*n*(p+q)+3*n*512+p*p+q*q+16*n)+8*MIB
    local = 15000*s*s+5000*(2*e+s)+4632*s+128*s*s+192000000+8*p*p+32*q*q
    target = 8*(3*L+3*32+2*p*L+L*L+p*32+R)+8*p*p
    stages = []

    def stage(name, reserved, transient=0, work=0, count=1):
        stages.append(dict(name=name, reserved_bytes=reserved, transient_bytes=transient,
                           numeric_bytes=reserved+transient, work_per_call=work,
                           calls=count, charged_work=work*count))

    stage('state and grid', b0)
    stage('ISA iteration and response-AUX Q', b0, isa_bytes, isa_work)
    stage('retained distributed moments', b0+8*p*q)
    stage('native factors', b0+8*p*q, factors, 8*p*m**3+4*p*p*m*m+2*p**3)
    stage('anchor fit', b0+8*p*q, constrained+contraction, 2*p**3+4*p*p*n)
    stage('distributed moment contraction', b0+8*p*q, constrained+contraction, 2*n*p*q)
    stage('target fit', b0, constrained+contraction, 2*p**3+4*p*p*n)
    stage('full-grid AUX kernel', 2*gr+gh, 8*(4*p*p+8*2048*p+64*2048)+2*gr, 2*r*p*p)
    stage('dense operator assembly', base, 8*(3*n*n+4*n*p+p*p)+16*MIB, 8*p*n*n+2*n*p*p)
    stage('operator stability', base, 32*n*n+MIB, 4*n**3)
    stage('native fit cloud', base, 96*L+8*MIB)
    stage('retained point-response targets', base+targets)
    stage('retained static reference tensors', base+targets+static)
    stage('H2H1 response operator', base+targets+static, operator+construction,
          int(2*n**3+2*n*n*(p+q)))
    live = base+targets+static+operator
    stage('frequency response', live, node, int(2*n**3/3+6*n*n*(p+q)), f)
    stage('production localization', live, local, 8*q**3, f)
    stage('point-response targets', live, target, 2*p*p*L+2*p*L*L, f)
    stage('complete-cloud refinement', base+targets, fit['complete_refinement_bytes'],
          fit['complete_refinement_work'])
    peak = max(t['numeric_bytes'] for t in stages)
    work = sum(t['charged_work'] for t in stages)
    io = (8*(2*n+2*p*o*o+7*p*n+2*p+2*p*v*v+2*n*q+2*p*p+6*n*n)
          +256*(21+2*((n+511)//512)+2*((v*v+511)//512)))
    frequency_bytes = operator+max(construction, node)+base
    frequency_work = f*int(2*n**3+2*n**3/3+8*n*n*(p+q))
    refusals = []
    for key, value, limit in (
        ('cumulative_work', work, resources['max_work']),
        ('checkpoint_io', io, resources['max_io_bytes']),
        ('numeric_bytes', peak, resources['max_bytes']),
        ('packed_replay_bytes', packed['planned_bytes']+packed_overhead+base+targets,
         resources['max_bytes']),
        ('quadrature_bytes', frequency_bytes, resources['max_bytes']),
        ('quadrature_work', frequency_work, resources['max_work']),
        ('initial_grid_bytes', 3*gr, resources['max_bytes'])):
        if value > limit:
            refusals.append(key)
    return dict(
        schema='native-campaign-ledger-prediction/2', policy_id=policy,
        status='REFUSED' if refusals else 'WITHIN_PREDICTED_LEDGER_LIMITS',
        policy_scientific_commit=scientific_commit, resources=dict(resources),
        qualification='Arithmetic prediction only; no source/build/descriptor receipt supplied '
                      'to this pure calculator. A separately verified supervisor is required.',
        max_replay_work=max_replay_work, pfit_maximum_work_bytes=cap, refusals=refusals,
        dimensions=dict(asdict(dimensions), nmo_envelope=m, nvir_envelope=v,
                        ov_envelope=n, nodes=f, parameters=k, channels=c, used_channels=u,
                        moment_channels=q, fit_points=L, isa_iterations=120,
                        drho_refinement_iterations=drho_refinement_iterations, response_grid_rows=r,
                        integration_grid_rows=h, isa_grid_accounting_rows=g),
        stages=stages, charged_work=work, peak_numeric_bytes=peak,
        peak_stage=max(stages, key=lambda t: t['numeric_bytes'])['name'],
        admission_numeric_bytes=peak, checkpoint_io_bytes=io, fit_plan=fit,
        work_by_stage={t['name']: t['charged_work'] for t in stages},
        packed_replay=dict(planned_bytes=packed['planned_bytes'], planned_work=packed['planned_work'],
                           overhead_bytes=packed_overhead, row_count=R, chunk=packed['chunk'],
                           block_rows=4096, passes=2),
        native_pfit=dict(base_bytes=fit['native_base_bytes'], maximum_work_bytes=cap,
                         source_bytes=fit['native_source_bytes'], solver='NormalEquationsDSYSV',
                         workspace_query='Runtime gate; queried workspace must fit above native base',
                         cap_reservation_bytes=cap, included_in_refinement_stage=True),
        exclusions=[
            'SCF, MAIN adaptation, integral-engine operations and vendor workspaces',
            'Python object overhead, total process RSS, actual disk/transport and wall time',
            'Uncharged isotropic contractions and export/serialization',
            'Input/output tensors are numeric plans, not measurements'],
        execution_gates=[
            'Genuine converged SCF, actual nmo<=nbf envelope and canonical positive-gap state',
            'ISA convergence and Q; finite metrics, H1/H2 stability and all response/LW residuals',
            'Native queried LAPACK workspace under enforced cap and PFIT solver success',
            'Full bound runtime/record audit; no scientific accuracy or covariance certification'])
