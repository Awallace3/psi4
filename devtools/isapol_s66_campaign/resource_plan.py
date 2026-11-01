"""Operational prediction of acf54's reference-H2H1 ledger, not a new producer.

No Psi4, wavefunction, integral, point grid or array is constructed here.
This mirrors pinned accounting formulas; it is not a bound on total process
memory, SCF work, vendor workspaces, wall time or scientific convergence.
"""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

SCIENTIFIC_COMMIT = 'acf54c28e50822b5b825f4d26e5d9f9a2835d773'
FILES = {
    'psi4/driver/procrouting/isapol_bounded.py': '84d124346ecb2151036b16a9a1030fa83b5e3cfe11ab74a4ec42cb9ea8cf36da',
    'psi4/driver/procrouting/isapol_bounded_response.py': '7e1c26462d659fc2226daa06facdb19b7709efcf7cff2e74ccdb31995573f65c',
    'psi4/driver/procrouting/isapol_distribution.py': '670d3b6a2cdfe3a2738a697405a7b769ac0291e5a998a9b8968e2571f41f87b9',
    'psi4/driver/procrouting/isapol_native_factors.py': '7d0187360751024d6774307d486beec7ab77c5d2c45f7bffa3fcf30b5f3bbc43',
    'psi4/driver/procrouting/isapol_pfit_stream.py': '7a58faf10fdcdfa5656e11ae3cd76b59ca759aee4cce35d81df9eac7dcfa0dbb',
    'psi4/driver/procrouting/isapol_molecular.py': '28b6be4e1043ecbfe7dad80d48cf432226016a288b2e07f58ff1f31174a05ef7',
    'psi4/driver/procrouting/isapol_recipes.py': '0d490627b7711a614e632080c373c7a94dd199dc8d5918b16422bd2913af277e',
    'psi4/driver/procrouting/isapol_native_partition.py': '321ea2530436277293f02843cc4c73b6d62892043dc6d77360500c4172d823fb',
    'psi4/src/psi4/libisapol/native_response.cc': 'd27e84107c4d975e7e02a97d5116e1f5d84b8767ebac2b789e91cd21ca753830',
    'psi4/src/psi4/libisapol/pfit.cc': '16354ebbe6212944fbc5018d3388bf2bfa2256239d0eb8a6ae5429564d948cbe',
    'psi4/src/psi4/libisapol/pfit.h': '7e0a092c5fd4ed887ef8c30ecd6a27f2a646a91398d292d0460cd90f8484cb8f',
    'psi4/src/psi4/libisapol/isa_grid.cc': 'd608c2850bef6d5bc4073c0e28f681a1de1549d20717af005f792c81e88958b6',
    'psi4/src/psi4/libfock/cubature.cc': '7458fddd6662582fe6a756314c456e03089706ea5518135f3fd0f012f929172f',
}
MIB = 1024**2
GIB = 1024**3


def source_binding(root):
    """Bind the external mirror to actual source bytes, including native plans."""
    observed = {p: hashlib.sha256((Path(root)/p).read_bytes()).hexdigest() for p in FILES}
    if observed != FILES:
        raise ValueError('scientific planning sources differ from acf54 policy')
    return dict(scientific_commit=SCIENTIFIC_COMMIT, files_sha256=observed)


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


def plan(dimensions, resources, *, max_replay_work):
    """Replay charged work, checkpoint I/O and live numeric reservations.

    The pre-SCF orbital envelope uses nmo <= nbf, never a synthetic SCF state.
    Native PFIT's whole enforced cap is additionally reserved as a separate
    conservative admission policy; queried LAPACK workspace remains a runtime
    gate. Neither this extra allowance nor quadrature prechecks alter ledger
    totals. Full rank2 variables, all sites/cloud rows and 11 nodes are retained.
    """
    if not isinstance(dimensions, Dimensions):
        raise ValueError('explicit Dimensions required')
    if type(resources) is not dict or set(resources) != {
            'max_bytes', 'max_work', 'max_io_bytes', 'capacity'}:
        raise ValueError('exact finite resource declaration required')
    ceilings = {'standard': 6*10**12, 'large': 100*10**12}
    if type(resources['capacity']) is not str or resources['capacity'] not in ceilings:
        raise ValueError('closed resource capacity required')
    if resources['capacity'] != 'large':
        raise ValueError('this molecular profile requires explicit large capacity')
    for key, ceiling in (('max_bytes', None), ('max_work', ceilings[resources['capacity']]),
                         ('max_io_bytes', 64*GIB)):
        value = resources[key]
        if type(value) is not int or value <= 0 or (ceiling is not None and value > ceiling):
            raise ValueError(f'invalid {key}')
    if type(max_replay_work) is not int or not 0 < max_replay_work <= 2*10**12:
        raise ValueError('explicit finite packed replay allowance required')
    s, e, b, z, o, p, d = (getattr(dimensions, k) for k in (
        'sites', 'bonds', 'main_nbf', 'main_nshell', 'nocc',
        'response_aux_width', 'density_aux_width'))
    m, v = b, b-o
    n, q, f, k, c, u, L = o*v, 25*s, 11, 36*s, 9*s, 8*s, 2000
    R = L*(L+1)//2
    # Exact profile policy: actual Lebedev230 response rows, ISA charge bound406.
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
                +10*(256*d*(d+1)+16384*d+16*d*d+64*d))
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
    fit_bytes = 8*(5*L*c+L*c*k+2*64*L*k+16*k*k*f+4096*(8*k+32*f))+8*MIB
    fit_work = (2*((R+64*L)*2*c*k+R*(4*k+8))
                +2*R*k*k+4*R*k*f+f*(64*k**3+64*k*k))
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
    stage('complete-cloud refinement', base+targets, fit_bytes, fit_work)
    peak = max(t['numeric_bytes'] for t in stages)
    work = sum(t['charged_work'] for t in stages)
    io = (8*(2*n+2*p*o*o+7*p*n+2*p+2*p*v*v+2*n*q+2*p*p+6*n*n)
          +256*(21+2*((n+511)//512)+2*((v*v+511)//512)))
    chunk = max(1, min(64, 16*4096//L))
    packed_bytes = (9*L*c+8*L*(u*(k+1)+1)
                    +16*(L*chunk*k+4096*(2*k+12))+256)
    packed_work = 2*((R+64*L)*2*u*k+R*(4*k+8))
    packed_overhead = 24*L+8*L*c+17*4096*f
    # Native streamed source owns a capacity*(parameters+nodes) block and a row.
    source_bytes = 8*(4096*(k+f)+k)
    native_base = 8*(f*(40*k*k+64*k+16)+4096*(k+2*f))+source_bytes
    native_cap = 256*MIB
    # Additional policy, not a correction or reinterpretation of ledger. It
    # conservatively adds the full native cap beside Python's whole fit plan.
    admission_peak = max(peak, base+targets+fit_bytes+native_cap)
    frequency_bytes = operator+max(construction, node)+base
    frequency_work = f*int(2*n**3+2*n**3/3+8*n*n*(p+q))
    refusals = []
    for key, value, limit in (
        ('cumulative_work', work, resources['max_work']),
        ('checkpoint_io', io, resources['max_io_bytes']),
        ('numeric_bytes', admission_peak, resources['max_bytes']),
        ('packed_replay_bytes', packed_bytes+packed_overhead+base+targets, resources['max_bytes']),
        ('packed_replay_work', packed_work, max_replay_work),
        ('quadrature_bytes', frequency_bytes, resources['max_bytes']),
        ('quadrature_work', frequency_work, resources['max_work']),
        ('initial_grid_bytes', 3*gr, resources['max_bytes']),
        ('native_pfit_base_bytes', native_base, native_cap)):
        if value > limit:
            refusals.append(key)
    return dict(
        schema='native-campaign-ledger-prediction/1',
        status='REFUSED' if refusals else 'WITHIN_PREDICTED_LEDGER_LIMITS',
        policy_scientific_commit=SCIENTIFIC_COMMIT, resources=dict(resources),
        qualification='Arithmetic prediction only; no source/build/descriptor receipt supplied '
                      'to this pure calculator. A separately verified supervisor is required.',
        max_replay_work=max_replay_work, refusals=refusals,
        dimensions=dict(asdict(dimensions), nmo_envelope=m, nvir_envelope=v,
                        ov_envelope=n, nodes=f, parameters=k, channels=c, used_channels=u,
                        moment_channels=q, fit_points=L, isa_iterations=120,
                        drho_refinement_iterations=10, response_grid_rows=r,
                        integration_grid_rows=h, isa_grid_accounting_rows=g),
        stages=stages, charged_work=work, peak_numeric_bytes=peak,
        peak_stage=max(stages, key=lambda t: t['numeric_bytes'])['name'],
        admission_numeric_bytes=admission_peak, checkpoint_io_bytes=io,
        work_by_stage={t['name']: t['charged_work'] for t in stages},
        packed_replay=dict(planned_bytes=packed_bytes, planned_work=packed_work,
                           overhead_bytes=packed_overhead,
                           row_count=R, chunk=chunk, block_rows=4096, passes=2),
        native_pfit=dict(base_bytes=native_base, maximum_work_bytes=native_cap,
                         source_bytes=source_bytes, solver='NormalEquationsDSYSV',
                         workspace_query='Runtime gate; queried workspace must fit above native base',
                         extra_cap_reservation_bytes=native_cap),
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
