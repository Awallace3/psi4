# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Wavefunction-first native atomic properties; no reference files or hidden SCF.

GENERATED_JKFIT_ISA_A is a deliberately modest, self-contained H/O recipe, NOT
modern CamCASP ISA-Pol. Density: lambda1000 Drho-C, ordinary ISA-A. Response:
direct occupied-fast OV, Slater/PW92 ALDA + 25% exact exchange, no GRAC by
DEFAULT, no PFIT. FIXED_GRAC is a separate explicit SCF-input acceptance policy,
not a generated matched preset or a GRAC response kernel.
The response grid and tensor localization are not density partition policies.
ATOMIC_RESPONSE_ALGORITHM names the native response arrangement and therefore
which calibrated ALDA work limit applies; it is not a screening, quadrature or
tolerance policy and relaxes no other limit.
"""
from dataclasses import dataclass
import math
import numpy as np
from psi4 import core
from . import isapol_native_partition as p
from . import isapol_native as n
from . import isapol_native_propagator as prop
from . import isapol_native_refinement as r
from . import isapol_refine as rf
from .isapol_response_preflight import estimate_response_work
from .isapol_native_correction import (functional_definition as _functional_definition,
                                       validate_correction, require_scf_seal)
from . import isapol_logging as lg
from . import isapol_df_multipoles as dfm
from . import isapol_ac_options as aco
from .isapol_native_ac import DeclaredAcProvenance

TASKS = frozenset(('ATOMIC_PARTITION', 'ATOMIC_POLARIZABILITIES', 'ATOMIC_DISPERSION',
                   'ATOMIC_REFINED_POLARIZABILITIES', 'ATOMIC_REFINED_DISPERSION'))
# The two refined tasks run the reference protocol's PFIT stage on top of the LW
# local tensors. Their numbers are a DIFFERENT MODEL from the unrefined ones, not
# a better-converged version of them, so they are published under their own
# variable names and the two are never quoted as agreeing.
REFINEMENT_TASKS = frozenset(('ATOMIC_REFINED_POLARIZABILITIES', 'ATOMIC_REFINED_DISPERSION'))
REFINEMENT_KEYS = ('ATOMIC_REFINEMENT_POINTS', 'ATOMIC_REFINEMENT_SEED',
                   'ATOMIC_REFINEMENT_LOWER_LIMIT', 'ATOMIC_REFINEMENT_UPPER_LIMIT',
                   'ATOMIC_REFINEMENT_WEIGHT_TYPE', 'ATOMIC_REFINEMENT_WEIGHT_COEFFICIENT',
                   'ATOMIC_REFINEMENT_CUTOFF', 'ATOMIC_REFINEMENT_RANK_LIMIT',
                   'ATOMIC_REFINEMENT_HYDROGEN_RANK_LIMIT')

# The declared Func-1 W-TAILS cutoff of each generated recipe NAME. The name is
# the model, so one ATOMIC_PROPERTY_RECIPE value names one tail policy and no
# combination of options can ask for a recipe that does not exist.
RECIPE_TAIL_POLICIES = {'GENERATED_JKFIT_ISA_A': 'flat_1.5_bohr',
                        'GENERATED_JKFIT_BRAGG_SLATER_TAIL_ISA_A': 'bragg_slater_1.5'}
# ATOMIC_RESPONSE_PROPAGATOR. NONE is deliberately not a PropagatorDeclaration:
# it means the propagator module is not entered at all, which is what makes
# every number recorded before this option existed still bitwise reproducible.
# EXACT_ORBITAL is that same shipped model stated rather than implied by an
# absent argument, and it is checked against the provider's own operators.
PROPAGATOR_DECLARATIONS = {'NONE': None,
                           'EXACT_ORBITAL': prop.EXACT_ORBITAL_PROPAGATOR,
                           'CAMCASP_DF': prop.CAMCASP_DF_PROPAGATOR}


def generated_recipe(wfn, radial=160, angular=590, aux_basis='cc-pVDZ-JKFIT', rank=3,
                     tail_policy='flat_1.5_bohr'):
    """Generate all effective Gaussian descriptors from shipped JKFIT and formulas.

    Molecular AUX: the Cartesian ``aux_basis`` JKFIT set, unchanged effective
    contractions.  It is a *declared* argument, not inferred from MAIN, and its
    default stays cc-pVDZ-JKFIT so this remains the same modest demo recipe it
    has always been.  The default is deliberately NOT MAIN-matched, and that is
    not free: the molecular AUX also carries the Drho-C/ISA-A density fit, so at
    a MAIN much larger than cc-pVDZ the fitted-auxiliary (constrained NN)
    response route can supply a charge-flow defect that strict production LW
    then rejects.  Measured on PBE0/aug-cc-pVTZ water with the reference GRAC
    shift, at the traced penalty lambda=1000: cc-pVDZ-JKFIT AUX gives
    input-sum-rule 6.65e-06 against the 1e-6 gate (9 of 11 nodes rejected),
    whereas aug-cc-pVTZ-JKFIT gives 2.79e-07 and passes every node.  Choosing
    the MAIN-matched AUX is therefore declaring a different model, with its own
    partition, and never a relaxation of the gate.
    AtomAux and Shape: normalized uncontracted even-tempered s Gaussians,
    exponents .1*2**k (O, k=0..16), .2*2**k (H, k=0..10). These separate roles
    are a compact radial ISA-A demonstration, not a claimed CamCASP basis alias.
    ``tail_policy`` declares the Func-1 W-TAILS cutoff and, like the AUX and the
    grid, is a model parameter rather than a tolerance, so each value gets its
    own recipe name and the two are never quoted as agreeing.  The default
    ``flat_1.5_bohr`` is this demo's own long-standing absolute cutoff, shared by
    every site.  ``bragg_slater_1.5`` is instead what a CamCASP
    ``W-TAILS R1-Multiplier = 1.5`` declaration means -- 1.5*R_Slater, hence a
    different cutoff on each element; see
    ``isapol_native_partition.bragg_slater_tail_cutoff``.  On PBE0/cc-pVDZ water
    the choice is not cosmetic: the r**4-weighted site multipoles and therefore
    C8/C10 move by percent, because the cutoff sets where the Gaussian shape is
    replaced by its fitted exponential.
    """
    mol = wfn.molecule()
    if any(mol.Z(i) not in (1, 8) for i in range(mol.natom())):
        raise ValueError('GENERATED_JKFIT_ISA_A currently supports real H/O nuclei only')
    centres = tuple((mol.x(i), mol.y(i), mol.z(i)) for i in range(mol.natom()))
    aux = core.BasisSet.build(mol, 'DF_BASIS_SCF', aux_basis, puream=0)
    shells = []
    for j in range(aux.nshell()):
        s = aux.shell(j)
        shells.append(p.ShellRecipe(int(aux.shell_to_center(j)), int(s.am),
            tuple(s.exp(k) for k in range(s.nprimitive)), tuple(s.coef(k) for k in range(s.nprimitive))))
    name = {'flat_1.5_bohr': 'GENERATED_JKFIT_ISA_A',
            'bragg_slater_1.5': 'GENERATED_JKFIT_BRAGG_SLATER_TAIL_ISA_A'}.get(tail_policy)
    if name is None:
        raise ValueError(f'Unknown declared tail policy {tail_policy!r}')
    origin = (f'runtime Psi4 shipped {aux_basis} plus normalized even-tempered radial s '
              'recipe; NOT modern CamCASP preset')
    if tail_policy != 'flat_1.5_bohr':   # keep the default recipe's origin text byte-identical
        origin += f'; {tail_policy} W-TAILS cutoff'
    auxiliary = p.BasisRecipe(f'{aux_basis} Cartesian molecular AUX', origin, 'Cartesian',
                              centres, tuple(shells))
    sites = []
    for i, c in enumerate(centres):
        oxygen = mol.Z(i) == 8
        exps = tuple((.1 if oxygen else .2)*2**k for k in range(17 if oxygen else 11))
        radial_shells = tuple(p.ShellRecipe(0, 0, (a,), ((2*a/math.pi)**.75,)) for a in exps)
        atom = p.BasisRecipe('even-tempered radial AtomAux', origin, 'Spherical', (c,), radial_shells)
        shape = p.BasisRecipe('even-tempered s Shape', origin, 'Spherical', (c,), radial_shells)
        cutoff = 1.5 if tail_policy == 'flat_1.5_bohr' else p.bragg_slater_tail_cutoff(int(mol.Z(i)), 1.5)
        sites.append(p.SiteRecipe(f'{mol.symbol(i)}{i+1}', c, atom, shape, tuple(range(len(exps))),
                                  rank, cutoff, True))
    return p.PartitionRecipe(name, origin, 'explicit_cartesian_drho_c_isa_a',
        auxiliary, tuple(sites), p.GridRecipe(radial, angular, 3, 1., 'native_tabulated_bragg_slater',
        'all_sites_unscreened_full_molecular_grid'),
        # w_eps_activation / positive_activation / tail_activation are CamCASP's
        # declared wEps_EpsNorm = 1e-5, PositiveW_EpsNorm = 1e-5 and
        # TailFix_EpsNorm = 1e-6.  The tail threshold is a declared model
        # parameter: a run at 1e-5 is a different model, not a looser one.
        p.ControllerRecipe(1e-9, 120, .17, .001, .2, True, 0., True, 1e-36,
                           1e-5, 1e-5, 1e-6, 0., 20, 20, True), 'strict1e-9')


@dataclass(frozen=True)
class AtomicPropertyResult:
    """Owned result of one request; no wavefunction or mutable global option cache.

    Core getters return copies; individual core records may be caller mutable.
    Taking the accessor transfers this request's result to the caller; later
    requests never mutate old results. Failure diagnostics remain accessible.
    """
    tasks: tuple
    options: tuple
    partition: object
    properties: object
    scf_commutator_maxabs: float
    correction_provenance: object = None
    refinement: object = None

    @property
    def atomic_scalars(self):
        if self.properties is None:
            raise RuntimeError('Atomic response was not requested')
        return self.properties.atomic_scalars.array

    @property
    def dispersion(self):
        return None if self.properties is None else self.properties.dispersion

    @property
    def refined_dispersion(self):
        """The PFIT-refined isotropic C_n, or None if no refinement was requested.

        Deliberately a separate accessor from ``dispersion``: the refined and the
        unrefined coefficients are two different models of the same molecule and
        neither is a correction to the other.
        """
        return None if self.refinement is None else self.refinement.dispersion


def atomic_property_result(wfn):
    """Return the owned latest native oeprop result (oeprop itself returns None).

    No property calculation is performed. Raises if no result is attached.
    """
    result = getattr(wfn, '_native_atomic_property_result', None)
    if result is None:
        raise ValueError('No native atomic property result on this wavefunction')
    return result


def _validate_pbe0(wfn, *, scf_correction='NONE', expected_grac_shift=None,
                   ac_declaration=None):
    return validate_correction(wfn, scf_correction=scf_correction,
                               expected_grac_shift=expected_grac_shift,
                               ac_declaration=ac_declaration, require_canonical=True)


def _distribution_options(effective):
    """Translate the declared distribution/response-basis options into kwargs.

    The two declarations are coupled -- the DF-centre rule has no meaning without
    auxiliary columns -- so they are read together here and refused together,
    rather than letting ``native_properties`` discover the contradiction after a
    partition has already been paid for. ``lambda``/``eta`` belong to the
    constrained transition fit only, so declaring either one under DIRECT_OV,
    which forms no fit, is a contradiction and is refused rather than ignored.
    """
    distribution = str(effective['ATOMIC_MULTIPOLE_DISTRIBUTION']).lower()
    basis = {'direct_ov': 'direct_ov',
             'fitted_auxiliary': 'fitted_auxiliary'}[str(effective['ATOMIC_RESPONSE_BASIS']).lower()]
    penalty = float(effective['ATOMIC_OV_CHARGE_PENALTY'])
    damping = float(effective['ATOMIC_OV_METRIC_DAMPING'])
    if distribution in dfm.DF_CENTRE_DISTRIBUTIONS and basis == 'direct_ov':
        raise ValueError(
            f'ATOMIC_MULTIPOLE_DISTRIBUTION {distribution.upper()} charges each auxiliary '
            'function to the centre it sits on, so it needs auxiliary columns; '
            'ATOMIC_RESPONSE_BASIS DIRECT_OV has occupied-virtual product columns and no '
            'centre for a function to sit on. Declare ATOMIC_RESPONSE_BASIS '
            'FITTED_AUXILIARY (the reference protocol declares '
            'ATOMIC_OV_CHARGE_PENALTY 1000 and ATOMIC_OV_METRIC_DAMPING 0.0005 with it).')
    if basis == 'direct_ov' and (penalty != 1. or damping != 0.):
        raise ValueError('ATOMIC_RESPONSE_BASIS DIRECT_OV forms no transition-density fit, '
                         'so ATOMIC_OV_CHARGE_PENALTY and ATOMIC_OV_METRIC_DAMPING have '
                         'nothing to apply to and must stay at their defaults')
    return dict(distribution=distribution, response_basis=basis,
                ov_charge_penalty=penalty, ov_metric_damping=damping)


def _propagator_option():
    """Translate the declared propagator option into the ``native_properties`` kwarg.

    Read from the ambient options rather than from a passed dict because it is
    validated before any SCF-derived object is touched: CAMCASP_DF projects its
    AUX-space ALDA kernel through the fitted transition density, so it is coupled
    to ATOMIC_RESPONSE_BASIS exactly as the DF-centre distribution is, and the
    contradiction is refused here instead of after a partition has been paid for.
    """
    declared = str(core.get_global_option('ATOMIC_RESPONSE_PROPAGATOR')).upper()
    if declared not in PROPAGATOR_DECLARATIONS:
        raise ValueError('ATOMIC_RESPONSE_PROPAGATOR must be a declared '
                         + ', '.join(PROPAGATOR_DECLARATIONS))
    declaration = PROPAGATOR_DECLARATIONS[declared]
    basis = str(core.get_global_option('ATOMIC_RESPONSE_BASIS')).lower()
    if declaration is not None and declaration.rebuilds_kernel and basis != 'fitted_auxiliary':
        raise ValueError(
            f'ATOMIC_RESPONSE_PROPAGATOR {declared} builds its ALDA kernel in the '
            'molecular AUX basis and projects it through the fitted transition density; '
            'ATOMIC_RESPONSE_BASIS DIRECT_OV forms no fit for it to project through, and '
            'there is no AUX expansion of an orbital product. Declare '
            'ATOMIC_RESPONSE_BASIS FITTED_AUXILIARY (the reference protocol declares '
            'ATOMIC_OV_CHARGE_PENALTY 1000 and ATOMIC_OV_METRIC_DAMPING 0.0005 with it).')
    return dict(propagator=declaration)


def _correction_options():
    """Read the declared correction policy and its own companion declaration.

    Only ONE of the two companion declarations is ever forwarded, because each
    belongs to exactly one policy: a GRAC shift describes a GRAC profile and an
    AcDeclaration describes the multipole/Tozer-Handy form, and forwarding the
    other one is a mis-declaration that ``validate_correction`` refuses rather
    than ignores. Nothing here produces orbitals; DECLARED_MULTPOLE_AC is an
    acceptance policy and the declaration resolved here is compared against the
    one a prior explicit producer call already recorded on the wavefunction.
    """
    policy = core.get_global_option('ATOMIC_SCF_ASYMPTOTIC_CORRECTION')
    shift = core.get_global_option('ATOMIC_SCF_EXPECTED_GRAC_SHIFT')
    # A zero sentinel is not a fixed-shift declaration under any policy. Nonzero
    # values are still forwarded and still fail outside FIXED_GRAC, so a leftover
    # declaration from another request cannot pass unnoticed.
    options = dict(scf_correction=policy,
                   expected_grac_shift=None if shift == 0 and policy != 'FIXED_GRAC' else shift)
    if policy == aco.AC_POLICY:
        options['ac_declaration'] = aco.declaration()
    return options


def _refinement_options():
    """Read the nine declared refinement options and refuse, never clamp, a bad one.

    Each of them names a model rather than a tolerance -- the lattice count and
    seed pick one specific reproducible point cloud, the two limits pick the shell
    it lives in, and the weight type/coefficient pick which points and which
    anchors the objective believes -- so an out-of-range value is a mis-declared
    model and is rejected here rather than silently moved into range.

    The two limits are in MULTIPLES OF THE VAN DER WAALS RADIUS, matching
    ``core.FitPointsOptions``; they are not bohr.
    """
    values = dict((k, core.get_global_option(k)) for k in REFINEMENT_KEYS)
    npoints = int(values['ATOMIC_REFINEMENT_POINTS'])
    seed = int(values['ATOMIC_REFINEMENT_SEED'])
    lower = float(values['ATOMIC_REFINEMENT_LOWER_LIMIT'])
    upper = float(values['ATOMIC_REFINEMENT_UPPER_LIMIT'])
    weight_type = int(values['ATOMIC_REFINEMENT_WEIGHT_TYPE'])
    coefficient = float(values['ATOMIC_REFINEMENT_WEIGHT_COEFFICIENT'])
    cutoff = float(values['ATOMIC_REFINEMENT_CUTOFF'])
    general = int(values['ATOMIC_REFINEMENT_RANK_LIMIT'])
    hydrogen = int(values['ATOMIC_REFINEMENT_HYDROGEN_RANK_LIMIT'])
    localization = int(core.get_global_option('ATOMIC_LOCALIZATION_RANK_LIMIT'))
    if not 1 <= npoints <= r.MAXIMUM_POINTS:
        raise ValueError(f'ATOMIC_REFINEMENT_POINTS must be in [1,{r.MAXIMUM_POINTS}]')
    if seed < 1:
        raise ValueError('ATOMIC_REFINEMENT_SEED must be a positive declared seed')
    if weight_type not in rf.WEIGHT_TYPES:
        raise ValueError(f'ATOMIC_REFINEMENT_WEIGHT_TYPE must be one of {rf.WEIGHT_TYPES}')
    if not np.isfinite(coefficient) or coefficient <= 0:
        raise ValueError('ATOMIC_REFINEMENT_WEIGHT_COEFFICIENT must be finite and positive')
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError('ATOMIC_REFINEMENT_CUTOFF must be finite and positive')
    if not np.isfinite(lower) or not np.isfinite(upper) or not 0 < lower < upper:
        raise ValueError('ATOMIC_REFINEMENT_LOWER_LIMIT/UPPER_LIMIT must satisfy '
                         '0 < lower < upper, in van der Waals radii')
    for name, value in (('ATOMIC_REFINEMENT_RANK_LIMIT', general),
                        ('ATOMIC_REFINEMENT_HYDROGEN_RANK_LIMIT', hydrogen)):
        if value not in (1, 2, 3, 4):
            raise ValueError(f'{name} must be a declared 1, 2, 3 or 4')
        if value > localization:
            raise ValueError(f'{name} exceeds ATOMIC_LOCALIZATION_RANK_LIMIT; a variable '
                             'cannot be refined at a rank the local tensors were never '
                             'localized at')
    return dict(npoints=npoints, seed=seed, lower_limit=lower, upper_limit=upper,
                weight_type=weight_type, weight_coefficient=coefficient, cutoff=cutoff,
                rank_limit=general, hydrogen_rank_limit=hydrogen)


def validate_request(wfn, tasks):
    if not tasks or any(t not in TASKS for t in tasks) or len(set(tasks)) != len(tasks):
        raise ValueError('Unknown or duplicate native atomic property request')
    if core.get_global_option('PARTITION_SCHEME') != 'ISA_A':
        raise ValueError('Only ISA_A has a validated native continuous-partition adapter; MBIS is unsupported here')
    if core.get_global_option('ATOMIC_PROPERTY_RECIPE') not in RECIPE_TAIL_POLICIES:
        raise ValueError('Unsupported atomic basis recipe')
    if not str(core.get_global_option('ATOMIC_PROPERTY_AUXILIARY_BASIS')).strip():
        raise ValueError('ATOMIC_PROPERTY_AUXILIARY_BASIS must name a declared molecular AUX')
    if core.get_global_option('ATOMIC_RESPONSE_LOCALIZATION') != 'LW':
        raise ValueError('Only LW distributed-tensor localization is supported; not a density partition')
    # Two separate declarations, validated as one statement: the localization cannot
    # run above the rank of the distributed response it is given.
    rank = core.get_global_option('ATOMIC_MULTIPOLE_RANK')
    limit = core.get_global_option('ATOMIC_LOCALIZATION_RANK_LIMIT')
    if rank not in (3, 4):
        raise ValueError('ATOMIC_MULTIPOLE_RANK must be a declared 3 or 4')
    if limit not in (1, 2, 3, 4):
        raise ValueError('ATOMIC_LOCALIZATION_RANK_LIMIT must be a declared 1, 2, 3 or 4')
    if limit > rank:
        raise ValueError('ATOMIC_LOCALIZATION_RANK_LIMIT exceeds ATOMIC_MULTIPOLE_RANK; rank-4 local '
                         'tensors cannot be localized out of a rank-3 distributed response')
    if set(tasks) & REFINEMENT_TASKS:
        _refinement_options()
    if tasks != ('ATOMIC_PARTITION',):
        _propagator_option()
    if not isinstance(wfn, core.Wavefunction) or wfn.nirrep() != 1 or not wfn.same_a_b_orbs():
        raise ValueError('Native atomic properties require an actual restricted C1 wavefunction')
    if wfn.nalpha() != wfn.nbeta() or wfn.nalpha() < 1:
        raise ValueError('Restricted closed-shell wavefunction required')
    provenance = validate_correction(wfn, **_correction_options(),
                                     require_canonical=tasks != ('ATOMIC_PARTITION',))
    if not np.isfinite(wfn.energy()) or wfn.energy() == 0:
        raise ValueError('A converged SCF wavefunction is required; oeprop never runs SCF')
    # Exactly one convergence record is required, and WHICH one is the declared
    # policy's. NONE and FIXED_GRAC describe orbitals Psi4's own SCF converged, so
    # the SCF seal is their evidence. DECLARED_MULTPOLE_AC describes orbitals that
    # deliberately invalidated that seal -- the state is no longer the one SCF
    # converged, and its energy is not a variational minimum -- so its evidence is
    # the producer's own convergence record, which the admission above has just
    # verified against this wavefunction's state signature, basis and stopping
    # diagnostics. The gate is keyed on the provenance OBJECT that only
    # ``validate_declared_ac`` can return, not on the option string, so no seal is
    # waived by a policy name alone and none is ever manufactured. The stationarity
    # prerequisite below stays unconditional: the corrected state must still be
    # stationary for the Fock matrix it carries.
    if not isinstance(provenance, DeclaredAcProvenance):
        require_scf_seal(wfn)
    S, D, F = map(np.asarray, (wfn.S(), wfn.Da(), wfn.Fa()))
    residual = float(np.max(np.abs(F @ D @ S - S @ D @ F)))
    if not np.isfinite(residual) or residual > 2e-7:
        raise ValueError(f'SCF stationarity prerequisite failed: maxabs(FDS-SDF)={residual}')
    return residual


def run(wfn, tasks):
    # Invalidate on entry, even if validation fails; never mutate returned objects.
    if isinstance(wfn, core.Wavefunction):
        wfn._native_atomic_property_result = None
    tasks = tuple(tasks)
    residual = validate_request(wfn, tasks)
    correction_options = _correction_options()
    keys = ('PARTITION_SCHEME', 'ATOMIC_RESPONSE_LOCALIZATION', 'ATOMIC_PROPERTY_RECIPE',
            'ATOMIC_PROPERTY_AUXILIARY_BASIS',
            'ATOMIC_PROPERTY_RADIAL_POINTS', 'ATOMIC_PROPERTY_SPHERICAL_POINTS',
            'ATOMIC_RESPONSE_RADIAL_POINTS', 'ATOMIC_RESPONSE_SPHERICAL_POINTS',
            'ATOMIC_SCF_ASYMPTOTIC_CORRECTION', 'ATOMIC_SCF_EXPECTED_GRAC_SHIFT',
            'ATOMIC_RESPONSE_ALGORITHM', 'ATOMIC_MULTIPOLE_RANK',
            'ATOMIC_LOCALIZATION_RANK_LIMIT', 'ATOMIC_MULTIPOLE_DISTRIBUTION',
            'ATOMIC_RESPONSE_BASIS', 'ATOMIC_RESPONSE_PROPAGATOR',
            'ATOMIC_OV_CHARGE_PENALTY',
            'ATOMIC_OV_METRIC_DAMPING', 'ATOMIC_PROPERTY_PRINT')
    # The refinement options are recorded only when a refinement was actually
    # requested; listing them on an unrefined request would read as though they
    # had applied to it.
    if set(tasks) & REFINEMENT_TASKS:
        keys += REFINEMENT_KEYS
    # Same rule for the declared-correction companions: they are recorded only
    # under the policy they belong to, so a NONE or FIXED_GRAC request never
    # prints an ATOMIC_AC_* value as though it had applied to it.
    if correction_options['scf_correction'] == aco.AC_POLICY:
        keys += aco.KEYS
    options = tuple((k, core.get_global_option(k)) for k in keys)
    effective = dict(options)
    recipe = generated_recipe(wfn, int(effective['ATOMIC_PROPERTY_RADIAL_POINTS']),
                              int(effective['ATOMIC_PROPERTY_SPHERICAL_POINTS']),
                              str(effective['ATOMIC_PROPERTY_AUXILIARY_BASIS']),
                              int(effective['ATOMIC_MULTIPOLE_RANK']),
                              RECIPE_TAIL_POLICIES[effective['ATOMIC_PROPERTY_RECIPE']])
    core.print_out('\n  Native atomic properties: '+recipe.origin+'\n')
    # The logger is built here because this is the only module that reads
    # ambient options; every stage below is narrated off records it already
    # returns, so the verbosity cannot change a number.
    log = lg.StageLog(int(effective['ATOMIC_PROPERTY_PRINT']))
    lg.report_request(log, wfn, tasks=tasks, options=options, recipe=recipe,
                      correction_options=correction_options, scf_residual=residual)
    properties = None
    if set(tasks) == {'ATOMIC_PARTITION'}:
        log.stage('density partition (Drho-C ISA-A)', lg.partition_parameters(recipe))
        partition = p.native_partition(wfn, recipe, caller_converged=True)
        partition_log = log
    else:
        partition_log = lg.silent()  # native_properties already narrated this stage
        mol = wfn.molecule()
        radii = {1: .31, 8: .66}  # covalent radii in angstrom; explicit H/O graph policy
        bonds = tuple((i,j) for i in range(mol.natom()) for j in range(i)
                      if np.linalg.norm(np.array(recipe.sites[i].origin)-recipe.sites[j].origin)
                      < 1.3*(radii[mol.Z(i)]+radii[mol.Z(j)])/.529177210903)
        # Dedicated generated response grid; never construct DFTGrid in oeprop.
        go = core.IsaGridOptions()
        go.radial_points = int(effective['ATOMIC_RESPONSE_RADIAL_POINTS'])
        go.spherical_points = int(effective['ATOMIC_RESPONSE_SPHERICAL_POINTS'])
        grid = core.IsaGrid(mol.clone(), go)
        response_grid = np.column_stack((grid.x(), grid.y(), grid.z(), grid.w()))
        # Actual IsaGrid rows, before partition/response work. Do not infer a
        # response row count from SCF options or a historical basis alias.
        # The named algorithm decides which calibrated ALDA limit applies, and
        # nothing else; it is recorded in the request options above.
        algorithm = str(effective['ATOMIC_RESPONSE_ALGORITHM']).lower()
        estimate = estimate_response_work(wfn.basisset().nbf(), wfn.nmo(), wfn.nalpha(),
                                          response_grid.shape[0], algorithm=algorithm)
        lg.report_work_estimate(log, estimate)
        estimate.require_pass()
        dispersion = 'ATOMIC_DISPERSION' in tasks
        # A refined C_n needs the same Casimir-Polder weights, so it needs the same
        # frequency grid; it does not need the unrefined LW pairing, so pair_self
        # stays off unless the unrefined coefficients were themselves requested.
        refined_dispersion = 'ATOMIC_REFINED_DISPERSION' in tasks
        quad = (n.Quadrature.from_casimir(core.CasimirGrid(10,.5))
                if dispersion or refined_dispersion else None)
        properties = n.native_properties(wfn, recipe, bonds=bonds, frames=None, caller_converged=True,
            kernel='alda_slater_pw92', exact_exchange=.25, local_scale=.75, response_grid=response_grid,
            frequencies=quad.frequencies if quad else (0.,), quadrature=quad, pair_self=dispersion,
            response_algorithm=algorithm, log=log,
            localization_rank_limit=int(effective['ATOMIC_LOCALIZATION_RANK_LIMIT']),
            **_distribution_options(effective), **_propagator_option(), **correction_options)
        partition = properties.partition
    correction = validate_correction(wfn, **correction_options)
    result = AtomicPropertyResult(tasks, options, partition, properties, residual, correction)
    wfn._native_atomic_property_result = result
    partition.require_q()
    # Machine-readable surface. Arrays are wrapped as core.Matrix so that
    # p4util's name-driven ndarray reshaping cannot re-interpret a labeled
    # table; large intermediates stay in atomic_property_result(wfn).
    lg.report_partition(partition_log, wfn, partition)
    partition_log.stage_end()
    if properties is not None:
        local = properties.require_local()
        if 'ATOMIC_DISPERSION' in tasks and properties.dispersion is None:
            raise RuntimeError('Native dispersion failed: '+repr(properties.failures))
        # native_properties narrated these stages with wfn=None on purpose: it
        # is a pure function of its records. The wavefunction surface is owned
        # here, so the same reporters are replayed silently to publish only the
        # machine-readable names.
        lg.report_context(lg.silent(), wfn, properties.context,
                          properties.context.response.provider)
        lg.report_localization(lg.silent(), wfn, local)
        lg.report_atomic_polarizabilities(lg.silent(), wfn, local)
        if properties.dispersion is not None:
            lg.report_dispersion(lg.silent(), wfn, properties.dispersion)
    # PFIT refinement of the accepted local tensors. It runs last, after the
    # unrefined result is already attached, so a refusal here leaves the stages
    # that did succeed inspectable instead of discarding them. The reporters are
    # given the real logger AND the wavefunction: unlike the stages above, this
    # stage is owned here, so there is no second silent replay to publish it.
    if properties is not None and set(tasks) & REFINEMENT_TASKS:
        declared = _refinement_options()
        general = declared.pop('rank_limit')
        hydrogen = declared.pop('hydrogen_rank_limit')
        mol = wfn.molecule()
        # The site types are the actual nuclei of this molecule, one per site in
        # site order, and never parsed out of the generated `O1`/`H2` labels.
        site_types = tuple(mol.symbol(i) for i in range(mol.natom()))
        rank_limits = {t: (hydrogen if t.upper() == 'H' else general) for t in set(site_types)}
        refinement = r.native_refinement(properties, wfn, site_types=site_types,
                                         rank_limits=rank_limits, log=log,
                                         dispersion='ATOMIC_REFINED_DISPERSION' in tasks,
                                         **declared)
        result = AtomicPropertyResult(tasks, options, partition, properties, residual,
                                      correction, refinement)
        wfn._native_atomic_property_result = result
    # Deliberately None, like ordinary oeprop.
