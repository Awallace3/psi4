# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Wavefunction-first native atomic properties; no reference files or hidden SCF.

GENERATED_JKFIT_ISA_A is a deliberately modest, self-contained H/O recipe, NOT
modern CamCASP ISA-Pol. Density: lambda1000 Drho-C, ordinary ISA-A. Response:
direct occupied-fast OV, Slater/PW92 ALDA + 25% exact exchange, no GRAC, no PFIT.
The response grid and tensor localization are not density partition policies.
"""
from dataclasses import dataclass
import math
import numpy as np
from psi4 import core
from . import isapol_native_partition as p
from . import isapol_native as n

TASKS = frozenset(('ATOMIC_PARTITION', 'ATOMIC_POLARIZABILITIES', 'ATOMIC_DISPERSION'))


def generated_recipe(wfn, radial=160, angular=590):
    """Generate all effective Gaussian descriptors from shipped JKFIT and formulas.

    Molecular AUX: Cartesian cc-pVDZ-JKFIT, unchanged effective contractions.
    AtomAux and Shape: normalized uncontracted even-tempered s Gaussians,
    exponents .1*2**k (O, k=0..16), .2*2**k (H, k=0..10). These separate roles
    are a compact radial ISA-A demonstration, not a claimed CamCASP basis alias.
    """
    mol = wfn.molecule()
    if any(mol.Z(i) not in (1, 8) for i in range(mol.natom())):
        raise ValueError('GENERATED_JKFIT_ISA_A currently supports real H/O nuclei only')
    centres = tuple((mol.x(i), mol.y(i), mol.z(i)) for i in range(mol.natom()))
    aux = core.BasisSet.build(mol, 'DF_BASIS_SCF', 'cc-pvdz-jkfit', puream=0)
    shells = []
    for j in range(aux.nshell()):
        s = aux.shell(j)
        shells.append(p.ShellRecipe(int(aux.shell_to_center(j)), int(s.am),
            tuple(s.exp(k) for k in range(s.nprimitive)), tuple(s.coef(k) for k in range(s.nprimitive))))
    origin = 'runtime Psi4 shipped cc-pVDZ-JKFIT plus normalized even-tempered radial s recipe; NOT modern CamCASP preset'
    auxiliary = p.BasisRecipe('cc-pVDZ-JKFIT Cartesian molecular AUX', origin, 'Cartesian', centres, tuple(shells))
    sites = []
    for i, c in enumerate(centres):
        oxygen = mol.Z(i) == 8
        exps = tuple((.1 if oxygen else .2)*2**k for k in range(17 if oxygen else 11))
        radial_shells = tuple(p.ShellRecipe(0, 0, (a,), ((2*a/math.pi)**.75,)) for a in exps)
        atom = p.BasisRecipe('even-tempered radial AtomAux', origin, 'Spherical', (c,), radial_shells)
        shape = p.BasisRecipe('even-tempered s Shape', origin, 'Spherical', (c,), radial_shells)
        sites.append(p.SiteRecipe(f'{mol.symbol(i)}{i+1}', c, atom, shape, tuple(range(len(exps))), 3, 1.5, True))
    return p.PartitionRecipe('GENERATED_JKFIT_ISA_A', origin, 'explicit_cartesian_drho_c_isa_a',
        auxiliary, tuple(sites), p.GridRecipe(radial, angular, 3, 1., 'native_tabulated_bragg_slater',
        'all_sites_unscreened_full_molecular_grid'),
        p.ControllerRecipe(1e-9, 120, .17, .001, .2, True, 0., True, 1e-36,
                           1e-5, 1e-5, 1e-5, 0., 20, 20, True), 'strict1e-9')


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

    @property
    def atomic_scalars(self):
        if self.properties is None:
            raise RuntimeError('Atomic response was not requested')
        return self.properties.atomic_scalars.array

    @property
    def dispersion(self):
        return None if self.properties is None else self.properties.dispersion


def atomic_property_result(wfn):
    """Return the owned latest native oeprop result (oeprop itself returns None).

    No property calculation is performed. Raises if no result is attached.
    """
    result = getattr(wfn, '_native_atomic_property_result', None)
    if result is None:
        raise ValueError('No native atomic property result on this wavefunction')
    return result


def _functional_definition(functional):
    """Effective PBE0 definition, independent of current SCF/DFT options.

    Require the canonical full LibXC representation (not a name-only custom
    functional). Mix data also checks the internal semilocal scales, which a
    LibXC PBEH tweak can change independently of SuperFunctional.x_alpha().
    """
    scalar_keys = ('x_alpha', 'x_beta', 'x_omega', 'c_alpha', 'c_omega',
                   'c_os_alpha', 'c_ss_alpha', 'vv10_b', 'vv10_c',
                   'grac_shift', 'grac_alpha', 'grac_beta', 'ansatz',
                   'is_libxc_func', 'needs_vv10', 'needs_grac', 'is_x_lrc', 'is_c_lrc')
    components = []
    for group in (functional.x_functionals(), functional.c_functionals()):
        entries = []
        for component in group:
            if not isinstance(component, core.LibXCFunctional):
                raise ValueError('Supported PBE0 requires canonical LibXC components')
            entries.append((component.name(), component.alpha(), component.omega(),
                            component.is_gga(), component.is_meta(), component.is_lrc(),
                            tuple(component.get_mix_data()),
                            tuple(sorted(component.query_libxc('XC_HYB_CAM_COEF').items()))))
        components.append(tuple(entries))
    return tuple(getattr(functional, key)() for key in scalar_keys), tuple(components)


def _validate_pbe0(wfn):
    # Direct unmodified C++ LibXC factory: no ambient options, registry lookup,
    # mutating global option stash, or custom definition used as its own oracle.
    canonical = core.SuperFunctional.XC_build('XC_HYB_GGA_XC_PBEH', True)
    functional = wfn.functional()
    if (functional.name().upper() != 'PBE0' or functional.x_alpha() != .25
            or _functional_definition(functional) != _functional_definition(canonical)
            or getattr(wfn, '_disp_functor', None) is not None):
        raise ValueError('Ordinary atomic response requires unmodified canonical PBE0 '
                         '(25% exact exchange, no range separation, dispersion or GRAC)')


def validate_request(wfn, tasks):
    if not tasks or any(t not in TASKS for t in tasks) or len(set(tasks)) != len(tasks):
        raise ValueError('Unknown or duplicate native atomic property request')
    if core.get_global_option('PARTITION_SCHEME') != 'ISA_A':
        raise ValueError('Only ISA_A has a validated native continuous-partition adapter; MBIS is unsupported here')
    if core.get_global_option('ATOMIC_PROPERTY_RECIPE') != 'GENERATED_JKFIT_ISA_A':
        raise ValueError('Unsupported atomic basis recipe')
    if core.get_global_option('ATOMIC_RESPONSE_LOCALIZATION') != 'LW':
        raise ValueError('Only LW distributed-tensor localization is supported; not a density partition')
    if not isinstance(wfn, core.Wavefunction) or wfn.nirrep() != 1 or not wfn.same_a_b_orbs():
        raise ValueError('Native atomic properties require an actual restricted C1 wavefunction')
    if wfn.nalpha() != wfn.nbeta() or wfn.nalpha() < 1:
        raise ValueError('Restricted closed-shell wavefunction required')
    if tasks != ('ATOMIC_PARTITION',):
        _validate_pbe0(wfn)
    if not np.isfinite(wfn.energy()) or wfn.energy() == 0:
        raise ValueError('A converged SCF wavefunction is required; oeprop never runs SCF')
    from .scf_proc.scf_iterator import _scf_state_signature
    evidence = getattr(wfn, '_scf_convergence_evidence', None)
    if evidence is None:
        raise ValueError('Successful SCF convergence evidence is required; oeprop never runs SCF')
    diagnostics, signature, basis = evidence
    iteration, delta_e, gradient_norm, e_threshold, d_threshold, rms_norm = diagnostics
    if (iteration < 0 or not np.all(np.isfinite((delta_e, gradient_norm, e_threshold, d_threshold)))
            or not (abs(delta_e) < e_threshold and 0 <= gradient_norm < d_threshold)):
        raise ValueError('SCF convergence stopping diagnostics are inconsistent')
    if basis != wfn.basisset() or signature != _scf_state_signature(wfn):
        raise ValueError('SCF convergence evidence is stale: wavefunction state has changed')
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
    keys = ('PARTITION_SCHEME', 'ATOMIC_RESPONSE_LOCALIZATION', 'ATOMIC_PROPERTY_RECIPE',
            'ATOMIC_PROPERTY_RADIAL_POINTS', 'ATOMIC_PROPERTY_SPHERICAL_POINTS',
            'ATOMIC_RESPONSE_RADIAL_POINTS', 'ATOMIC_RESPONSE_SPHERICAL_POINTS')
    options = tuple((k, core.get_global_option(k)) for k in keys)
    effective = dict(options)
    recipe = generated_recipe(wfn, int(effective['ATOMIC_PROPERTY_RADIAL_POINTS']),
                              int(effective['ATOMIC_PROPERTY_SPHERICAL_POINTS']))
    core.print_out('\n  Native atomic properties: '+recipe.origin+'\n')
    properties = None
    if set(tasks) == {'ATOMIC_PARTITION'}:
        partition = p.native_partition(wfn, recipe, caller_converged=True)
    else:
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
        dispersion = 'ATOMIC_DISPERSION' in tasks
        quad = n.Quadrature.from_casimir(core.CasimirGrid(10,.5)) if dispersion else None
        properties = n.native_properties(wfn, recipe, bonds=bonds, frames=None, caller_converged=True,
            kernel='alda_slater_pw92', exact_exchange=.25, local_scale=.75, response_grid=response_grid,
            frequencies=quad.frequencies if quad else (0.,), quadrature=quad, pair_self=dispersion,
            response_basis='direct_ov')
        partition = properties.partition
    result = AtomicPropertyResult(tasks, options, partition, properties, residual)
    wfn._native_atomic_property_result = result
    partition.require_q()
    wfn.set_variable('ISA ITERATIONS', float(partition.trajectory.state.iteration))
    wfn.set_variable('ISA DRHO FITTED ELECTRONS', float(partition.drho.fitted_electrons))
    if properties is not None:
        properties.require_local()
        if 'ATOMIC_DISPERSION' in tasks and properties.dispersion is None:
            raise RuntimeError('Native dispersion failed: '+repr(properties.failures))
        for i, site in enumerate(recipe.sites):
            wfn.set_variable(f'ATOM {site.label} DIPOLE POLARIZABILITY', float(result.atomic_scalars[0,i,0]))
        core.print_out('  Static atomic dipole trace polarizabilities (bohr^3): '+
                       repr(result.atomic_scalars[0,:,0].tolist())+'\n')
    # Deliberately None, like ordinary oeprop.
