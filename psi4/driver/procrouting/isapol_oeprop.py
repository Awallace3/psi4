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
from .isapol_response_preflight import estimate_response_work
from .isapol_native_correction import (functional_definition as _functional_definition,
                                       validate_correction, require_scf_seal)

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
    correction_provenance: object = None

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


def _validate_pbe0(wfn, *, scf_correction='NONE', expected_grac_shift=None):
    return validate_correction(wfn, scf_correction=scf_correction,
                               expected_grac_shift=expected_grac_shift, require_canonical=True)


def _correction_options():
    policy = core.get_global_option('ATOMIC_SCF_ASYMPTOTIC_CORRECTION')
    shift = core.get_global_option('ATOMIC_SCF_EXPECTED_GRAC_SHIFT')
    # NONE's zero sentinel is not a fixed-shift declaration. Nonzero mismatches
    # still fail, including a leftover declaration from another request.
    return dict(scf_correction=policy,
                expected_grac_shift=None if policy == 'NONE' and shift == 0 else shift)


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
    validate_correction(wfn, **_correction_options(),
                        require_canonical=tasks != ('ATOMIC_PARTITION',))
    if not np.isfinite(wfn.energy()) or wfn.energy() == 0:
        raise ValueError('A converged SCF wavefunction is required; oeprop never runs SCF')
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
            'ATOMIC_PROPERTY_RADIAL_POINTS', 'ATOMIC_PROPERTY_SPHERICAL_POINTS',
            'ATOMIC_RESPONSE_RADIAL_POINTS', 'ATOMIC_RESPONSE_SPHERICAL_POINTS',
            'ATOMIC_SCF_ASYMPTOTIC_CORRECTION', 'ATOMIC_SCF_EXPECTED_GRAC_SHIFT',
            'ATOMIC_RESPONSE_ALGORITHM')
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
        # Actual IsaGrid rows, before partition/response work. Do not infer a
        # response row count from SCF options or a historical basis alias.
        # The named algorithm decides which calibrated ALDA limit applies, and
        # nothing else; it is recorded in the request options above.
        algorithm = str(effective['ATOMIC_RESPONSE_ALGORITHM']).lower()
        estimate_response_work(wfn.basisset().nbf(), wfn.nmo(), wfn.nalpha(),
                               response_grid.shape[0], algorithm=algorithm).require_pass()
        dispersion = 'ATOMIC_DISPERSION' in tasks
        quad = n.Quadrature.from_casimir(core.CasimirGrid(10,.5)) if dispersion else None
        properties = n.native_properties(wfn, recipe, bonds=bonds, frames=None, caller_converged=True,
            kernel='alda_slater_pw92', exact_exchange=.25, local_scale=.75, response_grid=response_grid,
            frequencies=quad.frequencies if quad else (0.,), quadrature=quad, pair_self=dispersion,
            response_basis='direct_ov', response_algorithm=algorithm, **correction_options)
        partition = properties.partition
    correction = validate_correction(wfn, **correction_options)
    result = AtomicPropertyResult(tasks, options, partition, properties, residual, correction)
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
