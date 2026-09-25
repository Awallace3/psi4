# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Explicit bounded DF-centre presets for the wavefunction-first oeprop API.

No SCF, external program, reference file, or historical numerical result is
read here. A preset declares only molecule topology: local bond frames, bonds
and COPY-equivalent site types. Every numerical setting is shared by both
presets and is the CamCASP program/workflow default (CamCASP 6.0 source cited
per value), not the settings of a validated reference run. Geometry is always
taken from the live wavefunction; water atom order is O,H,H and benzene order
is cyclic C1..C6 then H1..H6. Changing geometry does not remove the explicitly
selected COPY constraints.

Precedence: explicit kwargs > explicitly changed Psi4 options > preset values.
Unchanged legacy defaults do not leak into this separately selected model.
"""
import numpy as np
from psi4 import core
from .isapol_bounded import bounded_properties, BoundedResources
from .isapol_native import Quadrature
from .isapol_native_partition import BasisRecipe, ShellRecipe
from .isapol_native_propagator import KernelSmoothing
from .isapol_refine import RefinementSite
from .isapol_logging import StageLog
from .isapol_native_correction import correction_state


TASKS = frozenset(('ATOMIC_REFINED_POLARIZABILITIES', 'ATOMIC_REFINED_DISPERSION'))
OPTION_MAP = {
    'auxiliary_basis': 'ATOMIC_PROPERTY_AUXILIARY_BASIS',
    'radial_points': 'ATOMIC_RESPONSE_RADIAL_POINTS',
    'spherical_points': 'ATOMIC_RESPONSE_SPHERICAL_POINTS',
    'npoints': 'ATOMIC_REFINEMENT_POINTS', 'seed': 'ATOMIC_REFINEMENT_SEED',
    'lower_limit': 'ATOMIC_REFINEMENT_LOWER_LIMIT', 'upper_limit': 'ATOMIC_REFINEMENT_UPPER_LIMIT',
    'weight_type': 'ATOMIC_REFINEMENT_WEIGHT_TYPE',
    'weight_coefficient': 'ATOMIC_REFINEMENT_WEIGHT_COEFFICIENT',
    'cutoff': 'ATOMIC_REFINEMENT_CUTOFF',
    'localization_rank_limit': 'ATOMIC_LOCALIZATION_RANK_LIMIT',
    'rank_limit': 'ATOMIC_REFINEMENT_RANK_LIMIT',
    'hydrogen_rank_limit': 'ATOMIC_REFINEMENT_HYDROGEN_RANK_LIMIT',
    'charge_penalty': 'ATOMIC_OV_CHARGE_PENALTY',
    'anchor_metric_damping': 'ATOMIC_OV_METRIC_DAMPING',
    'scf_correction': 'ATOMIC_SCF_ASYMPTOTIC_CORRECTION',
    'expected_grac_shift': 'ATOMIC_SCF_EXPECTED_GRAC_SHIFT',
    'verbosity': 'ATOMIC_PROPERTY_PRINT',
}


def settings(preset, overrides):
    """Resolve only documented inputs; reject typos instead of ignoring them."""
    if preset not in ('water', 'benzene'):
        raise ValueError("bounded preset must be 'water' or 'benzene'")
    for option, expected in (('ATOMIC_MULTIPOLE_DISTRIBUTION', 'DF_CENTRE_ANALYTIC'),
                             ('ATOMIC_RESPONSE_BASIS', 'FITTED_AUXILIARY'),
                             ('ATOMIC_RESPONSE_LOCALIZATION', 'LW'), ('ATOMIC_MULTIPOLE_RANK', 4)):
        if core.has_global_option_changed(option) and core.get_global_option(option) != expected:
            raise ValueError(f'BOUNDED_DF requires {option}={expected}; explicitly changed option conflicts')
    # CamCASP defaults. CIF = src/tools/cluster_file_interface.F90.
    values = dict(
        auxiliary_basis='aug-cc-pVTZ-RI',  # CIF aux aug-cc-pVTZ, Cartesian, limit G
        radial_points=100, spherical_points=200,  # CIF:2677-2681 pol-step grid
        npoints=2000, seed=1, lower_limit=2., upper_limit=4.,  # CIF random lattice
        # localize.py: weight 3, weightcoeff 1e-3, cutoff 1e-4, limit 2, hlimit=limit
        weight_type=3, weight_coefficient=1e-3, cutoff=1e-4,
        localization_rank_limit=2, rank_limit=2, hydrogen_rank_limit=2,
        charge_penalty=1000., anchor_metric_damping=.0005,  # DF lambda / NL4 anchor eta
        # CIF:2400-2412 attaches GRAC only when IP+HOMO > 0; AUTO follows the SCF.
        scf_correction='AUTO', expected_grac_shift=None, verbosity=1,
        # prop_parameters.F90:18-25 FD smoothing; parameters.f90:123 integral cutoff
        smoothing=KernelSmoothing(1e-8, 1000., .01, 1., 'FD'), shell_cutoff=1e-8,
        max_order=10,  # localize.py Casimir maxN = 4*wsmlimit+2
        quadrature=None, response_grid=None, lattice_options=None, auxiliary_recipe=None,
        sites=None, bonds=None,
        declared_variables=None,  # process_data.F90:2040-2150 cutoff-derived pdef
        scratch_directory=None,
        resources=None, log=None)  # None: numeric budget = psi4.set_memory
    unknown = set(overrides)-set(values)
    if unknown:
        raise TypeError('Unknown bounded property keyword(s): '+', '.join(sorted(unknown)))
    for key, option in OPTION_MAP.items():
        if core.has_global_option_changed(option):
            values[key] = core.get_global_option(option)
    values.update(overrides)
    return values


def default_resources():
    """Numeric budget from the Psi4 memory setting; fixed work and I/O ceilings."""
    return BoundedResources(int(core.get_memory()), 6_000_000_000_000, 64*1024**3)


def resolve_correction(wfn, scf_correction, expected_grac_shift):
    """AUTO reproduces CamCASP: FIXED_GRAC at the SCF's own attached shift, else NONE.

    The shift is read from the functional that produced the orbitals, never
    computed from an IP/HOMO here; bounded_properties still validates it.
    """
    if scf_correction != 'AUTO':
        return scf_correction, expected_grac_shift
    if expected_grac_shift is not None:
        return 'FIXED_GRAC', expected_grac_shift
    functional = wfn.functional() if hasattr(wfn, 'functional') else None
    shift = correction_state(functional)[0] if functional is not None else 0.
    return ('FIXED_GRAC', shift) if shift > 0 else ('NONE', None)


def _frame(z, x):
    z = z / np.linalg.norm(z)
    x = x-z*np.dot(z, x)
    x = x / np.linalg.norm(x)
    if not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError('degenerate geometry for preset local axes')
    return np.column_stack((x, np.cross(z, x), z))


def preset_sites(molecule, preset, rank_limit=2, hydrogen_rank_limit=2):
    """Declare frames and COPY site types, without reading fitted property values."""
    symbols = [molecule.symbol(i) for i in range(molecule.natom())]
    expected = ['O', 'H', 'H'] if preset == 'water' else ['C']*6+['H']*6
    if symbols != expected or any(molecule.Z(i) <= 0 for i in range(molecule.natom())):
        raise ValueError(f'{preset} preset requires atom order {expected} without ghost sites')
    xyz = molecule.geometry().np
    if preset == 'water':
        labels = ['O', 'H1', 'H2']
        frames = [np.eye(3), _frame(xyz[1]-xyz[0], xyz[1]-xyz[2]),
                  _frame(xyz[2]-xyz[0], xyz[2]-xyz[1])]
        bonds = [(1, 0), (2, 0)]
    else:
        labels = [f'C{i+1}' for i in range(6)]+[f'H{i+1}' for i in range(6)]
        frames = [_frame(xyz[i%6+6]-xyz[i%6], xyz[(i%6+1)%6]-xyz[i%6]) for i in range(12)]
        bonds = [(0,1),(0,5),(0,6),(1,2),(1,7),(2,3),(2,8),(3,4),(3,9),(4,5),(4,10),(5,11)]
    sites = [RefinementSite(label, symbol, origin, axes,
                            hydrogen_rank_limit if symbol == 'H' else rank_limit)
             for label, symbol, origin, axes in zip(labels, symbols, xyz, frames)]
    return sites, bonds


def run(wfn, tasks, *, preset, **kwargs):
    """Run explicit BOUNDED_DF oeprop tasks and attach the owned result on success.

    The two supported refined tasks execute the full common chain. Read its
    BoundedProperties record with psi4.atomic_property_result(wfn). Legacy
    ATOMIC_* requests without atomic_backend='BOUNDED_DF' are unchanged.
    """
    if not tasks or any(task not in TASKS for task in tasks) or len(set(tasks)) != len(tasks):
        raise ValueError('BOUNDED_DF accepts only ATOMIC_REFINED_POLARIZABILITIES and ATOMIC_REFINED_DISPERSION')
    if not isinstance(wfn, core.Wavefunction):
        raise TypeError('BOUNDED_DF requires a live Psi4 Wavefunction')
    v = settings(preset, kwargs)
    log = v['log'] if v['log'] is not None else StageLog(v['verbosity'])
    if not isinstance(log, StageLog):
        raise TypeError('log must be a StageLog')
    log.stage('Bounded preset preparation', (('preset', preset), ('model', 'DF-centre, not ISA-A')))
    mol = wfn.molecule()
    sites, bonds = preset_sites(mol, preset, v['rank_limit'], v['hydrogen_rank_limit'])
    if v['sites'] is not None:
        sites = v['sites']
    if v['bonds'] is not None:
        bonds = v['bonds']
    v['scf_correction'], v['expected_grac_shift'] = resolve_correction(
        wfn, v['scf_correction'], v['expected_grac_shift'])
    recipe = v['auxiliary_recipe']
    if recipe is None:
        basis = core.BasisSet.build(mol, 'DF_BASIS_SCF', v['auxiliary_basis'], puream=0)
        shells = []
        for i in range(basis.nshell()):
            s = basis.shell(i)
            shells.append(ShellRecipe(int(basis.shell_to_center(i)), int(s.am),
                tuple(s.exp(k) for k in range(s.nprimitive)), tuple(s.coef(k) for k in range(s.nprimitive))))
        recipe = BasisRecipe(v['auxiliary_basis'], 'Psi4 basis DATA, native normalized',
                             'Cartesian', tuple(map(tuple, mol.geometry().np)), tuple(shells))
    grid = v['response_grid']
    if grid is None:
        options = core.IsaGridOptions()
        options.radial_points, options.spherical_points = v['radial_points'], v['spherical_points']
        native_grid = core.IsaGrid(mol.clone(), options)
        grid = np.column_stack((native_grid.x(), native_grid.y(), native_grid.z(), native_grid.w()))
        del native_grid  # Do not retain a second full grid during the bounded solve.
    lattice = v['lattice_options']
    if lattice is None:
        lattice = core.FitPointsOptions()
        lattice.npoints, lattice.seed = v['npoints'], v['seed']
        lattice.lolim, lattice.hilim = v['lower_limit'], v['upper_limit']
    if v['resources'] is None:
        v['resources'] = default_resources()
    quadrature = v['quadrature'] if v['quadrature'] is not None else Quadrature.from_casimir(core.CasimirGrid(10, .5))
    log.items([(k, v[k]) for k in OPTION_MAP]+[('declared_variables', 'cutoff-derived'
               if v['declared_variables'] is None else len(v['declared_variables'])),
               ('numeric byte budget', v['resources'].max_bytes)])
    log.stage_end()
    keys = ('smoothing', 'shell_cutoff', 'charge_penalty', 'anchor_metric_damping',
            'localization_rank_limit', 'weight_type', 'weight_coefficient', 'cutoff',
            'resources', 'scf_correction', 'expected_grac_shift', 'max_order', 'scratch_directory')
    result = bounded_properties(wfn, recipe, caller_converged=True, distribution='df_centre_analytic',
        sites=sites, bonds=bonds, quadrature=quadrature, response_grid=grid,
        lattice_options=lattice, declared_variables=v['declared_variables'], log=log, **{k:v[k] for k in keys})
    wfn._native_atomic_property_result = result
