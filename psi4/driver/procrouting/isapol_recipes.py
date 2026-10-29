# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Explicit native-basis recipes, not inferred general ISA defaults.

The water recipe is the declared molecular regression model: its s-only
even-tempered shape basis and absolute 1.5-bohr tail cutoff are scientific
inputs. They must not be described as a Bragg-Slater multiplier or as a
CamCASP-matched production model. No external basis/product files are read.
"""
import numpy as np
from psi4 import core

from . import isapol_native_partition as isa
from .isapol_basis import BasisRecipe, ShellRecipe


def native_water_recipe(molecule, *, tail_fit_policy='always'):
    """Declare the shipped O,H,H regression recipe using only Psi4 basis data.

    AtomAux replaces the native s block by normalized primitive Gaussians:
    O 2**[8,...,-3], H 2**[5,...,-2]. Shape is exactly that s block.
    The caller still supplies the response/Q grids and resource limits.
    """
    if [molecule.symbol(i) for i in range(molecule.natom())] != ['O', 'H', 'H']:
        raise ValueError('declared water recipe requires O,H,H order')
    centres = tuple(map(tuple, molecule.geometry().np))
    native = core.BasisSet.build(molecule, 'DF_BASIS_SCF', 'aug-cc-pVTZ-RI', puream=0)
    shells = []
    for i in range(native.nshell()):
        shell = native.shell(i)
        shells.append(ShellRecipe(int(native.shell_to_center(i)), int(shell.am),
            tuple(shell.exp(k) for k in range(shell.nprimitive)),
            tuple(shell.coef(k) for k in range(shell.nprimitive))))
    origin = 'Psi4 aug-cc-pVTZ-RI; explicit O/H even-tempered s replacement'
    auxiliary = BasisRecipe('aug-cc-pVTZ-RI', origin, 'Cartesian', centres, tuple(shells))
    sites = []
    for index, (label, centre, powers) in enumerate(zip(
            ('O', 'H1', 'H2'), centres,
            (range(8, -4, -1), range(5, -3, -1), range(5, -3, -1)))):
        shape_shells = tuple(ShellRecipe(0, 0, (2.**power,),
                            ((2. * 2.**power / np.pi)**.75,)) for power in powers)
        angular = tuple(ShellRecipe(0, shell.l, shell.exponents, shell.coefficients)
                        for shell in shells if shell.centre == index and shell.l > 0)
        shape = BasisRecipe(label+' shape', origin, 'Spherical', (centre,), shape_shells)
        atomic = BasisRecipe(label+' atomic', origin, 'Spherical', (centre,), shape_shells+angular)
        sites.append(isa.SiteRecipe(label, centre, atomic, shape,
                                   tuple(range(len(shape_shells))), 4, 1.5, True))
    return isa.PartitionRecipe('Explicit water ISA-A with native basis data', origin,
        'explicit_cartesian_drho_c_isa_a', auxiliary, tuple(sites),
        isa.GridRecipe(100, 200, 3, 1., 'native_tabulated_bragg_slater',
                       'all_sites_unscreened_full_molecular_grid'),
        isa.ControllerRecipe(1e-9, 120, .17, .001, .2, True, 0., True, 1e-36,
                             1e-5, 1e-5, 1e-5, 0., 20, 20, True,
                             tail_fit_policy=tail_fit_policy),
        'Drho1e-2', atomic_initialization='zero_atomic_D0')


def benzene_declared_variables():
    """Full noncharge rank1/2 upper triangles, C1/H1 reference ownership."""
    from .isapol_refine import COMPONENT_NAMES
    return tuple(f'{ref}_{COMPONENT_NAMES[i]}_{COMPONENT_NAMES[j]}_A'
                 for ref in ('C1', 'H1') for i in range(1, 9) for j in range(i, 9))


def native_benzene_recipe(molecule, *, tail_fit_policy='always'):
    """Explicit C1..C6,H1..H6 model from current installed basis definitions.

    Unlike the water regression recipe, tail switches are at1.5 times the
    declared Bragg–Slater radius, not at absolute1.5bohr. No molecular products,
    historical basis manifests or external program are read.
    """
    from psi4.driver.p4util import OptionsState
    if [molecule.symbol(i) for i in range(molecule.natom())] != ['C']*6+['H']*6:
        raise ValueError('benzene recipe requires cyclic C1..C6 then matching H1..H6')
    xyz = np.asarray(molecule.geometry())
    for i in range(6):
        if (np.argmin(np.linalg.norm(xyz[:6]-xyz[i+6], axis=1)) != i
                or not 2.4 < np.linalg.norm(xyz[i]-xyz[(i+1) % 6]) < 2.9
                or not 1.8 < np.linalg.norm(xyz[i]-xyz[i+6]) < 2.3):
            raise ValueError('benzene cyclic ring / matching C-H topology required')
    stash = OptionsState(['PUREAM'])
    try:
        # A changed global PUREAM overrides the build argument in Psi4.
        core.set_global_option('PUREAM', False)
        native = core.BasisSet.build(molecule, 'DF_BASIS_SCF', 'aug-cc-pVTZ-RI', puream=0)
    finally:
        stash.restore()
    if native.nbf() != 1146 or native.has_puream():
        raise ValueError('benzene recipe requires actual1146-function Cartesian AUX')
    centres = tuple(map(tuple, xyz))
    shells = tuple(ShellRecipe(
        int(native.shell_to_center(i)), int(native.shell(i).am),
        tuple(native.shell(i).exp(k) for k in range(native.shell(i).nprimitive)),
        tuple(native.shell(i).coef(k) for k in range(native.shell(i).nprimitive)))
        for i in range(native.nshell()))
    origin = 'Current Psi4 aug-cc-pVTZ-RI; explicit C/H normalized even-tempered s replacement'
    auxiliary = BasisRecipe('aug-cc-pVTZ-RI', origin, 'Cartesian', centres, shells)
    sites = []
    for i, centre in enumerate(centres):
        symbol = 'C' if i < 6 else 'H'
        label = f'{symbol}{i % 6+1}'
        powers = range(8, -4, -1) if symbol == 'C' else range(5, -3, -1)
        ss = tuple(ShellRecipe(0, 0, (2.**p,), ((2.*2.**p/np.pi)**.75,)) for p in powers)
        angular = tuple(ShellRecipe(0, s.l, s.exponents, s.coefficients)
                        for s in shells if s.centre == i and s.l > 0)
        shape = BasisRecipe(label+' shape', origin, 'Spherical', (centre,), ss)
        atomic = BasisRecipe(label+' atomic', origin, 'Spherical', (centre,), ss+angular)
        sites.append(isa.SiteRecipe(
            label, centre, atomic, shape, tuple(range(len(ss))), 4,
            isa.bragg_slater_tail_cutoff(int(molecule.Z(i)), 1.5), True))
    return isa.PartitionRecipe(
        'Explicit native benzene ISA-A', origin, 'explicit_cartesian_drho_c_isa_a',
        auxiliary, tuple(sites),
        isa.GridRecipe(100, 200, 3, 1., 'native_tabulated_bragg_slater',
                       'all_sites_unscreened_full_molecular_grid'),
        isa.ControllerRecipe(1e-9, 120, .17, .001, .2, True, 0., True, 1e-36,
                             1e-5, 1e-5, 1e-6, 0., 20, 20, True,
                             tail_fit_policy=tail_fit_policy),
        'Drho1e-2', atomic_initialization='zero_atomic_D0')
