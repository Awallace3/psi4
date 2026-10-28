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
