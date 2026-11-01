"""Descriptor extraction fixtures only: no Psi4, wavefunction or quantum data."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

spec = importlib.util.spec_from_file_location(
    'campaign_descriptor_probe', Path(__file__).with_name('descriptor_probe.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def fixtures():
    molecule = NS(natom=lambda: 3, molecular_charge=lambda: 0, multiplicity=lambda: 1,
                  symbol=lambda i: ('O', 'H', 'H')[i], Z=lambda i: (8, 1, 1)[i])
    # Toy shell counts, deliberately not a real basis or a molecular result.
    main = NS(name=lambda: 'aug-cc-pVTZ', has_puream=lambda: True,
              nbf=lambda: 12, nshell=lambda: 4, shell=lambda i: NS(am=1))
    basis = lambda representation, am: NS(
        representation=representation, shells=tuple(NS(l=i) for i in am))
    recipe = NS(auxiliary=basis('Cartesian', (0, 1, 2)),
                sites=tuple(NS(atomic=basis('Spherical', (0, 1, 2)),
                               shape=basis('Spherical', (0, 0))) for _ in range(3)),
                grid=NS(radial_points=100, spherical_points=200),
                controller=NS(max_iterations=120))
    declaration = NS(frames=(0, 0, 0), bonds=((0, 1), (0, 2)))
    return molecule, main, recipe, declaration


def test_widths_come_from_representation_and_shells_not_formula_lookup():
    d = probe.descriptor_dimensions(*fixtures(), probe.helper())
    assert d.main_nbf == 12 and d.main_nshell == 4 and d.nocc == 5
    assert d.response_aux_width == d.density_aux_width == 10
    assert d.atomic_widths == (9, 9, 9) and d.shape_widths == (2, 2, 2)
    assert d.bonds == 2


@pytest.mark.parametrize('change', [
    lambda m, b, r, d: setattr(b, 'has_puream', lambda: False),
    lambda m, b, r, d: setattr(b, 'nbf', lambda: 13),
    lambda m, b, r, d: setattr(r.auxiliary, 'representation', 'Spherical'),
    lambda m, b, r, d: setattr(m, 'molecular_charge', lambda: 1),
    lambda m, b, r, d: setattr(m, 'Z', lambda i: 0 if i == 0 else 1),
    lambda m, b, r, d: setattr(r.controller, 'max_iterations', 119),
    lambda m, b, r, d: setattr(r.grid, 'spherical_points', 230),
    lambda m, b, r, d: setattr(d, 'frames', (0,)),
])
def test_mismatched_real_descriptors_are_refused(change):
    args = fixtures()
    change(*args)
    with pytest.raises(ValueError):
        probe.descriptor_dimensions(*args, probe.helper())


def test_help_does_not_import_quantum_or_manager_packages():
    process = subprocess.run([sys.executable, '-I', '-S', '-B', probe.__file__, '--help'],
                             text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    assert '--memory-bytes' in process.stdout


def test_no_quantum_state_or_grid_is_constructed_by_descriptor_child():
    tree = ast.parse(Path(probe.__file__).read_text())
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    forbidden = ('energy', 'properties', 'oeprop', 'Wavefunction', 'NativeRestrictedState',
                 'IsaGrid', 'FitPoints', 'native_partition', 'adapt_main')
    assert not any(c.split('.')[-1] in forbidden for c in calls)
