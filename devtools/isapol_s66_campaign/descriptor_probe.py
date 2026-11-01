"""Isolated descriptor-only prediction; never SCF, ISA or molecular properties.

Run with the qualified private scientific Python. A supervising allocation must
verify the complete sealed runtime and pinned source binding before and after,
and bind this receipt to the exact proposed runtime input and memory contract.
This child alone is not an installed-runtime or molecular qualification.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def helper():
    spec = importlib.util.spec_from_file_location(
        'native_campaign_resource_plan', Path(__file__).with_name('resource_plan.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def descriptor_dimensions(molecule, main, recipe, declaration, planner):
    """Extract real shell inventories, not chemical-name width predictions."""
    if (main.name().upper() != 'AUG-CC-PVTZ' or not main.has_puream()
            or recipe.auxiliary.representation != 'Cartesian'):
        raise ValueError('actual spherical MAIN / Cartesian AUX required')
    if (molecule.natom() != len(declaration.frames)
            or len(recipe.sites) != molecule.natom()
            or molecule.molecular_charge() != 0 or molecule.multiplicity() != 1):
        raise ValueError('declared neutral singlet site inventory mismatch')
    electrons = 0
    for i in range(molecule.natom()):
        value = molecule.Z(i)
        if value != {'C': 6, 'H': 1, 'N': 7, 'O': 8}.get(molecule.symbol(i)):
            raise ValueError('real CHNO descriptors required')
        electrons += int(value)
    if electrons % 2:
        raise ValueError('closed-shell electron inventory required')

    def width(basis):
        if basis.representation not in ('Cartesian', 'Spherical'):
            raise ValueError('explicit descriptor representation required')
        return sum((s.l+1)*(s.l+2)//2 if basis.representation == 'Cartesian'
                   else 2*s.l+1 for s in basis.shells)

    if sum(2*main.shell(i).am+1 for i in range(main.nshell())) != main.nbf():
        raise ValueError('MAIN function/shell inventory mismatch')
    if (recipe.grid.radial_points != 100 or recipe.grid.spherical_points != 200
            or recipe.controller.max_iterations != 120):
        raise ValueError('profile grid or iteration cap changed')
    # No state is constructed: nmo<=nbf is an envelope, not an assertion that
    # SCF retains every MAIN overlap direction.
    return planner.Dimensions(
        sites=molecule.natom(), bonds=len(declaration.bonds),
        main_nbf=main.nbf(), main_nshell=main.nshell(), nocc=electrons//2,
        response_aux_width=width(recipe.auxiliary), density_aux_width=width(recipe.auxiliary),
        atomic_widths=tuple(width(s.atomic) for s in recipe.sites),
        shape_widths=tuple(width(s.shape) for s in recipe.sites))


def scientific_probe(runtime, *, memory_bytes):
    """Only molecule/basis/recipe descriptors and their hashes are built."""
    import psi4
    from psi4.driver.procrouting.isapol_schema import (
        admit_native_schema, native_molecule_schema, validate_native_request, check_native_geometry)
    from psi4.driver.procrouting.isapol_recipes import native_monomer_recipe
    from psi4.driver.procrouting.isapol_distribution import basis_identity
    from psi4.driver.procrouting.isapol_native_partition import DRHO_REFINEMENT_ITERATIONS
    planner = helper()
    specification = runtime['specification']
    request, geometry = admit_native_schema(specification, runtime['molecule'])
    if type(memory_bytes) is not int or memory_bytes <= 0 or request.resources.max_bytes > memory_bytes:
        raise ValueError('explicit intended Psi4 memory must cover numeric budget')
    psi4.set_memory(memory_bytes, quiet=True)
    psi4.set_options({k: v for k, v in specification['keywords'].items()
                      if k != 'function_kwargs'})
    molecule = psi4.core.Molecule.from_schema(native_molecule_schema(runtime['molecule'], request))
    request, identity = validate_native_request(
        request.as_dict(), method=specification['model']['method'],
        basis=specification['model']['basis'], molecule=molecule)
    main = psi4.core.BasisSet.build(molecule, 'BASIS', 'aug-cc-pVTZ', puream=1)
    recipe = native_monomer_recipe(molecule, profile=request.declaration.profile)
    if DRHO_REFINEMENT_ITERATIONS != 10:
        raise ValueError('native Drho refinement accounting policy changed')
    dimensions = descriptor_dimensions(molecule, main, recipe, request.declaration, planner)
    resources = dict(max_bytes=request.resources.max_bytes, max_work=request.resources.max_work,
                     max_io_bytes=request.resources.max_io_bytes, capacity=request.resources.capacity)
    plan = planner.plan(dimensions, resources, max_replay_work=request.max_replay_work)
    shells = [
        dict(centre=int(main.shell_to_center(i)), am=int(main.shell(i).am),
             exponents=[main.shell(i).exp(j) for j in range(main.shell(i).nprimitive)],
             coefficients=[main.shell(i).coef(j) for j in range(main.shell(i).nprimitive)])
        for i in range(main.nshell())]
    return dict(
        schema='native-campaign-descriptor-probe/1',
        status='PASS_DESCRIPTOR_PLAN_ONLY' if not plan['refusals'] else 'REFUSED_DESCRIPTOR_PLAN',
        package=psi4.__file__, core=psi4.core.__file__, implementation=identity,
        version=str(psi4.__version__), intended_memory_bytes=memory_bytes,
        coordinate_change_bohr=check_native_geometry(molecule, geometry),
        main_shells=shells,
        main_descriptor_sha256=hashlib.sha256(
            json.dumps(shells, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        auxiliary_descriptor_sha256=basis_identity(recipe.auxiliary),
        atomic_descriptor_sha256=[basis_identity(s.atomic) for s in recipe.sites],
        shape_descriptor_sha256=[basis_identity(s.shape) for s in recipe.sites],
        recipe_sha256=hashlib.sha256(repr(recipe).encode()).hexdigest(),
        grid_policy='Source-bound native angular rounding: 200->230,974->974; no grid constructed',
        source_policy_sha256=planner.FILES, plan=plan,
        qualification='Descriptor-only prediction, not SCF/ISA/response/fit, full runtime qualification, '
                      'wall-time/total-memory guarantee or molecular scientific acceptance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--memory-bytes', type=int, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    report = scientific_probe(json.loads(raw), memory_bytes=args.memory_bytes)
    report.update(runtime_input_sha256=hashlib.sha256(raw).hexdigest(),
                  descriptor_probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  planner_sha256=hashlib.sha256(Path(__file__).with_name('resource_plan.py').read_bytes()).hexdigest())
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], report=str(args.output))))


if __name__ == '__main__':
    main()
