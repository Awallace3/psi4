"""Installed descriptor child for cb63305/request3; never SCF or properties.

The allocation supervisor must independently bind this source, input, code,
sealed runtime and receipt profile. A child result alone is not admission.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


def helper(name):
    if name not in ('resource_plan_cb63305', 'descriptor_probe'):
        raise ValueError('closed installed descriptor helpers required')
    path = Path(__file__).with_name(name+'.py')
    raw = path.read_bytes()
    spec = importlib.util.spec_from_file_location('_expanded_descriptor_'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def scientific_probe(runtime, *, memory_bytes):
    """Real basis/shell/recipe declarations only; no electronic state."""
    import psi4
    from psi4.driver.procrouting.isapol_schema import (
        admit_native_schema, native_molecule_schema, validate_native_request, check_native_geometry)
    from psi4.driver.procrouting.isapol_recipes import native_monomer_recipe
    from psi4.driver.procrouting.isapol_distribution import basis_identity
    from psi4.driver.procrouting.isapol_native_partition import DRHO_REFINEMENT_ITERATIONS
    planner, descriptors = helper('resource_plan_cb63305'), helper('descriptor_probe')
    specification = runtime['specification']
    if specification['keywords']['function_kwargs']['native_monomer']['schema'] != 'native-monomer-request/3':
        raise ValueError('expanded descriptor child requires explicit request3')
    request, geometry = admit_native_schema(specification, runtime['molecule'])
    if type(memory_bytes) is not int or memory_bytes <= 0 or request.resources.max_bytes > memory_bytes:
        raise ValueError('explicit intended Psi4 memory must cover numeric budget')
    psi4.set_memory(memory_bytes, quiet=True)
    psi4.set_options({k: v for k, v in specification['keywords'].items() if k != 'function_kwargs'})
    molecule = psi4.core.Molecule.from_schema(native_molecule_schema(runtime['molecule'], request))
    request, identity = validate_native_request(request.as_dict(), method=specification['model']['method'],
        basis=specification['model']['basis'], molecule=molecule)
    main = psi4.core.BasisSet.build(molecule, 'BASIS', 'aug-cc-pVTZ', puream=1)
    recipe = native_monomer_recipe(molecule, profile=request.declaration.profile)
    if DRHO_REFINEMENT_ITERATIONS != 10:
        raise ValueError('native Drho refinement policy changed')
    dimensions = descriptors.descriptor_dimensions(molecule, main, recipe, request.declaration, planner)
    resources = dict(max_bytes=request.resources.max_bytes, max_work=request.resources.max_work,
                     max_io_bytes=request.resources.max_io_bytes, capacity=request.resources.capacity)
    plan = planner.plan(dimensions, resources, max_replay_work=request.max_replay_work,
                        pfit_maximum_work_bytes=request.pfit_maximum_work_bytes)
    shells = [
        dict(centre=int(main.shell_to_center(i)), am=int(main.shell(i).am),
             exponents=[main.shell(i).exp(j) for j in range(main.shell(i).nprimitive)],
             coefficients=[main.shell(i).coef(j) for j in range(main.shell(i).nprimitive)])
        for i in range(main.nshell())]
    return dict(
        schema='native-campaign-descriptor-probe/2', policy_id=planner.POLICY,
        status='PASS_DESCRIPTOR_PLAN_ONLY' if not plan['refusals'] else 'REFUSED_DESCRIPTOR_PLAN',
        package=psi4.__file__, core=psi4.core.__file__, implementation=identity,
        version=str(psi4.__version__), method='PBE0/aug-cc-pVTZ',
        intended_memory_bytes=memory_bytes, request_schema='native-monomer-request/3',
        resources=resources, max_replay_work=request.max_replay_work,
        pfit_maximum_work_bytes=request.pfit_maximum_work_bytes,
        coordinate_change_bohr=check_native_geometry(molecule, geometry), main_shells=shells,
        main_descriptor_sha256=hashlib.sha256(
            json.dumps(shells, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        auxiliary_descriptor_sha256=basis_identity(recipe.auxiliary),
        atomic_descriptor_sha256=[basis_identity(s.atomic) for s in recipe.sites],
        shape_descriptor_sha256=[basis_identity(s.shape) for s in recipe.sites],
        recipe_sha256=hashlib.sha256(repr(recipe).encode()).hexdigest(),
        grid_policy='Source-bound angular rounding200->230,974->974; no grid constructed',
        source_policy_sha256=planner.FILES, plan=plan,
        qualification='Actual installed basis descriptors and ledger prediction only; '
                      'no SCF, ISA, response, fit, integral/grid generation, QCArchive, '
                      'molecular accuracy, total RSS or independent runtime acceptance. '
                      'AUX/atomic/shape/recipe fingerprints are trusted-child measurements, '
                      'not independently reconstructed shell-content/order proofs.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--memory-bytes', type=int, required=True)
    parser.add_argument('--monomer-id', required=True)
    args = parser.parse_args()
    if re.fullmatch(r'monomer-\d{4}', args.monomer_id) is None:
        raise ValueError('explicit bounded monomer identifier required')
    raw = args.input.read_bytes()
    report = scientific_probe(json.loads(raw), memory_bytes=args.memory_bytes)
    report.update(monomer_id=args.monomer_id, runtime_input_sha256=hashlib.sha256(raw).hexdigest(),
        operational_sources_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name('descriptor_probe.py'),
                         Path(__file__).with_name('resource_plan_cb63305.py'))})
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], monomer=args.monomer_id, report=str(args.output))))


if __name__ == '__main__':
    main()
