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
    if name not in ('resource_plan_cb63305', 'resource_plan_ed561666',
            'resource_plan_grid400', 'resource_plan_qomp', 'qualified_qomp',
            'descriptor_profiles', 'descriptor_probe'):
        raise ValueError('closed installed descriptor helpers required')
    path = Path(__file__).with_name(name+'.py')
    raw = path.read_bytes()
    spec = importlib.util.spec_from_file_location('_expanded_descriptor_'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def scientific_probe(runtime, *, memory_bytes, producer='cb63305', scope='default', runtime_view=None):
    """Real basis/shell/recipe declarations only; no electronic state."""
    profile = helper('descriptor_profiles').profile(producer, scope=scope)
    guard = None
    if producer == 'qomp':
        guard = helper('qualified_qomp').runtime()
        guard.load_view(runtime_view)
        guard.validate_python_path(sys.path)
    elif runtime_view is not None:
        raise ValueError('runtime view is qomp-only')
    import psi4
    if guard is not None:
        guard.validate_python_path(sys.path)
        if (str(Path(psi4.__file__).resolve()) != guard.PREFIX+'/lib/python3.13/site-packages/psi4/__init__.py'
                or str(Path(psi4.core.__file__).resolve().parent)
                != guard.PREFIX+'/lib/python3.13/site-packages/psi4'):
            raise ValueError('foreign qomp scientific import')
    from psi4.driver.procrouting.isapol_schema import (
        admit_native_schema, native_molecule_schema, validate_native_request, check_native_geometry)
    from psi4.driver.procrouting.isapol_recipes import native_monomer_recipe
    from psi4.driver.procrouting.isapol_distribution import basis_identity
    from psi4.driver.procrouting.isapol_native_partition import DRHO_REFINEMENT_ITERATIONS
    planner, descriptors = helper(profile['planner']), helper('descriptor_probe')
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
    if DRHO_REFINEMENT_ITERATIONS != profile['drho_refinement_iterations']:
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
    result = dict(
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
        grid_policy=('Source-bound response angular rounding200->230; integration400/2030; no grid constructed'
            if producer in ('grid400', 'qomp')
            else 'Source-bound angular rounding200->230,974->974; no grid constructed'),
        source_policy_sha256=planner.FILES, plan=plan,
        qualification='Actual installed basis descriptors and ledger prediction only; '
                      'no SCF, ISA, response, fit, integral/grid generation, QCArchive, '
                      'molecular accuracy, total RSS or independent runtime acceptance. '
                      'AUX/atomic/shape/recipe fingerprints are trusted-child measurements, '
                      'not independently reconstructed shell-content/order proofs.')
    if scope == 'full-bank':
        result['campaign_scope'] = scope
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--memory-bytes', type=int, required=True)
    parser.add_argument('--monomer-id', required=True)
    parser.add_argument('--producer', choices=('cb63305', 'ed561666', 'grid400', 'qomp'), default='cb63305')
    parser.add_argument('--runtime-view', type=Path)
    parser.add_argument('--scope', choices=('default', 'full-bank'), default='default')
    args = parser.parse_args()
    if bool(getattr(args, 'runtime_view', None)) != (args.producer == 'qomp'):
        raise ValueError('qomp requires its runtime view; historical children do not accept it')
    scope = getattr(args, 'scope', 'default')
    if re.fullmatch(r'monomer-\d{4}', args.monomer_id) is None:
        raise ValueError('explicit bounded monomer identifier required')
    profile = helper('descriptor_profiles').profile(args.producer, scope=scope)
    if scope == 'full-bank' and not 1 <= int(args.monomer_id[-4:]) <= 131:
        raise ValueError('full-bank descriptor monomer outside exact131 identities')
    if profile['selected_monomers'] is not None and args.monomer_id not in profile['selected_monomers']:
        raise ValueError('candidate descriptor is fixed to selected monomer0008')
    raw = args.input.read_bytes()
    report = scientific_probe(json.loads(raw), memory_bytes=args.memory_bytes, producer=args.producer,
                              **(dict(runtime_view=args.runtime_view.read_bytes())
                                 if args.producer == 'qomp' and args.runtime_view else {}),
                              **(dict(scope=scope) if scope != 'default' else {}))
    sources = [Path(__file__).with_name(name) for name in profile['child_sources']]
    report.update(monomer_id=args.monomer_id, runtime_input_sha256=hashlib.sha256(raw).hexdigest(),
        operational_sources_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources})
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], monomer=args.monomer_id, report=str(args.output))))


if __name__ == '__main__':
    main()
