import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('s66_entries', Path(__file__).with_name('prepare_entries.py'))
entries_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entries_module)


def inventory():
    return dict(source_pins={}, summary=dict(tolerance_bohr=1e-7), monomers=[
        dict(id='monomer-0001', symbols=['O', 'H', 'H'], charge=0., multiplicity=1,
             geometry_bohr=[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], formula='H2O',
             atom_order='explicit', representative=dict(dimer_id='dimer1', role='A'),
             members=[dict(dimer_id='dimer1', system='synthetic', role='A', R=1.,
                           fragment_indices=[0, 1, 2], rotation=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                           translation_bohr=[0., 0., 0.], max_abs_error_bohr=0.)])])


def test_reference_attributes_never_become_runtime_inputs():
    entries = entries_module.entries_from_inventory(inventory(), {
        'dimer1': {'SAPT0/aDZ DISP': '-1.5', 'SAPT2+3(CCD)/aTZ DISP': '-2.1',
                   'SAPT0/aQZ DISP': 'nan', 'Total_ref': '-5.'}})
    assert len(entries) == 1
    entry = entries[0]
    assert entry['additional_keywords'] == {} and 'extras' not in entry['molecule']
    values = entry['attributes']['dimer_contexts'][0]['sapt_reference_components_kcal_mol']
    assert values == {'SAPT0/aDZ DISP': -1.5, 'SAPT2+3(CCD)/aTZ DISP': -2.1, 'SAPT0/aQZ DISP': None}
    assert entry['molecule']['geometry'] == [0., 0., 0., 1., 0., 0., 0., 1., 0.]
    assert entry['molecule']['fix_com'] and entry['molecule']['fix_orientation']
    assert entry['molecule']['fix_symmetry'] == 'c1'


def test_infinite_reference_is_refused_not_silently_repaired():
    with pytest.raises(ValueError, match='infinite'):
        entries_module.entries_from_inventory(inventory(), {'dimer1': {'SAPT0/aDZ DISP': 'inf'}})
