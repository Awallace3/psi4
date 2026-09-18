# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""The public front end of the DECLARED_MULTPOLE_AC asymptotic correction.

``isapol_native_ac`` is the producer and deliberately reads no options at all,
so that every refusal it makes is a statement about its arguments rather than
about ambient global state.  ``isapol_ac_options`` is the only place the option
surface is turned into that producer's arguments, which puts one specific risk
here and nowhere else: the C++ option defaults and the ``AcDeclaration`` field
defaults are two independent spellings of the same model, and if they drift the
branch has two models sharing one name.  ``test_every_option_default_is_the_
declaration_field_default`` is the guard against exactly that.

These tests also hold the front end to the three things ``plan.md`` item 3 asks
of it.

*The policy is an acceptance declaration; the producer is a named call.*  No
option runs the correction.  ``run()`` refuses before it looks at a
wavefunction unless the policy has been declared, so the model produced is
always the model a later property request admits.

*The declaration is resolved once, before any orbital exists.*  Every
ATOMIC_AC_* option is read into one frozen ``AcDeclaration`` whose ``label()``
is the model's identity, and an undeclared ionization potential is refused
rather than inferred from a HOMO eigenvalue.

*The three policy values are three models.*  NONE, FIXED_GRAC and
DECLARED_MULTPOLE_AC each refuse the other two's orbitals; only one companion
declaration is ever forwarded.

Everything here is SCF-free.  The end-to-end guard -- what the correction does
to alpha and C6-C12, that the producer's energy sits above the SCF minimum, that
the invalidated SCF seal is replaced by the producer's own verified record, and
that every refusal leaves the wavefunction intact -- lives in the gitignored
``agent_scratch/pytests/test_isapol_declared_ac_full.py``.
"""
import ast
import dataclasses
import pathlib

import pytest
import psi4
from psi4 import core
from psi4.driver.procrouting import isapol_ac_options as aco
from psi4.driver.procrouting import isapol_native_ac as ac
from psi4.driver.procrouting import isapol_oeprop as api

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]
ROOT = pathlib.Path(__file__).parents[2]
IP = 0.46380


@pytest.fixture(autouse=True)
def pristine_options():
    """Every test starts from the built-in defaults; none leaks its declaration.

    `psi4.set_options` leaves unmentioned options alone, so without this a test
    that declares an IP hands it to the next one.
    """
    core.clean_options()
    yield
    core.clean_options()


def test_every_declared_field_has_exactly_one_option():
    """Both directions: a new field needs an option, a new option needs a field."""
    fields = {f.name for f in dataclasses.fields(ac.AcDeclaration)}
    mapped = [field for _, field, _ in aco.DECLARATION_OPTIONS]
    assert set(mapped) == fields
    assert len(mapped) == len(fields), 'one option per field, no field read twice'
    # The producer controls are NOT part of the model's identity, so they must
    # not be declaration fields: they choose how hard to converge it, and the
    # admission re-checks the thresholds they were converged to.
    assert not fields & {field for _, field, _ in aco.PRODUCER_OPTIONS}


def test_every_option_default_is_the_declaration_field_default():
    """The drift guard.  Two spellings of one model must stay one model."""
    defaults = {f.name: f.default for f in dataclasses.fields(ac.AcDeclaration)}
    for name, field, convert in aco.DECLARATION_OPTIONS:
        built_in = convert(core.get_global_option(name))
        if defaults[field] is dataclasses.MISSING:
            # The only required field: its option default must be the sentinel
            # that `declaration()` refuses, never a usable value.
            assert field == 'ionization_potential'
            assert built_in == 0.
            continue
        assert built_in == defaults[field], f'{name} has drifted from AcDeclaration.{field}'
    # And the resolved default declaration is the one the producer documents.
    psi4.set_options({'atomic_ac_ionization_potential': IP})
    assert aco.declaration() == ac.REFERENCE_WATER_DECLARATION
    assert aco.declaration().label() == (
        'tanh_b3-4_k1_psi4_fa0.75_L2_nuccharge_var_ip0.46380')


def test_the_producer_controls_are_the_tightest_the_admission_accepts():
    """Defaults that a property request would refuse would make the exposed
    producer unusable out of the box, so they are pinned against the admission
    ceilings rather than merely to a number."""
    assert aco.producer_controls() == {'maxiter': 200, 'energy_threshold': 1.e-10,
                                       'gradient_threshold': 1.e-8}
    controls = aco.producer_controls()
    assert controls['energy_threshold'] <= ac.MAX_ENERGY_THRESHOLD
    assert controls['gradient_threshold'] <= ac.MAX_GRADIENT_THRESHOLD


def test_the_recorded_keys_are_every_companion_option():
    """`KEYS` is what a request echoes back under the policy; a companion option
    missing from it would be an undeclared part of the model."""
    assert set(aco.KEYS) == {name for name, _, _ in
                             aco.DECLARATION_OPTIONS + aco.PRODUCER_OPTIONS}
    assert len(aco.KEYS) == len(set(aco.KEYS)) == 14
    assert all(k.startswith('ATOMIC_AC_') for k in aco.KEYS)


@pytest.mark.parametrize('name', [n for n, _, _ in
                                  aco.DECLARATION_OPTIONS + aco.PRODUCER_OPTIONS])
def test_psi4_accepts_every_companion_option(name):
    """Each name is a real declared option, not just a string this module reads."""
    assert core.has_global_option_changed(name) is False
    core.get_global_option(name)


def test_the_policy_is_a_closed_list_of_three_models():
    assert aco.POLICY_OPTION == 'ATOMIC_SCF_ASYMPTOTIC_CORRECTION'
    assert aco.AC_POLICY == 'DECLARED_MULTPOLE_AC'
    for value in ['NONE', 'FIXED_GRAC', 'DECLARED_MULTPOLE_AC']:
        psi4.set_options({aco.POLICY_OPTION: value})
        assert core.get_global_option(aco.POLICY_OPTION) == value
    with pytest.raises(Exception):
        psi4.set_options({aco.POLICY_OPTION: 'MULTPOLE'})


def test_an_undeclared_ionization_potential_is_refused_not_inferred():
    """Zero is the sentinel.  The correction is IP-driven, and guessing the IP
    from the HOMO would make the declared model depend on the orbitals it is
    supposed to correct."""
    psi4.set_options({aco.POLICY_OPTION: 'DECLARED_MULTPOLE_AC'})
    with pytest.raises(ValueError, match='ATOMIC_AC_IONIZATION_POTENTIAL'):
        aco.declaration()
    with pytest.raises(ValueError, match='undeclared sentinel'):
        aco.declaration()
    with pytest.raises(ValueError, match='never computed from a HOMO'):
        aco.declaration()


@pytest.mark.parametrize('opts,message', [
    ({'atomic_ac_splice_upper': 2.}, 'splice radii must satisfy'),
    ({'atomic_ac_splice_lower': 0.}, 'splice radii must satisfy'),
    ({'atomic_ac_tanh_sharpness': 0.}, 'sharpness must be positive'),
    ({'atomic_ac_fermi_amaldi_scale': 1.5}, r'scale must lie in \[0, 1\]'),
    ({'atomic_ac_multipole_order': 9}, 'multipole order must be'),
    ({'atomic_ac_ionization_potential': -1.}, 'must be positive'),
])
def test_the_declaration_refuses_what_the_producer_refuses(opts, message):
    """The front end resolves the model but does not relax it: the dataclass's
    own validation is what answers, so the option surface cannot admit a
    declaration the producer would reject."""
    psi4.set_options({aco.POLICY_OPTION: 'DECLARED_MULTPOLE_AC',
                      'atomic_ac_ionization_potential': IP, **opts})
    with pytest.raises(ValueError, match=message):
        aco.declaration()


def test_join_none_is_refused_as_a_policy_confusion_not_a_bad_value():
    """`join=none` is a WELL-FORMED declaration -- of no asymptotic branch, i.e.
    of the NONE policy -- so it resolves and is then refused by name.

    Which side refuses matters: `declaration()` does not, because `none` is a
    legitimate member of `JOIN_FORMS` that other callers of the module may
    declare, and duplicating the rule in the front end would put it in two
    places.  The producer refuses it, before it touches a wavefunction, which is
    why `run(None)` gets the ValueError rather than an AttributeError.
    """
    psi4.set_options({aco.POLICY_OPTION: 'DECLARED_MULTPOLE_AC',
                      'atomic_ac_ionization_potential': IP, 'atomic_ac_join': 'NONE'})
    assert aco.declaration().join == 'none'
    with pytest.raises(ValueError, match='that is the NONE policy'):
        aco.run(None)


@pytest.mark.parametrize('value', [2.7, 2.0, True, '2'])
def test_a_count_is_never_truncated_into_a_different_model(value):
    """A truncating `int()` would turn a mis-declared 2.7 into 2 silently, and
    the multipole order is part of the model's identity."""
    with pytest.raises(ValueError, match='must be an integer'):
        aco._exact_int(value)
    assert aco._exact_int(2) == 2


def test_producing_requires_declaring():
    """The policy gate, checked before any wavefunction is touched: passing
    `None` gets the policy refusal, not an AttributeError."""
    for value in ['NONE', 'FIXED_GRAC']:
        psi4.set_options({aco.POLICY_OPTION: value,
                          'atomic_ac_ionization_potential': IP})
        with pytest.raises(ValueError, match=f'{aco.POLICY_OPTION} is {value}'):
            aco.run(None)
        with pytest.raises(ValueError, match='requires declaring it'):
            aco.run(None)


def test_only_one_companion_declaration_is_ever_forwarded():
    """The two corrections' companions are not composable: a request carries the
    declaration of the policy it names and nothing from the other."""
    psi4.set_options({aco.POLICY_OPTION: 'DECLARED_MULTPOLE_AC',
                      'atomic_ac_ionization_potential': IP})
    opts = api._correction_options()
    assert opts['scf_correction'] == 'DECLARED_MULTPOLE_AC'
    assert isinstance(opts['ac_declaration'], ac.AcDeclaration)
    assert opts['expected_grac_shift'] is None

    core.clean_options()
    psi4.set_options({aco.POLICY_OPTION: 'FIXED_GRAC',
                      'atomic_scf_expected_grac_shift': .06,
                      'atomic_ac_ionization_potential': IP})
    opts = api._correction_options()
    assert 'ac_declaration' not in opts, 'a GRAC request must not carry an AC declaration'
    assert opts['expected_grac_shift'] == .06

    core.clean_options()
    opts = api._correction_options()
    assert opts == {'scf_correction': 'NONE', 'expected_grac_shift': None}


def test_a_stale_grac_shift_cannot_pass_unnoticed_under_another_policy():
    """A leftover nonzero shift is forwarded, so it fails loudly outside
    FIXED_GRAC rather than being dropped as inapplicable."""
    psi4.set_options({aco.POLICY_OPTION: 'NONE', 'atomic_scf_expected_grac_shift': .06})
    assert api._correction_options()['expected_grac_shift'] == .06


def test_the_producer_module_reads_no_options():
    """The promise the whole front end rests on.

    If `isapol_native_ac` ever read an option itself there would be two places a
    declaration comes from, and the drift guard above would stop covering the
    model actually produced.
    """
    tree = ast.parse((ROOT / 'psi4/driver/procrouting/isapol_native_ac.py').read_text())
    forbidden = {'get_option', 'get_global_option', 'get_local_option',
                 'set_options', 'set_global_option', 'set_local_option',
                 'has_option_changed', 'has_global_option_changed'}
    assert not [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr in forbidden]


def test_the_declaration_is_frozen_and_identity_bearing():
    """`label()` is the model's name; if two declarations could share a label the
    provenance check at admission would accept the wrong orbitals."""
    d = ac.REFERENCE_WATER_DECLARATION
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.ionization_potential = .5
    alternatives = {'ionization_potential': .5, 'join': 'linear', 'b1': 2.5, 'b2': 5.,
                    'tanh_k': 2., 'bragg_table': 'camcasp', 'fa_scale': .5,
                    'multipole_order': 1, 'origin': 'com', 'shift_mode': 'fixed',
                    'shift_value': .1}
    for f in dataclasses.fields(d):
        other = dataclasses.replace(d, **{f.name: alternatives[f.name]})
        assert other != d
        if f.name == 'shift_value':
            # The one field that is conditionally part of the identity: under the
            # variational mode the constant is a converged RESULT, so a declared
            # value neither enters the model nor belongs in its name.
            assert d.shift_mode == 'variational'
            assert other.label() == d.label()
            continue
        assert other.label() != d.label(), f'{f.name} does not reach the label'
    # Under the fixed mode it is declared, and then it must reach the label.
    fixed = dataclasses.replace(d, shift_mode='fixed', shift_value=.1)
    assert fixed.label() != dataclasses.replace(fixed, shift_value=.2).label()
    assert 'fix+0.100000' in fixed.label()


def test_the_public_producer_is_exported_and_names_its_mutation():
    """It is reachable as `psi4.atomic_asymptotic_correction`, and its docstring
    states the two things a caller cannot discover from the return value: the
    wavefunction is mutated, and the reported energy is not variational."""
    assert psi4.atomic_asymptotic_correction is not None
    assert 'atomic_asymptotic_correction' in psi4.driver.p4util.util.__all__
    doc = psi4.atomic_asymptotic_correction.__doc__
    assert 'MUTATES' in doc
    assert 'seal is deliberately invalidated' in doc
    assert 'not a variational one' in doc
    assert 'never runs the correction iteration itself' in doc
