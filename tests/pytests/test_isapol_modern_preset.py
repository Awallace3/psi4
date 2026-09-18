# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""The two modern ISA-Pol presets: what they declare, and what is refused.

``isapol_modern_preset`` is the record of every line of CamCASP's
``methods/isa-pol-from-isa-A`` and ``methods/isa-pol-from-isa-A+DF``.  These
tests hold it to three claims.

*The transcription is verbatim.*  When the reference tree is present, every
declaration text is checked to occur in the pinned source, AND every content
line of both files is checked to be covered by a declaration -- so a line cannot
be quietly dropped from the manifest, which is the only way the gap list could
understate the reduction.

*The reduction is named, never applied.*  ``options()`` refuses to emit anything
until the caller has named every gap, and what it emits never contains a
stand-in for an unhonourable declaration.

*A+DF is not a variant of A.*  Its ``options()`` refuses outright.  The two
presets differ by one hunk, ``ISA-Algorithm A+DF  Zeta = 0.1`` with
``DF-PARAMETERS Lambda = 1000.0``, and emitting the A-only subset of an A+DF
request is the substitution ``plan.md:377`` forbids.

Everything here is SCF-free.  The end-to-end guard -- that the emitted options
actually run, that ``EXACT_ORBITAL`` reproduces the shipped numbers bitwise, and
that ``CAMCASP_DF`` is refused at the default angular order by the propagator's
own work budget -- lives in the gitignored
``agent_scratch/pytests/test_isapol_modern_preset_full.py``, which also guards
every measured literal quoted in ``GAPS``.
"""
import hashlib
import pathlib

import pytest
import psi4
from psi4 import core
from psi4.driver.procrouting import isapol_modern_preset as mp
from psi4.driver.procrouting import isapol_native_partition as part
from psi4.driver.procrouting import isapol_native_propagator as prop
from psi4.driver.procrouting import isapol_oeprop as api

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]

#: The reference tree, if this machine has it. The manifest is self-contained;
#: only the verbatim/coverage checks need the files.
CAMCASP = pathlib.Path.home() / 'gits' / 'CamCASP'

#: Stripped lines that open or close a block, name the file, or are a bare
#: annotation: structure rather than declarations. Everything else in either
#: file has to be covered by a Declaration.
STRUCTURAL = frozenset({
    '#  -*- coding: utf-8 -*-', 'CamCASP-commands', 'End-CamCASP-commands', 'Edit',
    'SET QUAD', 'BEGIN GRID', 'SET Lattice', 'SET NEW-PROP', 'KERNEL-INTEGRAL-PARAMETERS',
    'SET PROPAGATOR', 'SET DF-INTEGRALS', 'SET DF', 'BEGIN DF', 'Begin ISA', 'Convergence',
    'W-TAILS', 'Begin Multipoles', 'BEGIN Polarizability', 'Molecule  MOL', 'Run-Type',
    'End', 'END', 'Finish', '...'})

#: What ``ISA_A.options()`` emits, in full. Two of these are the declarations
#: this branch exposed in order to reach the preset at all -- the Bragg-Slater
#: W-TAILS recipe name and the CAMCASP_DF propagator -- and four are NOT branch
#: defaults, so a run under this preset is never comparable with a number
#: recorded at the defaults.
DECLARED_OPTIONS = {'ATOMIC_PROPERTY_RADIAL_POINTS': 100,
                    'ATOMIC_RESPONSE_RADIAL_POINTS': 100,
                    'ATOMIC_REFINEMENT_LOWER_LIMIT': 2.0,
                    'ATOMIC_REFINEMENT_UPPER_LIMIT': 4.0,
                    'ATOMIC_REFINEMENT_SEED': 1,
                    'ATOMIC_RESPONSE_BASIS': 'FITTED_AUXILIARY',
                    'ATOMIC_RESPONSE_PROPAGATOR': 'CAMCASP_DF',
                    'ATOMIC_OV_METRIC_DAMPING': 0.0,
                    'ATOMIC_OV_CHARGE_PENALTY': 1000.0,
                    'ATOMIC_PROPERTY_RECIPE': 'GENERATED_JKFIT_BRAGG_SLATER_TAIL_ISA_A',
                    'ATOMIC_MULTIPOLE_DISTRIBUTION': 'ISA_A',
                    'ATOMIC_MULTIPOLE_RANK': 4}

#: Gaps of the A preset, in declared order. A+DF has these and the functional.
ISA_A_GAPS = ('overlap_neighbour_rule', 'lebedev_angular_grid',
              'point_response_512_leg_limit', 'kernel_integral_cutoff',
              'second_unconstrained_df_block', 'isa_charge_convergence_threshold',
              'isa_tail_iteration_count', 'spherical_auxiliary_basis',
              'declared_atomaux_basis', 'isa_basis_set2', 'ac_multipole_producer')


@pytest.fixture(autouse=True)
def _clean():
    psi4.core.clean_options()
    yield
    psi4.core.clean_options()


def _declaration_lines(preset):
    return [t.strip() for d in preset.declarations for t in d.text.splitlines() if t.strip()]


def _source(preset, rel):
    if not (CAMCASP / rel).is_file():
        pytest.skip(f'reference tree absent: {CAMCASP / rel}')
    return (CAMCASP / rel).read_text()


@pytest.mark.parametrize('preset', list(mp.PRESETS.values()), ids=list(mp.PRESETS))
def test_the_pinned_sources_are_the_files_that_were_transcribed(preset):
    """A stale transcription is detectable, not silent."""
    pins = dict(preset.source_sha256)
    for rel in (preset.method_path, preset.cluster_path):
        text = _source(preset, rel)
        assert hashlib.sha256(text.encode()).hexdigest() == pins[rel], rel


@pytest.mark.parametrize('preset', list(mp.PRESETS.values()), ids=list(mp.PRESETS))
def test_every_declaration_text_occurs_in_its_pinned_source(preset):
    """``text`` is a transcription, so it is checked against the file, not trusted."""
    lines = [l.strip() for rel in (preset.method_path, preset.cluster_path)
             for l in _source(preset, rel).splitlines()]
    for text in _declaration_lines(preset):
        assert any(text in line for line in lines), text


@pytest.mark.parametrize('preset', list(mp.PRESETS.values()), ids=list(mp.PRESETS))
def test_every_declaring_line_of_both_files_is_covered(preset):
    """The reverse direction: the manifest cannot understate the preset.

    Without this, dropping a line from the manifest would remove a gap and make
    the branch look closer to the preset than it is.
    """
    texts = _declaration_lines(preset)
    for rel in (preset.method_path, preset.cluster_path):
        for line in _source(preset, rel).splitlines():
            stripped = line.strip()
            if not stripped or stripped in STRUCTURAL or stripped.startswith(('!', '(')):
                continue
            assert any(t in stripped for t in texts), (rel, line)


def test_the_manifest_is_internally_consistent():
    """Routes, gap keys and option agreement, checked rather than assumed."""
    assert set(mp.GAPS) == {d.gap for p in mp.PRESETS.values() for d in p.unhonoured}
    for preset in mp.PRESETS.values():
        for d in preset.declarations:
            assert d.route in mp.ROUTES
            assert d.honoured == (d.route != 'unhonoured')
            assert bool(d.option) == (d.route == 'option')
            assert (d.gap is not None) == (d.route == 'unhonoured')
        assert set(preset.honoured) | set(preset.unhonoured) == set(preset.declarations)
        assert not set(preset.honoured) & set(preset.unhonoured)


def test_the_declared_gaps_are_the_recorded_ones():
    assert mp.ISA_A.gaps == ISA_A_GAPS
    assert mp.ISA_A_PLUS_DF.gaps == ISA_A_GAPS + ('isa_a_plus_df_functional',)
    # The A+DF hunk is the only declaration difference beyond the cluster
    # renames: the functional line, its DF-PARAMETERS line, and the Solver line
    # that moved with them.
    a = {d.text.strip() for d in mp.ISA_A.declarations}
    adf = {d.text.strip() for d in mp.ISA_A_PLUS_DF.declarations}
    assert adf - a == {'ISA-Algorithm A+DF  Zeta = 0.1', 'DF-PARAMETERS Lambda = 1000.0',
                       'Solver        LU', '#METHOD      isa-pol-from-isa-A+DF',
                       'Title  Template Cluster file for isa-pol-from-isa-A+DF',
                       'Aux-Basis      aVTZ   Type  MC   Spherical   Use-ISA-Basis  '
                       '( this basis must be spherical )',
                       'AtomAux-Basis  aVTZ   Type  MC   Spherical   Use-ISA-Basis  '
                       '( and identical to this.       )',
                       'File-Prefix  MOL-ISA-Pol-ISA-A+DF'}


def test_options_refuse_until_every_gap_is_accepted_by_name():
    with pytest.raises(ValueError, match='has to be accepted by name'):
        mp.ISA_A.options()
    with pytest.raises(ValueError, match='has to be accepted by name'):
        mp.ISA_A.options(ISA_A_GAPS[:-1])
    with pytest.raises(ValueError, match='no such gap'):
        mp.ISA_A.options(ISA_A_GAPS + ('angular_400_rounded_to_434',))
    assert mp.ISA_A.options(ISA_A_GAPS) == DECLARED_OPTIONS
    assert mp.ISA_A.options(reversed(ISA_A_GAPS)) == DECLARED_OPTIONS


def test_no_unhonourable_declaration_leaks_a_stand_in_value():
    """The options a preset cannot reach are ABSENT, not approximated."""
    emitted = mp.ISA_A.options(ISA_A_GAPS)
    # Angular 400 and Random 2000 are the two that a well-meaning reader would
    # round; neither appears, so the caller's own declaration stands.
    for name in ('ATOMIC_PROPERTY_SPHERICAL_POINTS', 'ATOMIC_RESPONSE_SPHERICAL_POINTS',
                 'ATOMIC_REFINEMENT_POINTS', 'ATOMIC_SCF_ASYMPTOTIC_CORRECTION',
                 'ATOMIC_PROPERTY_AUXILIARY_BASIS', 'ATOMIC_LOCALIZATION_RANK_LIMIT'):
        assert name not in emitted


def test_the_a_plus_df_preset_refuses_rather_than_running_a():
    """``plan.md:377``: do not silently switch the target to the other preset."""
    for accept in ((), mp.ISA_A_PLUS_DF.gaps):
        with pytest.raises(ValueError, match='ISA-A\\+DF functional is not implemented'):
            mp.ISA_A_PLUS_DF.options(accept)
    with pytest.raises(ValueError, match='not implemented'):
        mp.ISA_A_PLUS_DF.require_complete()
    assert 'none' in mp.ISA_A_PLUS_DF.model_name
    # It still records the whole preset, including the options the A preset
    # shares, so the refusal is a statement about the functional and not a
    # missing transcription.
    assert mp.ISA_A_PLUS_DF.declared_options == DECLARED_OPTIONS


def test_require_complete_lists_what_would_have_to_be_built():
    with pytest.raises(ValueError) as exc:
        mp.ISA_A.require_complete()
    for gap in ISA_A_GAPS:
        assert gap in str(exc.value)
    assert 'Angular 400' in str(exc.value)


def test_the_model_name_says_it_is_not_the_preset():
    assert 'NOT isa-pol-from-isa-A' in mp.ISA_A.model_name
    report = mp.ISA_A.reduction_report()
    assert 'isa-pol-from-isa-A' in report
    for gap in ISA_A_GAPS:
        assert f'dropped [{gap}]' in report
    assert 'REFUSED' in mp.ISA_A_PLUS_DF.reduction_report()


def test_a_declaration_cannot_be_half_recorded():
    kw = dict(block='SET DF', text='  Type NN')
    with pytest.raises(ValueError, match='route must be one of'):
        mp.Declaration(route='honoured', **kw)
    with pytest.raises(ValueError, match='needs the option and its value'):
        mp.Declaration(route='option', **kw)
    with pytest.raises(ValueError, match='needs the option and its value'):
        mp.Declaration(route='option', option='ATOMIC_MULTIPOLE_RANK', **kw)
    with pytest.raises(ValueError, match='only an option route carries'):
        mp.Declaration(route='fixed', option='ATOMIC_MULTIPOLE_RANK', value=4,
                       note='x', **kw)
    with pytest.raises(ValueError, match='needs a named gap'):
        mp.Declaration(route='unhonoured', **kw)
    with pytest.raises(ValueError, match='needs a named gap'):
        mp.Declaration(route='unhonoured', gap='no_such_gap', **kw)
    with pytest.raises(ValueError, match='only an unhonoured declaration names a gap'):
        mp.Declaration(route='fixed', gap='kernel_integral_cutoff', note='x', **kw)
    for route in ('fixed', 'inert'):
        with pytest.raises(ValueError, match=f'a {route} declaration must say why'):
            mp.Declaration(route=route, **kw)
    with pytest.raises(ValueError, match='verbatim text'):
        mp.Declaration(block='SET DF', text='   ', route='report')
    # A single option name may be given as a bare string or as a tuple.
    one = mp.Declaration(route='option', option='ATOMIC_MULTIPOLE_RANK', value=4, **kw)
    assert one.option == ('ATOMIC_MULTIPOLE_RANK',) and one.honoured


def test_a_manifest_cannot_pin_nothing_or_contradict_itself():
    kw = dict(name='x', method_path=mp.ISA_A.method_path,
              cluster_path=mp.ISA_A.cluster_path, model_name='x')
    fixed = mp.Declaration('SET DF', '  Type NN', 'fixed', note='x')
    with pytest.raises(ValueError, match='records nothing'):
        mp.PresetManifest(declarations=(), **kw)
    with pytest.raises(ValueError, match='not pinned in source_sha256'):
        mp.PresetManifest(declarations=(fixed,), source_sha256=(), **kw)
    with pytest.raises(ValueError, match='must pin a sha256'):
        mp.PresetManifest(declarations=(fixed,),
                          source_sha256=((mp.ISA_A.method_path, 'deadbeef'),
                                         (mp.ISA_A.cluster_path, 'deadbeef')), **kw)
    with pytest.raises(ValueError, match='declared as both'):
        mp.PresetManifest(declarations=(
            mp.Declaration('a', '  Rank 4', 'option', option='ATOMIC_MULTIPOLE_RANK', value=4),
            mp.Declaration('b', '  Rank 3', 'option', option='ATOMIC_MULTIPOLE_RANK', value=3)),
            **kw)
    # The reference's own redundancy is allowed: ISA-GRID twice, same value.
    assert mp.PresetManifest(declarations=(
        mp.Declaration('a', '  DF Type ISA-GRID', 'option',
                       option='ATOMIC_MULTIPOLE_DISTRIBUTION', value='ISA_A'),
        mp.Declaration('b', '  DIST-ALG ISA-GRID', 'option',
                       option='ATOMIC_MULTIPOLE_DISTRIBUTION', value='ISA_A')),
        **kw).declared_options == {'ATOMIC_MULTIPOLE_DISTRIBUTION': 'ISA_A'}


def test_psi4_itself_accepts_every_option_the_preset_declares():
    """Each emitted name/value is a real declared option, read back as given."""
    emitted = mp.ISA_A.options(ISA_A_GAPS)
    psi4.set_options(emitted)
    for name, value in emitted.items():
        assert core.get_global_option(name) == value


def test_the_two_newly_exposed_declarations_default_to_the_shipped_model():
    """Neither exposure moved a number that was already recorded."""
    assert core.get_global_option('ATOMIC_RESPONSE_PROPAGATOR') == 'NONE'
    assert core.get_global_option('ATOMIC_PROPERTY_RECIPE') == 'GENERATED_JKFIT_ISA_A'
    # NONE is deliberately not a declaration: the propagator module is not
    # entered at all, which is what keeps earlier numbers reproducible.
    assert api.PROPAGATOR_DECLARATIONS['NONE'] is None
    assert api.PROPAGATOR_DECLARATIONS['EXACT_ORBITAL'] is prop.EXACT_ORBITAL_PROPAGATOR
    assert api.PROPAGATOR_DECLARATIONS['CAMCASP_DF'] is prop.CAMCASP_DF_PROPAGATOR
    assert set(api.RECIPE_TAIL_POLICIES) == {'GENERATED_JKFIT_ISA_A',
                                             'GENERATED_JKFIT_BRAGG_SLATER_TAIL_ISA_A'}


def test_the_camcasp_df_propagator_is_the_reference_kernel_block():
    """The four KERNEL-INTEGRAL-PARAMETERS the option carries, and the fifth it does not."""
    d = prop.CAMCASP_DF_PROPAGATOR
    assert (d.two_electron, d.kernel_projection, d.kernel_density) == (
        'density_fitted', 'auxiliary_metric', 'fitted_auxiliary')
    s = d.smoothing
    assert (s.method, s.rho_epsilon, s.f_max, s.fd_delta, s.fd_alpha) == (
        'FD', 1.e-8, 1000.0, 0.01, 1.0)
    assert d.rebuilds_kernel
    assert 'KERNEL-INTEGRAL-CUTOFF' in mp.GAPS['kernel_integral_cutoff']


def test_camcasp_df_is_refused_without_the_fit_it_projects_through():
    """The propagator and the response basis are one declaration, checked early."""
    psi4.set_options({'atomic_response_propagator': 'CAMCASP_DF',
                      'atomic_response_basis': 'DIRECT_OV'})
    with pytest.raises(ValueError, match='forms no fit for it to project through'):
        api.validate_request(None, ('ATOMIC_POLARIZABILITIES',))
    psi4.set_options({'atomic_response_basis': 'FITTED_AUXILIARY'})
    # Now the coupling is satisfied and the refusal that remains is the missing
    # wavefunction, which is checked after the declarations.
    with pytest.raises(ValueError, match='actual restricted C1 wavefunction'):
        api.validate_request(None, ('ATOMIC_POLARIZABILITIES',))
    assert api._propagator_option() == dict(propagator=prop.CAMCASP_DF_PROPAGATOR)


def test_an_undeclared_propagator_or_recipe_is_refused():
    with pytest.raises(Exception):
        psi4.set_options({'atomic_response_propagator': 'DF'})
    with pytest.raises(Exception):
        psi4.set_options({'atomic_property_recipe': 'GENERATED_JKFIT_BRAGG_SLATER'})


def test_the_recipe_name_is_the_tail_policy_and_the_policies_differ():
    """``W-TAILS R1-Multiplier = 1.5`` is per-element, not 1.5 bohr."""
    assert api.RECIPE_TAIL_POLICIES['GENERATED_JKFIT_ISA_A'] == 'flat_1.5_bohr'
    assert api.RECIPE_TAIL_POLICIES[
        'GENERATED_JKFIT_BRAGG_SLATER_TAIL_ISA_A'] == 'bragg_slater_1.5'
    # Water's two elements, at the multiplier the preset declares.
    assert part.bragg_slater_tail_cutoff(8, 1.5) == 1.7007533897210307
    assert part.bragg_slater_tail_cutoff(1, 1.5) == 1.4172944914341925
    # Neither equals the demo recipe's flat cutoff, on either element, which is
    # why the two names are two models and not a tolerance.
    assert part.bragg_slater_tail_cutoff(8, 1.5) != 1.5
    assert part.bragg_slater_tail_cutoff(1, 1.5) != 1.5
