"""The declared MULTPOLE/Tozer-Handy asymptotic correction as its own policy.

Small genuine PBE0/cc-pVDZ water runs.  No reference eigenvalue table is
asserted here: the reference spectrum the form was derived against belongs to
the aug-cc-pVTZ audit, and this file checks the POLICY -- its declaration
identity, its refusals, its reproducibility and its routing -- not agreement
with DALTON.

Nothing in this file widens ``FIXED_GRAC``, refits a GRAC beta, or lets the
declared form be reported under another policy's provenance.  The three
policies are different models and the tests assert that each refuses the
others' wavefunctions.
"""
import ast
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import psi4
from psi4 import core
from psi4.driver.p4util import OptionsState
from psi4.driver.procrouting import isapol_native_ac as ac
from psi4.driver.procrouting import isapol_native_correction as correction
from psi4.driver.procrouting import isapol_native_response as response
from psi4.driver.procrouting.scf_proc.scf_iterator import _scf_state_signature

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'psi4/driver/procrouting/isapol_native_ac.py'
DECL = ac.REFERENCE_WATER_DECLARATION
GEOMETRY = ('0 1\nO 0. 0. 0.\nH -1.45365196 0. -1.12168732\n'
            'H 1.45365196 0. -1.12168732\nunits bohr\nsymmetry c1\n'
            'no_com\nno_reorient\n')

# Recorded from this file's own PBE0/cc-pVDZ run; a reproducibility anchor for
# the iteration, not a reference value and not a claim about DALTON.
RECORDED = dict(iterations=17, shift=.164581425171, homo=-.299218530996,
                lumo=.078224857057, energy=-76.338733436310,
                reference_energy=-76.338758964954, grid_points=66198)


@contextmanager
def options(values):
    saved = OptionsState(*[[k] if not isinstance(k, tuple) else list(k) for k in values])
    try:
        for key, value in values.items():
            if isinstance(key, tuple):
                core.set_local_option(*key, value)
            else:
                core.set_global_option(key, value)
        yield
    finally:
        saved.restore()


def _water():
    """One genuine uncorrected canonical PBE0 SCF.  Never a corrected SCF."""
    scf = dict(reference='rks', scf_type='pk', df_scf_guess=False, maxiter=100,
               fail_on_maxiter=True, e_convergence=1e-10, d_convergence=1e-10,
               dft_alpha=.25)
    with options({'BASIS': 'cc-pvdz',
                  **{('SCF', k.upper()): v for k, v in scf.items()}}):
        mol = psi4.geometry(GEOMETRY)
        _, wfn = psi4.energy('pbe0', molecule=mol, return_wfn=True)
    return wfn


@pytest.fixture(scope='module')
def pristine():
    """Module-wide uncorrected wavefunction.  Tests must not mutate it."""
    return _water()


@pytest.fixture(scope='module')
def record(pristine):
    return ac.declared_ac_orbitals(pristine, DECL)


@pytest.fixture(autouse=True)
def public_defaults():
    """The declared policy is never an ambient default; the option stays NONE."""
    with options({'ATOMIC_SCF_ASYMPTOTIC_CORRECTION': 'NONE',
                  'ATOMIC_SCF_EXPECTED_GRAC_SHIFT': 0.}):
        yield


@pytest.fixture
def fresh():
    """A wavefunction a single test may mutate by applying the correction."""
    return _water()


# ------------------------------------------------------------ the declaration
def test_reference_declaration_is_the_transcribed_input():
    # output_1/H2O_aTZ_A.dal: .DFTAC / MULTPOLE / TANH / 0.46380 0.46380 3.0 4.0
    assert (DECL.ionization_potential, DECL.join, DECL.b1, DECL.b2) == (.46380, 'tanh', 3., 4.)
    # Derivations, carried as declared fields rather than hidden: c_FA = 1 - a_x
    # for PBE0 and the variational Tozer-Handy constant Delta = I + eps_HOMO.
    assert (DECL.fa_scale, DECL.shift_mode, DECL.shift_value) == (.75, 'variational', 0.)
    assert DECL.label() == 'tanh_b3-4_k1_psi4_fa0.75_L2_nuccharge_var_ip0.46380'
    assert DECL != replace(DECL, fa_scale=1.) and DECL == replace(DECL)
    assert hash(DECL) == hash(replace(DECL))
    with pytest.raises(FrozenInstanceError):
        DECL.fa_scale = 1.
    # Integer-valued reals are coerced, so 3 and 3.0 are the same declaration.
    assert replace(DECL, b1=3, b2=4) == DECL
    assert isinstance(replace(DECL, ionization_potential=1).ionization_potential, float)


@pytest.mark.parametrize('bad', [
    dict(join='grac'), dict(join='GRAC'), dict(bragg_table='bragg'),
    dict(origin='homo'), dict(shift_mode='fitted'),
    dict(multipole_order=4), dict(multipole_order=-1), dict(multipole_order=2.),
    dict(multipole_order=True), dict(ionization_potential=0.),
    dict(ionization_potential=-.4638), dict(ionization_potential=float('nan')),
    dict(ionization_potential=float('inf')), dict(b1=0.), dict(b1=4., b2=3.),
    dict(b1=4.), dict(tanh_k=0.), dict(tanh_k=-1.), dict(fa_scale=-.1),
    dict(fa_scale=1.1), dict(shift_value=float('nan')),
])
def test_declaration_refuses_undeclared_or_impossible_fields(bad):
    with pytest.raises(ValueError):
        replace(DECL, **bad)


def test_a_different_beta_or_scale_is_a_different_declaration():
    # The rule that kept FIXED_GRAC from being widened applies inside this
    # policy too: any changed field is a different model, with its own label.
    for change in (dict(fa_scale=1.), dict(b1=3.5, b2=4.7), dict(bragg_table='camcasp'),
                   dict(join='linear'), dict(tanh_k=3.), dict(multipole_order=0),
                   dict(origin='com'), dict(shift_mode='fixed', shift_value=.1332)):
        other = replace(DECL, **change)
        assert other != DECL and other.label() != DECL.label()


# -------------------------------------------------- the geometric/potential parts
def test_multipole_potential_converges_to_two_point_charges():
    origin = np.zeros(3)
    charges = np.array([[.3, .1, -.2], [-.4, .05, .3]])
    q = np.array([1., -1.])
    m0 = q.sum()
    m1 = (q[:, None]*charges).sum(0)
    m2 = np.einsum('k,ka,kb->ab', q, charges, charges)
    m3 = np.einsum('k,ka,kb,kc->abc', q, charges, charges, charges)
    points = np.array([[6., 0., 0.], [0., 8., 1.], [-5., 5., 5.]])
    norms = np.linalg.norm(points, axis=1)
    exact = (q[None, :]/np.linalg.norm(points[:, None, :] - charges[None, :, :], axis=2)).sum(1)
    errors = [float(np.max(np.abs(ac.multipole_potential(points, norms, (m0, m1, m2, m3), L) - exact)))
              for L in range(4)]
    assert errors == sorted(errors, reverse=True)
    assert errors[3] < .05*errors[0]


def _fake_grid(points, blocksize=7):
    blocks = []
    for start in range(0, len(points), blocksize):
        chunk = np.asarray(points[start:start + blocksize], dtype=float)
        blocks.append(SimpleNamespace(npoints=lambda n=len(chunk): n,
                                      x=lambda c=chunk: c[:, 0].copy(),
                                      y=lambda c=chunk: c[:, 1].copy(),
                                      z=lambda c=chunk: c[:, 2].copy()))
    return SimpleNamespace(blocks=lambda: list(blocks))


@pytest.mark.parametrize('join', ['none', 'linear', 'tanh', 'tanh_raw'])
def test_splice_weight_is_identically_zero_inside_b1(join):
    # The multipole branch has a pole at its origin, so a small-but-nonzero f
    # inside the core is not a small perturbation.  Every join form must give
    # exactly zero there, not merely a tiny number.
    mol = psi4.geometry(GEOMETRY)
    radius = ac.BRAGG_SLATER_TABLES['psi4'][8]
    grid = _fake_grid(np.column_stack([np.linspace(.01, 8.*radius, 40),
                                       np.zeros(40), np.zeros(40)]))
    declaration = replace(DECL, join=join)
    f, vector, norm = ac.splice_weight(mol, grid, declaration)
    f = np.concatenate(f)
    x = np.concatenate([np.linalg.norm(v, axis=1) for v in vector])
    assert x.shape == f.shape == (40,)
    inside = np.linspace(.01, 8.*radius, 40) < DECL.b1*radius
    assert np.all(f[inside] == 0.)
    if join == 'none':
        assert np.all(f == 0.)
    else:
        assert np.all(np.diff(f) >= 0.) and np.all(f <= 1.)
        assert f[-1] > 0.
    if join in ('linear', 'tanh'):
        outside = np.linspace(.01, 8.*radius, 40) > DECL.b2*radius
        assert np.all(f[outside] == 1.)


def test_splice_weight_refuses_an_element_absent_from_the_declared_table():
    mol = psi4.geometry('0 1\nLi 0. 0. 0.\nH 0. 0. 3.\nunits bohr\nsymmetry c1\n'
                        'no_com\nno_reorient\n')
    grid = _fake_grid(np.array([[0., 0., 10.]]))
    with pytest.raises(ValueError, match='no fallback radius is substituted'):
        ac.splice_weight(mol, grid, replace(DECL, bragg_table='camcasp'))
    assert np.concatenate(ac.splice_weight(mol, grid, DECL)[0]).shape == (1,)


def test_multipole_origins_are_declared_not_inferred():
    mol = psi4.geometry(GEOMETRY)
    geometry = np.asarray(mol.geometry().to_array(), dtype=float)
    np.testing.assert_allclose(ac.multipole_origin(mol, 'atom0'), geometry[0])
    charge = np.array([8., 1., 1.])
    np.testing.assert_allclose(ac.multipole_origin(mol, 'nuccharge'),
                               (charge[:, None]*geometry).sum(0)/10.)
    com = ac.multipole_origin(mol, 'com')
    assert abs(com[2] - ac.multipole_origin(mol, 'nuccharge')[2]) > 1e-3


def test_analytic_moments_carry_number_density_not_charge(pristine):
    mints = core.MintsHelper(pristine.basisset())
    origin = ac.multipole_origin(pristine.molecule(), 'nuccharge')
    m0, m1, m2, m3 = ac.density_moments(mints, pristine.Da(), origin, 3)
    assert m0 == pytest.approx(10., abs=1e-9)
    assert np.trace(m2) > 0 and m2.shape == (3, 3)
    np.testing.assert_allclose(m2, m2.T, atol=0)
    np.testing.assert_allclose(m3, m3.transpose(1, 0, 2), atol=0)
    np.testing.assert_allclose(m3, m3.transpose(0, 2, 1), atol=0)
    # Order 0 must not silently supply higher moments.
    assert ac.density_moments(mints, pristine.Da(), origin, 0)[1].tolist() == [0., 0., 0.]


# ------------------------------------------------------------------ the producer
@pytest.mark.parametrize('kwargs,error', [
    (dict(declaration=None), TypeError),
    (dict(declaration=dict(ionization_potential=.4638)), TypeError),
    (dict(declaration=replace(DECL, join='none')), ValueError),
    (dict(shift_damping=0.), ValueError),
    (dict(shift_damping=1.5), ValueError),
    (dict(maxiter=0), ValueError),
    (dict(maxiter=2.), ValueError),
    (dict(diis_subspace=0), ValueError),
    (dict(energy_threshold=1e-6), ValueError),
    (dict(energy_threshold=0.), ValueError),
    (dict(gradient_threshold=1e-4), ValueError),
    (dict(gradient_threshold=float('nan')), ValueError),
])
def test_producer_refuses_before_iterating(pristine, monkeypatch, kwargs, error):
    monkeypatch.setattr(ac, '_AcKohnSham', lambda *a, **k: pytest.fail('iteration entered'))
    monkeypatch.setattr(psi4, 'energy', lambda *a, **k: pytest.fail('hidden SCF'))
    declaration = kwargs.pop('declaration', DECL)
    with pytest.raises(error):
        ac.declared_ac_orbitals(pristine, declaration, **kwargs)


def test_producer_verifies_the_uncorrected_seal_and_manufactures_none(pristine, monkeypatch):
    monkeypatch.setattr(ac, '_AcKohnSham', lambda *a, **k: pytest.fail('iteration entered'))
    monkeypatch.delattr(pristine, '_scf_convergence_evidence', raising=True)
    with pytest.raises(ValueError, match='SCF convergence evidence is required'):
        ac.declared_ac_orbitals(pristine, DECL)


def test_producer_applies_nothing(pristine, record):
    assert getattr(pristine, '_declared_ac_evidence', None) is None
    assert record.declaration == DECL
    # The starting wavefunction is still the sealed uncorrected one.
    correction.require_scf_seal(pristine)
    assert correction.validate_correction(pristine, require_canonical=True).policy == 'NONE'


def test_declared_ac_reproduces_its_recorded_run(record, pristine):
    c = record.convergence
    assert c.iterations == RECORDED['iterations']
    for key in ('shift', 'homo', 'lumo', 'energy', 'reference_energy'):
        assert getattr(c, key) == pytest.approx(RECORDED[key], abs=1e-8)
    assert c.grid_points == RECORDED['grid_points'] and c.shift_clamped == 0
    assert abs(c.delta_energy) < c.energy_threshold
    assert 0 <= c.orbital_gradient < c.gradient_threshold
    # No energy functional: the corrected energy must lie ABOVE the minimum.
    assert c.energy > c.reference_energy == pristine.energy()
    # The variational Tozer-Handy constant sits at its own fixed point.
    assert c.shift == pytest.approx(DECL.ionization_potential + c.homo, abs=1e-6)
    # The correction opens the gap relative to the uncorrected spectrum.
    epsilon = np.asarray(pristine.epsilon_a())
    nocc = record.nocc
    assert c.lumo - c.homo > epsilon[nocc] - epsilon[nocc - 1]
    # Exactly integer occupations: the native contract compares D to C_occ C_occ^T.
    S, C = np.asarray(pristine.S()), record.orbitals
    np.testing.assert_array_equal(record.density, C[:, :nocc] @ C[:, :nocc].T)
    assert np.abs(C.T @ S @ C - np.eye(C.shape[1])).max() < 1e-10


# -------------------------------------------------------- applying and validating
def test_apply_validate_and_every_mislabel_refusal(fresh, record):
    before = _scf_state_signature(fresh)
    produced = ac.declared_ac_orbitals(fresh, DECL)
    # Two wavefunction instances of the same declared SCF give the same record.
    assert produced.convergence.shift == pytest.approx(record.convergence.shift, abs=1e-10)
    assert _scf_state_signature(fresh) == before
    with pytest.raises(TypeError):
        ac.apply_declared_ac(fresh, produced.orbitals)
    assert ac.apply_declared_ac(fresh, produced) is fresh
    np.testing.assert_array_equal(np.asarray(fresh.Ca()), produced.orbitals)
    np.testing.assert_array_equal(np.asarray(fresh.Da()), produced.density)
    np.testing.assert_array_equal(np.asarray(fresh.epsilon_a()), produced.energies)
    assert fresh.energy() == produced.convergence.energy
    assert _scf_state_signature(fresh) != before

    provenance = ac.validate_declared_ac(fresh, DECL)
    assert provenance == correction.validate_correction(
        fresh, scf_correction='DECLARED_MULTPOLE_AC', ac_declaration=DECL)
    assert provenance.policy == 'DECLARED_MULTPOLE_AC'
    assert provenance.shift == produced.convergence.shift
    assert 'Fermi-Amaldi' in provenance.response_description
    assert 'no asymptotic-correction kernel derivative' in provenance.response_description
    with pytest.raises(FrozenInstanceError):
        provenance.shift = .1

    # The other two policies must never describe these orbitals.
    for policy, extra in (('NONE', {}), ('FIXED_GRAC', dict(expected_grac_shift=.0649))):
        with pytest.raises(ValueError, match='carry a declared asymptotic correction'):
            correction.validate_correction(fresh, scf_correction=policy, **extra)
    # And the declared policy must never be admitted without its declaration.
    with pytest.raises(TypeError, match='explicit AcDeclaration'):
        correction.validate_correction(fresh, scf_correction='DECLARED_MULTPOLE_AC')
    with pytest.raises(ValueError, match='does not match the orbitals'):
        correction.validate_correction(fresh, scf_correction='DECLARED_MULTPOLE_AC',
                                       ac_declaration=replace(DECL, fa_scale=1.))
    with pytest.raises(ValueError, match='join=none'):
        ac.validate_declared_ac(fresh, replace(DECL, join='none'))
    with pytest.raises(ValueError, match='requires no expected GRAC shift'):
        correction.validate_correction(fresh, scf_correction='DECLARED_MULTPOLE_AC',
                                       ac_declaration=DECL, expected_grac_shift=.0649)
    with pytest.raises(ValueError, match='requires no asymptotic-correction declaration'):
        correction.validate_correction(fresh, scf_correction='NONE', ac_declaration=DECL)
    with pytest.raises(ValueError, match='unsupported'):
        correction.validate_correction(fresh, scf_correction='MULTPOLE')
    # Corrections are not composed.
    with pytest.raises(ValueError, match='not composed'):
        ac.apply_declared_ac(fresh, produced)
    # Any further state change invalidates the record, exactly as a seal would be.
    np.asarray(fresh.Ca())[0, 0] += 1e-10
    with pytest.raises(ValueError, match='stale'):
        ac.validate_declared_ac(fresh, DECL)


def test_record_must_belong_to_this_wavefunction(fresh, record):
    # A record is admitted only against the SCF state it was produced from, and
    # that state must still be the sealed one.  Both are checked, in that order.
    elsewhere = replace(record, convergence=replace(
        record.convergence, reference_energy=record.convergence.reference_energy + 1e-9))
    with pytest.raises(ValueError, match='different SCF state'):
        ac.apply_declared_ac(fresh, elsewhere)
    smaller = replace(record, nocc=record.nocc - 1)
    with pytest.raises(ValueError, match='basis/occupation'):
        ac.apply_declared_ac(fresh, smaller)
    # Touching the wavefunction first is caught earlier still, by the seal.
    fresh.set_energy(record.convergence.reference_energy + 1e-9)
    with pytest.raises(ValueError, match='stale'):
        ac.apply_declared_ac(fresh, record)


@pytest.mark.parametrize('mutation', ['iterations', 'clamped', 'energy', 'gradient',
                                      'threshold', 'below_minimum', 'fixed_point'])
def test_validator_refuses_a_doctored_record(fresh, mutation):
    produced = ac.declared_ac_orbitals(fresh, DECL)
    ac.apply_declared_ac(fresh, produced)
    declaration, convergence, signature, basis = fresh._declared_ac_evidence
    changes = dict(
        iterations=dict(iterations=0),
        clamped=dict(shift_clamped=1),
        energy=dict(delta_energy=1e-6),
        gradient=dict(orbital_gradient=-1.),
        threshold=dict(energy_threshold=1e-6, delta_energy=1e-7),
        below_minimum=dict(energy=convergence.reference_energy - 1e-9),
        fixed_point=dict(shift=convergence.shift + 1e-4),
    )[mutation]
    fresh._declared_ac_evidence = (declaration, replace(convergence, **changes),
                                   signature, basis)
    with pytest.raises(ValueError):
        ac.validate_declared_ac(fresh, DECL)


def test_validator_refuses_a_malformed_or_absent_record(pristine):
    with pytest.raises(ValueError, match='evidence is required'):
        ac.validate_declared_ac(pristine, DECL)
    doubled = SimpleNamespace(_declared_ac_evidence=(DECL, object(), None, None))
    with pytest.raises(ValueError, match='malformed'):
        ac.validate_declared_ac(doubled, DECL)


# ------------------------------------------------------------------- the routing
def test_response_factory_routes_the_declared_policy(fresh):
    produced = ac.declared_ac_orbitals(fresh, DECL)
    ac.apply_declared_ac(fresh, produced)
    kwargs = dict(caller_converged=True, kernel='alda_slater_pw92', exact_exchange=.25,
                  local_scale=.75, scf_correction='DECLARED_MULTPOLE_AC')
    with pytest.raises(ValueError, match='requires an explicit ALDA response'):
        response.native_response_from_wavefunction(
            fresh, **{**kwargs, 'kernel': 'no_local', 'local_scale': 0.},
            ac_declaration=DECL)
    with pytest.raises(TypeError, match='explicit AcDeclaration'):
        response.native_response_from_wavefunction(fresh, grid=np.ones((1, 4)), **kwargs)
    # An independent small quadrature; not a production integration claim.
    go = core.IsaGridOptions()
    go.radial_points, go.spherical_points = 12, 50
    grid = core.IsaGrid(fresh.molecule().clone(), go)
    points = np.column_stack((grid.x(), grid.y(), grid.z(), grid.w()))
    result = response.native_response_from_wavefunction(fresh, grid=points,
                                                        ac_declaration=DECL, **kwargs)
    assert np.isfinite(result.at_frequency(.4).raw_coupled).all()
    np.testing.assert_array_equal(np.asarray(result.provider.orbitals()), produced.orbitals)
    np.testing.assert_array_equal(np.asarray(result.provider.energies()), produced.energies)
    owned = result.correction_provenance
    assert owned == ac.validate_declared_ac(fresh, DECL)
    assert 'not an SCF seal' in result.convergence_evidence
    assert 'seal' not in owned.response_description


# ---------------------------------------------------- the surface that stays fixed
def test_the_grac_policy_and_option_surface_were_not_widened():
    assert correction.POLICIES == ('NONE', 'FIXED_GRAC', 'DECLARED_MULTPOLE_AC')
    # The C++ option cannot carry a declaration object, so it is deliberately
    # NOT widened: the declared form is reachable only through the explicit
    # Python ac_declaration argument.
    opts = (ROOT / 'psi4/src/read_options.cc').read_text()
    assert 'options.add_str("ATOMIC_SCF_ASYMPTOTIC_CORRECTION", "NONE", "NONE FIXED_GRAC")' in opts
    assert 'DECLARED_MULTPOLE_AC' not in opts
    # No GRAC control, LibXC tweak, option write or SCF driver call inside the
    # AC module: the correction is applied outside the functional, by name.
    tree = ast.parse(SOURCE.read_text())
    forbidden = {'set_options', 'set_global_option', 'set_local_option', 'set_tweak',
                 'set_grac_shift', 'set_grac_alpha', 'set_grac_beta', 'set_grac_x_functional',
                 'set_grac_c_functional', 'compute_energy', 'optimize', 'gradient',
                 'set_variable', 'set_scalar_variable'}
    calls = [n.func for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)]
    assert not [c.attr for c in calls if c.attr in forbidden]
    # ``energy`` appears only as the wavefunction's own getter, never as the
    # driver entry point, so the module cannot run a hidden (corrected) SCF.
    receivers = {ast.unparse(c.value) for c in calls if c.attr == 'energy'}
    assert receivers == {'self.wfn', 'wfn'}
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Import)
                and any(a.name == 'psi4' for a in n.names)]
    # The declared form is not a GRAC profile: the module never builds GRAC's
    # LB94/VWN attachment, so it cannot have been reached by refitting a beta.
    text = SOURCE.read_text()
    assert 'XC_GGA_X_LB' not in text and 'XC_LDA_C_VWN' not in text
