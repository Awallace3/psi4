"""Independent algebraic basis checks; production fixtures are labelled separately."""
import importlib.util
import hashlib
import json
from pathlib import Path
from math import pi, sqrt

import numpy as np
import pytest

path = Path(__file__).parent / 'data_isapol/oracle/reconstruct_isa_basis.py'
spec = importlib.util.spec_from_file_location('reconstruct_isa_basis', path)
basis_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(basis_tool)
pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


def primitive(l, representation='S', exponent=.7):
    normalization = (2*exponent/pi)**.75 * sqrt((4*exponent)**l / basis_tool.double_factorial(2*l-1))
    return dict(representation=representation, labels=['X'], centres=np.zeros((1, 3)),
                charges=np.ones(1), exponents=np.array([exponent]),
                contractions=np.array([[normalization] * (l+1)]),
                shells=np.array([[1, l, 1, 1]]),
                nfunction=2*l+1 if representation == 'S' else (l+1)*(l+2)//2)


@pytest.mark.parametrize('l', range(5))
def test_regular_harmonics_addition_theorem(l):
    xyz = np.random.default_rng(551).normal(size=(31, 3))
    values = np.column_stack([basis_tool.polynomial_values(p, xyz)
                              for p in basis_tool.angular_polynomials(l, 'S')])
    np.testing.assert_allclose(np.sum(values**2, axis=1), np.sum(xyz**2, axis=1)**l,
                               atol=1e-12, rtol=3e-15)


@pytest.mark.parametrize('l', range(5))
@pytest.mark.parametrize('representation', ['C', 'S'])
def test_normalized_primitive_analytic_overlap(l, representation):
    b = primitive(l, representation)
    overlap = basis_tool.atomic_overlap(b)
    np.testing.assert_allclose(basis_tool.contraction_norms(b), [1.], atol=2e-15, rtol=0)
    np.testing.assert_allclose(np.diag(overlap), 1., atol=2e-14, rtol=0)
    if representation == 'S':
        np.testing.assert_allclose(overlap, np.eye(2*l+1), atol=2e-14, rtol=0)


def test_cartesian_mixed_normalization_and_order():
    p = np.array([[2., 3., 5.]])
    polynomials = basis_tool.angular_polynomials(2, 'C')
    values = [basis_tool.polynomial_values(poly, p)[0] for poly in polynomials]
    np.testing.assert_allclose(values, [4, 9, 25, sqrt(3)*6, sqrt(3)*10, sqrt(3)*15])
    spher = [basis_tool.polynomial_values(poly, p)[0] for poly in basis_tool.angular_polynomials(1, 'S')]
    np.testing.assert_allclose(spher, [2, 3, 5])


def test_effective_contraction_coefficients_are_not_renormalized():
    b = primitive(0)
    b['exponents'] = np.array([.4, 1.3])
    b['contractions'] = np.array([[2.], [-.3]])
    b['shells'] = np.array([[1, 0, 1, 2]])
    points = np.array([[0, 0, 0], [1, 2, 3]])
    r2 = np.sum(points**2, axis=1)
    want = 2*np.exp(-.4*r2) - .3*np.exp(-1.3*r2)
    np.testing.assert_allclose(basis_tool.evaluate_basis(b, points)[:, 0], want)
    metric = sum(ca*cb*(pi/(a+b_))**1.5 for a, ca in [(.4, 2), (1.3, -.3)]
                 for b_, cb in [(.4, 2), (1.3, -.3)])
    assert basis_tool.atomic_overlap(b)[0, 0] == pytest.approx(metric, rel=1e-14)


def test_density_neighbour_screening_and_translation():
    b = primitive(0)
    b['centres'] = np.array([[0., 0., 0.], [1., 0., 0.]])
    b['shells'] = np.array([[1, 0, 1, 1], [2, 0, 1, 1]])
    points = np.array([[1., 0., 0.], [2., 0., 0.]])
    full = basis_tool.evaluate_basis(b, points)
    screened = basis_tool.evaluate_basis(b, points, neighbours=[2])
    assert np.all(screened[:, 0] == 0)
    np.testing.assert_array_equal(screened[:, 1], full[:, 1])
    moved = dict(b, centres=b['centres']+[2, -3, 7])
    np.testing.assert_allclose(basis_tool.evaluate_basis(moved, points+[2, -3, 7]), full)
    with pytest.raises(ValueError, match='co-centred'):
        basis_tool.atomic_overlap(b)


#: A slice of the exported production CamCASP ISA atomic bases: for each atom the
#: tightest and most diffuse s shell plus the tightest shell of every higher
#: angular momentum present.  Evaluation and overlap are per-function and pairwise
#: quantities, so the values below are exactly the corresponding submatrices of
#: the full 97x109 (oxygen) and 97x49 (hydrogen) descriptor comparison -- this is
#: production parity on selected elements, not a reduced-accuracy substitute.  The
#: descriptors themselves and the complete replay live in the untracked
#: `agent_scratch/` tree; `test_isapol_basis_production.py` there re-derives every number
#: below from the fixture, so a regenerated fixture cannot leave these stale.
#:
#: Four points, one per regime of the 97-point radial grid:
#:   cusp    r ~ 1e-4  -- the tight s function is everything, and R_lm ~ r^l kills
#:                        the l=4 block down to 1.5e-15
#:   valence r ~ 0.9   -- the chemically relevant shell; the tight s has underflowed
#:                        to 1.8e-87 and the diffuse s carries the density
#:   angular           -- the point where the highest-l block is largest
#:   tail    r ~ 1.1e4 -- every function is exactly zero, signed zeros included
PRODUCTION_SLICE = {
    'O': dict(
        centre=[0.0, 0.0, 0.0],
        shells=[  # (l, exponent, contraction coefficient)
            (0, 256.0, 45.61315010271937),
            (0, 0.125, 0.14982786878830595),
            (1, 52.854386423, 203.1380982463263),
            (2, 15.907733132, 208.5566178742842),
            (3, 4.6873846701, 47.59357073611633),
            (4, 2.3270964878, 11.354682625596398),
        ],
        points=[  # (name, coordinate) -- one per regime of the 97-point grid
            ('cusp', [0.00011568570943800967, 0.0, 0.0]),
            ('valence', [-0.5873655370504208, -0.5873655370504208, -0.3241017618684983]),
            ('angular', [0.04579045170449098, 0.04579045170449098, -1.1319848601117335]),
            ('tail', [-9375.099047264035, -5405.963657496579, -2525.0127253571072]),
        ],
        axial=[  # m=0 component of every selected shell, point by point
            [45.612993828004214, 0.14982786853765923, 0.023500158386437736, 0.0, 0.0, 0.0],  # cusp
            [1.8501062583593413e-87, 0.13565396675100783, -6.71573021955797e-17, 0.00040083549196502774, -0.36709594828003606, 0.0],  # valence
            [5.354654156944445e-142, 0.1275858137150282, 2.8761711097952415e-29, 9.947218068906676e-10, 1.7449156430283208e-05, 0.0],  # angular
            [0.0, 0.0, -0.0, 0.0, -0.0, 0.0],  # tail
        ],
        block_maxabs=[  # largest |value| within each shell's m block
            [45.612993828004214, 0.14982786853765923, 0.023500158386437736, 2.4172075552597697e-06, 5.825422291089799e-11, 1.5039655151473554e-15],  # cusp
            [1.8501062583593413e-87, 0.13565396675100783, 6.71573021955797e-17, 0.00040083549196502774, 0.4961673095850148, 0.6285083562129506],  # valence
            [5.354654156944445e-142, 0.1275858137150282, 7.1101769696228155e-28, 3.503976343690515e-07, 0.16590685000485808, 0.9267995087914238],  # angular
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # tail
        ],
        s_functions=[0, 1],
        unweighted_s_diagonal=[0.9999999999999999, 0.9999999999999998],
        unweighted_ss=[(0, 1), 0.009283880039157184],
        w_eps=0.17,
        weighted_s_diagonal=[1.000498253664011, 5.524271728019906],
        weighted_ss=[(0, 1), 0.009293130815501514],
        nfunction=109, selected_nfunction=26,
    ),
    'H1': dict(
        centre=[-1.45365196, 0.0, -1.12168732],   # H1 is off the origin; points are absolute
        shells=[  # (l, exponent, contraction coefficient)
            (0, 32.0, 9.58898360245158),
            (0, 0.25, 0.25197943553838076),
            (1, 2.3725491635, 4.1971946787625845),
            (2, 1.8096373189, 4.64723525239862),
            (3, 1.8063060576, 5.56845816257023),
        ],
        points=[  # (name, coordinate) -- one per regime of the 97-point grid
            ('cusp', [-0.0008842431171817831, -0.0008842431171817831, -0.0015202240941318908]),
            ('valence', [0.5873655370504208, -0.5873655370504208, -0.3241017618684983]),
            ('angular', [-1.4154932517625483, 0.03815870823745165, -2.065007999275582]),
            ('tail', [-9375.099047264035, -5405.963657496579, -2525.0127253571072]),
        ],
        axial=[  # m=0 component of every selected shell, point by point
            [1.6317230992399595e-46, 0.10863788473477111, 0.0020777399918503843, -2.342472651065594e-05, -5.6465132388839646e-05],  # cusp
            [2.8393270312218714e-71, 0.06959022584345605, 4.2611224520979e-05, -0.0008699356913384737, -0.002881796984544358],  # valence
            [3.75512479409312e-12, 0.20157387906664048, 0.019260164585195836, 0.00232972099684953, 9.753014417415745e-05],  # angular
            [0.0, 0.0, -0.0, 0.0, -0.0],  # tail
        ],
        block_maxabs=[  # largest |value| within each shell's m block
            [1.6317230992399595e-46, 0.10863788473477111, 0.0020777399918503843, 0.029674653223720048, 0.058404422950991706],  # cusp
            [2.8393270312218714e-71, 0.06959022584345605, 4.2611224520979e-05, 0.0013862803200566673, 0.003013699212865847],  # valence
            [3.75512479409312e-12, 0.20157387906664048, 0.47613015163952943, 0.8206603297334842, 0.9273181236642408],  # angular
            [0.0, 0.0, 0.0, 0.0, 0.0],  # tail
        ],
        s_functions=[0, 1],
        w_eps=0.17,
        weighted_s_diagonal=[1.0039976454902542, 1.865022590595951],
        weighted_ss=[(0, 1), 0.07404759273810113],
        nfunction=49, selected_nfunction=17,
    ),
}


def sliced_basis(tag, role=None):
    """Build the selected production shells as a C++ basis. Indexing adapter only."""
    from psi4 import core
    want = PRODUCTION_SLICE[tag]
    shells = []
    for l, exponent, coefficient in want['shells']:
        s = core.IsaGaussianShell()
        s.centre, s.l = 0, l
        s.exponents = [exponent]
        s.coefficients = [coefficient]
        shells.append(s)
    basis = core.IsaExplicitBasis(role or core.IsaBasisRole.AtomAux,
                                  core.IsaBasisRepresentation.Spherical, [want['centre']], shells)
    blocks, n = [], 0
    for l, _, _ in want['shells']:
        blocks.append(list(range(n, n + 2 * l + 1)))
        n += 2 * l + 1
    return basis, want, blocks


@pytest.mark.parametrize('tag', ['O', 'H1'])
def test_production_slice_evaluates_to_the_exported_values(tag):
    """C++ evaluation of production shells, at selected points, against the export.

    `assert_array_equal` on the tail row is deliberate: the export records exact
    zeros there, so a tolerance would hide an underflow boundary moving.
    """
    basis, want, blocks = sliced_basis(tag)
    assert basis.nfunction == want['selected_nfunction']
    points = [point for _, point in want['points']]
    values = basis.evaluate(points).np
    assert values.shape == (len(points), want['selected_nfunction'])
    for row, (name, _) in enumerate(want['points']):
        axial = [values[row, b[0]] for b in blocks]
        maxabs = [np.abs(values[row, b]).max() for b in blocks]
        if name == 'tail':
            np.testing.assert_array_equal(np.abs(axial), np.abs(want['axial'][row]))
            np.testing.assert_array_equal(maxabs, want['block_maxabs'][row])
        else:
            np.testing.assert_allclose(axial, want['axial'][row], rtol=2e-13, atol=0)
            np.testing.assert_allclose(maxabs, want['block_maxabs'][row], rtol=2e-13, atol=0)
    # The independent polynomial oracle agrees on the same selection.
    np.testing.assert_allclose(values, basis_tool.evaluate_basis(sliced_descriptor(tag), np.array(points)),
                               rtol=3e-13, atol=2e-14)


@pytest.mark.parametrize('tag', ['O', 'H1'])
def test_production_slice_weighted_metric_touches_only_the_s_block(tag):
    """W-Eps is s-block-only, and that is visible in the exported metric itself.

    The unweighted metric of normalized primitives has a unit diagonal; switching
    on the production `w_eps` with `s_block_only` lifts the two s diagonals -- the
    diffuse oxygen s by a factor of 5.5 -- and leaves all 24 (or 15) higher-l
    diagonals at one. This is the scope claim from PRODUCTION_CHECKPOINT.md,
    checked against the production numbers rather than asserted in prose.
    """
    basis, want, blocks = sliced_basis(tag)
    s0, s1 = want['s_functions']
    higher = [i for b in blocks[2:] for i in b]

    plain = basis.overlap().np
    np.testing.assert_allclose(np.diag(plain), 1., atol=2e-14)
    np.testing.assert_allclose(plain, basis_tool.atomic_overlap(sliced_descriptor(tag)),
                               rtol=2e-13, atol=2e-14)
    if 'unweighted_s_diagonal' in want:
        exported = basis.overlap(0., True).np
        np.testing.assert_allclose([exported[s0, s0], exported[s1, s1]],
                                   want['unweighted_s_diagonal'], rtol=2e-13, atol=0)
        (i, j), value = want['unweighted_ss']
        np.testing.assert_allclose(exported[i, j], value, rtol=2e-13, atol=0)

    weighted = basis.overlap(want['w_eps'], True).np
    np.testing.assert_allclose([weighted[s0, s0], weighted[s1, s1]],
                               want['weighted_s_diagonal'], rtol=2e-13, atol=0)
    (i, j), value = want['weighted_ss']
    np.testing.assert_allclose(weighted[i, j], value, rtol=2e-13, atol=0)
    assert weighted[s1, s1] > 1.5 * weighted[s0, s0]   # nonvacuous weighting
    np.testing.assert_allclose(np.diag(weighted)[higher], 1., atol=2e-14)
    np.testing.assert_allclose(weighted[np.ix_(higher, higher)], plain[np.ix_(higher, higher)],
                               rtol=0, atol=2e-14)
    # The oracle reproduces the weighted metric from the same shells.
    np.testing.assert_allclose(weighted, basis_tool.atomic_overlap(sliced_descriptor(tag), want['w_eps']),
                               rtol=2e-13, atol=2e-14)


def sliced_descriptor(tag):
    """The same selected shells as a descriptor for the independent oracle."""
    want = PRODUCTION_SLICE[tag]
    lmax = max(l for l, _, _ in want['shells'])
    contractions = np.zeros((len(want['shells']), lmax + 1))
    for k, (l, _, coefficient) in enumerate(want['shells']):
        contractions[k, l] = coefficient
    return dict(representation='S', labels=[tag], centres=np.array([want['centre']]), charges=np.ones(1),
                exponents=np.array([e for _, e, _ in want['shells']]), contractions=contractions,
                shells=np.array([[1, l, k + 1, k + 1] for k, (l, _, _) in enumerate(want['shells'])]),
                nfunction=want['selected_nfunction'])


def test_bounded_fixture_roundtrip(tmp_path):
    # Synthetic plumbing only; independent normalization checks live above.
    b = primitive(0)
    points = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    values = basis_tool.evaluate_basis(b, points)
    d = dict(atomic_basis=b, density_basis=b, shape_basis=b,
             density_coefficients=np.array([2.]), density_neighbours=np.array([1]),
             shape_map=np.array([1]), shape_old=np.array([1.]), shape_new_raw=np.array([.8]))
    c = dict(schema_version=2, descriptors=d, options=dict(w_eps=0., s_block_only=True),
             tail_flags=[0, 0], overlap=np.ones((1, 1)), coefficients=np.array([.8]),
             atom=1, atom_label='synthetic', points=points, basis_values=values,
             density=2*values[:, 0], shape=values[:, 0])
    source = tmp_path / 'source.dat'
    source.write_text('synthetic source')
    fixture = tmp_path / 'fixture.json'
    basis_tool.write_fixture(fixture, c, [0, 2], source)
    restored = basis_tool.load_fixture(fixture)
    np.testing.assert_array_equal(restored['points'], points[[0, 2]])
    report = basis_tool.audit(restored)
    assert all(e['max_scaled'] < 1e-14 for e in report['errors'].values())
    with pytest.raises(ValueError, match='513'):
        basis_tool.write_fixture(fixture, c, [0]*514, source)
    with pytest.raises(ValueError, match='positive'):
        basis_tool.audit(c, sample_count=0)


def test_weighted_overlap_integrability_and_scope():
    s = primitive(0, exponent=.2)
    assert basis_tool.atomic_overlap(s, .1)[0, 0] == pytest.approx((.4/.3)**1.5)
    p = primitive(1, exponent=.2)
    np.testing.assert_allclose(basis_tool.atomic_overlap(p, .1), np.eye(3), atol=2e-15)
    np.testing.assert_allclose(basis_tool.atomic_overlap(p, .1, False), np.eye(3)*(.4/.3)**2.5, atol=3e-15)
    with pytest.raises(ValueError, match='Nonintegrable'):
        basis_tool.atomic_overlap(s, .4)
    with pytest.raises(ValueError, match='ranks'):
        basis_tool.angular_polynomials(5, 'S')
    with pytest.raises(ValueError, match='finite'):
        basis_tool.evaluate_basis(s, [[np.nan, 0, 0]])
    with pytest.raises(ValueError, match='v2'):
        basis_tool.audit({'descriptors': None})


def cpp_basis(b, role=None):
    """Descriptor indexing adapter only; all numerical evaluation stays in C++."""
    from psi4 import core
    shells = []
    for site, l, first, last in b['shells']:
        s = core.IsaGaussianShell()
        s.centre, s.l = int(site)-1, int(l)
        s.exponents = b['exponents'][first-1:last].tolist()
        s.coefficients = b['contractions'][first-1:last, l].tolist()
        shells.append(s)
    rep = (core.IsaBasisRepresentation.Cartesian if b['representation'] == 'C'
           else core.IsaBasisRepresentation.Spherical)
    return core.IsaExplicitBasis(role or core.IsaBasisRole.AtomAux, rep, b['centres'].tolist(), shells)


@pytest.mark.parametrize('l', range(5))
@pytest.mark.parametrize('representation', ['C', 'S'])
def test_cpp_contracted_shells_against_independent_oracle(l, representation):
    b = primitive(l, representation)
    b['exponents'] = np.array([.4, 1.3])
    b['contractions'] = np.array([[2.]*(l+1), [-.3]*(l+1)])
    b['shells'] = np.array([[1, l, 1, 2]])
    points = np.vstack([np.zeros((1, 3)), np.eye(3), np.random.default_rng(973).normal(size=(35, 3))])
    np.testing.assert_allclose(cpp_basis(b).evaluate(points.tolist()).np,
                               basis_tool.evaluate_basis(b, points), rtol=2e-13, atol=2e-14)


def test_cpp_density_screening_roles_translation_and_ownership():
    from psi4 import core
    b = primitive(0)
    b['centres'] = np.array([[0., 0., 0.], [1., 0., 0.]])
    b['shells'] = np.array([[1, 0, 1, 1], [2, 0, 1, 1]])
    p = np.array([[0., 0., 0.], [1., 2., 3.]])
    mol = cpp_basis(b, core.IsaBasisRole.MolecularAux)
    coefficients = [-2., 3.]
    rho = core.IsaFixedDensity(mol, coefficients)
    coefficients[0] = 100
    full = mol.evaluate(p.tolist()).np.copy()
    np.testing.assert_allclose(rho.evaluate(p.tolist(), [0, 1]), full @ [-2., 3.])
    assert rho.evaluate(p.tolist(), [0])[0] < 0  # no positivity clipping
    np.testing.assert_allclose(rho.evaluate(p.tolist(), [1]), 3*full[:, 1])
    np.testing.assert_array_equal(rho.evaluate(p.tolist(), []), [0, 0])
    np.testing.assert_array_equal(mol.evaluate_screened(p.tolist(), [1]).np[:, 0], [0, 0])
    moved = cpp_basis(dict(b, centres=b['centres']+[2, -3, 7]), core.IsaBasisRole.MolecularAux)
    np.testing.assert_allclose(moved.evaluate((p+[2, -3, 7]).tolist()).np, full)
    result = mol.evaluate(p.tolist())
    result.np[:] = 99
    np.testing.assert_array_equal(mol.evaluate(p.tolist()).np, full)
    with pytest.raises(ValueError, match='molecular AUX'):
        core.IsaFixedDensity(cpp_basis(b), [-2., 3.])
    for bad in ([0, 0], [-1], [2]):
        with pytest.raises(ValueError, match='[Nn]eighbour'):
            rho.evaluate(p.tolist(), bad)
    for bad in ([1.], [1., np.nan]):
        with pytest.raises(ValueError, match='coefficient'):
            core.IsaFixedDensity(mol, bad)
    with pytest.raises(ValueError, match='finite'):
        mol.evaluate([[np.nan, 0, 0]])


@pytest.mark.parametrize('field,value,match', [
    ('centre', -1, 'centre'), ('centre', 1, 'centre'), ('l', -1, 'S through G'),
    ('l', 5, 'S through G'), ('exponents', [], 'dimensions'),
    ('exponents', [0.], 'positive'), ('exponents', [np.inf], 'positive'),
    ('coefficients', [], 'dimensions'), ('coefficients', [np.nan], 'finite')])
def test_cpp_invalid_shell(field, value, match):
    from psi4 import core
    s = core.IsaGaussianShell()
    s.exponents, s.coefficients = [.7], [1.]
    setattr(s, field, value)
    with pytest.raises(ValueError, match=match):
        core.IsaExplicitBasis(core.IsaBasisRole.AtomAux, core.IsaBasisRepresentation.Spherical, [[0, 0, 0]], [s])


def test_cpp_basis_snapshot_and_shape_validation():
    from psi4 import core
    s = core.IsaGaussianShell()
    s.exponents, s.coefficients = [.7], [2.]
    centres = [[0., 0., 0.]]
    b = core.IsaExplicitBasis(core.IsaBasisRole.Shape, core.IsaBasisRepresentation.Spherical, centres, [s])
    s.coefficients = [99.]
    centres[0][0] = 4.
    assert b.evaluate([[0, 0, 0]]).np[0, 0] == 2.
    s.l = 1
    with pytest.raises(ValueError, match='s shells'):
        core.IsaExplicitBasis(core.IsaBasisRole.Shape, core.IsaBasisRepresentation.Spherical, [[0, 0, 0]], [s])
    for centres, shells, match in [([], [s], 'centres'), ([[0, 0, 0]], [], 'shells'),
                                    ([[np.inf, 0, 0]], [s], 'finite')]:
        with pytest.raises(ValueError, match=match):
            core.IsaExplicitBasis(core.IsaBasisRole.AtomAux, core.IsaBasisRepresentation.Spherical, centres, shells)


@pytest.mark.parametrize('representation', ['C', 'S'])
@pytest.mark.parametrize('w_eps,s_only', [(0., True), (.17, True), (.17, False)])
def test_cpp_mixed_contracted_analytic_overlap(representation, w_eps, s_only):
    # Signed, deliberately non-normalized contractions; all l/l' blocks tested.
    b = dict(representation=representation, centres=np.array([[1., -2., 3.]]),
             exponents=np.tile([.4, 1.3], 5), contractions=np.tile([[2.]*5, [-.3]*5], (5, 1)),
             shells=np.array([[1, l, 2*l+1, 2*l+2] for l in range(5)]))
    basis = cpp_basis(b)
    actual = basis.overlap(w_eps, s_only).np.copy()
    expected = basis_tool.atomic_overlap(b, w_eps, s_only)
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-12)
    np.testing.assert_array_equal(actual, actual.T)
    assert np.linalg.eigvalsh(actual).min() > 0
    # Translation and output ownership do not alter an analytic co-centred metric.
    moved = cpp_basis(dict(b, centres=b['centres']+[2, 3, 4]))
    np.testing.assert_array_equal(moved.overlap(w_eps, s_only).np, actual)
    output = basis.overlap(w_eps, s_only)
    output.np[:] = 0
    np.testing.assert_array_equal(basis.overlap(w_eps, s_only).np, actual)


@pytest.mark.parametrize('l', range(5))
@pytest.mark.parametrize('representation', ['C', 'S'])
def test_cpp_normalized_metric_and_weight_scaling(l, representation):
    b = cpp_basis(primitive(l, representation, exponent=.2))
    plain = b.overlap().np.copy()
    np.testing.assert_allclose(np.diag(plain), 1., atol=3e-15)
    if representation == 'S':
        np.testing.assert_allclose(plain, np.eye(2*l+1), atol=3e-15)
    np.testing.assert_allclose(b.overlap(.1, False).np, plain*(.4/.3)**(l+1.5), atol=5e-14)
    np.testing.assert_allclose(b.overlap(.1, True).np, plain*((.4/.3)**1.5 if l == 0 else 1), atol=5e-14)


@pytest.mark.parametrize('w_eps', [-1., np.nan, np.inf, .4, .5])
def test_cpp_overlap_rejects_invalid_weight(w_eps):
    with pytest.raises(ValueError, match='W-Eps|Nonintegrable'):
        cpp_basis(primitive(0, exponent=.2)).overlap(w_eps)


def test_cpp_overlap_scope_and_integrability():
    from psi4 import core
    b = primitive(0)
    with pytest.raises(ValueError, match='molecular AUX'):
        cpp_basis(b, core.IsaBasisRole.MolecularAux).overlap()
    b['centres'] = np.array([[0., 0., 0.], [1., 0., 0.]])
    b['shells'] = np.array([[1, 0, 1, 1], [2, 0, 1, 1]])
    with pytest.raises(ValueError, match='co-centred'):
        cpp_basis(b).overlap()
    # Different site indices at the same location are valid.
    b['centres'][1] = b['centres'][0]
    np.testing.assert_allclose(cpp_basis(b).overlap().np, np.ones((2, 2)), atol=2e-15)
    p = cpp_basis(primitive(1, exponent=.2))
    np.testing.assert_allclose(p.overlap(.4, True).np, np.eye(3), atol=2e-15)
    with pytest.raises(ValueError, match='Nonintegrable'):
        p.overlap(.4, False)
    # Divergent zero-coefficient primitives still fail; no hidden dropping of terms.
    zero = primitive(0, exponent=.2)
    zero['contractions'][:] = 0
    with pytest.raises(ValueError, match='Nonintegrable'):
        cpp_basis(zero).overlap(.4)
    huge = primitive(0)
    huge['contractions'][:] = 1e308
    with pytest.raises(ValueError, match='Nonfinite'):
        cpp_basis(huge).overlap()


def mapped_test_bases():
    """s shells deliberately follow non-s blocks; shape is an explicit permutation."""
    from psi4 import core
    def shell(l, alpha):
        s = core.IsaGaussianShell()
        s.l, s.exponents, s.coefficients = l, [alpha], [2.]
        return s
    shells = [shell(1, .7), shell(0, .4), shell(2, .7), shell(0, 1.3)]
    atomic = core.IsaExplicitBasis(core.IsaBasisRole.AtomAux, core.IsaBasisRepresentation.Cartesian,
                                   [[0, 0, 0]], shells)
    shape = core.IsaExplicitBasis(core.IsaBasisRole.Shape, core.IsaBasisRepresentation.Spherical,
                                  [[0, 0, 0]], [shells[3], shells[1]])
    return atomic, shape, shells


def test_cpp_shape_map_permutation_subset_and_ownership():
    from psi4 import core
    atomic, shape, shells = mapped_test_bases()
    indices = [3, 1]
    mapping = core.IsaShapeMap(atomic, shape, indices)
    assert mapping.function_indices == [10, 3]  # not shell indices
    indices[0] = 0
    shells[3].coefficients = [99.]
    assert mapping.project(list(range(11))) == [10., 3.]
    returned = mapping.function_indices
    returned[0] = 0
    assert mapping.function_indices == [10, 3]
    one = core.IsaExplicitBasis(core.IsaBasisRole.Shape, core.IsaBasisRepresentation.Spherical,
                                [[0, 0, 0]], [shells[1]])
    assert core.IsaShapeMap(atomic, one, [1]).project(list(range(11))) == [3.]
    with pytest.raises(ValueError, match='dimension'):
        mapping.project([1.])
    values = list(range(11))
    values[0] = np.nan  # validate even unselected non-s coefficients
    with pytest.raises(ValueError, match='finite'):
        mapping.project(values)


@pytest.mark.parametrize('indices,match', [([3], 'dimension'), ([3, 1, 0], 'dimension'),
    ([-1, 1], 'range'), ([4, 1], 'range'), ([3, 3], 'Duplicate'), ([0, 1], 's shell'),
    ([1, 3], 'matching')])
def test_cpp_shape_map_rejects_bad_indices(indices, match):
    from psi4 import core
    atomic, shape, _ = mapped_test_bases()
    with pytest.raises(ValueError, match=match):
        core.IsaShapeMap(atomic, shape, indices)


def test_cpp_shape_map_rejects_role_and_descriptor_mismatch():
    from psi4 import core
    atomic, shape, shells = mapped_test_bases()
    with pytest.raises(ValueError, match='roles'):
        core.IsaShapeMap(shape, atomic, [3, 1])
    for centre, coefficient in [([1, 0, 0], 2.), ([0, 0, 0], 2.00000000001)]:
        shells[1].coefficients = [coefficient]
        mismatch = core.IsaExplicitBasis(core.IsaBasisRole.Shape, core.IsaBasisRepresentation.Spherical,
                                         [centre], [shells[1]])
        with pytest.raises(ValueError, match='matching'):
            core.IsaShapeMap(atomic, mismatch, [1])


def fit_provider_case():
    from psi4 import core
    b = primitive(1)
    # p followed by s checks noncontiguous metadata/normalization roles.
    b['shells'] = np.array([[1, 1, 1, 1], [1, 0, 1, 1]])
    b['centres'] = np.array([[1., -2., 3.]])
    b['nfunction'] = 4
    atomic = cpp_basis(b)
    mol = cpp_basis(b, core.IsaBasisRole.MolecularAux)
    density = core.IsaFixedDensity(mol, [.1, -.3, .2, 2.])
    samples = core.IsaAFitSamples()
    samples.points = (np.random.default_rng(711).normal(size=(23, 3))+b['centres'][0]).tolist()
    samples.weights = [.05]*23
    samples.shape = [-.2]+[.7]*22  # signed supplied shape is not re-evaluated/clipped
    samples.shape_sum = [1., 0.]+[1.3]*21
    samples.previous = [.2, -.1, .3, -.4]
    samples.density_sites = [0]
    return b, atomic, density, samples


def test_cpp_fit_provider_assembly_and_direct_solve():
    from psi4 import core
    b, atomic, density, samples = fit_provider_case()
    provider = core.IsaAFitProvider(atomic, density)
    options = core.IsaAFitOptions()
    options.w_eps, options.damping = .17, .1
    options.positive_lambda, options.positive_max_alpha = .001, 1.
    data = provider.assemble(samples, options)
    expected_values = basis_tool.evaluate_basis(b, samples.points)
    np.testing.assert_allclose(data.basis_values.np, expected_values, atol=1e-15)
    np.testing.assert_allclose(data.density, expected_values @ [.1, -.3, .2, 2.], atol=1e-15)
    np.testing.assert_allclose(data.overlap.np, basis_tool.atomic_overlap(b, .17), atol=3e-15)
    np.testing.assert_allclose(data.radius_squared, np.sum((np.asarray(samples.points)-b['centres'][0])**2, axis=1))
    assert data.angular_momenta == [1, 1, 1, 0]
    assert data.exponents == [.7]*4
    assert data.shape == samples.shape and data.shape_sum == samples.shape_sum
    direct = provider.fit(samples, options)
    frozen = core.isa_a_fit_step(data, options)
    for field in ('metric', 'rhs', 'coefficients'):
        np.testing.assert_array_equal(getattr(direct, field).np, getattr(frozen, field).np)
    assert direct.excluded_points == 1
    assert direct.relative_residual < 1e-14
    np.testing.assert_allclose(direct.coefficients.np, np.linalg.solve(direct.metric.np, direct.rhs.np), atol=2e-15)
    # Returned mutable data never aliases either samples or the provider.
    data.basis_values.np[:] = 99
    data.shape = [99]*23
    np.testing.assert_allclose(provider.assemble(samples, options).basis_values.np, expected_values, atol=1e-15)
    samples.density_sites = []
    assert provider.assemble(samples, options).density == [0.]*23


@pytest.mark.parametrize('field,value,match', [
    ('points', [], 'nonempty'), ('weights', [], 'dimension'), ('previous', [1.], 'dimension'),
    ('shape', [np.nan]*23, 'finite'), ('shape_sum', [np.inf]*23, 'finite'),
    ('points', [[np.nan, 0, 0]]*23, 'finite'), ('points', [[1e308, 0, 0]]*23, 'squared'),
    ('density_sites', [-1], 'range'), ('density_sites', [0, 0], 'Duplicate')])
def test_cpp_fit_provider_rejects_samples(field, value, match):
    from psi4 import core
    _, atomic, density, samples = fit_provider_case()
    setattr(samples, field, value)
    with pytest.raises(ValueError, match=match):
        core.IsaAFitProvider(atomic, density).assemble(samples)


@pytest.mark.parametrize('field', ['w_eps', 'damping', 'positive_lambda', 'positive_max_alpha', 'density_cutoff'])
def test_cpp_fit_provider_rejects_options(field):
    from psi4 import core
    _, atomic, density, samples = fit_provider_case()
    o = core.IsaAFitOptions()
    setattr(o, field, -1.)
    with pytest.raises(ValueError, match='options'):
        core.IsaAFitProvider(atomic, density).assemble(samples, o)


def test_cpp_fit_provider_rejects_contracts():
    from psi4 import core
    b, _, density, _ = fit_provider_case()
    with pytest.raises(ValueError, match='AtomAux'):
        core.IsaAFitProvider(cpp_basis(b, core.IsaBasisRole.MolecularAux), density)
    b['exponents'] = np.array([.7, .4])
    b['contractions'] = np.ones((2, 2))
    b['shells'] = np.array([[1, 1, 1, 2]])
    with pytest.raises(ValueError, match='primitive'):
        core.IsaAFitProvider(cpp_basis(b), density)
    b['centres'] = np.array([[0., 0., 0.], [1., 0., 0.]])
    b['shells'] = np.array([[1, 1, 1, 1], [2, 0, 2, 2]])
    with pytest.raises(ValueError, match='co-centred'):
        core.IsaAFitProvider(cpp_basis(b), density)
