# Psi4: Copyright (c) 2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Expected basis for a traced NON-ISA properties protocol, not a preset.

Self-contained O/H numeric records only. See NATIVE_REFERENCE_BASIS.md and
isapol_reference_basis.NOTICE for source/data attribution and license notices.
No runtime file reads, SCF, options, registry, or historical export inference.
Actual orbital MAIN must still come from adapt_main(wavefunction).
"""
from dataclasses import dataclass
from math import pi, sqrt, prod, fsum


def _odd_factorial(l):
    return prod(range(1, 2*l, 2))


@dataclass(frozen=True)
class ReferenceShell:
    """One contraction, raw normalized-primitive coefficients retained separately.

    Effective coefficients multiply GAMINT Cartesian polynomials (or regular
    spherical harmonics), not unit-normalized primitives. The existing native
    explicit basis API supplies the Cartesian component GAMINT scaling itself.
    """
    l: int
    exponents: tuple
    source_coefficients: tuple

    @property
    def effective_coefficients(self):
        d = _odd_factorial(self.l)
        pre = tuple(c * 2**self.l * (2/pi)**.75 * a**(self.l/2+.75) / sqrt(d)
                    for a, c in zip(self.exponents, self.source_coefficients))
        k = fsum(cp*cq*d*pi**1.5 / (2**self.l*(ap+aq)**(self.l+1.5))
                 for ap, cp in zip(self.exponents, pre)
                 for aq, cq in zip(self.exponents, pre))
        return tuple(c / sqrt(k) for c in pre)

    @property
    def spherical_functions(self):
        return 2*self.l + 1

    @property
    def cartesian_functions(self):
        return (self.l+1)*(self.l+2)//2


@dataclass(frozen=True)
class ElementReference:
    symbol: str
    main_shells: tuple
    aux_shells: tuple


@dataclass(frozen=True)
class ReferenceProtocolManifest:
    name: str
    reference_revision: str
    inspected_revision: str
    source_sha256: tuple
    elements: tuple
    geometry_bohr: tuple = (("O", (0., 0., 0.)),
                            ("H1", (-1.4536519600, 0., -1.1216873200)),
                            ("H2", (1.4536519600, 0., -1.1216873200)))
    charge: int = 0
    multiplicity: int = 1
    ip_ev: float = 12.62063
    ip_hartree_reference: float = 0.46380004527520863
    homo_hartree_reference: float = -0.3989
    fixed_grac_shift_hartree_reference: float = 0.06490004527520865
    grac_provenance: str = "explicit accepted fixed reference value; not recomputed from a wavefunction"
    main_harmonics: str = "spherical; expected aug-cc-pVTZ contractions; actual MAIN remains adapt_main(wfn)"
    native_harmonic_convention: str = (
        "native explicit spherical uses real regular DALTON harmonics, no Condon-Shortley phase; "
        "p=x,y,z; l>=2 descending sine, m=0, ascending cosine; "
        "not an assertion about historical Psi4 component order")
    native_main_comparison: str = (
        "Native Psi4 H contractions match as a multiset; O omits 1.752/.2384 terms "
        "from both contracted s shells and .7156/.214 from contracted p, all "
        "separately present unit shells. Same span, not literal contractions. "
        "Expected 62 primitive entries versus native 56; neither verifies historical export.")
    aux_harmonics: str = "Cartesian GAMINT; ordinary aug-cc-pVTZ RI; source order; Limit G"
    atomaux_mode: str = "same_as_aux_fallback"
    isa_algorithm: None = None
    shape: str = "shape_not_applicable"
    response_route: str = "ALDA+CHF constrained NN; eta=0 then 0.0005; lambda=1000"
    localization: str = "LW L2/H1 then PFIT wt3 coefficient .001; NoRefine=False"
    historical_grid_request: tuple = (("angular", 200), ("radial", 100))
    known_modern_grid_case: tuple = (
        ("provenance", "source-derived current IsaGrid, NOT historical/SCF grid"),
        ("atoms", 3), ("radial_points", 99), ("angular_points", 590),
        ("rows", 173460), ("formula", "3*(99-1)*590; supported Lebedev; no weight pruning"))
    provenance_caveat: str = (
        "Reference-file commit is not proof of generating executable/environment. "
        "CLT/potential/O-H data bytes agree at reference and inspected revisions; "
        "current generator adds only three Psi4 charge/multiplicity/comment lines. "
        "Archived 2016 DALTON output and modern ISA templates are different protocols.")
    missing_artifacts: tuple = (
        "generating H2O.cks and overrides", "historical H2O-A.basis SCF export",
        "SCF input/output/fchk, version and generating executable revision",
        "included run-specific basis snapshots", "actual response grid/propagator metadata",
        "unlocalized per-frequency polarizabilities", "actual point/response .p2p files",
        "H2O.pdef and per-frequency PFIT inputs/refined polarizabilities")

    @property
    def historical_scf_export_verified(self):
        # Deliberately not an init/replace flag that can accidentally certify data.
        return False

    @property
    def expected_main_nfunction(self):
        return sum((1 if e.symbol == "O" else 2) *
                   sum(s.spherical_functions for s in e.main_shells) for e in self.elements)

    @property
    def expected_aux_nfunction(self):
        return sum((1 if e.symbol == "O" else 2) *
                   sum(s.cartesian_functions for s in e.aux_shells) for e in self.elements)


def h2o_props_psi4_777f904_manifest():
    """Immutable expected reference metadata; NOT callable as PartitionRecipe."""
    return _MANIFEST


def build_expected_reference_aux():
    """Build ONLY the expected reference Cartesian molecular RI AUX (246).

    No wavefunction, MAIN substitution, AtomAux/shape/controller construction,
    partition, NN response, or matched-property recipe is provided. This uses
    the fixed reference geometry, not the geometry of any caller wavefunction.
    Native IsaExplicitBasis applies GAMINT component scaling; coefficients here
    are already primitive/contraction normalized, not source 1.0 coefficients.
    """
    from psi4 import core
    m = h2o_props_psi4_777f904_manifest()
    by_symbol = {e.symbol: e for e in m.elements}
    shells = []
    for centre, (label, _) in enumerate(m.geometry_bohr):
        for record in by_symbol[label[0]].aux_shells:
            shell = core.IsaGaussianShell()
            shell.centre, shell.l = centre, record.l
            shell.exponents = list(record.exponents)
            shell.coefficients = list(record.effective_coefficients)
            shells.append(shell)
    return core.IsaExplicitBasis(core.IsaBasisRole.MolecularAux,
        core.IsaBasisRepresentation.Cartesian, [list(xyz) for _, xyz in m.geometry_bohr], shells)


# Literal records below: tuple (l, exponents, source coefficients), one shell
# per contraction. In particular, diffuse AUX shells follow ALL tight ranks.
_O_MAIN = (
    ReferenceShell(0, (15330.0, 2299.0, 522.4, 147.3, 47.55, 16.76, 6.207, 1.752, 0.6882, 0.2384), (0.000508, 0.003929, 0.020243, 0.079181, 0.230687, 0.433118, 0.35026, 0.042728, -0.008154, 0.002381)),
    ReferenceShell(0, (15330.0, 2299.0, 522.4, 147.3, 47.55, 16.76, 6.207, 1.752, 0.6882, 0.2384), (-0.000115, -0.000895, -0.004636, -0.018724, -0.058463, -0.136463, -0.17574, 0.160934, 0.603418, 0.378765)),
    ReferenceShell(0, (1.752,), (1.0,)),
    ReferenceShell(0, (0.2384,), (1.0,)),
    ReferenceShell(0, (0.07376,), (1.0,)),
    ReferenceShell(1, (34.46, 7.749, 2.28, 0.7156, 0.214), (0.015928, 0.09974, 0.310492, 0.491026, 0.336337)),
    ReferenceShell(1, (0.7156,), (1.0,)),
    ReferenceShell(1, (0.214,), (1.0,)),
    ReferenceShell(1, (0.05974,), (1.0,)),
    ReferenceShell(2, (2.314,), (1.0,)),
    ReferenceShell(2, (0.645,), (1.0,)),
    ReferenceShell(2, (0.214,), (1.0,)),
    ReferenceShell(3, (1.428,), (1.0,)),
    ReferenceShell(3, (0.5,), (1.0,)),
)

_O_AUX = (
    ReferenceShell(0, (366.54531503,), (1.0,)),
    ReferenceShell(0, (76.693314858,), (1.0,)),
    ReferenceShell(0, (24.573835091,), (1.0,)),
    ReferenceShell(0, (8.4165585072,), (1.0,)),
    ReferenceShell(0, (3.0202080356,), (1.0,)),
    ReferenceShell(0, (1.3331563617,), (1.0,)),
    ReferenceShell(0, (0.81250658727,), (1.0,)),
    ReferenceShell(0, (0.36505253662,), (1.0,)),
    ReferenceShell(1, (52.854386423,), (1.0,)),
    ReferenceShell(1, (15.021795824,), (1.0,)),
    ReferenceShell(1, (3.9137719482,), (1.0,)),
    ReferenceShell(1, (2.1678840611,), (1.0,)),
    ReferenceShell(1, (0.97419243984,), (1.0,)),
    ReferenceShell(1, (0.51845615412,), (1.0,)),
    ReferenceShell(2, (15.907733132,), (1.0,)),
    ReferenceShell(2, (5.3799329215,), (1.0,)),
    ReferenceShell(2, (2.9552623921,), (1.0,)),
    ReferenceShell(2, (1.1624458997,), (1.0,)),
    ReferenceShell(2, (0.50353399421,), (1.0,)),
    ReferenceShell(3, (4.6873846701,), (1.0,)),
    ReferenceShell(3, (2.1512789489,), (1.0,)),
    ReferenceShell(3, (1.0894068169,), (1.0,)),
    ReferenceShell(4, (2.3270964878,), (1.0,)),
    ReferenceShell(0, (0.11211820118,), (1.0,)),
    ReferenceShell(1, (0.21354459926,), (1.0,)),
    ReferenceShell(2, (0.16734574676,), (1.0,)),
    ReferenceShell(3, (0.40383704543,), (1.0,)),
    ReferenceShell(4, (0.78655757162,), (1.0,)),
)

_H_MAIN = (
    ReferenceShell(0, (33.87, 5.095, 1.159, 0.3258, 0.1027), (0.006068, 0.045308, 0.202822, 0.503903, 0.383421)),
    ReferenceShell(0, (0.3258,), (1.0,)),
    ReferenceShell(0, (0.1027,), (1.0,)),
    ReferenceShell(0, (0.02526,), (1.0,)),
    ReferenceShell(1, (1.407,), (1.0,)),
    ReferenceShell(1, (0.388,), (1.0,)),
    ReferenceShell(1, (0.102,), (1.0,)),
    ReferenceShell(2, (1.057,), (1.0,)),
    ReferenceShell(2, (0.247,), (1.0,)),
)

_H_AUX = (
    ReferenceShell(0, (8.512827591,), (1.0,)),
    ReferenceShell(0, (1.8730891166,), (1.0,)),
    ReferenceShell(0, (0.52618426941,), (1.0,)),
    ReferenceShell(0, (0.28973989936,), (1.0,)),
    ReferenceShell(1, (2.3725491635,), (1.0,)),
    ReferenceShell(1, (1.1804084468,), (1.0,)),
    ReferenceShell(1, (0.60382204241,), (1.0,)),
    ReferenceShell(2, (1.8096373189,), (1.0,)),
    ReferenceShell(2, (1.1439726055,), (1.0,)),
    ReferenceShell(3, (1.8063060576,), (1.0,)),
    ReferenceShell(0, (0.12719436063,), (1.0,)),
    ReferenceShell(1, (0.23551289521,), (1.0,)),
    ReferenceShell(2, (0.43665405053,), (1.0,)),
    ReferenceShell(3, (0.25297787763,), (1.0,)),
)

_MANIFEST = ReferenceProtocolManifest(
    name="tracedH2O_props_psi4_777f904",
    reference_revision='777f90498868d33847de525612628b2dc8448523',
    inspected_revision="63b16a22b9bae597fe81ecdb8b8d91c21868c814",
    source_sha256=(
        ('tests/H2O_props/psi4/H2O-avtz.clt', '8c09be634edd593c5966522fac73d1a59a1fbc7fbb748e67a5539dfab5f63a11'),
        ('tests/H2O_props/psi4/check/L2H1/H2O_ref_wt3_L2_Cn.pot', '25ffd1d2ac0e4202fb194cc7786df13a9d4953fa020a81b998400b375b5f2fac'),
        ('bin/camcasp.py', '4e1e7c068b48278b0d76a6a4c0dda318539316f2e9db6dd8159f1df6feef4205'),
        ('bin/runcamcasp.py', 'db24d2a382107a1863bdb7e4e3aad0d9a350ab4b4f1a323ee7d89d96eda9c359'),
        ('tests/test_H2O_props.py', 'c82d212d83fd70a0e2add53960fe71a3caf898410d1f94dd081aae22fe936d6a'),
        ('src/tools/cluster_file_interface.F90', 'dc4bcc76e9e9c16e1677b93f0c783d0387152391514c974df064761e5f93cee5'),
        ('basis/gamess_us/aug-cc-pVTZ/O', '67e3395ce4ab7053a4ff6f86d508bb74e97403f7194ac8828c599974265e3acc'),
        ('basis/gamess_us/aug-cc-pVTZ/H', '8590af2f125999d2888ac3599a42e8735654b4e93fe4ecc8b18878dd672f80d0'),
        ('basis/psi4/aug-cc-pvtz.gbs', '4e044373a426868d570392a535eed0f439c84ab796e901cd6e8f72e0ce8fd26f'),
        ('basis/auxiliary/aug-cc-pVTZ/O', 'c730aa433ce6c0d604bbc2616fb9b44631c2699f0223cc5e260864c43ca8c6ca'),
        ('basis/auxiliary/aug-cc-pVTZ/H', '384e43bb427e911de6c900a071fc1e3adcc6970572afc3c5d05fed8f1c78e066'),
        ('src/molecule_parser.F90', '3210d7d0e49d38f824ebc74bc4d295ced136ef5bf43bf631be04d264acc1585c'),
        ('src/basis_operations.F90', '33f87d26c0b9198afc7a73a1b7c985c2dede96a3e313dfc9d82aa210c0599ab9'),
        ('777f904:src/tools/cluster_file_interface.F90', '794389fe9b49fc408c859ed9aa2a78b3bfc3c47efbab6ed86b175e5d04887a60'),
        ('Psi4:d52fb16a19:psi4/share/psi4/basis/aug-cc-pvtz.gbs', '235b70809d1f205815c72046ab7319b291c4fe826b93f83dea876502dbb0a53c'),
        ('https://www.basissetexchange.org/api/basis/aug-cc-pvtz-rifit/format/json/?elements=1,8', '5b91e4286ce95bbca28bd04d17435c69adc29fb24f4482e3f8f5e1824ae10d4d'),
        ('https://www.basissetexchange.org/api/basis/aug-cc-pvtz/format/json/?elements=1,8&version=0', 'd5188d878e3e6b2a4938369a5838385f3c7e179202d6cc6270bf8dedc9cf7c3e'),
    ),
    elements=(ElementReference("O", _O_MAIN, _O_AUX), ElementReference("H", _H_MAIN, _H_AUX)),
)
