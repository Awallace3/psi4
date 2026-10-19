"""Reuse parsed entries within one build without caching mutable basis objects."""

from collections import Counter

import pytest

import psi4
from psi4.driver.qcdb.libmintsbasisset import basishorde
from psi4.driver.qcdb.libmintsbasissetparser import Gaussian94BasisSetParser

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


def _count_parses(monkeypatch):
    calls = Counter()
    original = Gaussian94BasisSetParser.parse

    def parse(self, entry, lines):
        calls[(entry, tuple(lines) if not isinstance(lines, str) else lines)] += 1
        return original(self, entry, lines)

    monkeypatch.setattr(Gaussian94BasisSetParser, "parse", parse)
    return calls


@pytest.mark.parametrize("atomlist", [False, True])
def test_parse_once_per_source_entry(monkeypatch, atomlist):
    calls = _count_parses(monkeypatch)
    mol = psi4.geometry("H 0 0 0\nH 0 0 1\nH 0 0 2\nH 0 0 3\nsymmetry c1")
    basis = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pvdz",
                                    puream=True, return_atomlist=atomlist)
    nbf = sum(b.nbf() for b in basis) if atomlist else basis.nbf()
    assert nbf == 20
    assert calls
    assert max(calls.values()) == 1


def test_label_ghost_and_redefined_source(monkeypatch):
    template = """spherical
****
H 0
S 1 1.0
{ordinary} 1.0
****
H_SPECIAL 0
S 1 1.0
2.0 1.0
****
"""
    source = {"basis": template.format(ordinary=1.0)}

    def spec(mol, role):
        mol.set_basis_all_atoms("basis", role=role)
        return source

    monkeypatch.setitem(basishorde, "PARSE_REUSE", spec)
    mol = psi4.geometry("0 2\nH_SPECIAL 0 0 0\nH 0 0 1\nH 0 0 2\n@H 0 0 3\nsymmetry c1")
    for ordinary in (1.0, 3.0):
        source["basis"] = template.format(ordinary=ordinary)
        basis = psi4.core.BasisSet.build(mol, "ORBITAL", "PARSE_REUSE", puream=True)
        assert basis.nbf() == 4
        assert [basis.shell(i).exp(0) for i in range(4)] == [2.0, ordinary, ordinary, ordinary]


def test_parse_reuse_respects_puream(monkeypatch):
    def spec(mol, role):
        mol.set_basis_all_atoms("basis", role=role)
        return {"basis": "spherical\n****\nH 0\nD 1 1.0\n1.0 1.0\n****"}

    monkeypatch.setitem(basishorde, "PARSE_PUREAM", spec)
    mol = psi4.geometry("H 0 0 0\nH 0 0 1\nsymmetry c1")
    spherical = psi4.core.BasisSet.build(mol, "ORBITAL", "PARSE_PUREAM", puream=True)
    cartesian = psi4.core.BasisSet.build(mol, "ORBITAL", "PARSE_PUREAM", puream=False)
    assert spherical.nbf() == 10
    assert cartesian.nbf() == 12


def test_two_sources_for_same_element(monkeypatch):
    def spec(mol, role):
        for atom, name in enumerate(("first", "second", "first", "second")):
            mol.set_basis_by_number(atom, name, role=role)
        return {
            name: f"spherical\n****\nH 0\nS 1 1.0\n{exponent} 1.0\n****"
            for name, exponent in (("first", 1.0), ("second", 3.0))
        }

    monkeypatch.setitem(basishorde, "TWO_SOURCES", spec)
    calls = _count_parses(monkeypatch)
    mol = psi4.geometry("H1 0 0 0\nH2 0 0 1\nH3 0 0 2\nH4 0 0 3\nsymmetry c1")
    basis = psi4.core.BasisSet.build(mol, "ORBITAL", "TWO_SOURCES", puream=True)
    assert [basis.shell(i).exp(0) for i in range(4)] == [1.0, 3.0, 1.0, 3.0]
    assert max(calls.values()) == 1


@pytest.mark.parametrize("forced_puream,expected", [(True, 10), (False, 12)])
def test_parser_configuration_freshness(forced_puream, expected):
    qcdb = psi4.driver.qcdb
    mol = qcdb.Molecule("H 0 0 0\nH 0 0 1\nsymmetry c1")
    mol.set_basis_all_atoms("inline", role="BASIS")
    basis, _, _ = qcdb.BasisSet.construct(
        Gaussian94BasisSetParser(forced_puream=forced_puream), mol, "BASIS",
        basstrings={"inline": "spherical\n****\nH 0\nD 1 1.0\n1.0 1.0\n****"},
    )
    assert basis.nbf() == expected


def test_repeated_ecp_atoms():
    qcdb = psi4.driver.qcdb
    mol = qcdb.Molecule("I1 0 0 0\nI2 0 0 3\nsymmetry c1")
    mol.set_basis_all_atoms("def2-svp", role="BASIS")
    _, _, ecp = qcdb.BasisSet.construct(Gaussian94BasisSetParser(), mol, "BASIS")
    assert ecp.ecp_coreinfo == {"I1": 28, "I2": 28}
    shells = [[ecp.shell(i) for i in range(ecp.nshell()) if ecp.shell(i).nc == atom]
              for atom in (0, 1)]
    assert shells[0] and len(shells[0]) == len(shells[1])
    for first, second in zip(*shells):
        assert (first.l, first.PYexp, first.PYcoef, first.rpowers) == (
            second.l, second.PYexp, second.PYcoef, second.rpowers)
