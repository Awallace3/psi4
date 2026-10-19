"""Portable PRINT3 parser tests. Real archive replay is mandatory via replay_print3.py."""
import ast
import copy
from pathlib import Path
import unittest

import config
import science as legacy
import validation


PRIMARY = """  ==> Primary Basis <==

  -AO BASIS SET INFORMATION:
    Name                   = 6-31+G**
    Blend                  = 6-31+G**
    Total number of shells = 122
    Number of primitives   = 240
    Number of AO           = 260
    Number of SO           = 250
    Maximum AM             = 2
    Spherical Harmonics    = TRUE

  -Contraction Scheme:
"""


def loader(name, role, keyword):
    return (f"   => Loading Basis Set <=\n\n    Name: {name}\n"
            f"    Role: {role}\n    Keyword: {keyword}\n    atoms 1 entry H\n\n")


def parser_fixture():
    """Only format grammar, not a synthetic scientific record."""
    text = (loader("6-31+G**", "ORBITAL", "BASIS") +
            loader("DEF2-UNIVERSAL-JKFIT", "JKFIT", "DF_BASIS_SCF") +
            loader("AUG-CC-PVDZ-RI", "RIFIT", "DF_BASIS_MP2"))
    for i, stage in enumerate(legacy.STAGES):
        scf = "RHF" if i < 3 else "RKS"
        text += (f"SAPT(DFT): {stage}\n"+PRIMARY+
                 f"@DF-{scf} iter SAD: 0\n@DF-{scf} Final Energy: -1.0\n")
    return text


class PrimaryParserTests(unittest.TestCase):
    def check(self, text):
        validation.print3_basis(text, {"basis": "6-31+g**", "nbf": 250},
                               {"DF_BASIS_SCF": "def2-universal-jkfit",
                                "DF_BASIS_MP2": "aug-cc-pvdz-ri"})

    def reject(self, text):
        with self.assertRaises(RuntimeError):
            self.check(text)

    def test_five_primary_blocks_with_ao_different_from_so(self):
        self.check(parser_fixture())

    def test_wrong_primary_despite_correct_loader(self):
        self.reject(parser_fixture().replace("Name                   = 6-31+G**",
                                            "Name                   = WRONG", 1))

    def test_one_false_among_four_true(self):
        self.reject(parser_fixture().replace("= TRUE", "= FALSE", 1))

    def test_wrong_so_not_ao(self):
        self.reject(parser_fixture().replace("Number of SO           = 250",
                                            "Number of SO           = 260", 1))

    def test_duplicate_primary(self):
        self.reject(parser_fixture().replace(PRIMARY, PRIMARY+PRIMARY, 1))

    def test_missing_primary(self):
        self.reject(parser_fixture().replace(PRIMARY, "", 1))

    def test_loader_alone_is_not_orbital_evidence(self):
        self.reject(parser_fixture().replace(PRIMARY, ""))

    def test_primary_before_its_stage_is_displaced(self):
        text = parser_fixture().replace(PRIMARY, "", 1)
        self.reject(PRIMARY+text)

    def test_primary_after_final_energy_is_late(self):
        text = parser_fixture().replace(PRIMARY, "", 1)
        self.reject(text.replace("Final Energy: -1.0\n", "Final Energy: -1.0\n"+PRIMARY, 1))

    def test_primary_after_iteration_is_late(self):
        text = parser_fixture().replace(PRIMARY, "", 1)
        self.reject(text.replace("iter SAD: 0\n", "iter SAD: 0\n"+PRIMARY, 1))

    def test_banner_and_primary_moved_together_after_first_row(self):
        pair = "SAPT(DFT): delta HF Dimer\n"+PRIMARY
        text = parser_fixture().replace(pair, "", 1)
        # There remains an iteration after the moved banner, as on the real SCF.
        self.reject(text.replace("iter SAD: 0\n", "iter SAD: 0\n"+pair+"@DF-RHF iter 1: 0\n", 1))

    def test_reordered_stages(self):
        text = parser_fixture().replace("delta HF Monomer A", "TEMP")
        self.reject(text.replace("delta HF Monomer B", "delta HF Monomer A")
                    .replace("TEMP", "delta HF Monomer B"))

    def test_missing_duplicate_or_conflicting_primary_fields(self):
        for line in PRIMARY.splitlines(keepends=True):
            if "=" in line:
                with self.subTest(field=line.strip(), corruption="missing"):
                    self.reject(parser_fixture().replace(line, "", 1))
                with self.subTest(field=line.strip(), corruption="duplicate"):
                    self.reject(parser_fixture().replace(line, line+line, 1))
        self.reject(parser_fixture().replace("= TRUE", "= TRUE\n Spherical Harmonics = FALSE", 1))

    def test_malformed_primary_headers_and_fields(self):
        for old, new in (("==> Primary Basis <==", "prefix ==> Primary Basis <=="),
                         ("==> Primary Basis <==", "==> Primary Basis <="),
                         ("Number of SO           = 250", "Number of SO : 250"),
                         ("Number of SO           = 250", "Number of SO = 250 junk"),
                         ("= TRUE", "= true"), ("= TRUE", "= TRUE suffix"),
                         ("Name                   =", "Name:"),
                         ("Blend                  = 6-31+G**", "Blend = WRONG"),
                         ("Number of AO           = 260", "Number of AO = nan")):
            with self.subTest(new=new):
                self.reject(parser_fixture().replace(old, new, 1))

    def test_aux_names_are_role_and_keyword_bound(self):
        text = parser_fixture()
        for old, new in (("Role: JKFIT", "Role: RIFIT"),
                         ("Role: RIFIT", "Role: JKFIT"),
                         ("Keyword: DF_BASIS_SCF", "Keyword: DF_BASIS_MP2"),
                         ("Keyword: DF_BASIS_MP2", "Keyword: DF_BASIS_SCF"),
                         ("Name: DEF2-UNIVERSAL-JKFIT", "Name: WRONG"),
                         ("Name: AUG-CC-PVDZ-RI", "Name: WRONG"),
                         ("Name: 6-31+G**", "Name: WRONG"),
                         ("Role: JKFIT", "Role: JKFIT\n    Role: RIFIT"),
                         ("Keyword: DF_BASIS_MP2", ""),
                         ("=> Loading Basis Set <=", "prefix => Loading Basis Set <=")):
            with self.subTest(new=new):
                self.reject(text.replace(old, new, 1))

    def test_missing_aux_loader_not_rescued_by_stray_name(self):
        for name, role, keyword in (("DEF2-UNIVERSAL-JKFIT", "JKFIT", "DF_BASIS_SCF"),
                                    ("AUG-CC-PVDZ-RI", "RIFIT", "DF_BASIS_MP2")):
            with self.subTest(role=role):
                self.reject(parser_fixture().replace(loader(name, role, keyword), name+"\n"))

    def test_conflicting_extra_loader_is_rejected(self):
        self.reject(parser_fixture()+loader("WRONG", "JKFIT", "DF_BASIS_SCF"))

    def test_blank_separated_conflicting_loader_fields_are_rejected(self):
        for field in ("Role: WRONG", "Name: WRONG", "Keyword: WRONG"):
            with self.subTest(field=field):
                self.reject(parser_fixture().replace(
                    "    atoms 1 entry H\n", "    atoms 1 entry H\n\n    "+field+
                    "\n    atoms 2 entry H\n", 1))


class ParityTests(unittest.TestCase):
    def test_all_other_legacy_gates_are_ast_identical(self):
        def function(path):
            return copy.deepcopy(next(n for n in ast.parse(path.read_text()).body
                                      if isinstance(n, ast.FunctionDef) and n.name == "validate_record"))
        old = function(Path(legacy.__file__))
        new = function(Path(validation.__file__))
        removed = []
        for node in ast.walk(old):
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "token":
                removed = [n.value for n in node.iter.elts
                           if n.value in ("Spherical Harmonics?: true", "Basis Set: 6-31+G**")]
                node.iter.elts = [n for n in node.iter.elts if n.value not in removed]
        self.assertEqual(removed, ["Spherical Harmonics?: true", "Basis Set: 6-31+G**"])
        additions = [n for n in new.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                     and isinstance(n.value.func, ast.Name) and n.value.func.id == "print3_basis"]
        self.assertEqual(len(additions), 1)
        self.assertEqual(ast.unparse(additions[0]), "print3_basis(text, expected, science)")
        new.body.remove(additions[0])
        self.assertEqual(ast.dump(old), ast.dump(new))
        for name in ("require", "finite", "geometry", "sections", "solvers", "timers", "profile"):
            self.assertIs(getattr(validation, name), getattr(legacy, name))

    def test_controller_delta_is_only_validator_import_and_call(self):
        import subprocess
        here = Path(__file__).resolve().parent
        original = subprocess.check_output(
            ["git", "show", "6ae8a733279f6b830775fc0ceb7061a7ec3a6f9a:"
             "devtools/major2/whole_gpu_scaling/controller.py"], cwd=here, text=True)
        expected = original.replace("import science\n", "import validation\n").replace(
            "science.validate_record(", "validation.validate_record(")
        self.assertEqual((here/"controller.py").read_text(), expected)


if __name__ == "__main__":
    unittest.main()
