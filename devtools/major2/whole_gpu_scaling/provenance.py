"""Exact minimal owned runner delta; no generated runtime overlay."""
import ast
import difflib
from pathlib import Path
import config as s

INSERT = (
    '    # Symmetric native trace verbosity; no JK lifecycle modifications.\n'
    '    psi4.set_options({"PRINT": 3, "SCF__PRINT": 3, "SAPT__PRINT": 3})\n')
ANCHOR = '        psi4.set_options(options)\n'
METADATA = (
    '        record["trace_print_options"] = {\n'
    '            "global": psi4.core.get_global_option("PRINT"),\n'
    '            "scf": psi4.core.get_option("SCF", "PRINT"),\n'
    '            "sapt": psi4.core.get_option("SAPT", "PRINT"),\n'
    '        }\n')


def expected_runner(old):
    # Intentionally outside SCIENCE: its inherited hash/options stay unchanged.
    value = old.replace('    proofs = []\n', INSERT+'    proofs = []\n', 1)
    return value.replace(ANCHOR, ANCHOR+METADATA, 1)


def verify_runner(root):
    root = Path(root)
    old = root/"devtools/saptdft_cpks_suite_run.py"
    new = s.HERE/"runner.py"
    s.require(s.sha(old) == s.p.RUNNER_SHA, "original runner SHA")
    s.require(new.read_text() == expected_runner(old.read_text()), "unreviewed runner delta")
    tree = ast.parse(new.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and
             isinstance(node.func, ast.Attribute) and node.func.attr == "energy"]
    s.require(len(calls) == 1, "exactly one energy endpoint")
    return dict(original_path="devtools/saptdft_cpks_suite_run.py", original_sha256=s.sha(old),
                owned_sha256=s.sha(new), patch="".join(difflib.unified_diff(
                    old.read_text().splitlines(True), new.read_text().splitlines(True),
                    fromfile="original", tofile="owned")))
