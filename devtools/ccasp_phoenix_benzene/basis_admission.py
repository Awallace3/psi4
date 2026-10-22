# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Benzene basis metadata admission only; stdlib import, no SCF or integrals.

Native normalized shell data are exact float.hex snapshots, NOT a reconstruction
from raw GBS coefficients. A private, reviewed native probe supplies the reference.
"""
import hashlib
import json
import math
from pathlib import Path
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract():
    return json.loads(Path(__file__).with_name("basis_contract.json").read_text())


def puream_state(core):
    return {"value": bool(core.get_global_option("PUREAM")),
            "changed": bool(core.has_global_option_changed("PUREAM"))}


def scoped_auxiliary(core, options_state, molecule, audit, *, builder=None):
    """Scope ONLY Cartesian AUX construction; preserve value AND changed flag.

    builder is a controlled test/probe error seam, never a patched production
    function. Science uses the real BasisSet.build exactly once in this scope.
    """
    before = puream_state(core)
    audit.update(before=before, requested_puream=0)
    stash = options_state(["PUREAM"])
    try:
        core.set_global_option("PUREAM", False)
        audit["during"] = puream_state(core)
        if builder is not None:
            return builder()
        return core.BasisSet.build(molecule, "DF_BASIS_SCF", "aug-cc-pvtz-ri", puream=0)
    finally:
        stash.restore()
        audit["after"] = puream_state(core)
        audit["restored_exactly"] = audit["after"] == before
        if not audit["restored_exactly"]:
            raise ValueError("PUREAM value/changed flag restoration failed: " + repr(audit))


def raw_gbs_elements(text):
    """Read C/H contraction declarations as raw tokens; no normalization/math."""
    elements = {}
    for block in text.split("****"):
        lines = [s for line in block.splitlines() if (s := line.split("!")[0].strip())]
        if not lines or not re.fullmatch(r"[A-Za-z]+\s+0", lines[0]):
            continue
        symbol = lines[0].split()[0]
        if symbol not in ("C", "H"):
            continue
        if symbol in elements:
            raise ValueError("duplicate C/H GBS block")
        shells, i = [], 1
        while i < len(lines):
            fields = lines[i].split()
            if len(fields) != 3 or len(fields[0]) != 1 or fields[0].upper() not in "SPDFGHI":
                raise ValueError("unexpected GBS shell declaration: " + lines[i])
            l, n = "SPDFGHI".index(fields[0].upper()), int(fields[1])
            rows = [line.split() for line in lines[i + 1:i + 1 + n]]
            if n < 1 or len(rows) != n or any(len(row) != 2 for row in rows):
                raise ValueError("incomplete/non-single-contraction GBS primitive records")
            shells.append(dict(l=l, nprimitive=n, raw_scale_token=fields[2],
                               raw_exponent_coefficient_tokens=rows))
            i += 1 + n
        elements[symbol] = shells
    if set(elements) != {"C", "H"}:
        raise ValueError("canonical GBS must contain C and H")
    return elements


def static_counts(elements):
    """Combinatorial function counts only; not native measurement or normalization."""
    result = {}
    for symbol, shells in elements.items():
        result[symbol] = dict(
            shells_by_l={str(l): sum(s["l"] == l for s in shells)
                         for l in sorted({s["l"] for s in shells})},
            nshell=len(shells),
            cartesian=sum((s["l"] + 1) * (s["l"] + 2) // 2 for s in shells),
            spherical=sum(2 * s["l"] + 1 for s in shells))
    result["6C6H"] = {key: 6 * (result["C"][key] + result["H"][key])
                       for key in ("nshell", "cartesian", "spherical")}
    return result


def basis_files(native_source, stage, receipt):
    records = receipt.setdefault("basis_files", {})
    for name, pin in contract()["basis_files"].items():
        source = Path(native_source) / "psi4/share/psi4/basis" / name
        staged = Path(stage) / "share/psi4/basis" / name
        rec = records[name] = dict(source=str(source), staged=str(staged),
            expected_sha256=pin, source_sha256=file_sha(source), staged_sha256=file_sha(staged))
        if rec["source_sha256"] != pin or rec["staged_sha256"] != pin:
            raise ValueError("canonical basis file hash mismatch: " + repr(rec))
        raw = raw_gbs_elements(staged.read_text())
        rec.update(raw_coefficient_convention="unmodified GBS contraction tokens, NOT native normalization",
                   raw_C_H_shells=raw, raw_shell_sha256=digest(raw), static_counts=static_counts(raw))
    return records


def _hex(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite native basis datum")
    return value.hex()


def native_snapshot(basis):
    molecule = basis.molecule()
    centres = [[_hex(molecule.x(i)), _hex(molecule.y(i)), _hex(molecule.z(i))]
               for i in range(molecule.natom())]
    metadata = {name: getattr(basis, name)() for name in
                ("nbf", "nao", "has_puream", "nshell", "max_am")}
    metadata = {k: bool(v) if k == "has_puream" else int(v) for k, v in metadata.items()}
    shells = []
    for i in range(basis.nshell()):
        s = basis.shell(i)
        shells.append(dict(centre=int(basis.shell_to_center(i)), l=int(s.am),
            nprimitive=int(s.nprimitive), is_pure=bool(s.is_pure()),
            exponents_hex=[_hex(s.exp(k)) for k in range(s.nprimitive)],
            native_normalized_coefficients_hex=[_hex(s.coef(k)) for k in range(s.nprimitive)]))
    data = dict(metadata=metadata, centres_bohr_hex=centres, ordered_shells=shells,
                coefficient_convention="GaussianShell.coef native-normalized binary64 float.hex")
    return dict(data, fingerprint_sha256=digest(data))


def check_shape(snapshot, expected, label):
    if snapshot["metadata"] != expected:
        raise ValueError(f"{label} basis representation/count mismatch: "
                         f"ACTUAL {snapshot['metadata']}; EXPECTED {expected}")


def build_bases(core, options_state, molecule, receipt):
    """Caller applies SCF options first. No density, integrals, SCF or ISA call."""
    receipt["puream_before_main"] = puream_state(core)
    main = core.BasisSet.build(molecule, "ORBITAL", "aug-cc-pvtz", puream=1)
    receipt["main"] = native_snapshot(main)
    check_shape(receipt["main"], contract()["main"], "spherical MAIN")
    audit = receipt.setdefault("auxiliary_puream_scope", {})
    auxiliary = scoped_auxiliary(core, options_state, molecule, audit)
    receipt["auxiliary"] = native_snapshot(auxiliary)
    check_shape(receipt["auxiliary"], contract()["auxiliary"], "Cartesian response AUX")
    return main, auxiliary


def compare_manifest(receipt, manifest, binding):
    if (manifest.get("schema") != "ccasp-benzene-native-basis-admission/1"
            or manifest.get("status") != "basis_metadata_only_complete"
            or manifest.get("no_scf") is not True):
        raise ValueError("not a completed basis-only probe manifest")
    if manifest["binding"] != binding:
        raise ValueError("basis manifest source/core/fixture/config binding mismatch")
    if manifest["admission"]["basis_files"] != receipt["basis_files"]:
        raise ValueError("basis file identity/raw-shell manifest mismatch")
    for label in ("main", "auxiliary"):
        actual, reference = receipt[label], manifest["admission"][label]
        for item in (actual, reference):
            data = {k: v for k, v in item.items() if k != "fingerprint_sha256"}
            if digest(data) != item["fingerprint_sha256"]:
                raise ValueError("corrupt native shell fingerprint: " + label)
        if actual != reference:
            raise ValueError("exact ordered native shell fingerprint mismatch: " + label)


def admit_before_scf(core, options_state, molecule, native_source, stage,
                     receipt, manifest, binding, recipe_builder):
    basis_files(native_source, stage, receipt)
    main, auxiliary = build_bases(core, options_state, molecule, receipt)
    compare_manifest(receipt, manifest, binding)
    # Retain this recipe, constructed from precisely the admitted Cartesian AUX.
    part = recipe_builder(auxiliary)
    return main, part


def compare_scf_main(basis, admitted_main, receipt=None):
    actual = native_snapshot(basis)
    if receipt is not None:
        receipt["actual_scf_main"] = actual
    if actual != admitted_main:
        raise ValueError("SCF wavefunction MAIN differs from admitted exact shell fingerprint: "
                         + repr(actual["metadata"]))
    return actual
