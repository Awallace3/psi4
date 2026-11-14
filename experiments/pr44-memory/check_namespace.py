"""Run with staged PsiAPI Python: check_namespace.py OUTPUT [MODE].

Without MODE, run red/green/reference in fresh processes and compare components.
"""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) == 2:
        for mode in ("broken", "namespace", "reference"):
            subprocess.run([sys.executable, __file__, str(output), mode], check=True)
        fixed = json.loads((output / "namespace.json").read_text())
        reference = json.loads((output / "reference.json").read_text())
        assert fixed["completed"] and reference["completed"]
        delta = {k: fixed["variables"][k] - v for k, v in reference["variables"].items()}
        (output / "comparison.json").write_text(json.dumps(delta, indent=2) + "\n")
        assert all(abs(v) < 1e-8 for v in delta.values()), delta
        print(delta)
        if os.environ.get("PR44_AUDIT_PROTEIN") == "1":
            audit_protein(output)
        return

    mode = sys.argv[2]
    assert mode in ("broken", "namespace", "reference")
    with tempfile.TemporaryDirectory(prefix="pr44-disk-jk-") as scratch:
        os.environ["PSI_SCRATCH"] = scratch
        import psi4
        with open(psi4.core.__file__, "rb") as handle:
            core_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        expected_hash = os.environ.get("PR44_EXPECTED_CORE_SHA256")
        if expected_hash:
            assert core_hash == expected_hash, (psi4.core.__file__, core_hash)
        expected_stage = os.environ.get("PR44_EXPECTED_STAGE")
        if expected_stage:
            assert Path(psi4.core.__file__).resolve().is_relative_to(Path(expected_stage).resolve())

        workaround = None
        if mode == "namespace":
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from namespace_workaround import install
            workaround = install()
        psi4.core.IOManager.shared_object().set_default_path(scratch + "/")
        psi4.core.set_output_file(str(output / f"{mode}.out"), False)
        psi4.set_memory("1 GiB")
        psi4.set_num_threads(2)
        molecule = psi4.geometry("""
0 1
O -2.930978458 -0.216411437 0.0
H -3.655219777 1.440921844 0.0
H -1.133225297 0.076934530 0.0
--
0 1
O 2.552311356 0.210645882 0.0
H 3.175492012 -0.706268134 -1.433472544
H 3.175492012 -0.706268134 1.433472544
units bohr
symmetry c1
""")
        psi4.set_options({
            "basis": "aug-cc-pvdz", "scf_type": "mem_df" if mode == "reference" else "disk_df",
            "scf_subtype": "out_of_core", "guess": "sad", "sad_scf_type": "df",
            "df_scf_guess": False, "e_convergence": 1e-8, "d_convergence": 1e-8,
            "sapt_dft_grac_compute": "none", "sapt_dft_grac_shift_a": 0.136,
            "sapt_dft_grac_shift_b": 0.136, "sapt_dft_functional": "pbe0",
            "sapt_dft_do_hybrid": True, "sapt_dft_do_dhf": True,
        })
        result = {"completed": False, "core": psi4.core.__file__,
                  "core_sha256": core_hash, "workaround": workaround}
        try:
            result["energy"] = psi4.energy("sapt(dft)", molecule=molecule)
            assert mode != "broken", "Expected failure not reproduced"
            result["variables"] = {k: psi4.variable(k) for k in [
                "SAPT ELST ENERGY", "SAPT EXCH ENERGY", "SAPT IND ENERGY",
                "SAPT DISP ENERGY", "SAPT TOTAL ENERGY"]}
            result["completed"] = True
        except RuntimeError as error:
            if mode != "broken" or "PSIO_ERROR: 13" not in str(error) or "manage_JK_disk" not in str(error):
                raise
            result.update(expected_failure_reproduced=True, exception=str(error))
        finally:
            psi4.core.flush_outfile()
            (output / f"{mode}.json").write_text(json.dumps(result, indent=2) + "\n")
        psi4.core.clean()


def audit_protein(output):
    """Cheap old-binary check of the *actual* full-run options and basis sizes."""
    import psi4
    here = Path(__file__).resolve().parent.parent / "pr44-protein157"
    config = json.loads((here / "config.json").read_text())
    psi4.set_options(config["options"])
    atoms = (here / "protein157.xyz").read_text().splitlines()[2:]
    split = config["fragment_a_atoms"]
    molecule = psi4.geometry(
        "0 1\n" + "\n".join(atoms[:split]) + "\n--\n0 1\n" + "\n".join(atoms[split:])
        + "\nunits angstrom\nsymmetry c1\nno_reorient\nno_com\n")
    primary = psi4.core.BasisSet.build(molecule, "ORBITAL", config["options"]["basis"])
    auxiliary = psi4.core.BasisSet.build(
        molecule, "DF_BASIS_MP2", config["options"]["df_basis_mp2"], "RIFIT", config["options"]["basis"])
    result = {
        "scf_subtype": psi4.core.get_global_option("SCF_SUBTYPE"),
        "scf_subtype_explicitly_changed": psi4.core.has_global_option_changed("SCF_SUBTYPE"),
        "nbf": primary.nbf(), "naux": auxiliary.nbf(),
        "max_nprimitive": auxiliary.max_nprimitive(),
        "max_function_per_shell": auxiliary.max_function_per_shell(),
    }
    assert result["scf_subtype"] == "AUTO" and result["scf_subtype_explicitly_changed"]
    assert result["nbf"] == 2491 and result["naux"] == 7384
    (output / "protein-options.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
