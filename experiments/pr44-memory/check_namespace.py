"""Run with staged PsiAPI Python: check_namespace.py OUTPUT [MODE].

Without MODE, run red/green/reference in fresh processes and compare components.
"""
import json
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
        return

    mode = sys.argv[2]
    assert mode in ("broken", "namespace", "reference")
    with tempfile.TemporaryDirectory(prefix="pr44-disk-jk-") as scratch:
        os.environ["PSI_SCRATCH"] = scratch
        import psi4

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
        result = {"completed": False, "core": psi4.core.__file__, "workaround": workaround}
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


if __name__ == "__main__":
    main()
