"""Print comparable per-term wall seconds and fragment closure from d4i_timing.py results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


# D4 pair extraction is inside D4. The HF interaction timers occur in PBE0 delta-HF.
TIMERS = {
    "grac_a": "SAPT(DFT):GRAC Shift Monomer A",
    "grac_b": "SAPT(DFT):GRAC Shift Monomer B",
    "ks_a": "SAPT(DFT): Monomer A DFT",
    "ks_b": "SAPT(DFT): Monomer B DFT",
    "hf_dimer": "SAPT(DFT):Dimer SCF",
    "hf_a": "SAPT(DFT):Monomer A SCF",
    "hf_b": "SAPT(DFT):Monomer B SCF",
    "hf_elst_for_dhf": "SAPT(HF):elst",
    "hf_exch_for_dhf": "SAPT(HF):exch",
    "hf_ind_for_dhf": "SAPT(HF):ind",
    "build_jk": "SAPT(DFT):Build JK",
    "elst": "SAPT(DFT):elst",
    "exch": "SAPT(DFT):exch",
    "ind": "SAPT(DFT):ind",
    "disp_d4_inclusive": "SAPT(DFT):D4 Interaction Energy",
    "fdisp_d4_pairs_nested": "SAPT(DFT):D4 intermolecular pairs",
    "ibo_setup": "SAPT(DFT): F-SAPT Setup + Localization (IBO)",
    "felst": "SAPT(DFT): F-SAPT Electrostatics",
    "fexch": "SAPT(DFT): F-SAPT Exchange",
    "find": "SAPT(DFT): F-SAPT Induction",
}


def summarize(path):
    d = json.loads(path.read_text())
    opt = d["options"]
    frag = bool(d["fragment_sums_hartree"])
    field = d["external_potentials"] or {}
    row = dict(case=path.parent.name, functional=opt["sapt_dft_functional"],
               arm="gpu" if opt["use_cuest"] else "cpu", fsapt=frag, nbf=d["nbf"],
               total_s=d["wall_seconds"], group_analysis_s=d.get("group_seconds"), total_Eh=d["energy_hartree"],
               fragment_algorithm=opt.get("fisapt_fsapt_integral_algorithm") if frag else None,
               integral_direct=opt.get("cuest_df_integral_direct", False),
               einsums=opt["sapt_dft_use_einsums"], geometry_sha256=d["geometry_sha256"],
               mm_sites=sum(len(sites) for sites in field.values()),
               field_sha256=hashlib.sha256(json.dumps(field, sort_keys=True).encode()).hexdigest(),
               host=d.get("host"), threads=d.get("threads"), memory=d.get("memory"),
               libxc="|".join(d.get("libxc") or []),
               psi4_core=d["psi4_core"], cuda_runtime=d["cuda_runtime"], cuest_sha256=d["cuest_sha256"])
    timers = {v["timer_name"]: v["wall_time"] for v in d["timers"].values()}
    for label, timer in TIMERS.items():
        if (label.startswith(("ks_", "grac_")) or label.endswith("_for_dhf")) and row["functional"] == "HF":
            row[label + "_s"] = None
        elif label in ("ibo_setup", "felst", "fexch", "find") and not frag:
            row[label + "_s"] = None
        elif label == "fdisp_d4_pairs_nested":
            # Earlier production revisions timed the inclusive D4 operation only.
            row[label + "_s"] = timers.get(timer)
        else:
            row[label + "_s"] = timers[timer]
    sums = d["fragment_sums_hartree"]
    if frag:
        partitions = dict(ELST=sums["ELST_AB"], EXCH=sums["EXCH_AB"],
                          IND=sums["INDAB_AB"] + sums["INDBA_AB"], DISP=sums["EMPIRICAL_DISP"])
    for term, value in d["scalar_hartree"].items():
        row[term.lower() + "_Eh"] = value
        row[term.lower() + "_closure_Eh"] = partitions[term] - value if frag else None
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, nargs="+", help="result.json files")
    args = parser.parse_args()
    rows = [summarize(path) for path in args.result]
    writer = csv.DictWriter(sys.stdout, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
