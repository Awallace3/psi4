"""Manual dimer-only HF/DF SCF cold/warm diagnostic; NOT full SAPT or speedup proof."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import textwrap

INPUT_SHA="988d3b5c6b80b611c6fecc5b6292d5fde51d8f1be89b0a1bb7a017561f4e337a"


def timer_summary(records,width=78):
    """Native timer paths use ';'; show the leaf in a narrow indented tree."""
    if width<40:raise ValueError("Timer table width must be at least 40")
    lines=["\n=== DIMER SCF TIMERS ===","Inclusive host wall time; nested, NOT additive.",
           f"{'seconds':>10} {'calls':>5}  timer"]
    for path,row in sorted(records.items(),key=lambda item:tuple(item[0].split(";"))):
        parts=path.split(";")
        depth=len(parts)-1
        indent="  "*min(depth,6)+("… " if depth>6 else "")
        numbers=f"{row['wall_time']:10.6f} {int(row['n_calls']):5d}  "
        wrapped=textwrap.wrap(parts[-1],width=width-len(numbers)-len(indent)) or [""]
        lines.append(numbers+indent+wrapped[0])
        lines.extend(" "*len(numbers)+indent+line for line in wrapped[1:])
    lines.extend(["Compute API may only enqueue; waits can land in free/D2H.",
                  "No added CUDA barriers/warm-up. Full paths: *-timers.json."])
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix",type=Path,required=True)
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--backend",choices=("gpu","cpu"),default="gpu")
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--repeats",type=int,default=2,
                        help="First call cold, subsequent calls warm in this same process")
    parser.add_argument("--e-convergence",type=int,default=10)
    parser.add_argument("--d-convergence",type=int,default=12)
    parser.add_argument("--maxiter",type=int,default=100)
    args=parser.parse_args()
    if not 1<=args.threads<=8 or not 1<=args.repeats<=10:
        parser.error("Use 1–8 threads and 1–10 repeats on cachy")
    raw=args.inputs.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=INPUT_SHA:
        raise RuntimeError("Frozen jun-cc-pVDZ timing-suite input differs")
    inputs=json.loads(raw)
    point=inputs["systems"]["peptide"]
    if hashlib.sha256(point["geometry"].encode()).hexdigest()!=point["geometry_sha256"]:
        raise RuntimeError("Peptide geometry checksum differs")
    # Import only after environment selection; do not create/warm any CUDA
    # session before the first actual SCF call.
    import psi4
    from psi4.driver.procrouting.proc import scf_helper
    prefix=args.prefix.resolve(strict=True)
    for module in (psi4,psi4.core):
        if not Path(module.__file__).resolve().is_relative_to(prefix):
            raise RuntimeError("Wrong manual Psi4 import: "+module.__file__)
    psi4.core.set_output_file("stdout",False)
    psi4.set_memory("6 GiB")
    psi4.set_num_threads(args.threads)
    options={"basis":"jun-cc-pvdz","df_basis_scf":"aug-cc-pvdz-jkfit",
             "reference":"rhf","scf_type":"df","puream":True,
             "use_cuest":args.backend=="gpu","cuest_mixed_precision":False,
             "cuest_gemm_min_dim":0,"ints_tolerance":1e-16,
             "save_jk":True,"df_ints_io":"save","freeze_core":True,
             "e_convergence":args.e_convergence,"d_convergence":args.d_convergence,
             "maxiter":args.maxiter,"guess":"sad"}
    psi4.set_options(options)
    report={"scope":__doc__,"backend":args.backend,"options":options,
            "threads":args.threads,"input_sha256":INPUT_SHA,"psi4_file":psi4.__file__,
            "core_file":psi4.core.__file__,"version":psi4.__version__,
            "timer_switch":os.environ.get("PSI4_CUEST_SCF_TIMERS","0"),
            "cuda_cache_disable":os.environ.get("CUDA_CACHE_DISABLE"),
            "samples":[],"status":"RUNNING"}
    try:
        for index in range(args.repeats):
            label="cold" if index==0 else f"warm-{index}"
            print(f"\n=== {label}: peptide dimer SCF ===",flush=True)
            psi4.core.clean_timers()
            sample={"label":label,"status":"RUNNING"}
            report["samples"].append(sample)
            start=time.perf_counter()
            psi4.core.timer_on("Manual peptide dimer")
            try:
                psi4.core.timer_on("Manual molecule construction")
                try:molecule=psi4.geometry(point["geometry"])
                finally:psi4.core.timer_off("Manual molecule construction")
                # Same helper and retained-JK policy as SAPT(DFT)'s HF dimer,
                # without performing either monomer or any SAPT term.
                psi4.core.timer_on("SAPT(DFT):Dimer SCF")
                try:
                    wfn=scf_helper("SCF",molecule=molecule,banner="Manual peptide HF dimer")
                    energy=wfn.energy()
                finally:psi4.core.timer_off("SAPT(DFT):Dimer SCF")
                if wfn.basisset().nbf()!=point["spherical_basis_counts"]["primary"]:
                    raise RuntimeError("Unexpected peptide AO basis dimension")
                sample.update(status="PASS",energy_Eh=float(energy),nbf=wfn.basisset().nbf())
                psi4.core.timer_on("Manual wavefunction cleanup")
                try:
                    del wfn,molecule
                    gc.collect()
                finally:psi4.core.timer_off("Manual wavefunction cleanup")
            except BaseException as error:
                sample.update(status="FAILED",error=f"{type(error).__name__}: {error}")
                raise
            finally:
                psi4.core.timer_off("Manual peptide dimer")
                sample["wall_seconds"]=time.perf_counter()-start
                sample["timers"]=psi4.core.get_timer_dict()
                text=f"\n{label}: {sample['status']}, wall={sample['wall_seconds']:.6f} s\n"
                text+=timer_summary(sample["timers"])
                print(text,flush=True)
                Path(f"{label}-timers.json").write_text(json.dumps(sample,indent=2)+"\n")
            psi4.core.clean()
        report["status"]="PASS_DIAGNOSTIC_ONLY"
    finally:
        if report["status"]=="RUNNING":report["status"]="FAILED"
        Path("scf-result.json").write_text(json.dumps(report,indent=2)+"\n")
    print("\n=== SUMMARY: dimer only; cold/warm are NOT independent timing repeats ===")
    for sample in report["samples"]:
        print(f"{sample['label']:10s} {sample['wall_seconds']:12.6f} s  {sample['energy_Eh']:.14f} Eh")


if __name__=="__main__":
    main()
