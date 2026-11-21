"""Submit ONE consolidated 8h H200 embers suite job behind its CPU build."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from phoenix_timing import clean, digest, save


def command(source, root, build_job):
    if not str(build_job).isdecimal():
        raise ValueError("Numeric prerequisite job required")
    return [
        "sbatch", "--parsable",
        "--job-name=fsapt-suite-h200",
        "--account=gts-cs207-chemx",
        "--qos=embers",
        "--partition=gpu-h200",
        "--constraint=H200",
        "--gres=gpu:h200:1",
        "--nodes=1", "--ntasks=1",
        "--cpus-per-task=8",
        "--mem=160G",
        "--time=08:00:00",
        "--dependency=afterany:" + str(build_job),
        "--kill-on-invalid-dep=yes",
        "--no-requeue",
        "--signal=TERM@120",
        "--export=ALL",
        "--chdir=" + str(root),
        "--output=" + str(root / "logs/%x-%j.out"),
        "--error=" + str(root / "logs/%x-%j.err"),
        str(source / "devtools/fsapt_integration/phoenix_timing.sbatch"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--sapt0-inputs", type=Path, required=True)
    parser.add_argument("--protein83-cpu48-0", type=Path, required=True)
    parser.add_argument("--protein83-cpu48-1", type=Path, required=True)
    parser.add_argument("--protein83-svp-r0", type=Path, required=True)
    parser.add_argument("--protein83-svp-inputs", type=Path, required=True)
    parser.add_argument("--protein83-svp-inputs-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    source = Path(__file__).resolve().parents[2]
    helper_pin = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    clean(source, helper_pin)

    build_root = args.build_root.resolve(strict=True)
    build_manifest = json.loads((build_root / "manifest.json").read_text())
    build_job = (build_root / "job-id.txt").read_text().strip()
    science = Path(build_manifest["source"]).resolve(strict=True)
    pin = build_manifest["source_pin"]
    clean(science, pin)
    subprocess.run(
        ["git", "-C", str(source), "diff", "--exit-code", pin, helper_pin,
         "--", "psi4", "psi4_runtime_support"],
        check=True,
    )
    if not build_manifest["prepare_gpu_runtime"]:
        raise RuntimeError("Build must prepare the sealed runtime")

    inputs = args.inputs.resolve(strict=True)
    if digest(inputs) != args.inputs_sha256:
        raise RuntimeError("Frozen jun-cc-pVDZ inputs changed")
    svp_inputs = args.protein83_svp_inputs.resolve(strict=True)
    if digest(svp_inputs) != args.protein83_svp_inputs_sha256:
        raise RuntimeError("Frozen def2-SVP inputs changed")
    sapt0_inputs = args.sapt0_inputs.resolve(strict=True)
    sapt0_hashes = {}
    for name in ("water", "peptide", "nanotube", "protein83", "protein157"):
        p = (sapt0_inputs / (name + ".json")).resolve(strict=True)
        sapt0_hashes[name] = digest(p)

    cpu48_0 = args.protein83_cpu48_0.resolve(strict=True)
    cpu48_1 = args.protein83_cpu48_1.resolve(strict=True)
    svp_r0 = args.protein83_svp_r0.resolve(strict=True)

    python = Path(build_manifest["read_only_inputs"]["DISP0_DEPENDENCIES"]) / "bin/python"
    if not python.is_file():
        raise RuntimeError("Missing pinned environment")
    root = args.run_root
    if not root.is_absolute():
        raise ValueError("Absolute fresh run root required")
    root.mkdir(mode=0o700)
    (root / "logs").mkdir()

    argv = command(source, root, build_job)
    manifest = {
        "helper_source": str(source),
        "helper_pin": helper_pin,
        "science_source": str(science),
        "science_pin": pin,
        "build_root": str(build_root),
        "build_job": build_job,
        "build_manifest_sha256": digest(build_root / "manifest.json"),
        "inputs": str(inputs),
        "inputs_sha256": args.inputs_sha256,
        "sapt0_inputs": str(sapt0_inputs),
        "sapt0_input_hashes": sapt0_hashes,
        "protein83_cpu48_references": [
            {"result": str(cpu48_0), "result_sha256": digest(cpu48_0)},
            {"result": str(cpu48_1), "result_sha256": digest(cpu48_1)},
        ],
        "fixed_fdisp0_snapshots": {
            "protein83_jun_cc_pvdz_cpu48": {
                "result": str(cpu48_0),
                "result_sha256": digest(cpu48_0),
                "inputs": str(inputs),
                "inputs_sha256": args.inputs_sha256,
            },
            "protein83_def2_svp_r0": {
                "result": str(svp_r0),
                "result_sha256": digest(svp_r0),
                "inputs": str(svp_inputs),
                "inputs_sha256": args.protein83_svp_inputs_sha256,
            },
        },
        "kind": "gpu",
        "account": "gts-cs207-chemx",
        "qos": "embers",
        "allocation_wall_seconds": 28800,
        "command": argv,
        "basis": "jun-cc-pvdz",
        "scf_E": 10,
        "scf_D": 12,
        "threads_both_arms": 8,
        "factor_cache_bytes": 0,
        "authorization": "User requested full timing_test_suite + fsapt1 in one H200 embers job",
    }
    save(root / "manifest.json", manifest)

    env = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}
    env.update(
        FSAPT_SOURCE=str(source),
        FSAPT_PYTHON=str(python),
        FSAPT_RUN_ROOT=str(root),
        FSAPT_MANIFEST_SHA256=digest(root / "manifest.json"),
        FSAPT_DRIVER=str(source / "devtools/fsapt_integration/phoenix_suite_timing.py"),
    )
    test = subprocess.run([argv[0], "--test-only", *argv[1:]], env=env,
                          text=True, capture_output=True, timeout=60)
    save(root / "test-only.json", vars(test))
    print(test.stdout + test.stderr, flush=True)
    if test.returncode or "gpu-h200" not in test.stdout + test.stderr:
        raise RuntimeError("Exact H200 route test failed")

    (root / "SUBMISSION_ATTEMPTED").write_text("Single call; no automatic retry.\n")
    result = subprocess.run(argv, env=env, text=True, capture_output=True, timeout=60)
    save(root / "submission.json", vars(result))
    print(result.stdout + result.stderr, flush=True)
    if result.returncode:
        raise RuntimeError("Submission failed/ambiguous; inspect queue, never blindly retry")
    job = result.stdout.strip().split(";")[0]
    if not job.isdecimal():
        raise RuntimeError("Unparseable job ID")
    (root / "job-id.txt").write_text(job + "\n")
    print("SUBMITTED_JOB=" + job, "\nOUTPUT_ROOT=" + str(root), flush=True)


if __name__ == "__main__":
    main()
