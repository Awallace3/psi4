"""One bounded embers campaign: two CPU48 samples, two H200 samples, CPU oracle."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from phoenix_timing import clean, digest, save
from protein83_timing import ROLES, resource_policy


def command(source, directory, role, parents):
    policy = resource_policy(role)
    if not parents or any(not str(job).isdecimal() for job in parents):
        raise ValueError("Explicit numeric prerequisite jobs required")
    argv = ["sbatch", "--parsable", "--job-name=fsapt-p83-" + role,
            "--account=gts-cs207-chemx", "--qos=embers",
            "--partition=" + policy["partition"], "--constraint=" + policy["constraint"],
            "--nodes=1", "--ntasks=1", "--cpus-per-task=" + str(policy["threads"]),
            "--mem=" + str(policy["memory_GiB"]) + "G",
            "--time=" + ("04:00:00" if role == "gpu" else "08:00:00"),
            "--dependency=afterany:" + ":".join(map(str, parents)), "--kill-on-invalid-dep=yes",
            "--no-requeue", "--signal=TERM@120", "--export=ALL",
            "--chdir=" + str(directory), "--output=" + str(directory / "logs/%x-%j.out"),
            "--error=" + str(directory / "logs/%x-%j.err")]
    if role == "gpu":
        argv.append("--gres=gpu:h200:1")
    return argv + [str(source / "devtools/fsapt_integration/phoenix_timing.sbatch")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    pin = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                   text=True).strip()
    clean(source, pin)
    build_root = args.build_root.resolve(strict=True)
    build = json.loads((build_root / "manifest.json").read_text())
    build_job = (build_root / "job-id.txt").read_text().strip()
    if (build["source_pin"] != pin or Path(build["source"]).resolve() != source
            or not build["prepare_gpu_runtime"]):
        raise RuntimeError("Same-source whole build with 48-thread oracle contract required")
    inputs = args.inputs.resolve(strict=True)
    if digest(inputs) != args.inputs_sha256:
        raise RuntimeError("Frozen timing suite changed")
    python = Path(build["read_only_inputs"]["DISP0_DEPENDENCIES"]) / "bin/python"
    if not python.is_file() or not args.run_root.is_absolute():
        raise RuntimeError("Pinned Python and absolute fresh campaign root required")
    root = args.run_root
    root.mkdir(mode=0o700)
    aliases = root / "sealed-workers"
    aliases.mkdir(mode=0o700)
    base = {"source": str(source), "pin": pin, "campaign": str(root),
            "build_root": str(build_root), "build_job": build_job,
            "build_manifest_sha256": digest(build_root / "manifest.json"),
            "inputs": str(inputs), "inputs_sha256": args.inputs_sha256,
            "alias_parent": str(aliases), "system": "protein83", "basis": "jun-cc-pvdz",
            "scf_E": 10, "scf_D": 12, "cache_bytes": 0, "block_size": 24,
            "timed_samples_per_arm": 2, "source_tracing": False,
            "sanitizers": "OUTSTANDING_LOADER_INCOMPATIBILITY",
            "cpu_baseline_threads": 48, "gpu_host_threads": 8,
            "host_runtime_memory_GiB": 96,
            "memory_estimates": {
                "primary_AO": 922, "JKFIT": 4762, "RIFIT": 3820,
                "in_core_DF_upper_bytes": 8*(4762*922**2 + 4762**2 + 48*922**2 + 3*922**2*7),
                "triangular_DF_upper_bytes": 8*(4762*922*923//2 + 4762**2),
                "scope": "Sizing only, not a full FSAPT peak-memory guarantee"},
            "time_policy": "One CPU repeat per 8h embers allocation; no automatic retries or inferno",
            "comparison_policy": "Different CPU resource classes, CPU local scratch/GPU shared scratch"}
    save(root / "campaign.json", base)
    env_base = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}
    # Validate every resource shape before the first actual submission.
    for role in ROLES:
        directory = root / role
        directory.mkdir()
        (directory / "logs").mkdir()
        argv = command(source, directory, role, [build_job])
        test = subprocess.run([argv[0], "--test-only", *argv[1:]], env=env_base,
                              text=True, capture_output=True, timeout=60)
        save(directory / "test-only.json", vars(test))
        print(role, test.stdout + test.stderr, flush=True)
        if test.returncode or resource_policy(role)["partition"] not in test.stdout + test.stderr:
            raise RuntimeError("Resource route rejected before campaign submission: " + role)
    submissions = {"status": "SUBMITTING", "jobs": {}}
    for role in ROLES:
        directory = root / role
        parents = ([submissions["jobs"][peer] for peer in ("cpu0", "cpu1", "gpu")]
                   if role == "compare" else [build_job])
        argv = command(source, directory, role, parents)
        pinned_producers = ({"producer_jobs": dict(submissions["jobs"]),
            "producer_manifest_sha256": {peer: digest(root / peer / "manifest.json")
                                        for peer in ("cpu0", "cpu1", "gpu")}}
                            if role == "compare" else {})
        save(directory / "manifest.json", {**base, **pinned_producers, "role": role,
             "resources": resource_policy(role), "command": argv})
        env = {**env_base, "FSAPT_SOURCE": str(source), "FSAPT_PYTHON": str(python),
               "FSAPT_RUN_ROOT": str(directory),
               "FSAPT_MANIFEST_SHA256": digest(directory / "manifest.json"),
               "FSAPT_DRIVER": str(source / "devtools/fsapt_integration/protein83_timing.py"),
               "FSAPT_THREADS": str(resource_policy(role)["threads"])}
        (directory / "SUBMISSION_ATTEMPTED").write_text("One scheduler call; no blind retry.\n")
        try:
            result = subprocess.run(argv, env=env, text=True, capture_output=True, timeout=60)
            save(directory / "submission.json", vars(result))
            print(role, result.stdout + result.stderr, flush=True)
            job = result.stdout.strip().split(";")[0]
            if result.returncode or not job.isdecimal():
                raise RuntimeError("Failed/ambiguous scheduler response")
            (directory / "job-id.txt").write_text(job + "\n")
            submissions["jobs"][role] = job
            save(root / "submissions.json", submissions)
            print("SUBMITTED", role, job, flush=True)
        except BaseException as error:
            submissions.update(status="PARTIAL_OR_AMBIGUOUS", failed_role=role,
                               error=f"{type(error).__name__}: {error}")
            save(root / "submissions.json", submissions)
            print("PARTIAL_SUBMISSIONS", submissions["jobs"], flush=True)
            raise
    submissions["status"] = "SUBMITTED"
    save(root / "submissions.json", submissions)
    print("CAMPAIGN_ROOT=" + str(root), flush=True)


if __name__ == "__main__":
    main()
