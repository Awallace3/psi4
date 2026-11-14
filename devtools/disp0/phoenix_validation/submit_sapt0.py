"""One explicit matched timing job after a successful coherent build."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from sapt0_timing import ACCEPTANCE, SYSTEMS, save, sha
from submit import clean_environment
from sapt0_hardware import PROFILES
from sealed_h200 import CONTRACT, require_cpu_admission


def route_matches(fields, profile, account, qos, hours):
    # Phoenix appends its default GPU partition. A typed GRES still prevents
    # allocation on V100 hardware; do not admit arbitrary alternate partitions.
    partitions = set(fields.get("Partition", "").split(","))
    return (
        fields.get("Account") == account and fields.get("QOS") == qos
        and profile["partition"] in partitions
        and partitions <= {profile["partition"], "gpu-v100"}
        and fields.get("TresPerNode") == "gres/" + profile["gres"]
        and fields.get("TimeLimit") == f"{hours:02}:00:00"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pin", required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--hours", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--device-family", choices=PROFILES, default="H200")
    parser.add_argument("--qos", choices=("embers", "inferno"), default="embers")
    parser.add_argument("--account", choices=("gts-cs207", "gts-cs207-chemx"),
                        default="gts-cs207-chemx")
    parser.add_argument("--inferno-approved", action="store_true")
    parser.add_argument("--identity-contract", choices=("fresh-bytes-v1", CONTRACT),
                        default=CONTRACT)
    parser.add_argument("--runtime-ab", action="store_true",
                        help="Also measure the unchanged fresh-byte path on this same node")
    args = parser.parse_args()
    if args.qos == "inferno" and not args.inferno_approved:
        raise ValueError("Explicit per-job inferno approval required")
    if args.runtime_ab and args.identity_contract != CONTRACT:
        raise ValueError("A/B requires the sealed candidate and explicit fresh-byte control")
    if args.device_family == "RTX_PRO_BLACKWELL" and args.system != "peptide":
        raise ValueError("Blackwell admission is limited to the water/peptide pilot")
    profile = PROFILES[args.device_family]
    source = args.source.resolve(strict=True)
    run = args.run_root
    if not run.is_absolute():
        raise ValueError("Absolute fresh run root required")
    if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                               text=True).strip() != args.pin:
        raise ValueError("Source is not pinned")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"],
                               text=True).strip():
        raise ValueError("Source is dirty")
    build_root, inputs = args.build_root.resolve(strict=True), args.inputs.resolve(strict=True)
    job = (build_root / "job-id.txt").read_text().strip()
    build_run = build_root / ("job-" + job)
    status = (build_run / "job-status.env").read_text()
    if "exit_code=0\n" not in status or "term_received=0\n" not in status:
        raise RuntimeError("Build failed or was interrupted")
    proof_path = build_run / "result.json"
    proof = json.loads(proof_path.read_text())
    if proof["status"] != "PASS":
        raise RuntimeError("Build qualification missing")
    if args.identity_contract == CONTRACT:
        if proof["pin"] != args.pin or Path(proof["source"]).resolve() != source:
            raise RuntimeError("Sealed workers require the exact proof-authenticated helper checkout")
        for path, checksum in proof["provenance_sha256"].items():
            if sha(path) != checksum:
                raise RuntimeError("Proof-authenticated helper changed before submission: " + path)
    admission_sha = (require_cpu_admission(build_run, proof["pin"])
                     if args.identity_contract == CONTRACT else None)
    # Helper-only evolution is allowed; scientific code must match the native build.
    subprocess.run(["git", "-C", str(source), "diff", "--exit-code", proof["pin"],
                    args.pin, "--", "psi4/", "psi4_runtime_support/"], check=True)
    input_hashes = {name: sha(inputs / (name + ".json")) for name in ("water", args.system)}
    env = clean_environment(os.environ)
    env.update(SAPT0_SOURCE=str(source), SAPT0_PIN=args.pin, SAPT0_BUILD_ROOT=str(build_root),
               SAPT0_INPUTS=str(inputs), SAPT0_SYSTEM=args.system, SAPT0_RUN_ROOT=str(run),
               SAPT0_PYTHON=proof["dependency_environment"] + "/bin/python")
    run.mkdir(mode=0o700)
    (run / "logs").mkdir()
    argv = ["sbatch", "--parsable", "--job-name=sapt0-" + args.system,
            "--account=" + args.account, "--partition=" + profile["partition"], "--qos=" + args.qos,
            "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--gres=" + profile["gres"],
            "--mem=160G", f"--time={args.hours:02}:00:00", "--exclusive",
            "--no-requeue", "--signal=TERM@90", "--chdir=" + str(run),
            "--output=" + str(run / "logs/%x-%j.out"),
            "--error=" + str(run / "logs/%x-%j.err"),
            "--export=ALL", str(source / "devtools/disp0/phoenix_validation/sapt0_job.sbatch")]
    save(run / "manifest.json", {
        "source_pin": args.pin, "source": str(source), "build_proof": str(proof_path),
        "build_proof_sha256": sha(proof_path), "native_pin": proof["pin"],
        "input_hashes": input_hashes, "command": argv, "qos": args.qos,
        "account": args.account, "device_family": args.device_family,
        "hardware_profile": profile, "inferno_approved": args.inferno_approved,
        "identity_contract": args.identity_contract,
        "runtime_ab": args.runtime_ab,
        "cpu_sealed_admission_sha256": admission_sha,
        "diagnostics": ["cold_cprofile", "warm_energy"] if args.identity_contract == CONTRACT else [],
        "system": args.system, "cpu_threads_each_arm": 8,
        "repeats_each_arm": 3, "gate": "water CPU/GPU first",
        "acceptance": dict(ACCEPTANCE),
        "environment": {k: v for k, v in env.items() if k.startswith("SAPT0_")}})
    test = subprocess.run([argv[0], "--test-only", *argv[1:]], env=env, capture_output=True,
                          text=True, timeout=60)
    save(run / "test-only.json", dict(returncode=test.returncode, stdout=test.stdout, stderr=test.stderr))
    print(test.stdout + test.stderr, flush=True)
    if test.returncode:
        raise RuntimeError("Exact route rejected; nothing submitted")
    with (run / "SUBMISSION_ATTEMPTED").open("x") as stream:
        stream.write("One scheduler call. Do not retry an ambiguous submission.\n")
    submitted = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=60)
    save(run / "submission.json", dict(returncode=submitted.returncode,
                                     stdout=submitted.stdout, stderr=submitted.stderr))
    print(submitted.stdout + submitted.stderr, flush=True)
    if submitted.returncode:
        raise RuntimeError("Submission failed/ambiguous; do not retry")
    job = submitted.stdout.strip().split(";")[0]
    if not job.isdecimal():
        raise RuntimeError("Unknown scheduler response")
    with (run / "job-id.txt").open("x") as stream:
        stream.write(job + "\n")
    print("SUBMITTED_JOB=" + job, flush=True)
    effective = subprocess.run(["scontrol", "show", "job", "-o", job],
                               capture_output=True, text=True, timeout=30)
    save(run / "effective-route.json", dict(returncode=effective.returncode,
         stdout=effective.stdout, stderr=effective.stderr))
    if effective.returncode:
        raise RuntimeError("Submitted, but route readback failed; do not resubmit")
    fields = dict(item.split("=", 1) for item in effective.stdout.split() if "=" in item)
    if not route_matches(fields, profile, args.account, args.qos, args.hours):
        raise RuntimeError("Submitted route differs; inspect receipt, do not resubmit")
    print("VERIFIED_ROUTE=" + " ".join(f"{key}={fields.get(key)}" for key in
          ("Account", "QOS", "Partition", "TimeLimit", "JobState", "StartTime")), flush=True)


if __name__ == "__main__":
    main()
