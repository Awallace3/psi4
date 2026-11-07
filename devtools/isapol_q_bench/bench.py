#!/usr/bin/env python3
"""Benchmark-only controller/children. Never a production acceptance route."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
BASE_COMMIT = "a1e5231f1f5a21116caec91c03e143fe9728e1f4"
PROFILE_COMMIT = "8359155a10b8ebff4ba4995adc4f082400e801b5"
PROFILE_TREE = "dced1568f707951632bd5845be4cc4ff6099b829"
VALIDATION_SHA = "1582867ceb6f1ab0b8bd9f7d3ddeba443c35b745436713cf3e828e8abac0b3f8"
EVIDENCE_SHA = "44582da052154970c01f309d0f623299896559f1a1ce58dad1d8b42ba7b1e2f9"
CAND_ID = dict(core="676ea1e1b616a4c4df4f221688b31b8ae7a367539e9187676141b3109c85c013",
               isapol_python="2b7758f3eec48bb8cfc3565d50505ba1ce1fd911cbf1d762aab7773549d3713d")
FULL_ORDER = ("old", "candidate", "candidate", "old", "old", "candidate")
SUCCESS = "BENCHMARK_COMPLETE_NOT_ACCEPTED"
ALLOWED = {f"devtools/isapol_q_bench/{n}" for n in
           ("bench.py", "benchmark.sbatch", "test_bench.py", "README.md")}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


ops = module("bench_ops", HERE.parent/"isapol_q_omp/ops.py")
validation = module("bench_validation", HERE.parent/"isapol_q_omp/validate.py")
OLD, CAND = ops.OLD, validation.CANDIDATE
QUAL = OLD.parent/"ccasp-native-s66-grid400-1aaa9e0a30-installed001/payload/installed-validation.json"


def source_gate(args):
    ops.need(HERE == args.source/"devtools/isapol_q_bench", "running benchmark source differs")
    ops.need(ops.git(args.source, "rev-parse", "HEAD") == args.commit
             and ops.git(args.source, "rev-parse", "HEAD^{tree}") == args.tree,
             "benchmark source commit/tree differs")
    ops.need(not ops.git(args.source, "status", "--porcelain", "--untracked-files=all"),
             "benchmark source must be clean")
    changed = set(ops.git(args.source, "diff", "--name-only", BASE_COMMIT, args.commit).splitlines())
    ops.need(changed <= ALLOWED, "benchmark child changed frozen source/ops/tests")
    ops.need(ops.git(args.profiler_source, "rev-parse", "HEAD") == PROFILE_COMMIT
             and ops.git(args.profiler_source, "rev-parse", "HEAD^{tree}") == PROFILE_TREE
             and not ops.git(args.profiler_source, "status", "--porcelain", "--untracked-files=all"),
             "frozen835 profiler source differs")


def profile(args):
    return module("frozen_profile", args.profiler_source/"devtools/isapol_native_profile/profile.py")


def validation_gate(path):
    ops.need(ops.sha(path) == VALIDATION_SHA, "parent-approved passing validation receipt differs")
    value = ops.read(path)
    ops.need(value["status"] == "READY_DIAGNOSTIC_NOT_ACCEPTANCE"
             and value["acceptance"] is False and value["adoption"] is False
             and value["identity"]["implementation"] == CAND_ID
             and value["compiled_source"]["commit"] == validation.COMPILED_COMMIT
             and value["compiled_source"]["tree"] == validation.COMPILED_TREE
             and value["source"]["commit"] == BASE_COMMIT
             and value["evidence_sha256"] == EVIDENCE_SHA
             and all(value[k] is True for k in ("old_runtime_prechecked", "old_runtime_unchanged",
                         "candidate_prefix_unchanged", "compiled_tree_unchanged")),
             "candidate native/integrity validation differs")
    marker = ops.read(path.parent/"COMPLETED")
    ops.need(marker == dict(exit_code=0, acceptance=False, mode="validate")
             and not (path.parent/"FAILED").exists(), "candidate validation did not complete")
    junit = ops.junit_gate(path.parent/"junit.xml", False)
    ops.need(junit == value["tests"] and junit["cases"] == 82, "candidate82-case JUnit differs")
    before = ops.read(path.parent/"compiled-tree-before.json")
    ops.need(before == ops.read(path.parent/"compiled-tree-after.json"),
             "candidate validation snapshot pair differs")
    return value, before


def changes(a, b, path=()):
    if type(a) is not type(b):
        return [path]
    if isinstance(a, dict):
        if set(a) != set(b):
            return [path]
        return [p for k in sorted(a) for p in changes(a[k], b[k], path+(k,))]
    if isinstance(a, list):
        if len(a) != len(b):
            return [path]
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in changes(x, y, path+(i,))]
    return [] if a == b else [path]


def derive_claim(accepted, arm):
    """Fresh input only. Exactly two approved leaf replacements for candidate."""
    result = copy.deepcopy(accepted)
    path = ("keywords", "function_kwargs", "native_monomer", "implementation_sha256")
    request = result
    for key in path:
        request = request[key]
    ops.need(request == ops.OLD_ID, "accepted claim identity differs")
    if arm == "candidate":
        request.update(CAND_ID)
    expected = sorted([path+(k,) for k in CAND_ID]) if arm == "candidate" else []
    diff = changes(accepted, result)
    ops.need(sorted(diff) == expected, "benchmark derivation changed non-identity fields")
    return result, [dict(path=list(p), before=ops.OLD_ID[p[-1]], after=CAND_ID[p[-1]])
                    for p in diff]


def protected(args):
    return [args.source, args.profiler_source, args.compiled_source, OLD, CAND,
            args.fixtures, args.validation_result.parent, args.evidence.parent, QUAL.parent]


def ensure_threads(snapshot, count, affinity):
    ops.need(snapshot["process_threads"] == snapshot["native_mkl_max_threads"] == count,
             "requested Process/MKL thread count differs")
    ops.need(snapshot["affinity"] == affinity and len(affinity) >= 8, "fixed allocation affinity differs")
    for pool in snapshot.get("threadpools") or []:
        if pool.get("internal_api") == "openblas":
            ops.need(pool["num_threads"] == 1, "OpenBLAS is not single-threaded")
    for lib in snapshot["readonly_pool_queries"]:
        for name, number in lib["queries"].items():
            if "openblas" in name.lower():
                ops.need(number == 1, "mapped OpenBLAS is not single-threaded")


def imported_identity(psi4, pair, prefix):
    from psi4.driver.procrouting.isapol_schema import implementation_identity
    actual = implementation_identity()["sha256"]
    ops.need(actual == pair and Path(psi4.__file__).resolve().is_relative_to(prefix)
             and Path(psi4.core.__file__).resolve().is_relative_to(prefix)
             and Path(sys.prefix).resolve() == prefix, "foreign installed benchmark identity")
    return actual


def proxy(psi4, args):
    """Synthetic width-matched all-s fixture: NOT the molecular Q/grid."""
    import numpy as np
    oracle = module("frozen_q_fixture", args.compiled_source/ops.TEST)
    start = time.perf_counter_ns()
    aux = oracle.auxiliary(psi4.core.IsaBasisRepresentation.Spherical, 246)
    xyz = np.random.default_rng(4103).normal(size=(65543, 3))
    sites = [oracle.frozen_site(f"proxy-{i}", 4, xyz, [0], seed)
             for i, seed in enumerate((41, 97, 173))]
    arrays = {"points": xyz}
    for i, site in enumerate(sites):
        for key in ("weights", "shape", "shape_sum"):
            arrays[f"{i}-{key}"] = np.asarray(getattr(site.samples, key), dtype=np.float64)
    bits = {k: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
            for k, a in arrays.items()}
    fixture = dict(label="SYNTHETIC_ALL_S_WIDTH_MATCHED_NOT_MOLECULAR_Q", seed=4103,
                   site_seeds=[41, 97, 173], points=65543, sites=3, rank=4, aux=246,
                   screens=[[0]]*3, origins=[s.origin for s in sites], cutoff=1.e-36,
                   basis="frozen5ae auxiliary(Spherical, columns=246); 1centre,246s shells",
                   builder_sha256=ops.sha(args.compiled_source/ops.TEST),
                   basis_recipe=dict(representation="Spherical", centres=[[0., 0., 0.]],
                       shells=[dict(centre=0, l=0, exponents=[.3+i*.001],
                                    coefficients=[-1. if i % 3 == 0 else 1.]) for i in range(246)]),
                   array_bits_sha256=bits, q_shape=[75, 246])
    ops.write(args.root/"proxy-fixture.json", fixture)
    np.savez(args.root/"proxy-operands.npz", **arrays)
    preparation = (time.perf_counter_ns()-start)/1.e9
    expected = None
    timings = []
    for i in range(6):  # One untimed warm-up, five measured constructors.
        start = time.perf_counter_ns()
        q = psi4.core.IsaPartitionedMultipoles(aux, sites, "benchmark-only synthetic proxy", 1.e-36)
        seconds = (time.perf_counter_ns()-start)/1.e9
        values = np.ascontiguousarray(q.values.np)
        signature = dict(bits_sha256=hashlib.sha256(values.view(np.uint64).tobytes()).hexdigest(),
                         offsets=list(q.offsets), excluded=list(q.excluded_denominators),
                         negative=list(q.negative_ratios), shape=list(values.shape))
        if expected is None:
            expected = signature
            np.save(args.root/"proxy-Q.npy", values, allow_pickle=False)
        ops.need(signature == expected, "repeated frozen Q changed bits/diagnostics")
        if i:
            timings.append(seconds)
        del values, q
    return dict(fixture=fixture, signature=expected, constructor_seconds=timings,
                preparation_export_seconds=preparation, warmup_constructors=1,
                note="Constructor includes collocation/harmonics/screening/accumulation, not just accumulator.")


def water(psi4, args, p, normalized, claim, recorder, patches):
    from psi4 import schema_wrapper
    original = schema_wrapper._read_output
    reads = []

    def capture(path):
        text = original(path)
        name = f"psi4-output-{len(reads):03d}.txt"
        (args.root/name).write_text(text)
        reads.append(name)
        return text

    patches.attribute(schema_wrapper, "_read_output", capture, "benchmark_stdout_sidecar_only")
    p.install_hooks(patches, recorder, schema_wrapper)  # Frozen835 helper, unchanged.
    ops.write(args.root/"hooks.json", patches.inventory)
    with recorder.span("standalone_run_qcschema"):
        result = schema_wrapper.run_qcschema(copy.deepcopy(claim))
    plain = json.loads(result.model_dump_json() if hasattr(result, "model_dump_json") else result.json())
    ops.write(args.root/"qcschema-result.json", plain)
    envelope = p.preserve_exports(plain, args.root)  # Complete NPZ/envelope/Cn, no scientific trimming.
    expected = normalized["specification"]["keywords"]["function_kwargs"]["native_monomer"]
    ops.need(envelope["request"] == expected
             and envelope["result"]["provenance"]["response"] == "reference_h2h1",
             "computed request/response differs")
    return dict(captured_output_files=reads, status="COMPUTED_NOT_ACCEPTED",
                exported_tensors_sha256=envelope["result"]["tensors_sha256"])


def child(args):
    source_gate(args)
    validation_gate(args.validation_result)
    ops.fresh_paths(args.root, args.scratch_root, protected(args))
    ops.need(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_CPUS_PER_TASK") == "8",
             "scientific/native execution only in parent's finite allocation")
    for key, value in (("OMP_NUM_THREADS", str(args.threads)), ("MKL_NUM_THREADS", str(args.threads)),
                       ("OPENBLAS_NUM_THREADS", "1")):
        ops.need(os.environ.get(key) == value, "thread environment differs: "+key)
    args.root.mkdir(mode=0o700)
    args.scratch_root.mkdir(mode=0o700)
    cwd = args.scratch_root/"cwd"
    cwd.mkdir(mode=0o700)
    os.chdir(cwd)  # Before ANY Psi4 import, retained through core finalization.
    p = profile(args)
    normalized, envelope, metadata = p.read_fixtures(args.fixtures)
    ops.need(metadata["monomer_id"] == "monomer-0001", "only accepted water0001")
    accepted = ops.read(args.fixtures/"claimed-input.json")["function_kwargs"]["input_data"]
    claim, diff = derive_claim(accepted, args.arm)
    pair = ops.OLD_ID if args.arm == "old" else CAND_ID
    normalized = copy.deepcopy(normalized)
    normalized["specification"]["keywords"]["function_kwargs"]["native_monomer"]["implementation_sha256"] = dict(pair)
    ops.write(args.root/"derivation.json", dict(benchmark_only=True, acceptance=False,
              source_fixture_sha256=metadata["fixture_sha256"], exact_leaf_diff=diff,
              validation_sha256=VALIDATION_SHA, implementation=pair,
              protocols_unchanged=True, partition=[400, 2030], response=[100, 200], Drho=32))
    ops.write(args.root/"used-qcschema-input.json", claim)
    ops.write(args.root/"normalized-input.json", normalized)
    import psi4
    from psi4.driver.procrouting import isapol_schema, isapol_native_partition
    prefix = (OLD if args.arm == "old" else CAND)/"env"
    imported_identity(psi4, pair, prefix)
    ops.need(isapol_native_partition.DRHO_REFINEMENT_ITERATIONS == 32, "Drho cap differs")
    psi4.set_memory(20*1024**3, quiet=True)
    psi4.core.set_num_threads(args.threads, quiet=True)
    psi4.core.IOManager.shared_object().set_default_path(str(args.scratch_root)+"/")
    isapol_schema.admit_native_schema(normalized["specification"], normalized["molecule"])
    affinity = json.loads(args.affinity)
    initial = p.thread_snapshot(psi4)
    ensure_threads(initial, args.threads, affinity)
    sampler = p.telemetry.Sampler(args.scratch_root, args.root/"samples.jsonl", .5)
    recorder = p.telemetry.Recorder(args.root/"events.jsonl")
    patches = p.telemetry.Patches()
    outcome = dict(status="FAILED", acceptance=False, adoption=False, arm=args.arm,
                   task=args.task, threads=args.threads, initial_threads=initial, implementation=pair,
                   installed=dict(prefix=str(prefix), package=str(Path(psi4.__file__).resolve()),
                                  core=str(Path(psi4.core.__file__).resolve()),
                                  python=sys.executable, version=str(psi4.__version__)))
    def terminate(signum, unused):
        raise SystemExit(128+signum)
    handlers = {sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
    sampler.start()
    try:
        if args.task == "proxy":
            outcome.update(proxy(psi4, args))
        elif args.task == "water":
            outcome.update(water(psi4, args, p, normalized, claim, recorder, patches))
        else:
            ops.need(args.task == "warm", "unknown child task")
        outcome["final_threads"] = p.thread_snapshot(psi4)
        ensure_threads(outcome["final_threads"], args.threads, affinity)
        imported_identity(psi4, pair, prefix)
        outcome["status"] = "RETURNED_NOT_ACCEPTED"
    except BaseException as error:
        outcome["exception"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        patches.restore()
        recorder.close()
        outcome["telemetry"] = sampler.stop()
        if sampler.errors:
            outcome["status"] = "FAILED_TELEMETRY"
        ops.write(args.root/"summary.json", outcome)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    ops.need(not sampler.errors, "child telemetry failed")


def seal(args, p, expected=None):
    """Read-only qualified OLD and entire existing compiled CAND, pre/post."""
    source_gate(args)
    manifest = ops.old_seal()
    p.verify_receipts(OLD/"build-identity.json", QUAL)
    p.read_fixtures(args.fixtures)
    value, audited = validation_gate(args.validation_result)
    proof = validation.evidence_gate(args)
    validation.candidate_gate(args, proof, manifest)
    tree = validation.tree_snapshot(CAND, lambda: None, (args.compiled_source, ops.BASE))
    ops.need(tree == audited, "current compiled candidate differs from passed validation snapshot")
    if expected is not None:
        ops.need(tree == expected, "compiled candidate changed during benchmark")
    return tree


def run(args):
    source_gate(args)
    ops.fresh_paths(args.root, args.scratch_root, protected(args))
    ops.need(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_CPUS_PER_TASK") == "8",
             "fresh eight-CPU allocation required")
    allocation = ops.allocation("validate")  # Same approved8CPU/32GiB/2h embers shape.
    args.root.mkdir(parents=True, mode=0o700)
    ops.write(args.root/"owner.json", dict(token=args.launch_token, mode="benchmark"))
    args.scratch_root.mkdir(parents=True, mode=0o700)
    cwd = args.root/"cwd"
    cwd.mkdir(mode=0o700)
    os.chdir(cwd)
    p = profile(args)
    sampler = p.telemetry.Sampler(args.scratch_root, args.root/"independent-scratch.jsonl", .5)
    runner = ops.Runner(args.root, cwd, ops.environment(OLD/"env", args.scratch_root), 6900)
    original_check = runner.check
    def check():
        original_check()
        ops.need(not sampler.errors, "independent sampler failed")
        ops.need(max(sampler.peaks.get(k, 0) for k in ("logical_bytes", "allocated_bytes"))
                 <= 256*1024**3, "sampled accepted scratch byte cap exceeded")
    runner.check = check
    handlers = {sig: signal.signal(sig, runner.stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    result = dict(status="FAILED", acceptance=False, adoption=False,
                  launch_token=args.launch_token,
                  source=dict(path=str(args.source), commit=args.commit, tree=args.tree),
                  profiler_commit=PROFILE_COMMIT, validation_sha256=VALIDATION_SHA,
                  benchmark_sha256=ops.sha(HERE/"bench.py"),
                  telemetry_sha256=ops.sha(args.profiler_source/"devtools/isapol_native_profile/telemetry.py"),
                  candidate_identity=CAND_ID, allocation=allocation, full_order=list(FULL_ORDER),
                  fixed_operand_bits_equal=False, full_recompute_bitwise_expected=False,
                  historical_build_status="FAILED_UNCHANGED", historical_diagnostic001_status="FAILED_UNCHANGED")
    before = None
    sampler.start()
    try:
        before = seal(args, p)
        ops.write(args.root/"compiled-tree-before.json", before)
        affinity = sorted(os.sched_getaffinity(0))
        ops.need(len(affinity) >= 8, "insufficient allocation affinity")
        result["affinity"] = affinity
        shared = ["--source", args.source, "--commit", args.commit, "--tree", args.tree,
                  "--profiler-source", args.profiler_source, "--fixtures", args.fixtures,
                  "--validation-result", args.validation_result, "--compiled-source", args.compiled_source,
                  "--evidence", args.evidence, "--launch-token", args.launch_token,
                  "--affinity", json.dumps(affinity)]
        reference = fixture = None
        steps = [("warm-"+arm, arm, "warm", 8) for arm in ("old", "candidate")]
        steps += [(f"proxy-{n}-{arm}", arm, "proxy", n)
                  for n in (1, 2, 4, 8) for arm in ("old", "candidate")]
        steps += [(f"water-{i+1}-{arm}", arm, "water", 8) for i, arm in enumerate(FULL_ORDER)]
        for name, arm, task, threads in steps:
            runner.env = ops.environment((OLD if arm == "old" else CAND)/"env", args.scratch_root/name)
            runner.env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                              OPENBLAS_NUM_THREADS="1")
            command = [(OLD if arm == "old" else CAND)/"env/bin/python", "-I", "-B", "-X",
                       f"pycache_prefix={args.scratch_root/name}/cache", HERE/"bench.py", "child",
                       *shared, "--root", args.root/name, "--scratch-root", args.scratch_root/name,
                       "--arm", arm, "--task", task, "--threads", str(threads)]
            start = time.perf_counter_ns()
            runner.run(name, command)
            ops.write(args.root/(name+"-process-time.json"), dict(seconds=(time.perf_counter_ns()-start)/1.e9))
            item = ops.read(args.root/name/"summary.json")
            ops.need(item["status"] == "RETURNED_NOT_ACCEPTED", "child did not complete")
            if task == "proxy":
                if reference is None:
                    reference, fixture = item["signature"], item["fixture"]
                ops.need(item["signature"] == reference and item["fixture"] == fixture,
                         "OLD1 vs candidate/thread frozen proxy bits/operands/diagnostics differ")
        result["fixed_operand_bits_equal"] = True
        runner.check()
        result["status"] = SUCCESS
    except BaseException as error:
        result["exception"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        try:
            after = seal(args, p, before)
            ops.write(args.root/"compiled-tree-after.json", after)
            result["old_and_candidate_unchanged"] = True
            if runner.interrupted:
                result["status"] = "FAILED"
        except BaseException as error:
            result.update(status="FAILED", integrity_error=repr(error))
            raise
        finally:
            result["independent_telemetry"] = sampler.stop()
            if sampler.errors:
                result["status"] = "FAILED_TELEMETRY"
            ops.write(args.root/"result.json", result)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    ops.need(not runner.interrupted and not sampler.errors, "interrupted/incomplete benchmark")


def finalize(args):
    status = args.exit_code
    try:
        ops.need(ops.read(args.root/"owner.json") == dict(token=args.launch_token, mode="benchmark")
                 and args.root.stat().st_uid == os.getuid(), "foreign benchmark root")
    except Exception:
        return status or 1
    failed, completed = args.root/"FAILED", args.root/"COMPLETED"
    if failed.exists():
        previous = ops.read(failed)
        ops.need(type(previous["exit_code"]) is int and previous["exit_code"] != 0
                 and previous["mode"] == "benchmark" and previous["acceptance"] is False,
                 "invalid existing failure marker")
        status = status or previous["exit_code"]
    if status == 0:
        try:
            source_gate(args)
            value = ops.read(args.root/"result.json")
            ops.need(value["launch_token"] == args.launch_token and value["status"] == SUCCESS
                     and value["old_and_candidate_unchanged"] is True
                     and value["fixed_operand_bits_equal"] is True
                     and value["acceptance"] is False and value["adoption"] is False,
                     "benchmark did not finish successfully")
        except Exception:
            status = 1
    if status and completed.exists():
        archived = args.root/"COMPLETED_INTERRUPTED"
        if not archived.exists():
            os.link(completed, archived)
        else:
            ops.need(ops.sha(archived) == ops.sha(completed), "completion evidence differs")
        completed.unlink()
    target = failed if status else completed
    if not target.exists():
        ops.write(target, dict(exit_code=status, acceptance=False, mode="benchmark"))
    return status


def main():
    os.environ["GIT_OPTIONAL_LOCKS"] = "0"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("run", "child", "finalize"))
    for name in ("source", "commit", "tree", "root", "scratch-root", "profiler-source",
                 "fixtures", "validation-result", "evidence", "launch-token"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--compiled-source", default="/storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-native-q-omp003")
    parser.add_argument("--arm", choices=("old", "candidate"))
    parser.add_argument("--task", choices=("warm", "proxy", "water"))
    parser.add_argument("--threads", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--affinity")
    parser.add_argument("--exit-code", type=int)
    args = parser.parse_args()
    for name in ("source", "root", "scratch_root", "profiler_source", "fixtures",
                 "validation_result", "evidence", "compiled_source"):
        path = Path(getattr(args, name))
        ops.need(path.is_absolute(), "absolute CLI paths required")
        setattr(args, name, path.resolve())
    args.candidate_root, args.evidence_sha256 = CAND, EVIDENCE_SHA
    if args.operation == "finalize":
        return finalize(args)
    if args.operation == "child":
        child(args)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
