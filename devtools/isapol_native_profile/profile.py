#!/usr/bin/env python3
"""Standalone exact-input profile, never a QCArchive client or scientific acceptance."""
import argparse
import ast
import base64
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import sys

# -I deliberately removes script-directory imports. Load ONLY this adjacent helper.
_helper = Path(__file__).resolve().with_name("telemetry.py")
_spec = importlib.util.spec_from_file_location("_native_profile_telemetry", _helper)
telemetry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(telemetry)

SCIENCE_COMMIT = "1aaa9e0a306373b42a3decc25c7dfa106dbc8ea2"
SCIENCE_TREE = "8f4705983f81bc5acf42625192ece9b0379c8209"
IMPLEMENTATION = dict(
    core="ef69cddb2dd7b675688a733e4c5dd68b78fc78ac35b77b363114d638ac9dae44",
    isapol_python="e1476e2d0b8e395c40b48219ca8bc8def9e69035cab3ab970f6b88940bebc74b")
RECEIPTS = {
    "build-identity.json": "d2b4049c2e5818e462ad3c6ac36f93d6a9623a6a205c701b885887ec2a9f3bfd",
    "installed-validation.json": "a9f7179ec4721a79b462a9516639d474f9ca952b7adbe89b4069438b4ed8e8dc",
    "imported-identity.json": "b8f5f5afcc3ffb372cc9524406159327c9bdb6894e8bb5deabf275781d264e78",
    "junit.xml": "06ca24994e7fa11a0c5b4ab9d463068872f4b02b3554e5ab94965d84a62fd188",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def canonical_input(raw):
    """Only translate schema layout, never scientific/protocol values."""
    result = copy.deepcopy(raw)
    if "specification" in result:
        require(set(result) <= {"molecule", "specification", "schema_name", "schema_version"},
                "expected copied runtime/QCSchema input, not a campaign receipt")
        result.setdefault("schema_name", "qcschema_atomic_input")
        result.setdefault("schema_version", 2)
        require(result["schema_name"] == "qcschema_atomic_input" and result["schema_version"] == 2,
                "nested input must be QCSchema v2")
    else:
        require(result.get("schema_name") == "qcschema_input" and result.get("schema_version") == 1,
                "expected copied runtime, QCSchema v1 or v2")
        molecule = result.pop("molecule")
        result.pop("schema_name")
        result.pop("schema_version")
        require(set(result) <= {"driver", "model", "keywords", "protocols", "extras"},
                "unexpected flattened input fields")
        result = dict(schema_name="qcschema_atomic_input", schema_version=2,
                      molecule=molecule, specification=result)
    # QCSchema molecule versions differ, but no physical fields are altered.
    molecule = result["molecule"]
    molecule["schema_name"], molecule["schema_version"] = "qcschema_molecule", 2
    return result


def validate_case(raw, envelope, metadata):
    require(metadata.get("schema") == "native-profile-case/1", "case metadata schema required")
    monomer = metadata["monomer_id"]
    require(monomer in ("monomer-0001", "monomer-0002", "monomer-0025"),
            "only explicitly accepted 0001/0002/0025 fixtures permitted")
    expected_memory = 60 if monomer == "monomer-0025" else 20
    require(type(metadata["science_memory_gib"]) is int
            and metadata["science_memory_gib"] == expected_memory, "accepted science memory differs")
    data = canonical_input(raw)
    spec = data["specification"]
    request = spec["keywords"]["function_kwargs"]["native_monomer"]
    require(spec["driver"] == "properties"
            and spec["model"]["method"].lower() == "pbe0"
            and spec["model"]["basis"].lower() == "aug-cc-pvtz", "exact accepted method/basis required")
    require(envelope["schema"] == "native-monomer-artifacts/2"
            and request == envelope["request"], "accepted envelope/request differs")
    require(request["implementation_sha256"] == IMPLEMENTATION
            and envelope["implementation"]["sha256"] == IMPLEMENTATION,
            "qualified grid400 implementation required")
    require(envelope["result"]["provenance"]["response"] == "reference_h2h1",
            "accepted H2H1 response required")
    require(request["resources"]["max_bytes"] <= expected_memory * 1024**3,
            "numeric budget exceeds accepted science memory")
    return data


def read_fixtures(directory):
    directory = Path(directory).resolve()
    inventory = load(Path(__file__).with_name("fixtures.json"))
    require(directory.name in inventory, "only accepted 0001/0002/0025 fixture directories permitted")
    pins = inventory[directory.name]
    for name, expected in pins.items():
        require(digest(directory / name) == expected, "copied accepted fixture changed: " + name)
    case, outcome = (load(directory / name) for name in ("case.json", "pilot-outcome.json"))
    raw = load(directory / "expected-runtime.json")
    envelope = load(directory / "server-native-envelope.json")
    claim = load(directory / "claimed-input.json")
    claimed = claim["function_kwargs"]["input_data"]
    require(claim["function"] == "qcengine.compute" and claim["function_kwargs"]["program"] == "psi4",
            "unexpected claimed execution path")
    # QCElemental added defaults and stripped fix_symmetry on the original v1
    # claim. The native request independently restores computational C1.
    for key, value in raw["molecule"].items():
        if key == "fix_symmetry" and key not in claimed["molecule"]:
            require(value == "c1", "only the known v1 C1 metadata omission is allowed")
        else:
            require(claimed["molecule"].get(key) == value, "claimed molecule differs: " + key)
    for key, value in raw["specification"].items():
        require(claimed[key] == value, "claimed specification differs: " + key)
    require(case["monomer_id"] == outcome["monomer_id"] == directory.name
            and outcome["scientific_build_commit"] == SCIENCE_COMMIT
            and case["execution_profile"] == outcome["execution_profile"]
            and case["intended_memory_bytes"] == case["execution_profile"]["science_memory_GiB"] * 1024**3,
            "copied case/outcome science identity differs")
    require(case["descriptor_dimensions"]["drho_refinement_iterations"] == 32,
            "accepted descriptor is not Drho32")
    require(envelope["result"]["tensors_sha256"] == pins["server-tensors.npz"],
            "accepted tensor checksum differs")
    metadata = dict(schema="native-profile-case/1", monomer_id=directory.name,
                    science_memory_gib=case["execution_profile"]["science_memory_GiB"],
                    accepted_task_id=outcome["task_id"], accepted_job_id=outcome["allocation"]["JobId"],
                    input_sha256=pins["expected-runtime.json"],
                    accepted_envelope_sha256=pins["server-native-envelope.json"],
                    fixture_sha256=pins, accepted_dimensions=case["descriptor_dimensions"])
    return validate_case(raw, envelope, metadata), envelope, metadata


def verify_receipts(build_path, qualification_path):
    paths = {"build-identity.json": Path(build_path),
             "installed-validation.json": Path(qualification_path)}
    for name in ("imported-identity.json", "junit.xml"):
        paths[name] = Path(qualification_path).with_name(name)
    for name, path in paths.items():
        require(digest(path) == RECEIPTS[name], "qualified receipt changed: " + name)
    manifest = load(build_path)
    require(manifest["commit"] == SCIENCE_COMMIT and manifest["tree"] == SCIENCE_TREE,
            "scientific build must be exactly 1aaa")
    package = Path(manifest["installed_package"]).resolve()
    # Full scientific installed-file set, including SCF seal/iterator files.
    for name, expected in manifest["installed_files"].items():
        require(digest(package / name) == expected, "installed file changed: " + name)
    return manifest, load(paths["imported-identity.json"])


def source_audit(package, expected_harness):
    """Read actual deployed QCEngine producer; do not import/execute it."""
    package = Path(package).resolve()
    result = {"qualification": "Source scope evidence only; not a timing breakdown.",
              "package": str(package), "files": {}}
    selections = {"util.py": ("compute_wrapper", "handle_output_metadata"),
                  "compute.py": ("compute",), "programs/psi4.py": ("compute",)}
    for relative, names in selections.items():
        path = package / relative
        text = path.read_text()
        tree = ast.parse(text)
        functions = [node for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
        snippets = {}
        for node in functions:
            snippets[node.name] = dict(first_line=node.lineno, last_line=node.end_lineno,
                                       source="\n".join(text.splitlines()[node.lineno-1:node.end_lineno]))
        require(len(snippets) == len(names), "wall-time producer functions missing: " + relative)
        result["files"][relative] = dict(path=str(path), sha256=digest(path), functions=snippets)
    require(result["files"]["programs/psi4.py"]["sha256"] == expected_harness,
            "deployed harness differs from copied case metadata")
    util = result["files"]["util.py"]["functions"]
    wrapper = util["compute_wrapper"]["source"]
    metadata = util["handle_output_metadata"]["source"]
    compute = result["files"]["compute.py"]["functions"]["compute"]["source"]
    compute_tree = ast.parse(compute)
    wrapped_executor = any(
        any(isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Name)
            and item.context_expr.func.id == "compute_wrapper" for item in node.items)
        and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name) and call.func.value.id == "executor"
                and call.func.attr == "compute" for call in ast.walk(node))
        for node in ast.walk(compute_tree) if isinstance(node, ast.With))
    # Conservatively identify the familiar producer, retaining complete source for review.
    recognized = ('metadata["wall_time"] = time.time() - comp_time' in wrapper
                  and "yield metadata" in wrapper
                  and wrapper.index("comp_time = time.time()") < wrapper.index("yield metadata")
                  < wrapper.index('metadata["wall_time"]')
                  and 'provenance_augments["wall_time"] = metadata["wall_time"]' in metadata
                  and wrapped_executor)
    result["recognized_qcengine_wrapper_timer"] = recognized
    result["attribution"] = (
        "QCEngine wrapper wall clock includes executor and wrapper setup; not native-stage time. "
        "Review saved source for retries, launch/transport and exact context boundaries."
        if recognized else "UNRESOLVED: inspect captured producer source; no scope assertion.")
    return result


def thread_snapshot(psi4):
    result = dict(process_threads=psi4.core.get_num_threads(),
                  native_mkl_max_threads=psi4.core._isa_blas_max_threads(),
                  affinity=sorted(os.sched_getaffinity(0)),
                  environment={k: os.environ.get(k) for k in (
                      "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES")},
                  process=telemetry.process_snapshot())
    result["cpu_model"] = next((line.split(":", 1)[1].strip()
                                for line in Path("/proc/cpuinfo").read_text().splitlines()
                                if line.startswith("model name")), None)
    result["blas_mapped_libraries"] = sorted({
        line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
        if any(token in line.lower() for token in ("mkl", "openblas", "blis", "libgomp", "libiomp", "libomp"))})
    try:
        from threadpoolctl import threadpool_info
        result["threadpools"] = threadpool_info()
    except ImportError:
        result["threadpools"] = None
        result["threadpool_limit"] = "threadpoolctl unavailable; no installation or pool mutation attempted"
    # Query only libraries ALREADY mapped: no new backend or initializer loaded.
    # Covers minimal qualified environments without optional threadpoolctl.
    import ctypes
    result["readonly_pool_queries"] = []
    symbols = ("mkl_get_max_threads", "MKL_Get_Max_Threads", "openblas_get_num_threads",
               "openblas_get_num_threads64_", "scipy_openblas_get_num_threads",
               "scipy_openblas_get_num_threads64_", "omp_get_max_threads")
    for path in result["blas_mapped_libraries"]:
        query = dict(path=path, queries={})
        try:
            library = ctypes.CDLL(path, mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
            for symbol in symbols:
                try:
                    getter = getattr(library, symbol)
                except AttributeError:
                    continue
                getter.argtypes, getter.restype = [], ctypes.c_int
                query["queries"][symbol] = int(getter())
        except OSError as error:
            query["error"] = str(error)
        result["readonly_pool_queries"].append(query)
    return result


def install_hooks(patches, recorder, schema):
    prefix = "psi4.driver.procrouting."
    modules = {name: importlib.import_module(prefix + "isapol_" + name) for name in (
        "logging", "schema", "molecular", "bounded", "bounded_response", "distribution",
        "native_partition", "native_factors", "pfit_stream")}
    telemetry.install_stages(patches, modules["logging"].StageLog, recorder)
    targets = {
        "schema": ("execute_native_request",),
        "molecular": ("run_molecular_properties", "export_molecular_result"),
        "distribution": ("isa_moments", "_stockholder_q"),
        "native_partition": ("native_partition", "final_shape_samples"),
        "native_factors": ("native_plain_df_operators", "native_constrained_ov"),
        "pfit_stream": ("fitted_point_targets", "refine_streamed"),
        "bounded_response": ("_kernel", "_assemble", "_stability"),
    }
    # Patch imported aliases as well as definitions, with ONE wrapper per original.
    for module_name, names in targets.items():
        module = modules[module_name]
        for name in names:
            original = getattr(module, name)
            label = "isapol_" + module_name + "." + name

            probes = {}
            observations = {}
            if module_name == "native_partition" and name == "native_partition":
                probes = dict(drho="Drho-C fit", density=None,
                              samples="fixed density grid sampling",
                              controller=("controller construction/cache", "controller cache release"),
                              declared=None, initial="controller initialization",
                              trajectory="ISA controller iteration/final tails",
                              provenance=None)
                observations = dict(
                    density=lambda local: dict(
                        drho_refinement_iterations=local["drho"].refinement_iterations,
                        drho_relative_residual=local["drho"].relative_residual),
                    declared=lambda local: dict(
                        prepared_cache_enabled=local["controller"].prepared_cache_enabled,
                        iteration_grid_points=len(local["pts"]), sites=len(local["recipe"].sites)))
            elif module_name == "pfit_stream" and name == "fitted_point_targets":
                probes = dict(provider="point potential construction P",
                              packed="point response contraction/packing")
                observations = dict(packed=lambda local: dict(
                    potential_shape=list(local["potentials"].shape),
                    potential_bytes=local["potentials"].nbytes))

            def wrap(function, tag, boundaries, metadata):
                def measured(*args, **kwargs):
                    with recorder.span(tag):
                        if boundaries:
                            with telemetry.line_intervals(function, recorder, boundaries, metadata):
                                return function(*args, **kwargs)
                        return function(*args, **kwargs)
                return measured

            replacement = wrap(original, label, probes, observations)
            for other in modules.values():
                for alias, value in list(vars(other).items()):
                    if value is original:
                        patches.attribute(other, alias, replacement, other.__name__ + "." + alias)
    response = modules["bounded_response"]
    for cls, names in ((response._H2H1Response, ("__init__", "solve")),
                       (response.BoundedResponse, ("__enter__", "prepare", "solve"))):
        for name in names:
            patches.wrap(cls, name, recorder)
    proc = importlib.import_module(prefix + "proc")
    patches.wrap(proc, "scf_helper", recorder, "ordinary_scf_helper")
    original = schema.methods_dict_["properties"]

    def ordinary(*args, **kwargs):
        with recorder.span("ordinary_properties_driver"):
            return original(*args, **kwargs)

    patches.mapping(schema.methods_dict_, "properties", ordinary, "schema.properties_dispatch")
    require(modules["native_partition"].DRHO_REFINEMENT_ITERATIONS == 32, "Drho32 changed")
    return modules


def preserve_exports(result, output):
    require(result.get("success") is True, "QCSchema did not succeed; raw result retained")
    envelope = result["extras"]["psi4:native_monomer"]
    artifact = envelope["artifacts"]["tensors.npz"]
    raw = base64.b64decode(artifact["data"], validate=True)
    require(len(raw) == artifact["size"] and hashlib.sha256(raw).hexdigest() == artifact["sha256"]
            == envelope["result"]["tensors_sha256"], "tensor transport checksum differs")
    telemetry.write_json(output / "server-native-envelope.json", envelope)
    telemetry.write_json(output / "native-result.json", envelope["result"])
    with (output / "server-tensors.npz").open("xb") as stream:
        stream.write(raw)
    return envelope


def enter_owned_cwd(scratch):
    """Keep CWD owned through interpreter shutdown, including import failures.

    Psi4 finalizers write relative files even for check-only. Do NOT restore the
    caller's CWD on return/exception: atexit/core finalization happens afterward.
    """
    cwd = scratch / "cwd"
    cwd.mkdir(mode=0o700)
    require(cwd.stat().st_uid == os.getuid(), "profiling CWD must be owned by this user")
    with (cwd / ".owned-profile-cwd").open("x") as stream:
        stream.write("Owned profiling CWD retained through interpreter finalization.\n")
    os.chdir(cwd)
    return cwd


def run(args):
    # Resolve before changing CWD, including paths revisited by final checks.
    for name in ("fixtures", "build_manifest", "qualification", "qcengine_package", "output", "scratch"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    output, scratch = Path(args.output).resolve(), Path(args.scratch).resolve()
    require(not output.exists() and not scratch.exists(), "fresh output AND scratch paths required")
    inputs = [Path(p).resolve() for p in (args.fixtures, args.build_manifest, args.qualification,
                                         args.qcengine_package)]
    require(not output.is_relative_to(scratch) and not scratch.is_relative_to(output),
            "output and scratch must be disjoint")
    for path in inputs:
        require(not path.is_relative_to(output) and not path.is_relative_to(scratch),
                "read-only input cannot be inside new output/scratch")
        protected = path if path.is_dir() else path.parent
        require(not output.is_relative_to(protected) and not scratch.is_relative_to(protected),
                "output/scratch cannot be inside fixtures, sealed build, receipts or QCEngine")
    require(sys.dont_write_bytecode and sys.pycache_prefix
            and Path(sys.pycache_prefix).resolve().is_relative_to(scratch),
            "launch with -I -B -X pycache_prefix=NEW_SCRATCH/cache")
    require(0.1 <= args.sample_seconds <= 60, "sample interval must be 0.1..60 seconds")
    data, accepted_envelope, metadata = read_fixtures(args.fixtures)
    actual_input = load(Path(args.fixtures) / "claimed-input.json")["function_kwargs"]["input_data"]
    manifest, imported = verify_receipts(args.build_manifest, args.qualification)
    require(Path(sys.prefix).resolve() == Path(args.build_manifest).resolve().parent / "env",
            "must launch the qualified build's installed interpreter")
    expected_cli = Path(args.build_manifest).resolve().parent / "env/bin/psi4"
    require(shutil.which("psi4") is not None
            and Path(shutil.which("psi4")).resolve() == expected_cli.resolve(),
            "PATH must select the qualified science CLI, never the production/manager launcher")
    source = source_audit(args.qcengine_package, args.qcengine_harness_sha256)
    require(source["recognized_qcengine_wrapper_timer"], "wall-time producer scope unresolved")
    if args.run:
        require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_CPUS_PER_TASK"),
                "scientific execution permitted only in parent's fresh compute allocation")
        require(int(os.environ["SLURM_CPUS_PER_TASK"]) >= args.threads, "insufficient allocated CPUs")
    for name, expected in (("OMP_NUM_THREADS", str(args.threads)), ("MKL_NUM_THREADS", str(args.threads)),
                           ("OPENBLAS_NUM_THREADS", "1")):
        require(os.environ.get(name) == expected, "set " + name + "=" + expected + " before launch")
    output.mkdir(parents=True)
    scratch.mkdir(parents=True)
    os.environ.update(TMPDIR=str(scratch), SCRATCH=str(scratch), PSI_SCRATCH=str(scratch))
    cwd = enter_owned_cwd(scratch)  # MUST precede any Psi4 import, including check-only.
    import psi4
    from psi4 import schema_wrapper
    from psi4.driver.procrouting import isapol_schema, isapol_native_partition
    require(Path(psi4.__file__).resolve() == Path(imported["package"]).resolve()
            and Path(psi4.core.__file__).resolve() == Path(imported["core"]).resolve(),
            "foreign imported science package/core")
    require(isapol_schema.implementation_identity()["sha256"] == IMPLEMENTATION, "imported identity changed")
    require(isapol_native_partition.DRHO_REFINEMENT_ITERATIONS == 32, "Drho32 changed")
    psi4.set_memory(metadata["science_memory_gib"] * 1024**3, quiet=True)
    psi4.core.set_num_threads(args.threads, quiet=True)
    psi4.core.IOManager.shared_object().set_default_path(str(scratch) + "/")
    # Descriptor admission only, before any ordinary SCF is possible.
    isapol_schema.admit_native_schema(data["specification"], data["molecule"])
    require(len(os.sched_getaffinity(0)) >= args.threads, "insufficient CPU affinity")
    start = dict(schema="native-stage-profile/1", status="ADMITTED_NOT_EXECUTED",
                 qualification="Instrumented standalone profile, NOT scientific/operational acceptance.",
                 scientific_commit=SCIENCE_COMMIT, implementation=IMPLEMENTATION,
                 input_sha256=metadata["input_sha256"], case_metadata=metadata,
                 normalized_input_sha256=hashlib.sha256(json.dumps(
                     data, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                 harness_sha256=digest(__file__), telemetry_sha256=digest(_helper),
                 fixture_inventory_sha256=digest(Path(__file__).with_name("fixtures.json")),
                 threads=args.threads, science_memory_gib=metadata["science_memory_gib"],
                 instrumentation=args.instrumentation, protocols_unchanged=True,
                 execution_input="Exact claimed-input.json:function_kwargs.input_data, unchanged.",
                 owned_cwd=str(cwd),
                 cwd_lifecycle="Not restored; retained through interpreter/core finalization.",
                 output_capture="Separate _read_output copy; input protocols/verbosity unchanged.",
                 declarations=dict(partition_grid=[400, 2030], response_grid=[100, 200],
                                   drho_refinement_iterations=32, response="reference_h2h1"),
                 allocation={k: os.environ.get(k) for k in (
                     "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURMD_NODENAME")},
                 launch_argv=sys.argv, initial_threads=thread_snapshot(psi4))
    start["psi4_cli"] = str(expected_cli)
    telemetry.write_json(output / "manifest.json", start)
    telemetry.write_json(output / "wall-time-source.json", source)
    telemetry.write_json(output / "normalized-input.json", data)
    telemetry.write_json(output / "used-qcschema-input.json", actual_input)
    if not args.run:
        return
    recorder = telemetry.Recorder(output / "events.jsonl")
    patches = telemetry.Patches()
    sampler = telemetry.Sampler(scratch, output / "samples.jsonl", args.sample_seconds)
    original_output = schema_wrapper._read_output
    output_reads = []

    def capture(path):
        text = original_output(path)
        name = "psi4-output-%03d.txt" % len(output_reads)
        with (output / name).open("x") as stream:
            stream.write(text)
        output_reads.append(name)
        return text  # Never change stdout or its protocols.

    patches.attribute(schema_wrapper, "_read_output", capture, "schema._read_output_sidecar")
    try:
        if args.instrumentation == "stages":
            install_hooks(patches, recorder, schema_wrapper)
    except BaseException:
        patches.restore()
        recorder.close()
        raise
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signals = {sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
    summary = dict(status="FAILED_OR_INTERRUPTED", acceptance=False)
    sampler.start()
    try:
        telemetry.write_json(output / "hooks.json", patches.inventory)
        with recorder.span("standalone_run_qcschema"):
            result = schema_wrapper.run_qcschema(copy.deepcopy(actual_input))
        # QCSchema properties may contain ndarrays in a Python model dump.
        plain = json.loads(result.model_dump_json() if hasattr(result, "model_dump_json") else result.json())
        telemetry.write_json(output / "qcschema-result.json", plain)
        envelope = preserve_exports(plain, output)
        require(envelope["request"] == data["specification"]["keywords"]["function_kwargs"]["native_monomer"],
                "exported scientific request differs")
        require(envelope["result"]["provenance"]["response"] == "reference_h2h1", "response changed")
        summary.update(status="COMPUTED_NOT_ACCEPTED",
                       resources=envelope["result"]["resources"],
                       partition=envelope["result"]["provenance"]["partition"],
                       final_threads=thread_snapshot(psi4))
    except BaseException as error:
        summary["exception"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        patches.restore()
        for sig, handler in signals.items():
            signal.signal(sig, handler)
        summary["telemetry"] = sampler.stop()
        if sampler.errors:
            summary["status"] = "FAILED_TELEMETRY"
        summary["captured_output_files"] = output_reads
        recorder.close()
        try:
            verify_receipts(args.build_manifest, args.qualification)
            read_fixtures(args.fixtures)
            require(isapol_schema.implementation_identity()["sha256"] == IMPLEMENTATION,
                    "science identity changed during profile")
            summary["installed_science_unchanged"] = True
        except Exception as error:
            summary.update(status="FAILED_IDENTITY", identity_error=repr(error))
            telemetry.write_json(output / "summary.json", summary)
            raise
        summary["final_process"] = telemetry.process_snapshot()
        telemetry.write_json(output / "summary.json", summary)
        require(not sampler.errors, "telemetry sampling failed; result is not a complete profile")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    for name in ("fixtures", "build-manifest", "qualification", "qcengine-package",
                 "qcengine-harness-sha256", "scratch", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--threads", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--instrumentation", choices=("stages", "off"), default="stages")
    parser.add_argument("--sample-seconds", type=float, default=1.)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
