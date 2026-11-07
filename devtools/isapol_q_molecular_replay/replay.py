#!/usr/bin/env python3
"""One OLD molecular Q capture, exact constructor replay; NOT acceptance/timing."""
import argparse
import array
import ast
import hashlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import signal
import struct
import sys
import time
import zipfile
import importlib.util

HERE = Path(__file__).resolve().parent
BASE = "dc916e9549ce1d2438facc493d69d15a171c21b3"
BUILD_SHA = "ab8a01fef0bb715bf31e084ae7f9c56251b9cae1471f3d1b4389a795ebd041cf"
ALLOWED = {str(Path("devtools/isapol_q_molecular_replay")/n)
           for n in ("replay.py", "test_replay.py", "replay.sbatch")}
spec = importlib.util.spec_from_file_location("frozen_dc916", HERE.parent/"isapol_q_bench/bench.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
ops = b.ops
N, LIMIT, SUCCESS = 2429910, 256*1024**2, "MOLECULAR_Q_REPLAY_NOT_ACCEPTED"
FLOAT_NAMES = {"points", "weights", "shape", "shape_sum", "origins", "centres",
               "exponents", "coefficients", "cutoff", "Q"}


def need(ok, text):
    ops.need(ok, text)


def flatten(value):
    # ndarray.flat is a view/iterator, not a second full array.
    if hasattr(value, "flat"):
        yield from value.flat
    elif isinstance(value, (list, tuple, array.array)):
        for item in value:
            yield from flatten(item)
    else:
        yield value


def blocks(value):
    it = iter(flatten(value))
    while True:
        part = array.array("d", itertools.islice(it, 65536))
        if not part:
            break
        need(all(math.isfinite(x) for x in part), "nonfinite capture/replay value")
        if sys.byteorder != "little":
            part.byteswap()
        yield part.tobytes()


def bits(value):
    h, count = hashlib.sha256(), 0
    for chunk in blocks(value):
        h.update(chunk)
        count += len(chunk)
    return count, h.hexdigest()


def same(a, c):
    offset = 0
    for x, y in itertools.zip_longest(blocks(a), blocks(c)):
        if x != y:
            if x is None or y is None or len(x) != len(y):
                raise ValueError("constructor input/output uint64 bits differ in length")
            first = next(i for i in range(0, len(x), 8) if x[i:i+8] != y[i:i+8])
            raise ValueError(f"constructor input/output uint64 bits differ at flat index {(offset+first)//8}: "
                             f"actual {x[first:first+8].hex()}, reference {y[first:first+8].hex()}")
        offset += len(x)


def small_hex(value):
    # Only small floating output metadata (origins/cutoff), not grid data.
    return b"".join(blocks(value)).hex()


def header(shape):
    raw = repr(dict(descr="<f8", fortran_order=False, shape=tuple(shape))).encode("ascii")
    raw += b" "*((64-(10+len(raw)+1) % 64) % 64)+b"\n"
    return b"\x93NUMPY\x01\x00"+struct.pack("<H", len(raw))+raw


class CappedStream:
    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def write(self, data):
        need(self.stream.tell()+len(data) <= LIMIT, "incremental archive byte ceiling exceeded")
        return self.stream.write(data)


def archive_write(path, meta, arrays):
    """Streaming ZIP_STORED, bounded temporary chunks; exclusive creation."""
    need(set(arrays) == FLOAT_NAMES, "capture array inventory differs")
    need({k: list(v[0]) for k, v in arrays.items()} == expected_shapes(meta),
         "capture declared shape differs")
    declared = sum(math.prod(shape)*8 for shape, value in arrays.values())
    need(declared <= LIMIT-1024**2, "numeric capture budget exceeds archive ceiling")
    rows = {}
    with path.open("xb") as stream, zipfile.ZipFile(CappedStream(stream), "w", compression=zipfile.ZIP_STORED) as z:
        for name, (shape, value) in arrays.items():
            h, count = hashlib.sha256(), 0
            with z.open(name+".npy", "w") as member:
                member.write(header(shape))
                for chunk in blocks(value):
                    count += len(chunk)
                    need(count <= math.prod(shape)*8, "capture dimensions exceed declaration")
                    member.write(chunk)
                    h.update(chunk)
            need(count == math.prod(shape)*8, "capture dimensions disagree")
            rows[name] = dict(shape=list(shape), dtype="<f8", bytes=count, sha256=h.hexdigest())
        document = dict(meta, arrays=rows)
        payload = json.dumps(document, sort_keys=True, allow_nan=False).encode()
        need(len(payload) <= 1024**2, "capture metadata budget exceeded")
        z.writestr("manifest.json", payload)
    need(path.stat().st_size <= LIMIT, "archive byte ceiling exceeded")
    return ops.sha(path)


def expected_shapes(meta):
    need(set(meta["basis"]) == {"name", "origin", "representation", "shells"}, "unknown basis metadata")
    shells = meta["basis"]["shells"]
    need(0 < len(shells) <= 246, "invalid shell count")
    count = 0
    for shell in shells:
        need(set(shell) == {"centre", "l", "first", "count"}, "unknown shell metadata")
        need(type(shell["centre"]) is int and 0 <= shell["centre"] < 3
             and type(shell["l"]) is int and 0 <= shell["l"] <= 4
             and type(shell["first"]) is int and shell["first"] == count and type(shell["count"]) is int
             and shell["count"] > 0, "invalid shell primitive mapping")
        count += shell["count"]
    need(count <= 10000, "primitive budget exceeded")
    need(meta["basis"]["representation"] in ("Spherical", "Cartesian"), "unknown basis representation")
    spherical = meta["basis"]["representation"] == "Spherical"
    need(sum(2*s["l"]+1 if spherical else (s["l"]+1)*(s["l"]+2)//2 for s in shells) == 246,
         "declared AUX width differs")
    need(len(meta["sites"]) == 3, "expected water's three sites")
    labels = []
    for site in meta["sites"]:
        need(set(site) == {"label", "rank", "screen"}, "unknown site metadata")
        need(type(site["label"]) is str and site["label"] and site["rank"] == 4, "invalid site")
        labels.append(site["label"])
        screen = site["screen"]
        need(type(screen) is list and len(set(screen)) == len(screen)
             and all(type(x) is int and 0 <= x < 3 for x in screen), "invalid AUX screen")
    need(len(set(labels)) == 3, "duplicate site labels")
    need(type(meta["provenance"]) is str and meta["provenance"], "missing provenance")
    return dict(points=[N, 3], weights=[N], shape=[3, N], shape_sum=[N],
                origins=[3, 3], centres=[3, 3], exponents=[count], coefficients=[count],
                cutoff=[1], Q=[75, 246])


def archive_read(path, digest, materialize=True):
    """No pickle/extraction; validate all members even for metadata-only audit."""
    b.digest_literal(digest)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= LIMIT
         and ops.sha(path) == digest, "archive path/size/hash differs")
    # Bound the central-directory parser itself, before ZipFile materializes
    # ZipInfo objects. Our canonical writer has eleven entries, no ZIP comment,
    # split disks or ZIP64. Unknown containers fail instead of growing a heap.
    with path.open("rb") as stream:
        stream.seek(-22, os.SEEK_END)
        end = struct.unpack("<4s4H2IH", stream.read(22))
    need(end[0] == b"PK\x05\x06" and end[1] == end[2] == end[7] == 0
         and end[3] == end[4] == len(FLOAT_NAMES)+1 and end[5] <= 16384
         and end[6]+end[5]+22 == path.stat().st_size, "archive directory inventory/budget differs")
    arrays = {}
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        need(len(set(names)) == len(names) and set(names) == {"manifest.json"}|{n+".npy" for n in FLOAT_NAMES},
             "duplicate/unknown/missing archive member")
        need(all(i.compress_type == zipfile.ZIP_STORED and not i.flag_bits & 1 for i in infos)
             and sum(i.file_size for i in infos) <= LIMIT, "archive compression/encryption/byte budget differs")
        need(z.getinfo("manifest.json").file_size <= 1024**2, "manifest budget exceeded")
        meta = json.loads(z.read("manifest.json"))
        need(meta["schema"] == "molecular-q-capture/1" and meta["acceptance"] is False
             and meta["implementation"] == ops.OLD_ID and meta["build_result_sha256"] == BUILD_SHA,
             "foreign capture identity/schema")
        need(set(meta["arrays"]) == FLOAT_NAMES, "manifest array inventory differs")
        shapes = expected_shapes(meta)
        for name, shape in shapes.items():
            row = meta["arrays"][name]
            nbytes = math.prod(shape)*8
            need(row["shape"] == shape and row["dtype"] == "<f8" and row["bytes"] == nbytes,
                 "manifest shape/dtype/byte budget differs")
            with z.open(name+".npy") as f:
                need(f.read(8) == b"\x93NUMPY\x01\x00", "unsupported NPY format")
                size = struct.unpack("<H", f.read(2))[0]
                need(size <= 512, "NPY header budget exceeded")
                hdr = ast.literal_eval(f.read(size).decode("ascii").strip())
                need(hdr == dict(descr="<f8", fortran_order=False, shape=tuple(shape)), "NPY header differs")
                need(z.getinfo(name+".npy").file_size == 10+size+nbytes, "NPY payload size differs")
        # ALL inventory, dimensions/products, headers and aggregate budgets have
        # passed before any numeric array allocation. Then bounded chunk reads.
        for name, shape in shapes.items():
            row = meta["arrays"][name]
            nbytes = math.prod(shape)*8
            h, values, count = hashlib.sha256(), array.array("d"), 0
            with z.open(name+".npy") as f:
                f.read(8)
                size = struct.unpack("<H", f.read(2))[0]
                f.read(size)
                while True:
                    chunk = f.read(65536*8)
                    if not chunk:
                        break
                    need(len(chunk) % 8 == 0, "truncated float64 member")
                    h.update(chunk)
                    count += len(chunk)
                    part = array.array("d")
                    part.frombytes(chunk)
                    if sys.byteorder != "little":
                        part.byteswap()
                    need(all(math.isfinite(x) for x in part), "nonfinite archived value")
                    if materialize:
                        values.extend(part)
            need(count == nbytes and h.hexdigest() == row["sha256"], "array payload checksum differs")
            if materialize:
                arrays[name] = values
    return meta, arrays


def metadata(q):
    result = {name: getattr(q, name) for name in (
        "offsets", "ranks", "labels", "components", "excluded_denominators", "negative_ratios",
        "provenance", "representation", "frame", "units")}
    # Floating metadata is compared through the same binary codec, not == on -0.
    result["origins_bits"] = small_hex(q.origins)
    result["cutoff_bits"] = small_hex([q.denominator_cutoff])
    return json.loads(json.dumps(result))


class ReturnObserver:
    """Observe one successful original code-object return; never replace a call."""
    def __init__(self, target, capture):
        self.code, self.capture, self.count = target.__code__, capture, 0
        self.error = None

    def observe(self, frame, event, value):
        if frame.f_code is self.code and event == "return" and value is not None:
            sys.setprofile(None)  # Before getter copies/I/O; no retained frame.
            self.count += 1
            try:
                self.capture(frame.f_locals, value)
            except Exception as error:
                # Do not retain exception/traceback/frame. Even if the scientific
                # driver catches a callback error, it cannot publish success.
                self.error = dict(type=type(error).__name__, message=str(error)[:1000])

    def __enter__(self):
        need(sys.getprofile() is None, "existing profile callback must not be replaced")
        sys.setprofile(self.observe)
        return self

    def __exit__(self, typ, value, tb):
        sys.setprofile(None)
        if typ is None:
            need(self.error is None and self.count == 1,
                 "exactly one successful original Q adapter capture required: "+str(self.error))


def capture(locals_, returned, path, fixture_hashes, code_sha):
    q, values, charge_error = returned
    aux, sites = locals_["auxiliary"], locals_["qsites"]
    points, weights, sampled, total = (locals_[n] for n in ("points", "weights", "sampled", "total"))
    need(len(points) == len(weights) == len(total) == N and len(sites) == 3
         and tuple(sampled.shape) == (3, N), "unexpected molecular capture dimensions")
    need(locals_["rank"] == 4 and tuple(values.shape) == (75, 246), "unexpected molecular Q dimensions")
    need(len(aux.centres) == 3 and locals_["response_aux"].nfunction == 246, "unexpected AUX dimensions")
    rows = []
    for i, site in enumerate(sites):
        s = site.samples
        # STL getters are owned copies. Fetch/check/release one field at a time.
        for name, expected in (("points", points), ("weights", weights),
                               ("shape", sampled[i]), ("shape_sum", total)):
            same(getattr(s, name), expected)
        rows.append(dict(label=site.label, rank=site.rank, screen=list(s.auxiliary_sites)))
    output_meta = metadata(q)
    need(output_meta["origins_bits"] == small_hex([s.origin for s in sites])
         and output_meta["cutoff_bits"] == small_hex([locals_["cutoff"]]), "original Q floating metadata differs")
    same(q.values.np, values)
    shells, exponents, coefficients = [], [], []
    for shell in aux.shells:
        shells.append(dict(centre=shell.centre, l=shell.l, first=len(exponents), count=len(shell.exponents)))
        exponents.extend(shell.exponents)
        coefficients.extend(shell.coefficients)
    meta = dict(schema="molecular-q-capture/1", acceptance=False,
                implementation=ops.OLD_ID, build_result_sha256=BUILD_SHA, fixture_sha256=fixture_hashes,
                basis=dict(name=aux.name, origin=aux.origin, representation=aux.representation, shells=shells),
                shell_layout=locals_["response_aux"].shell_layout(), sites=rows,
                provenance=locals_["provenance"], original_output=output_meta,
                adapter_code_sha256=code_sha,
                charge_error_observed=charge_error)
    shapes = expected_shapes(meta)
    data = dict(points=points, weights=weights, shape=sampled, shape_sum=total,
                origins=[s.origin for s in sites], centres=aux.centres, exponents=exponents,
                coefficients=coefficients, cutoff=[locals_["cutoff"]], Q=values)
    return archive_write(path, meta, {name: (shapes[name], value) for name, value in data.items()})


def reconstruct(core, meta, data):
    shells = []
    for row in meta["basis"]["shells"]:
        shell = core.IsaGaussianShell()
        first, end = row["first"], row["first"]+row["count"]
        shell.centre, shell.l = row["centre"], row["l"]
        shell.exponents, shell.coefficients = data["exponents"][first:end], data["coefficients"][first:end]
        same(shell.exponents, data["exponents"][first:end])
        same(shell.coefficients, data["coefficients"][first:end])
        need(shell.centre == row["centre"] and shell.l == row["l"], "native shell order/centre/rank differs")
        shells.append(shell)
    centres = [list(data["centres"][i:i+3]) for i in range(0, 9, 3)]
    aux = core.IsaExplicitBasis(core.IsaBasisRole.MolecularAux,
                               getattr(core.IsaBasisRepresentation, meta["basis"]["representation"]), centres, shells)
    need(aux.nfunction == 246 and aux.shell_layout() == meta["shell_layout"], "reconstructed AUX layout differs")
    points = [list(data["points"][i:i+3]) for i in range(0, 3*N, 3)]
    weights, total = data["weights"].tolist(), data["shape_sum"].tolist()
    sites = []
    for i, row in enumerate(meta["sites"]):
        s = core.IsaMultipoleSamples()
        s.points, s.weights = points, weights
        s.shape, s.shape_sum = data["shape"][i*N:(i+1)*N].tolist(), total
        s.auxiliary_sites = row["screen"]
        item = core.IsaMultipoleSite()
        item.label, item.rank, item.origin = row["label"], row["rank"], list(data["origins"][3*i:3*i+3])
        item.samples = s
        # Validate actual reconstructed native inputs, not just the archive.
        native = item.samples
        for field, expected in (("points", points), ("weights", weights),
                                ("shape", data["shape"][i*N:(i+1)*N]), ("shape_sum", total)):
            same(getattr(native, field), expected)
        need(list(native.auxiliary_sites) == row["screen"], "reconstructed screen differs")
        sites.append(item)
    return aux, sites


def capture_ready(directory, digest, fixture_hashes):
    item = ops.read(directory/"summary.json")
    need(item["status"] == "RETURNED_NOT_ACCEPTED" and item["arm"] == "old" and item["threads"] == 8
         and item["implementation"] == ops.OLD_ID and item["capture_sha256"] == digest
         and item["capture_adapter_calls"] == 1 and item["fixture_sha256"] == fixture_hashes
         and "exception" not in item and item["acceptance"] is item["adoption"] is False,
         "fixture unusable: entire OLD capture must succeed")
    schema = ops.read(directory/"qcschema-result.json")
    envelope = ops.read(directory/"server-native-envelope.json")
    need(schema["success"] is True and schema["extras"]["psi4:native_monomer"] == envelope
         and envelope["implementation"]["sha256"] == ops.OLD_ID
         and ops.sha(directory/"server-tensors.npz") == envelope["result"]["tensors_sha256"]
         == item["exported_tensors_sha256"], "capture full-result transport differs")
    return item


def adapter_span(path):
    events = [json.loads(line) for line in path.read_text().splitlines()]
    begin = [e for e in events if e["event"] == "begin" and e["name"] == "isapol_distribution._stockholder_q"]
    end = [e for e in events if e["event"] == "end" and e["name"] == "isapol_distribution._stockholder_q"]
    need(len(begin) == len(end) == 1 and end[0]["status"] == "returned"
         and begin[0]["id"] == end[0]["id"], "exactly one successful original adapter span required")
    return 1


def saved_capture_Q(path):
    # Called only AFTER archive_read validated every size/member/hash.
    with zipfile.ZipFile(path) as z:
        return z.read("Q.npy")


def source_gate(args):
    need(HERE == args.source/"devtools/isapol_q_molecular_replay", "foreign diagnostic source")
    need(ops.git(args.source, "rev-parse", "HEAD") == args.commit
         and ops.git(args.source, "rev-parse", "HEAD^{tree}") == args.tree
         and ops.git(args.source, "rev-parse", "--show-toplevel") == str(args.source)
         and not ops.git(args.source, "status", "--porcelain", "--untracked-files=all"), "unclean/unbound source")
    need(ops.git(args.source, "merge-base", BASE, args.commit) == BASE
         and set(ops.git(args.source, "diff", "--name-only", BASE, args.commit).splitlines()) <= ALLOWED,
         "diagnostic changed frozen source/helpers")
    need(args.build_result_sha256 == BUILD_SHA, "parent-approved literal build digest required")
    need(ops.git(b.PROFILE_SOURCE, "rev-parse", "HEAD") == b.PROFILE_COMMIT
         and ops.git(b.PROFILE_SOURCE, "rev-parse", "HEAD^{tree}") == b.PROFILE_TREE
         and not ops.git(b.PROFILE_SOURCE, "status", "--porcelain", "--untracked-files=all"), "frozen835 differs")
    ops.source_gate(b.COMPILED_SOURCE, b.BASE_COMMIT, b.COMPILED_TREE)


def seal(args, p, expected=None):
    source_gate(args)
    p.verify_receipts(b.OLD/"build-identity.json", b.QUAL)
    p.read_fixtures(b.FIXTURES)
    b.build_gate(args.build_result_sha256)  # Actual normal84 build/files/dependencies and OLD.
    current = b.tree_snapshot(b.CAND)
    need(expected is None or current == expected, "candidate tree changed")
    return current


def protected(args):
    return b.protected(args)+[b.COMPILED_SOURCE.parent/"psi4.ccasp-native-q-lowthreads-bench001"]


def enter_child(args):
    """All paths/preflights are stdlib; owned CWD precedes ANY Psi4 import."""
    source_gate(args)
    build = b.build_gate(args.build_result_sha256)
    ops.fresh_paths(args.root, args.scratch_root, protected(args))
    need(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_CPUS_PER_TASK") == "8", "allocation required")
    need(sys.flags.isolated and sys.dont_write_bytecode and sys.pycache_prefix
         and Path(sys.pycache_prefix).resolve().is_relative_to(args.scratch_root), "owned isolated cache required")
    for k, v in (("OMP_NUM_THREADS", str(args.threads)), ("MKL_NUM_THREADS", str(args.threads)),
                 ("OPENBLAS_NUM_THREADS", "1")):
        need(os.environ.get(k) == v, "thread environment differs")
    args.root.mkdir(mode=0o700)
    args.scratch_root.mkdir(mode=0o700)
    (args.scratch_root/"cwd").mkdir(mode=0o700)
    os.chdir(args.scratch_root/"cwd")
    return build


def child(args):
    build = enter_child(args)
    p = b.profile(args)
    normalized, envelope, fixture = p.read_fixtures(b.FIXTURES)
    need(fixture["monomer_id"] == "monomer-0001", "only accepted water0001")
    pair = ops.OLD_ID if args.arm == "old" else build["identity"]["implementation"]
    prefix = (b.OLD if args.arm == "old" else b.CAND)/"env"
    import psi4
    b.imported_identity(psi4, pair, prefix)
    if args.arm == "candidate":
        need(str(psi4.__version__) == build["identity"]["version"]
             and Path(psi4.core.__file__).resolve() == Path(build["identity"]["core"]).resolve(), "candidate version/core differs")
    psi4.set_memory(20*1024**3, quiet=True)
    psi4.core.set_num_threads(args.threads, quiet=True)
    psi4.core.IOManager.shared_object().set_default_path(str(args.scratch_root)+"/")
    from psi4.driver.procrouting import isapol_distribution as distribution
    from psi4.driver.procrouting import isapol_native_partition as partition
    need(partition.DRHO_REFINEMENT_ITERATIONS == 32, "Drho cap differs")
    affinity = json.loads(args.affinity)
    initial = p.thread_snapshot(psi4)
    b.ensure_threads(initial, args.threads, affinity)
    sampler = p.telemetry.Sampler(args.scratch_root, args.root/"samples.jsonl", .5)
    recorder, patches = p.telemetry.Recorder(args.root/"events.jsonl"), p.telemetry.Patches()
    outcome = dict(status="FAILED", acceptance=False, adoption=False, arm=args.arm, threads=args.threads,
                   implementation=pair, initial_threads=initial,
                   installed=dict(prefix=str(prefix), package=str(Path(psi4.__file__).resolve()),
                                  core=str(Path(psi4.core.__file__).resolve()), version=str(psi4.__version__)))
    def terminate(signum, unused):
        raise SystemExit(128+signum)
    handlers = {sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
    sampler.start()
    try:
        if args.operation == "capture":
            need(args.arm == "old" and args.threads == 8, "capture only unchanged OLD8")
            target = distribution._stockholder_q  # ORIGINAL body before835 installs wrappers.
            code_sha = hashlib.sha256(inspect.getsource(target).encode()).hexdigest()
            archive = args.root/"molecular-q-input.npz"
            def snapshot(local, returned):
                outcome["capture_sha256"] = capture(local, returned, archive, fixture["fixture_sha256"], code_sha)
            claim = ops.read(b.FIXTURES/"claimed-input.json")["function_kwargs"]["input_data"]
            need(claim == b.derive_claim(claim, "old", build["identity"]["implementation"])[0], "OLD claim differs")
            from psi4.driver.procrouting import isapol_schema
            isapol_schema.admit_native_schema(normalized["specification"], normalized["molecule"])
            ops.write(args.root/"used-qcschema-input.json", claim)
            ops.write(args.root/"normalized-input.json", normalized)
            outcome["adapter_code_sha256"] = code_sha
            with ReturnObserver(target, snapshot) as observer:
                outcome.update(b.water(psi4, args, p, normalized, claim, recorder, patches))
            need(observer.count == 1 and distribution._stockholder_q is not None, "capture count differs")
            # Only the original adapter body is observed; hooks restored in finally below.
            archive_read(archive, outcome["capture_sha256"], materialize=False)
            outcome["fixture_sha256"] = fixture["fixture_sha256"]
        else:
            need(args.operation == "replay", "unknown child operation")
            archive = args.root.parent/"capture/molecular-q-input.npz"
            capture_item = capture_ready(archive.parent, args.capture_sha256, fixture["fixture_sha256"])
            meta, data = archive_read(archive, args.capture_sha256)
            need(meta["fixture_sha256"] == fixture["fixture_sha256"], "capture accepted fixture differs")
            need(meta["adapter_code_sha256"] == capture_item["adapter_code_sha256"], "capture code binding differs")
            aux, sites = reconstruct(psi4.core, meta, data)
            start = time.perf_counter_ns()
            q = psi4.core.IsaPartitionedMultipoles(aux, sites, meta["provenance"], data["cutoff"][0])
            outcome["constructor_call_including_marshalling_seconds"] = (time.perf_counter_ns()-start)/1.e9
            values, info = q.values.np, metadata(q)  # Getter clones: fetch once.
            # Preserve actual output and metadata EVEN on mismatch; no fresh SCF,
            # denominator, shapes or grid. Comparison never repairs the result.
            with (args.root/"Q.npy").open("xb") as stream:
                stream.write(header((75, 246)))
                for chunk in blocks(values):
                    stream.write(chunk)
            outcome.update(capture_sha256=args.capture_sha256, output_metadata=info,
                           Q_bits_sha256=bits(values)[1], fixed_bits_equal=False)
            same(values, data["Q"])
            need(info == meta["original_output"], "molecular Q metadata/counters differ")
            outcome["fixed_bits_equal"] = True
        outcome["final_threads"] = p.thread_snapshot(psi4)
        b.ensure_threads(outcome["final_threads"], args.threads, affinity)
        b.imported_identity(psi4, pair, prefix)
        outcome["status"] = "RETURNED_NOT_ACCEPTED"
    except BaseException as error:
        outcome.update(status="FAILED", exception=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        patches.restore()
        recorder.close()
        outcome["telemetry"] = sampler.stop()
        try:
            if sampler.errors:
                outcome["status"] = "FAILED_TELEMETRY"
            if args.operation == "capture" and outcome["status"] == "RETURNED_NOT_ACCEPTED":
                need(distribution._stockholder_q is target, "original adapter restoration required")
                outcome["capture_adapter_calls"] = adapter_span(args.root/"events.jsonl")
        except BaseException as error:
            outcome.update(status="FAILED", capture_postcheck_error=repr(error))
            raise
        finally:
            ops.write(args.root/"summary.json", outcome)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    need(not sampler.errors, "child telemetry failed")


def run(args):
    source_gate(args)
    ops.fresh_paths(args.root, args.scratch_root, protected(args))
    need(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_CPUS_PER_TASK") == "8", "allocation required")
    allocation = ops.allocation("oracle")
    args.root.mkdir(parents=True, mode=0o700)
    args.scratch_root.mkdir(parents=True, mode=0o700)
    ops.write(args.root/"owner.json", dict(token=args.launch_token, mode="molecular-replay"))
    (args.root/"cwd").mkdir(mode=0o700)
    os.chdir(args.root/"cwd")
    p = b.profile(args)
    sampler = p.telemetry.Sampler(args.scratch_root, args.root/"independent-scratch.jsonl", .5)
    runner = ops.Runner(args.root, args.root/"cwd", {}, 6600)
    original_check = runner.check
    def check():
        original_check()
        need(not sampler.errors and max(sampler.peaks.get(k, 0) for k in ("logical_bytes", "allocated_bytes"))
             <= 256*1024**3, "independent scratch sampler/cap failed")
    runner.check = check
    handlers = {sig: signal.signal(sig, runner.stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    result = dict(status="FAILED", acceptance=False, adoption=False, launch_token=args.launch_token,
                  source=dict(path=str(args.source), commit=args.commit, tree=args.tree),
                  build_result_sha256=BUILD_SHA, compiled_source=b.compiled_source(), allocation=allocation,
                  diagnostic_memory_allowance_bytes=4*1024**3, capture_timing_is_instrumented=True,
                  diagnostic_memory_allowance_is_estimate_not_heap_bound=True,
                  constructor_timing_is_not_speed_result=True, fixed_bits_equal=False)
    before = None
    sampler.start()
    try:
        before = seal(args, p)
        ops.write(args.root/"compiled-tree-before.json", before)
        result["candidate_identity"] = b.build_gate(BUILD_SHA)["identity"]["implementation"]
        affinity = sorted(os.sched_getaffinity(0))
        need(len(affinity) >= 8, "insufficient affinity")
        result["affinity"] = affinity
        shared = ["--source", args.source, "--commit", args.commit, "--tree", args.tree,
                  "--build-result-sha256", BUILD_SHA, "--launch-token", args.launch_token,
                  "--affinity", json.dumps(affinity)]
        digest = None
        steps = [("capture", "old", 8)] + [(f"replay-{n}-{arm}", arm, n)
                for n in (1, 2, 4, 8) for arm in ("old", "candidate")]
        for name, arm, n in steps:
            runner.check()
            prefix = (b.OLD if arm == "old" else b.CAND)/"env"
            runner.env = ops.environment(prefix, args.scratch_root/name)
            runner.env.update(OMP_NUM_THREADS=str(n), MKL_NUM_THREADS=str(n), OPENBLAS_NUM_THREADS="1")
            command = [prefix/"bin/python", "-I", "-B", "-X", f"pycache_prefix={args.scratch_root/name}/cache",
                       HERE/"replay.py", "capture" if name == "capture" else "replay", *shared,
                       "--root", args.root/name, "--scratch-root", args.scratch_root/name,
                       "--arm", arm, "--threads", str(n)]
            if digest:
                command += ["--capture-sha256", digest]
            start = time.monotonic()
            global_deadline = runner.deadline
            runner.deadline = min(global_deadline, time.monotonic()+600)
            try:
                runner.run(name, command)
            finally:
                runner.deadline = global_deadline
            ops.write(args.root/(name+"-process-time.json"), dict(seconds=time.monotonic()-start))
            item = ops.read(args.root/name/"summary.json")
            need(item["status"] == "RETURNED_NOT_ACCEPTED", "incomplete child")
            if name == "capture":
                digest = item["capture_sha256"]
                captured, unused = archive_read(args.root/name/"molecular-q-input.npz", digest, materialize=False)
                reference_Q = saved_capture_Q(args.root/name/"molecular-q-input.npz")
            else:
                need(item["fixed_bits_equal"] is True and item["capture_sha256"] == digest, "replay failed")
                raw = (args.root/name/"Q.npy").read_bytes()
                prefix = header((75, 246))
                need(raw == reference_Q and raw.startswith(prefix) and len(raw) == len(prefix)+75*246*8
                     and hashlib.sha256(raw[len(prefix):]).hexdigest()
                     == item["Q_bits_sha256"] == captured["arrays"]["Q"]["sha256"]
                     and item["output_metadata"] == captured["original_output"],
                     "controller saved Q/metadata mismatch")
        runner.check()
        result.update(status=SUCCESS, capture_sha256=digest, fixed_bits_equal=True)
    except BaseException as error:
        result["exception"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        try:
            after = seal(args, p, before)
            ops.write(args.root/"compiled-tree-after.json", after)
            result["old_and_candidate_unchanged"] = before is not None
        except BaseException as error:
            result.update(status="FAILED", integrity_error=repr(error))
            raise
        finally:
            result["independent_telemetry"] = sampler.stop()
            if runner.interrupted or sampler.errors:
                result["status"] = "FAILED"
            ops.write(args.root/"result.json", result)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    need(not runner.interrupted and not sampler.errors, "interrupted diagnostic")


def finalize(args):
    # Small owned diagnostic marker only; no normal-build/qualification adoption.
    status = args.exit_code
    try:
        need(ops.read(args.root/"owner.json") == dict(token=args.launch_token, mode="molecular-replay")
             and args.root.stat().st_uid == os.getuid(), "foreign output root")
    except Exception:
        return status or 1
    failed, done = args.root/"FAILED", args.root/"COMPLETED"
    if failed.exists():
        previous = ops.read(failed)
        need(previous["mode"] == "molecular-replay" and previous["acceptance"] is False
             and type(previous["exit_code"]) is int and previous["exit_code"] != 0, "invalid failure marker")
        status = status or previous["exit_code"]
    if status == 0:
        try:
            source_gate(args)
            value = ops.read(args.root/"result.json")
            need(value["launch_token"] == args.launch_token and value["status"] == SUCCESS
                 and value["build_result_sha256"] == BUILD_SHA
                 and value["fixed_bits_equal"] and value["old_and_candidate_unchanged"]
                 and value["acceptance"] is value["adoption"] is False, "diagnostic incomplete")
        except Exception:
            status = 1
    if status and done.exists():
        archived = args.root/"COMPLETED_INTERRUPTED"
        if not archived.exists():
            os.link(done, archived)
        else:
            need(ops.sha(archived) == ops.sha(done), "completion evidence differs")
        done.unlink()
    target = failed if status else done
    if not target.exists():
        ops.write(target, dict(exit_code=status, acceptance=False, mode="molecular-replay"))
    return status


def main():
    os.environ["GIT_OPTIONAL_LOCKS"] = "0"
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("operation", choices=("run", "capture", "replay", "finalize"))
    for name in ("source", "commit", "tree", "root", "scratch-root", "build-result-sha256", "launch-token"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--arm", choices=("old", "candidate"))
    parser.add_argument("--threads", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--affinity")
    parser.add_argument("--capture-sha256")
    parser.add_argument("--exit-code", type=int)
    args = parser.parse_args()
    need(args.build_result_sha256 == BUILD_SHA, "parent-approved explicit build digest required")
    for name in ("source", "root", "scratch_root"):
        path = Path(getattr(args, name))
        need(path.is_absolute() and not path.is_symlink(), "absolute non-symlink CLI path required")
        setattr(args, name, path.resolve())
    args.profiler_source, args.fixtures, args.compiled_source = b.PROFILE_SOURCE, b.FIXTURES, b.COMPILED_SOURCE
    if args.operation == "finalize":
        need(type(args.exit_code) is int, "finalizer exit code required")
        return finalize(args)
    if args.operation == "run":
        run(args)
    else:
        need(args.arm is not None and args.threads is not None and args.affinity, "child flags required")
        child(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
