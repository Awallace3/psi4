"""Private developer input guards; fixture bytes only, no download or compiler."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import zipfile

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "devtools/disp0"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("provision_sdk_inputs", TOOLS / "provision_sdk_inputs.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path.remove(str(TOOLS))


class UncompressedFixture:
    def stream_reader(self, stream):
        return stream


def archive(item, corruption=None):
    index = {k: item[k] for k in ("name", "version", "build")}
    index["subdir"] = "linux-64"
    if corruption == "index":
        index["version"] = "0"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as zipstream:
        for label in ("info", "pkg"):
            data = io.BytesIO()
            with tarfile.open(fileobj=data, mode="w") as stream:
                members = {"info/index.json": json.dumps(index).encode()} if label == "info" else {
                    "targets/x86_64-linux/include/" + item["header"]: b"// exact fixture\n"}
                if corruption == "traversal" and label == "pkg":
                    members["../../escape"] = b"no"
                if corruption == "absolute" and label == "pkg":
                    members["/escape"] = b"no"
                for name, value in members.items():
                    entry = tarfile.TarInfo(name)
                    entry.size = len(value)
                    stream.addfile(entry, io.BytesIO(value))
                if label == "pkg":
                    for directory in ("lib", "targets/x86_64-linux/lib"):
                        entry = tarfile.TarInfo(f"{directory}/lib{item['family']}.so")
                        entry.type = tarfile.SYMTYPE
                        target = (("../targets/x86_64-linux/lib/" if directory == "lib" else "")
                                  + item["runtime_file"])
                        if corruption == "soname":
                            target = item["soname"]
                        elif corruption == "link-escape":
                            target = "/outside/" + item["runtime_file"]
                        elif corruption == "wrong-link-dir":
                            target = "../../../outside/" + item["runtime_file"]
                        entry.linkname = target
                        stream.addfile(entry)
            zipstream.writestr(f"{label}-fixture.tar.zst", data.getvalue())
    value = output.getvalue()
    item = {**item, "size": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    return value, item


@pytest.mark.parametrize("item", module.PACKAGES)
def test_pinned_package_index_header_and_genuine_link(item):
    data, item = archive(item)
    headers, receipt = module.read_package(data, item, UncompressedFixture())
    assert headers[item["header"]] == b"// exact fixture\n"
    assert receipt["index"]["version"] == item["version"]


@pytest.mark.parametrize("corrupt", ["index", "traversal", "absolute", "soname", "size", "sha",
                                   "link-escape", "wrong-link-dir"])
def test_bad_archive_fails_before_header_writes(corrupt):
    data, item = archive(module.PACKAGES[0], corrupt)
    if corrupt == "size":
        item["size"] += 1
    elif corrupt == "sha":
        item["sha256"] = "0" * 64
    with pytest.raises(RuntimeError):
        module.read_package(data, item, UncompressedFixture())


def test_writer_never_overwrites_or_exceeds_budget(tmp_path, monkeypatch):
    writer = module.Writer()
    monkeypatch.setattr(module, "MAX_WRITE", 4)
    writer.write(tmp_path / "one", b"1234")
    with pytest.raises(RuntimeError):
        writer.write(tmp_path / "two", b"x")
    assert not (tmp_path / "two").exists()
    writer.bytes = 0
    with pytest.raises(FileExistsError):
        writer.write(tmp_path / "one", b"1")
    assert (tmp_path / "one").read_bytes() == b"1234"


def test_include_copy_dereferences_without_mutating_input(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.h").write_bytes(b"header")
    (source / "alias.h").symlink_to(source / "a.h")
    writer = module.Writer()
    module.copy_include(source, tmp_path / "private", writer)
    assert (tmp_path / "private/alias.h").read_bytes() == b"header"
    assert not (tmp_path / "private/alias.h").is_symlink()
    assert (source / "alias.h").is_symlink()


def test_include_copy_rejects_cycles(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "cycle").symlink_to(source, target_is_directory=True)
    with pytest.raises(RuntimeError, match="cycle"):
        module.copy_include(source, tmp_path / "partial", module.Writer())


def test_fresh_prefix_and_lock_reject_before_download(tmp_path):
    source = tmp_path / "sdk"
    source.mkdir()
    vendor = tmp_path / "vendor"
    (vendor / "lib").mkdir(parents=True)
    lock = tmp_path / "lock"
    lock.write_text("@EXPLICIT\nnot-approved\n")
    with pytest.raises(RuntimeError, match="lock mismatch"):
        module.provision(source, tmp_path / "never", lock, UncompressedFixture(), vendor)
    assert not (tmp_path / "never").exists()
    with pytest.raises(RuntimeError, match="Fresh absent"):
        module.provision(source, source, lock, UncompressedFixture(), vendor)


@pytest.mark.parametrize("closure_failure", [False, True])
def test_complete_private_provision_preserves_sources_and_binds_separate_devlinks(
        tmp_path, monkeypatch, closure_failure):
    source, vendor = tmp_path / "original-sdk", tmp_path / "vendor"
    for path in (source / "include", source / "bin", source / "nvvm",
                 source / "lib64", vendor / "lib", vendor / "bin"):
        path.mkdir(parents=True)
    (source / "include/original.h").write_bytes(b"original header")
    (vendor / "bin/nvcc").write_bytes(b"fixture compiler, never executed")
    lock = tmp_path / "environment-explicit.txt"
    lock.write_text("@EXPLICIT\nfixture\n")
    monkeypatch.setattr(module, "LOCK_SHA", module.contract.digest(lock))
    packages, payloads = [], {}
    for item in module.PACKAGES:
        data, item = archive(item)
        packages.append(item)
        payloads[module.package_url(item)] = data
        (source / "lib64" / item["runtime_file"]).write_bytes(item["soname"].encode())
    monkeypatch.setattr(module, "PACKAGES", packages)
    monkeypatch.setattr(module, "COMPONENT_BINDINGS", {
        item["family"]: {"source_root": str(source / "lib64"),
                         "runtime_file": item["runtime_file"], "soname": item["soname"],
                         "sha256": module.contract.digest(source / "lib64" / item["runtime_file"])}
        for item in packages})
    monkeypatch.setattr(module.contract, "elf", lambda p: {
        "soname": Path(p).resolve().read_text(), "needed": []})
    monkeypatch.setattr(module.contract, "inspect_sdk", lambda p: {
        "library_root": str((p / "lib64").resolve()),
        "libraries": {item["family"]: {
            "path": str((p / "lib64" / item["runtime_file"]).resolve()),
            "sha256": module.contract.digest(p / "lib64" / item["runtime_file"]),
            "soname": item["soname"]} for item in packages},
        "allowed_files": {str(q.resolve()): module.contract.digest(q)
                          for q in (p / "lib64").iterdir()}})
    monkeypatch.setattr(module, "closure", lambda *args: {"fixture": "no actual ldd"})
    monkeypatch.setattr(module.contract, "command", lambda args: "fixture-tool-pin")
    def download(request, timeout):
        result = io.BytesIO(payloads[request.full_url])
        result.url = request.full_url
        return result
    monkeypatch.setattr(module.urllib.request, "urlopen", download)
    before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    destination = tmp_path / "new-inputs"
    if closure_failure:
        def fail(*args):
            raise RuntimeError("captured loader mapping failure")
        monkeypatch.setattr(module, "closure", fail)
        with pytest.raises(RuntimeError, match="loader mapping"):
            module.provision(source, destination, lock, UncompressedFixture(), vendor)
        assert not (destination / "sdk-manifest.json").exists()
        assert not (destination / "provision-receipt.json").exists()
        return
    result = module.provision(source, destination, lock, UncompressedFixture(), vendor)
    receipt = json.loads((destination / "provision-receipt.json").read_text())
    manifest = json.loads((destination / "sdk-manifest.json").read_text())
    assert result["submission_go"] is False
    assert destination.stat().st_mode & 0o777 == 0o700
    assert manifest["library_root"] == str((source / "lib64").resolve())
    for item in packages:
        link = destination / "sdk/lib" / f"lib{item['family']}.so"
        assert receipt["developer_links"][str(link)]["sha256"] == module.contract.digest(link)
        assert str(link.resolve()) in manifest["allowed_files"]
        assert not (source / "lib64" / link.name).exists()
        assert (destination / "sdk/include" / item["header"]).is_file()
    assert before == {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    assert receipt["compiler_discovery_ready"] is False


@pytest.mark.parametrize("nested", ["sdk", "vendor", "14196330"])
def test_protected_destination_rejected_before_write(tmp_path, nested):
    source, vendor = tmp_path / "sdk", tmp_path / "vendor"
    source.mkdir()
    (vendor / "lib").mkdir(parents=True)
    lock = tmp_path / "lock"
    destination = {"sdk": source / "new", "vendor": vendor / "new",
                   "14196330": tmp_path / "14196330-new"}[nested]
    with pytest.raises(RuntimeError):
        module.provision(source, destination, lock, UncompressedFixture(), vendor)
    assert not destination.exists()


def test_real_streaming_zstd_frame_window_units():
    import zstandard
    parameters = zstandard.ZstdCompressionParameters.from_level(1, window_log=23)
    compressor = zstandard.ZstdCompressor(compression_params=parameters).compressobj()
    frame = compressor.compress(b"x" * 8192) + compressor.flush()
    assert zstandard.get_frame_parameters(frame).window_size == 8 * 1024**2
    # Reproduce the consumed remote attempt's guard bug with a real frame;
    # the earlier uncompressed archive injection could not detect it.
    with pytest.raises(zstandard.ZstdError, match="memory"):
        with zstandard.ZstdDecompressor(max_window_size=128 * 1024).stream_reader(frame) as stream:
            stream.read()
    with module.decoder().stream_reader(frame) as stream:
        assert stream.read() == b"x" * 8192


def test_startup_proof_version_is_fail_closed_then_exercises_actual_codec(monkeypatch):
    import zstandard
    monkeypatch.setattr(zstandard, "__version__", "not-the-frozen-remote-version")
    with pytest.raises(RuntimeError, match="Expected frozen remote"):
        module.verify_remote_decoder()
    # Only this unit test simulates version metadata; actual CLI accepts no
    # version override. Codec is real; this is NOT a remote0.19 pass.
    monkeypatch.setattr(zstandard, "__version__", "0.19.0")
    proof = module.verify_remote_decoder()
    assert proof["negative_rejected"] is True
    assert proof["positive_bound_bytes"] == 128 * 1024**2


def test_input_job_has_exact_bounded_resource_and_no_requeue_contract():
    batch = (TOOLS / "input_prep.sbatch").read_text()
    for option in ["--account=gts-cs207-chemx", "--partition=cpu-small", "--qos=embers",
                   "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--mem=2G",
                   "--time=00:10:00", "--no-requeue"]:
        assert "#SBATCH " + option in batch
    assert "#SBATCH --gres" not in batch
    assert 'trap on_term TERM INT' in batch
    assert 'term_received == 0' in batch
    payload = (TOOLS / "input_prep_payload.sh").read_text()
    assert "--kill-after=5s 110s" in payload
    assert ' -B ' in payload
    assert "cmake" not in payload and "nvcc " not in payload


def virtual_nodes(item):
    name = f"lib{item['family']}.so"
    return {f"lib/{name}": {"type": "2", "link": f"../targets/x86_64-linux/lib/{name}"},
            f"targets/x86_64-linux/lib/{name}": {"type": "2", "link": item["runtime_file"]}}


@pytest.mark.parametrize("item", module.PACKAGES)
def test_virtual_multihop_resolves_only_to_explicit_external_filename(item):
    graph = module.resolve_developer_links(virtual_nodes(item), item)
    start = f"lib/lib{item['family']}.so"
    assert len(graph[start]["chain"]) == 3
    assert graph[start]["external_terminal"].endswith("/" + item["runtime_file"])


@pytest.mark.parametrize("corrupt", ["cycle", "escape", "absolute", "broken", "runtime-shadow",
                                    "too-deep", "non-link", "ancestor-link", "ancestor-file"])
def test_virtual_chains_reject_cycles_escape_broken_or_unbound_terminals(corrupt):
    item = module.PACKAGES[0]
    nodes = virtual_nodes(item)
    start = "lib/libcusparse.so"
    second = "targets/x86_64-linux/lib/libcusparse.so"
    if corrupt == "cycle":
        nodes[second]["link"] = "../../../lib/libcusparse.so"
    elif corrupt == "escape":
        nodes[start]["link"] = "../../outside/" + item["runtime_file"]
    elif corrupt == "absolute":
        nodes[start]["link"] = "/host/" + item["runtime_file"]
    elif corrupt == "broken":
        nodes[second]["link"] = "not-present.so"
    elif corrupt == "runtime-shadow":
        nodes["targets/x86_64-linux/lib/" + item["runtime_file"]] = {"type": "0", "link": ""}
    elif corrupt == "non-link":
        nodes[second]["type"] = "0"
    elif corrupt.startswith("ancestor-"):
        nodes["targets/x86_64-linux/lib"] = {
            "type": "2" if corrupt == "ancestor-link" else "0", "link": "/outside"}
    else:
        nodes[start]["link"] = "step0"
        for n in range(20):
            nodes[f"lib/step{n}"] = {"type": "2", "link": f"step{n+1}"}
    with pytest.raises(RuntimeError):
        module.resolve_developer_links(nodes, item)


def test_explicit_directory_ancestors_are_allowed():
    item = module.PACKAGES[0]
    nodes = virtual_nodes(item)
    for directory in ("lib", "targets", "targets/x86_64-linux", "targets/x86_64-linux/lib"):
        nodes[directory] = {"type": "5", "link": ""}
    assert len(module.resolve_developer_links(nodes, item)) == 2


@pytest.mark.parametrize("corrupt", [None, "hash", "soname", "filename", "symlink"])
def test_explicit_external_runtime_binding_checks_actual_bytes_and_elf(tmp_path, monkeypatch, corrupt):
    item = module.PACKAGES[0]
    path = tmp_path / item["runtime_file"]
    path.write_bytes(b"fixture runtime bytes, no native load")
    binding = {"runtime_file": item["runtime_file"], "soname": item["soname"],
               "source_root": str(tmp_path), "sha256": module.contract.digest(path)}
    monkeypatch.setattr(module.contract, "elf", lambda p: {
        "soname": item["soname"] if corrupt != "soname" else "libcusparse.so.11"})
    if corrupt == "hash":
        binding["sha256"] = "0" * 64
    elif corrupt == "filename":
        binding["runtime_file"] = "libcusparse.so.12"
    elif corrupt == "symlink":
        path.unlink()
        path.symlink_to("/unapproved/host/file")
    if corrupt:
        with pytest.raises(RuntimeError):
            module.verify_runtime_binding(item, binding)
    else:
        assert module.verify_runtime_binding(item, binding)["actual_sha256_and_soname_verified"] is True
