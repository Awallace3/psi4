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
                    entry = tarfile.TarInfo(f"targets/x86_64-linux/lib/lib{item['family']}.so")
                    entry.type = tarfile.SYMTYPE
                    entry.linkname = item["soname"] if corruption != "soname" else "libfake.so.12"
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


@pytest.mark.parametrize("corrupt", ["index", "traversal", "absolute", "soname", "size", "sha"])
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


def test_complete_private_provision_preserves_sources_and_binds_separate_devlinks(tmp_path, monkeypatch):
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
    monkeypatch.setattr(module.contract, "elf", lambda p: {
        "soname": Path(p).resolve().read_text(), "needed": []})
    monkeypatch.setattr(module.contract, "inspect_sdk", lambda p: {
        "library_root": str((p / "lib64").resolve()), "libraries": {},
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
