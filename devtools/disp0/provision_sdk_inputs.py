"""One bounded private INPUT provision. No solve, compile, device or shared write."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import resource
import re
import signal
import tarfile
import time
import urllib.request
import zipfile

import runtime_contract as contract

PACKAGES = (
    {"name": "libcusparse-dev", "version": "12.8.2.51", "build": "hecca717_0",
     "size": 54443, "sha256": "c472d4f6687017a42b1a392ba4064dbe07ee499d77a61558c04bfd7e6b81b163",
     "family": "cusparse", "soname": "libcusparse.so.12",
     "runtime_file": "libcusparse.so.12.8.2.51", "header": "cusparse.h"},
    {"name": "libnvjitlink-dev", "version": "13.3.33", "build": "hecca717_0",
     "size": 28126, "sha256": "4d0c356194dc359142147236efb56099c0f3d745e6239e51627fa8a97cfaf229",
     "family": "nvJitLink", "soname": "libnvJitLink.so.13",
     "runtime_file": "libnvJitLink.so.13.3.33", "header": "nvJitLink.h"},
)
LOCK_SHA = "3f9a508fd601306f548a9088f5cd304367a5fde3988e8bbd4fe00d9e6efb0205"
MAX_WRITE = 512 * 1024**2
MAX_DOWNLOAD = 128 * 1024**2
MAX_MEMBER = 32 * 1024**2
MAX_WINDOW_BYTES = 128 * 1024**2


def decoder():
    import zstandard
    # This binding passes the raw integer to ZSTD_DCtx_setMaxWindowSize:
    # bytes, not KiB. A real streaming-frame regression covers this boundary.
    return zstandard.ZstdDecompressor(max_window_size=MAX_WINDOW_BYTES)


def package_url(item):
    return ("https://conda.anaconda.org/conda-forge/linux-64/"
            f"{item['name']}-{item['version']}-{item['build']}.conda")


def limited_read(stream, maximum):
    value = stream.read(maximum + 1)
    contract.require(len(value) <= maximum, "Input exceeds byte bound")
    return value


class Writer:
    def __init__(self):
        self.bytes = 0

    def write(self, path, value):
        contract.require(self.bytes + len(value) <= MAX_WRITE, "512MiB write bound")
        with open(path, "xb") as stream:
            stream.write(value)
        self.bytes += len(value)

    def copy(self, source, target):
        contract.require(source.stat().st_size + self.bytes <= MAX_WRITE,
                         "Include tree exceeds write bound")
        with open(source, "rb") as stream, open(target, "xb") as out:
            while block := stream.read(1024**2):
                contract.require(self.bytes + len(block) <= MAX_WRITE, "512MiB write bound")
                out.write(block)
                self.bytes += len(block)

    def json(self, path, value):
        self.write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def safe_member(name):
    path = PurePosixPath(name)
    contract.require(not path.is_absolute() and ".." not in path.parts and "\\" not in name,
                     f"Unsafe package member: {name}")
    return path


def read_package(data, item, decompressor):
    """Validate entire archive identity before streaming selected data, never extractall."""
    contract.require(len(data) == item["size"], "Package size mismatch")
    contract.require(hashlib.sha256(data).hexdigest() == item["sha256"], "Package SHA mismatch")
    headers, links, index = {}, {}, None
    entries = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        contract.require(len(names) == len(set(names)), "Duplicate ZIP member")
        payloads = [n for n in names if n.startswith(("pkg-", "info-")) and n.endswith(".tar.zst")]
        contract.require(len(payloads) == 2 and
                         sum(n.startswith("pkg-") for n in payloads) == 1, "Invalid conda payloads")
        for name in names:
            safe_member(name)
            contract.require(archive.getinfo(name).file_size <= MAX_MEMBER, "Oversize ZIP member")
        for name in payloads:
            expanded = 0
            with archive.open(name) as compressed, decompressor.stream_reader(compressed) as raw:
                with tarfile.open(fileobj=raw, mode="r|") as stream:
                    seen = set()
                    for member in stream:
                        path = safe_member(member.name)
                        contract.require(str(path) not in seen, "Duplicate TAR member")
                        seen.add(str(path))
                        expanded += member.size
                        contract.require(member.size <= MAX_MEMBER and expanded <= MAX_WRITE,
                                         "Expanded package exceeds bound")
                        contract.require(not member.islnk() and not member.isdev() and not member.isfifo(),
                                         "Unsupported package member type")
                        entries.append({"path": str(path), "size": member.size,
                                        "type": member.type.decode("ascii"),
                                        "link": member.linkname})
                        parts = path.parts
                        include = ("targets", "x86_64-linux", "include")
                        if parts[:3] == include and len(parts) > 3:
                            contract.require(member.isfile() or member.isdir(), "Linked package header")
                            if member.isfile():
                                headers[str(PurePosixPath(*parts[3:]))] = limited_read(
                                    stream.extractfile(member), member.size)
                        if member.name == "info/index.json":
                            contract.require(member.isfile(), "Invalid package index")
                            index = json.loads(limited_read(stream.extractfile(member), MAX_MEMBER))
                        if path.name == f"lib{item['family']}.so":
                            contract.require(member.issym() and
                                             PurePosixPath(member.linkname).name == item["soname"],
                                             "Developer link is not the genuine SONAME")
                            links[str(path)] = member.linkname
    contract.require(index is not None and all(index.get(k) == item[k]
                     for k in ("name", "version", "build")), "Package index identity mismatch")
    contract.require(index.get("subdir") == "linux-64", "Wrong package architecture")
    contract.require(item["header"] in headers and links, "Missing header/developer link")
    return headers, {"package": item, "url": package_url(item), "index": index, "entries": entries,
                     "headers": {p: hashlib.sha256(v).hexdigest() for p, v in headers.items()}}


def copy_include(source, destination, writer, ancestors=frozenset()):
    real = source.resolve(strict=True)
    contract.clean_location(real)
    contract.require(real not in ancestors, "Include directory cycle")
    destination.mkdir(mode=0o700)
    for path in sorted(source.iterdir()):
        if path.is_dir():
            copy_include(path, destination / path.name, writer, ancestors | {real})
        else:
            contract.clean_location(path.resolve(strict=True))
            contract.require(path.is_file(), "Non-file include input")
            writer.copy(path, destination / path.name)


def closure(sdk, vendor_environment):
    expected = {v["soname"]: v for v in sdk["libraries"].values()}
    environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C",
                   "LD_LIBRARY_PATH": f"{sdk['library_root']}:{vendor_environment}/lib"}
    result = {}
    for family, item in sdk["libraries"].items():
        selected = contract.parse_loader(contract.command(
            ["/usr/bin/ldd", item["path"]], env=environment))
        for name, path in selected.items():
            if name in expected:
                contract.require(str(path) == expected[name]["path"] and
                                 contract.digest(path) == expected[name]["sha256"],
                                 "Wrong CUDA dependency closure")
            contract.require(not re.match(
                r"lib(cudart|cublas|cublasLt|nvJitLink)\.so\.12(?:$|\.)", name),
                "CUDA12 dependency closure")
        result[family] = {name: contract.identity(path) for name, path in selected.items()}
    return result


def provision(source, destination, lock, decompressor, vendor_environment=None):
    start = time.monotonic()
    contract.require(destination.is_absolute() and source.is_absolute(), "Absolute paths required")
    contract.require(not destination.exists() and not destination.is_symlink(), "Fresh absent prefix required")
    contract.require(vendor_environment is not None and vendor_environment.is_absolute()
                     and (vendor_environment / "lib").is_dir(),
                     "Explicit read-only vendor environment required")
    contract.clean_location(vendor_environment.resolve(strict=True))
    canonical_destination = destination.resolve()
    contract.clean_location(canonical_destination)
    contract.require(not any(canonical_destination.is_relative_to(root.resolve(strict=True))
                             for root in (source, vendor_environment)),
                     "Destination cannot nest beneath protected SDK/vendor inputs")
    parent = destination.parent.stat()
    contract.require(parent.st_uid == os.getuid() and not (parent.st_mode & 0o022),
                     "Private owned parent required")
    contract.require(contract.digest(lock) == LOCK_SHA, "Explicit dependency lock mismatch")
    contract.require("@EXPLICIT" in lock.read_text(), "Not an explicit dependency lock")
    contract.require(sum(p["size"] for p in PACKAGES) <= MAX_DOWNLOAD, "Download budget")
    # Exact selected runtime files must preexist. No runtime package is downloaded.
    for item in PACKAGES:
        runtime = (source / "lib64" / item["runtime_file"]).resolve(strict=True)
        contract.require(contract.elf(runtime)["soname"] == item["soname"], "Runtime SONAME mismatch")
    destination.mkdir(mode=0o700)
    writer, records, all_headers = Writer(), [], {}
    for item in PACKAGES:
        request = urllib.request.Request(package_url(item), headers={"User-Agent": "disp0-private-input/1"})
        with urllib.request.urlopen(request, timeout=15) as response:
            contract.require(response.url.startswith("https://"), "Non-HTTPS package response")
            data = limited_read(response, item["size"])
        headers, record = read_package(data, item, decompressor)
        contract.require(not set(headers) & set(all_headers), "Conflicting package headers")
        all_headers.update(headers)
        writer.write(destination / Path(package_url(item)).name, data)
        records.append(record)
    sdk = destination / "sdk"
    sdk.mkdir(mode=0o700)
    copy_include(source / "include", sdk / "include", writer)
    for name, value in all_headers.items():
        path = sdk / "include" / name
        contract.require(not path.exists(), "Package would replace existing SDK header")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        writer.write(path, value)
    for name in ("bin", "nvvm", "lib64"):
        (sdk / name).symlink_to((source / name).resolve(strict=True), target_is_directory=True)
    (sdk / "lib").mkdir(mode=0o700)
    developer_links = {}
    for item in PACKAGES:
        link = sdk / "lib" / f"lib{item['family']}.so"
        link.symlink_to((source / "lib64" / item["runtime_file"]).resolve(strict=True))
        developer_links[str(link)] = {**contract.identity(link),
                                     "soname": contract.elf(link)["soname"]}
    manifest = contract.inspect_sdk(sdk)
    dependencies = closure(manifest, vendor_environment)
    # Existing nvcc locator delegates to this vendor compiler. Bind it rather
    # than presenting a wrapper hash as the underlying compiler identity.
    compiler_chain = {"vendor_nvcc": contract.identity(vendor_environment / "bin/nvcc"),
                      "profiles": {str(p): contract.identity(p)
                                   for p in (vendor_environment / "bin").glob("*nvcc*.profile")}}
    writer.json(destination / "sdk-manifest.json", manifest)
    writer.json(destination / "provision-receipt.json", {
        "scope": "INPUTS_ONLY_NOT_BUILD_OR_DEVICE_READINESS", "packages": records,
        "source_sdk": str(source), "destination": str(destination), "sdk": str(sdk),
        "recipe_pin": contract.command(["git", "rev-parse", "HEAD"]).strip(),
        "helper_sha256": contract.digest(__file__),
        "package_metadata_sources": [
            f"https://api.anaconda.org/release/conda-forge/{p['name']}/{p['version']}"
            for p in PACKAGES],
        "vendor_environment": str(vendor_environment), "limits": {
            "wall_seconds": 110, "outer_kill_grace_seconds": 5, "threads": 1,
            "address_space_bytes": 1024**3, "download_bytes": MAX_DOWNLOAD, "write_bytes": MAX_WRITE},
        "decoder_window_bound_bytes": MAX_WINDOW_BYTES,
        "developer_links": developer_links,  # input lib/, separate from lib64 SDK inventory
        "compiler_chain": compiler_chain,
        "compiler_discovery_ready": False,
        "compiler_discovery_caveat": "Existing wrapper delegates to original vendor prefix; "
            "copied input include selection is not proven by --version or ELF metadata.",
        # The old explicit lock is an authorized read-only INPUT, not a loader path.
        "dependency_lock": {"path": str(lock.resolve()), "sha256": contract.digest(lock)},
        "sdk_manifest_sha256": contract.digest(destination / "sdk-manifest.json"),
        "closure": dependencies, "new_download_bytes": sum(p["size"] for p in PACKAGES),
        "written_before_receipt_bytes": writer.bytes, "elapsed_seconds": time.monotonic() - start,
        "cpu_build_submission_go": False})
    return {"prefix": str(destination), "sdk_manifest_sha256": contract.digest(
        destination / "sdk-manifest.json"), "written_bytes": writer.bytes,
        "elapsed_seconds": time.monotonic() - start, "submission_go": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--vendor-environment", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (110, 110))
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    # External timeout --kill-after=5s 110s is additionally mandatory.
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("110s input deadline")))
    signal.alarm(110)
    result = provision(args.source, args.destination, args.lock,
                       decoder(),
                       args.vendor_environment.resolve(strict=True))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
