"""Streaming LOCAL SDK manifest: no extraction, import, install or execution."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ARCHIVE_SHA = "a06e81489dea0b652bfc4451bcfe50d3032881a7eb6decc34cdbfc50a2076c0d"
NATIVE_SHA = "65c9e559201ee589cb742b919cc1c5d80c45aa480b64aede17ac96aff9e4963c"


def digest(stream):
    result = hashlib.sha256()
    for data in iter(lambda: stream.read(1024**2), b""):
        result.update(data)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    with args.archive.open("rb") as stream:
        sha = digest(stream)
    if args.archive.stat().st_size != 206185073 or sha != ARCHIVE_SHA:
        raise RuntimeError("Not the canonical cuEST0.2.50 archive")
    headers, binaries, links = {}, {}, {}
    with tarfile.open(args.archive, "r|gz") as archive:
        for member in archive:
            if not member.name.startswith(("cuest_PID/include/", "cuest_PID/lib/13/")):
                continue
            if member.isfile():
                record = {"bytes": member.size, "sha256": digest(archive.extractfile(member))}
                (headers if member.name.startswith("cuest_PID/include/") else binaries)[member.name] = record
            elif member.issym():
                links[member.name] = member.linkname
    native = binaries["cuest_PID/lib/13/libcuest.so.0.2.50"]
    if native["sha256"] != NATIVE_SHA:
        raise RuntimeError("Canonical archive's CUDA13 native library differs")
    header_json = json.dumps(headers, sort_keys=True, separators=(",", ":")).encode()
    print(json.dumps({"sdk_version": "0.2.50", "archive": str(args.archive.resolve()),
                      "archive_bytes":206185073, "archive_sha256":sha,
                      "header_tree_manifest_sha256":hashlib.sha256(header_json).hexdigest(),
                      "header_files":headers, "cuda13_binary_files":binaries,
                      "symlinks":links, "scope":"LOCAL ARCHIVE ONLY; no loaded/runtime/device proof",
                      "python_binding_record":None}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
