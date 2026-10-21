"""Effective compiler/link command and actual Ninja header-dependency audit."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex

from runtime_contract import command, digest, require, write_new

BAD = ("cuda-12.", "14182584", "14193865", "14196330")
CUDA_HEADER = re.compile(r"^(cuda(?:_.*)?|cublas.*|cusolver.*|cusparse.*|nvJitLink)\.h$")
CUDA_NAMESPACES = {"cuda", "crt", "nv", "nvtx3", "cub", "thrust", "cooperative_groups"}
# Standard host headers can legitimately coexist with libcudacxx copies.
# Namespace-specific CUDA copies still require the canonical SDK identity.
HOST_HEADERS = {"assert.h", "complex.h", "ctype.h", "errno.h", "fenv.h", "float.h", "inttypes.h",
                "limits.h", "math.h", "omp.h", "setjmp.h", "signal.h", "stdarg.h", "stdbool.h",
                "stddef.h", "stdint.h", "stdio.h", "stdlib.h", "string.h", "time.h", "wchar.h",
                "wctype.h", "unistd.h", "threads.h", "stdatomic.h", "tgmath.h"}


def roots(run, source, sdk, scratch):
    return [Path(p).resolve() for p in (
        run, source, scratch, sdk["source"], sdk["include"], sdk["library_root"],
        *(str(Path(item["path"]).parent) for item in sdk["support"].values()),
        "/usr/include", "/usr/lib", "/usr/lib64", "/lib", "/lib64", "/usr/bin", "/bin")]


def safe_path(value, directory, allowed):
    value = value.strip("'\"")
    if not value or value == "/dev/null":
        return
    # Ninja's shell-escaped linker token is still the literal ELF $ORIGIN.
    # Recognize ONLY that token, not arbitrary escaped variables/prefix lookalikes.
    if re.fullmatch(r"\\?\$(?:ORIGIN|\{ORIGIN\})(?:/[A-Za-z0-9_.+-]*)*", value):
        require(not any(token in value for token in BAD), f"Foreign origin-relative build path: {value}")
        return
    require("$" not in value and not any(token in value for token in BAD),
            f"Uninspectable/foreign build path: {value}")
    path = Path(value)
    if not path.is_absolute():
        path = Path(directory) / path
    path = path.resolve()
    require(any(path.is_relative_to(root) for root in allowed), f"Build path outside isolated inputs: {path}")


def compiler_allowed(program, run, sdk):
    path = Path(program).resolve()
    env = Path(run) / "environment/bin"
    host = {str((env / name).resolve()) for name in
            ("x86_64-conda-linux-gnu-cc", "x86_64-conda-linux-gnu-c++",
             "x86_64-conda-linux-gnu-gfortran")}
    return str(path) in host or str(path) == sdk["support"]["bin/nvcc"]["path"]


def audit_linker_parts(parts, directory, allowed):
    rpaths = {"-rpath", "--rpath", "-rpath-link", "--rpath-link", "-R"}
    files = {"-T", "--script", "--version-script", "--dynamic-list", "--retain-symbols-file",
             "--just-symbols", "--symbol-ordering-file", "--section-ordering-file",
             "--dependency-file", "-Map", "--Map", "--out-implib", "--default-script", "-L", "--library-path"}
    index = 0
    while index < len(parts):
        part = parts[index]
        require(not part.startswith("@"), "Unexpanded linker input response file")
        if part in rpaths | files:
            require(index + 1 < len(parts), "Truncated linker file/path option")
            index += 1
            values = parts[index].split(":") if part in rpaths else [parts[index]]
            for value in values:
                if value:  # Build-only RPATH padding; installed policy forbids empty entries.
                    safe_path(value, directory, allowed)
        elif "=" in part:
            key, value = part.split("=", 1)
            if key in rpaths:
                for entry in value.split(":"):
                    if entry:
                        safe_path(entry, directory, allowed)
            elif key in files or "/" in value:
                safe_path(value, directory, allowed)
        elif part.startswith(("-T", "-L", "-R")) and len(part) > 2:
            for value in (part[2:].split(":") if part.startswith("-R") else [part[2:]]):
                if value:
                    safe_path(value, directory, allowed)
        elif "/" in part:
            require(not part.startswith("-"), f"Unsupported opaque linker path: {part}")
            safe_path(part, directory, allowed)
        index += 1


def audit_tokens(tokens, directory, allowed):
    joined = " ".join(tokens)
    require(not any(token in joined for token in BAD), "Foreign historical/site12 command")
    require(not any(Path(token).name in ("ccache", "sccache") for token in tokens),
            "Unapproved compiler-cache launcher")
    # GNU driver forwarding is one linker stream, not compiler filenames.
    # In particular: --version-script -Xlinker FILE must not consume the
    # intervening forwarding marker as the script name.
    plain, forwarded = [], []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "-Xlinker":
            require(index + 1 < len(tokens), "Truncated -Xlinker forwarding")
            index += 1
            forwarded.append(tokens[index])
        elif token.startswith("-Xlinker="):
            forwarded.append(token.split("=", 1)[1])
        elif token.startswith("-Wl,"):
            forwarded.extend(token.split(",")[1:])
        else:
            plain.append(token)
        index += 1
    if forwarded:
        audit_linker_parts(forwarded, directory, allowed)
    tokens = plain
    path_options = ("-I", "-L", "-isystem", "-iquote", "-idirafter", "-include", "-imacros",
                    "--sysroot", "-B", "-ccbin", "--compiler-bindir", "--include-path", "--library-path",
                    "-T", "--script", "--version-script", "--dynamic-list", "--retain-symbols-file",
                    "--just-symbols", "--symbol-ordering-file", "--section-ordering-file")
    pending = False
    for token in tokens:
        require(not token.startswith(("@", "-Wl,@")), "Unexpanded compiler/link response file")
        if pending:
            safe_path(token, directory, allowed)
            pending = False
        if token in path_options:
            pending = True
        for option in path_options:
            if token.startswith(option + "="):
                safe_path(token[len(option) + 1:], directory, allowed)
            elif not option.startswith("--") and token.startswith(option) and len(token) > len(option):
                safe_path(token[len(option):], directory, allowed)
        if token.startswith("--") and "=" in token and "/" in token.split("=", 1)[1]:
            audit_linker_parts([token], directory, allowed)
        if token.startswith("/"):
            safe_path(token, directory, allowed)
        if token.startswith("-Xcompiler="):
            audit_tokens(token.split("=", 1)[1].split(","), directory, allowed)
    require(not pending, "Truncated include/library-path option")


def audit_compile_database(entries, run, source, sdk, scratch, response_expansions=None):
    allowed = roots(run, source, sdk, scratch)
    require(entries, "Empty actual compile-command database")
    for entry in entries:
        directory = Path(entry["directory"]).resolve()
        require(directory.is_relative_to(Path(run).resolve()), "Foreign compilation directory")
        tokens = entry.get("arguments") or shlex.split(entry["command"])
        require(tokens and compiler_allowed(tokens[0], run, sdk), f"Wrong actual compiler: {tokens[:1]}")
        safe_path(entry["file"], directory, allowed)
        if any(token.startswith(("@", "-Wl,@")) for token in tokens):
            key = (str(directory), str((directory / entry["file"]).resolve()))
            expansions = (response_expansions or {}).get(key, [])
            require(expansions, "No effective Ninja response-file expansion for compiler command")
            for expanded in expansions:
                effective = shlex.split(expanded)
                require(compiler_allowed(effective[0], run, sdk), "Foreign expanded compiler")
                audit_tokens(effective, directory, allowed)
        else:
            audit_tokens(tokens, directory, allowed)
    return len(entries)


def audit_dependencies(text, directory, run, source, sdk, scratch, hash_cache=None):
    allowed = roots(run, source, sdk, scratch)
    sdk_headers = {item["path"]: item["sha256"] for item in sdk["input_trees"]["include"].values()}
    relative = set(sdk["input_trees"]["include"])
    cuda_root_names = {name for name in relative if len(Path(name).parts) == 1 and name not in HOST_HEADERS}
    cache = hash_cache if hash_cache is not None else {}
    count, observed = 0, set()
    for line in text.splitlines():
        if not line.startswith("    "):
            continue
        value = line.strip()
        safe_path(value, directory, allowed)
        path = Path(value)
        if not path.is_absolute():
            path = Path(directory) / path
        path = path.resolve(strict=True)
        namespaced = any("/".join(path.parts[i:]) in relative
                         for i, part in enumerate(path.parts) if part in CUDA_NAMESPACES)
        if str(path) in sdk_headers or CUDA_HEADER.match(path.name) or path.name in cuda_root_names or namespaced:
            stat = path.stat()
            key = (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if key not in cache:
                cache[key] = digest(path)
            require(sdk_headers.get(str(path)) == cache[key], f"Foreign/changed actual CUDA header: {path}")
            observed.add(path.name)
        count += 1
    return count, observed


def audit_link_commands(text, directory, run, source, sdk, scratch):
    allowed = roots(run, source, sdk, scratch)
    count = 0
    compiler_name = re.compile(
        r"^(?:[A-Za-z0-9_]+-)*(?:g\+\+|c\+\+|gcc|cc|gfortran|nvcc|clang\+\+|clang|icx|icpx|ifort|ifx|flang)"
        r"(?:-[0-9.]+)?$")
    for line in text.splitlines():
        if not line.strip():
            continue
        lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        segments, segment = [], []
        for token in lexer:
            if token in ("&&", "||", ";", "|"):
                if segment:
                    segments.append(segment)
                    segment = []
            else:
                segment.append(token)
        if segment:
            segments.append(segment)
        working = Path(directory)
        for tokens in segments:
            program = tokens[0]
            require(not any(Path(t).name in ("ccache", "sccache") for t in tokens),
                    "Unapproved compiler-cache wrapper")
            if program == "cd":
                require(len(tokens) == 2, "Uninspectable working-directory command")
                safe_path(tokens[1], working, allowed)
                working = (working / tokens[1]).resolve()
            elif compiler_name.fullmatch(Path(program).name):
                require(compiler_allowed(program, run, sdk),
                        f"Foreign compiler/linker command: {program}")
                audit_tokens(tokens, working, allowed)
                if "-shared" in tokens or any(t.startswith("-Wl,-soname,") for t in tokens):
                    count += 1
            else:
                # Full compdb includes Python/CMake code-generation rules.
                # Their @fragment lists are not compiler response files.
                # Preserve their raw metadata, but never silently skip a
                # compiler hidden behind an unrecognized wrapper.
                require(not any(compiler_name.fullmatch(Path(t).name) for t in tokens[1:]),
                        "Unrecognized compiler wrapper in build metadata")
    return count


def audit_output_directory(run, artifact_dir=None):
    run = Path(run).resolve()
    output = Path(artifact_dir) if artifact_dir is not None else run / "effective-build-audit"
    if artifact_dir is not None:
        require(output.is_absolute() and not output.resolve().is_relative_to(run),
                "Revalidation evidence must be outside the preserved failed run")
    output.mkdir()  # Never overwrite old partial audits or a previous validation.
    return output


def capture(run, source, scratch, artifact_dir=None):
    run = Path(run).resolve()
    sdk = json.loads((run / "cuda13/sdk-contract.json").read_text())
    output = audit_output_directory(run, artifact_dir)
    databases, counts = [], {"compile": 0, "dependencies": 0, "links": 0}
    observed = set()
    records = {}
    header_hashes = {}
    for name in ("einsums-build", "psi4-build"):
        base = run / name
        dbs = [p for p in base.rglob("compile_commands.json") if "CMakeScratch" not in p.parts]
        require(dbs, f"No effective compile database for {name}")
        ninjas = [p for p in base.rglob("build.ninja") if "CMakeScratch" not in p.parts]
        require(ninjas, f"No Ninja graph for {name}")
        responses = {}
        for index, ninja in enumerate(sorted(ninjas)):
            for tool in ("commands", "deps", "compdb"):
                args = [str(run / "environment/bin/ninja"), "-C", str(ninja.parent), "-t", tool]
                if tool == "compdb":
                    args.append("-x")  # Expand configured rspfile_content, even after Ninja deleted the file.
                text = command(args)
                path = output / f"{name}-{index}-{tool}.txt"
                with path.open("x") as stream:
                    stream.write(text)
                records[str(path)] = digest(path)
                if tool == "commands":
                    require(not any(token in text for token in BAD), "Foreign raw Ninja command")
                elif tool == "deps":
                    count, seen = audit_dependencies(text, ninja.parent, run, source, sdk, scratch, header_hashes)
                    counts["dependencies"] += count
                    observed.update(seen)
                else:
                    expanded = json.loads(text)
                    for entry in expanded:
                        if not entry.get("command"):
                            continue
                        counts["links"] += audit_link_commands(
                            entry["command"], entry["directory"], run, source, sdk, scratch)
                        key = (str(Path(entry["directory"]).resolve()),
                               str((Path(entry["directory"]) / entry["file"]).resolve()))
                        responses.setdefault(key, []).append(entry["command"])
        for db in dbs:
            entries = json.loads(db.read_text())
            if entries:
                counts["compile"] += audit_compile_database(entries, run, source, sdk, scratch, responses)
                databases.append(str(db))
                records[str(db)] = digest(db)
    require(counts["compile"] > 0 and counts["links"] > 0 and counts["dependencies"] > 0,
            "Incomplete actual compiler/link/header evidence")
    require(observed & {"cuda_runtime.h", "cuda_runtime_api.h"}
            and observed & {"cublas_v2.h", "cublas_api.h"}, "No actual CUDA/runtime/cuBLAS include evidence")
    return {"status": "EFFECTIVE_BUILD_INPUTS_AUDITED", "counts": counts,
            "observed_cuda_headers": sorted(observed), "compile_databases": databases,
            "file_sha256": records, "source": str(Path(source).resolve()), "run": str(run),
            "scope": "Compiler/link/dependency paths audited; auxiliary Ninja rules captured raw; no device correctness"}


def main():
    parser = argparse.ArgumentParser()
    for name in ("run", "source", "scratch", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--artifact-dir", help="Fresh external evidence directory; input build stays read-only")
    args = parser.parse_args()
    write_new(args.out, capture(args.run, args.source, args.scratch, args.artifact_dir))


if __name__ == "__main__":
    main()
