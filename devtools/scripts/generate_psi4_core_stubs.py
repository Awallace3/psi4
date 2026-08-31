#!/usr/bin/env python3
"""Generate Python type stubs for the built ``psi4.core`` extension.

The generator loads only the extension module, not ``psi4/__init__.py``. This
keeps stub generation usable when optional Python-side dependencies are absent
or temporarily incompatible with a development build. Declarations for APIs
attached later by the Python driver are merged from ``psi4_core_dynamic.pyi``.
"""

from __future__ import annotations

import argparse
import ast
import runpy
import sys
import types
from pathlib import Path


def find_stage_lib(repo: Path) -> Path:
    candidates = sorted(
        repo.glob("build*/stage/lib/psi4/core*.so"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            "No built psi4.core extension found under build*/stage/lib/psi4. "
            "Build Psi4 first or pass --stage-lib."
        )
    if len(candidates) > 1:
        print(f"Using newest extension: {candidates[0]}", file=sys.stderr)
    return candidates[0].parent.parent


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-lib",
        type=Path,
        help="Directory containing the staged psi4 package (for example <objdir>/stage/lib)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / ".typings",
        help="Stub root directory (default: .typings)",
    )
    return parser.parse_args()


def source_segment(lines: list[str], node: ast.AST) -> str:
    starts = [node.lineno]
    starts.extend(decorator.lineno for decorator in getattr(node, "decorator_list", []))
    return "\n".join(lines[min(starts) - 1 : node.end_lineno])


def merge_dynamic_overlay(stub_text: str, overlay_path: Path) -> str:
    overlay_text = overlay_path.read_text()
    overlay_lines = overlay_text.splitlines()
    overlay = ast.parse(overlay_text, filename=str(overlay_path))

    module_members: list[str] = []
    for node in overlay.body:
        if isinstance(node, ast.ClassDef):
            declaration = overlay_lines[node.lineno - 1]
            marker = declaration + "\n"
            if stub_text.count(marker) != 1:
                raise SystemExit(f"Could not uniquely locate {node.name} in generated stub")
            members = "\n".join(source_segment(overlay_lines, member) for member in node.body)
            stub_text = stub_text.replace(marker, marker + members + "\n")
        else:
            module_members.append(source_segment(overlay_lines, node))

    if module_members:
        stub_text += "\n# APIs installed dynamically by the Python driver.\n"
        stub_text += "\n\n".join(module_members) + "\n"
    return stub_text


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    stage_lib = (args.stage_lib or find_stage_lib(repo)).resolve()
    package_dir = stage_lib / "psi4"
    if not list(package_dir.glob("core*.so")):
        raise SystemExit(f"No core extension found in {package_dir}")

    try:
        import pybind11_stubgen  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"pybind11-stubgen is not installed for {sys.executable}. Run:\n"
            f"  {sys.executable} -m pip install pybind11-stubgen"
        ) from exc

    # Import psi4.core under its real qualified name without executing the
    # Python package initializer, which imports all driver dependencies.
    package = types.ModuleType("psi4")
    package.__package__ = "psi4"
    package.__path__ = [str(package_dir)]
    sys.modules["psi4"] = package
    sys.path.insert(0, str(stage_lib))

    output_dir = args.output_dir.resolve()
    sys.argv = [
        "pybind11-stubgen",
        "psi4.core",
        "--output-dir",
        str(output_dir),
        # Existing bindings leak C++ spellings such as psi::Matrix into some
        # signatures. Stubgen replaces those individual annotations with ... .
        "--ignore-all-errors",
    ]
    runpy.run_module("pybind11_stubgen", run_name="__main__")

    generated = output_dir / "psi4" / "core.pyi"
    if not generated.is_file():
        raise SystemExit(f"Stub generator did not create {generated}")

    # FrozenOrbitals exports an enum member named ``None``. It passes
    # str.isidentifier(), so stubgen emits it, but it is a Python keyword and
    # therefore an illegal annotated assignment.
    stub_lines = generated.read_text().splitlines(keepends=True)
    stub_text = "".join(
        line
        for line in stub_lines
        if not line.lstrip().startswith("None: typing.ClassVar[")
    )
    overlay = repo / "devtools" / "psi4_core_dynamic.pyi"
    stub_text = merge_dynamic_overlay(stub_text, overlay)
    generated.write_text(stub_text)

    try:
        display_path = generated.relative_to(repo)
    except ValueError:
        display_path = generated
    print(f"Generated {display_path} ({generated.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
