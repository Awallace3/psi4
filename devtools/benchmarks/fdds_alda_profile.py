#!/usr/bin/env python3
"""Diagnostic-only ALDA attribution on an unchanged, pinned staged FDDS build.

Private AST copies wrap existing calls without changing operands, association,
return values, solvers or science. Nested records are not additive timings.
"""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import textwrap
import time

from fdds_product_profile import load_driver


def canonical(expression):
    return ast.unparse(ast.parse(expression, mode="eval").body)


DRIVER_SITES = {
    canonical(expression): label for expression, label in (
        ('fdds_obj.project_densities([cache["D_A"], cache["D_B"]])', "density_projection"),
        ("half_Saux.power(-0.5, 1.e-12)", "overlap_power_negative"),
        ("halfp_Saux.power(0.5, 1.e-12)", "overlap_power_positive"),
        ("_compute_fxc(D[0], half_Saux, halfp_Saux, x_alpha, rho_thresh=rho_thresh)", "alda_A_inclusive"),
        ("_compute_fxc(D[1], half_Saux, halfp_Saux, x_alpha, rho_thresh=rho_thresh)", "alda_B_inclusive"),
        ("W_A.to_array()", "potential_A_conversion"),
        ("W_B.to_array()", "potential_B_conversion"),
    )
}
FXC_SITES = {
    canonical(expression): label for expression, label in (
        ("core.triplet(half_Saux, PQrho, half_Saux, False, False, False)", "density_leveling"),
        ("PQrho_lvl.diagonalize(U, rho, core.DiagonalizeOrder.Ascending)", "density_diagonalization"),
        ("func_x.compute_functional(inp, out, dft_size, 2)", "libxc_exchange"),
        ("func_c.compute_functional(inp, out, dft_size, 2)", "libxc_correlation"),
        ("core.doublet(Ul, U, False, True)", "back_rotation"),
        ("core.triplet(halfp_Saux, tmp, halfp_Saux, False, False, False)", "undo_leveling"),
    )
}


def instrument(function, records, expected, overrides=None):
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    sites = []

    class Calls(ast.NodeTransformer):
        def visit_Call(self, node):
            expression = ast.unparse(node)
            if expression in expected:
                sites.append(expression)
                return ast.copy_location(ast.Call(
                    func=ast.Name(id="_fdds_alda_profile_call", ctx=ast.Load()),
                    args=[ast.Constant(expected[expression]), node.func, *node.args],
                    keywords=node.keywords), node)
            return self.generic_visit(node)

    tree = Calls().visit(tree)
    if len(sites) != len(expected) or set(sites) != set(expected):
        raise ValueError(f"Unexpected ALDA sites: {sites!r}; required {list(expected)!r}")
    ast.fix_missing_locations(tree)

    def call(label, function, *args, **kwargs):
        entry = time.perf_counter()
        start = time.perf_counter()
        succeeded = False
        try:
            result = function(*args, **kwargs)
            succeeded = True
            return result
        finally:
            seconds = time.perf_counter() - start
            # Metadata only, after the timed native call; no array conversion.
            shapes = [list(a.shape) for a in args if hasattr(a, "shape")]
            records.append({"label": label, "seconds": seconds, "succeeded": succeeded,
                            "argument_shapes": shapes,
                            "bookkeeping_seconds_lower_bound": time.perf_counter() - entry - seconds})

    namespace = dict(function.__globals__)
    namespace.update(overrides or {})
    namespace["_fdds_alda_profile_call"] = call
    exec(compile(tree, inspect.getsourcefile(function), "exec"), namespace)
    copied = namespace[function.__name__]
    hashes = {"function_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "instrumented_ast_sha256": hashlib.sha256(ast.dump(tree).encode()).hexdigest()}
    return copied, hashes


def main():
    runner, expected_driver_sha, destination, *arguments = sys.argv[1:]
    driver = load_driver()  # Preserve NumPy-before-Psi4 library initialization.
    path = Path(driver.__file__).resolve()
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != expected_driver_sha:
        raise ValueError(f"Driver hash mismatch: {path} {actual_sha}")
    output = Path(destination)
    if output.exists() or output.with_suffix(".tmp").exists():
        raise FileExistsError(output)
    records = []
    original = driver.df_fdds_dispersion
    fxc, fxc_hashes = instrument(driver._compute_fxc, records, FXC_SITES)
    copied, driver_hashes = instrument(original, records, DRIVER_SITES, {"_compute_fxc": fxc})
    driver.df_fdds_dispersion = copied
    sys.argv = [runner, *arguments]
    completed = False
    try:
        runpy.run_path(runner, run_name="__main__")
        completed = True
    finally:
        driver.df_fdds_dispersion = original
        expected_counts = {label: 1 for label in DRIVER_SITES.values()}
        expected_counts.update({label: 2 for label in FXC_SITES.values()})
        counts = {label: sum(r["label"] == label for r in records) for label in expected_counts}
        valid = completed and counts == expected_counts and len(records) == 19 and all(
            r["succeeded"] for r in records)
        result = {
            "diagnostic_only": True, "valid": valid, "sites": 13,
            "scope": "ALDA call attribution only; inclusive ALDA records contain nested records; not a speedup benchmark",
            "bookkeeping_scope": "Lower bound excludes dispatch and argument evaluation; no corrected whole-wall claim",
            "driver": str(path), "driver_sha256": actual_sha,
            "driver_function": driver_hashes, "fxc_function": fxc_hashes,
            "profile_harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "counts": counts, "records": records,
            "bookkeeping_seconds_lower_bound": sum(r["bookkeeping_seconds_lower_bound"] for r in records)}
        temporary = output.with_suffix(".tmp")
        with temporary.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
        if completed and not valid:
            raise RuntimeError("ALDA diagnostic requires exactly13 sites/19 successful calls")


if __name__ == "__main__":
    main()
