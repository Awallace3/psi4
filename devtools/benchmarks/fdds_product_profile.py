#!/usr/bin/env python3
"""Diagnostic only: time the original left-associated FDDS NumPy products.

No monkeypatch of NumPy, changed multiplication association, solver, or science.
The imported staged function is copied into an instrumented private namespace.
The benchmark and its provenance/route/energy checks remain authoritative.
"""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import time


EXPECTED = {
    "K_A.dot(Rtinv_A).dot(metric)", "K_B.dot(Rtinv_B).dot(metric)",
    "X_A.dot(metric_inv).dot(W_A)", "X_B.dot(metric_inv).dot(W_B)",
    "XSW_A.dot(amplitude).dot(X_A)", "XSW_B.dot(amplitude).dot(X_B)",
    "metric_inv.dot(X_A_uc).dot(metric_inv)",
    "metric_inv.dot(X_A_coupled).dot(metric_inv)",
}


def instrument(function, records):
    """Fail closed on a different source structure; retain exact dot ordering."""
    source = inspect.getsource(function)
    tree = ast.parse(source)
    sites = []

    class Products(ast.NodeTransformer):
        def visit_Call(self, node):
            if (isinstance(node.func, ast.Attribute) and node.func.attr == "dot"
                    and isinstance(node.func.value, ast.Call)):
                inner = node.func.value
                if (isinstance(inner.func, ast.Attribute) and inner.func.attr == "dot"
                        and len(inner.args) == len(node.args) == 1
                        and not inner.keywords and not node.keywords):
                    expression = ast.unparse(node)
                    sites.append(expression)
                    return ast.copy_location(ast.Call(
                        func=ast.Name(id="_fdds_profile_product", ctx=ast.Load()),
                        args=[inner.func.value, inner.args[0], node.args[0],
                              ast.Constant(expression)], keywords=[]), node)
            return self.generic_visit(node)

    tree = Products().visit(tree)
    if len(sites) != 8 or set(sites) != EXPECTED:
        raise ValueError(f"Unexpected FDDS product sites: {sites!r}")
    ast.fix_missing_locations(tree)

    def product(a, b, c, label):
        entry = time.perf_counter()
        start = time.perf_counter()
        try:
            return a.dot(b).dot(c)
        finally:
            seconds = time.perf_counter() - start
            records.append({"expression": label, "shape": list(a.shape),
                            "seconds": seconds})
            # Lower bound: excludes extra Python dispatch/argument evaluation
            # and these final accounting instructions. Never subtract this
            # from the diagnostic whole-FDDS time to invent a benchmark.
            records[-1]["bookkeeping_seconds_lower_bound"] = (
                time.perf_counter() - entry - seconds)

    namespace = dict(function.__globals__)
    namespace["_fdds_profile_product"] = product
    exec(compile(tree, inspect.getsourcefile(function), "exec"), namespace)
    result = namespace[function.__name__]
    result.profile_ast_sha256 = hashlib.sha256(ast.dump(tree).encode()).hexdigest()
    return result, hashlib.sha256(source.encode()).hexdigest()


def main():
    runner, expected_driver_sha, destination, *arguments = sys.argv[1:]
    # The runner has independent exact stage/core guards. Hash the actual
    # imported scientific driver too; don't silently instrument another stage.
    from psi4.driver.procrouting.sapt import sapt_mp2_terms
    driver = Path(sapt_mp2_terms.__file__).resolve()
    actual_sha = hashlib.sha256(driver.read_bytes()).hexdigest()
    if actual_sha != expected_driver_sha:
        raise ValueError(f"Driver hash mismatch: {driver} {actual_sha}")
    output = Path(destination)
    if output.exists() or output.with_suffix(".tmp").exists():
        raise FileExistsError(output)
    records = []
    original = sapt_mp2_terms.df_fdds_dispersion
    instrumented, source_sha = instrument(original, records)
    sapt_mp2_terms.df_fdds_dispersion = instrumented
    sys.argv = [runner, *arguments]
    completed = False
    try:
        runpy.run_path(runner, run_name="__main__")
        completed = True
    finally:
        sapt_mp2_terms.df_fdds_dispersion = original
        counts = {site: sum(r["expression"] == site for r in records) for site in EXPECTED}
        valid = completed and len(records) == 80 and all(n == 10 for n in counts.values())
        result = {"diagnostic_only": True, "driver": str(driver),
                  "driver_sha256": actual_sha, "function_sha256": source_sha,
                  "instrumented_ast_sha256": instrumented.profile_ast_sha256,
                  "profile_harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "scope": "Original eight left-associated dot chains only; not a speedup benchmark",
                  "bookkeeping_scope": "Lower bound excludes added Python dispatch; no wall-time correction applied",
                  "valid": valid, "sites": len(counts), "counts": counts,
                  "records": records, "total_product_seconds": sum(r["seconds"] for r in records),
                  "bookkeeping_seconds_lower_bound": sum(
                      r["bookkeeping_seconds_lower_bound"] for r in records)}
        temporary = output.with_suffix(".tmp")
        with temporary.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
        if completed and not valid:
            raise RuntimeError("FDDS diagnostic requires exactly eight sites /80 calls")


if __name__ == "__main__":
    main()
