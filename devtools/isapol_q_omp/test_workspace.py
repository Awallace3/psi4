"""Pure planning/source tests; no Psi4/numpy imports or native computation."""
import ast
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "psi4/driver/procrouting/isapol_distribution.py"
CPP = ROOT / "psi4/src/psi4/libisapol/partitioned_response.cc"
ORIGINAL = "1aaa9e0a306373b42a3decc25c7dfa106dbc8ea2"


def cpp_body(source, marker):
    """Brace-delimited body for these fixed reviewed code sections."""
    start = source.index("{", source.index(marker))
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start+1:end-1], end


class WorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(PYTHON.read_text())
        helper = next(n for n in cls.tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_q_accumulation_workspace_bytes")
        namespace = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(PYTHON), "exec"), namespace)
        cls.workspace = staticmethod(namespace[helper.name])

    def test_payload_bound_and_grid_size_independence(self):
        for width in (1, 2, 63, 64, 65, 246, 255, 256, 257, 1024, 1048576):
            for rank in range(5):
                for points in (0, 1, 255, 256, 4095, 4096, 4097, 2429910, 10**12):
                    with self.subTest(width=width, rank=rank, points=points):
                        count = min(points, 4096, (8*1024**2)//8//width)
                        self.assertEqual(self.workspace(width, rank, points),
                                         count*(8*(rank+1)**2+1)+4096)
                        self.assertLessEqual(self.workspace(width, rank, points), 827392)
        self.assertEqual(self.workspace(246, 4, 2429910), 827392)
        self.assertEqual(self.workspace(246, 4, 2429910), self.workspace(246, 4, 10**12))

    def test_existing_collocation_memory_admission_edge(self):
        self.assertEqual(self.workspace(1048576, 4, 100), 201+4096)
        with self.assertRaisesRegex(ValueError, "AUX multipole row exceeds collocation scratch budget"):
            self.workspace(1048577, 4, 1)
        for dimensions in ((0, 4, 1), (-1, 4, 1), (1, -1, 1), (1, 5, 1),
                           (1, 4, -1), (True, 4, 1), (1, True, 1), (1, 4, .5)):
            with self.subTest(dimensions=dimensions), self.assertRaises(ValueError):
                self.workspace(*dimensions)

    def test_both_native_stockholder_adapters_charge_workspace(self):
        for name in ("isa_resource_plan", "mbis_resource_plan"):
            function = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == "_q_accumulation_workspace_bytes"]
            self.assertEqual(len(calls), 1)
            self.assertEqual([n.id for n in calls[0].args], ["p", "rank", "h"])
        # No cap/threshold/model declaration or old profiler file is rewritten.
        self.assertIn("256*1024**2", PYTHON.read_text())

    def test_cpp_keeps_batch_and_bounded_disjoint_tasks(self):
        source = CPP.read_text()
        self.assertIn("std::min<size_t>(4096,(8*1024*1024)/sizeof(double)/columns)", source)
        self.assertIn("std::min<size_t>(8,(static_cast<size_t>(columns)+63)/64)", source)
        self.assertIn("std::vector<double> prefactors(components*count)", source)
        self.assertIn("std::vector<unsigned char> active(count,0)", source)
        self.assertIn("detail::parallel_work(tasks,count >= 256 ? 1 : tasks+1", source)
        self.assertIn("if (!active[p]) continue;", source)
        self.assertIn("row[k] += prefactor*phi[p][k];", source)
        self.assertNotIn("omp_set_num_threads", source)
        self.assertNotIn("reduction(", source)
        self.assertLess(source.index("auxiliary.evaluate_screened("), source.index("std::vector<double> prefactors"))
        self.assertLess(source.index('"Nonfinite weighted multipole"'), source.index("detail::parallel_work"))

    def test_task_intervals_cover_each_output_once_with_constant_fanout(self):
        for columns in (1, 2, 63, 64, 65, 127, 129, 246, 255, 256, 257, 1048576):
            for rank in range(5):
                components = (rank+1)**2
                tiles = min(8, (columns+63)//64)
                width = (columns+tiles-1)//tiles
                rows = [[] for _ in range(components)]
                for task in range(components*tiles):
                    row, tile = divmod(task, tiles)
                    rows[row].append((tile*width, min(columns, (tile+1)*width)))
                self.assertLessEqual(components*tiles, 200)
                for intervals in rows:
                    self.assertEqual(intervals[0][0], 0)
                    self.assertEqual(intervals[-1][1], columns)
                    self.assertTrue(all(a < b for a, b in intervals))
                    self.assertTrue(all(left[1] == right[0]
                                        for left, right in zip(intervals, intervals[1:])))


    def test_low_thread_path_restores_exact_original_point_loop(self):
        source = CPP.read_text()
        original = subprocess.check_output(
            ["git", "-C", str(ROOT), "show",
             ORIGINAL+":psi4/src/psi4/libisapol/partitioned_response.cc"], text=True)
        branch, _ = cpp_body(source, "if (original_accumulation)")
        marker = "for (size_t p = begin; p < end; ++p)"
        actual, _ = cpp_body(branch, marker)
        expected, _ = cpp_body(original, marker)
        self.assertEqual("".join(actual.split()), "".join(expected.split()))
        self.assertIn("continue;", branch)
        self.assertNotIn("prefactors", branch)
        self.assertNotIn("active[", branch)
        self.assertNotIn("parallel_work", branch)

    def test_dispatch_is_per_constructor_before_dynamic_parallel_scratch(self):
        source = CPP.read_text()
        selector = "const bool original_accumulation = Process::environment.get_n_threads() <= 2;"
        self.assertEqual(source.count(selector), 1)
        self.assertLess(source.index(selector), source.index("for (size_t a"))
        _, end = cpp_body(source, "if (original_accumulation)")
        self.assertLess(source.index("auxiliary.evaluate_screened("), source.index("if (original_accumulation)"))
        self.assertLess(end, source.index("std::vector<double> prefactors"))
        self.assertLess(end, source.index("std::vector<unsigned char> active"))
        self.assertNotIn("static const bool original_accumulation", source)
        self.assertNotIn("set_n_threads", source)
        # Both callers conservatively reserve the same workspace, even when
        # the low-thread branch uses no staged-prefactor/active payload.
        for workers in (1, 2, 4, 8):
            self.assertEqual(self.workspace(65, 4, 8199), 827392, workers)

    def test_high_thread_tail_and_python_outside_isa_plan_unchanged_from_5ae(self):
        base = subprocess.check_output(
            ["git", "-C", str(ROOT), "show",
             "5ae597cd9305588d14681d0da435834ef761c264:"+str(CPP.relative_to(ROOT))], text=True)
        marker = "            // Per-batch scratch only"
        self.assertEqual(CPP.read_text().split(marker, 1)[1], base.split(marker, 1)[1])
        planner = subprocess.check_output(
            ["git", "-C", str(ROOT), "show",
             "5ae597cd9305588d14681d0da435834ef761c264:"+str(PYTHON.relative_to(ROOT))], text=True)
        # The separately tested ISA accounting revision may change ONLY this
        # function. Keep every other byte (including MBIS/Q workspace) frozen.
        def without_isa(source):
            function = next(n for n in ast.parse(source).body
                            if isinstance(n, ast.FunctionDef) and n.name == "isa_resource_plan")
            lines = source.splitlines(keepends=True)
            return "".join(lines[:function.lineno-1]+lines[function.end_lineno:])
        self.assertEqual(without_isa(PYTHON.read_text()), without_isa(planner))


if __name__ == "__main__":
    unittest.main(verbosity=2)
