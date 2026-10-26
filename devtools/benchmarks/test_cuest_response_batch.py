"""CPU-only exhaustive check of resident cuEST density dispatch (no SDK needed)."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_density_batch_schedules(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    source = r"""
#include "psi4/src/psi4/libfock/cuest_response_batch.h"
#include <cassert>
using psi::cuest_response::density_batches;
int main() {
    for (std::size_t storage = 1; storage <= 8; ++storage)
      for (std::size_t actual = 1; actual <= storage; ++actual)
        for (std::size_t spin = 1; spin <= 2; ++spin)
          for (std::size_t limit = 1; limit <= 17; ++limit) {
            std::vector<unsigned> seen(storage * spin, 0);
            for (const auto& b : density_batches(storage, actual, spin, limit)) {
                assert(b.count && b.count <= limit && b.first + b.count <= seen.size());
                for (std::size_t i = b.first; i < b.first + b.count; ++i) ++seen[i];
            }
            for (std::size_t i = 0; i < seen.size(); ++i)
                assert(seen[i] == (i % storage < actual ? 1u : 0u));
          }
    for (const auto& shape : {std::vector<std::size_t>{0,1,1,1}, {1,0,1,1},
                            {1,2,1,1}, {1,1,0,1}, {1,1,3,1}, {1,1,1,0}}) {
        bool threw = false;
        try { density_batches(shape[0], shape[1], shape[2], shape[3]); }
        catch (const std::invalid_argument&) { threw = true; }
        assert(threw);
    }
    bool overflow = false;
    try { density_batches(std::numeric_limits<std::size_t>::max(), 1, 2, 1); }
    catch (const std::overflow_error&) { overflow = true; }
    assert(overflow);
}
"""
    binary = tmp_path / "density-batches"
    subprocess.run([compiler, "-std=c++14", "-Wall", "-Wextra", "-Werror",
                    "-x", "c++", "-", "-I", str(root), "-o", str(binary)],
                   input=source, text=True, check=True, capture_output=True)
    subprocess.run([str(binary)], check=True)


def test_gridk_workspace_budget(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    source = r"""
#include "psi4/src/psi4/libfock/cuest_workspace_policy.h"
#include <cassert>
#include <initializer_list>
int main() {
    using psi::cuest_workspace::gridk_variable_bytes;
    assert(gridk_variable_bytes(1) == 1000000);
    assert(gridk_variable_bytes(64) == 64000000);
    assert(gridk_variable_bytes(256) == 256000000);
    if (std::numeric_limits<std::size_t>::max() >= 8000000000ULL)
        assert(gridk_variable_bytes(8000) == 8000000000ULL);
    for (int invalid : {0, -1}) {
        bool threw = false;
        try { gridk_variable_bytes(invalid); }
        catch (const std::invalid_argument&) { threw = true; }
        assert(threw);
    }
}
"""
    binary = tmp_path / "gridk-workspace"
    subprocess.run([compiler, "-std=c++14", "-Wall", "-Wextra", "-Werror",
                    "-x", "c++", "-", "-I", str(root), "-o", str(binary)],
                   input=source, text=True, check=True, capture_output=True)
    subprocess.run([str(binary)], check=True)
