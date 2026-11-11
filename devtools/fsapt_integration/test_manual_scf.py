import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from peptide_scf import timer_summary


class ManualSCFContracts(unittest.TestCase):
    def test_timer_counts_are_doubles_in_native_api(self):
        text=timer_summary({"SCF/cuEST potential compute API":
                            {"wall_time":.12345,"n_calls":2.0}})
        self.assertIn("0.123450",text)
        self.assertIn("      2",text)
        self.assertIn("NOT additive",text)
        self.assertIn("enqueue",text)

    @unittest.skipUnless(shutil.which("g++"),"Host C++ compiler required")
    def test_opt_in_timer_balances_phase_changes_and_exceptions(self):
        root=Path(__file__).resolve().parents[2]
        header=(root/"psi4/src/psi4/libfock/cuESTCommon.h").read_text()
        begin=header.index("class SCFProfileTimer {")
        end=header.index("\n};",begin)+3
        code=r'''
#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept>
#include <cassert>
std::vector<std::string> events;
void timer_on(const char* p) { events.push_back(std::string("+")+p); }
void timer_off(const char* p) { events.push_back(std::string("-")+p); }
''' + header[begin:end] + r'''
int main() {
  unsetenv("PSI4_CUEST_SCF_TIMERS");
  { SCFProfileTimer t("a"); t.next("b"); }
  assert(events.empty());
  setenv("PSI4_CUEST_SCF_TIMERS","1",1);
  try {
    SCFProfileTimer outer("total");
    SCFProfileTimer inner("a");
    inner.next("b");
    throw std::runtime_error("fixture");
  } catch (const std::runtime_error&) {}
  assert((events == std::vector<std::string>{"+total","+a","-a","+b","-b","-total"}));
}
'''
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            (p/"test.cc").write_text(code)
            subprocess.run(["g++","-std=c++17",str(p/"test.cc"),"-o",str(p/"test")],
                           check=True,capture_output=True,timeout=30)
            subprocess.run([str(p/"test")],check=True,timeout=10)


if __name__=="__main__":
    unittest.main()
