"""Exercise the real Ninja ExternalProject concurrency-inheritance seam."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class CachyBuildContracts(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cmake") and shutil.which("ninja"),
                         "CMake and Ninja required for nested-build regression")
    def test_parallel_limit_reaches_external_project(self):
        script=Path(__file__).with_name("cachy_build.sh").read_text().splitlines()
        # Execute the production build invocation and its parallelism exports,
        # rather than replacing the nested CMake command with a mock.
        commands=[line for line in script
                  if line.startswith("export CMAKE_BUILD_PARALLEL_LEVEL=")
                  or line.startswith("cmake --build ")]
        self.assertTrue(any(line.startswith("cmake --build ") for line in commands))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            child=root/"child"
            child.mkdir()
            result=root/"observed.json"
            worker=root/"observe.py"
            worker.write_text(
                "import json,os,sys\n"
                "from pathlib import Path\n"
                "Path(sys.argv[1]).write_text(json.dumps("
                "{'parallel':os.environ.get('CMAKE_BUILD_PARALLEL_LEVEL')}))\n")
            (child/"CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.20)\nproject(child NONE)\n"
                f'add_custom_target(observe ALL COMMAND "{sys.executable}" '
                f'"{worker}" "{result}" VERBATIM)\n')
            (root/"CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.20)\nproject(parent NONE)\n"
                "include(ExternalProject)\n"
                f'ExternalProject_Add(child SOURCE_DIR "{child}" '
                'INSTALL_COMMAND "" BUILD_ALWAYS 1)\n')
            env={key:value for key,value in os.environ.items()
                 if key not in ("CMAKE_BUILD_PARALLEL_LEVEL","MAKEFLAGS","MFLAGS")}
            subprocess.run(["cmake","-S",str(root),"-B",str(root/"build"),"-G","Ninja"],
                           check=True,capture_output=True,text=True,env=env,timeout=30)
            env.update(ROOT=str(root),JOBS="2")
            run=subprocess.run(["bash","-eu","-c","\n".join(commands)],
                               capture_output=True,text=True,env=env,timeout=30)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertEqual(json.loads(result.read_text())["parallel"],"2",
                             "Outer -j does not propagate into Ninja ExternalProject")


if __name__=="__main__":
    unittest.main()
