"""CPU fixture of normal support-package import and early loader ordering.

Tiny C ELF stubs stand in for CUDA/cuEST; no vendor or scientific API executes.
This is not a whole-install or GPU qualification.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[3]
PREFIX = r'''
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch
repo,root=map(Path,sys.argv[1:3])
sys.path.insert(0,str(repo/"devtools/disp0/cachy"))
sys.path.insert(0,str(repo/"devtools/disp0/phoenix_validation"))
import bootstrap_gate as bootstrap
from gpu_runtime import require_runtime_support,sealed_startup_path
package=root/"site-packages/psi4"
folder=package/"driver/procrouting/sapt";folder.mkdir(parents=True)
for name in ("sapt_cuest250_identity.py","sapt_cuest_factory_abi.py","sapt_cuest_sealed_runtime.py"):
    shutil.copy2(repo/"psi4/driver/procrouting/sapt"/name,folder/name)
support=package.parent/"psi4_runtime_support"
shutil.copytree(repo/"psi4_runtime_support",support,ignore=shutil.ignore_patterns("__pycache__"))
inventory={p.relative_to(support).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for p in support.rglob("*.py")}
build={"stage":str(root),"python_package":str(package),
       "runtime_support_package":str(support),"runtime_support_sha256":inventory,
       "scientific_python_sha256":{"driver/procrouting/sapt/sapt_cuest_sealed_runtime.py":"fixture"}}
environment=root/"environment";(environment/"lib").mkdir(parents=True)
runtime=root/"runtime";sdk=runtime/"cuest";(sdk/"lib/13").mkdir(parents=True)
def library(path,body):
    source=path.with_name(path.name+".c");source.write_text(body)
    subprocess.run(["cc","-shared","-fPIC",str(source),"-o",str(path)],check=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()
names=bootstrap.native_symbols(package)
native=sdk/"lib/13/libcuest.so.0.2.50"
native_sha=library(native,"\n".join("int "+name+"(void){return 0;}" for name in names))
cuda=runtime/"cuda";(cuda/"lib").mkdir(parents=True)
components=[]
for name in ("libcudart.so.13","libcublas.so.13","libcublasLt.so.13"):
    path=cuda/"lib"/name
    sha=library(path,"int fixture(void){return 7;}")
    components.append({"path":str(path),"sha256":sha})
manifest={"cuda_root":str(cuda),"cuda_runtime_version":13030,
          "nvrtc_version":[13,3],"components":components}
config={"schema":2,"identity_contract":bootstrap.SEALED,"cuest_root":str(sdk),
        "dependency_environment":str(environment),"startup_library_path":str(environment/"lib"),
        "sealed_alias_parent":str(root/"workers"),
        "native_preload_paths":[r["path"] for r in components]+[str(native)]}
'''


class SealedBootstrap(unittest.TestCase):
    def run_case(self, body, *, code=0):
        with tempfile.TemporaryDirectory(prefix="psi4-sealed-bootstrap-") as temporary:
            root = Path(temporary)
            aliases = root/"workers/child"
            aliases.mkdir(parents=True, mode=0o700)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1",
                       LD_LIBRARY_PATH=str(root/"environment/lib"),
                       DISP0_SEALED_ALIAS_DIRECTORY=str(aliases))
            env.pop("LD_PRELOAD", None)
            run = subprocess.run(
                [sys.executable,"-B","-c",PREFIX+body,str(REPO),str(root)],
                env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(run.returncode,code,run.stdout+run.stderr)
            self.assertNotIn("UNSAFE",run.stdout)

    def test_normal_authenticated_support_import_and_all_symbols_precede_psi4(self):
        self.run_case(r'''
with patch.object(bootstrap,"NATIVE_SHA",native_sha):
    owner=bootstrap.prepare_sealed_owner(config,build,manifest)
import psi4_runtime_support
assert owner is psi4_runtime_support.prepared_runtime()
assert Path(psi4_runtime_support.__file__).parent==support
assert "psi4" not in sys.modules and "cuda" not in sys.modules
assert len(owner.report()["files"])==4
owner.verify_symbols(str(native),names)
assert (root/"workers/child/libcuest.so.0").is_symlink()
assert len(owner.report()["files"])==4
''')

    def test_changed_support_inventory_rejects_before_import_or_preload(self):
        self.run_case(r'''
(support/"sealed.py").write_text("# changed test fixture\n")
try:bootstrap.prepare_sealed_owner(config,build,manifest)
except RuntimeError as error:assert "inventory changed" in str(error),error
else:raise AssertionError("Changed support admitted")
assert "psi4_runtime_support" not in sys.modules
''')

    def test_wrong_startup_environment_rejects_before_native_import(self):
        self.run_case(r'''
os.environ["LD_LIBRARY_PATH"]+=":/foreign/toolkit"
try:bootstrap.prepare_sealed_owner(config,build,manifest)
except RuntimeError as error:assert "startup environment" in str(error),error
else:raise AssertionError("Unexpected loader environment admitted")
assert "psi4_runtime_support" not in sys.modules
''')

    def test_missing_native_symbol_is_fail_stop(self):
        self.run_case(r'''
native_sha=library(native,"int incomplete(void){return 0;}")
with patch.object(bootstrap,"NATIVE_SHA",native_sha):
    try:bootstrap.prepare_sealed_owner(config,build,manifest)
    finally:print("UNSAFE incomplete symbol cleanup",flush=True)
''',code=86)

    def test_source_loader_path_or_escaping_worker_directory_rejects(self):
        self.run_case(r'''
config["startup_library_path"]+=":"+str(cuda/"lib")
try:sealed_startup_path(config,root/"workers/child")
except RuntimeError as error:assert "only snapshots" in str(error),error
else:raise AssertionError("Source toolkit remained in the startup path")
config["startup_library_path"]=str(environment/"lib")
outside=root/"outside";outside.mkdir(mode=0o700)
try:sealed_startup_path(config,outside)
except RuntimeError as error:assert "per-worker" in str(error),error
else:raise AssertionError("Escaping alias directory accepted")
''')


if __name__ == "__main__":
    unittest.main()
