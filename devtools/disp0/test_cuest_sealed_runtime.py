"""CPU-only real-kernel/ELF coverage of the explicit sealed-runtime interface.

ELFs here are tiny C fixtures, NOT cuEST/CUDA and NOT GPU evidence.
Each dedicated worker owns its process-lifetime handles and exits after a case.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SAPT = Path(__file__).resolve().parents[2] / "psi4/driver/procrouting/sapt"
PREFIX = r'''
import ctypes as ct
import errno
import fcntl
import hashlib
import importlib
import mmap
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import types
from unittest.mock import patch
package=types.ModuleType("_sealed_runtime_fixture")
package.__path__=[sys.argv[1]]
sys.modules[package.__name__]=package
owned=importlib.import_module(package.__name__+".sapt_cuest_owned_runtime")
sealed=importlib.import_module("psi4_runtime_support.sealed")
identity=importlib.import_module(package.__name__+".sapt_cuest250_identity")
root=Path(sys.argv[2])
aliases=root/"aliases"
def library(name,body="int value(void) { return 7; }",flags=()):
    source=root/(name+".c")
    source.write_text(body)
    target=root/name
    subprocess.run(["cc","-shared","-fPIC",str(source),*flags,"-o",str(target)],check=True)
    return target,hashlib.sha256(target.read_bytes()).hexdigest()
path,checksum=library("libfixture.so")
pins={str(path):checksum}
def admit(selected=pins):
    from psi4_runtime_support import prepare_sealed_runtime
    prepare_sealed_runtime(selected,alias_directory=aliases,dedicated_worker=True)
    return owned.admit_sealed_runtime(selected,alias_directory=aliases,dedicated_worker=True)
'''


class SealedRuntime(unittest.TestCase):
    def run_case(self, body, *, code=0):
        with tempfile.TemporaryDirectory(prefix="psi4-sealed-cpu-") as temporary:
            root = Path(temporary)
            aliases = root / "aliases"
            aliases.mkdir(mode=0o700)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1",
                       LD_LIBRARY_PATH="")
            env.pop("LD_PRELOAD", None)
            run = subprocess.run(
                [sys.executable, "-B", "-c", PREFIX + body, str(SAPT), str(root)],
                env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(run.returncode, code, run.stdout + run.stderr)
            self.assertNotIn("UNSAFE", run.stdout)
            return run

    def test_loaded_image_survives_source_overwrite_removal_and_has_no_hot_file_io(self):
        self.run_case(r'''
owner=admit()
assert owned.current_owned_runtime() is owner and owned.owner_for_session() is owner
assert admit() is owner
owner.preload([str(path)])
native,logical=owner.load_library(path,checksum)
native.value.restype=ct.c_int
assert native.value()==7 and logical==str(path)
path.write_bytes(b"source is now different")
path.unlink()
before=owner.report()
with patch.object(sealed.os,"stat",side_effect=AssertionError("hot stat")), \
     patch.object(sealed.os,"fstat",side_effect=AssertionError("hot fstat")), \
     patch.object(sealed.fcntl,"fcntl",side_effect=AssertionError("hot fcntl")), \
     patch.object(sealed.Path,"resolve",side_effect=AssertionError("hot resolve")), \
     patch.object(sealed,"_descriptor_digest",side_effect=AssertionError("hot rehash")):
    for _ in range(100):
        owner.check_live()
        assert owner.require_library(str(path),checksum)==checksum
        assert owner.load_library(path,checksum)[0] is native
        assert owner.guarded_call(native.value)==7
after=owner.report()
assert before["full_hash_calls"]==after["full_hash_calls"]==2
assert after["contract"]==sealed.CONTRACT
assert signal.getsignal(signal.SIGIO)==signal.SIG_DFL
''')

    def test_kernel_seals_block_mutations_and_reopening_without_read_leases(self):
        self.run_case(r'''
owner=admit()
alias=aliases/path.name
fd=os.open(alias,os.O_RDWR)
try:
    size=os.fstat(fd).st_size
    for action in (lambda:os.pwrite(fd,b"X",0),lambda:os.ftruncate(fd,size+1),
                   lambda:os.ftruncate(fd,size-1),
                   lambda:mmap.mmap(fd,0,flags=mmap.MAP_SHARED,
                                    prot=mmap.PROT_READ|mmap.PROT_WRITE),
                   lambda:fcntl.fcntl(fd,sealed.F_ADD_SEALS,0)):
        try:action()
        except OSError as error:assert error.errno==errno.EPERM,error
        else:raise AssertionError("Snapshot mutation succeeded")
finally:os.close(fd)
owner.preload([str(path)])
native,_=owner.load_library(path,checksum)
assert native.value()==7
assert list(owner.report()["files"].values())[0]["seals"]&sealed.REQUIRED_SEALS==sealed.REQUIRED_SEALS
''')

    def test_source_byte_mismatch_rolls_back_without_publishing_then_can_admit(self):
        self.run_case(r'''
fds_before=set(os.listdir("/proc/self/fd"))
try:admit({str(path):"0"*64})
except RuntimeError as error:assert "SHA256" in str(error),error
else:raise AssertionError("Bad source admitted")
assert owned.current_owned_runtime() is None
assert not list(aliases.iterdir())
assert set(os.listdir("/proc/self/fd"))==fds_before
admit().check_live()
''')

    def test_source_changes_during_copy_are_rejected(self):
        self.run_case(r'''
read=sealed.os.read
changed=False
def mutate(fd,*args):
    global changed
    result=read(fd,*args)
    if result and not changed:
        changed=True
        with path.open("ab") as out:out.write(b"mutation during copy")
    return result
with patch.object(sealed.os,"read",side_effect=mutate):
    try:admit()
    except RuntimeError as error:assert "changed" in str(error),error
    else:raise AssertionError("Racing source admitted")
assert owned.current_owned_runtime() is None and not list(aliases.iterdir())
''')

    def test_aggregate_snapshot_budget_rejects_before_copy_and_rolls_back(self):
        self.run_case(r'''
with patch.object(sealed,"MAX_SNAPSHOT_BYTES",path.stat().st_size-1):
    try:admit()
    except MemoryError as error:assert "snapshot budget" in str(error),error
    else:raise AssertionError("Oversized closure admitted")
assert owned.current_owned_runtime() is None and not list(aliases.iterdir())
admit().check_live()
''')

    def test_preparation_serializes_legacy_selection_and_lease_admission(self):
        self.run_case(r'''
import psi4_runtime_support as support
assert owned._STATE_LOCK is support.selection_lock()
original=sealed._descriptor_digest
errors=[]
def hashing(fd):
    def competing_selection():
        for operation in (owned.owner_for_session,
                lambda:owned.admit_owned_runtime(pins,dedicated_worker=True)):
            try:operation()
            except RuntimeError as error:errors.append(str(error))
            else:raise AssertionError("Selected legacy/lease runtime during preparation")
    thread=threading.Thread(target=competing_selection)
    thread.start();thread.join(5)
    assert not thread.is_alive()
    return original(fd)
with patch.object(sealed,"_descriptor_digest",side_effect=hashing):
    owner=admit()
assert len(errors)==2 and not owned._LEGACY_SESSION_STARTED,errors
assert owned.owner_for_session() is owner
''')
        self.run_case(r'''
from psi4_runtime_support import prepare_sealed_runtime,sealed_preparation_active
original=owned._hash_descriptor
def hashing(fd):
    try:prepare_sealed_runtime(pins,alias_directory=aliases,dedicated_worker=True)
    except RuntimeError as error:assert "runtime/session selection" in str(error),error
    else:raise AssertionError("Sealed preparation raced active lease admission")
    return original(fd)
with patch.object(owned,"_hash_descriptor",side_effect=hashing):
    owner=owned.admit_owned_runtime(pins,dedicated_worker=True)
assert not sealed_preparation_active()
owner.check_live()
''')

    def test_short_snapshot_writes_are_completed_and_seal_failure_rolls_back(self):
        self.run_case(r'''
write=sealed.os.write
def partial(fd,data):
    return write(fd,data[:max(1,len(data)//3)])
with patch.object(sealed.os,"write",side_effect=partial):
    owner=admit()
owner.preload([str(path)])
assert owner.load_library(path,checksum)[0].value()==7
''')
        self.run_case(r'''
with patch.object(sealed.fcntl,"fcntl",side_effect=OSError(errno.EOPNOTSUPP,"no seals")):
    try:admit()
    except OSError as error:assert error.errno==errno.EOPNOTSUPP
    else:raise AssertionError("Unsupported seals admitted")
assert owned.current_owned_runtime() is None and not list(aliases.iterdir())
admit().check_live()
''')

    def test_fresh_kernel_rows_translate_only_exact_snapshot_identity(self):
        self.run_case(r'''
owner=admit();owner.preload([str(path)])
native,_=owner.load_library(path,checksum)
text=Path("/proc/self/maps").read_text()
rows=owner.mapped_rows(text)
address=ct.cast(native.value,ct.c_void_p).value
selected=[row for row in rows if row[0]<=address<row[1]]
assert len(selected)==1 and selected[0][2]==str(path)
assert owner.mapped_sha256(path,rows)==checksum
try:identity.mapped_rows(text)
except RuntimeError as error:assert "Deleted mapped binary" in str(error),error
else:raise AssertionError("Legacy contract admitted a deleted sealed ELF")
''')

    def test_changed_snapshot_identity_is_fail_stop_before_unwinding(self):
        self.run_case(r'''
owner=admit();owner.preload([str(path)])
text=Path("/proc/self/maps").read_text()
record=owner.report()["files"][str(path)]
changed=[]
for line in text.splitlines():
    fields=line.split(maxsplit=5)
    if len(fields)==6 and fields[5]==record["kernel_path"]:
        fields[4]=str(int(fields[4])+1)
        line=" ".join(fields)
    changed.append(line)
try:owner.mapped_rows("\n".join(changed))
finally:print("UNSAFE finalizer after foreign inode",flush=True)
''', code=86)

    def test_foreign_toolkit_mappings_reject_all_recognized_families(self):
        # Metadata parser checks, not CUDA loads; the file name alone cannot
        # authorize an unsealed inode even if a source locator was pinned.
        for name in ("libcudart.so.13", "libcudnn.so.9", "libcupti.so.13",
                     "libcuest.so.0.2.50"):
            with self.subTest(name=name):
                self.run_case(f'''
owner=admit()
owner.mapped_rows("1000-1001 r-xp 0 00:01 7 /foreign/{name}")
print("UNSAFE foreign toolkit accepted")
''', code=86)

    def test_admission_and_session_selection_cannot_mix_contracts(self):
        for mode in ("lease-first", "sealed-first", "legacy-first"):
            with self.subTest(mode=mode):
                self.run_case(f'''
mode={mode!r}
if mode=="lease-first":
    owned.admit_owned_runtime(pins,dedicated_worker=True)
    action=admit
elif mode=="sealed-first":
    admit()
    action=lambda:owned.admit_owned_runtime(pins,dedicated_worker=True)
else:
    assert owned.owner_for_session() is None
    action=admit
try:action()
except RuntimeError:pass
else:raise AssertionError("Contract/session changed in place")
''')

    def test_explicit_dedicated_worker_and_private_directory_required(self):
        self.run_case(r'''
try:owned.admit_sealed_runtime(pins,alias_directory=aliases)
except ValueError:pass
else:raise AssertionError("Implicit dedicated worker accepted")
aliases.chmod(0o755)
try:admit()
except RuntimeError as error:assert "mode0700" in str(error),error
else:raise AssertionError("Shared writable directory accepted")
assert owned.current_owned_runtime() is None
aliases.chmod(0o700)
admit().check_live()
''')

    def test_thread_or_fork_cannot_use_published_owner(self):
        self.run_case(r'''
owner=admit()
thread=threading.Thread(target=owner.check_live)
thread.start();thread.join()
print("UNSAFE foreign thread returned")
''', code=86)
        self.run_case(r'''
owner=admit()
owned._STATE_LOCK.acquire()
child=os.fork()
if child==0:
    owned.owner_for_session()
    print("UNSAFE child returned",flush=True)
    os._exit(0)
_,status=os.waitpid(child,0)
owned._STATE_LOCK.release()
assert os.waitstatus_to_exitcode(status)==86,status
owner.check_live()
''')

    def test_retained_loaded_handle_does_not_depend_on_alias_or_open_fd_lifetime(self):
        self.run_case(r'''
owner=admit();owner.preload([str(path)])
native,_=owner.load_library(path,checksum)
alias=aliases/path.name
fd=int(os.readlink(alias).rsplit("/",1)[1])
alias.unlink();os.close(fd)
# The loader still retains its own immutable inode mapping. No later dlopen
# uses this closed/reusable descriptor or the removed source/alias locator.
owner.check_live()
assert owner.load_library(path,checksum)[0] is native and native.value()==7
''')

    def test_unpreloaded_or_wrong_hash_fail_stop(self):
        for body in (r'''
owner=admit()
owner.load_library(path,checksum)
''', r'''
owner=admit();owner.preload([str(path)])
owner.require_library(path,"0"*64)
'''):
            self.run_case(body + '\nprint("UNSAFE invalid library returned")\n', code=86)

    def test_primary_alias_changes_do_not_affect_descriptor_bound_loads(self):
        self.run_case(r'''
owner=admit()
foreign,_=library("libforeign.so",
    '#include <unistd.h>\n'
    '__attribute__((constructor)) static void bad(void){'
    'write(1,"UNSAFE_FOREIGN_CONSTRUCTOR\\n",27);}\n'
    'int value(void){return 999;}')
alias=aliases/path.name
alias.unlink();alias.symlink_to(foreign)
load=sealed.ct.CDLL
first_loads=[]
def observed(locator,*args,**kwargs):
    if not kwargs.get("mode",0)&os.RTLD_NOLOAD:
        first_loads.append(locator)
    return load(locator,*args,**kwargs)
with patch.object(sealed.ct,"CDLL",side_effect=observed):
    owner.preload([str(path)])
assert first_loads and all(str(name).startswith("/proc/self/fd/") for name in first_loads)
assert owner.load_library(path,checksum)[0].value()==7
''')

    def test_substituted_dependency_alias_never_runs_its_foreign_constructor(self):
        self.run_case(r'''
dependency,dep_sha=library("libdep.so","int dep(void){return 7;}",
                          ["-Wl,-soname,libdep.so"])
parent,parent_sha=library("libmain.so",
    "extern int dep(void); int value(void){return dep();}",
    ["-L"+str(root),"-ldep","-Wl,-rpath,$ORIGIN"])
foreign,_=library("libforeign.so",
    '#include <unistd.h>\n'
    '__attribute__((constructor)) static void bad(void){'
    'write(1,"UNSAFE_FOREIGN_CONSTRUCTOR\\n",27);}\n'
    'int dep(void){return 999;}')
owner=admit({str(dependency):dep_sha,str(parent):parent_sha})
alias=aliases/dependency.name
alias.unlink();alias.symlink_to(foreign)
owner.preload([str(parent)])
native,_=owner.load_library(parent,parent_sha)
assert native.value()==7,"Foreign dependency was used"
owner.verify_symbols(str(dependency),["dep"])
''')

    def test_unpinned_dependency_or_missing_soname_rejects_before_constructors(self):
        for selected_dependency in (False,True):
            with self.subTest(pinned_without_soname=selected_dependency):
                self.run_case(f'''
dependency,dep_sha=library("libdep.so",
    '#include <unistd.h>\\n'
    '__attribute__((constructor)) static void bad(void){{'
    'write(1,"UNSAFE_DEPENDENCY_CONSTRUCTOR\\\\n",29);}}\\n'
    'int dep(void){{return 7;}}')
parent,parent_sha=library("libmain.so",
    "extern int dep(void); int value(void){{return dep();}}",
    ["-L"+str(root),"-ldep","-Wl,-rpath,$ORIGIN"])
selected={{str(parent):parent_sha}}
if {selected_dependency!r}:selected[str(dependency)]=dep_sha
owner=admit(selected)
if not {selected_dependency!r}:(aliases/dependency.name).symlink_to(dependency)
owner.preload([str(parent)])
print("UNSAFE unsupported dependency admitted",flush=True)
''',code=86)

    def test_existing_foreign_soname_rejects_before_parent_constructor(self):
        self.run_case(r'''
foreign,_=library("libforeign.so","int dep(void){return 999;}",
                   ["-Wl,-soname,libdep.so"])
foreign_handle=ct.CDLL(str(foreign),mode=os.RTLD_NOW|os.RTLD_GLOBAL)
dependency,dep_sha=library("libdep.so","int dep(void){return 7;}",
                          ["-Wl,-soname,libdep.so"])
parent,parent_sha=library("libmain.so",
    '#include <unistd.h>\n'
    '__attribute__((constructor)) static void bad(void){'
    'write(1,"UNSAFE_PARENT_CONSTRUCTOR\\n",26);}\n'
    'extern int dep(void); int value(void){return dep();}',
    ["-L"+str(root),"-ldep","-Wl,-rpath,$ORIGIN"])
owner=admit({str(dependency):dep_sha,str(parent):parent_sha})
owner.preload([str(parent)])
print("UNSAFE foreign SONAME reused",flush=True)
''',code=86)

    def test_filesystem_rpath_is_rejected_during_preparation_and_rolls_back(self):
        self.run_case(r'''
parent,parent_sha=library("libmain.so",flags=["-Wl,-rpath,"+str(root)])
try:admit({str(parent):parent_sha})
except ValueError:pass
else:raise AssertionError("Filesystem RPATH admitted")
assert owned.current_owned_runtime() is None and not list(aliases.iterdir())
admit().check_live()
''')

    def test_elf_loader_extensions_cannot_add_an_unpinned_world(self):
        for option in ("--filter=libforeign.so","--audit=libforeign.so"):
            with self.subTest(option=option):
                self.run_case(f'''
parent,parent_sha=library("libmain.so",flags=["-Wl,{option}"])
try:admit({{str(parent):parent_sha}})
except ValueError as error:assert "loader extensions" in str(error),error
else:raise AssertionError("Unmodeled loader extension admitted")
assert owned.current_owned_runtime() is None and not list(aliases.iterdir())
''')

    def test_actual_identity_adapter_accepts_only_snapshot_backed_cu_est_and_cuda(self):
        self.run_case(r'''
functions=("cuestGetMajorVersion","cuestGetMinorVersion","cuestGetPatchVersion",
    "cuestGetMathMode","cuestParametersCreate","cuestParametersDestroy",
    "cuestDFMOIntegralsCompute","cuestDFMOIntegralsComputeWorkspaceQuery")
native_path,native_sha=library("libcuest.so.0.2.50",
    "\n".join("int "+name+"(void) { return 0; }" for name in functions))
components=[]
for name in ("libcudart.so.13","libcublas.so.13","libcublasLt.so.13"):
    p,sha=library(name)
    components.append({"path":str(p),"sha256":sha})
selected={str(native_path):native_sha,**{r["path"]:r["sha256"] for r in components}}
owner=admit(selected)
owner.preload(list(selected))
native,_=owner.load_library(native_path,native_sha)
manifest={"cuda_root":str(root),"components":components,
          "cuda_runtime_version":13030,"nvrtc_version":[13,3]}
# Explicit CPU-fixture native hash; no real SDK/device is being admitted.
with patch.object(identity,"NATIVE_SHA256",native_sha):
    result=identity.audit_maps(native,{"native_path":str(native_path)},manifest,owner=owner)
assert result["identity_contract"]==sealed.CONTRACT
assert result["cuda_mapped_sha256"]=={r["path"]:r["sha256"] for r in components}
assert "psi4" not in sys.modules and "cuda" not in sys.modules
''')

    def test_normal_support_import_prepares_before_psi4_and_exact_owner_is_adopted(self):
        self.run_case(r'''
from psi4_runtime_support import prepare_sealed_runtime
prepared=prepare_sealed_runtime(pins,alias_directory=aliases,dedicated_worker=True)
assert "psi4" not in sys.modules
prepared.preload([str(path)])
try:owned.owner_for_session()
except RuntimeError as error:assert "must be adopted" in str(error),error
else:raise AssertionError("Selected a legacy session during sealed handoff")
# A package-import marker is enough for this CPU handoff fixture. No scientific
# module is overlaid/injected in a real bootstrap.
sys.modules["psi4"]=types.ModuleType("psi4")
assert admit() is prepared
assert owned.owner_for_session() is prepared
assert prepared.load_library(path,checksum)[0].value()==7
''')


if __name__ == "__main__":
    unittest.main()
