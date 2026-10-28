"""Real Linux lease/lifetime tests in isolated CPU-only worker processes."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest


SAPT = Path(__file__).resolve().parents[2]/"psi4/driver/procrouting/sapt"
PREFIX = r'''
import errno, fcntl, functools, hashlib, importlib, json, mmap, os, signal, stat, sys
import tempfile, threading, types
from pathlib import Path
from unittest.mock import Mock, patch
package=types.ModuleType("_owned_runtime_kernel_tests")
package.__path__=[sys.argv[1]]
sys.modules[package.__name__]=package
owned=importlib.import_module(package.__name__+".sapt_cuest_owned_runtime")
raw_admit=owned.admit_owned_runtime
owned.admit_owned_runtime=functools.partial(raw_admit,dedicated_worker=True)
root=Path(sys.argv[2])  # Parent owns cleanup even after deliberate os._exit.
path=root/"libfixture.so.1"
payload=b"CPU immutable library fixture; not an ELF"
path.write_bytes(payload)
checksum=hashlib.sha256(payload).hexdigest()
pins={str(path):checksum}
def mapping(p, *, inode_delta=0, device=None):
    s=p.stat()
    return (4096,8192,str(p),s.st_ino+inode_delta,
            (os.major(s.st_dev),os.minor(s.st_dev)) if device is None else device)
def invalid(function, *args, **kwargs):
    try:
        function(*args,**kwargs)
    except RuntimeError as error:
        assert "new process required" in str(error),str(error)
        return error
    raise AssertionError("invalidated immutable proof remained usable")
def pinned_descriptors():
    result=[]
    for entry in Path("/proc/self/fd").iterdir():
        try:
            if os.readlink(entry)==str(path):result.append(int(entry.name))
        except FileNotFoundError:
            pass
    return result
'''


@unittest.skipUnless(sys.platform.startswith("linux"),"Linux kernel lease contract")
class OwnedRuntime(unittest.TestCase):
    def run_case(self, body, *, revoked=False, notification=False, executable=False):
        with tempfile.TemporaryDirectory(prefix="disp0-owned-kernel-",
                dir="/dev/shm" if not executable and Path("/dev/shm").is_dir() else None) as directory:
            completed = subprocess.run(
                [sys.executable,"-B","-c",PREFIX+"\n"+textwrap.dedent(body),str(SAPT),directory],
                capture_output=True,text=True,timeout=15,
                env={**os.environ,"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
            expected = int(signal.SIGIO) if notification else 86 if revoked else 0
            self.assertEqual(completed.returncode,expected,completed.stdout+"\n"+completed.stderr)
            if revoked or notification:
                self.assertNotIn("UNSAFE",completed.stdout)
            # Process exit must release every lease, including rollback faults.
            for path in Path(directory).rglob("*"):
                if path.is_file() and not path.is_symlink():
                    path.chmod(0o600)
                    descriptor = os.open(path,os.O_WRONLY|os.O_NONBLOCK)
                    os.close(descriptor)
            return completed

    def test_native_load_uses_pinned_descriptor_not_retargeted_library_alias(self):
        self.run_case('''
            import ctypes as ct
            import subprocess
            source=root/"fixture.c"
            source.write_text("int owned_fixture(void) { return 7; }")
            subprocess.run(["cc","-shared","-fPIC",str(source),"-o",str(path)],check=True)
            other=root/"other.so"
            source.write_text("int owned_fixture(void) { return 42; }")
            subprocess.run(["cc","-shared","-fPIC",str(source),"-o",str(other)],check=True)
            checksum=hashlib.sha256(path.read_bytes()).hexdigest()
            pins={str(path):checksum}
            alias=root/"libcusolver.so.13"
            alias.symlink_to(path)
            # Bootstrap may already have loaded the actual file.
            original=ct.CDLL(str(path))
            owner=owned.admit_owned_runtime(pins)
            real_load=owned.ct.CDLL
            def racing_load(locator, *, mode):
                alias.unlink()
                alias.symlink_to(other)
                assert locator.startswith("/proc/self/fd/"),locator
                assert mode & os.RTLD_NOLOAD
                return real_load(locator,mode=mode)
            with patch.object(owned.ct,"CDLL",side_effect=racing_load):
                loaded,canonical=owner.load_library(alias,checksum)
            assert canonical==str(path)
            assert loaded._handle==original._handle
            assert loaded.owned_fixture()==7
            assert owner.report()["full_hash_calls"]==1
            # The actual cuSOLVER constructor must pass this same owner into
            # descriptor loading, not re-open its manifest's mutable alias.
            power=importlib.import_module(package.__name__+".sapt_cuda_symmetric_power")
            native_owner=power.DiagnosticPlanOwner.__new__(power.DiagnosticPlanOwner)
            native_owner._check_construction=Mock()
            native_owner.assert_live=Mock()
            native_owner.session=types.SimpleNamespace(identity_owner=owner,stream=9)
            native_owner.api=types.SimpleNamespace(status=Mock(),
                manifest={"components":[{"path":str(alias),"sha256":checksum}]})
            native_owner._new_native=Mock(return_value=object())
            native_owner._raw=lambda value:ct.c_void_p(9)
            loaded.cusolverDnCreate=Mock()
            loaded.cusolverDnDestroy=Mock()
            loaded.cusolverDnSetStream=Mock(return_value=0)
            alias.unlink()
            alias.symlink_to(path)
            def solver_load(locator, *, mode):
                racing_load(locator,mode=mode)
                return loaded
            with patch.object(owned.ct,"CDLL",side_effect=solver_load), \
                    patch.object(power.CuSolverSymmetricPower,"_bind"):
                solver=power.CuSolverSymmetricPower(native_owner)
            assert solver.native.owned_fixture()==7
            assert solver.path==str(path)
        ''',executable=True)

    def test_origin_dependency_closure_must_be_preloaded_before_descriptor_reuse(self):
        for state in ("absent","mmap","linker"):
            self.run_case(f'''
                import ctypes as ct
                import subprocess
                child=root/"libchild.so"
                source=root/"child.c"
                source.write_text("int child(void) {{ return 7; }}")
                subprocess.run(["cc","-shared","-fPIC",str(source),"-o",str(child)],check=True)
                source=root/"parent.c"
                source.write_text("extern int child(void); int parent(void) {{ return child(); }}")
                subprocess.run(["cc","-shared","-fPIC",str(source),"-L"+str(root),"-lchild",
                    "-Wl,-rpath,$ORIGIN","-o",str(path)],check=True)
                pins={{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (path,child)}}
                checksum=pins[str(path)]
                state={state!r}
                if state=="mmap":
                    views=[]
                    for filename in (path,child):
                        with filename.open("rb") as stream:
                            views.append(mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ))
                if state=="linker":
                    original=ct.CDLL(str(path))
                    assert original.parent()==7
                owner=owned.admit_owned_runtime(pins)
                loaded,_=owner.load_library(path,checksum)
                assert state=="linker"
                assert loaded._handle==original._handle
                assert loaded.parent()==7
            ''',executable=True,revoked=state!="linker")

    def test_full_stderr_cannot_prevent_fail_stop_worker_retirement(self):
        with tempfile.TemporaryDirectory(prefix="disp0-owned-stderr-") as directory:
            read,write=os.pipe()
            process=None
            try:
                os.set_blocking(write,False)
                try:
                    while True:os.write(write,b"x"*4096)
                except BlockingIOError:
                    pass
                os.set_blocking(write,True)
                body="owner=owned.admit_owned_runtime(pins)\npath.chmod(0o400)\nowner.check_live()\n"
                process=subprocess.Popen([sys.executable,"-B","-c",PREFIX+body,str(SAPT),directory],
                    stderr=write,stdout=subprocess.PIPE)
                self.assertEqual(process.wait(timeout=5),86)
                self.assertEqual(process.stdout.read(),b"")
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=5)
                    process.stdout.close()
                os.close(write)
                os.close(read)

    def test_no_legacy_session_can_start_while_byte_admission_releases_the_gil(self):
        self.run_case('''
            entered=threading.Event()
            checked=threading.Event()
            errors=[]
            original=owned._hash_descriptor
            def hashing(fd):
                entered.set()
                assert checked.wait(5)
                return original(fd)
            def select_session():
                assert entered.wait(5)
                try:owned.owner_for_session()
                except RuntimeError as error:errors.append(str(error))
                finally:checked.set()
            thread=threading.Thread(target=select_session)
            thread.start()
            with patch.object(owned,"_hash_descriptor",side_effect=hashing):
                owner=owned.admit_owned_runtime(pins)
            thread.join(5)
            assert len(errors)==1 and "during immutable admission" in errors[0],errors
            assert not owned._LEGACY_SESSION_STARTED
            assert not owned._ADMISSION_IN_PROGRESS
            assert owned.owner_for_session() is owner
        ''')
        self.run_case('''
            owner=owned.admit_owned_runtime(pins)
            owned._STATE_LOCK.acquire()
            child=os.fork()
            if child==0:
                owned.owner_for_session()
                os._exit(99)
            owned._STATE_LOCK.release()
            _,status=os.waitpid(child,0)
            assert os.waitstatus_to_exitcode(status)==86,status
            owner.check_live()  # Child never unlocked the parent's lease.
        ''')

    def test_file_admission_preserves_header_archive_checks_without_rehashing_owned_native(self):
        self.run_case('''
            identity=importlib.import_module(package.__name__+".sapt_cuest250_identity")
            sdk=root/"sdk"
            (sdk/"include").mkdir(parents=True)
            native=sdk/"lib/13/libcuest.so.0.2.50"
            native.parent.mkdir(parents=True)
            path.rename(native)
            owner=owned.admit_owned_runtime({str(native):checksum})
            with patch.object(identity,"canonical_headers",return_value={}) as archive_check, \
                    patch.object(identity,"NATIVE_SHA256",checksum), \
                    patch.object(identity,"file_digest",wraps=identity.file_digest) as hashes:
                for _ in range(2):
                    proof=identity.verify_files(sdk,root/"archive",owner=owner)
                    assert proof["native_sha256"]==checksum
                assert archive_check.call_count==2
                assert not any(str(call.args[0])==str(native) for call in hashes.call_args_list)
            assert owner.report()["full_hash_calls"]==1
        ''')

    def test_native_cleanup_checks_consumed_break_before_any_native_entry(self):
        for kind in ("plan","provider"):
            self.run_case(f'''
                plans=importlib.import_module(package.__name__+".sapt_cuest_plan_owner")
                provider=importlib.import_module(package.__name__+".sapt_cuest_dfmo_cuda")
                owner=owned.admit_owned_runtime(pins)
                assert len(list(Path("/proc/self/task").iterdir()))==1
                previous=signal.pthread_sigmask(signal.SIG_BLOCK,{{signal.SIGIO}})
                try:
                    try:os.open(path,os.O_WRONLY|os.O_NONBLOCK)
                    except BlockingIOError:pass
                    else:raise AssertionError("write lease unexpectedly admitted")
                    assert signal.sigtimedwait({{signal.SIGIO}},1) is not None
                finally:signal.pthread_sigmask(signal.SIG_SETMASK,previous)
                assert os.pread(owner._files[str(path)].fd,len(payload),0)==payload
                def forbidden(*args):
                    print("UNSAFE native cleanup entry",flush=True)
                    raise AssertionError("native cleanup preceded owned guard")
                session=types.SimpleNamespace(identity_owner=owner,
                    runtime=types.SimpleNamespace(cudaGetDevice=forbidden))
                if {kind!r}=="plan":
                    plans._cleanup_admission(types.SimpleNamespace(session=session,pid=os.getpid()))
                else:
                    value=provider.CuESTDFMOProvider.__new__(provider.CuESTDFMOProvider)
                    value.session=session
                    value._device_owner=True
                    value.cleanup_complete=False
                    value.close()
                print("UNSAFE cleanup returned",flush=True)
            ''',revoked=True)

    def test_hash_once_under_real_read_lease_and_keep_receipts_independent(self):
        self.run_case('''
            # The same pre-existing read mapping as a loaded ELF is permitted.
            with path.open("rb") as stream:
                view=mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ)
            owner=owned.admit_owned_runtime(pins)
            assert owned.current_owned_runtime() is owner
            assert owned.owner_for_session() is owner
            assert owned.admit_owned_runtime(dict(pins)) is owner
            with patch.object(owned,"_hash_descriptor",side_effect=AssertionError("rehashed")):
                for _ in range(2):
                    owner.check_live()
                    owner.require_library(path,checksum)
                    assert owner.mapped_sha256(str(path),[mapping(path)])==checksum
                receipt=owner.report()
            assert receipt["full_hash_calls"]==1
            assert receipt["full_hash_bytes"]==len(payload)
            assert receipt["mapped_checks"]==2
            assert not receipt["invalidated"]
            receipt["files"][str(path)]["sha256"]="corrupted report"
            owner.require_library(path,checksum)
            assert "psi4" not in sys.modules and "cuda" not in sys.modules
            view.close()
        ''')

    def test_write_open_is_blocked_and_revokes_before_any_content_change(self):
        self.run_case('''
            owner=owned.admit_owned_runtime(pins)
            try:
                fd=os.open(path,os.O_WRONLY|os.O_NONBLOCK)
            except BlockingIOError:
                pass
            else:
                os.close(fd)
                raise AssertionError("kernel allowed write under a read lease")
            assert path.read_bytes()==payload
            invalid(owner.check_live)
            assert owner.report()["invalidated"]
            invalid(owned.admit_owned_runtime,pins)
            invalid(owner.mapped_sha256,str(path),[mapping(path)])
            assert owner.report()["full_hash_calls"]==1
        ''',notification=True)

    def test_pending_break_is_detected_even_when_sigio_has_not_run(self):
        self.run_case('''
            owner=owned.admit_owned_runtime(pins)
            original=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGIO})
            try:
                try:
                    fd=os.open(path,os.O_WRONLY|os.O_NONBLOCK)
                except BlockingIOError:
                    pass
                else:
                    os.close(fd)
                    raise AssertionError("kernel allowed write")
                # Consume pending SIGIO without invoking the native exit handler.
                assert signal.sigtimedwait({signal.SIGIO},1) is not None
                assert not owner.failed
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK,original)
            invalid(owner.check_live)  # F_GETLEASE reports target F_UNLCK.
        ''',revoked=True)

    def test_existing_writable_descriptor_and_shared_mapping_reject_admission(self):
        for kind in ("descriptor","mapping"):
            self.run_case(f'''
                writer=os.open(path,os.O_RDWR)
                view=None
                if {kind!r}=="mapping":
                    view=mmap.mmap(writer,0,flags=mmap.MAP_SHARED,prot=mmap.PROT_READ|mmap.PROT_WRITE)
                    os.close(writer)
                    writer=None
                previous=signal.getsignal(signal.SIGIO)
                try:
                    owned.admit_owned_runtime(pins)
                except RuntimeError as error:
                    assert "Cannot establish immutable read lease" in str(error),str(error)
                else:
                    raise AssertionError("existing writer was admitted")
                assert owned.current_owned_runtime() is None
                assert signal.getsignal(signal.SIGIO)==previous
                if writer is not None:os.close(writer)
                if view is not None:view.close()
                owner=owned.admit_owned_runtime(pins)
                owner.check_live()
            ''')

    def test_byte_mismatch_and_unsupported_leases_roll_back_without_fd_leaks(self):
        for kind in ("bytes","unsupported","fstat"):
            self.run_case(f'''
                previous=signal.getsignal(signal.SIGIO)
                assert pinned_descriptors()==[]
                original=fcntl.fcntl
                def unsupported(fd,command,*args):
                    if command==fcntl.F_SETLEASE:
                        raise OSError(errno.EOPNOTSUPP,"unsupported fixture filesystem")
                    return original(fd,command,*args)
                mode={kind!r}
                try:
                    if mode=="bytes":
                        owned.admit_owned_runtime({{str(path):"0"*64}})
                    elif mode=="fstat":
                        with patch.object(owned.os,"fstat",side_effect=OSError(errno.EIO,"fstat failed")):
                            owned.admit_owned_runtime(pins)
                    else:
                        with patch.object(owned.fcntl,"fcntl",side_effect=unsupported):
                            owned.admit_owned_runtime(pins)
                except (RuntimeError,OSError):
                    pass
                else:
                    raise AssertionError("bad admission succeeded")
                assert pinned_descriptors()==[],pinned_descriptors()
                assert signal.getsignal(signal.SIGIO)==previous
                assert owned.current_owned_runtime() is None
                owner=owned.admit_owned_runtime(pins)
                owner.check_live()
            ''')

    def test_replacement_metadata_or_descriptor_loss_permanently_invalidates(self):
        for kind in ("replace","chmod","descriptor"):
            self.run_case(f'''
                owner=owned.admit_owned_runtime(pins)
                mode={kind!r}
                if mode=="replace":
                    replacement=root/"replacement"
                    replacement.write_bytes(payload)
                    replacement.replace(path)
                elif mode=="chmod":
                    path.chmod(0o400)
                else:
                    descriptors=pinned_descriptors()
                    assert len(descriptors)==1,descriptors
                    os.close(descriptors[0])
                invalid(owner.check_live)
                invalid(owned.admit_owned_runtime,pins)
                assert owner.report()["invalidated"]
            ''',revoked=True)

    def test_wrong_mapping_library_or_signal_owner_cannot_reuse_byte_proof(self):
        for kind in ("inode","device","unmapped","unowned","hash","signal",
                     "notification-owner","notification-signal"):
            self.run_case(f'''
                owner=owned.admit_owned_runtime(pins)
                mode={kind!r}
                if mode=="inode":
                    invalid(owner.mapped_sha256,str(path),[mapping(path,inode_delta=1)])
                elif mode=="device":
                    invalid(owner.mapped_sha256,str(path),[mapping(path,device=(65534,65533))])
                elif mode=="unmapped":
                    invalid(owner.mapped_sha256,str(path),[])
                elif mode=="unowned":
                    invalid(owner.mapped_sha256,str(root/"foreign.so"),[mapping(path)])
                elif mode=="hash":
                    invalid(owner.require_library,path,"0"*64)
                elif mode=="notification-owner":
                    fcntl.fcntl(pinned_descriptors()[0],fcntl.F_SETOWN,0)
                    invalid(owner.check_live)
                elif mode=="notification-signal":
                    fcntl.fcntl(pinned_descriptors()[0],fcntl.F_SETSIG,signal.SIGUSR1)
                    invalid(owner.check_live)
                else:
                    previous=signal.signal(signal.SIGIO,signal.SIG_IGN)
                    invalid(owner.check_live)
                    signal.signal(signal.SIGIO,previous)
                invalid(owner.check_live)
                invalid(owned.admit_owned_runtime,pins)
            ''',revoked=True)

    def test_fork_rejects_child_without_unlocking_the_parent_lease(self):
        self.run_case('''
            owner=owned.admit_owned_runtime(pins)
            child=os.fork()
            if child==0:
                try:
                    invalid(owner.check_live)
                    invalid(owned.admit_owned_runtime,pins)
                except BaseException:
                    os._exit(31)
                os._exit(0)
            _,status=os.waitpid(child,0)
            assert os.waitstatus_to_exitcode(status)==owned.REVOKED_EXIT_CODE,status
            owner.check_live()  # Child must never issue F_UNLCK on shared OFD.
            assert not owner.report()["invalidated"]
        ''')

    def test_owned_descriptors_are_not_inherited_across_exec(self):
        self.run_case('''
            import subprocess
            owner=owned.admit_owned_runtime(pins)
            code=("import os,sys\\n"
                  "from pathlib import Path\\n"
                  "for p in Path('/proc/self/fd').iterdir():\\n"
                  "    try:target=os.readlink(p)\\n"
                  "    except FileNotFoundError:continue\\n"
                  "    assert target!=sys.argv[1], 'owned descriptor escaped exec'\\n")
            result=subprocess.run([sys.executable,"-B","-c",code,str(path)],
                close_fds=False,capture_output=True,text=True,timeout=5)
            assert result.returncode==0,result.stderr
            owner.check_live()
        ''')

    def test_foreign_thread_revokes_but_nonmain_admission_leaves_no_owner(self):
        self.run_case('''
            errors=[]
            def admission():
                try:owned.admit_owned_runtime(pins)
                except RuntimeError as error:errors.append(str(error))
            worker=threading.Thread(target=admission)
            worker.start();worker.join()
            assert len(errors)==1 and "main thread" in errors[0],errors
            assert owned.current_owned_runtime() is None
            owner=owned.admit_owned_runtime(pins)
            def misuse():
                try:owner.check_live()
                except RuntimeError as error:errors.append(str(error))
            worker=threading.Thread(target=misuse)
            worker.start();worker.join()
            assert len(errors)==2 and "Foreign thread" in errors[1],errors
            invalid(owner.check_live)
        ''',revoked=True)

    def test_guarded_call_preserves_ordinary_errors_and_exits_on_revocation(self):
        self.run_case('''
            owner=owned.admit_owned_runtime(pins)
            assert owner.guarded_call(lambda value:value+1,3)==4
            primary=ValueError("ordinary native input error")
            def failed():
                raise primary
            try:owner.guarded_call(failed)
            except ValueError as error:assert error is primary
            else:raise AssertionError("primary error lost")
            owner.check_live()
            primary=KeyboardInterrupt("original interruption")
            def revoked_failure():
                path.chmod(0o400)
                raise primary
            try:owner.guarded_call(revoked_failure)
            except KeyboardInterrupt as error:
                assert error is primary
                assert any("post-call check" in note for note in error.__notes__)
            else:raise AssertionError("primary error masked")
            invalid(owner.check_live)
        ''',revoked=True)
        self.run_case('''
            owner=owned.admit_owned_runtime(pins)
            def would_publish():
                path.chmod(0o400)
                return "MUST NOT ESCAPE"
            invalid(owner.guarded_call,would_publish)
        ''',revoked=True)

    def test_preexisting_signal_owner_wrong_pins_or_late_upgrade_reject(self):
        self.run_case('''
            previous=signal.signal(signal.SIGIO,lambda *_:None)
            try:
                try:owned.admit_owned_runtime(pins)
                except RuntimeError as error:assert "unclaimed" in str(error),str(error)
                else:raise AssertionError("clobbered caller SIGIO")
                assert owned.current_owned_runtime() is None
            finally:
                signal.signal(signal.SIGIO,previous)
            owner=owned.admit_owned_runtime(pins)
            other=root/"other.so"
            other.write_bytes(payload)
            try:owned.admit_owned_runtime({str(other):checksum})
            except RuntimeError as error:assert "different immutable runtime" in str(error),str(error)
            else:raise AssertionError("changed runtime within process")
            owner.check_live()
        ''')
        self.run_case('''
            assert owned.owner_for_session() is None
            try:owned.admit_owned_runtime(pins)
            except RuntimeError as error:assert "legacy CUDA session" in str(error),str(error)
            else:raise AssertionError("upgraded a live native process")
            assert owned.current_owned_runtime() is None
        ''')

    def test_invalid_configuration_has_no_admission_or_signal_side_effect(self):
        self.run_case('''
            previous=signal.getsignal(signal.SIGIO)
            try:raw_admit(pins)
            except ValueError as error:assert "dedicated_worker=True" in str(error)
            else:raise AssertionError("fatal worker policy was not acknowledged")
            alias=root/"alias.so";alias.symlink_to(path.name)
            for invalid_pins in ({},{str(path):"bad"}, {"relative.so":checksum},
                                 {str(path):checksum,str(alias):checksum}):
                try:owned.admit_owned_runtime(invalid_pins)
                except ValueError:pass
                else:raise AssertionError("bad pins admitted")
                assert signal.getsignal(signal.SIGIO)==previous
                assert owned.current_owned_runtime() is None
                assert pinned_descriptors()==[]
        ''')

    def test_rejected_results_and_callable_finally_never_enter_native_style_teardown(self):
        self.run_case('''
            owner=owned.admit_owned_runtime(pins)
            class NativeResult:
                def __del__(self):
                    os.write(1,b"UNSAFE native result destructor\\n")
            def result():
                path.chmod(0o400)
                return NativeResult()
            owner.guarded_call(result)
            print("UNSAFE result published",flush=True)
        ''',revoked=True)
        self.run_case('''
            owner=owned.admit_owned_runtime(pins)
            def result():
                try:
                    os.open(path,os.O_WRONLY|os.O_NONBLOCK)
                finally:
                    os.write(1,b"UNSAFE callable cleanup\\n")
            owner.guarded_call(result)
            print("UNSAFE result published",flush=True)
        ''',notification=True)

    def test_terminal_receipt_checks_do_not_return_after_suppressed_notification(self):
        self.run_case('''
            owner=owned.admit_owned_runtime(pins)
            row=mapping(path)
            class RevokingMappings:
                def __iter__(self):
                    owner.failed=True
                    owner.reason="simulated notification during mapping iteration"
                    yield row
            owner.mapped_sha256(str(path),RevokingMappings())
            print("UNSAFE digest receipt",flush=True)
        ''',revoked=True)
        self.run_case('''
            owner=owned.admit_owned_runtime(pins)
            original=Path.stat
            def changed(self,*args,**kwargs):
                value=original(self,*args,**kwargs)
                if self==path:
                    owner.failed=True
                    owner.reason="simulated notification during stat"
                return value
            with patch.object(Path,"stat",changed):
                owner.require_library(path,checksum)
            print("UNSAFE library receipt",flush=True)
        ''',revoked=True)

    def test_multifile_admission_rollback_attempts_all_closes_after_baseexception(self):
        self.run_case('''
            other=root/"other.so"
            other.write_bytes(payload)
            before=signal.getsignal(signal.SIGIO)
            original=os.close
            calls=[]
            def interrupted(fd):
                original(fd)
                calls.append(fd)
                if len(calls)==1:raise KeyboardInterrupt("after observed close")
            with patch.object(owned.os,"close",side_effect=interrupted):
                try:owned.admit_owned_runtime({str(path):"0"*64,str(other):checksum})
                except RuntimeError as error:
                    assert "bytes differ" in str(error),str(error)
                    assert any("rollback also failed" in note for note in error.__notes__)
                else:raise AssertionError("bad admission succeeded")
            assert len(calls)==2,calls
            assert signal.getsignal(signal.SIGIO)==before
            assert owned.current_owned_runtime() is None
            assert pinned_descriptors()==[]
            owned.admit_owned_runtime({str(path):checksum,str(other):checksum}).check_live()
        ''')
        completed=self.run_case('''
            other=root/"other.so"
            other.write_bytes(payload)
            original=os.close
            calls=[]
            def interrupted(fd):
                calls.append(fd)
                if len(calls)==1:raise KeyboardInterrupt("before unknown close")
                original(fd)
                os.write(1,b"SECOND_CLOSE_ATTEMPTED\\n")
            with patch.object(owned.os,"close",side_effect=interrupted):
                owned.admit_owned_runtime({str(path):"0"*64,str(other):checksum})
            print("UNSAFE incomplete rollback returned",flush=True)
        ''',revoked=True)
        self.assertIn("SECOND_CLOSE_ATTEMPTED",completed.stdout)

    def test_notification_during_hash_or_rollback_never_publishes_an_owner(self):
        for phase in ("hash","rollback"):
            self.run_case(f'''
                previous=signal.getsignal(signal.SIGIO)
                phase={phase!r}
                original_hash=owned._hash_descriptor
                original_close=os.close
                def hashing(fd):
                    result=original_hash(fd)
                    os.kill(os.getpid(),signal.SIGIO)
                    return result
                def closing(fd):
                    original_close(fd)
                    os.kill(os.getpid(),signal.SIGIO)
                try:
                    if phase=="hash":
                        with patch.object(owned,"_hash_descriptor",side_effect=hashing):
                            owned.admit_owned_runtime(pins)
                    else:
                        with patch.object(owned.os,"close",side_effect=closing):
                            owned.admit_owned_runtime({{str(path):"0"*64}})
                except RuntimeError:pass
                else:raise AssertionError("failed admission was published")
                assert owned.current_owned_runtime() is None
                assert signal.getsignal(signal.SIGIO)==previous
                assert pinned_descriptors()==[]
            ''',notification=True)

    def test_actual_kernel_sigio_owner_is_checked_not_only_python_cached_disposition(self):
        self.run_case('''
            probe=owned._KernelSignal()
            assert probe.current()==0
            assert signal.getsignal(signal.SIGIO)==signal.SIG_DFL
            probe.set(1)  # Native code installs SIG_IGN; Python's cache stays DFL.
            try:owned.admit_owned_runtime(pins)
            except RuntimeError as error:assert "actual kernel SIGIO" in str(error),str(error)
            else:raise AssertionError("native signal owner was overwritten")
            assert probe.current()==1
            assert signal.getsignal(signal.SIGIO)==signal.SIG_DFL
            probe.set(0)
            owner=owned.admit_owned_runtime(pins)
            assert probe.current()==probe.exit_address
            assert signal.getsignal(signal.SIGIO)==signal.SIG_DFL
            probe.set(1)
            invalid(owner.check_live)
        ''',revoked=True)

    def test_owned_mapping_adapter_keeps_legacy_fresh_hashes_and_revokes_on_drift(self):
        self.run_case('''
            sys.path.insert(0,str(Path(sys.argv[1]).parents[3]/"devtools/disp0"))
            import test_cuest250_identity as fixture
            identity=fixture.identity
            module=importlib.import_module(fixture.PACKAGE+".sapt_cuest_owned_runtime")
            case=fixture.Identity()
            create_temp=fixture.tempfile.TemporaryDirectory
            with patch.object(fixture.tempfile,"TemporaryDirectory",
                    side_effect=lambda *a,**kw:create_temp(*a,dir=root,**kw)):
                _,native,proof,manifest,text=case.fixture()
            admitted={proof["native_path"]:identity.file_digest(proof["native_path"])}
            admitted.update({r["path"]:r["sha256"] for r in manifest["components"]})
            with patch.object(identity,"NATIVE_SHA256",admitted[proof["native_path"]]):
                original=identity.mapped_file_digest
                with patch.object(identity,"mapped_file_digest",wraps=original) as hashes:
                    before=identity.audit_maps(native,proof,manifest,text=text)
                    assert hashes.call_count==4,hashes.call_count
                    owner=module.admit_owned_runtime(admitted,dedicated_worker=True)
                    with patch.object(module,"_hash_descriptor",side_effect=AssertionError("rehash")):
                        for _ in range(2):
                            result=identity.audit_maps(native,proof,manifest,text=text,owner=owner)
                    assert hashes.call_count==4,hashes.call_count
                    assert result.pop("identity_contract")=="process-owned-kernel-read-leases-v1"
                    assert result==before
                    assert identity.audit_maps(native,proof,manifest,text=text)==before
                    assert hashes.call_count==8,hashes.call_count
                    assert owner.report()["full_hash_calls"]==4
                    foreign=text+"\\n9000-9001 r-xp 0 00:00 1 /foreign/libcusolver.so.12"
                    try:identity.audit_maps(native,proof,manifest,text=foreign,owner=owner)
                    except RuntimeError as error:assert "Foreign/changed" in str(error),str(error)
                    else:raise AssertionError("foreign CUDA mapping admitted")
                    invalid(owner.check_live)
                    # Removing the foreign map never renews this process proof.
                    invalid(identity.audit_maps,native,proof,manifest,text=text,owner=owner)
        ''',revoked=True)

    def test_sync_checks_before_and_after_without_real_cuda_calls(self):
        for method in ("synchronize","synchronize_device"):
            self.run_case(f'''
                runtime=importlib.import_module(package.__name__+".sapt_cuda_runtime")
                owner=owned.admit_owned_runtime(pins)
                session=runtime.CUDASession.__new__(runtime.CUDASession)
                session.identity_owner=owner
                session.closed=session.failed=False
                session.stream=9
                calls=[]
                def native_sync(*args):
                    calls.append("native synchronization")
                    path.chmod(0o400)
                    return (0,)
                session.runtime=types.SimpleNamespace(
                    cudaStreamSynchronize=native_sync,cudaDeviceSynchronize=native_sync)
                invalid(getattr(session,{method!r}))
                assert calls==["native synchronization"],calls
                assert session.failed
                invalid(getattr(session,{method!r}))
                assert calls==["native synchronization"],calls
                assert "cuda.bindings" not in sys.modules
            ''',revoked=True)

    def test_cleanup_revocation_preserves_observed_free_and_quarantines_parents(self):
        self.run_case('''
            runtime=importlib.import_module(package.__name__+".sapt_cuda_runtime")
            state={"live":True}
            def check_live():
                if not state["live"]:
                    raise RuntimeError("injected identity failure; new process required")
            owner=types.SimpleNamespace(check_live=check_live)
            session=runtime.CUDASession.__new__(runtime.CUDASession)
            session.identity_owner=owner
            session.closed=session.failed=session.cleanup_complete=False
            session.native_owners=[]
            session.live_bytes=8
            session.handle=session.module=session.stream=9
            calls=[]
            def free(pointer):
                calls.append("free")
                state["live"]=False
                return (0,)
            def sync():
                calls.append("sync")
                return (0,)
            session.runtime=types.SimpleNamespace(cudaDeviceSynchronize=sync,cudaFree=free,
                cudaStreamDestroy=lambda *args:calls.append("UNSAFE stream destroy"))
            session.driver=types.SimpleNamespace(
                cuModuleUnload=lambda *args:calls.append("UNSAFE module unload"))
            session._blas=lambda *args:calls.append("UNSAFE BLAS destroy")
            allocation=runtime._Allocation(session,1234,8)
            session.allocations=[allocation]
            invalid(session.close)
            assert calls==["sync","free"],calls
            assert allocation.freed and session.live_bytes==0
            assert session.handle==session.module==session.stream==9
            assert session.allocations==[]
            assert session.failed and not session.cleanup_complete
            invalid(session.close)
            allocation.free()  # Observed successful free must not run twice.
            assert calls==["sync","free"],calls
        ''')


if __name__ == "__main__":
    unittest.main()
