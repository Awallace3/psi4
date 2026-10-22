"""Synthetic CPU-only identity negatives; no SDK load/native/device execution."""
import ctypes as ct
import importlib
import json
import mmap
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SAPT = Path(__file__).resolve().parents[2]/"psi4/driver/procrouting/sapt"
PACKAGE = "_disp0_cuda_fixtures"
package = types.ModuleType(PACKAGE); package.__path__=[str(SAPT)]
sys.modules[PACKAGE]=package
identity=importlib.import_module(f"{PACKAGE}.sapt_cuest250_identity")


class Identity(unittest.TestCase):
    def fixture(self, version=(0,2,50)):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name)
        native_path=root/"sdk/lib/13/libcuest.so.0.2.50"
        native_path.parent.mkdir(parents=True);native_path.write_bytes(b"CPU FIXTURE NOT AN ELF")
        functions={}
        version_type=ct.CFUNCTYPE(ct.c_int,ct.c_void_p,ct.POINTER(ct.c_uint32))
        for name,value in zip(("Major","Minor","Patch"),version):
            def callback(handle,out,value=value):
                out[0]=value
                return 0 if handle else 1
            functions[f"cuestGet{name}Version"]=version_type(callback)
        math_type=ct.CFUNCTYPE(ct.c_int,ct.c_void_p,ct.POINTER(ct.c_int))
        def math_mode(handle,out):
            out[0]=1
            return 0 if handle else 1
        functions["cuestGetMathMode"]=math_type(math_mode)
        noop_type=ct.CFUNCTYPE(ct.c_int)
        for name in ("cuestParametersCreate","cuestParametersDestroy",
                     "cuestDFMOIntegralsCompute","cuestDFMOIntegralsComputeWorkspaceQuery"):
            functions[name]=noop_type(lambda:0)
        native=types.SimpleNamespace(**functions)
        proof={"native_path":str(native_path)}
        rows=[]
        def row(lo,hi,path):
            stat=path.stat()
            return (f"{lo:x}-{hi:x} r-xp 0 {os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x} "
                    f"{stat.st_ino} {path}")
        for function in functions.values():
            address=ct.cast(function,ct.c_void_p).value
            rows.append(row(address,address+1,native_path))
        records=[]
        cuda_root=root/"cuda";cuda_root.mkdir()
        for i,name in enumerate(("libcudart.so.13","libcublas.so.13","libcublasLt.so.13")):
            path=cuda_root/name;path.write_bytes(f"CPU CUDA FIXTURE{i}".encode())
            records.append({"path":str(path),"sha256":identity.file_digest(path)})
            rows.append(row(4096+i*256,4097+i*256,path))
        manifest={"cuda_root":str(cuda_root),"components":records,
                  "cuda_runtime_version":13030,"nvrtc_version":[13,3]}
        return root,native,proof,manifest,"\n".join(rows)

    def audit(self,native,proof,manifest,text):
        # Fake bytes are admitted ONLY through this test-local expected-hash patch.
        with patch.object(identity,"NATIVE_SHA256",identity.file_digest(proof["native_path"])):
            return identity.audit_maps(native,proof,manifest,text=text)

    def test_actual_version_abi_uses_handle_and_uint32_output(self):
        _,native,_,_,_=self.fixture()
        identity.configure_abi(native)
        self.assertEqual(identity.native_version(native,123),(0,2,50))
        with self.assertRaises(RuntimeError):
            identity.native_version(native,0)
        with self.assertRaises(RuntimeError):
            identity.native_version(native,True)
        self.assertEqual(native.cuestGetPatchVersion.argtypes,
                         [ct.c_void_p,ct.POINTER(ct.c_uint32)])
        self.assertEqual(native.cuestDFMOIntegralsCompute.argtypes[5],ct.c_uint64)
        self.assertEqual(native.cuestGetMathMode.argtypes,
                         [ct.c_void_p,ct.POINTER(ct.c_int)])
        self.assertEqual(identity.native_math_mode(native,123),1)
        for handle in (0,True):
            with self.assertRaises(RuntimeError):
                identity.native_math_mode(native,handle)

    def test_self_consistent_old_native_version_is_rejected(self):
        _,native,_,_,_=self.fixture((0,2,2))
        with self.assertRaisesRegex(RuntimeError,"must be0.2.50"):
            identity.native_version(native,123)

    def test_math_getter_must_exist_and_belong_to_canonical_native(self):
        _,native,_,_,_=self.fixture()
        del native.cuestGetMathMode
        with self.assertRaisesRegex(RuntimeError,"capability missing"):
            identity.configure_abi(native)
        _,native,proof,manifest,text=self.fixture()
        address=ct.cast(native.cuestGetMathMode,ct.c_void_p).value
        lines=[line.replace(proof["native_path"],"/foreign/math_getter.so")
               if line.startswith(f"{address:x}-") else line
               for line in text.splitlines()]
        with self.assertRaisesRegex(RuntimeError,"outside canonical0.2.50: cuestGetMathMode"):
            self.audit(native,proof,manifest,"\n".join(lines))

    def test_wrong_archive_rejected_before_sdk_or_library_use(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"cuest_v0.2.50.tar.gz";path.write_bytes(b"actually0.2.2")
            with self.assertRaisesRegex(RuntimeError,"not canonical0.2.50"):
                identity.canonical_headers(path)

    def test_mapped_native_symbol_and_cuda_byte_proofs(self):
        _,native,proof,manifest,text=self.fixture()
        self.assertEqual(self.audit(native,proof,manifest,text)["native_mapped_path"],proof["native_path"])
        with self.assertRaises(RuntimeError):
            self.audit(native,proof,manifest,text+ "\n7000-7001 r-xp 0 00:00 1 /foreign/libcuest.so.0.2.2")
        wrong=text.replace(proof["native_path"],"/foreign/libcuest.so.0.2.50")
        with self.assertRaises(RuntimeError):
            self.audit(native,proof,manifest,wrong)
        Path(manifest["components"][0]["path"]).write_bytes(b"changed CUDA runtime")
        with self.assertRaisesRegex(RuntimeError,"Foreign/changed"):
            self.audit(native,proof,manifest,text)

    def test_no_python_binding_can_bypass_native_admission(self):
        _,native,proof,manifest,text=self.fixture()
        with patch.dict(sys.modules,{"cuest.bindings":object()}):
            with self.assertRaisesRegex(RuntimeError,"Unadmitted Python"):
                self.audit(native,proof,manifest,text)

    def test_missing_capability_missing_cuda_and_foreign_root_reject(self):
        _,native,proof,manifest,text=self.fixture()
        delattr(native,"cuestDFMOIntegralsCompute")
        with self.assertRaisesRegex(RuntimeError,"capability missing"):
            identity.configure_abi(native)
        _,native,proof,manifest,text=self.fixture()
        with self.assertRaisesRegex(RuntimeError,"not mapped"):
            self.audit(native,proof,manifest,"\n".join(text.splitlines()[:-1]))
        manifest["cuda_root"]="/wrong/root"
        with self.assertRaisesRegex(RuntimeError,"inside the selected"):
            self.audit(native,proof,manifest,text)

    def test_cuda12_or_unversioned_closure_rejects(self):
        _,native,proof,manifest,text=self.fixture()
        manifest["cuda_runtime_version"]=12080
        with self.assertRaisesRegex(RuntimeError,"selected13.3"):
            self.audit(native,proof,manifest,text)

    def test_self_consistent_cuda12_blas_with_cudart13_still_rejects(self):
        _,native,proof,manifest,text=self.fixture()
        path=Path(manifest["cuda_root"])/"libcublas.so.12"
        path.write_bytes(b"CUDA12 BLAS, SELF CONSISTENT TEST INVENTORY")
        manifest["components"].append({"path":str(path),"sha256":identity.file_digest(path)})
        stat=path.stat()
        text+=(f"\n9000-9001 r-xp 0 {os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x} "
               f"{stat.st_ino} {path}")
        with self.assertRaisesRegex(RuntimeError,"CUDA13"):
            self.audit(native,proof,manifest,text)

    def test_mapped_inode_device_must_match_current_hashed_file(self):
        for field,value in ((4,"99999999999999"),(3,"ff:ff")):
            _,native,proof,manifest,text=self.fixture()
            lines=text.splitlines();parts=lines[0].split(maxsplit=5)
            parts[field]=value;lines[0]=" ".join(parts)
            with self.subTest(field=field),self.assertRaisesRegex(RuntimeError,"inode/device"):
                self.audit(native,proof,manifest,"\n".join(lines))

    def test_live_non_executable_mapping_uses_kernel_filesystem_device_identity(self):
        _,_,proof,_,_=self.fixture()
        path=Path(proof["native_path"])
        with path.open("rb") as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ):
            rows=identity.mapped_rows(Path("/proc/self/maps").read_text())
            stat=path.stat()
            observed={(inode,device) for _,_,name,inode,device in rows if name==str(path)}
            expected={(stat.st_ino,(os.major(stat.st_dev),os.minor(stat.st_dev)))}
            if observed==expected:
                self.assertEqual(identity.mapped_file_digest(str(path),rows),identity.file_digest(path))
            else:
                with self.assertRaisesRegex(RuntimeError,"inode/device"):
                    identity.mapped_file_digest(str(path),rows)

    def btrfs_fixture(self):
        _,_,proof,_,_=self.fixture()
        path=proof["native_path"];stat=Path(path).stat()
        device=(65534,65533)  # Deliberately distinct from the descriptor's st_dev.
        rows=[(4096,8192,path,stat.st_ino,device)]
        record={"mnt_id":64,"inode":stat.st_ino,"device":device,
                "fstype":"btrfs","record":"synthetic descriptor-bound mount record"}
        return path,rows,record

    def test_btrfs_subvolume_device_cannot_be_admitted_by_descriptor_path_metadata(self):
        path,rows,record=self.btrfs_fixture()
        with patch.object(identity,"_fd_mount_record",return_value=record,create=True), \
                patch.object(identity,"_current_path_mappings",return_value=rows,create=True), \
                patch.object(identity.os,"readlink",return_value=path):
            with self.assertRaisesRegex(RuntimeError,"inode/device"):
                identity.mapped_file_digest(path,rows)

    def test_stable_overmount_with_colliding_btrfs_inode_observations_must_reject(self):
        # A VMA from subvolume A and an opened file in overmounted subvolume B
        # can have the same path/inode/raw superblock device, even though their
        # contents differ. Descriptor mountinfo identifies B, not A's backing
        # mount. These observations must never authorize hashing B as mapped A.
        path,rows,record=self.btrfs_fixture()
        mapped_payload=b"noncanonical old mapped subvolume A"
        self.assertNotEqual(identity.file_digest(path),
                            __import__("hashlib").sha256(mapped_payload).hexdigest())
        with patch.object(identity,"_fd_mount_record",return_value=record,create=True), \
                patch.object(identity,"_current_path_mappings",return_value=rows,create=True), \
                patch.object(identity.os,"readlink",return_value=path):
            with self.assertRaisesRegex(RuntimeError,"inode/device"):
                identity.mapped_file_digest(path,rows)

    def test_btrfs_normalization_rejects_wrong_inode_device_fs_link_or_liveness(self):
        for failure in ("inode","fd-inode","device","filesystem","link","unmapped","mixed-devices"):
            path,rows,record=self.btrfs_fixture()
            link,current=path,list(rows)
            if failure=="inode":rows=[(*rows[0][:3],rows[0][3]+1,rows[0][4])]
            elif failure=="fd-inode":record["inode"]+=1
            elif failure=="device":record["device"]=(1,2)
            elif failure=="filesystem":record["fstype"]="ext4"
            elif failure=="link":link="/foreign/replaced.so"
            elif failure=="unmapped":current=[]
            else:rows.append((8192,12288,path,record["inode"],(1,2)))
            with self.subTest(failure=failure), \
                    patch.object(identity,"_fd_mount_record",return_value=record,create=True), \
                    patch.object(identity,"_current_path_mappings",return_value=current,create=True), \
                    patch.object(identity.os,"readlink",return_value=link):
                with self.assertRaisesRegex(RuntimeError,"inode/device"):
                    identity.mapped_file_digest(path,rows)

    def test_opened_mapped_inode_cannot_be_replaced_during_hash(self):
        _,native,proof,manifest,text=self.fixture()
        path=Path(proof["native_path"])
        original=identity.digest
        def replace_after_read(stream):
            value=original(stream)
            replacement=path.with_name("replacement")
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
            return value
        expected=identity.file_digest(path)
        with patch.object(identity,"NATIVE_SHA256",expected),patch.object(
                identity,"digest",side_effect=replace_after_read):
            with self.assertRaisesRegex(RuntimeError,"changed/replaced"):
                identity.audit_maps(native,proof,manifest,text=text)

    def test_header_tree_changed_byte_or_extra_header_reject(self):
        root,_,proof,_,_=self.fixture()
        sdk=root/"sdk";(sdk/"include").mkdir()
        header=sdk/"include/cuest.h";header.write_bytes(b"CPU HEADER FIXTURE")
        records={"include/cuest.h":{"bytes":header.stat().st_size,"sha256":identity.file_digest(header)}}
        with patch.object(identity,"canonical_headers",return_value=records),patch.object(
                identity,"NATIVE_SHA256",identity.file_digest(proof["native_path"])):
            result=identity.verify_files(sdk,root/"unused-test-archive")
            self.assertEqual(result["sdk_version"],"0.2.50")
            header.write_bytes(b"MODIFIED OLD HEADER")
            with self.assertRaises(RuntimeError):
                identity.verify_files(sdk,root/"unused-test-archive")
            (sdk/"include/extra-old.h").write_bytes(b"old0.2.2")
            with self.assertRaisesRegex(RuntimeError,"inventory differs"):
                identity.verify_files(sdk,root/"unused-test-archive")


if __name__=="__main__":
    unittest.main()
