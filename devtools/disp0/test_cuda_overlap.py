"""CPU serial-shim overlap maps/metadata, NOT native GPU overlap evidence."""
import ctypes as ct
import importlib
import unittest
from unittest.mock import Mock,patch
import numpy as np
from test_cuda_backend import PACKAGE,Kernels,host,runtime
from test_cuda_hf_setup import Session
from test_cuest_basis import Basis,Shell

overlap=importlib.import_module(PACKAGE+".sapt_cuest_overlap_cuda")


class MetadataSession(Session):
    upload_indices=runtime.CUDASession.upload_indices
    _upload_array=Session.upload  # Explicit CPU transport double.


class Overlap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Kernels.setUpClass.__func__(cls)
        cls.lib.gather_indexed_matrix.argtypes=[ct.c_longlong]*3+[ct.c_void_p]*4
        cls.lib.gather_indexed_matrix.restype=None

    @classmethod
    def tearDownClass(cls):Kernels.tearDownClass.__func__(cls)

    def test_union_offsets_keep_original_parent_order_and_unchanged_guards(self):
        primary=Basis();minimal=Basis([Shell(0,0),Shell(1,0)])
        p,m=overlap.snapshot_pair(primary,minimal)
        union=overlap.union_snapshot(p,m)
        self.assertEqual(union.nbf,15)
        self.assertEqual(union.primary_indices,tuple(range(9))+tuple(range(10,14)))
        self.assertEqual(union.minimal_indices,(9,14))
        self.assertEqual(union.shells_per_center,(4,3))
        self.assertEqual([s.function_offset for s in union.shells],[0,1,4,9,10,11,14])
        union.assert_unchanged()
        minimal.shells[0].exponents[0]*=1.01
        with self.assertRaises(ValueError):union.assert_unchanged()
        with self.assertRaises(TypeError):overlap.CuESTOverlapBlocks(object(),minimal)

    def test_union_gather_and_filtered_minimal_all_matrix_values_survive_scratch(self):
        p,m=overlap.snapshot_pair(Basis(),Basis([Shell(0,0),Shell(1,0)]))
        union=overlap.union_snapshot(p,m)
        full=np.arange(union.nbf**2,dtype=np.float64).reshape(union.nbf,union.nbf)
        for minimal_indices in (union.minimal_indices,union.minimal_indices[:1]):
            session=MetadataSession(self.lib)
            session.download=lambda *_:self.fail("Overlap extraction staged a matrix")
            with session.scope():
                actual=overlap.gather_overlap_blocks(session,session.upload(full),
                    np.array(union.primary_indices,dtype=np.int64),np.array(minimal_indices,dtype=np.int64))
                for value,rows,cols in ((actual.s11,union.primary_indices,union.primary_indices),
                    (actual.s12,union.primary_indices,minimal_indices),(actual.s22,minimal_indices,minimal_indices)):
                    np.testing.assert_array_equal(host(value),full[np.ix_(rows,cols)])
                    self.assertFalse(value.allocation.freed)
            self.assertEqual(session.live_bytes,0)

    def test_scattered_changed_metadata_and_nonsymmetric_values(self):
        for seed in (211,212):
            rng=np.random.default_rng(seed)
            for N in (4,37,65):
                full=rng.normal(size=(N,N))
                p=np.arange(0,N,2,dtype=np.int64);m=np.arange(1,N,3,dtype=np.int64)
                session=MetadataSession(self.lib)
                with session.scope():
                    values=overlap.gather_overlap_blocks(session,session.upload(full),p,m)
                    for value,rows,cols in ((values.s11,p,p),(values.s12,p,m),(values.s22,m,m)):
                        np.testing.assert_array_equal(host(value),full[np.ix_(rows,cols)])

    def test_integer_metadata_does_not_weaken_numerical_upload_policy(self):
        session=MetadataSession(self.lib)
        with session.scope():
            valid=np.array([3,2,1],dtype=np.int64)
            np.testing.assert_array_equal(host(session.upload_indices(valid)),valid)
            for invalid in (np.array([],dtype=np.int64),np.array([-1],dtype=np.int64),
                            np.array([1.]),np.array([1],dtype=np.int32),np.array([[1]],dtype=np.int64)):
                with self.assertRaises(ValueError):session.upload_indices(invalid)
            with self.assertRaisesRegex(ValueError,"FP64"):
                runtime.CUDASession.upload(session,valid)

    def test_bad_metadata_and_output_aliases_reject_before_launch(self):
        session=MetadataSession(self.lib)
        with session.scope():
            full=session.empty((4,4));p=np.array([0,2],dtype=np.int64);q=np.array([1],dtype=np.int64)
            for invalid in (np.array([4],dtype=np.int64),np.array([-1],dtype=np.int64),
                            np.array([0,0],dtype=np.int64),np.array([1.])):
                with self.assertRaises(ValueError):
                    overlap.gather_overlap_blocks(session,full,invalid,q)
            alias=overlap.OverlapBlocksDevice(full.view((2,2)),session.empty((2,1)),session.empty((1,1)))
            with self.assertRaises(ValueError):
                overlap.gather_overlap_blocks(session,full,p,q,outputs=alias)


    def test_large_union_admission_is_memory_based_before_native_construction(self):
        class ReachedNative(RuntimeError):pass
        for N in (356,742,1293):
            primary=Basis([Shell(0,0) for _ in range(N-1)]+[Shell(1,0)])
            minimal=Basis([Shell(0,0) for _ in range(199)]+[Shell(1,0)])
            p,_=overlap.snapshot_pair(primary,minimal)
            session=MetadataSession(self.lib)
            owner=overlap.DiagnosticPlanOwner.__new__(overlap.DiagnosticPlanOwner)
            owner.primary_snapshot,owner.session=p,session
            owner._check_construction=Mock();owner.assert_live=Mock()
            with patch.object(overlap.CuESTOverlapBlocks,"_bind",side_effect=ReachedNative) as native:
                with self.assertRaises(ReachedNative):
                    overlap.CuESTOverlapBlocks(owner,minimal)
                native.assert_called_once()
                native.reset_mock()
                session.limit=1
                with self.assertRaises(MemoryError):overlap.CuESTOverlapBlocks(owner,minimal)
                native.assert_not_called()
                session.limit=2**40
                with self.assertRaises(NotImplementedError):
                    overlap.CuESTOverlapBlocks(owner,minimal,max_union_nbf=512)
                native.assert_not_called()

    def test_large_native_overlap_output_gather_has_no_512_limit(self):
        N,M=742,137;U=N+M
        full=np.arange(U*U,dtype=np.float64).reshape(U,U)
        p=np.arange(N,dtype=np.int64);q=np.arange(N,U,dtype=np.int64)
        session=MetadataSession(self.lib)
        with session.scope():
            outputs=overlap.gather_overlap_blocks(session,session.upload(full),p,q)
            np.testing.assert_array_equal(host(outputs.s11),full[:N,:N])
            np.testing.assert_array_equal(host(outputs.s12),full[:N,N:])
            np.testing.assert_array_equal(host(outputs.s22),full[N:,N:])
        self.assertEqual(session.live_bytes,0)


if __name__=="__main__":unittest.main()
