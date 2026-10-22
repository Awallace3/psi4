"""CPU serial-shim equation/descriptor tests, NOT CUDA execution evidence."""
import ctypes as ct
import importlib
import unittest

import numpy as np

from test_cuda_backend import PACKAGE, CPUSession, Kernels, host, runtime

setup = importlib.import_module(PACKAGE+".sapt_dispersion_cuda_setup")


class Session(CPUSession):
    def gemm(self, left, right):
        result = self.empty((left.shape[0],right.shape[1]))
        host(result)[...] = host(left) @ host(right)
        self.gemms += 1
        return result


def fixture(seed):
    rng = np.random.default_rng(seed)
    n = 9
    matrices = {name:rng.normal(size=(n,n))*.1 for name in
                ("S","D_A","P_A","V_A","J_A","K_A","D_B","P_B","V_B","J_B","K_B","K_O")}
    matrices.update({name:rng.normal(size=(n,size))*.2 for name,size in
                     (("Cocc_A",3),("Cocc_B",2),("Cvir_A",4),("Cvir_B",5))})
    energies = {"a":np.array([-1.1,-.8,-.5]),"b":np.array([-.9,-.6]),
                "r":np.linspace(.1,.4,4),"s":np.linspace(.2,.7,5)}
    return matrices,energies


def reference(d,fa,fb):
    a,b,r,s = d["Cocc_A"][:,fa:],d["Cocc_B"][:,fb:],d["Cvir_A"],d["Cvir_B"]
    S,DA,DB,PA,PB = (d[name] for name in ("S","D_A","D_B","P_A","P_B"))
    JA,JB,VA,VB,KA,KB,KO = (d[name] for name in ("J_A","J_B","V_A","V_B","K_A","K_B","K_O"))
    spaces = {
        "a":a,"b":b,"r":r,"s":s,
        "r1":DB@S@r-r,"s1":DA@S@s-s,
        "a2":DB@S@a,"b2":DA@S@b,
        "r3":2*DB@S@r-2*DA@S@DB@S@r,
        "s3":2*DA@S@s-2*DB@S@DA@S@s,
        "a4":-2*DA@S@DB@S@a,"b4":-2*DB@S@DA@S@b}
    corrections = {
        "Qar":4*a.T@JB@r+2*a.T@VB@r,
        "Qbs":4*b.T@JA@s+2*b.T@VA@s,
        "Qas":2*a.T@JB@s-a.T@KB@s+a.T@KO@s
              -2*a.T@JB@DA@S@s-2*a.T@S@DB@JA@s
              -a.T@S@DB@VA@s+a.T@VB@PA@S@s,
        "Qbr":2*b.T@JA@r-b.T@KA@r+b.T@KO.T@r
              -2*b.T@S@DA@JB@r-2*b.T@JA@DB@S@r
              -b.T@S@DA@VB@r+b.T@VA@PB@S@r,
        "Sas":a.T@S@s,"Sbr":b.T@S@r,
        "SBar":a.T@S@DB@S@r,"SAbs":b.T@S@DA@S@s}
    return spaces,corrections


class HFSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Kernels.setUpClass.__func__(cls)
        cls.lib.matrix_window.argtypes = [ct.c_longlong]*4+[ct.c_int]+[ct.c_void_p]*2
        cls.lib.linear_combination.argtypes = [ct.c_longlong]*3+[ct.c_void_p]*3
        cls.lib.reduce_components.argtypes = [ct.c_longlong]+[ct.c_void_p]*3
        for name in ("matrix_window","linear_combination","reduce_components"):
            getattr(cls.lib,name).restype = None

    @classmethod
    def tearDownClass(cls):
        Kernels.tearDownClass.__func__(cls)

    def test_all_setup_equations_unequal_spaces_frozen_inputs_and_nonsymmetric_ko(self):
        for seed in (91,92):
            for fa,fb in ((0,0),(1,0),(2,1)):
                d,e = fixture(seed)
                expected = reference(d,fa,fb)
                session = Session(self.lib)
                # Any intermediate download is a test failure.
                session.download = lambda *_:self.fail("Setup downloaded an intermediate")
                with session.scope():
                    actual = setup.prepare_hf(session,d,e,frozen_a=fa,frozen_b=fb)
                    for fields,targets in zip(actual[:2],expected):
                        self.assertEqual(set(fields),set(targets))
                        for name,value in fields.items():
                            np.testing.assert_allclose(host(value),targets[name],rtol=1e-12,atol=1e-14)
                    for name,start in (("a",fa),("b",fb),("r",0),("s",0)):
                        np.testing.assert_array_equal(host(actual[2][name]),e[name][start:])
                    self.assertGreater(session.gemms,0)
                self.assertEqual(session.live_bytes,0)
                # A deliberately wrong K_O convention must be distinguishable.
                wrong = dict(d); wrong["K_O"] = d["K_O"].T
                self.assertGreater(np.max(np.abs(expected[1]["Qbr"]-reference(wrong,fa,fb)[1]["Qbr"])),1e-6)

    def test_windows_and_transpose_keep_occupied_slice_on_device(self):
        session = Session(self.lib)
        source = np.arange(63,dtype=np.float64).reshape(9,7)
        with session.scope():
            value = session.upload(source)
            m = setup.DeviceMath(session)
            for start,columns in ((0,7),(1,3),(6,1)):
                for transpose in (False,True):
                    actual = m.window(value,start,columns,transpose=transpose)
                    expected = source[:,start:start+columns]
                    np.testing.assert_array_equal(host(actual),expected.T if transpose else expected)
            for args in ((-1,1),(0,0),(6,2),(True,1)):
                with self.assertRaises(ValueError):
                    m.window(value,*args)

    def test_final_pairs_and_three_global_components_have_no_host_sum_in_path(self):
        session = Session(self.lib)
        rng = np.random.default_rng(17)
        for shape in ((1,1),(2,3),(3,7)):
            d,x = rng.normal(size=shape),rng.normal(size=shape)
            with session.scope():
                combined,totals = setup.finish_components(session,session.upload(d),session.upload(x))
                np.testing.assert_allclose(host(combined),d+x,rtol=1e-14,atol=1e-14)
                np.testing.assert_allclose(host(totals),[d.sum(),x.sum(),d.sum()+x.sum()],
                                           rtol=1e-14,atol=1e-14)

    def test_gemm_descriptor_rectangular_and_scalar_lifetimes(self):
        session = Session(self.lib)
        left,right = session.empty((3,7)),session.empty((7,5))
        calls = []
        session._blas = lambda *args:calls.append(args)
        session.handle = ct.c_void_p(123)
        out = runtime.CUDASession.gemm(session,left,right)
        args = calls[0]
        self.assertEqual(args[:7],("cublasDgemm_v2",session.handle,0,0,5,3,7))
        self.assertEqual((args[8],args[9],args[10],args[11],args[13],args[14]),
                         (right.pointer,5,left.pointer,7,out.pointer,5))
        self.assertEqual(ct.cast(args[7],ct.POINTER(ct.c_double))[0],1.)
        self.assertEqual(ct.cast(args[12],ct.POINTER(ct.c_double))[0],0.)

    def test_input_validation_before_allocation_and_no_fallback(self):
        d,e = fixture(7)
        invalid = [
            (dict(d,S=np.zeros((9,8))),e,{}),
            (dict(d,K_O=np.full((9,9),np.nan)),e,{}),
            (dict(d,Cocc_A=d["Cocc_A"].astype(np.float32)),e,{}),
            (d,dict(e,a=np.array([1.,2.,3.])),{}),
            (d,e,{"frozen_a":3}),(d,e,{"frozen_b":True}),
            (d,dict(e,r=e["r"].astype(np.float32)),{})]
        for matrices,energies,kwargs in invalid:
            session = Session(self.lib)
            with self.assertRaises(ValueError):
                setup.prepare_hf(session,matrices,energies,**kwargs)
            self.assertEqual(session.allocations,[])

    def test_device_math_rejects_cross_session_and_wrong_layout(self):
        session,other = Session(self.lib),Session(self.lib)
        m = setup.DeviceMath(session)
        value = session.empty((2,3))
        with self.assertRaises(ValueError): m.combine(value,other.empty((2,3)))
        with self.assertRaises(ValueError): m.window(value.view((2,3),order="F"))
        with self.assertRaises(ValueError): m.combine(value,value,alpha=1.5)
        with self.assertRaises(ValueError): m.combine(value,value,alpha=5)


if __name__ == "__main__":
    unittest.main()
