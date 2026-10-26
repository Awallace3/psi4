"""CPU checks for standalone probe fixtures; not native CUDA evidence."""
import unittest
import numpy as np
from response_reuse_probe import numerical_case


class Fixture(unittest.TestCase):
    def test_reference_equations_and_changed_dimensions(self):
        for dims in ((2,3,5),(8,32,65),(16,96,129)):
            factors,eo,ev,x,truth=numerical_case(np,*dims)
            ov,oo,vv=(factors[n] for n in ("ov","oo","vv"))
            np.testing.assert_array_equal(oo,oo.transpose(0,2,1))
            np.testing.assert_array_equal(vv,vv.transpose(0,2,1))
            coulomb=np.tensordot(ov,np.sum(ov*x,axis=(1,2)),axes=(0,0))
            reference=(ev[None,:]-eo[:,None])*x+4*coulomb
            reference-=np.sum((oo@x)@vv.transpose(0,2,1),axis=0)
            reference-=np.sum((ov@x.T)@ov,axis=0)
            np.testing.assert_allclose(truth,reference,rtol=1e-12,atol=1e-13)
            if dims[0]==2:
                O,V,_=dims;eri=np.einsum("qij,qab->iajb",oo,vv)
                H=4*np.einsum("qia,qjb->iajb",ov,ov)-eri
                H-=np.einsum("qib,qja->iajb",ov,ov)
                H=H.reshape(O*V,O*V)+np.diag((ev[None,:]-eo[:,None]).ravel())
                np.testing.assert_allclose(H,H.T,atol=1e-14)
                self.assertGreater(np.linalg.eigvalsh(H).min(),0)
                np.testing.assert_allclose(truth.ravel(),H@x.ravel(),rtol=1e-12,atol=1e-13)


if __name__=="__main__":unittest.main()
