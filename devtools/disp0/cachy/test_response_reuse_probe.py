"""CPU checks for standalone probe fixtures; not native CUDA evidence."""
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from response_reuse_probe import numerical_case
from response_cuda_closure import relocate,verify_mapped_cuda


class Fixture(unittest.TestCase):
    def test_relocation_preserves_exact_component_hashes_and_rejects_alias_names(self):
        original={"cuda_root":"/old","components":[
            {"path":"/old/lib/libcublas.so.13.6.0.2","sha256":"one"},
            {"path":"/old/lib/libnvrtc.so.13.3.33","sha256":"two"}]}
        result=relocate(original,Path("/new"))
        self.assertEqual(result["cuda_root"],"/new/cuda")
        self.assertEqual([c["sha256"] for c in result["components"]],["one","two"])
        self.assertEqual(result["components"][0]["path"],"/new/cuda/lib/libcublas.so.13.6.0.2")
        self.assertEqual(original["cuda_root"],"/old")
        for name in ("bad","libcublas.so.13.6.0.2"):
            bad={"components":original["components"]+[{"path":"/other/"+name,"sha256":"three"}]}
            with self.assertRaises(RuntimeError):relocate(bad,Path("/new"))

    def test_actual_map_proof_rejects_missing_foreign_or_changed_cuda_bytes(self):
        manifest={"components":[{"path":"/new/"+name,"sha256":"expected"}
            for name in ("libcublas.so.13.6.0.2","libcudart.so.13.3.29","libnvrtc.so.13.3.33")]}
        paths=[c["path"] for c in manifest["components"]]
        identity=SimpleNamespace(mapped_rows=lambda text:[(0,1,p,1,(0,1)) for p in paths],
                                 mapped_file_digest=lambda path,rows:"expected")
        self.assertEqual(len(verify_mapped_cuda(identity,manifest)),3)
        paths.pop()
        with self.assertRaisesRegex(RuntimeError,"missing"):verify_mapped_cuda(identity,manifest)
        paths.append("/foreign/libnvrtc.so.13.3.33")
        with self.assertRaisesRegex(RuntimeError,"Unlisted"):verify_mapped_cuda(identity,manifest)
        paths[-1]=manifest["components"][-1]["path"]
        identity.mapped_file_digest=lambda path,rows:"changed"
        with self.assertRaisesRegex(RuntimeError,"bytes"):verify_mapped_cuda(identity,manifest)

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
