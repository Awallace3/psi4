"""CPU-only checks for native experiment metadata/oracles, not device proof."""
import unittest
import numpy as np
from dispersion_cache_probe import GaussianBasis,fixture,reference
from test_cuest_basis import basis_api
from test_cuda_backend import oracle
from test_cuda_fisapt import factor_reference


class Probe(unittest.TestCase):
    def test_explicit_gaussian_metadata_and_mutation_guard(self):
        primary,aux=GaussianBasis(),GaussianBasis(True)
        p,a=basis_api.snapshot_pair(primary,aux)
        self.assertEqual((p.nbf,p.ncartesian),(18,20))
        self.assertEqual((a.nbf,a.ncartesian),(32,40))
        p.assert_unchanged();a.assert_unchanged()
        primary.shells[0].exponent+=.1
        with self.assertRaisesRegex(ValueError,"changed"):p.assert_unchanged()

    def test_probe_oracle_against_independent_direct_and_localized_equations(self):
        for seed in (521,522):
            sp,c,e,rot=fixture(np,18,seed)
            rng=np.random.default_rng(seed+1)
            ao=rng.normal(size=(32,18,18))*.1
            def f(l,r):return np.einsum("Qmn,mi,nj->Qij",ao,sp[l],sp[r])
            rf={"Aar":f("r","a"),"Bbr":f("r1","b"),"Cbr":f("r","b2"),
                "Dar":f("r3","a"),"Ear":f("r","a4")}
            sf={"Abs":f("s","b"),"Bas":f("s1","a"),"Cas":f("s","a2"),
                "Dbs":f("s3","b"),"Ebs":f("s","b4")}
            for rotations in (None,rot):
                actual=reference(np,rf,sf,c,e,rotations)
                expected=(oracle(ao,sp,c,e) if rotations is None
                          else factor_reference(ao,sp,c,e,*rot))
                for a,b in zip(actual,expected):
                    np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-12)


if __name__=="__main__":unittest.main()
