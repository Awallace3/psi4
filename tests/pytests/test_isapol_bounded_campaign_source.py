# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Actual bounded loop, targets and row producer; synthetic engine boundaries.

No response, grid, integral, LW or compiled PFIT implementation executes.
The fake normal-equation boundary is NOT validation of the native solver.
"""
from contextlib import ExitStack
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from isapol_source_test_support import load_source

B = load_source("isapol_bounded")
R = load_source("isapol_refine")
BR = load_source("isapol_bounded_response")
LG = load_source("isapol_logging")
N = load_source("isapol_native")
core = R.core


class CampaignTests(unittest.TestCase):
    def campaign(self, *, distribution="df_centre_analytic", response="reference_h2h1",
                 recoupled=False, failure=None, budget=512*1024**2, work=10**12):
        case = SimpleNamespace(closed=False, released=False, directory=None, ledger=None,
                               solves=[], runner_keywords=None, native_declarations=[])
        class Matrix:
            def __init__(self,a):
                self.np = np.asarray(a)
            @classmethod
            def from_array(cls,a):
                return cls(a)
        class Aux:
            nfunction = 1
        class Options:
            npoints, seed, lolim, hilim = 2, 1, 2., 4.
        class Coulomb:
            def __init__(self,aux):
                pass
            def point_potentials(self,points):
                return Matrix(1/points.np[:,2][None,:])
        class Runner:
            def __init__(self,*args,**kwargs):
                case.runner_keywords = kwargs
                self.ledger = BR._Ledger(kwargs["resources"])
                case.ledger = self.ledger
                self.auxiliary = Aux()
                self.state_sha256, self.auxiliary_sha256, self.grid_sha256 = "a"*64,"b"*64,"c"*64
                self.correction, self.model, self.stability = "fake NONE","fake response boundary",{}
                self.lw_residual_policy, self.lw_disclosure = "production","fake"
                self.provenance = None
            def __enter__(self):
                self.temp = tempfile.TemporaryDirectory()
                case.directory = self.temp.name
                return self
            def prepare(self,producer):
                self.partition = distribution
                if producer is not None:
                    self.supplied = producer(self.ledger)
                if failure == "prepare":
                    raise RuntimeError("injected prepare failure")
            def solve(self,omega):
                case.solves.append(omega)
                self.ledger.admit("synthetic response boundary",64,1)
                if failure == "response" and omega:
                    raise RuntimeError("injected response failure")
                return SimpleNamespace(target_response=-np.ones((1,1))*(1+omega),
                    nonlocal_response=np.zeros((25,25)), residual=0.,diagnostics={})
            def release(self):
                case.released = True
            def __exit__(self,*exc):
                self.release()
                self.temp.cleanup()
                case.closed = True
        def localized(**kw):
            omega = kw["frequencies"][0]
            if failure == "localization" and omega:
                raise RuntimeError("injected localization failure")
            a = np.zeros((1,1,3,3))
            a[0,0,0,0] = 2. if not omega else 8e-5
            return SimpleNamespace(raw_local=SimpleNamespace(array=a),
                frequency_diagnostics=[SimpleNamespace(residuals=SimpleNamespace(maximum=0.))])
        def solve_rows(problems,blocks,options):
            if failure == "fit":
                raise RuntimeError("injected fit failure")
            first,second = list(blocks()),list(blocks())
            for (i,d,t),(j,e,u) in zip(first,second):
                self.assertEqual(i,j)
                np.testing.assert_array_equal(d,e)
                np.testing.assert_array_equal(t,u)
            design = np.vstack([d for _,d,_ in first])
            targets = np.vstack([t for _,_,t in first])
            result = []
            for k,p in enumerate(problems):
                n = len(p.model.parameter_labels)
                strength = np.asarray(p.penalty.matrix.values).reshape(n,n)
                anchor = np.asarray(p.penalty.anchor)
                z = np.linalg.solve(design.T@design+strength,
                                    design.T@targets[:,k]+strength@anchor)
                case.native_declarations.append(p.model.provenance)
                result.append(SimpleNamespace(parameters=z,status="Solved",diagnostics={},
                                               model_provenance=p.model.provenance))
            return result
        replacements = dict(Matrix=Matrix, print_out=lambda text:None, FitPointsOptions=Options,
            FitPoints=lambda *a:SimpleNamespace(x=lambda:[0.,0.],y=lambda:[0.,0.],z=lambda:[4.,5.]),
            IsaMultipoleSite=SimpleNamespace, IsaExplicitBasis=Aux, IsaAuxCoulomb=Coulomb,
            IsaPfitTargetOrigin=SimpleNamespace(NativeFittedPointResponse="NativeFittedPointResponse"),
            IsaPfitTargetConvention=SimpleNamespace(
                NegativeInducedPotentialPerUnitSourceChargeAtomicUnits="fake convention"),
            IsaPfitTargetProvenance=SimpleNamespace, IsaPfitRowModel=SimpleNamespace,
            IsaPfitCloudRows=SimpleNamespace, IsaPfitRowProblem=SimpleNamespace,
            IsaPfitMatrix=SimpleNamespace, IsaPfitMatrixPenalty=SimpleNamespace,
            IsaPfitOptions=SimpleNamespace,
            IsaPfitSolver=SimpleNamespace(NormalEquationsDSYSV="fake normal equations"),
            IsaPfitStatus=SimpleNamespace(Solved="Solved"), isa_pfit_solve_rows_multi=solve_rows,
            isa_t_functions=lambda rank,point,*a:[0.,1/point[2]**2,0.,0.])
        def dispersion(*args,**kwargs):
            if failure == "dispersion":
                raise RuntimeError("injected dispersion failure")
            return SimpleNamespace(pairs=())
        A = load_source("isapol_native_recoupled")
        def angular(*args,**kwargs):
            if failure == "angular":
                raise RuntimeError("injected angular failure")
            kwargs["ledger"].admit("synthetic angular boundary",128,7)
            return SimpleNamespace(pairs=(),marker="explicit full-tensor request")
        site = R.RefinementSite("A","A",(0.,0.,0.),np.eye(3),1)
        wfn = SimpleNamespace(molecule=lambda:SimpleNamespace(clone=lambda:object()))
        with ExitStack() as stack:
            stack.enter_context(patch.dict(core.__dict__,replacements))
            stack.enter_context(patch.object(B,"BoundedResponse",Runner))
            stack.enter_context(patch.object(B,"DistributedMoments",SimpleNamespace))
            stack.enter_context(patch.object(B.lw,"supplied_nonlocal_properties",localized))
            stack.enter_context(patch.object(R,"refined_isotropic_dispersion",dispersion))
            stack.enter_context(patch.object(A,"recoupled_plan",lambda *a:(128,7)))
            stack.enter_context(patch.object(A,"refined_recoupled_dispersion",angular))
            fdds = object()
            kwargs = dict(caller_converged=True, distribution=distribution, sites=[site],bonds=[],
                quadrature=N.Quadrature((0.,.7),(0.,.4),B.lw.Provenance("fake CP","a"*64,"analytic","fake")),
                response_grid=np.array([[0.,0.,0.,1.]]), smoothing=object(),shell_cutoff=1e-8,
                charge_penalty=1000.,anchor_metric_damping=.0005,lattice_options=Options(),
                localization_rank_limit=1,weight_type=3,weight_coefficient=1e-3,cutoff=1e-4,
                resources=BR.BoundedResources(budget,work,1024**3),scf_correction="NONE",
                log=LG.StageLog(0),max_order=6,response=response,
                fdds=fdds if response=="native_fdds" else None,recoupled_dispersion=recoupled,
                distributed_moments=SimpleNamespace(values=np.ones((25,1))) if distribution=="supplied" else None)
            if failure or budget<200*1024**2 or work<1000:
                with self.assertRaises((ValueError,RuntimeError)):
                    B.bounded_properties(wfn,object(),**kwargs)
                output = None
            else:
                output = B.bounded_properties(wfn,object(),**kwargs)
        if case.directory is not None:
            from pathlib import Path
            self.assertFalse(Path(case.directory).exists())
            self.assertTrue(case.closed)
            self.assertTrue(case.released)
        return output,case

    def test_published_routes_and_static_default_share_actual_targets_rows_and_ledger(self):
        for distribution in ("df_centre_analytic","supplied"):
            for response in ("reference_h2h1","native_fdds"):
                with self.subTest(distribution=distribution,response=response):
                    out,case = self.campaign(distribution=distribution,response=response)
                    self.assertEqual(case.solves,[0.,.7])
                    self.assertEqual(case.runner_keywords["response"],response)
                    self.assertEqual(case.runner_keywords["fdds"] is not None,response=="native_fdds")
                    self.assertEqual(out.refinements[1].model.anchors,(8e-5,))
                    self.assertGreater(out.refinements[1].model.strengths[0],0.)
                    self.assertEqual(out.provenance["penalty_selection"],"static_reference")
                    self.assertIsNone(out.recoupled_dispersion)
                    self.assertEqual(out.resources["charged_work"],sum(s["work"] for s in case.ledger.stages))
                    self.assertEqual(case.ledger.reserved,0)
                    self.assertEqual(case.native_declarations[0],case.native_declarations[1])
                    self.assertIn("shared native declaration",case.native_declarations[0])
                    self.assertNotIn("; anchor_sha256=",case.native_declarations[0])
                    # Independent analytic one-variable synthetic fit.
                    d,t = np.array([1/256,1/400,1/625]),np.array([1/16,1/20,1/25])
                    for r in out.refinements:
                        a,s,f = r.model.anchors[0],r.model.strengths[0],r.model.frequency_au
                        expected = (d@(t*(1+f))+s*a)/(d@d+s)
                        self.assertAlmostEqual(r.parameters[0],expected,places=12)

    def test_optional_angular_uses_same_ledger_and_owned_result(self):
        out,case = self.campaign(recoupled=True)
        self.assertEqual(out.recoupled_dispersion.marker,"explicit full-tensor request")
        self.assertIn("synthetic angular boundary",[s["stage"] for s in case.ledger.stages])
        self.assertEqual(out.resources["charged_work"],sum(s["work"] for s in case.ledger.stages))

    def test_all_scientific_boundary_failures_close_owned_context(self):
        for failure in ("prepare","response","localization","fit","dispersion","angular"):
            with self.subTest(failure=failure):
                self.campaign(failure=failure,recoupled=failure=="angular")

    def test_cumulative_work_and_memory_refusals_close_before_later_response(self):
        _,case = self.campaign(work=1)
        self.assertEqual(case.solves,[0.])
        _,case = self.campaign(budget=128*1024**2)
        self.assertEqual(case.solves,[0.])


if __name__ == "__main__":
    unittest.main(verbosity=2)
