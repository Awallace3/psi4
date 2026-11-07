"""Admission + compiled native-solver algebra, not end-to-end Psi4 qualification."""
import importlib.util
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CPP = ROOT / "psi4/src/psi4/fisapt"
spec = importlib.util.spec_from_file_location(
    "_cpks_admission", ROOT / "psi4/driver/procrouting/sapt/sapt_fisapt_cpks_find.py")
admission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(admission)


def build_solver(directory):
    """Compile the actual PCG and XC-action branch against small matrix doubles."""
    header = (CPP / "fisapt.h").read_text()
    text = (CPP / "fisapt.cc").read_text()
    declaration = header[header.index("class CPHF_FISAPT {"):header.index("}  // Namespace fisapt")]
    methods = text[text.index("CPHF_FISAPT::CPHF_FISAPT()"):text.index("// method for updating/setting matrices_")]
    # Only the original-monomer action is exercised here; the unchanged JK arm
    # needs the real library and is covered by matching-build tests.
    methods = methods[:methods.index('    bool do_A = b.count("A");')]
    methods += '    throw PSIEXCEPTION("legacy arm not in algebra fixture");\n}\n'
    source = Path(directory) / "solver.cc"
    source.write_text(Path(__file__).with_name("fsapt_cpks_fixture.hpp").read_text()
                      + declaration + methods + MAIN)
    executable = Path(directory) / "solver"
    subprocess.run([shutil.which("c++"), "-std=c++17", "-O1", str(source), "-o", str(executable)],
                   check=True, capture_output=True, text=True, timeout=60)
    return executable


MAIN = r"""
int main() {
    int oa,va,ob,vb,maxiter; double tolerance,offset;
    std::cin>>oa>>va>>ob>>vb>>maxiter>>tolerance>>offset;
    CPHF_FISAPT solver; solver.delta_=tolerance; solver.maxiter_=maxiter;
    auto read=[&](int o,int v,SharedMatrix& rhs,std::shared_ptr<Vector>& eo,
                  std::shared_ptr<Vector>& ev,std::shared_ptr<scf::HF>& wfn) {
        wfn=std::make_shared<scf::HF>(); wfn->offset=offset; wfn->hessian.resize(o*v*o*v);
        for(auto& x:wfn->hessian) std::cin>>x;
        eo=std::make_shared<Vector>(o); ev=std::make_shared<Vector>(v);
        for(auto& x:eo->data) std::cin>>x; for(auto& x:ev->data) std::cin>>x;
        rhs=std::make_shared<Matrix>(o,v); for(auto& x:rhs->data) std::cin>>x;
    };
    read(oa,va,solver.w_A_,solver.eps_occ_A_,solver.eps_vir_A_,solver.monomer_A_);
    read(ob,vb,solver.w_B_,solver.eps_occ_B_,solver.eps_vir_B_,solver.monomer_B_);
    try {
        solver.compute_cphf(); std::cout<<"OK\n"<<std::setprecision(17);
        for(auto x:solver.x_A_->data) std::cout<<x<<" ";
        for(auto x:solver.x_B_->data) std::cout<<x<<" ";
    } catch(const std::exception& e) { std::cout<<"ERROR\n"<<e.what(); }
}
"""


def solve(executable, problems, *, maxiter=128, tolerance=1e-12, offset=0.):
    fields = [*problems[0][1].shape, *problems[1][1].shape, maxiter, tolerance, offset]
    for h, rhs, eo, ev in problems:
        fields += [*h.ravel(), *eo, *ev, *rhs.ravel()]
    result = subprocess.run([str(executable)], input=" ".join(map(str, fields)),
                            text=True, capture_output=True, check=True, timeout=30).stdout
    status, values = result.split("\n", 1)
    if status != "OK":
        raise RuntimeError(values)
    a, b = problems[0][1], problems[1][1]
    numbers = np.fromstring(values, sep=" ")
    return numbers[:a.size].reshape(a.shape), numbers[a.size:].reshape(b.shape)


class NativePCGAlgebra(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("c++"):
            raise unittest.SkipTest("requires an existing C++ compiler, not a native Psi4 build")
        cls.directory = tempfile.TemporaryDirectory(prefix="fsapt-cpks-pcg-")
        cls.executable = build_solver(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def fixture(self):
        rng = np.random.default_rng(73)
        result = []
        for o, v in ((3, 2), (1, 4)):
            a = rng.normal(size=(o*v, o*v))
            h = np.eye(o*v)*2 + .1*a@a.T
            result.append((h, rng.normal(size=(o, v)), -np.linspace(.2,.6,o), np.linspace(.3,.8,v)))
        return result

    def test_actual_native_loop_matches_dense_solve_and_source_additivity(self):
        problems = self.fixture()
        full = solve(self.executable, problems)
        for actual, (h, rhs, _, _) in zip(full, problems):
            np.testing.assert_allclose(actual, np.linalg.solve(h, rhs.ravel()).reshape(rhs.shape),
                                       atol=1e-12, rtol=1e-11)
        pieces = [solve(self.executable, [(h, fraction*r, eo, ev) for h,r,eo,ev in problems])
                  for fraction in (1.7, -.7, 0.)]
        for i in (0, 1):
            np.testing.assert_allclose(sum(p[i] for p in pieces), full[i], atol=1e-12, rtol=1e-11)
            np.testing.assert_array_equal(pieces[-1][i], np.zeros_like(full[i]))

    def test_zero_unequal_direction_negative_curvature_and_bad_gaps(self):
        p = self.fixture()
        p[0] = (p[0][0], np.zeros_like(p[0][1]), *p[0][2:])
        np.testing.assert_array_equal(solve(self.executable,p)[0], p[0][1])
        p = self.fixture(); p[0] = (-p[0][0], *p[0][1:])
        with self.assertRaisesRegex(RuntimeError, "curvature"):
            solve(self.executable,p)
        p = self.fixture(); p[0] = (*p[0][:3], np.full(2,-10.))
        with self.assertRaisesRegex(RuntimeError, "gaps"):
            solve(self.executable,p)
        with self.assertRaisesRegex(RuntimeError, "converge"):
            solve(self.executable,self.fixture(),maxiter=1)

    def test_zero_rhs_still_checks_true_hessian_residual(self):
        p = [(h,np.zeros_like(r),eo,ev) for h,r,eo,ev in self.fixture()]
        with self.assertRaisesRegex(RuntimeError, "true residual"):
            solve(self.executable,p,offset=.1)

    def test_raw_source_pair_readouts_and_equal_totals_do_not_imply_equal_weights(self):
        p=self.fixture();rng=np.random.default_rng(19)
        parts=[[.4*r+rng.normal(size=r.shape)*.1, None] for _,r,_,_ in p]
        for pieces,(_,r,_,_) in zip(parts,p): pieces[1]=r-pieces[0]
        amplitudes=[solve(self.executable,[(h,parts[i][s],eo,ev) for i,(h,_,eo,ev) in enumerate(p)])
                    for s in (0,1)]
        for i,(h,r,eo,ev) in enumerate(p):
            u=np.linalg.qr(rng.normal(size=(r.shape[0],r.shape[0])))[0]
            target=u.T@r;exchange=rng.normal(size=r.shape)*.05
            full=[];uncoupled=[]
            for s in (0,1):
                physical=-amplitudes[s][i]
                exact=-np.linalg.solve(h,parts[i][s].ravel()).reshape(r.shape)
                for t in (target,exchange):
                    np.testing.assert_allclose(2*np.sum((u.T@physical)*t,axis=1),
                                               2*np.sum((u.T@exact)*t,axis=1),atol=1e-12,rtol=1e-11)
                full.append(2*np.sum((u.T@physical)*(target+exchange),axis=1))
                raw=-parts[i][s]/(ev[None,:]-eo[:,None])
                uncoupled.append(2*np.sum((u.T@raw)*(target+exchange),axis=1))
            full=np.array(full);uncoupled=np.array(uncoupled)
            fitted=uncoupled*full.sum()/uncoupled.sum()
            self.assertAlmostEqual(fitted.sum(),full.sum(),places=12)
            self.assertFalse(np.allclose(fitted,full))


class Admission(unittest.TestCase):
    def fixture(self):
        options = {
            "FISAPT": dict(FISAPT_FSAPT_IND_XC_RESPONSE=True, FISAPT_FSAPT_IND_RESPONSE=True,
                           FISAPT_LINK_ASSIGNMENT="C", SSAPT0_SCALE=False, FISAPT_EXCH_PARPERP=False),
            "SAPT": dict(SAPT_DFT_DO_FSAPT="FISAPT", SAPT_DFT_INDUCTION_TYPE="CPKS",
                         SAPT_DFT_FUNCTIONAL="PBE", SAPT_DFT_DO_DDFT=False,
                         SAPT_DFT_GRAC_COMPUTE="NONE", SAPT_DFT_GRAC_SHIFT_A=0.,
                         SAPT_DFT_GRAC_SHIFT_B=0., DO_IND_EXCH_SINF=False),
            "SCF": dict(LEVEL_SHIFT=0., PCM=False, PE=False, DDX=False),
        }
        core = SimpleNamespace(get_option=lambda mod,key: options[mod][key],
                               get_global_option_list=lambda: ["PCM","PE","DDX","BRIANQC_ENABLE"],
                               get_global_option=lambda name: False,
                               prepare_options_for_module=lambda name: None)
        return core,options

    def test_default_off_never_installs_monomers(self):
        core = SimpleNamespace(get_option=lambda *args: False, prepare_options_for_module=lambda x: None)
        admission.install(core, object(), object(), object())

    def test_install_only_passes_original_monomers_and_restores_option_context(self):
        core,_ = self.fixture()
        recorded=[];a,b=object(),object()
        core.prepare_options_for_module=recorded.append
        native=SimpleNamespace(set_cpks_monomers=lambda *x: recorded.append(x))
        admission.install(core,native,a,b)
        self.assertEqual(recorded,[(a,b),"FISAPT"])
        with self.assertRaisesRegex(RuntimeError,"matching native"):
            admission.install(core,object(),a,b)
        self.assertEqual(recorded[-1],"FISAPT")

    def test_unsupported_modes_and_effective_SCF_embedding_fail_closed(self):
        core,options=self.fixture()
        for mod,key,value in (("SAPT","SAPT_DFT_FUNCTIONAL","PBE0"),
                               ("SAPT","SAPT_DFT_INDUCTION_TYPE","CPHF"),
                               ("SAPT","SAPT_DFT_DO_DDFT",True),
                               ("SAPT","SAPT_DFT_GRAC_SHIFT_A",.1),
                               ("SCF","LEVEL_SHIFT",.1),
                               *(("SCF",key,True) for key in ("PCM","PE","DDX")),
                               ("FISAPT","FISAPT_FSAPT_IND_RESPONSE",False),
                               ("FISAPT","SSAPT0_SCALE",True)):
            old=options[mod][key];options[mod][key]=value
            with self.assertRaises((ValueError,NotImplementedError)):
                admission.validate_request(core)
            options[mod][key]=old
        with self.assertRaises(NotImplementedError):
            admission.validate_request(core,external_potentials={"A":[1.]})
        core.get_global_option=lambda key: True
        with self.assertRaisesRegex(ValueError,"CPU"):
            admission.validate_request(core)

    def test_effective_embedding_and_legacy_source_wiring_remain_explicit(self):
        text = (CPP / "fisapt.cc").read_text()
        self.assertIn('fill_tensor("WAbs", wA, {C, C + 1})', text)
        self.assertIn("wfn->cphf_Hx({item.second})", text)
        self.assertIn('cpks_close(amplitude_sum_A, cphf->x_A_', text)
        self.assertIn('raw_doubles > raw_budget', text)
        self.assertIn('get_option("SCF", name)', Path(spec.origin).read_text())
        self.assertNotIn("sapt_fisapt_cpks_solver", Path(spec.origin).read_text())

    def test_legacy_product_body_is_byte_identical_to_target_branch(self):
        text=(CPP/"fisapt.cc").read_text().split("CPHF_FISAPT::product(",1)[1]
        body=text[text.index('    bool do_A = b.count("A");'):text.index("// method for updating/setting matrices_")]
        self.assertEqual(hashlib.sha256(body.encode()).hexdigest(),
                         "b313ba78bdeaf2149f4804e3e00000709c33f1eea33adcedd693a9a9daa2aee6")
