# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Engine-free diagnostic integration toys; no native runtime or molecular evidence."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
ROUTING = HERE.parents[1]/"psi4/driver/procrouting"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def function(name, file, env):
    tree = ast.parse((ROUTING/file).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROUTING/file),"exec"),env)
    return env[name]


class Matrix:
    """Small stdlib-only live operand, not a NumPy stand-in for ABI claims."""
    dtype = SimpleNamespace(str="<f8")

    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, key):
        a, b = key
        return self.rows[a][b]

    def tobytes(self, order="C"):
        assert order == "C"
        return b"".join(struct.pack("<d", x) for row in self.rows for x in row)


class Vector:
    dtype = Matrix.dtype

    def __init__(self, values):
        self.values, self.shape = values, (len(values),)

    def __getitem__(self, i):
        return self.values[i]


class BytesView:
    dtype = Matrix.dtype

    def __init__(self, shape, data):
        self.shape, self.data = shape, data

    def tobytes(self, order="C"):
        assert order == "C"
        return self.data


def toy():
    F = Matrix([[2.,1.],[-1.,3.]])
    Q = Matrix([[float(t+1), float(t%3-1)] for t in range(50)])
    Q.rows[0], Q.rows[25] = [1.,2.], [3.,4.]
    B = Matrix([[sum(F[i,k]*Q[j,k] for k in range(2)) for j in range(50)] for i in range(2)])
    X = Matrix([[float(j+1) for j in range(50)], [float(2-j) for j in range(50)]])
    raw = Matrix([[-sum(B[k,i]*X[k,j] for k in range(2)) for j in range(50)] for i in range(50)])
    return F, Q, B, X, raw


class IntegratedDiagnosticTests(unittest.TestCase):
    def sink(self, tmp):
        module = load("integrated_sink_test", HERE/"diagnostic_sink.py")
        helpers = load("capture_toy_inputs", HERE/"test_lw_capture.py")
        return module.DiagnosticSink(Path(tmp)/"diagnostic",helpers.identities(),kind="synthetic")

    def test_live_signed_attribution_matches_independent_integer_oracle(self):
        module = load("integrated_sink_test", HERE/"diagnostic_sink.py")
        helpers = load("capture_toy_inputs", HERE/"test_lw_capture.py")
        with tempfile.TemporaryDirectory() as tmp:
            sink = module.DiagnosticSink(Path(tmp)/"diagnostic", helpers.identities(), kind="synthetic")
            sink.plan(2, 2, 50, 1)
            F,Q,B,X,raw = toy()
            sink.auxiliary_charges(Vector([3.,5.]))
            sink.anchor(F,Q,B)
            sink.solved(0.,B,X)
            values = sink.vectors
            self.assertEqual(list(values["deltaq"]), [1.,1.])
            self.assertEqual(list(values["Fq"]), [11.,12.])
            self.assertEqual(list(values["Fdeltaq"]), [3.,2.])
            self.assertEqual(list(values["Bc"]), [14.,14.])
            self.assertEqual(list(sink.node_vectors["sfit"]), [float(j-35) for j in range(50)])
            self.assertEqual(list(sink.node_vectors["sQ"]), [float(-j-7) for j in range(50)])
            self.assertEqual(list(sink.node_vectors["minus_BcT_X"]), [-42.]*50)
            for operand in (F,Q,B,X,raw):
                self.assertFalse(any(item is operand for item in vars(sink).values()))

    def test_real_pre_lw_seam_captures_once_then_preserves_original_refusal(self):
        events = []
        error = RuntimeError("original unchanged strict LW refusal")
        def strict(**kwargs):
            events.append(("lw",kwargs))
            raise error
        sink = SimpleNamespace(before_lw=lambda **kwargs: events.append(("capture",kwargs)))
        localize = function("_localize_node","isapol_bounded.py",
                            {"lw":SimpleNamespace(supplied_nonlocal_properties=strict)})
        kwargs = dict(tensors=object(),provenance=object(),frequencies=[0.],input_rank=4,
            residual_policy="production",localization_rank_limit=2, truncation="discard_rank4_rows_and_columns")
        with self.assertRaises(RuntimeError) as caught:
            localize(diagnostic_sink=sink, raw="actual raw", source={"actual":"source"}, **kwargs)
        self.assertIs(caught.exception,error)
        self.assertEqual([event[0] for event in events],["capture","lw"])
        self.assertEqual(events[1][1],kwargs)
        self.assertEqual(events[0][1],dict(kwargs,raw="actual raw",source={"actual":"source"}))

    def test_default_none_pre_lw_path_has_no_capture_or_keyword_change(self):
        strict = mock.Mock(return_value="unchanged result")
        localize = function("_localize_node","isapol_bounded.py",
                            {"lw":SimpleNamespace(supplied_nonlocal_properties=strict)})
        payload = object()
        self.assertEqual(localize(raw=object(),source=object(),tensors=payload),"unchanged result")
        strict.assert_called_once_with(tensors=payload)

    def test_capture_failure_prevents_strict_lw_and_downstream(self):
        strict = mock.Mock(side_effect=AssertionError("must not reach LW"))
        localize = function("_localize_node","isapol_bounded.py",
                            {"lw":SimpleNamespace(supplied_nonlocal_properties=strict)})
        failure = OSError("exclusive diagnostic storage unavailable")
        sink = SimpleNamespace(before_lw=mock.Mock(side_effect=failure))
        with self.assertRaises(OSError) as caught:
            localize(diagnostic_sink=sink,raw=object(),source={},tensors=object())
        self.assertIs(caught.exception,failure)
        strict.assert_not_called()
        sink.before_lw.assert_called_once()

    def test_complete_real_seam_captures_full_bytes_and_signed_axes_before_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink = self.sink(tmp)
            sink.plan(2,2,50,1)
            F,Q,B,X,raw = toy()
            before = [m.tobytes() for m in (F,Q,B,X,raw)]
            sink.auxiliary_charges(Vector([3.,5.]))
            sink.anchor(F,Q,B)
            sink.solved(0.,B,X)
            pair = sink.tools.site_major_to_pairs(sink.tools.Float64Array((50,50),raw.tobytes()))
            refusal = RuntimeError("strict production refusal; not caught or retried")
            def strict(**kwargs):
                marker = sink.root/"node-01/pre-lw-complete.json"
                self.assertTrue(marker.is_file())
                info = json.loads(marker.read_bytes())
                self.assertFalse(info["properties_accepted"])
                capture = sink.tools.load_capture(info["capture"],info["lw_manifest_sha256"])
                self.assertEqual(capture.array("tensors").data,pair.data)
                self.assertEqual(capture.array("raw_site_major").data,raw.tobytes())
                columns = struct.unpack("<50d",capture.array("sum_response_site").data)
                self.assertEqual(columns,(-42.,)*50)  # integer oracle, independent of sink
                row_expected = tuple(-27*B[0,j]+21*B[1,j] for j in range(50))
                self.assertEqual(struct.unpack("<50d",capture.array("sum_potential_site").data),row_expected)
                self.assertFalse((Path(tmp)/"properties.json").exists())
                raise refusal
            boundary = mock.Mock(side_effect=strict)
            localize = function("_localize_node","isapol_bounded.py",
                {"lw":SimpleNamespace(supplied_nonlocal_properties=boundary)})
            with self.assertRaises(RuntimeError) as caught:
                localize(raw=raw,source={"toy":"not molecular evidence"},diagnostic_sink=sink,
                    tensors=BytesView(pair.shape,pair.data),origins=Matrix([[0.,0.,0.],[1.,2.,3.]]),
                    frames=[[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]]*2,
                    labels=["Left","Right"],bonds=[(1,0)],frequencies=[0.],input_rank=4,
                    truncation="discard_rank4_rows_and_columns",residual_policy="production",
                    localization_rank_limit=2,provenance="toy")
            self.assertIs(caught.exception,refusal)
            boundary.assert_called_once()
            self.assertEqual(before,[m.tobytes() for m in (F,Q,B,X,raw)])
            vectors = json.loads((sink.root/"node-01/raw-axis-comparisons/manifest.json").read_bytes())
            for descriptor in vectors["arrays"].values():
                data = (sink.root/"node-01/raw-axis-comparisons"/descriptor["file"]).read_bytes()
                self.assertEqual(struct.unpack("<50d",data),(0.,)*50)
            self.assertLess(sink.written,sink.output_bytes)

    def test_solved_seam_selects_actual_anchor_not_target_columns_once(self):
        observe = function("_diagnose_solved","isapol_bounded_response.py",{})
        events = []
        class Slice:
            def __init__(self,name): self.name = name
            def __getitem__(self,key):
                events.append((self.name,key))
                return self.name+" actual anchor"
        sink = SimpleNamespace(solved=mock.Mock())
        observe(sink,.375,Slice("legs"),Slice("solved"),2)
        self.assertEqual(events,[("legs",(slice(None),slice(2,None))),
                                 ("solved",(slice(None),slice(2,None)))])
        sink.solved.assert_called_once_with(.375,"legs actual anchor","solved actual anchor")
        observe(None,.375,object(),object(),2)  # default None never even slices operands

    def test_actual_native_charge_callsite_observes_existing_vector_and_failure_stops_metric(self):
        tree = ast.parse((ROUTING/"isapol_native_factors.py").read_text())
        fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="native_constrained_ov")
        index = next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id=="charges" for t in n.targets))
        code = compile(ast.Module(body=fn.body[index:index+3],type_ignores=[]),"actual-charge-seam","exec")
        vector = Vector([3.,5.])
        provider = SimpleNamespace(charges=mock.Mock(return_value=vector),metric=mock.Mock(return_value="metric"))
        observer = mock.Mock()
        env = dict(np=SimpleNamespace(asarray=lambda x:x),provider=provider,
                   diagnostic_sink=SimpleNamespace(auxiliary_charges=observer))
        exec(code,env)
        provider.charges.assert_called_once_with()
        observer.assert_called_once_with(vector)
        provider.metric.assert_called_once_with()
        provider.metric.reset_mock()
        error = OSError("diagnostic refusal")
        env["diagnostic_sink"].auxiliary_charges = mock.Mock(side_effect=error)
        with self.assertRaises(OSError) as caught:
            exec(code,env)
        self.assertIs(caught.exception,error)
        provider.metric.assert_not_called()
        env["diagnostic_sink"] = None
        exec(code,env)
        provider.metric.assert_called_once_with()

    def test_resource_declaration_and_io_fail_before_observation(self):
        allowance = function("_diagnostic_bytes","isapol_bounded_response.py",{})
        admission = function("_admit_diagnostics","isapol_bounded_response.py",{})
        self.assertEqual(allowance(None),0)
        with tempfile.TemporaryDirectory() as tmp:
            sink = self.sink(tmp)
            self.assertEqual(allowance(sink),32*1024**2)
            error = ValueError("shared byte budget refused")
            ledger = SimpleNamespace(admit=mock.Mock(side_effect=error),charge_io=mock.Mock())
            with self.assertRaises(ValueError) as caught:
                admission(sink,ledger,2,2,50,1)
            self.assertIs(caught.exception,error)
            ledger.charge_io.assert_not_called()
            self.assertFalse(sink.vectors)
        ledger = SimpleNamespace(admit=mock.Mock(),charge_io=mock.Mock(side_effect=ValueError("I/O budget")))
        with tempfile.TemporaryDirectory() as tmp:
            sink = self.sink(tmp)
            with self.assertRaisesRegex(ValueError,"I/O budget"):
                admission(sink,ledger,2,2,50,1)
            self.assertFalse(sink.vectors)
        with self.assertRaises(ValueError):
            allowance(SimpleNamespace(memory_bytes=True,io_bytes=100))

    def test_dimension_output_and_repeated_capture_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink = self.sink(tmp)
            with self.assertRaisesRegex(ValueError,"dimensions"):
                sink.plan(2049,2,50,1)
            sink.output_bytes = sink.written
            with self.assertRaisesRegex(ValueError,"output byte"):
                sink.plan(2,2,50,1)
        with tempfile.TemporaryDirectory() as tmp:
            sink = self.sink(tmp)
            sink.plan(2,2,50,1)
            sink.auxiliary_charges(Vector([3.,5.]))
            with self.assertRaisesRegex(ValueError,"exactly once"):
                sink.auxiliary_charges(Vector([3.,5.]))
            F,Q,B,X,raw = toy()
            sink.anchor(F,Q,B)
            sink.solved(0.,B,X)
            with self.assertRaisesRegex(ValueError,"one existing anchor solve"):
                sink.solved(0.,B,X)

    def test_compensated_fit_diagnostics_are_separate_and_never_replace_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink = self.sink(tmp)
            sink.plan(3,1,25,1)
            sink.auxiliary_charges(Vector([1.,1.,1.]))
            F = Matrix([[1e16,1.,-1e16]])
            Q = Matrix([[1.,1.,1.]]+[[0.,0.,0.]]*24)
            B = Matrix([[0.]*25])  # actual ordinary toy contraction, not repaired
            before = F.tobytes(), Q.tobytes(), B.tobytes()
            sink.anchor(F,Q,B)
            self.assertEqual(list(sink.vectors["Fq"]),[0.])
            self.assertEqual(list(sink.vectors["Fq_fsum"]),[1.])
            sink.solved(0.,B,Matrix([[1.]*25]))
            self.assertEqual(list(sink.node_vectors["sfit"]),[0.]*25)
            self.assertEqual(before,(F.tobytes(),Q.tobytes(),B.tobytes()))

    def test_oeprop_keyword_is_optin_and_default_none(self):
        settings = function("settings","isapol_bounded_oeprop.py",dict(
            MAX_PARAMETERS=64,OPTION_MAP={},KernelSmoothing=lambda *args:"unchanged smoothing"))
        self.assertIsNone(settings("benzene",{})["diagnostic_sink"])
        marker = object()
        self.assertIs(settings("benzene",{"diagnostic_sink":marker})["diagnostic_sink"],marker)
        tree = ast.parse((ROUTING/"isapol_bounded.py").read_text())
        fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="bounded_properties")
        args = dict(zip((n.arg for n in fn.args.kwonlyargs),fn.args.kw_defaults))
        self.assertIsNone(ast.literal_eval(args["diagnostic_sink"]))
        # Native-build and basis inputs remain byte-identical in the new worktree.
        self.assertEqual(subprocess.check_output(["git","diff","--name-only",
            "19f7b23255382b8551b864cbbfa62ffb13efaadb","--","psi4/src","psi4/share/psi4/basis"],
            cwd=HERE,text=True),"")

    def test_default_native_fit_and_response_arithmetic_ast_is_unchanged(self):
        # Remove only documented observation plumbing and docs, compare actual
        # formula AST against accepted19f7. No native package import is required.
        def target(text, name, cls=None):
            tree = ast.parse(text)
            body = tree.body if cls is None else next(n.body for n in tree.body
                if isinstance(n,ast.ClassDef) and n.name==cls)
            return deepcopy(next(n for n in body if isinstance(n,ast.FunctionDef) and n.name==name))
        def normalized(node):
            node.body = [n for n in node.body if not isinstance(n,ast.Expr)
                         or not isinstance(n.value,ast.Constant) or not isinstance(n.value.value,str)]
            return ast.dump(node,include_attributes=False)
        for file,name,cls in (("isapol_native_factors.py","native_constrained_ov",None),
                              ("isapol_bounded_response.py","solve","_H2H1Response")):
            old = subprocess.check_output(["git","show",
                "19f7b23255382b8551b864cbbfa62ffb13efaadb:psi4/driver/procrouting/"+file],
                cwd=HERE,text=True)
            baseline, current = target(old,name,cls), target((ROUTING/file).read_text(),name,cls)
            if cls is None:
                keep = [(a,d) for a,d in zip(current.args.kwonlyargs,current.args.kw_defaults)
                        if a.arg!="diagnostic_sink"]
                current.args.kwonlyargs, current.args.kw_defaults = map(list,zip(*keep))
                current.body = [n for n in current.body if not isinstance(n,ast.If)
                    or "diagnostic_sink" not in ast.unparse(n.test)]
            else:
                current.body = [n for n in current.body if not isinstance(n,ast.Expr)
                    or not isinstance(n.value,ast.Call) or not isinstance(n.value.func,ast.Name)
                    or n.value.func.id!="_diagnose_solved"]
            self.assertEqual(normalized(current),normalized(baseline),file)


if __name__ == "__main__":
    unittest.main()
