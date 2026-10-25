# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Engine-free source accounting at the ACTUAL 72-column inventory."""
import ast
from pathlib import Path
from types import SimpleNamespace

ROUTING = Path(__file__).resolve().parents[2]/"psi4/driver/procrouting"


def expression(node, env):
    return eval(compile(ast.Expression(node), "<source resource expression>", "eval"), dict(env))


def stage_work(file, stage, env):
    tree = ast.parse((ROUTING/file).read_text())
    matches = [n.args[2] for n in ast.walk(tree) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == "admit" and len(n.args) > 2
        and isinstance(n.args[0], ast.Constant) and n.args[0].value == stage]
    if len(matches) != 1:
        raise AssertionError((file, stage, len(matches)))
    return expression(matches[0], env)


def design_admission(stream, model, npoint=2000, max_bytes=512*1024**2):
    """Execute unchanged __init__ prefix through byte/work guards, NO allocations/replay."""
    tree = ast.parse((ROUTING/"isapol_pfit_stream.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PackedDesignRows")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    body = []
    for n in init.body:
        if isinstance(n, ast.If) and ast.unparse(n.test) == "not np.isfinite(fields).all()":
            break
        body.append(n)
    else:
        raise AssertionError("source admission boundary changed")
    body.append(ast.parse("return dict(planned_bytes=planned, planned_work=max_passes*work, used_channels=len(used))").body[0])
    fn = ast.FunctionDef(name="metadata_init", args=init.args, body=body,
                        decorator_list=[], returns=None, type_comment=None)
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    env = dict(vars(stream), MAX_BYTES=stream.PackedDesignRows.MAX_BYTES)
    exec(compile(module, "<unmodified PackedDesignRows admission prefix>", "exec"), env)
    # An ndarray subclass backed by only one scalar; metadata is actual full shape.
    # No content is read by the prefix and no full-sized field array is allocated.
    import numpy as np
    fields = np.lib.stride_tricks.as_strided(np.zeros(1), shape=(npoint, model.channel_count),
                                            strides=(0, 0), writeable=False)
    return env["metadata_init"](stream.PackedDesignRows.__new__(stream.PackedDesignRows),
        model, fields, block_rows=4096, max_passes=2, max_bytes=max_bytes)


def evidence(bounded, isa_evidence):
    """Source scalar work + checkpoint byte arithmetic; no tensors/solver work."""
    p, no, nv, q, nf, npoint = 1146, 21, 393, 300, 11, 2000
    n = no*nv
    env = dict(p=p, no=no, nv=nv, n=n, q=q, nf=nf, npoint=npoint, rows=273240)
    file = "isapol_bounded_response.py"
    multiplicity = {
        "native factor work reservation": 1, "anchor fit work reservation": 0,
        "distributed moment contraction": 1, "full-grid AUX kernel": 1,
        "dense operator assembly": 1, "operator stability": 1,
        "H2H1 response operator": 1, "frequency response": nf}
    # The two fit stages have dynamic names; extract their one shared expression.
    tree = ast.parse((ROUTING/file).read_text())
    dynamic = [c.args[2] for c in ast.walk(tree) if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Attribute) and c.func.attr == "admit" and len(c.args) == 3
        and ast.unparse(c.args[0]) == "name + ' fit work reservation'"]
    assert len(dynamic) == 1
    work = {s: times*stage_work(file, s, dict(env, moments=True,
                    n=p if s == "full-grid AUX kernel" else n))
            for s, times in multiplicity.items() if times}
    # supplied ISA moments: the conditional 2000*p*q term is zero.
    work["distributed moment contraction"] = stage_work(file, "distributed moment contraction",
                                                         dict(env, moments=True))
    work["two constrained fits"] = 2*expression(dynamic[0], env)
    work["ISA"] = isa_evidence["ISA_work"]
    diag = ast.parse((Path(__file__).with_name("diagnostic_sink.py")).read_text())
    plan = next(c for c in ast.walk(diag) if isinstance(c, ast.FunctionDef) and c.name == "plan")
    diag_work = next(c.value for c in plan.body if isinstance(c, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "work" for t in c.targets))
    work["diagnostics"] = expression(diag_work, dict(env, m=q))
    work["LW"] = nf*8*q**3
    work["point targets"] = nf*stage_work("isapol_bounded.py", "point-response targets", env)
    pfit_bytes, work["PFIT"] = bounded._pfit_plan(npoint, 108, 72, nf)
    # recoupled_plan enforces each of its two work terms <=100M before return.
    # Use the conservative sum, not a native table query or inherited magic total.
    work["recoupled source ceiling"] = 200_000_000
    # Exact _Store charge rule is array payload + 256 on EVERY save/load.
    # factor tile sizes follow native_plain_df_operators(... tile_columns=512).
    records = dict(gaps=n, oo=p*no*no, ov=p*n, density=p,
                   anchor=n*q, target=n*p, kernel=p*p, h1=n*n, h2=n*n)
    for kind, size in (("ov", n), ("vv", nv*nv)):
        for i, start in enumerate(range(0, size, 512)):
            records[f"dual{kind}{i}"] = p*min(512, size-start)
    loads = ["density", "oo", "ov", "target", "kernel", "gaps",
             "h1", "h2", "h1", "h2", "target", "anchor"]
    loads += [key for key in records if key.startswith("dual")]
    store = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "_Store")
    charges = {}
    for action in ("save", "load"):
        method = next(c for c in store.body if isinstance(c, ast.FunctionDef) and c.name == action)
        charges[action] = next(c.args[0] for c in ast.walk(method) if isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute) and c.func.attr == "charge_io")
    import numpy as np
    checkpoint_io = sum(expression(charges["save"], dict(a=SimpleNamespace(nbytes=8*size)))
                        for size in records.values())
    checkpoint_io += sum(expression(charges["load"], dict(np=np, shape=(records[key],)))
                         for key in loads)
    checkpoint_io = int(checkpoint_io)
    return dict(dimensions=env, work_stages=work, fullchain_source_work_bound=sum(work.values()),
        checkpoint_writes=len(records), checkpoint_reads=len(loads), checkpoint_io_bytes=checkpoint_io,
        diagnostics_io_bytes=256*1024**2, total_io_bytes=checkpoint_io+256*1024**2,
        pfit_plan_bytes=pfit_bytes, pfit_plan_work=work["PFIT"])
