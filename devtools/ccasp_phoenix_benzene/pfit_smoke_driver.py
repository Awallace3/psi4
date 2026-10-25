# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Bounded synthetic PFIT procedure. No molecule, native field generation or LW."""
import hashlib


def require(ok, message):
    if not ok:
        raise ValueError("synthetic PFIT smoke: "+message)


def digest(array):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest()


def synthetic_inputs(F, p):
    import numpy as np
    # Fictitious line of sites, not a molecular geometry. Alternating proper
    # rotations demonstrate local COPY placement, not physical frame parity.
    frames = (np.eye(3), np.diag([-1., -1., 1.]))
    sites = tuple(F.RefinementSite(f"{t}{k+1}", t, (float(3*k), float(t == "H"), 0.),
                                  frames[k % 2], 2) for t in "CH" for k in range(6))
    static = []
    for _ in sites:
        a = np.zeros((9, 9))
        a[1:, 1:] = .02
        static.append(a)
    static[6][3, 5] = static[6][5, 3] = 4e-5
    # Deliberate COPY disagreements must not change reference ownership.
    static[1][1, 1] = 9.
    static[7][3, 5] = static[7][5, 3] = 7.
    nodes = []
    for k in range(p["nodes"]):
        aa = [a.copy() for a in static]
        if k:
            aa[0][1, 1] = 3e-5
        if k >= 3:
            aa[6][3, 5] = aa[6][5, 3] = 2e-4
        nodes.append((k*.03, aa))
    rng = np.random.Generator(np.random.PCG64(p["seed"]))
    fields = rng.standard_normal((p["points"], p["channels"]))/np.sqrt(6.)
    points = np.column_stack((np.arange(p["points"]), np.zeros((p["points"], 2)))).astype(float)
    return sites, nodes, points, fields


def independent_design(model, fields):
    """Explicit packed-pair algebra, independent of PackedDesignRows/BLAS H."""
    import numpy as np
    i, j = np.tril_indices(len(fields))
    a = np.zeros((len(i), model.parameter_count))
    for k, entries in enumerate(model.parameter_entries):
        for site, r, c in entries:
            r, c = model.channel_offsets[site]+r, model.channel_offsets[site]+c
            a[:, k] += fields[i, r]*fields[j, c]
            if r != c:
                a[:, k] += fields[i, c]*fields[j, r]
    return a


def oracle(model, design, target):
    """min ||Az-y||² + sum w*(z-a)²; no normalization, pair weight or 1/2."""
    import numpy as np
    root = np.sqrt(model.strengths)
    augmented = np.vstack((design, np.diag(root)))
    rhs = np.r_[target, root*np.asarray(model.anchors)]
    z, _, rank, _ = np.linalg.lstsq(augmented, rhs, rcond=None)
    require(rank == model.parameter_count, "independent augmented design rank")
    return z


def objective(model, design, target, z):
    import numpy as np
    return float(np.sum((design@z-target)**2) +
                 np.sum(np.asarray(model.strengths)*(z-np.asarray(model.anchors))**2))


def resource_evidence(B, rows, p):
    """Real planner plus conservative source-counted dense/reference overhead."""
    n, c, k, nt = p["points"], p["channels"], p["max_parameters"], p["nodes"]
    budget = B.BoundedResources(p["max_bytes"], p["max_work"], p["max_io_bytes"])
    nr = n*(n+1)//2
    plan_bytes, work = B._pfit_plan(n, c, k, nt)
    # pfit.cc data_rows: two multiplies plus one add per (row,param,u,v),
    # rounded up to four arithmetic units for zeroing/writes. Not instruction
    # count, wall time or an assertion about validation-check overhead.
    # One dense fit at a time. data_rows is replayed for BOTH assembly and
    # residuals at every successful node; singular refusal is assembly only.
    dense_work = (2*nt+1)*nr*k*c*c*4
    # All independent design checks/SVD/lstsq and negative cases: generous
    # fixed 500M ceiling at these exact closed dimensions, no adaptive repeats.
    total_work = work + rows.planned_work + dense_work + 500_000_000
    # Real plan + streamed default native cap + dense default native cap +
    # 128MiB Python tensor/list/oracle overlap allowance (not measured RSS).
    numeric_bytes = plan_bytes + rows.planned_bytes + 2*256*1024**2 + 128*1024**2
    require(total_work <= budget.max_work, "work admission")
    require(numeric_bytes <= budget.max_bytes, "numeric byte admission")
    require(p["max_output_files"]*p["max_snapshot_bytes"] <= p["max_output_bytes"]
            <= budget.max_io_bytes, "output/I/O admission")
    return dict(real_pfit_plan_bytes=plan_bytes, real_pfit_plan_work=work,
        packed_plan_bytes=rows.planned_bytes, packed_plan_work=rows.planned_work,
        dense_source_work_bound=dense_work, total_work_bound=total_work,
        numeric_bytes_bound=numeric_bytes, numerical_cap=p["max_bytes"],
        output_bytes_bound=p["max_output_bytes"], io_bytes_bound=p["max_io_bytes"],
        files_bound=p["max_output_files"], not_measured_RSS=True)


def exercise(B, F, S, core, labels, p, emit):
    """Dependency-injected engine-free seam; production entry is run_native below."""
    import numpy as np
    sites, nodes, points, fields = synthetic_inputs(F, p)
    kw = dict(cutoff=1e-4, weight_type=3, weight_coefficient=.001,
              provenance="synthetic complete72; NOT molecular response",
              declared_variables=labels)
    models = B.refinement_models(sites, nodes, **kw)
    require(len(models) == 11, "eleven models")
    up, down = labels.index("H1_11s_21c_A"), labels.index("C1_10_10_A")
    static_hash = F._anchor_hash(F._validated_anchor_tensors(sites, nodes[0][1]))
    for m, (omega, aa) in zip(models, nodes):
        require(m.parameter_count == 72 and m.channel_count == 108 and
                m.parameter_labels == tuple(labels) and m.declared_variables == tuple(labels),
                "complete actual model inventory")
        require(m.reference_sites == (0, 6) and m.equivalent_sites == (tuple(range(6)), tuple(range(6, 12))),
                "reference/COPY ownership")
        require(m.static_reference_sha256 == static_hash and m.penalty_selection == "static_reference"
                and m.unpenalized_variables == ("H1_11s_21c_A",), "static identity/mask")
        require(m.anchors[up] == m.strengths[up] == 0., "upward extra remains FREE")
        require(m.anchors[down] == aa[0][1, 1] and m.strengths[down] == F.penalty_weight(
            weight_type=3, weight_coefficient=.001, alpha=aa[0][1, 1],
            frequency=omega, rank1=1, rank2=1), "downward current anchor/weight")
        for entries in m.parameter_entries:
            s, r, c = entries[0]
            require(1 <= r <= c <= 8 and entries == tuple(
                (member, r, c) for member in (range(6) if s == 0 else range(6, 12))),
                "all72 actual entry tables")
    refusal = {}
    for name, bad in (("duplicate", labels+(labels[0],)),
                      ("incomplete", tuple(v for v in labels if v != labels[up])),
                      ("out_of_rank", ("C1_30_30_A",)+labels[1:])):
        try:
            B.refinement_models(sites, nodes, **dict(kw, declared_variables=bad))
        except ValueError as error:
            refusal[name] = str(error)
        else:
            raise ValueError("declaration guard did not refuse "+name)

    independent = independent_design(models[0], fields)
    rows = S.PackedDesignRows(models[0], fields, block_rows=p["block_rows"],
                              max_passes=2, max_bytes=16*1024**2)
    resources = resource_evidence(B, rows, p)
    assembled = []
    pass_digests = []
    for pass_index in range(2):
        h = hashlib.sha256()
        count = 0
        for start, block in rows.blocks():
            require(start == count, "packed order/coverage")
            np.testing.assert_allclose(block, independent[start:start+len(block)], atol=1e-13, rtol=1e-13)
            h.update(np.ascontiguousarray(block, dtype="<f8").tobytes())
            if pass_index == 0:
                assembled.append(block)
            count += len(block)
        require(count == p["points"]*(p["points"]+1)//2, "all point pairs")
        pass_digests.append(h.hexdigest())
    require(rows.passes_used == 2 and pass_digests[0] == pass_digests[1], "bitwise design replay")
    try:
        rows.blocks()
    except RuntimeError:
        refusal["third_pass"] = "refused"
    else:
        raise ValueError("unbounded replay")
    design = np.concatenate(assembled)
    del assembled
    spectrum = np.linalg.svd(design, compute_uv=False)
    condition = float(spectrum[0]/spectrum[-1]) if spectrum[-1] > 0. else float("inf")
    require(np.linalg.matrix_rank(design) == 72 and condition <= p["max_design_condition"],
            "unpenalized full column rank/conditioning before solve")
    truths = []
    refs = []
    targets = np.empty((len(design), len(models)))
    conditions = []
    for t, m in enumerate(models):
        z = np.asarray(m.anchors) + .04*(1.+np.sin(np.arange(72)+t*.1))
        z[up] = .125+.001*t
        targets[:, t] = design@z
        truths.append(z)
        refs.append(oracle(m, independent, targets[:, t]))
        normal_condition = float(np.linalg.cond(independent.T@independent+np.diag(m.strengths)))
        require(normal_condition <= p["max_normal_condition"], "normal conditioning before solve")
        conditions.append(normal_condition)
        # Objective must actually test penalties, rather than a truth equal to
        # every anchor, or an oracle accidentally normalized by the row count.
        require(objective(m, independent, targets[:, t], refs[-1]) <
                objective(m, independent, targets[:, t], z), "active penalty objective")
        require(np.max(np.abs(refs[-1]-z)) > 1e-8, "penalty has measurable effect")
        require(np.max(np.abs(refs[-1]-z)) < 1e-4, "bounded known-target recovery bias")
    provenance = dict(target_origin=core.IsaPfitTargetOrigin.SyntheticAnalyticTest,
        source_id="ccasp-synthetic-complete72-pfit-smoke",
        generation_record=p["identity"]+"; supplied PCG64 fields; complete design-generated targets")
    emit("design.json", dict(status="DESIGN_ADMITTED_NOT_SOLVED", labels=list(labels),
        entries=models[0].parameter_entries, sites=[dict(label=s.label, type=s.site_type,
            origin=s.origin_bohr, frame=s.frame, rank=s.rank_limit) for s in sites],
        static_reference_sha256=static_hash, unpenalized=list(models[0].unpenalized_variables),
        design_rank=72, design_condition=condition, normal_conditions=conditions,
        singular_extrema=[float(spectrum[-1]), float(spectrum[0])],
        design_replay_sha256=pass_digests, independent_design_sha256=digest(independent),
        fields_sha256=digest(fields), targets_sha256=digest(targets), resources=resources))
    for t, m in enumerate(models):
        emit(f"model-{t:02d}.json", dict(status="MODEL_ADMITTED_NOT_SOLVED", node=t,
            frequency=m.frequency_au, labels=m.parameter_labels, entries=m.parameter_entries,
            anchors=m.anchors, strengths=m.strengths, anchor_sha256=m.anchor_sha256,
            static_reference_sha256=m.static_reference_sha256,
            static_mask=[v not in m.unpenalized_variables for v in labels],
            raw_upward_anchor=float(nodes[t][1][6][3, 5]), copy_discrepancy=m.copy_anchor_discrepancy,
            known_target_parameters=truths[t].tolist(), independent_solution=refs[t].tolist(),
            normal_condition=conditions[t]))

    # options=None deliberately reaches the actual Python default DSYSV path.
    # Its native default budget is 256MiB, included in the admission above.
    fits = S.refine_streamed(models, points, targets, fields=fields,
        block_rows=p["block_rows"], max_bytes=16*1024**2, **provenance)
    require(len(fits) == 11, "eleven streamed results")
    records = []
    for t, (m, fit) in enumerate(zip(models, fits)):
        # Also exercise actual IsaPfitProblem/full channel tensors, one at a
        # time (never retain eleven large Python tensor/list problems).
        problem = F.refinement_problem(m, points, targets[:, t], fields=fields, **provenance)
        require(len(problem.model.parameter_tensors) == 72 and
                list(problem.model.fixed) == [False]*72 and
                list(problem.model.parameter_labels) == list(labels),
                "actual72 dense entries all unfixed")
        for tensor, entries in zip(problem.model.parameter_tensors, m.parameter_entries):
            require(tensor.rows == tensor.cols == 108, "actual native tensor dimensions")
            expected_tensor = np.zeros((108, 108))
            for s, r, c in entries:
                offset = m.channel_offsets[s]
                expected_tensor[offset+r, offset+c] = 1.
                expected_tensor[offset+c, offset+r] = 1.
            require(np.array_equal(np.asarray(tensor.values).reshape(108, 108), expected_tensor),
                    "actual native COPY/mirror channel tensor entries")
        options = core.IsaPfitOptions()
        options.solver = core.IsaPfitSolver.NormalEquationsDSYSV
        dense = core.isa_pfit_solve(problem, options)
        require(dense.status == core.IsaPfitStatus.Solved and fit.status == core.IsaPfitStatus.Solved,
                "dense/streamed Solved")
        z = np.array(fit.parameters)
        for candidate in (z, np.array(dense.parameters)):
            np.testing.assert_allclose(candidate, refs[t], atol=p["parameter_atol"], rtol=p["parameter_rtol"])
        require(abs(z[up]) > .1 and abs(z[up]-refs[t][up]) <= p["parameter_atol"]+
                p["parameter_rtol"]*abs(refs[t][up]), "nonzero FREE recovery")
        for result in (fit.result, dense):
            d = result.diagnostics
            require(result.settings.solver == core.IsaPfitSolver.NormalEquationsDSYSV,
                    "no QR production substitution")
            require(list(result.parameter_labels) == list(labels) and
                    list(result.channel_labels) == list(m.channel_labels) and
                    result.frequency_au == m.frequency_au, "actual native result inventory/frequency")
            require(list(d.free_indices) == list(range(72)) and d.numerical_rank == 72
                    and d.data_rows == len(design) and d.objective_available,
                    "actual native rank/free/data/objective inventory")
            require(d.work_budget_bytes <= 256*1024**2 and d.lapack_info == 0 and
                    d.condition_estimate_available and d.normal_h_rcond >= 1e-6,
                    "native budget/condition/status diagnostics")
            expected = objective(m, independent, targets[:, t], np.array(result.parameters))
            np.testing.assert_allclose(d.total_objective, expected,
                atol=p["objective_atol"], rtol=p["objective_rtol"])
            np.testing.assert_allclose(np.asarray(result.matrix_penalty_rhs),
                np.asarray(m.strengths)*m.anchors, atol=1e-15, rtol=1e-13)
            np.testing.assert_allclose(np.asarray(result.effective_penalty_matrix.values).reshape(72, 72),
                np.diag(m.strengths), atol=1e-15, rtol=1e-13)
        for value, entries in zip(z, m.parameter_entries):
            for s, r, c in entries:
                require(fit.refined_tensors[s][r, c] == value ==
                        fit.refined_tensors[s][c, r], "COPY/mirror tensor placement")
        require(all(np.array_equal(fit.refined_tensors[0], fit.refined_tensors[s]) for s in range(6))
                and all(np.array_equal(fit.refined_tensors[6], fit.refined_tensors[s]) for s in range(6, 12)),
                "all COPY tensors identical in local frames")
        error = float(np.max(np.abs(z-refs[t])))
        record = dict(status="Solved", node=t, frequency=m.frequency_au,
            static_reference_sha256=m.static_reference_sha256,
            parameters=z.tolist(), dense_parameters=list(dense.parameters),
            independent_reference=refs[t].tolist(), reference_maxabs_error=error,
            known_target_maxabs_shift=float(np.max(np.abs(z-truths[t]))),
            free_extra=float(z[up]), normal_condition=conditions[t],
            normal_h_rcond=fit.diagnostics.normal_h_rcond,
            native_work_budget_bytes=fit.diagnostics.work_budget_bytes,
            total_objective=fit.diagnostics.total_objective,
            parameters_sha256=digest(z), native_verified=bool(fit.diagnostics.native_verified),
            no_molecular_acceptance=True)
        emit(f"solution-{t:02d}.json", record)
        records.append(record)
        del problem, dense

    # Zero design AND zero penalties: rank must fail, never regularize/fix or
    # expose parameters. Use the real bounded constructor (weight_type=0).
    free = B.refinement_models(sites, nodes[:1], **dict(kw, weight_type=0))[0]
    zero_fields = np.zeros_like(fields)
    problem = F.refinement_problem(free, points, targets[:, 0], fields=zero_fields, **provenance)
    options = core.IsaPfitOptions()
    options.solver = core.IsaPfitSolver.NormalEquationsDSYSV
    singular = core.isa_pfit_solve(problem, options)
    require(singular.status == core.IsaPfitStatus.RankDeficient, "singular unpenalized refusal")
    try:
        singular.parameters
    except RuntimeError:
        refusal["singular_parameters"] = "withheld"
    else:
        raise ValueError("singular parameters not withheld")
    try:
        S.refine_streamed([free], points, targets[:, :1], fields=zero_fields,
            block_rows=p["block_rows"], max_bytes=16*1024**2, **provenance)
    except ValueError as error:
        require("did not solve" in str(error), "singular stream failed for wrong reason")
        refusal["singular_stream"] = str(error)
    else:
        raise ValueError("singular stream falsely succeeded")
    emit("negative-cases.json", dict(status="EXPECTED_REFUSALS", cases=refusal))
    return dict(status="SYNTHETIC_CHECKS_PASSED_PENDING_FINAL_CLOSURE", solved=11,
        design_condition=condition, max_reference_error=max(r["reference_maxabs_error"] for r in records),
        static_reference_sha256=static_hash, design_replay_sha256=pass_digests,
        resources=resources, refusal_cases=refusal, no_molecular_acceptance=True)


def run_native(runtime, emit):
    # First statement is deliberately a compute guard. No NumPy/scientific
    # imports or native calls at module import or ahead of this guard.
    runtime.compute_guard()
    from psi4 import core
    from psi4.driver.procrouting import isapol_bounded as B
    from psi4.driver.procrouting import isapol_refine as F
    from psi4.driver.procrouting import isapol_pfit_stream as S
    profile = runtime.module("_smoke_science_profile", "pfit_smoke_profile.py")
    p, _ = profile.configuration(runtime)
    return exercise(B, F, S, core, profile.declaration(runtime.HERE), p, emit)
