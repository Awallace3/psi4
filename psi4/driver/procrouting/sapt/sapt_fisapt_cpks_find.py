"""CPU reference callbacks for explicitly XC-coupled native FISAPT::find.

The native source/readout loop stays native. One original whole-monomer
cphf_Hx operator is borrowed per direction; only source solves cross this
reference seam. XC-off never installs these callbacks and retains legacy K.
"""
import numpy as np

from . import sapt_fisapt_cpks as response
from .sapt_fisapt_cpks_solver import solve_positive


def validate_request(core, *, external_potentials=None):
    """Native-only, opt-in admission; never reinterpret an existing default."""
    if not core.get_option("FISAPT", "FISAPT_FSAPT_IND_XC_RESPONSE"):
        return False
    response._require_cpu(core)
    if external_potentials:
        raise NotImplementedError("XC-coupled find does not admit external potentials")
    if not core.get_option("FISAPT", "FISAPT_FSAPT_IND_RESPONSE"):
        raise ValueError("FISAPT_FSAPT_IND_XC_RESPONSE requires FISAPT_FSAPT_IND_RESPONSE=true")
    if core.get_option("SAPT", "SAPT_DFT_DO_FSAPT") != "FISAPT":
        raise NotImplementedError("XC-coupled find requires the native SAPT_DFT_DO_FSAPT=FISAPT backend")
    if core.get_option("SAPT", "SAPT_DFT_INDUCTION_TYPE") != "CPKS":
        raise NotImplementedError("XC-coupled find requires canonical SAPT_DFT_INDUCTION_TYPE=CPKS")
    if core.get_option("SAPT", "SAPT_DFT_DO_DDFT"):
        raise NotImplementedError("XC-coupled find does not admit deltaDFT redistribution")
    if core.get_option("SAPT", "SAPT_DFT_FUNCTIONAL").upper() not in ("HF", "SVWN", "PBE"):
        raise NotImplementedError("XC-coupled find currently admits only HF limit and unshifted SVWN/PBE")
    if (core.get_option("SAPT", "SAPT_DFT_GRAC_COMPUTE") != "NONE"
            or core.get_option("SAPT", "SAPT_DFT_GRAC_SHIFT_A") != 0.
            or core.get_option("SAPT", "SAPT_DFT_GRAC_SHIFT_B") != 0.):
        raise NotImplementedError("XC-coupled find does not admit GRAC computation or shifted references")
    if (core.get_option("FISAPT", "FISAPT_LINK_ASSIGNMENT") != "C"
            or core.get_option("FISAPT", "SSAPT0_SCALE")
            or core.get_option("FISAPT", "FISAPT_EXCH_PARPERP")
            or core.get_option("SAPT", "DO_IND_EXCH_SINF")):
        raise NotImplementedError("XC-coupled find currently requires C assignment and unscaled S2 readout")
    for name in ("PCM", "PE", "DDX"):
        if name in core.get_global_option_list() and core.get_option("SCF", name):
            raise NotImplementedError("External response contributions are not admitted by XC-coupled find")
    return True


class FindCPKS:
    """Single-use native-find solve/finish contract, with bounded O(ov) state.

    ``solve`` returns +H^-1 W: native find applies the existing minus sign
    exactly once before localized readout. Internally, physical amplitudes
    are -H^-1 W, as in the standalone CPU reference.
    """
    def __init__(self, core, dimer, monomers, cache):
        self.core = core
        self.rtol = float(core.get_option("SAPT", "CPHF_R_CONVERGENCE"))
        self.maxiter = int(core.get_option("SAPT", "MAXITER"))
        if not 0 < self.rtol <= 1e-10 or not 1 <= self.maxiter <= 1024:
            raise ValueError("XC-coupled find requires CPHF_R_CONVERGENCE<=1e-10 and bounded MAXITER")
        if any(dimer.has_potential_variable(side) for side in ("A", "B", "C")):
            raise NotImplementedError("External source/reference potentials are not admitted by XC-coupled find")
        if len(dimer.molecule().get_fragments()) != 2:
            raise NotImplementedError("XC-coupled find currently requires two unembedded monomers")
        self.state, self.finished = {}, False
        natoms = dimer.molecule().natom()
        # Admit both directions' reference scratch BEFORE native tensor work.
        budget = min(64 * 1024**2, core.get_memory() // 8)
        required = 0
        for side in ("A", "B"):
            nao, occupied = cache["Cocc_" + side].np.shape
            virtual = cache["Cvir_" + side].np.shape[1]
            required += 8 * 20 * occupied * virtual
            required += response._validation_work_bytes(nao, occupied, virtual)
        if required > budget:
            raise MemoryError("XC-coupled find exceeds its bounded CPU response-array admission")
        for side, other in (("A", "B"), ("B", "A")):
            wfn = monomers[side]
            # The canonical SAPT response attachment is borrowed, not rebuilt,
            # finalized, or replaced with the bare readout JK configuration.
            operator = response.NativeMonomerCPKS(
                core, wfn, wfn.jk(), grac_shift=0.,
                orbital_shift=core.get_option("SCF", "LEVEL_SHIFT"),
                external_response=False, max_validation_bytes=budget)
            co, cv = (np.asarray(cache[key + side].np) for key in ("Cocc_", "Cvir_"))
            for cached, actual in (
                (co, wfn.Ca_subset("AO", "OCC").np),
                (cv, wfn.Ca_subset("AO", "VIR").np),
                (cache["eps_occ_" + side].np, wfn.epsilon_a_subset("AO", "OCC").np),
                (cache["eps_vir_" + side].np, wfn.epsilon_a_subset("AO", "VIR").np),
            ):
                response._array(cached, np.shape(actual), "native find cached canonical reference")
                response._close(cached, actual, "native find original canonical reference", 1e-12, 0.)
            total = co.T @ (cache["V_" + other].np + 2 * cache["J_" + other].np) @ cv
            self.state[side] = {
                "operator": operator, "total": np.array(total, copy=True),
                "rhs_sum": np.zeros_like(total), "amplitude_sum": np.zeros_like(total),
                "expected_sources": natoms + cache["Cocc_" + other].np.shape[1] + 1,
                "sources": 0, "max_residual": 0., "iterations": 0,
            }

    def solve(self, side, source_index, matrix):
        if self.finished or side not in self.state:
            raise ValueError("Live native CPKS direction required")
        state = self.state[side]
        if (source_index != state["sources"] or source_index >= state["expected_sources"]):
            raise ValueError("Native CPKS sources must arrive once in physical slot order")
        shape = state["operator"].gap.shape
        value = np.asarray(matrix.np)
        # DFHelper retains a singleton disk-tensor axis on a 2-D core.Matrix.
        if value.shape == (1, *shape):
            value = value[0]
        rhs = response._array(value, shape, "native find source RHS").copy()
        result = solve_positive(state["operator"].apply, state["operator"].gap, rhs,
                                rtol=self.rtol, atol=1e-12, max_iterations=self.maxiter)
        physical_residual = float(np.linalg.norm(state["operator"].apply(result.amplitudes) + rhs))
        if physical_residual > max(1e-12, self.rtol * np.linalg.norm(rhs)):
            raise RuntimeError("Native CPKS physical true residual exceeds unchanged tolerance")
        state["rhs_sum"] += rhs
        state["amplitude_sum"] += result.amplitudes
        state["sources"] += 1
        state["max_residual"] = max(state["max_residual"], physical_residual)
        state["iterations"] += result.iterations
        return self.core.Matrix.from_array(-result.amplitudes)

    def finish(self):
        if self.finished:
            raise ValueError("Native CPKS callbacks are single-use")
        diagnostics = {}
        for side, direction in (("A", "A<-B"), ("B", "B<-A")):
            state = self.state[side]
            if state["sources"] != state["expected_sources"]:
                raise RuntimeError("Incomplete native CPKS source partition")
            response._close(state["rhs_sum"], state["total"], "native CPKS source completeness", 2e-9, 1e-7)
            aggregate = solve_positive(
                state["operator"].apply, state["operator"].gap, state["total"],
                rtol=self.rtol, atol=1e-12, max_iterations=self.maxiter)
            aggregate_residual = float(np.linalg.norm(
                state["operator"].apply(aggregate.amplitudes) + state["total"]))
            if aggregate_residual > max(1e-12, self.rtol*np.linalg.norm(state["total"])):
                raise RuntimeError("Native CPKS aggregate physical true residual exceeds tolerance")
            response._close(state["amplitude_sum"], aggregate.amplitudes,
                            "native CPKS source amplitude additivity", 2e-9, 1e-7)
            for name, value in (("SOURCES", state["sources"]), ("ITERATIONS", state["iterations"]),
                                ("MAX TRUE RESIDUAL", state["max_residual"]),
                                ("AGGREGATE TRUE RESIDUAL", aggregate_residual)):
                diagnostics[f"CPKS {name} ({direction})"] = float(value)
            diagnostics[f"CPKS HF REFERENCE LIMIT ({direction})"] = float(
                state["operator"].mode == response.Mode.CPHF)
        self.finished = True
        return diagnostics


def install(core, fisapt, dimer, wfn_a, wfn_b, cache):
    try:
        if not validate_request(core):
            return None
        if not hasattr(fisapt, "set_cpks_response_callbacks"):
            raise RuntimeError("XC-coupled find requires a matching native build with CPKS callback support")
        driver = FindCPKS(core, dimer, {"A": wfn_a, "B": wfn_b}, cache)
        # Bound methods retain the driver and the original live references/JKs.
        # No callback captures fisapt, so there is no Python/native ownership cycle.
        fisapt.set_cpks_response_callbacks(driver.solve, driver.finish)
        return driver
    finally:
        # get_option(module, key) MUTATES the shared current-module context.
        # The SCF LEVEL_SHIFT lookup must not hide FISAPT-local flags from
        # the following native felst/fexch/find calls (or from error cleanup).
        core.prepare_options_for_module("FISAPT")
