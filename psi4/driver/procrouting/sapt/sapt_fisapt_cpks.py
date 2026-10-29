"""Opt-in source-resolved CPU CPKS reference; no F-SAPT driver/default changes.

Native cphf_Hx has the *negative* positive-Hessian convention. Sources are
canonical (source, occupied, virtual); readout uses one shared localized gauge.
No functional is ever evaluated on an isolated source density.
"""
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

import numpy as np

from .sapt_fisapt_cpks_solver import solve_positive


class Mode(str, Enum):
    UNCOUPLED_SCALED = "uncoupled-scaled-compatibility"
    CPHF = "source-resolved-hf-cphf"
    CPHF_TO_CPKS_TOTAL = "hf-kernel-partition-to-cpks-total"
    CPKS = "source-resolved-cpks"


def _array(value, shape, name):
    value = np.asarray(value)
    if value.dtype != np.float64 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite FP64 with shape {shape}")
    return value


def _snapshot(value):
    value = np.array(value, copy=True)
    value.flags.writeable = False
    return value


def _close(actual, expected, name, atol, rtol):
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise RuntimeError(f"Nonfinite {name}")
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise RuntimeError(f"Unnormalized {name} mismatch")


def _require_cpu(core):
    # BrianQC's option is registered only in builds with that backend.
    if ("BRIANQC_ENABLE" in core.get_global_option_list()
            and core.get_global_option("BRIANQC_ENABLE")):
        raise ValueError("CPKS CPU reference requires BrianQC disabled")


def _validation_work_bytes(nao, occupied, virtual):
    """Conservative scratch bound for eager Fock blocks, diagonals and comparisons."""
    if any(type(n) is not int or n < 1 for n in (nao, occupied, virtual)):
        raise ValueError("Positive full reference dimensions required")
    quadratic = max(occupied**2, virtual**2, occupied * virtual)
    return 8 * (8 * quadratic + 4 * nao * (occupied + virtual) + nao**2
                + 32 * occupied * virtual)


class NativeMonomerCPKS:
    """Borrow the original live restricted monomer and initialized canonical JK.

    Caller retains both, must not mutate/finalize them or run concurrent JK
    work, and declares the reference's shifts/external-response state explicitly.
    set_jk follows canonical SAPT; this adapter does not own/finalize that JK.
    """
    def __init__(self, core, wfn, jk, *, grac_shift, orbital_shift,
                 external_response, max_validation_bytes=64 * 1024**2):
        _require_cpu(core)
        if not hasattr(wfn, "has_external_cpscf_terms"):
            raise RuntimeError("CPKS requires a matching build with external-response state introspection")
        if wfn.has_external_cpscf_terms():
            raise NotImplementedError("Stored external potentials/CPSCF perturbations are not admitted")
        if (type(grac_shift) not in (float, int) or grac_shift != 0
                or type(orbital_shift) not in (float, int) or orbital_shift != 0
                or external_response is not False):
            raise NotImplementedError("Shifted or external-perturbation CPKS is not qualified")
        if (wfn.nirrep() != 1 or wfn.nalpha() != wfn.nbeta()
                or not wfn.same_a_b_orbs() or not wfn.same_a_b_dens()):
            raise NotImplementedError("CPKS reference requires restricted closed-shell C1")
        functional = wfn.functional()
        name = functional.name().upper()
        hf = name == "HF"
        if (name not in ("HF", "SVWN", "PBE") or functional.is_x_lrc()
                or functional.is_c_lrc() or functional.is_c_hybrid()
                or functional.is_c_scs_hybrid() or functional.is_meta()
                or functional.needs_vv10()
                or not np.isfinite(functional.x_beta())
                or functional.needs_grac() or functional.grac_shift() != 0
                # HF's builder may retain beta=1 with omega=0. Native
                # twoel_Hx uses beta only under is_x_lrc(), rejected above.
                or (hf and (functional.needs_xc() or functional.x_alpha() != 1))
                or (not hf and (functional.is_x_hybrid()
                                or functional.x_alpha() != 0
                                or functional.x_beta() != 0))):
            raise NotImplementedError("Only HF limit and unshifted SVWN/PBE CPKS are admitted")
        co = np.asarray(wfn.Ca_subset("AO", "OCC").np)
        cv = np.asarray(wfn.Ca_subset("AO", "VIR").np)
        eo = np.asarray(wfn.epsilon_a_subset("AO", "OCC").np)
        ev = np.asarray(wfn.epsilon_a_subset("AO", "VIR").np)
        if co.ndim != 2 or cv.ndim != 2 or co.shape[0] != cv.shape[0]:
            raise ValueError("Canonical full monomer orbital spaces required")
        o, v = co.shape[1], cv.shape[1]
        if (type(max_validation_bytes) is not int or max_validation_bytes < 1
                or _validation_work_bytes(co.shape[0], o, v) > max_validation_bytes):
            raise MemoryError("CPKS Fock/reference validation exceeds the explicit scratch budget")
        for value, shape, label in ((co, co.shape, "occupied"), (cv, cv.shape, "virtual"),
                                    (eo, (o,), "occupied energies"), (ev, (v,), "virtual energies")):
            _array(value, shape, label)
        if min(o, v) < 1:
            raise ValueError("Nonempty full occupied and virtual spaces required")
        self.gap = _snapshot(ev[None, :] - eo[:, None])
        _array(self.gap, (o, v), "canonical gap")
        if np.any(self.gap <= 0):
            raise ValueError("Positive canonical gaps required")
        fock = _array(wfn.Fa().np, (co.shape[0], co.shape[0]), "Fock")
        # Native onel_Hx uses Fock, not the supplied eigenvalue preconditioner.
        for actual, expected in ((co.T @ fock @ co, np.diag(eo)),
                                 (cv.T @ fock @ cv, np.diag(ev)),
                                 (co.T @ fock @ cv, np.zeros((o, v)))):
            _close(actual, expected, "Fock/gap reference", 1e-8, 1e-8)
        if jk is None:
            raise ValueError("Live initialized canonical CPU response JK required")
        self.core, self.wfn, self.jk = core, wfn, jk
        self.mode = Mode.CPHF if hf else Mode.CPKS
        self.metadata = MappingProxyType({
            "mode": self.mode.value, "functional": name,
            "reference": "HF" if hf else "KS", "response_kernel": "HF" if hf else "KS-XC",
            "hessian": "-original-monomer.cphf_Hx",
            "xc_reference": "whole original monomer; stored potential/grid",
            "response_exchange": "native functional coefficients, applied once",
            "exchange_alpha": functional.x_alpha(), "exchange_beta": functional.x_beta(),
            "response_exchange_beta_effective": 0.,
            "grac_shift": grac_shift, "orbital_shift": orbital_shift,
        })
        wfn.set_jk(jk)

    def apply(self, trial):
        trial = _array(trial, self.gap.shape, "canonical response trial")
        products = self.wfn.cphf_Hx([self.core.Matrix.from_array(trial)])
        if len(products) != 1:
            raise RuntimeError("Native CPKS returned wrong vector count")
        return -_array(products[0].np, self.gap.shape, "native Hessian action").copy()


@dataclass(frozen=True)
class SourceResponse:
    """Unscaled snapshots, [localized responding occupied, physical source]."""
    mode: Mode
    amplitudes: np.ndarray
    induction: np.ndarray
    exchange_induction: np.ndarray
    diagnostics: tuple
    aggregate_diagnostic: object
    metadata: object


def source_resolved(operator, sources, rotation, target, target_exchange, *,
                    total_source, canonical_induction, canonical_exchange_induction,
                    rtol=1e-10, atol=1e-12, max_iterations=128,
                    check_atol=2e-9, check_rtol=1e-7,
                    max_response_bytes=64 * 1024**2):
    """Solve/validate a single direction BEFORE allowing any normalization.

    operator provides positive ``gap``, ``apply`` and an explicitly labeled
    coupled ``mode``. Targets are already in the one shared localized gauge;
    target_exchange is bare SAPT S^2 readout, not functional-weighted response K.
    Component canonical totals and complete canonical RHS are mandatory.
    """
    if operator.mode not in (Mode.CPHF, Mode.CPKS):
        raise ValueError("Explicit coupled CPHF/CPKS operator required")
    gap = np.asarray(operator.gap)
    if gap.ndim != 2 or not gap.size:
        raise ValueError("Full occupied-virtual gaps required")
    _array(gap, gap.shape, "gap")
    if np.any(gap <= 0):
        raise ValueError("Positive gaps required")
    o, v = gap.shape
    sources = np.asarray(sources)
    if sources.ndim != 3 or sources.shape[0] < 1:
        raise ValueError("At least one physical source required")
    _array(sources, (sources.shape[0], o, v), "sources")
    if (type(max_response_bytes) is not int or max_response_bytes < 1
            or 4 * sources.nbytes + 8 * (32 * o * v + 6 * o * sources.shape[0] + o * o)
            > max_response_bytes):
        raise MemoryError("CPKS response arrays exceed explicit byte budget")
    rotation = _array(rotation, (o, o), "shared occupied rotation")
    target = _array(target, (o, v), "localized induction target")
    target_exchange = _array(target_exchange, (o, v), "localized exchange target")
    total_source = _array(total_source, (o, v), "canonical total RHS")
    if (any(type(t) not in (float, int) or not np.isfinite(t) or t < 0
            for t in (check_atol, check_rtol)) or max(check_atol, check_rtol) == 0):
        raise ValueError("Finite positive validation policy required")
    _close(rotation.T @ rotation, np.eye(o), "occupied gauge", 1e-10, 1e-10)
    _close(sources.sum(axis=0), total_source, "source completeness", check_atol, check_rtol)
    # Prevent an incorrect supplied target or gauge from passing total-only tests.
    _close(rotation @ target, total_source, "induction target/gauge", check_atol, check_rtol)
    expected = _array([canonical_induction, canonical_exchange_induction],
                      (2,), "separate canonical component totals")

    def solve(rhs):
        solved = solve_positive(operator.apply, gap, rhs, rtol=rtol, atol=atol,
                                max_iterations=max_iterations)
        # Includes zero RHS: verify H(0)=0 rather than hiding an affine action.
        residual = _array(operator.apply(solved.amplitudes), gap.shape, "H kappa") + rhs
        absolute = float(np.linalg.norm(residual))
        norm = float(np.linalg.norm(rhs))
        if absolute > max(atol, rtol * norm):
            raise RuntimeError("CPKS true physical residual exceeds tolerance")
        return solved.amplitudes, MappingProxyType({
            "iterations": solved.iterations, "absolute_residual": absolute,
            "relative_residual": absolute / norm if norm else absolute,
        })

    amplitudes = np.empty_like(sources)
    diagnostics = []
    for column, rhs in enumerate(sources):
        amplitudes[column], diagnostic = solve(rhs)
        diagnostics.append(diagnostic)
    aggregate, aggregate_diagnostic = solve(total_source)
    _close(amplitudes.sum(axis=0), aggregate, "source amplitude additivity",
           check_atol, check_rtol)
    localized = np.einsum("il,siv->slv", rotation, amplitudes)
    induction = 2 * np.einsum("slv,lv->ls", localized, target)
    exchange = 2 * np.einsum("slv,lv->ls", localized, target_exchange)
    _close(np.array([induction.sum(), exchange.sum()]), expected,
           "separate canonical Ind20,r / Exch-Ind20,r", check_atol, check_rtol)
    metadata = dict(getattr(operator, "metadata", {}))
    metadata.update(mode=operator.mode.value, response_scale=1.0, delta_hf_scale=1.0,
                    normalized=False, amplitude_equation="H_positive kappa = -W",
                    layout="localized responding occupied, source",
                    readout_exchange="bare SAPT S^2",
                    source_additivity_max_abs=float(np.max(np.abs(amplitudes.sum(axis=0) - aggregate))))
    return SourceResponse(operator.mode, _snapshot(amplitudes), _snapshot(induction),
                          _snapshot(exchange), tuple(diagnostics), aggregate_diagnostic,
                          MappingProxyType(metadata))


def build_sources(core, jk, occupied, virtual, source_localized, nuclei):
    """Small CPU source builder using a borrowed bare-SAPT CPU JK.

    ``nuclei`` is an ordered iterable of (Z, (x,y,z) in bohr). Each electronic
    column is 2J[L_s L_s.T]. Returns canonical RHS columns, nuclei first.
    Kernel/potential on the responding monomer is never changed. Source labels
    and their physical placement are owned by the caller.
    """
    _require_cpu(core)
    co, cv, lo = (np.asarray(x) for x in (occupied, virtual, source_localized))
    if (any(x.ndim != 2 for x in (co, cv, lo)) or co.shape[0] != cv.shape[0]
            or lo.shape[0] != co.shape[0] or min(co.shape + cv.shape + lo.shape) < 1):
        raise ValueError("Common AO and full orbital spaces required")
    for name, value in (("occupied", co), ("virtual", cv), ("source localized", lo)):
        _array(value, value.shape, name)
    columns = []
    for charge, xyz in nuclei:
        _array(np.array([charge, *xyz], dtype=np.float64), (4,), "nucleus")
        potential = core.ExternalPotential()
        potential.addCharge(float(charge), *map(float, xyz))
        ao = potential.computePotentialMatrix(jk.basisset())
        columns.append(co.T @ np.asarray(ao.np) @ cv)
    for orbital in lo.T:
        jk.C_clear()
        matrix = core.Matrix.from_array(orbital[:, None])
        jk.C_left_add(matrix)
        jk.C_right_add(matrix)
        jk.compute()
        columns.append(2 * co.T @ np.asarray(jk.J()[0].np) @ cv)
    jk.C_clear()
    return np.stack(columns)


def export_direction(induction, exchange_induction, *, mode, direction,
                     reference=None, canonical_combined=None, delta_hf_scale=1.0,
                     ratio_atol=1e-12, ratio_rtol=1e-10):
    """Explicit export; preserve raw matrices and record every scale.

    Approximate modes require an explicit HF/KS reference label. HF-like kernel
    on KS is never called true HF CPHF; fitting totals cannot change reference.
    A coupled mode rejects even a redundant request to fit its response total.
    Caller must use the same global delta_hf_scale for both directional exports.
    """
    mode = Mode(mode)
    if mode in (Mode.CPKS, Mode.CPHF):
        implied = "KS" if mode == Mode.CPKS else "HF"
        if reference not in (None, implied):
            raise ValueError("Reference label contradicts coupled scientific mode")
        reference = implied
    elif reference not in ("HF", "KS"):
        raise ValueError("Approximate partition requires explicit HF or KS reference")
    if direction not in ("A<-B", "B<-A"):
        raise ValueError("Explicit physical direction required")
    induction, exchange_induction = map(np.asarray, (induction, exchange_induction))
    if induction.ndim != 2 or not induction.size:
        raise ValueError("Nonempty directional matrices required")
    for name, value in (("induction", induction), ("exchange", exchange_induction)):
        _array(value, induction.shape, name)
    if (type(delta_hf_scale) not in (int, float) or not np.isfinite(delta_hf_scale)
            or any(type(t) not in (int, float) or not np.isfinite(t) or t < 0
                   for t in (ratio_atol, ratio_rtol)) or max(ratio_atol, ratio_rtol) == 0):
        raise ValueError("Finite explicit scaling policy required")
    response_scale = 1.0
    with np.errstate(over="raise", invalid="raise"):
        raw_total = float(induction.sum() + exchange_induction.sum())
        magnitude = float(np.abs(induction).sum() + np.abs(exchange_induction).sum())
    if not np.isfinite([raw_total, magnitude]).all():
        raise FloatingPointError("Nonfinite raw directional response")
    if mode in (Mode.CPKS, Mode.CPHF):
        if canonical_combined is not None:
            raise ValueError("Coupled response must not receive a legacy response ratio")
    else:
        if (type(canonical_combined) not in (int, float)
                or not np.isfinite(canonical_combined)):
            raise ValueError("Canonical directional combined target required")
        if abs(raw_total) <= max(ratio_atol, ratio_rtol * magnitude):
            raise ValueError("Undefined/cancellation-dominated directional response ratio")
        response_scale = canonical_combined / raw_total
        if not np.isfinite(response_scale):
            raise FloatingPointError("Nonfinite directional response ratio")
    with np.errstate(over="raise", invalid="raise"):
        combined = (induction + exchange_induction) * response_scale * delta_hf_scale
    if not np.isfinite(combined).all():
        raise FloatingPointError("Nonfinite exported response")
    return MappingProxyType({
        "raw_induction": _snapshot(induction),
        "raw_exchange_induction": _snapshot(exchange_induction),
        "combined": _snapshot(combined),
        "metadata": MappingProxyType({
            "mode": mode.value, "direction": direction, "raw_combined_total": raw_total,
            "reference": reference,
            "response_kernel": ("none" if mode == Mode.UNCOUPLED_SCALED else
                                "KS-XC" if mode == Mode.CPKS else
                                "HF" if reference == "HF" else "HF-like on fixed KS"),
            "normalization_target": ("canonical CPKS combined" if mode == Mode.CPHF_TO_CPKS_TOTAL
                                     else "canonical coupled combined" if mode == Mode.UNCOUPLED_SCALED
                                     else "none; raw coupled components validated"),
            "response_scale": response_scale, "delta_hf_scale": delta_hf_scale,
            "delta_hf_redistribution": "not applied" if delta_hf_scale == 1 else "global scalar approximation",
        }),
    })
