# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""PFIT local tensors → native local-axis Cn(t,u,J), without model repair.

Real Racah/no-CS components and local-to-global frame columns are already the
core convention. Only the identically zero 00 row/column is removed; no trace,
rotation, normalization, symmetrization or external-model parser enters here.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np
from psi4 import core

from .isapol_bounded_response import _Ledger
from .isapol_refine import COMPONENT_NAMES, RefinementResult, _validated_sites
from .isapol_lw import Provenance


@dataclass(frozen=True)
class DeclaredPairCoverage:
    site_a: int
    site_b: int
    order: int
    declared_model_complete: bool
    missing_declared_rank_quadruples: tuple


@dataclass(frozen=True)
class NativeRecoupledDispersion:
    """Owned native result, with a separate declared-model coverage inventory.

    ``pairs`` retains ALL core coefficients and coverage, including odd orders,
    zero records, missing_table_rank_quadruples and missing_unrestricted ranks.
    ``table_complete`` is NOT a synonym for declared-model completeness.
    t/u are one-based real angular-component indices (00,10,11c,11s,...);
    J is angular momentum. These are NOT oriented scalar Cn or -Cn/R**n.
    Frequencies, origins, tensors, coefficients and CP weights are atomic units;
    weights already include 1/(2*pi), and static nodes have zero weight.
    """
    native_result: object
    pair_labels: tuple
    declared_model_coverage: tuple
    quadrature_provenance: Provenance
    provenance: str
    charge_policy: str = '00 row/column exactly zero; omitted, not rank 1'
    convention: str = 'local-axis real Racah/no-CS Cn(t,u,J); no dimer placement or energy'

    @property
    def pairs(self):
        return self.native_result.pairs

    @property
    def model(self):
        return self.native_result.model_a

    @property
    def frequencies(self):
        return self.native_result.frequencies

    @property
    def cp_weights(self):
        return self.native_result.cp_weights

    @property
    def max_order(self):
        return self.native_result.max_order


def recoupled_plan(sites, nfrequency, max_order):
    """Preflight existing core ceilings using its actual tables, not new policy.

    Inventory/size arithmetic only: no recoupling mathematics is duplicated.
    The conservative numeric plan includes snapshot/copy overlap and the two
    model copies in the native result. Python/container overhead is not a
    numeric-buffer ledger promise.
    """
    sites = _validated_sites(sites)
    if any(s.rank_limit < 1 for s in sites):
        raise ValueError('recoupled dispersion requires positive per-site ranks; rank-zero-only sites unsupported')
    if type(max_order) is not int or not 6 <= max_order <= 12:
        raise ValueError('recoupled max_order must be an integer in 6..12')
    if type(nfrequency) is not int or not 1 <= nfrequency <= 65536:
        raise ValueError('recoupled frequency resource limit')
    ranks = [tuple(range(1, s.rank_limit+1)) for s in sites]
    tensor_elements = nfrequency*sum((s.component_count-1)**2 for s in sites)
    if tensor_elements > 8_000_000 or len(sites)*nfrequency > 65536:
        raise ValueError('recoupled local model resource limit')
    elements = recouple_work = 0
    for rr in ranks:
        for l, p in product(rr, repeat=2):
            if core.isapol_realcg_defined(l, p):
                elements += nfrequency*((l+p+1)**2-(l-p)**2)
                recouple_work += nfrequency*len(core.isapol_realcg_terms(l, p))
    pairs, records, work = len(sites)**2, 0, 0
    blocks = core.isapol_recoupling_blocks()
    for a, b in product(ranks, repeat=2):
        for n, L1, L2, J, terms in blocks:
            if n > max_order:
                continue
            count = sum(t.la in a and t.lap in a and t.lb in b and t.lbp in b
                        and core.isapol_realcg_defined(t.la, t.lap)
                        and core.isapol_realcg_defined(t.lb, t.lbp) for t in terms)
            if count:
                size = (2*L1+1)*(2*L2+1)
                records += size
                work += size*count*nfrequency
    if (pairs > 4096 or records > 2_000_000 or work > 100_000_000
            or elements > 8_000_000 or recouple_work > 100_000_000):
        raise ValueError('recoupled core resource budget exceeded')
    # Three coverage lists per core pair/order, plus declared-model inventory,
    # conservatively bounded by all positive-rank quadruples through C12.
    inventory = pairs*sum((n-3)*(n-4)*(n-5)//6 for n in range(6, max_order+1))
    numeric = (8*8*tensor_elements + 16*5*elements + 96*records
               + 128*inventory + 8*1024**2)
    return numeric, recouple_work+work


def refined_recoupled_dispersion(results, *, cp_weights, quadrature_provenance,
                                 max_order, ledger):
    """Single native conversion/contraction seam, reusable outside oeprop.

    Only solved RefinementResults with invariant site/channel declarations are
    accepted. Exact reciprocal FULL tensors, including mixed-rank entries, are
    checked before any native recoupling/contraction. Nonzero charge response
    cannot be represented by this positive-rank core and is refused honestly.
    """
    if not isinstance(ledger, _Ledger):
        raise TypeError('shared bounded ledger required')
    results = tuple(results)
    if not results or any(not isinstance(r, RefinementResult) for r in results):
        raise TypeError('nonempty RefinementResult sequence required')
    sites = _validated_sites(results[0].model.sites)
    numeric, work = recoupled_plan(sites, len(results), max_order)
    frequencies = tuple(r.model.frequency_au for r in results)
    weights = np.asarray(cp_weights, dtype=float)
    f = np.asarray(frequencies, dtype=float)
    if (not np.isfinite(f).all() or (f < 0).any() or (np.diff(f) <= 0).any()):
        raise ValueError('frequencies must be finite, nonnegative and strictly increasing')
    if (weights.shape != f.shape or not np.isfinite(weights).all()
            or (weights[f == 0] != 0).any() or (weights[f > 0] <= 0).any()
            or not (weights > 0).any()):
        raise ValueError('CP weights: static zero, dynamic positive finite, matching frequency shape required')
    if not isinstance(quadrature_provenance, Provenance):
        raise ValueError('quadrature provenance required')
    channels = tuple(f'{s.label}_{c}' for s in sites for c in COMPONENT_NAMES[:s.component_count])
    for r in results:
        if r.status != core.IsaPfitStatus.Solved:
            raise ValueError('recoupled dispersion requires solved PFIT')
        if (r.model.sites != sites or tuple(r.model.channel_labels) != channels
                or r.model.parameter_labels != results[0].model.parameter_labels
                or r.model.parameter_entries != results[0].model.parameter_entries):
            raise ValueError('identical site/rank/frame/component ordering required at every node')
        if len(r.refined_tensors) != len(sites):
            raise ValueError('one refined tensor per site required')
        for s, tensor in zip(sites, r.refined_tensors):
            a = np.asarray(tensor)
            if (a.shape != (s.component_count, s.component_count)
                    or a.dtype != np.float64 or not np.isfinite(a).all()):
                raise ValueError('finite float64 full refined tensor shape/ranks required')
            if not np.array_equal(a, a.T):
                raise ValueError('reciprocity requires exactly symmetric refined tensors; no repair')
            if np.any(a[0, :] != 0) or np.any(a[:, 0] != 0):
                raise ValueError('nonzero rank-zero charge response unsupported by recoupled core')
    ledger.admit('native recoupled dispersion', numeric, work)
    native_sites = []
    for i, s in enumerate(sites):
        site = core.IsaAnisotropicSite()
        site.label, site.origin, site.frame = s.label, s.origin_bohr, s.frame
        site.ranks = list(range(1, s.rank_limit+1))
        site.responses = [core.Matrix.from_array(np.asarray(r.refined_tensors[i])[1:, 1:])
                          for r in results]
        native_sites.append(site)
    provenance = ('native PFIT-refined full local tensors; '
                  + results[0].model.provenance + '; ' + repr(quadrature_provenance))
    # "supplied_local_response" is the core's explicit local-tensor declaration,
    # NOT a claim that a supplied-file bridge produced these native tensors.
    local = core.IsaAnisotropicModel(frequencies, native_sites, 'supplied_local_response', provenance)
    model = core.IsaRecoupledModel(local)
    native = core.isa_recoupled_dispersion(model, model, weights.tolist(), max_order)
    coverage = []
    labels = []
    for pair in native.pairs:
        a, b = sites[pair.site_a], sites[pair.site_b]
        labels.append((a.label, b.label))
        for cov in pair.coverage:
            included = {tuple(q) for q in cov.included_rank_quadruples}
            declared = product(range(1, a.rank_limit+1), range(1, a.rank_limit+1),
                               range(1, b.rank_limit+1), range(1, b.rank_limit+1))
            missing = tuple(q for q in declared if sum(q) == cov.order-2 and q not in included)
            coverage.append(DeclaredPairCoverage(pair.site_a, pair.site_b, cov.order,
                                                 not missing, missing))
    return NativeRecoupledDispersion(native, tuple(labels), tuple(coverage),
                                     quadrature_provenance, provenance)
