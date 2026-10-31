# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Explicit experimental CHNO monomer declarations and native execution.

No SCF is hidden here: the caller supplies a genuinely converged live PBE0
wavefunction. A separate versioned declaration owns frames and bonds; no
chemical-name lookup, element-wide COPY, external result or reference reader
is available. Installed/runtime and OS-trace qualification remain the caller's
responsibility; this module does not turn source tests into accepted runs.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from psi4 import core

from .isapol_bounded import BoundedResources, refinement_limits, _pfit_plan
from .isapol_native import Quadrature
from .isapol_native_correction import require_scf_seal, validate_correction
from .isapol_native_propagator import KernelSmoothing
from .isapol_pfit_stream import packed_design_plan
from .isapol_recipes import native_monomer_recipe
from .isapol_refine import COMPONENT_NAMES, RefinementSite

PROFILE = 's66-chno-isa-a-v1'
DECLARATION_SCHEMA = 'native-monomer-declaration/1'


@dataclass(frozen=True)
class MolecularDeclaration:
    """Versioned JSON input; every atom's proper frame and bonds are explicit."""
    profile: str
    frames: tuple
    bonds: tuple

    def __post_init__(self):
        if self.profile != PROFILE:
            raise ValueError('explicit supported monomer profile required')
        if not 3 <= len(self.frames) <= 17:
            raise ValueError('molecular declaration requires3..17 frames')
        # Use the same proper local-to-global frame validation as the engine.
        frames = tuple(RefinementSite('validation', 'validation', (0., 0., 0.), frame, 2).frame
                       for frame in self.frames)
        bonds, seen = [], set()
        for edge in self.bonds:
            if (not isinstance(edge, (tuple, list)) or len(edge) != 2
                    or any(type(i) is not int or not 0 <= i < len(frames) for i in edge)
                    or edge[0] == edge[1] or tuple(sorted(edge)) in seen):
                raise ValueError('invalid molecular bond declaration')
            seen.add(tuple(sorted(edge)))
            bonds.append(tuple(edge))
        object.__setattr__(self, 'frames', frames)
        object.__setattr__(self, 'bonds', tuple(bonds))

    @classmethod
    def from_dict(cls, value):
        if (type(value) is not dict or set(value) != {'schema', 'profile', 'frames', 'bonds'}
                or value['schema'] != DECLARATION_SCHEMA):
            raise ValueError('exact versioned molecular declaration required')
        return cls(value['profile'], tuple(value['frames']), tuple(value['bonds']))

    def as_dict(self):
        return dict(schema=DECLARATION_SCHEMA, profile=self.profile,
                    frames=[[list(row) for row in frame] for frame in self.frames],
                    bonds=[list(edge) for edge in self.bonds])


def independent_sites(molecule, declaration):
    """Every atom has its own type: shared element is never COPY equivalence."""
    if not isinstance(declaration, MolecularDeclaration):
        raise TypeError('explicit MolecularDeclaration required')
    if molecule.natom() != len(declaration.frames):
        raise ValueError('frames and every molecular atom must match')
    counts, sites = {}, []
    xyz = molecule.geometry().np
    for index in range(molecule.natom()):
        symbol = molecule.symbol(index)
        if symbol not in ('C', 'H', 'N', 'O') or molecule.Z(index) <= 0:
            raise ValueError('profile permits real CHNO atoms only')
        counts[symbol] = counts.get(symbol, 0)+1
        label = f'{symbol}{counts[symbol]}'
        sites.append(RefinementSite(label, label, xyz[index], declaration.frames[index], 2))
    return tuple(sites)


def full_rank2_variables(sites):
    return tuple(f'{site.label}_{COMPONENT_NAMES[i]}_{COMPONENT_NAMES[j]}_A'
                 for site in sites for i in range(1, 9) for j in range(i, 9))


def plan_molecular_refinement(declaration, resources, *, max_replay_work):
    """Cheap refinement-only preflight; NOT a complete SCF/ISA/response plan."""
    if not isinstance(declaration, MolecularDeclaration):
        raise TypeError('explicit MolecularDeclaration required')
    count = 36*len(declaration.frames)
    work = refinement_limits(count, resources, capacity='molecular',
                             max_replay_work=max_replay_work)
    plan = packed_design_plan(
        npoint=2000, channel_count=9*len(declaration.frames), parameter_count=count,
        used_channels=8*len(declaration.frames), block_rows=4096,
        max_bytes=resources.max_bytes, max_work=work)
    fit_bytes, fit_work = _pfit_plan(2000, 9*len(declaration.frames), count, 11)
    if fit_work > resources.max_work:
        raise ValueError('complete refinement alone exceeds cumulative work resource limit')
    if fit_bytes > resources.max_bytes:
        raise ValueError('complete refinement alone exceeds numeric byte resource limit')
    return dict(scope='refinement_only_not_complete_molecular_admission',
                parameters=count, sites=len(declaration.frames), nodes=11,
                planned_work_scope='packed_replay_only',
                complete_refinement_work=fit_work, complete_refinement_bytes=fit_bytes, **plan)


def run_molecular_properties(wfn, declaration, resources, *, max_replay_work,
                             scratch_directory, log=None):
    """Compute from the caller's live state through the ordinary oeprop route."""
    from psi4.driver.p4util.util import oeprop, atomic_property_result
    from .isapol_bounded_oeprop import invalidate
    if not isinstance(wfn, core.Wavefunction):
        raise TypeError('live Psi4 Wavefunction required')
    invalidate(wfn)
    plan_molecular_refinement(declaration, resources, max_replay_work=max_replay_work)
    require_scf_seal(wfn)
    validate_correction(wfn, scf_correction='NONE', require_canonical=True)
    if resources.max_bytes > int(core.get_memory()):
        raise ValueError('molecular numeric budget exceeds the Psi4 memory setting')
    basis = wfn.basisset()
    if basis.name().upper() != 'AUG-CC-PVTZ' or not basis.has_puream():
        raise ValueError('profile requires actual spherical aug-cc-pVTZ MAIN basis')
    molecule = wfn.molecule()
    sites = independent_sites(molecule, declaration)
    recipe = native_monomer_recipe(molecule, profile=declaration.profile)
    options = core.IsaGridOptions()
    options.radial_points, options.spherical_points = 200, 974
    grid = core.IsaGrid(molecule.clone(), options)
    integration = np.column_stack((grid.x(), grid.y(), grid.z(), grid.w()))
    del grid
    lattice = core.FitPointsOptions()
    lattice.npoints, lattice.seed, lattice.lolim, lattice.hilim = 2000, 1, 2., 4.
    quadrature = Quadrature.from_casimir(core.CasimirGrid(10, .5))
    oeprop(wfn, 'ATOMIC_REFINED_POLARIZABILITIES', 'ATOMIC_REFINED_DISPERSION',
        atomic_backend='BOUNDED_DF', preset='explicit', sites=sites, bonds=declaration.bonds,
        distribution='isa', partition_recipe=recipe, partition_grid=integration,
        auxiliary_recipe=recipe.auxiliary, quadrature=quadrature,
        radial_points=100, spherical_points=200,
        smoothing=KernelSmoothing(1e-8, 600., .01, 1., 'FD'),
        shell_cutoff=1e-8, charge_penalty=1000., anchor_metric_damping=.0005,
        lattice_options=lattice, rank_limit=2, hydrogen_rank_limit=2, localization_rank_limit=2,
        weight_type=3, weight_coefficient=.001, cutoff=1e-4,
        declared_variables=full_rank2_variables(sites), max_refinement_parameters=36*len(sites),
        refinement_capacity='molecular', max_replay_work=max_replay_work,
        max_order=10, scf_correction='NONE', recoupled_dispersion=False,
        resources=resources, scratch_directory=scratch_directory, log=log)
    return atomic_property_result(wfn)


def export_molecular_result(wfn, result, declaration, *, output):
    """Export numeric tensors/scalars; execution is not an accuracy guarantee."""
    from .isapol_acceptance import _plain, _isotropic_export, _write_tensor_artifact
    from .isapol_bounded_oeprop import atomic_property_result
    if atomic_property_result(wfn) is not result:
        raise RuntimeError('molecular result is not owned by the supplied wavefunction')
    sites = independent_sites(wfn.molecule(), declaration)
    variables = full_rank2_variables(sites)
    nf, ns = len(result.frequencies), len(sites)
    expected_entries = tuple(((s, i, j),) for s in range(ns)
                             for i in range(1, 9) for j in range(i, 9))
    if (nf != 11 or len(result.refinements) != nf or len(result.diagnostics) != nf
            or len(result.local_tensors) != nf):
        raise RuntimeError('incomplete molecular frequency sweep')
    for node, refined in enumerate(result.refinements):
        model = refined.model
        if (refined.status != core.IsaPfitStatus.Solved or model.sites != sites
                or model.parameter_labels != variables or model.declared_variables != variables
                or model.parameter_entries != expected_entries
                or model.reference_sites != tuple(range(ns))
                or model.equivalent_sites != tuple((s,) for s in range(ns))
                or model.penalty_selection != 'static_reference'
                or model.frequency_au != result.frequencies[node]):
            raise RuntimeError('molecular independent-site/full-variable fit inventory mismatch')
    if any(not np.isfinite(d['response_residual']) or d['response_residual'] >= 1e-10
           or not np.isfinite(d['localization_residual']) or d['localization_residual'] >= 1e-6
           for d in result.diagnostics):
        raise RuntimeError('molecular response/localization residual acceptance failed')
    pairs = _isotropic_export(result.dispersion, max_order=10)
    keys = {(p['site_a'], p['site_b'], p['order']) for p in pairs}
    expected_keys = {(a.label, b.label, n) for a in sites for b in sites for n in (6, 8, 10)}
    if keys != expected_keys or len(pairs) != len(expected_keys):
        raise RuntimeError('incomplete molecular isotropic coverage')
    for pair in pairs:
        if not np.isfinite(pair['value']) or pair['unrestricted_complete'] != (pair['order'] < 10):
            raise RuntimeError('invalid molecular isotropic coefficient/coverage')
        if pair['order'] == 10 and pair['missing_rank_pairs'] != ((1, 3), (3, 1)):
            raise RuntimeError('partial C10 coverage must remain explicit')
    # Positivity is reported rather than silently repaired. A completed
    # native calculation with nonpositive scalar terms cannot use the present
    # positive-C8/C6 BJ radius workflow without an explicit further decision.
    ineligible = [p for p in pairs if p['order'] in (6, 8) and p['value'] <= 0]
    require_scf_seal(wfn)
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    layout = _write_tensor_artifact(directory/'tensors.npz', result, localization_rank_limit=2)
    if layout['local_shape'] != (11, ns, 15, 15) or layout['refined_shape'] != (11, ns, 9, 9):
        raise RuntimeError('incorrect molecular tensor layout')
    record = dict(
        schema='native-monomer-properties/1', status='complete',
        method='PBE0/aug-cc-pVTZ', declaration=declaration.as_dict(),
        energy_Eh=float(wfn.energy()), frequencies=result.frequencies,
        cp_weights=result.dispersion.cp_weights, sites=sites, isotropic_pairs=pairs,
        independent_parameters=len(variables), tensor_layout=layout,
        tensors_sha256=hashlib.sha256((directory/'tensors.npz').read_bytes()).hexdigest(),
        diagnostics=result.diagnostics, resources=result.resources, provenance=result.provenance,
        bj_positive_eligible=not ineligible, nonpositive_C6_C8_pairs=ineligible,
        pfit_native_verified=[r.diagnostics.native_verified for r in result.refinements],
        qualification='Experimental uniform-tail, independent-site rank2 profile; C10 partial(2,2), no convergence/parity/accuracy or independent solver certification claim.')
    (directory/'result.json').write_text(json.dumps(_plain(record), indent=2, allow_nan=False))
    return _plain(record)
