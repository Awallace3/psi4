# Psi4: Copyright (c) 2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""What CamCASP's two modern ISA-Pol presets declare, and what this branch honours.

The reference distribution ships two ready-made ISA-Pol protocols,
``methods/isa-pol-from-isa-A`` and ``methods/isa-pol-from-isa-A+DF`` with their
cluster templates. They are the closest thing the reference has to "the modern
recommended way to get distributed polarizabilities", and this module is the
record of every declaration in them: which Psi4 option carries it, which ones
this branch fixes internally, and which ones it cannot honour at all.

It is a MANIFEST, not a driver. Nothing here runs a calculation, reads a file or
touches ``core``; the declarations are transcribed verbatim from the two method
files at the pinned revision so that a later reader can diff the text rather
than trust this summary. ``options()`` returns a plain dict that a caller may
hand to ``psi4.set_options``, and it refuses to return one until the caller has
named every declaration the branch drops on the way.

Two rules govern the whole module.

First, **the name is the model.** A preset is not a set of tolerances that may be
loosened until it runs. Every line in those files names a model, so a run that
drops one of them is a different model, and its numbers may never be quoted as
agreeing with numbers produced by the reference under the full preset. That is
why ``options()`` takes ``accept_reduction`` and why the thing it returns is
labelled by :attr:`PresetManifest.model_name` rather than by the preset's name.

Second, and specifically, ``plan.md:377``: *ISA-A and ISA-A+DF templates are
different protocols. Do not silently switch the target to one of them.* The two
files differ by exactly one hunk -- ``ISA-Algorithm A`` becomes
``ISA-Algorithm A+DF  Zeta = 0.1`` plus ``DF-PARAMETERS Lambda = 1000.0`` -- and
that hunk is the A+DF preset's whole reason to exist. This branch implements the
A functional only. So :data:`ISA_A_PLUS_DF` records the A+DF preset in full and
its ``options()`` **refuses**: emitting the A-only subset of an A+DF request
would be exactly the silent substitution that line forbids, and it would be
worse than a refusal because the result would look like a successful run.

The declaration text reproduced below is CamCASP input protocol, not CamCASP
code or results. See ``isapol_modern_preset.NOTICE`` for its provenance and the
retained MIT notice.
"""
from dataclasses import dataclass

#: The inspected reference tree. Every ``text`` below is verbatim from these
#: files at this revision, and :data:`SOURCE_SHA256` pins them so a test can
#: detect that the transcription has gone stale rather than silently drift.
CAMCASP_REVISION = '63b16a22b9bae597fe81ecdb8b8d91c21868c814'
CAMCASP_REVISION_SHORT = '63b16a2'
SOURCE_SHA256 = (
    ('methods/isa-pol-from-isa-A',
     '464b738145db3f03bc544fbcdc6558a5a4f1ce15d77a0203582112eb3c24de4a'),
    ('methods/isa-pol-from-isa-A.clt_tmpl',
     '7b89fba240de586eec8d1bebe22bbfd5f917b566abd99f263679b750092403f6'),
    ('methods/isa-pol-from-isa-A+DF',
     'dae14e0df349445eacd1ec0084e1429122b41271051205b824cd29b196cc6c22'),
    ('methods/isa-pol-from-isa-A+DF.clt_tmpl',
     '980608f847c9125b584e9c5c3e64fce037f328f1b4b6748865959cb6f454caa4'),
)

#: How a declaration reaches (or fails to reach) a Psi4 run.
#:
#: ``option``      carried by a named Psi4 option, which ``options()`` emits.
#: ``fixed``       honoured, but built into the branch rather than declarable.
#:                 The value is not a default that a caller may change; there is
#:                 no option for it, which is why the ``note`` must say where it
#:                 lives.
#: ``caller``      not the preset's to supply -- geometry, orbital basis,
#:                 functional, SCF code. The caller's wavefunction is the
#:                 declaration, so ``options()`` must not invent one.
#: ``report``      selects no number: printing, file naming, output requests.
#: ``inert``       a real keyword that provably cannot change any number *under
#:                 the rest of this same preset*. Not the same as ``fixed``: the
#:                 branch has no counterpart at all, and the ``note`` carries the
#:                 proof from the reference source that none is needed.
#: ``unhonoured``  the branch cannot do this. Requires a key in :data:`GAPS`.
ROUTES = ('option', 'fixed', 'caller', 'report', 'inert', 'unhonoured')

#: Every declaration this branch cannot honour, keyed by a stable name. The text
#: says what the reference asks for, what happens instead, and what closing it
#: would take -- so that "we do not do this" is never recorded as "this does not
#: matter". Nothing in here is a tolerance; each one changes the model.
GAPS = {
    'overlap_neighbour_rule':
        'The preset opens with an overlap-based neighbour list, OVR with EPS = 0.01, which '
        'is what decides which site pairs are treated as neighbours downstream. This branch '
        'has no overlap neighbour rule: its bond/frame input is supplied by the caller and '
        'the response is formed for every ordered site pair. Closing it means an actual '
        'overlap criterion over the converged shapes, not a distance cutoff standing in for '
        'one.',
    'lebedev_angular_grid':
        'The preset declares Angular 400 on the shared integration grid, deliberately -- '
        '"We need larger grids for the polarizability integration". Psi4 offers Lebedev '
        'orders, and 400 is not one of them; the neighbouring available orders are 302 and '
        '434. There is no defensible rounding here, because the grid is what converges the '
        'ISA-A shapes and the kernel alike, so options() emits no angular value at all and '
        "the caller's own ATOMIC_PROPERTY_SPHERICAL_POINTS / "
        'ATOMIC_RESPONSE_SPHERICAL_POINTS declaration stands and is part of their model. '
        'That declaration is also BOUNDED by the propagator this preset asks for: the '
        'AUX-space ALDA kernel of ATOMIC_RESPONSE_PROPAGATOR CAMCASP_DF is charged '
        'grid_rows*(naux + nbf*nocc) against a measured kernel_sampling limit of '
        '33,000,000, and on PBE0/cc-pVDZ water the default angular order 590 exceeds it '
        '(about 4.4e7 at the declared Radial 100) and is refused outright, while 302 runs '
        '(88,788 kernel rows, about 2.2e7). So under this preset the angular order is not '
        'merely unhonourable at 400; it has an upper bound well below it. Nor can the PROPERTY grid be lowered toward 400 instead: measured at the declared Radial 100, an order-302 property grid leaves an LW charge sum-rule residual of about 1.1e-06, past the localization postcondition, and the request yields no accepted local tensors. So 400 is not reachable by rounding on either grid.',
    'point_response_512_leg_limit':
        'The preset declares a 2000-point point-to-point lattice, Random 2000 with Seed 1. '
        'This branch generates the same seeded Maclaren cloud but its certified lattice is '
        '500 points, and the refinement leg is budgeted for at most 512. A 2000-point '
        'refinement is a different fit -- the underdetermined rank-1-penalized refinement '
        'is lattice-selected, so refined C8/C10/C12 move with the point set -- so '
        'options() does not request 2000 and a caller who accepts this gap is refining at '
        'their own declared ATOMIC_REFINEMENT_POINTS.',
    'kernel_integral_cutoff':
        'KERNEL-INTEGRAL-CUTOFF = 0.10E-07 is the fifth KERNEL-INTEGRAL-PARAMETERS line and '
        'the one ATOMIC_RESPONSE_PROPAGATOR CAMCASP_DF does not carry: the native '
        'propagator applies the RHO-EPS floor and the F-MAX cap and then integrates every '
        'surviving kernel element, with no magnitude screen on the assembled integrals. '
        'Screening them is a smaller kernel, not the same kernel computed faster, so it is '
        'labelled rather than approximated by reusing RHO-EPS.',
    'second_unconstrained_df_block':
        'The method file declares TWO BEGIN DF blocks, the constrained one at Lambda = '
        '1000.0 and a second at Eta = 0.0, Lambda = 0.0, Gamma = 0.0, i.e. an unconstrained '
        'fit alongside it. The reference forms both and uses each where its own protocol '
        'says to. This branch forms exactly one transition-density fit, the one '
        'ATOMIC_OV_CHARGE_PENALTY and ATOMIC_OV_METRIC_DAMPING declare. Which of the two '
        'the reference consumes at which stage is not settled here, so honouring one and '
        'calling it both would be a guess.',
    'isa_charge_convergence_threshold':
        'The ISA convergence block declares two thresholds, EPS-Norm = 1.0e-09 on the shape '
        'norms and EPS-Q = 1.0e-4 on the site charges. The native controller converges on '
        'the shape norm only, at 1e-9. The charge criterion is not implied by the norm one: '
        'it can stop an iteration the norm test would continue, and it can refuse one the '
        'norm test would accept, so the shipped controller is a different stopping rule.',
    'isa_tail_iteration_count':
        'Tail-Iterations = 30 against a hard-coded 20 in the native controller. The tail '
        'refit is what fixes each site shape beyond its W-TAILS cutoff, so the count is a '
        'model declaration and not an iteration budget to be traded for time: a 20- and a '
        '30-tail-iteration partition are different converged shapes.',
    'declared_atomaux_basis':
        'The cluster template declares a SEPARATE ATOMAUX basis for the ISA expansions, '
        'aVQZ Type MC Spherical Use-ISA-Basis, larger than the aVTZ AUX used for the '
        'density fitting (the A+DF template instead requires ATOMAUX == AUX exactly). The '
        'generated recipe of this branch builds its ATOMAUX and Shape sets from one named '
        'JKFIT set; there is no option that declares an independent ATOMAUX basis, so the '
        'two-basis structure of the preset cannot be reproduced.',
    'isa_basis_set2':
        'ISA-Basis set2 Min-S-exp-H = 0.2 augments both auxiliary sets with the '
        'reference-owned ISA shape set, under a floor on the hydrogen s exponents. This '
        'branch has no ISA-set augmentation and no exponent floor: its shape functions come '
        'from the generated recipe. The augmentation changes the space the shapes are '
        'expanded in, which is the partition itself.',
    'spherical_auxiliary_basis':
        'Both templates declare Spherical AUX, and the A+DF template requires it (its '
        'ATOMAUX must be spherical and must equal AUX). The generated recipe of this branch '
        'is Cartesian throughout and pureAM is not a declarable part of it. The reference '
        'notes Cartesian AUX converges the ISA-A less reliably, which is a statement about '
        'the same functional in a different space, not a tolerance.',
    'ac_multipole_producer':
        'AC MULTIPOLE asks for the asymptotically corrected potential built from the '
        "molecule's own multipole expansion, using the declared IP. That producer IS now "
        'reachable: ATOMIC_SCF_ASYMPTOTIC_CORRECTION DECLARED_MULTPOLE_AC plus the '
        'ATOMIC_AC_* declaration, applied by an explicit psi4.atomic_asymptotic_correction'
        '(wfn) call. What the preset cannot do is hand it over as an option. Two reasons, '
        'both of which stay the caller\'s: the IP is caller data -- the cluster template '
        'declares the placeholder IP 0.0 a.u., which is exactly this branch\'s undeclared '
        'sentinel -- and production is a separate named call, because a property request '
        'never runs the correction iteration. So options() emits no correction and a '
        'caller who accepts this gap is running the preset on UNCORRECTED orbitals unless '
        'they produce the corrected ones themselves; the two are different models and '
        'their numbers may not be quoted as agreeing. One piece is unhonourable even then: '
        'the ALDA+CHF response kernel is not differentiated through the correction, which '
        'the model string records as "no asymptotic-correction kernel derivative".',
    'isa_a_plus_df_functional':
        'ISA-Algorithm A+DF  Zeta = 0.1 with DF-PARAMETERS Lambda = 1000.0 is the single '
        'hunk that distinguishes the A+DF preset: the partition functional becomes a '
        'zeta-weighted combination of the ISA-A functional and the density-fitting '
        'functional. This branch implements the A functional alone. There is no reduction '
        'of A+DF to A -- they are different partitions of the same density -- so the A+DF '
        'manifest refuses to emit options at all rather than quietly running A.',
}


@dataclass(frozen=True)
class Declaration:
    """One line of a preset, verbatim, and what this branch does with it."""
    block: str
    text: str
    route: str
    option: tuple = ()
    value: object = None
    gap: str = None
    note: str = None

    def __post_init__(self):
        if isinstance(self.option, str):
            object.__setattr__(self, 'option', (self.option,))
        object.__setattr__(self, 'option', tuple(self.option))
        if not self.block or not self.text.strip():
            raise ValueError('a declaration needs its block and its verbatim text')
        if self.route not in ROUTES:
            raise ValueError(f'{self.text!r}: route must be one of {ROUTES}')
        if self.route == 'option':
            if not self.option or self.value is None:
                raise ValueError(f'{self.text!r}: an option route needs the option and its value')
        elif self.option or self.value is not None:
            raise ValueError(f'{self.text!r}: only an option route carries an option and a value')
        if self.route == 'unhonoured':
            if self.gap not in GAPS:
                raise ValueError(f'{self.text!r}: an unhonoured declaration needs a named gap')
        elif self.gap is not None:
            raise ValueError(f'{self.text!r}: only an unhonoured declaration names a gap')
        if self.route in ('fixed', 'inert') and not self.note:
            # 'fixed' has to say where the value lives, and 'inert' has to carry
            # its proof; neither claim is checkable from the preset text alone.
            raise ValueError(f'{self.text!r}: a {self.route} declaration must say why')

    @property
    def honoured(self):
        return self.route != 'unhonoured'


@dataclass(frozen=True)
class PresetManifest:
    """A whole preset: its declarations, the reduction, and what it refuses."""
    name: str
    method_path: str
    cluster_path: str
    declarations: tuple
    model_name: str
    refusal: str = None
    camcasp_revision: str = CAMCASP_REVISION
    source_sha256: tuple = SOURCE_SHA256

    def __post_init__(self):
        object.__setattr__(self, 'declarations', tuple(self.declarations))
        if not self.declarations:
            raise ValueError('a preset manifest without declarations records nothing')
        for path, digest in self.source_sha256:
            if len(digest) != 64 or digest.strip('0123456789abcdef'):
                raise ValueError(f'{path}: source_sha256 must pin a sha256 hex digest')
        if not any(p == self.method_path for p, _ in self.source_sha256):
            raise ValueError(f'{self.method_path} is not pinned in source_sha256')
        if not any(p == self.cluster_path for p, _ in self.source_sha256):
            raise ValueError(f'{self.cluster_path} is not pinned in source_sha256')
        # Two declarations may carry the same option only if they agree about it,
        # which is the reference's own redundancy (ISA-GRID is declared once for
        # the multipoles and once for the polarizabilities) and not a merge.
        seen = {}
        for d in self.declarations:
            for name in d.option:
                if name in seen and seen[name] != d.value:
                    raise ValueError(f'{name} is declared as both {seen[name]!r} and {d.value!r}')
                seen[name] = d.value

    @property
    def honoured(self):
        return tuple(d for d in self.declarations if d.honoured)

    @property
    def unhonoured(self):
        return tuple(d for d in self.declarations if not d.honoured)

    @property
    def gaps(self):
        """Named gaps, in first-declared order, each one a model difference."""
        out = []
        for d in self.unhonoured:
            if d.gap not in out:
                out.append(d.gap)
        return tuple(out)

    @property
    def declared_options(self):
        """Option name -> declared value, ignoring whether a caller accepts it."""
        return {name: d.value for d in self.declarations for name in d.option}

    def options(self, accept_reduction=()):
        """The Psi4 options this preset declares, once the caller names the gaps.

        ``accept_reduction`` must name EVERY key in :attr:`gaps`, exactly. It is
        deliberately not a boolean and deliberately not defaulted: a caller who
        has to spell out ``'spherical_auxiliary_basis'`` has been told that their
        run is not the preset, and a caller who spells out a gap that no longer
        exists gets an error rather than a silently ignored argument.

        The returned dict is the honoured subset. It never carries a stand-in
        value for an unhonoured declaration, so options a preset cannot reach
        are simply absent and the caller's own declarations remain theirs.
        """
        if self.refusal:
            raise ValueError(f'{self.name}: {self.refusal}')
        accepted, gaps = set(accept_reduction), set(self.gaps)
        unknown = accepted - gaps
        if unknown:
            raise ValueError(f'{self.name} has no such gap to accept: '
                             + ', '.join(sorted(unknown)))
        missing = gaps - accepted
        if missing:
            raise ValueError(
                f'{self.name} cannot be run in full by this branch, and the options it '
                'CAN declare are a different model, so every dropped declaration has to '
                'be accepted by name. Not yet accepted: ' + ', '.join(sorted(missing))
                + '. See isapol_modern_preset.GAPS for what each one changes.')
        return dict(self.declared_options)

    def require_complete(self):
        """Raise unless the branch honours every declaration of this preset.

        For callers that want the preset or nothing. Both shipped manifests
        raise today; the message is the list of what would have to be built.
        """
        if self.refusal:
            raise ValueError(f'{self.name}: {self.refusal}')
        if self.gaps:
            raise ValueError(f'{self.name} is not honoured in full by this branch. '
                             'Unhonoured declarations: '
                             + '; '.join(f'{d.text.strip()} [{d.gap}]' for d in self.unhonoured))

    def reduction_report(self):
        """Human-readable record of what running :meth:`options` actually means."""
        lines = [f'{self.name} ({self.method_path} @ {self.camcasp_revision[:7]})',
                 f'  model actually run: {self.model_name}']
        if self.refusal:
            lines.append(f'  REFUSED: {self.refusal}')
        for key in self.gaps:
            lines.append(f'  dropped [{key}]: ' + '; '.join(
                d.text.strip() for d in self.unhonoured if d.gap == key))
        return '\n'.join(lines)


#: The declarations shared by both presets: everything outside the one hunk that
#: distinguishes them, plus the cluster template lines. Order follows the files.
_SHARED = (
    Declaration('Edit', '  NEIGHBOURS TYPE = OVR EPS = 0.01 PRINT',
                'unhonoured', gap='overlap_neighbour_rule'),
    Declaration('SET QUAD', '  Type Gauss-Legendre', 'fixed',
                note='core.CasimirGrid is Gauss-Legendre by construction; '
                     'isapol_oeprop.py builds CasimirGrid(10, .5).'),
    Declaration('SET QUAD', '  Beta 0.5', 'fixed',
                note='the omega0 of that same CasimirGrid(10, .5).'),
    Declaration('BEGIN GRID', '  Angular 400', 'unhonoured', gap='lebedev_angular_grid'),
    Declaration('BEGIN GRID', '  Radial  100', 'option',
                option=('ATOMIC_PROPERTY_RADIAL_POINTS', 'ATOMIC_RESPONSE_RADIAL_POINTS'),
                value=100,
                note='the reference integrates the partition and the response on ONE grid, '
                     'where this branch declares two, so honouring the single declaration '
                     'means declaring both radial counts.'),
    Declaration('SET Lattice', '  Charge   1.0', 'fixed',
                note='the refinement point charge is the unit source charge of '
                     'libisapol/fit_points; there is no option to scale it.'),
    Declaration('SET Lattice', '  LoLim    2.0', 'option',
                option='ATOMIC_REFINEMENT_LOWER_LIMIT', value=2.0),
    Declaration('SET Lattice', '  HiLim    4.0', 'option',
                option='ATOMIC_REFINEMENT_UPPER_LIMIT', value=4.0),
    Declaration('SET Lattice', '  Random   2000', 'unhonoured',
                gap='point_response_512_leg_limit'),
    Declaration('SET Lattice', '  Seed     1', 'option',
                option='ATOMIC_REFINEMENT_SEED', value=1),
    Declaration('SET NEW-PROP', '  Kernel ALDA', 'fixed',
                note="the native kernel is 'alda_slater_pw92'; the cluster template's "
                     'ALDA+CHF is the operative declaration for the polarizability step '
                     'and isapol_oeprop.py pairs it with exact_exchange=.25, '
                     "local_scale=.75 for PBE0. Neither is declarable."),
    Declaration('SET NEW-PROP', '  C-DF  ( For unconstrained DF use: DF )', 'option',
                option='ATOMIC_RESPONSE_BASIS', value='FITTED_AUXILIARY',
                note='C-DF is the CONSTRAINED density-fitted transition density; the '
                     'constraints themselves are the BEGIN DF block below.'),
    Declaration('SET NEW-PROP / KERNEL-INTEGRAL-PARAMETERS',
                '    INFINITY-CONTROL-METHOD FD', 'option',
                option='ATOMIC_RESPONSE_PROPAGATOR', value='CAMCASP_DF',
                note="CAMCASP_DF carries this whole block: its KernelSmoothing declares "
                     "method 'FD' with rho_epsilon 1e-8, f_max 1000., fd_delta .01, "
                     'fd_alpha 1., which is the reference block line for line except for '
                     'KERNEL-INTEGRAL-CUTOFF.'),
    Declaration('SET NEW-PROP / KERNEL-INTEGRAL-PARAMETERS', '    RHO-EPS  = 1e-8', 'option',
                option='ATOMIC_RESPONSE_PROPAGATOR', value='CAMCASP_DF'),
    Declaration('SET NEW-PROP / KERNEL-INTEGRAL-PARAMETERS', '    F-MAX    = 1000.0', 'option',
                option='ATOMIC_RESPONSE_PROPAGATOR', value='CAMCASP_DF'),
    Declaration('SET NEW-PROP / KERNEL-INTEGRAL-PARAMETERS', '    FD-DELTA = 0.01', 'option',
                option='ATOMIC_RESPONSE_PROPAGATOR', value='CAMCASP_DF'),
    Declaration('SET NEW-PROP / KERNEL-INTEGRAL-PARAMETERS', '    FD-ALPHA = 1.0', 'option',
                option='ATOMIC_RESPONSE_PROPAGATOR', value='CAMCASP_DF'),
    Declaration('SET NEW-PROP / KERNEL-INTEGRAL-PARAMETERS',
                '    KERNEL-INTEGRAL-CUTOFF =   0.10E-07', 'unhonoured',
                gap='kernel_integral_cutoff'),
    Declaration('SET NEW-PROP', '  SOLVER LU ( Options: GELSS )', 'fixed',
                note='the native response solve is an LU factorization; GELSS is not '
                     'implemented and so cannot be declared away from LU.'),
    Declaration('SET PROPAGATOR', '  Type CKS', 'fixed',
                note='coupled Kohn-Sham is the only response this branch forms.'),
    Declaration('SET PROPAGATOR', '  Hessians Internal', 'fixed',
                note='H1/H2 are assembled inside libisapol from its own operators; there '
                     'is no external Hessian import to select.'),
    Declaration('SET PROPAGATOR', '  DF with constraints', 'option',
                option='ATOMIC_RESPONSE_PROPAGATOR', value='CAMCASP_DF'),
    Declaration('SET PROPAGATOR', '  DF-integrals', 'option',
                option='ATOMIC_RESPONSE_PROPAGATOR', value='CAMCASP_DF'),
    Declaration('SET DF-INTEGRALS', '  DF-TYPE-MONOMER NN', 'fixed',
                note='the constrained NN fit is the only monomer DF type this branch '
                     'builds; its constraints are declared by the two option values '
                     'below, not by a type name.'),
    Declaration('SET DF', '  Solver LU (Options are LU and GELSS )', 'fixed',
                note='as for the response solve: LU, with no GELSS to choose.'),
    Declaration('BEGIN DF', '  Type     NN', 'fixed',
                note='the same constrained NN fit as DF-TYPE-MONOMER above.'),
    Declaration('BEGIN DF', '  Eta    =    0.0', 'option',
                option='ATOMIC_OV_METRIC_DAMPING', value=0.0,
                note='Eta scales every off-centre Coulomb-metric element by 1-eta; the '
                     'declared 0.0 is the undamped fit.'),
    Declaration('BEGIN DF', '  Lambda = 1000.0', 'option',
                option='ATOMIC_OV_CHARGE_PENALTY', value=1000.0,
                note='the rank-1 charge penalty A += lambda*q q^T. Note this is NOT the '
                     'branch default of 1.0, so a run under this preset must never be '
                     'compared with a number recorded at lambda 1.'),
    Declaration('BEGIN DF', '  Gamma  =    0.0', 'fixed',
                note='the native constrained fit forms the metric damping and the charge '
                     'penalty and no third constraint term at all, which is exactly what '
                     'Gamma = 0.0 asks for. A nonzero Gamma would be unhonourable.'),
    Declaration('BEGIN DF', '  Print only normalization constraints', 'report'),
    Declaration('BEGIN DF (second block)',
                '  Type     NN\n  Eta    = 0.0\n  Lambda = 0.0\n  Gamma  = 0.0\n'
                '  Print only normalization constraints', 'unhonoured',
                gap='second_unconstrained_df_block'),
    Declaration('Begin ISA', '  DF = Drho-C', 'fixed',
                note='the partition is converged against the Drho-C fitted density; this '
                     'branch forms no other ISA density.'),
    Declaration('Begin ISA', '  W-INIT = ONE-GTO   ALPHA0 = 1.0', 'fixed',
                note='the native controller starts from one Gaussian per site at alpha0 = '
                     '1.0. The starting shapes are not declarable, and the converged '
                     'solution is checked rather than assumed independent of them.'),
    Declaration('Begin ISA', '  Solver LU', 'fixed',
                note='the ISA linear solve is an LU factorization.'),
    Declaration('Begin ISA / Convergence', '    Convergence-Type W', 'fixed',
                note='the controller converges on the shape coefficients W.'),
    Declaration('Begin ISA / Convergence', '    EPS-Norm    = 1.0e-09', 'fixed',
                note='the native shape-norm convergence threshold is 1e-9.'),
    Declaration('Begin ISA / Convergence', '    EPS-Q       = 1.0e-4', 'unhonoured',
                gap='isa_charge_convergence_threshold'),
    Declaration('Begin ISA / Convergence', '    Max-Iter    = 120', 'fixed',
                note='the native iteration limit is 120.'),
    Declaration('Begin ISA / Convergence', '    W-Damping   = 0.0', 'fixed',
                note='the native controller applies no shape damping.'),
    Declaration('Begin ISA / Convergence',
                '    W-Mix-Fraction = 0.0  Skip-Iterations = 20', 'fixed',
                note='no mixing, and the first 20 iterations skip the mixing machinery, '
                     'as in the native controller.'),
    Declaration('Begin ISA / Convergence', '    W-Eps = 0.17', 'fixed',
                note='the native small-coefficient threshold is 0.17.'),
    Declaration('Begin ISA / Convergence', '    S-Block-Only', 'fixed',
                note='that threshold is applied to the s block only.'),
    Declaration('Begin ISA / Convergence', '    Couple', 'inert',
                note='src/stockholder.F90 at this revision sets LwEps = 0 at both sites '
                     'that read Couple under the declared Activate-at 1.0e-05, so the '
                     'keyword cannot change a number in this preset. The branch has no '
                     'coupling to omit.'),
    Declaration('Begin ISA / Convergence', '    Activate-at 1.0e-05', 'fixed',
                note='the native W-Eps/positivity machinery activates at 1e-5.'),
    Declaration('Begin ISA / Convergence',
                '    Positive-W   Lambda = 0.001  Auto  Max-Alpha = 0.2', 'fixed',
                note='the native positivity constraint is lambda 0.001, automatic, with '
                     'max alpha 0.2, at the 1e-5 activation above.'),
    Declaration('Begin ISA / Convergence', '    Tail-Iterations = 30', 'unhonoured',
                gap='isa_tail_iteration_count'),
    Declaration('Begin ISA / W-TAILS', '    Func = 1', 'fixed',
                note='the native tail replacement is the Func-1 form.'),
    Declaration('Begin ISA / W-TAILS', '    R1-Multiplier = 1.5', 'option',
                option='ATOMIC_PROPERTY_RECIPE',
                value='GENERATED_JKFIT_BRAGG_SLATER_TAIL_ISA_A',
                note='the recipe NAME is the tail policy: this value declares the '
                     'reference meaning of R1-Multiplier = 1.5, a per-element '
                     '1.5*R_Slater cutoff, rather than the demo recipe\'s flat 1.5 bohr. '
                     'Not the branch default, and the two move r**4-weighted site '
                     'multipoles and hence C8/C10 by percent.'),
    Declaration('Begin ISA / W-TAILS', '    R2-Multiplier = 2.5', 'inert',
                note='src/stockholder.F90 reads r(2) only in case(2) of the tail fit; the '
                     'Fit-Type = 3 declared two lines below uses r(1) alone, so this line '
                     'cannot change a number in this preset.'),
    Declaration('Begin ISA / W-TAILS', '    Fit-Type = 3', 'fixed',
                note='the native tail fit is type 3, the one-radius exponential fit.'),
    Declaration('Begin ISA / W-TAILS', '    W-Tests', 'report'),
    Declaration('Begin Multipoles', '  DF Type ISA-GRID', 'option',
                option='ATOMIC_MULTIPOLE_DISTRIBUTION', value='ISA_A',
                note='the ISA-GRID distribution IS the converged Drho-C stockholder '
                     'partition, which this branch names ISA_A.'),
    Declaration('Begin Multipoles', '  Rank 4', 'option',
                option='ATOMIC_MULTIPOLE_RANK', value=4,
                note='rank 4 is what makes the C12 rank pairs (1,4) and (4,1) exist. '
                     'ATOMIC_LOCALIZATION_RANK_LIMIT is a SEPARATE declaration that this '
                     'preset does not make -- it hands the distributed polarizabilities to '
                     "ORIENT -- so options() leaves it at the caller's value, which the "
                     'rank-4 site space permits at 1 through 4.'),
    Declaration('BEGIN Polarizability',
                '  DIST-ALG ISA-GRID   ( this sets the ISA-GRID-based partitioning )',
                'option', option='ATOMIC_MULTIPOLE_DISTRIBUTION', value='ISA_A'),
    Declaration('BEGIN Polarizability',
                '  Quad   10           ( and this is you want 1+10 static+freq-dependent pols )',
                'fixed', note='the n of the same hard-coded CasimirGrid(10, .5): one static '
                              'plus ten imaginary-frequency points.'),
    Declaration('BEGIN Polarizability', '  Invert No', 'fixed',
                note='the native response never inverts the propagator matrix; it solves.'),
    Declaration('BEGIN Polarizability', '  Spherical', 'fixed',
                note='distributed polarizabilities are formed in the real spherical '
                     'multipole basis throughout; there is no Cartesian output to select.'),
    Declaration('BEGIN Polarizability', '  Rank 4', 'option',
                option='ATOMIC_MULTIPOLE_RANK', value=4),
    Declaration('BEGIN Polarizability',
                '  Calculate only total and distributed polarizabilities and perturbations',
                'report'),
    Declaration('BEGIN Polarizability', '  Print only static total pols upto rank 2', 'report'),
    Declaration('BEGIN Polarizability', '  Pert-file DEFAULT', 'report'),
    Declaration('BEGIN Polarizability', '  Pol-file  DEFAULT', 'report'),
    # --- cluster template ---
    Declaration('clt: Molecule MOL', '  UNITS  Bohr', 'caller'),
    Declaration('clt: Molecule MOL', '  IP     0.0 a.u.              ( the AC needs the IP. )',
                'caller', note='the IP is an input to the asymptotic correction and is '
                               'never derived from a HOMO eigenvalue; it goes in '
                               'ATOMIC_AC_IONIZATION_POTENTIAL, where 0.0 -- the '
                               'placeholder this template carries -- is the undeclared '
                               'sentinel. See ac_multipole_producer.'),
    Declaration('clt: Molecule MOL', '  X   Z    Rx   Ry   Rz', 'caller'),
    Declaration('clt', 'Show MOL in XYZ format         ( always a useful check )', 'report'),
    Declaration('clt: Run-Type', '  Properties', 'caller',
                note='the tasks are the oeprop argument list, not an option.'),
    Declaration('clt: Run-Type', '  Molecule     MOL', 'caller',
                note='selects the molecule declared above; the wavefunction is that '
                     'selection here.'),
    Declaration('clt: Run-Type', '  Main-Basis     aVTZ   Type  MC', 'caller',
                note='the orbital basis comes from the wavefunction through '
                     'adapt_main(wfn); the preset does not own it.'),
    Declaration('clt: Run-Type',
                '  Aux-Basis      aVTZ   Type  MC   Spherical   Use-ISA-Basis',
                'unhonoured', gap='spherical_auxiliary_basis'),
    Declaration('clt: Run-Type',
                '  AtomAux-Basis  aVQZ   Type  MC   Spherical   Use-ISA-Basis',
                'unhonoured', gap='declared_atomaux_basis'),
    Declaration('clt: Run-Type', '  ISA-Basis      set2   Min-S-exp-H = 0.2',
                'unhonoured', gap='isa_basis_set2'),
    Declaration('clt: Run-Type', '  Func         PBE0', 'caller',
                note='the branch validates that the caller converged PBE0 rather than '
                     'running the SCF itself.'),
    Declaration('clt: Run-Type', '  AC           MULTIPOLE', 'unhonoured',
                gap='ac_multipole_producer'),
    Declaration('clt: Run-Type', '  Kernel       ALDA+CHF        ( needed for the Pol calculation )',
                'fixed', note='alda_slater_pw92 with exact_exchange=.25 and local_scale=.75, '
                              'fixed in isapol_oeprop.py for PBE0.'),
    Declaration('clt: Run-Type', '  SCF-code     <name of code>', 'caller',
                note='Psi4 itself; the wavefunction is the declaration.'),
    Declaration('clt: Run-Type', '  File-Prefix  MOL-ISA-Pol-ISA-A', 'report'),
)


def _with(declarations, replacements):
    """Substitute declarations by their verbatim text, refusing a silent no-op."""
    out, used = [], set()
    for d in declarations:
        new = replacements.get(d.text)
        if new is None:
            out.append(d)
        else:
            used.add(d.text)
            out.extend(new)
    missing = set(replacements) - used
    if missing:
        raise ValueError('nothing to replace: ' + '; '.join(sorted(missing)))
    return tuple(out)


#: ``methods/isa-pol-from-isa-A``: the ISA-A functional alone. This is the preset
#: whose options() can be obtained, once its eleven gaps are accepted by name.
ISA_A = PresetManifest(
    name='isa-pol-from-isa-A',
    method_path='methods/isa-pol-from-isa-A',
    cluster_path='methods/isa-pol-from-isa-A.clt_tmpl',
    declarations=_SHARED + (
        Declaration('Begin ISA', '  ISA-Algorithm A', 'fixed',
                    note='the ISA-A functional is the one this branch implements.'),
        Declaration('clt: Run-Type', '  #METHOD      isa-pol-from-isa-A', 'caller',
                    note='names which of the two presets applies, which is what choosing '
                         'this manifest does.'),
        Declaration('clt', 'Title  Template Cluster file for isa-pol-from-isa-A', 'report'),
    ),
    model_name=('ISA-A partition on a Cartesian generated-JKFIT recipe with the '
                'Bragg-Slater 1.5*R_Slater Func-1 tails, constrained fitted-AUX transition '
                'densities at lambda=1000 eta=0, CAMCASP_DF ALDA propagator without the '
                'kernel-integral cutoff, rank-4 ISA_A distribution, radial 100 at the '
                "caller's own angular order, 1+10 Gauss-Legendre beta=0.5 Casimir grid, "
                "refinement on the caller's own seeded 2.0-4.0 vdW lattice. NOT "
                'isa-pol-from-isa-A: see reduction_report().'),
)

#: ``methods/isa-pol-from-isa-A+DF``: the same protocol with a zeta-weighted
#: ISA-A + density-fitting functional. Recorded in full, and refused in full.
ISA_A_PLUS_DF = PresetManifest(
    name='isa-pol-from-isa-A+DF',
    method_path='methods/isa-pol-from-isa-A+DF',
    cluster_path='methods/isa-pol-from-isa-A+DF.clt_tmpl',
    declarations=_with(_SHARED, {
        '  Aux-Basis      aVTZ   Type  MC   Spherical   Use-ISA-Basis': (
            Declaration('clt: Run-Type',
                        '  Aux-Basis      aVTZ   Type  MC   Spherical   Use-ISA-Basis  '
                        '( this basis must be spherical )',
                        'unhonoured', gap='spherical_auxiliary_basis'),),
        '  AtomAux-Basis  aVQZ   Type  MC   Spherical   Use-ISA-Basis': (
            Declaration('clt: Run-Type',
                        '  AtomAux-Basis  aVTZ   Type  MC   Spherical   Use-ISA-Basis  '
                        '( and identical to this.       )',
                        'unhonoured', gap='declared_atomaux_basis'),),
        '  File-Prefix  MOL-ISA-Pol-ISA-A': (
            Declaration('clt: Run-Type', '  File-Prefix  MOL-ISA-Pol-ISA-A+DF', 'report'),),
    }) + (
        Declaration('Begin ISA', '  ISA-Algorithm A+DF  Zeta = 0.1', 'unhonoured',
                    gap='isa_a_plus_df_functional'),
        Declaration('Begin ISA', '  DF-PARAMETERS Lambda = 1000.0', 'unhonoured',
                    gap='isa_a_plus_df_functional'),
        Declaration('Begin ISA', '  Solver        LU', 'fixed',
                    note='the same LU solve; only the functional line differs from the A '
                         'preset.'),
        Declaration('clt: Run-Type', '  #METHOD      isa-pol-from-isa-A+DF', 'caller'),
        Declaration('clt', 'Title  Template Cluster file for isa-pol-from-isa-A+DF', 'report'),
    ),
    model_name='none: this branch has no ISA-A+DF partition to run.',
    refusal=('the ISA-A+DF functional is not implemented. Its defining declaration, '
             'ISA-Algorithm A+DF Zeta = 0.1 with DF-PARAMETERS Lambda = 1000.0, is a '
             'zeta-weighted combination of the ISA-A and density-fitting functionals, and '
             'A is not a reduction of it but a different partition. Emitting the A-only '
             'subset of these options would silently switch the target to the other '
             'preset, which plan.md:377 forbids: run ISA_A and say so, or implement A+DF.')

)

#: Both shipped presets, by name.
PRESETS = {ISA_A.name: ISA_A, ISA_A_PLUS_DF.name: ISA_A_PLUS_DF}

# Every gap must be reachable from a preset, so that GAPS cannot outlive the
# declaration that motivated it, and every preset gap must be in GAPS (checked
# per declaration above).
_referenced = {d.gap for p in PRESETS.values() for d in p.unhonoured}
if _referenced != set(GAPS):
    raise ValueError('GAPS and the presets disagree: '
                     + ', '.join(sorted(set(GAPS) ^ _referenced)))
del _referenced
