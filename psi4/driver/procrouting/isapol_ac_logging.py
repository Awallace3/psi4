# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Reporting retained for the separate declared-AC producer.

The native bounded ISA route does not call this producer. Keep its reporting
independent of bounded-response orchestration without removing the cleanup
branch's explicit asymptotic-correction functionality.
"""
import numpy as np

from .isapol_logging import _fmt, _matrix, _set, dataclass_parameters


def ac_parameters(declaration, *, maxiter, energy_threshold, gradient_threshold,
                  diis_subspace, shift_damping, max_energy_threshold=None,
                  max_gradient_threshold=None):
    return (dataclass_parameters(declaration)
            + (('label', declaration.label()), ('maxiter', maxiter),
               ('energy_threshold', energy_threshold),
               ('gradient_threshold', gradient_threshold),
               ('loosest admitted energy_threshold', max_energy_threshold),
               ('loosest admitted gradient_threshold', max_gradient_threshold),
               ('diis_subspace', diis_subspace), ('shift_damping', shift_damping)))


def report_ac_splice(log, *, exact_exchange, fermi_amaldi_scale, electrons, occupied,
                     basis_functions, origin_bohr, bragg_radii, grid_points, grid_blocks,
                     active_blocks, active_points, level=1):
    log.line('Splice geometry (the SCF exchange-correlation grid, unchanged):', level=level)
    log.items((('exact exchange a_x', exact_exchange),
               ('Fermi-Amaldi coefficient c_FA/N', fermi_amaldi_scale),
               ('electrons N', electrons), ('occupied orbitals', occupied),
               ('basis functions', basis_functions),
               ('multipole origin [bohr]', tuple(origin_bohr)),
               ('Bragg-Slater radii [bohr]', tuple(bragg_radii)),
               ('grid points', grid_points), ('grid blocks', grid_blocks),
               ('blocks with f > 0', active_blocks), ('points with f > 0', active_points),
               ('fraction of grid corrected',
                0. if not grid_points else active_points / float(grid_points))), level=level)
    log.line(level=level)


AC_ITERATION_HEADERS = ('iter', 'energy [Eh]', 'dE', 'max|[F,D]|', 'shift [Eh]',
                        'HOMO', 'LUMO', 'clamps')


class AcIterationLog:
    """Write actual iteration rows without retaining a scientific trajectory."""

    def __init__(self, log, level=2):
        self._log, self._level = log, int(level)
        self._open = False

    def row(self, *, iteration, energy, delta, gradient, shift, homo, lumo, clamped):
        if not self._log.enabled(self._level):
            return
        if not self._open:
            self._log.line('Declared asymptotic-correction SCF iterations '
                           '(Tozer-Handy shift; DIIS):', level=self._level)
            self._open = True
        cells = (int(iteration), float(energy),
                 None if delta is None else float(delta), float(gradient),
                 float(shift), float(homo), float(lumo), int(clamped))
        if not hasattr(self, '_widths'):
            self._widths = [max(len(h), 14) for h in AC_ITERATION_HEADERS]
            self._log.line('  '.join(h.rjust(w) for h, w in
                                     zip(AC_ITERATION_HEADERS, self._widths)),
                           level=self._level, indent=4)
        self._log.line('  '.join(_fmt(c).rjust(w) for c, w in zip(cells, self._widths)),
                       level=self._level, indent=4)


AC_SPECTRUM_VIRTUALS = 10


def report_ac_spectrum(log, wfn, energies, nocc, *, mark='', level=1):
    energies = np.asarray(energies, dtype=float).ravel()
    nocc = int(nocc)
    if nocc < 1 or energies.size <= nocc:
        raise ValueError('orbital energies must cover the occupied set and one virtual')
    stop = min(energies.size, nocc + AC_SPECTRUM_VIRTUALS)
    log.table('Corrected orbital energies (HOMO %s, LUMO %s, gap %s Eh):'
              % (_fmt(float(energies[nocc - 1])), _fmt(float(energies[nocc])),
                 _fmt(float(energies[nocc] - energies[nocc - 1]))),
              ('orbital', 'occupation', 'energy [Eh]'),
              [(i + 1, 2. if i < nocc else 0., float(energies[i])) for i in range(stop)],
              level=level,
              note=('%d higher virtual orbitals not printed' % (energies.size - stop))
              if energies.size > stop else None)
    _set(wfn, 'ATOMIC DECLARED AC ORBITAL ENERGIES' + mark, _matrix(energies.reshape(1, -1)))


def report_ac_convergence(log, wfn, convergence, declaration, *, converged):
    log.items((('declaration', declaration.label()),
               ('converged', bool(converged)),
               ('iterations', int(convergence.iterations)),
               ('delta energy', float(convergence.delta_energy)),
               ('orbital gradient', float(convergence.orbital_gradient)),
               ('energy threshold', float(convergence.energy_threshold)),
               ('gradient threshold', float(convergence.gradient_threshold)),
               ('Tozer-Handy shift', float(convergence.shift)),
               ('shift clamp hits', int(convergence.shift_clamped)),
               ('HOMO', float(convergence.homo)), ('LUMO', float(convergence.lumo)),
               ('gap', float(convergence.lumo - convergence.homo)),
               ('declared IP', float(declaration.ionization_potential)),
               ('I + eps_HOMO', float(declaration.ionization_potential + convergence.homo)),
               ('energy (not variational)', float(convergence.energy)),
               ('plain SCF reference energy', float(convergence.reference_energy)),
               ('energy above plain SCF',
                float(convergence.energy - convergence.reference_energy)),
               ('grid points', int(convergence.grid_points))))
    if wfn is None:
        return
    mark = '' if converged else ' UNCONVERGED'
    _set(wfn, 'ATOMIC DECLARED AC CONVERGED', 1. if converged else 0.)
    for key, value in (('ITERATIONS', float(convergence.iterations)),
                       ('SHIFT', float(convergence.shift)),
                       ('SHIFT CLAMP HITS', float(convergence.shift_clamped)),
                       ('HOMO', float(convergence.homo)),
                       ('LUMO', float(convergence.lumo)),
                       ('GAP', float(convergence.lumo - convergence.homo)),
                       ('DELTA ENERGY', float(convergence.delta_energy)),
                       ('ORBITAL GRADIENT', float(convergence.orbital_gradient)),
                       ('DECLARED IP', float(declaration.ionization_potential)),
                       ('ENERGY NOT VARIATIONAL', float(convergence.energy)),
                       ('REFERENCE SCF ENERGY', float(convergence.reference_energy)),
                       ('GRID POINTS', float(convergence.grid_points))):
        _set(wfn, 'ATOMIC DECLARED AC ' + key + mark, value)


def ac_application_parameters(record):
    return (('declaration', record.declaration.label()),
            ('basis functions', int(np.asarray(record.orbitals).shape[0])),
            ('occupied orbitals', int(record.nocc)),
            ('Tozer-Handy shift', float(record.convergence.shift)),
            ('iterations', int(record.convergence.iterations)),
            ('replaced state', 'Ca, Cb, Da, Db, Fa, Fb, epsilon_a, epsilon_b, energy'),
            ('energy replaced by', 'the plain functional at the corrected density, '
                                   'which is not a variational minimum'),
            ('SCF seal', 'verified, then deliberately invalidated by this mutation'))
