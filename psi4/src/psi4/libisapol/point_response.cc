/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2026 The Psi4 Developers.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * @END LICENSE
 */
// Restricted C1 admission, shell/Libint validation and resource accounting
// follow native_response.cc in this module. The point-charge operator itself is
// new native work: no CamCASP or ORIENT source was consulted for it.
#include "point_response.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <libint2.hpp>

#include "psi4/libmints/basisset.h"
#include "psi4/libmints/gshell.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"

namespace psi {
namespace isapol {
namespace {
void require_point(bool ok, const std::string& message) {
    if (!ok) throw std::invalid_argument("IsaPointChargeOperators: " + message);
}
std::size_t mul(std::size_t a, std::size_t b) {
    require_point(!b || a <= std::numeric_limits<std::size_t>::max() / b, "resource size overflow");
    return a * b;
}
std::size_t add(std::size_t a, std::size_t b) {
    require_point(a <= std::numeric_limits<std::size_t>::max() - b, "resource size overflow");
    return a + b;
}
void matrix_ok(const SharedMatrix& m, int rows, int cols, const char* name) {
    require_point(m && m->nirrep() == 1 && m->nrow() == rows && m->ncol() == cols,
                  std::string(name) + " dimensions/C1 symmetry mismatch");
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            require_point(std::isfinite(m->get(i, j)), std::string(name) + " must be finite");
}
// The caller's Libint shells are read, not reconstructed: nothing derived from
// the basis is retained by this class. Validate that they are consistent with
// the Gaussian shells and not stale before any engine buffer is indexed.
void validate_shells(const std::shared_ptr<BasisSet>& basis) {
    require_point(!basis->has_ECP(), "ECP bases are not supported");
    require_point(basis->nshell() > 0 && basis->nshell() <= basis->nbf(), "invalid shell count");
    int offset = 0, high_am_purity = -1;
    for (int s = 0; s < basis->nshell(); ++s) {
        const auto& sh = basis->shell(s);
        require_point(sh.am() >= 0 && sh.am() <= 4 && sh.am() <= basis->max_am() && sh.nprimitive() > 0 &&
                          sh.nprimitive() <= 64 && sh.nprimitive() <= basis->max_nprimitive() &&
                          sh.ncenter() >= 0 && sh.ncenter() < basis->molecule()->natom() &&
                          sh.function_index() == offset && sh.nfunction() > 0 &&
                          sh.nfunction() <= basis->nbf() - offset,
                      "invalid Gaussian shell bounds");
        const auto& l2 = basis->l2_shell(s);
        require_point(l2.contr.size() == 1, "Libint shell must have exactly one contraction");
        const auto& contraction = l2.contr[0];
        require_point(contraction.l == sh.am() && l2.size() == static_cast<std::size_t>(sh.nfunction()) &&
                          l2.alpha.size() == static_cast<std::size_t>(sh.nprimitive()) &&
                          contraction.coeff.size() == l2.alpha.size(),
                      "inconsistent Libint shell dimensions");
        if (sh.am() >= 2) {
            const int purity = sh.is_pure() ? 1 : 0;
            require_point(contraction.pure == sh.is_pure() && (high_am_purity < 0 || high_am_purity == purity),
                          "mixed or inconsistent high-AM shell purity unsupported");
            high_am_purity = purity;
        }
        for (int axis = 0; axis < 3; ++axis)
            require_point(std::isfinite(sh.coord(axis)) && l2.O[axis] == sh.coord(axis),
                          "stale or nonfinite Libint shell origin");
        for (int p = 0; p < sh.nprimitive(); ++p)
            require_point(std::isfinite(sh.exp(p)) && sh.exp(p) > 0 && l2.alpha[p] == sh.exp(p) &&
                              std::isfinite(contraction.coeff[p]),
                          "invalid Libint shell primitives");
        offset += sh.nfunction();
    }
    require_point(offset == basis->nbf(), "incomplete Gaussian shell function coverage");
}
}  // namespace

IsaPointChargeOperators::IsaPointChargeOperators(std::shared_ptr<Wavefunction> wfn, bool caller_converged,
                                                 SharedMatrix points_bohr, std::size_t max_bytes,
                                                 std::size_t max_points) {
    require_point(wfn && caller_converged, "explicit wavefunction and caller convergence declaration required");
    require_point(wfn->nirrep() == 1 && wfn->same_a_b_orbs() && wfn->same_a_b_dens() &&
                      wfn->nalpha() == wfn->nbeta() && wfn->soccpi()[0] == 0,
                  "only restricted closed-shell C1 wavefunctions are supported");
    auto basis = wfn->basisset();
    require_point(basis && basis->molecule() && basis->nbf() > 0, "missing orbital basis/molecule");
    const int nbf = basis->nbf(), nmo = wfn->nmo();
    nocc_ = wfn->nalpha();
    nvir_ = nmo - nocc_;
    require_point(nocc_ > 0 && nvir_ > 0 && nmo <= nbf && wfn->doccpi()[0] == nocc_,
                  "invalid integer Aufbau occupations or empty OV space");
    const std::size_t nov = mul(nocc_, nvir_);
    require_point(nov <= 512, "dense OV resource limit (maximum 512)");
    require_point(basis->max_am() <= 4 && basis->max_nprimitive() <= 64 && nbf <= 256,
                  "point operator basis resource limit");
    // The source lattice is caller-owned data, never inferred from a fit, a
    // final potential or an ISA/AUX site list.
    require_point(points_bohr && points_bohr->nirrep() == 1 && points_bohr->ncol() == 3 &&
                      points_bohr->nrow() > 0,
                  "explicit source points require [x,y,z] rows in bohr");
    npoint_ = points_bohr->nrow();
    matrix_ok(points_bohr, npoint_, 3, "source points");
    require_point(max_points > 0 && static_cast<std::size_t>(npoint_) <= max_points &&
                      static_cast<std::size_t>(npoint_) <= 512,
                  "source point resource limit (maximum 512)");
    // Conservative dense envelope: the retained operator plus one getter copy,
    // per-point AO operator, overlap, Gram and coefficient scratch, and a flat
    // engine/basis metadata allowance. Libint's private allocations are
    // backend-dependent; this is not an OS total-memory guarantee.
    auto elements = add(mul(3, mul(nov, static_cast<std::size_t>(npoint_))), mul(6, mul(nbf, nbf)));
    elements = add(elements, add(mul(4, mul(nbf, nmo)), mul(4, static_cast<std::size_t>(npoint_))));
    planned_bytes_ = add(mul(elements, sizeof(double)), 16ULL * 1024 * 1024);
    require_point(max_bytes > 0 && planned_bytes_ <= max_bytes, "dense workspace byte resource limit");

    matrix_ok(wfn->Ca(), nbf, nmo, "Ca");
    matrix_ok(wfn->Cb(), nbf, nmo, "Cb");
    matrix_ok(wfn->Da(), nbf, nbf, "Da");
    matrix_ok(wfn->Db(), nbf, nbf, "Db");
    auto ea = wfn->epsilon_a(), eb = wfn->epsilon_b();
    require_point(ea && eb && ea->nirrep() == 1 && eb->nirrep() == 1 && ea->dim() == nmo && eb->dim() == nmo,
                  "orbital energy dimension/C1 mismatch");
    for (int p = 0; p < nmo; ++p) {
        require_point(std::isfinite(ea->get(p)) && ea->get(p) == eb->get(p),
                      "energies must be finite and restricted");
        for (int mu = 0; mu < nbf; ++mu)
            require_point(wfn->Ca()->get(mu, p) == wfn->Cb()->get(mu, p), "alpha/beta orbitals differ");
    }
    for (int mu = 0; mu < nbf; ++mu)
        for (int nu = 0; nu < nbf; ++nu) {
            double density = 0;
            for (int i = 0; i < nocc_; ++i) density += wfn->Ca()->get(mu, i) * wfn->Ca()->get(nu, i);
            require_point(std::isfinite(density) && wfn->Da()->get(mu, nu) == wfn->Db()->get(mu, nu) &&
                              std::abs(density - wfn->Da()->get(mu, nu)) <= 1.e-9 * std::max(1.0, std::abs(density)),
                          "density inconsistent with integer occupied orbitals (fractional occupations unsupported)");
        }
    require_point(std::isfinite(wfn->energy()), "wavefunction energy must be finite");
    validate_shells(basis);

    points_ = points_bohr->clone();
    auto c = wfn->Ca()->clone();
    auto co = std::make_shared<Matrix>(nbf, nocc_), cv = std::make_shared<Matrix>(nbf, nvir_);
    for (int mu = 0; mu < nbf; ++mu) {
        for (int i = 0; i < nocc_; ++i) co->set(mu, i, c->get(mu, i));
        for (int a = 0; a < nvir_; ++a) cv->set(mu, a, c->get(mu, nocc_ + a));
    }
    // Local serial engines: no OneBodyAOInt, global INTS_TOLERANCE or Process
    // thread state. Ordered shell pairs, all functions retained. No screening
    // of our own; libint2 still returns a null buffer for shell pairs its own
    // precision test discards, and those blocks are then left as exact zeros.
    libint2::Engine overlap_engine(libint2::Operator::overlap, basis->max_nprimitive(), basis->max_am(), 0);
    overlap_engine.set_precision(1.e-15);
    auto overlap = std::make_shared<Matrix>(nbf, nbf);
    for (int s0 = 0; s0 < basis->nshell(); ++s0)
        for (int s1 = 0; s1 < basis->nshell(); ++s1) {
            const auto& sh0 = basis->shell(s0);
            const auto& sh1 = basis->shell(s1);
            overlap_engine.compute(basis->l2_shell(s0), basis->l2_shell(s1));
            const double* buf = overlap_engine.results()[0];
            if (!buf) continue;
            std::size_t index = 0;
            for (int m = 0; m < sh0.nfunction(); ++m)
                for (int n = 0; n < sh1.nfunction(); ++n, ++index) {
                    require_point(std::isfinite(buf[index]), "nonfinite overlap integral");
                    overlap->set(sh0.function_index() + m, sh1.function_index() + n, buf[index]);
                }
        }
    auto gram = linalg::triplet(c, overlap, c, true, false, false);
    for (int p = 0; p < nmo; ++p)
        for (int q = 0; q < nmo; ++q)
            require_point(std::isfinite(gram->get(p, q)) && std::abs(gram->get(p, q) - (p == q ? 1.0 : 0.0)) <= 1.e-8,
                          "orbitals must be AO-overlap orthonormal");

    // Geometry diagnostics: recorded for the caller, never acted upon. Exactly
    // coincident sources are rejected because they duplicate a target row, not
    // as a conditioning heuristic; near-coincident sources are admitted and
    // reported through minimum_point_separation_bohr.
    auto molecule = basis->molecule();
    min_nuclear_ = std::numeric_limits<double>::infinity();
    min_separation_ = npoint_ > 1 ? std::numeric_limits<double>::infinity() : 0.;
    for (int p = 0; p < npoint_; ++p) {
        for (int atom = 0; atom < molecule->natom(); ++atom) {
            const double dx = points_->get(p, 0) - molecule->x(atom);
            const double dy = points_->get(p, 1) - molecule->y(atom);
            const double dz = points_->get(p, 2) - molecule->z(atom);
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            require_point(std::isfinite(r), "nonfinite source/nucleus separation");
            min_nuclear_ = std::min(min_nuclear_, r);
        }
        for (int q = 0; q < p; ++q) {
            const double dx = points_->get(p, 0) - points_->get(q, 0);
            const double dy = points_->get(p, 1) - points_->get(q, 1);
            const double dz = points_->get(p, 2) - points_->get(q, 2);
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            require_point(std::isfinite(r) && r > 0, "duplicate (exactly coincident) source point");
            min_separation_ = std::min(min_separation_, r);
        }
    }
    // Unreachable under the admission above (a validated basis implies at least
    // one centre); kept so the sentinel cannot leak if that ever changes.
    if (molecule->natom() == 0) min_nuclear_ = 0.;

    // Libint's nuclear operator with charge q returns -q*int chi chi/|r-C| dr,
    // exactly as libmints/electrostatic.cc documents. q=-1 therefore selects the
    // positive Coulomb kernel published by convention(); it is minus the
    // charge-inclusive electron ESP operator oeprop accumulates with q=+1.
    libint2::Engine potential(libint2::Operator::nuclear, basis->max_nprimitive(), basis->max_am(), 0);
    potential.set_precision(1.e-15);
    w_ = std::make_shared<Matrix>(static_cast<int>(nov), npoint_);
    for (int p = 0; p < npoint_; ++p) {
        potential.set_params(std::vector<std::pair<double, std::array<double, 3>>>{
            {-1.0, {points_->get(p, 0), points_->get(p, 1), points_->get(p, 2)}}});
        auto v = std::make_shared<Matrix>(nbf, nbf);
        for (int s0 = 0; s0 < basis->nshell(); ++s0)
            for (int s1 = 0; s1 < basis->nshell(); ++s1) {
                const auto& sh0 = basis->shell(s0);
                const auto& sh1 = basis->shell(s1);
                potential.compute(basis->l2_shell(s0), basis->l2_shell(s1));
                const double* buf = potential.results()[0];
                if (!buf) continue;  // engine precision zero; no external screening
                std::size_t index = 0;
                for (int m = 0; m < sh0.nfunction(); ++m)
                    for (int n = 0; n < sh1.nfunction(); ++n, ++index) {
                        require_point(std::isfinite(buf[index]), "nonfinite point-charge AO integral");
                        v->set(sh0.function_index() + m, sh1.function_index() + n, buf[index]);
                    }
            }
        for (int mu = 0; mu < nbf; ++mu)
            for (int nu = 0; nu < mu; ++nu)
                require_point(std::abs(v->get(mu, nu) - v->get(nu, mu)) <=
                                  1.e-10 * std::max({1.0, std::abs(v->get(mu, nu)), std::abs(v->get(nu, mu))}),
                              "AO point-charge operator reciprocity failure (not symmetrized)");
        auto ov = linalg::triplet(co, v, cv, true, false, false);
        for (int a = 0; a < nvir_; ++a)
            for (int i = 0; i < nocc_; ++i) {
                const double value = ov->get(i, a);
                require_point(std::isfinite(value), "nonfinite point-charge OV operator element");
                w_->set(a * nocc_ + i, p, value);
                max_element_ = std::max(max_element_, std::abs(value));
            }
    }
}
SharedMatrix IsaPointChargeOperators::operators() const { return w_->clone(); }
SharedMatrix IsaPointChargeOperators::points() const { return points_->clone(); }
}  // namespace isapol
}  // namespace psi
