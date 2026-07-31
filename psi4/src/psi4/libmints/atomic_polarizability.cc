/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2025 The Psi4 Developers.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * @END LICENSE
 */

#include "psi4/libmints/atomic_polarizability.h"

#include <utility>

#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsi4util/exception.h"

namespace psi {

AtomicPolarizabilityCalculator::AtomicPolarizabilityCalculator(std::shared_ptr<Wavefunction> wfn)
    : wfn_(std::move(wfn)) {
    if (!wfn_) {
        throw PSIEXCEPTION("AtomicPolarizabilityCalculator: wavefunction is null");
    }
}

void AtomicPolarizabilityCalculator::validate_wavefunction_prerequisites() const {
    bool has_orbital_response_data = false;
    try {
        has_orbital_response_data =
            wfn_->molecule() && wfn_->basisset() && wfn_->Ca() && wfn_->Da() && wfn_->epsilon_a();
    } catch (const PsiException&) {
        // Some Wavefunction accessors reject incomplete, safely constructed wavefunctions.
    }

    if (!has_orbital_response_data) {
        throw PSIEXCEPTION(
            "AtomicPolarizabilityCalculator: unsupported wavefunction is missing required orbital response data");
    }
}

void AtomicPolarizabilityCalculator::compute() {
    // Output arrays must not be allocated or published until every native response
    // prerequisite has been validated. The response provider is added in a later stage.
    validate_wavefunction_prerequisites();
    throw PSIEXCEPTION("AtomicPolarizabilityCalculator: required native response data are unavailable");
}

}  // namespace psi
