/*
 * Psi4: an open-source quantum chemistry software package
 * Copyright (c) 2007-2026 The Psi4 Developers.
 * Licensed under the GNU Lesser General Public License, version 3.
 * This software is distributed WITHOUT ANY WARRANTY; see the repository COPYING.
 */
#ifndef CUEST_DIRECT_JK_H
#define CUEST_DIRECT_JK_H

#include "jk.h"

#if defined(USING_cuEST) && defined(USING_cuEST_DIRECT_JK)
namespace psi {

/// Conventional four-center J/K: no auxiliary basis or stored DF tensor.
/// Like cuESTJK, K contains alpha*K + beta*K_erf(omega); wK is not separate.
class PSI_API cuESTDirectJK : public JK {
   private:
    struct State;
    std::unique_ptr<State> state_;
    Options& options_;

   protected:
    std::string name() override { return "cuESTDirectJK"; }
    bool C1() const override { return true; }
    size_t memory_estimate() override { return 0; }  // No stored host ERI/DF tensor.
    void preiterations() override;
    void compute_JK() override;
    void postiterations() override;

   public:
    cuESTDirectJK(std::shared_ptr<BasisSet> primary, Options& options);
    ~cuESTDirectJK() override;
    void print_header() const override;
    void set_do_wK(bool value) override {
        do_wK_ = value;
        do_K_ = do_K_ || value;
    }
    void set_do_K(bool value) override { do_K_ = value || do_wK_; }
};

}  // namespace psi
#endif
#endif
