/* SPDX-License-Identifier: LGPL-3.0-only */
#ifndef PSI4_CUEST_DIIS_H
#define PSI4_CUEST_DIIS_H
#ifdef USING_cuEST
#include "psi4/pragma.h"
#include "psi4/libmints/typedefs.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace psi {
/// GPU history and projected commutators; only small Gram matrices/coefficient
/// vectors cross the host boundary. SCF still supplies host F/D and consumes F.
class PSI_API CuestDIIS {
   public:
    CuestDIIS(int max_vectors, int memory_mb, bool errors, bool densities, bool oldest);
    ~CuestDIIS();
    double scf_update(const std::vector<SharedMatrix>& focks, const std::vector<SharedMatrix>& densities,
                      SharedMatrix overlap, SharedMatrix orthogonalizer, bool save, bool rms);
    void add_entry(const std::vector<SharedMatrix>& errors, const std::vector<SharedMatrix>& targets,
                   const std::vector<SharedMatrix>& densities);
    void extrapolate(const std::vector<double>& coefficients, const std::vector<SharedMatrix>& targets);
    std::vector<double> error_gram() const;
    std::vector<double> density_fock_gram() const;
    /// Explicit diagnostic download, never called by the SCF iteration.
    std::vector<SharedMatrix> error_vectors();
    std::map<std::string, double> stats() const;
    int size() const;
    int capacity() const;
    int last_pivot() const;
    void test_fail_allocation_after(int successful_allocations);
    void reset_subspace();
    void clear();
   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace psi
#endif
#endif
